import logging
import time
from typing import List, Tuple, Union, Dict, Set, Generator, Any, Iterable
from string import Template
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from tqdm import tqdm
from collections import defaultdict
from datetime import date
import os
from meta_prov_fixer.utils import *

# # OC Meta published RDF dumps publication dates and DOIs at the time of writing this code (2025-07-01).
# meta_dumps_pub_dates = [
#     ('2022-12-19', 'https://doi.org/10.6084/m9.figshare.21747536.v1'),
#     ('2022-12-20', 'https://doi.org/10.6084/m9.figshare.21747536.v2'),
#     ('2023-02-15', 'https://doi.org/10.6084/m9.figshare.21747536.v3'),
#     ('2023-06-28', 'https://doi.org/10.6084/m9.figshare.21747536.v4'),
#     ('2023-10-26', 'https://doi.org/10.6084/m9.figshare.21747536.v5'),
#     ('2024-04-06', 'https://doi.org/10.6084/m9.figshare.21747536.v6'),
#     ('2024-06-17', 'https://doi.org/10.6084/m9.figshare.21747536.v7'),
#     ('2025-02-02', 'https://doi.org/10.6084/m9.figshare.21747536.v8')
# ]


class ProvenanceIssueFixer:
    def __init__(self, endpoint: str, issues_log_dir:Union[str, None]=None, checkpoint='checkpoint.json'):
        """
        Base class for fixing provenance issues via SPARQL queries.
        Initializes the SPARQL endpoint and sets up the query method.
        Classes dedicated to fixing specific issues should inherit from this class and implement the `detect_issue` and `fix_issue` methods.
        
        :param sparql_endpoint: The SPARQL endpoint URL.
        :type sparql_endpoint: str
        :param issues_log_dir: If provided, the path to the directory where the data involved in a query-detected issue will be logged. If None, data is kept in memory.
        :type issues_log_fp: Union[str, None]
        """

        self.endpoint = endpoint
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)
        self.checkpoint = CheckpointManager(checkpoint)
        if issues_log_dir:
            self.issues_log_fp = os.path.join(issues_log_dir, f"{type(self).__qualname__}.jsonl")
        else:
            self.issues_log_fp = None

        if self.issues_log_fp:
            if os.path.exists(self.issues_log_fp):
                logging.warning(f"The issues log file {self.issues_log_fp} already exists but process was \
interrupted while writing on it. It will be overwritten.")
            os.makedirs(os.path.dirname(self.issues_log_fp), exist_ok=True)

    def _query(self, query: str, retries: int = 3, delay: float = 2.0) -> Union[dict, None]:
        for attempt in range(retries):
            try:
                self.sparql.setQuery(query)
                return self.sparql.query().convert()
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    logging.error("Max retries reached. Query failed.")
                    return None
            finally:
                time.sleep(0.1)  # slight delay to avoid overwhelming the endpoint

    def _update(self, update_query: str, retries: int = 3, delay: float = 2.0) -> None:
        for attempt in range(retries):
            try:
                self.sparql.setQuery(update_query)
                self.sparql.query()
                return
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    logging.error("Max retries reached. Update failed.")

    def _paginate_query(self, query_template: str, limit: int = 100000, sleep: float = 0.5) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Executes a paginated SPARQL SELECT query and yields result bindings in batches.

        This method is designed to handle pagination by incrementing the OFFSET in the 
        provided SPARQL query template. It assumes that the query returns results in a 
        structure compatible with the SPARQL JSON results format.

        :param query_template: A SPARQL query string with two `%d` placeholders for offset and limit values (in that order).
        :type query_template: str
        :param limit: The number of results to fetch per page. Defaults to 100000.
        :type limit: int
        :param sleep: Number of seconds to wait between successive queries to avoid overwhelming the endpoint. Defaults to 0.5 seconds.
        :type sleep: float

        :yield: A list of SPARQL result bindings (each a dictionary with variable bindings).
        :rtype: Generator[List[Dict[str, Any]], None, None]
        """
        offset = 0
        while True:
            query = query_template % (offset, limit)
            logging.debug(f"Fetching results {offset} to {offset + limit}...")
            result_batch = self._query(query)

            if result_batch is None:
                break

            bindings = result_batch["results"]["bindings"]
            if not bindings:
                break

            yield bindings
            offset += limit
            time.sleep(sleep)
    
    def detect_issue(self):
        raise NotImplementedError("Subclasses must implement `detect_issue()`.")
    
    def fix_issue(self):
        raise NotImplementedError("Subclasses must implement `fix_issue()`.")
    

# (1) Delete filler snapshots and fix the rest of the graph -> move to daughter class FillerFixer
class FillerFixer(ProvenanceIssueFixer):
    """
    A class to fix issues related to filler snapshots in the OpenCitations Meta graph.
    """
    def __init__(self, endpoint: str, issues_log_dir:Union[str, None]=None):
        super().__init__(endpoint, issues_log_dir=issues_log_dir)

    def detect_issue(self, limit=10000) -> List[Tuple[str, Dict[str, Set[str]]]]:
        """
        Fetch snapshots that are fillers and need to be deleted, grouped by their named graph.

        :param limit: The number of results to fetch per page.
        :type limit: int
        :returns: A list of tuples, where the first element is a graph URI and the second element is a dictionary with 'to_delete' and 'remaining_snapshots' as keys and a set as value of both keys.
        :rtype: List[Tuple[str, Dict[str, Set[str]]]]
        """
        grouped_result = defaultdict(lambda: {'to_delete': set(), 'remaining_snapshots': set()})

        template = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX oc: <https://w3id.org/oc/ontology/>

        SELECT ?g ?snapshot ?other_se
        WHERE {
          {
            SELECT DISTINCT ?g ?snapshot ?other_se
            WHERE {
              GRAPH ?g {
                {
                  ?snapshot a prov:Entity .
                  OPTIONAL {
                    ?snapshot dcterms:description ?description .
                  }
                  FILTER (!regex(str(?snapshot), "/prov/se/1$"))
                  FILTER (!bound(?description) || !CONTAINS(LCASE(str(?description)), "merged"))
                  FILTER NOT EXISTS {
                    ?snapshot oc:hasUpdateQuery ?q .
                  }
                }
                ?other_se ?p ?o .
                FILTER (?other_se != ?snapshot)
              }
            }
            ORDER BY ?g
          }
        }
        OFFSET %d
        LIMIT %d
        """
        paradata = get_process_paradata(self)
        if self.issues_log_fp:
            res_log = open(self.issues_log_fp, 'w', encoding='utf-8')
            res_log.write(json.dumps(paradata, ensure_ascii=False) + '\n')
        else:
            logging.info(paradata)
        try:
            # Pagination loop
            logging.debug("Fetching filler snapshots (to be deleted) with pagination...")
            for current_bindings in self._paginate_query(template, limit):
                # group query results
                for b in current_bindings:
                    g = b['g']['value']
                    snapshot = b['snapshot']['value']
                    other_se = b['other_se']['value']
                    
                    grouped_result[g]['to_delete'].add(snapshot)
                    grouped_result[g]['remaining_snapshots'].add(other_se)
                    # the following is necessary when there are multiple filler snapshots in the same graph
                    grouped_result[g]['remaining_snapshots'].difference_update(grouped_result[g]['to_delete'])

            logging.info(f"{sum([len(d['to_delete']) for d in grouped_result.values()])} snapshots marked for deletion.")
            if self.issues_log_fp:
                for obj in list(dict(grouped_result).items()):
                    obj = make_json_safe(obj)
                    res_log.write(json.dumps(obj, ensure_ascii=False) + '\n')

        finally:
            if self.issues_log_fp:
                res_log.close()

        if self.issues_log_fp:
            save_detection_checkpoint(self.__class__.__name__, 'detection_done', self.checkpoint)
        return list(dict(grouped_result).items()) if not self.issues_log_fp else None

    def batch_delete_filler_snapshots(self, deletions: Union[str, List[Tuple[str, Dict[str, Set[str]]]]], batch_size=200, checkpoint=None) -> None:
        """
        Deletes snapshots from the triplestore based on the provided deletions dictionary.
        :param deletions: A list of tuples where the first element is a graph URI,  and the second is a dictionary with 'to_delete' and 'remaining_snapshots' sets.
        :type deletions: List[Tuple[str, Dict[str, Set[str]]]]
        """

        template = Template("""
            $dels
        """)

        logging.debug("Deleting filler snapshots in batches...")

        for batch_idx, (batch, line_num) in checkpointed_batch(
            deletions, 
            batch_size, 
            fixer_name=self.__class__.__name__, 
            phase="batch_delete",
            checkpoint=checkpoint
        ):
            try:
                dels = []
                for g_uri, values in batch:
                    for se_to_delete in values['to_delete']:
                        single_del = f"DELETE WHERE {{ GRAPH <{g_uri}> {{ <{se_to_delete}> ?p ?o . }}}};\n"
                        dels.append(single_del)
                dels_str = "    ".join(dels)

                query = template.substitute(dels=dels_str)

                self._update(query)
                logging.info(f"[{self.__class__.__name__}] Batch {batch_idx} (deletion) completed.")
            except Exception as e:
                logging.error(f"Error while deleting filler snapshots in Batch {batch_idx}, lines {line_num-batch_size} to {line_num}: {e}")
                print(f"Error while deleting filler snapshots in Batch {batch_idx}, lines {line_num-batch_size} to {line_num}: {e}")
                raise e
        return None


    @staticmethod
    def map_se_names(to_delete:set, remaining: set) -> dict:
        """
        For each snapshot in the union of ``to_delete`` and ``remaining`` (containing snapshot URIs), generates a new URI.

        Values in the mapping dictionary are not unique, i.e., multiple old URIs can be mapped to the same new URI.
        If ``to_delete`` is empty, the returned dictionary will have identical keys and values, i.e., the URIs will not change.
        Each URI in ``to_delete`` will be mapped to the new name of the URI in ``remaining`` that immediately precedes it in
        a sequence ordered by sequence number.

        **Examples:**

        .. code-block:: python

            to_delete = {'https://w3id.org/oc/meta/br/06101234191/prov/se/3'}
            remaining = {'https://w3id.org/oc/meta/br/06101234191/prov/se/1', 'https://w3id.org/oc/meta/br/06101234191/prov/se/2', 'https://w3id.org/oc/meta/br/06101234191/prov/se/4'}

            # The returned mapping will be:
            {
                'https://w3id.org/oc/meta/br/06101234191/prov/se/1': 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/2': 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/3': 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/4': 'https://w3id.org/oc/meta/br/06101234191/prov/se/3'
            }

        :param to_delete: A set of snapshot URIs that should be deleted.
        :type to_delete: set
        :param remaining: A set of snapshot URIs that should remain in the graph (AFTER BEING RENAMED).
        :type remaining: set
        :returns: A dictionary mapping old snapshot URIs to their new URIs.
        :rtype: dict
        """
        to_delete = set(to_delete)
        remaining = set(remaining)
        all_snapshots:list = sorted(to_delete|remaining, key=lambda x: get_seq_num(x))  # sorting is required!

        mapping = {}
        sorted_remaining = []
        base_uri = remove_seq_num(all_snapshots[0])

        if not all(u.startswith(base_uri) for u in all_snapshots):
            logging.error(f"All snapshots must start with the same base URI: {base_uri}. Found: {all_snapshots}")
            raise ValueError(f"Can rename only snapshots that are included in the same named graph.")

        for old_uri in all_snapshots:
            if old_uri in remaining:
                new_uri = f"{base_uri}{len(sorted_remaining)+1}"
                mapping[old_uri] = new_uri
                sorted_remaining.append(new_uri)

            else:  # i.e., elif old_uri in to_delete
                try:
                    new_uri = f"{base_uri}{get_seq_num(sorted_remaining[-1])}"
                except IndexError:
                    # all snapshots are fillers (must be deleted), including the first one (creation)
                    logging.error(f"The first snapshot {old_uri} is a filler. Cannot rename the remaining snapshots.")

                mapping[old_uri] = new_uri

        return mapping
    
    def rename_snapshots(self, mapping):
        """
        Renames snapshots in the triplestore according to the provided mapping.
        :param mapping: A dictionary where keys are old snapshot URIs and values are new snapshot URIs.
        :type mapping: dict
        """
        # TODO: consider modifying the query template to support bulk updates (using UNION in the WHERE block)

        # !IMPORTANT: mapping is sorted ascendingly by the sequence number of old URI (get_sequence_number on mapping's keys)
        # This is required, otherwise newly inserted URIs might be deleted when iterating over mapping's items
        mapping = dict(sorted(mapping.items(), key=lambda i: get_seq_num(i[0])))
        
        template = Template("""
        DELETE {
          GRAPH ?g {
            <$old_uri> ?p ?o .
            ?s ?p2 <$old_uri> .
          }
        }
        INSERT {
          GRAPH ?g {
            <$new_uri> ?p ?o .
            ?s ?p2 <$new_uri> .
          }
        }
        WHERE {
          GRAPH ?g {
            {
              <$old_uri> ?p ?o .
            }
            UNION
            {
              ?s ?p2 <$old_uri> .
            }
          }
        }
        """)

        for old_uri, new_uri in mapping.items():
            if old_uri == new_uri:
                continue
            query = template.substitute(old_uri=old_uri, new_uri=new_uri)
            self._update(query)
    
    def adapt_invalidatedAtTime(self, graph_uri: str, snapshots: list) -> None:
        """
        Update the ``prov:invalidatedAtTime`` property of each snapshot in the provided list to match 
        the value of ``prov:generatedAtTime`` of the following snapshot.

        :param graph_uri: The URI of the named graph containing the snapshots.
        :type graph_uri: str
        :param snapshots: A list of snapshot URIs sorted by their sequence number.
        :type snapshots: list
        :returns: None
        """
        
        # TODO: consider modifying the query template to support bulk updates
        snapshots = sorted(snapshots, key=lambda x: get_seq_num(x))  # sorting is required!
        template = Template("""
        PREFIX prov: <http://www.w3.org/ns/prov#>

        # WITH <$graph>
        DELETE {
          GRAPH <$graph> { <$snapshot> prov:invalidatedAtTime ?old_time . }
        }
        INSERT {
          GRAPH <$graph> { <$snapshot> prov:invalidatedAtTime ?new_time . }
        }
        WHERE {
          GRAPH <$graph> {
            OPTIONAL {
              <$snapshot> prov:invalidatedAtTime ?old_time .
            }
            <$following_snapshot> prov:generatedAtTime ?new_time .
          }
        }
        """)

        for s, following_se in zip(snapshots, snapshots[1:]):
            query = template.substitute(
                graph=graph_uri,
                snapshot=s,
                following_snapshot=following_se
            )

            self._update(query)

    def fix_issue(self, batch_size=200, checkpoint=None):

        # step 0: detect filler snapshots
        if not self.issues_log_fp:
            to_fix = self.detect_issue() # keep all the issues in memory
        else:
            if not detection_completed(self.__class__.__name__, 'detection_done', self.checkpoint):
                self.detect_issue() # writes all issues to self.issues_log_fp
            else:
                logging.warning(f"[{self.__class__.__name__}] Issues already detected: reading from file {self.issues_log_fp}")

        # step 1: delete filler snapshots in the role of subjects
        if not self.issues_log_fp:
            self.batch_delete_filler_snapshots(to_fix, checkpoint=checkpoint)
        else:
            self.batch_delete_filler_snapshots(self.issues_log_fp, checkpoint=checkpoint)

        logging.debug(f"Updating the graphs that had filler snapshots and the related resources in other graphs...")

        if not self.issues_log_fp:
            stream = to_fix
        else:
            stream = self.issues_log_fp
        
        # batch_size = 500

        for batch_idx, (batch, _) in checkpointed_batch(
            stream, 
            batch_size, 
            fixer_name=self.__class__.__name__, 
            phase="rename_and_adapt_datetime_sequence", 
            checkpoint=checkpoint
        ):
            # step 2: delete filler snapshots in the role of objects and rename rename remaining snapshots
            for g, _dict in batch:
                mapping = self.map_se_names(_dict['to_delete'], _dict['remaining_snapshots'])
                self.rename_snapshots(mapping)

                # step 3: adapt values of prov:invalidatedAtTime for the entities existing now, identified by "new" URIs
                new_names = list(set(mapping.values()))
                self.adapt_invalidatedAtTime(g, new_names)
            
            logging.info(f"[{self.__class__.__name__}] Batch {batch_idx} (renaming snapshots + adapting time sequence) completed.")
        
        logging.info(f"[{self.__class__.__name__}] Fixing filler snapshots terminated.")


    
# (2) DATETIME values correction -> move in daughter class DateTimeFixer
class DateTimeFixer(ProvenanceIssueFixer):
    """
    A class to fix issues related to ill-formed datetime values in the OpenCitations Meta graph.
    This class provides methods to fetch quads with ill-formed datetime values, correct them, and batch fix them in the triplestore.
    """
    def __init__(self, endpoint: str, issues_log_dir:Union[str, None]=None):
        super().__init__(endpoint, issues_log_dir=issues_log_dir)
    
    def detect_issue(self, limit=10000) -> List[Tuple[str]]:
        """
        Fetch all quads where the datetime object value is not syntactically correct or complete, including cases where
        the timezone is not specified (making the datetime impossible to compare with other offset-aware datetimes) 
        and/or where the time value includes microseconds. Querying is paginated.

        :param limit: The number of results to fetch per page.
        :type limit: int
        :returns: List of tuples (graph URI, subject, predicate, datetime value).
        :rtype: List[Tuple[str, str, str, str]]
        """
        result = []
        counter = 0

        template = r'''
        PREFIX prov: <http://www.w3.org/ns/prov#>
        
        SELECT ?g ?s ?p ?dt
        WHERE {
          {
            SELECT ?g ?s ?p ?dt WHERE {
              GRAPH ?g {
                VALUES ?p {prov:generatedAtTime prov:invalidatedAtTime}
                ?s ?p ?dt .
                FILTER (!REGEX(STR(?dt), "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:(?:\\+00:00)|Z)$"))
              }
            }
            ORDER BY ?s
          }
        }
        OFFSET %d
        LIMIT %d
        '''
        paradata = get_process_paradata(self)
        if self.issues_log_fp:
            res_log = open(self.issues_log_fp, 'w', encoding='utf-8')
            res_log.write(json.dumps(paradata, ensure_ascii=False) + '\n')
        else:
            logging.info(paradata)
        
        try:
            # Pagination loop
            logging.debug("Fetching ill-formed datetime values with pagination...")
            for current_bindings in self._paginate_query(template, limit):
                # group query results
                for b in current_bindings:
                    g = b['g']['value']
                    s = b['s']['value']
                    p = b['p']['value']
                    dt = b['dt']['value']
                    
                    counter +=1
                    out_row = (g, s, p, dt)

                    if not self.issues_log_fp:
                        result.append(out_row)
                    else:
                        out_row = make_json_safe(out_row)
                        res_log.write(json.dumps(out_row, ensure_ascii=False) + '\n')

            logging.info(f"Fetched {counter} quads with a badly formed datetime object value.")
        
        finally:
            if self.issues_log_fp:
                res_log.close()

        if self.issues_log_fp:
            save_detection_checkpoint(self.__class__.__name__, 'detection_done', self.checkpoint)
        
        return result if not self.issues_log_fp else None  # if issues_log_fp is provided, the results are logged to the file and not returned

    def batch_fix_illformed_datetimes(self, quads: Union[list, str], batch_size=200, checkpoint=None) -> None:
        """
        Replace the datetime object of each quad in ``quads`` with its correct version (offset-aware and without microseconds).
        Note that ``xsd:dateTime`` is always made explicit in newly inserted values.

        .. note::
           If a snapshot has multiple objects for ``prov:invalidatedAtTime`` or ``prov:generatedAtTime`` (though this should never 
           be the case), they all get deleted and replaced with a single, correct, value.

        :param quads: List of quads to fix (in memory) or a path to a file containing quads in JSON Lines format.
        :type quads: Union[list, str]
        :param batch_size: Number of quads to process per batch.
        :type batch_size: int
        :returns: None
        """

        template = Template('''
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        DELETE DATA {
          $to_delete
        } ;
        INSERT DATA {
          $to_insert
        }
        ''')

        for batch_idx, (batch, line_num) in checkpointed_batch(
            quads, 
            batch_size, 
            fixer_name=self.__class__.__name__, 
            phase="batch_fixed_illformed_datetimes", 
            checkpoint=checkpoint
        ):
            try:
                to_delete = []
                to_insert = []
                for g, s, p, dt in batch:
                    
                    to_delete.append(f'GRAPH <{g}> {{ <{s}> <{p}> "{dt}"^^xsd:dateTime . }}\n')
                    correct_dt = normalise_datetime(dt)
                    to_insert.append(f'GRAPH <{g}> {{ <{s}> <{p}> "{correct_dt}"^^xsd:dateTime . }}\n')
                
                to_delete_str = "  ".join(to_delete)
                to_insert_str = "  ".join(to_insert)
                query = template.substitute(to_delete=to_delete_str, to_insert=to_insert_str)

                self._update(query)
                logging.info(f"[{self.__class__.__name__}] Batch {batch_idx} completed.")

            except Exception as e:
                logging.error(f"Error while fixing datetime values in Batch {batch_idx} for quads {line_num-batch_size} to {line_num}: {e}")
                print(f"Error while fixing datetime values in Batch {batch_idx} for quads {line_num-batch_size} to {line_num}: {e}")
                raise e
        return None

    def fix_issue(self, checkpoint=None):

        # step 0: detect ill-formed datetime values
        if not self.issues_log_fp:
            to_fix = self.detect_issue() # keep all the issues in memory
        else:
            if not detection_completed(self.__class__.__name__, 'detection_done', self.checkpoint):
                self.detect_issue() # writes all issues to self.issues_log_fp
            else:
                logging.warning(f"[{self.__class__.__name__}] Issues already detected: reading from file {self.issues_log_fp}")

        # step 1:
        if not self.issues_log_fp:
            self.batch_fix_illformed_datetimes(to_fix, checkpoint=checkpoint)
        else:
            self.batch_fix_illformed_datetimes(self.issues_log_fp, checkpoint=checkpoint)
        logging.info(f"[{self.__class__.__name__}] Fixing ill-formed datetime values terminated.")

# (3) correct creation snapshots without primary source -> move to daughter class MissingPrimSourceFixer
class MissingPrimSourceFixer(ProvenanceIssueFixer):
    """
    A class to fix issues related to creation snapshots that do not have a primary source in the OpenCitations Meta graph.
    """
    def __init__(self, endpoint: str, meta_dumps_pub_dates: List[Tuple[str, str]], issues_log_dir:Union[str, None]=None):
        """
        :param endpoint: The SPARQL endpoint URL.
        :type endpoint: str
        :param meta_dumps_pub_dates: Register of published OpenCitations Meta dumps, in the form: [(<ISO format date 1>, <dump DOI1>), (<ISO format date 2>, <dump DOI2>), ...]
        :type meta_dumps_pub_dates: List[Tuple[str, str]]
        """
        super().__init__(endpoint, issues_log_dir=issues_log_dir)
        validate_meta_dumps_pub_dates(meta_dumps_pub_dates) # raises errors if something wrong
        self.meta_dumps_pub_dates = sorted([(date.fromisoformat(d), doi) for d, doi in meta_dumps_pub_dates], key=lambda x: x[0])
    
    def detect_issue(self, limit=10000) -> Union[str, List[Tuple[str, str]]]:
        """
        Fetch creation snapshots that do not have a primary source.

        :param limit: The number of results to fetch per page.
        :type limit: int
        :returns: A list of tuples with snapshot URI and generation time.
        :rtype: Union[str, List[Tuple[str, str]]]
        """
        results = []
        counter = 0

        template = """
        PREFIX prov: <http://www.w3.org/ns/prov#>

        SELECT ?s ?genTime WHERE {
          {
            SELECT DISTINCT ?s ?genTime WHERE {
              GRAPH ?g {
                ?s a prov:Entity ;
                prov:generatedAtTime ?genTime .
                FILTER NOT EXISTS { ?s prov:hadPrimarySource ?anySource }
                FILTER(REGEX(STR(?s), "/prov/se/1$"))  # escape the dollar sign if using string.Template compiling
              }
            }
            ORDER BY ?s
          }
        }
        OFFSET %d
        LIMIT %d
        """

        paradata = get_process_paradata(self)
        if self.issues_log_fp:
            res_log = open(self.issues_log_fp, 'w', encoding='utf-8')
            res_log.write(json.dumps(paradata, ensure_ascii=False) + '\n')
        else:
            logging.info(paradata)

        # Pagination loop
        logging.debug("Fetching creation snapshots without a primary source with pagination...")
        try:
            for current_bindings in self._paginate_query(template, limit):
                for b in current_bindings:
                    s = b["s"]["value"]
                    gen_time = b["genTime"]["value"]
                    # results.append((s, gen_time))
                    counter +=1
                    out_row = (s, gen_time)

                    if not self.issues_log_fp:
                        results.append(out_row)
                    else:
                        out_row = make_json_safe(out_row)
                        res_log.write(json.dumps(out_row, ensure_ascii=False) + '\n')
            
            logging.info(f"Found {counter} creation snapshots without a primary source.")
        finally:
            if self.issues_log_fp:
                res_log.close()

        if self.issues_log_fp:
            save_detection_checkpoint(self.__class__.__name__, 'detection_done', self.checkpoint)
        
        return results if not self.issues_log_fp else None  # (<snapshot uri>, <gen. time>)
    
    def batch_insert_missing_primsource(self, creations_to_fix: Union[str, List[Tuple[str, str]]], batch_size=200, checkpoint=None):
        """
        Insert primary sources for creation snapshots that do not have one, in batches.

        :param creations_to_fix: A list of tuples where each tuple contains the snapshot URI and the generation time, representing all the creation snapshots that must be fixed.
        :type creations_to_fix: Union[str, List[Tuple[str, str]]]
        :param batch_size: The number of snapshots to process in each batch.
        :type batch_size: int
        :returns: None
        """
        template = Template("""
        PREFIX prov: <http://www.w3.org/ns/prov#>

        INSERT DATA {
            $quads
        }
        """)

        for batch_idx, (batch, line_num) in checkpointed_batch(
            creations_to_fix,
            batch_size,
            fixer_name=self.__class__.__name__, 
            phase="batch_insert_missing_primsource",
            checkpoint=checkpoint
        ):
            try:
                quads = []
                for snapshot_uri, gen_time in batch:
                    prim_source_uri = get_previous_meta_dump_uri(self.meta_dumps_pub_dates, gen_time)
                    graph_uri = get_graph_uri_from_se_uri(snapshot_uri)
                    quads.append(f"GRAPH <{graph_uri}> {{ <{snapshot_uri}> prov:hadPrimarySource <{prim_source_uri}> . }}\n")
                quads_str = "    ".join(quads)
                query = template.substitute(quads=quads_str)

                self._update(query)
                logging.info(f"[{self.__class__.__name__}] Batch {batch_idx} completed.")
            except Exception as e:
                logging.error(f"Error while fixing multiple primary source in Batch {batch_idx} for snapshots {line_num-batch_size} to {line_num}: {e}")
                print(f"Error while fixing multiple primary source in Batch {batch_idx} for snapshots {line_num-batch_size} to {line_num}: {e}")
                raise e
        return None
        

    def fix_issue(self, checkpoint=None):
        
        # step 0: detect creation snapshots missing a primary source
        if not self.issues_log_fp:
            to_fix = self.detect_issue()
        else:
            if not detection_completed(self.__class__.__name__, 'detection_done', self.checkpoint):
                self.detect_issue() # writes all issues to self.issues_log_fp
            else:
                logging.warning(f"[{self.__class__.__name__}] Issues already detected: reading from file {self.issues_log_fp}")

        # step 1: insert primary source for the snapshots
        if not self.issues_log_fp:
            self.batch_insert_missing_primsource(to_fix, checkpoint=checkpoint)
        else:
            self.batch_insert_missing_primsource(self.issues_log_fp, checkpoint=checkpoint)
        logging.info(f"[{self.__class__.__name__}] Fixing creation snapshots without a primary source terminated.")


# TODO: (4) Correct snapshots with multiple objects for prov:wasAttributedTo -> move in daughter class MultiPAFixer
class MultiPAFixer(ProvenanceIssueFixer):
    """
    A class to fix issues related to snapshots that have multiple objects for the ``prov:wasAttributedTo`` property in the OpenCitations Meta graph.
    """
    def __init__(self, endpoint: str, issues_log_dir:Union[str, None]=None):
        super().__init__(endpoint, issues_log_dir=issues_log_dir)

    def detect_issue(self, limit=10000):
        """
        Fetch graph-snapshot pairs where the snapshot has more than one object for the ``prov:wasAttributedTo`` property.

        :param limit: The number of results to fetch per page.
        :type limit: int
        :returns: List of tuples (graph URI, snapshot URI).
        :rtype: List[Tuple[str, str]]
        """
        result = []
        counter = 0

        template = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?g ?s
        WHERE {
          {
            SELECT ?g ?s
            WHERE {
              GRAPH ?g {
                ?s prov:wasAttributedTo <https://w3id.org/oc/meta/prov/pa/1> .
                FILTER EXISTS {
                  ?s prov:wasAttributedTo ?other_pa
                  FILTER (?other_pa != <https://w3id.org/oc/meta/prov/pa/1>)
                }
              }
            }
            ORDER BY ?g
          }
        }
        OFFSET %d
        LIMIT %d
        """
        paradata = get_process_paradata(self)
        if self.issues_log_fp:
            res_log = open(self.issues_log_fp, 'w', encoding='utf-8')
            res_log.write(json.dumps(paradata, ensure_ascii=False) + '\n')
        else:
            logging.info(paradata)

        logging.debug("Fetching snapshots with multiple objects for prov:wasAttributedTo...")
        try:
            for current_bindings in self._paginate_query(template, limit):
                for b in current_bindings:
                    g = b['g']['value']
                    s = b['s']['value']
                    # result.append((g, s))
                    counter +=1
                    out_row = (g, s)

                    if not self.issues_log_fp:
                        result.append(out_row)
                    else:
                        out_row = make_json_safe(out_row)
                        res_log.write(json.dumps(out_row, ensure_ascii=False) + '\n')
            logging.info(f"Found {counter} snapshots with multiple objects for prov:wasAttributedTo.")
        finally:
            if self.issues_log_fp:
                res_log.close()

        if self.issues_log_fp:
            save_detection_checkpoint(self.__class__.__name__, 'detection_done', self.checkpoint)
        
        return result if not self.issues_log_fp else None 
    

    def batch_fix_extra_pa(self, multi_pa_snapshots:Union[str, List[Tuple[str]]], batch_size=200, checkpoint=None):
        """
        Delete triples where the value of ``prov:wasAttributedTo`` is <https://w3id.org/oc/meta/prov/pa/1> if there is at least another processing agent for the same snapshot subject.

        :param multi_pa_snapshots: A list of tuples where each tuple contains a graph URI and a snapshot URI.
        :type multi_pa_snapshots: Union[str, List[Tuple[str]]]
        :param batch_size: Number of snapshots to process per batch.
        :type batch_size: int
        :returns: None
        """
        template = Template("""
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        DELETE DATA {
          $quads_to_delete
        } ;
        INSERT DATA {
          $quads_to_insert
        }
        """)

        for batch_idx, (batch, line_num) in checkpointed_batch(
            multi_pa_snapshots,
            batch_size,
            fixer_name=self.__class__.__name__, 
            phase="batch_fix_extra_pa", 
            checkpoint=checkpoint
        ):
            try:
                to_delete = []
                to_insert = []
                for g, s in batch:
                    to_delete.append(f"GRAPH <{g}> {{ <{s}> prov:wasAttributedTo <https://orcid.org/0000-0002-8420-0696> . }}\n")  # deletes Arcangelo's ORCID
                    to_delete.append(f"GRAPH <{g}> {{ <{s}> prov:wasAttributedTo <https://w3id.org/oc/meta/prov/pa/1> . }}\n")  # deletes Meta's default processing agent (for ingestions only)
                    to_insert.append(f"GRAPH <{g}> {{ <{s}> prov:wasAttributedTo <https://w3id.org/oc/meta/prov/pa/2> . }}\n")  # inserts Meta's processsing agent for modification processes

                to_delete_str = "  ".join(to_delete)
                to_insert_str = "  ".join(to_insert)
                query = template.substitute(quads_to_delete=to_delete_str, quads_to_insert=to_insert_str)

                self._update(query)
                logging.info(f"[{self.__class__.__name__}] Batch {batch_idx} completed.")
            except Exception as e:
                logging.error(f"Error while fixing multiple processing agents in Batch {batch_idx} for snapshots {line_num-batch_size} to {line_num}: {e}")
                print(f"Error while fixing multiple processing agents in Batch {batch_idx} for snapshots {line_num-batch_size} to {line_num}: {e}")
                raise e
        return None
    
    def fix_issue(self, checkpoint=None):

        # step 0: detect graph and snapshots that have an additional object besides the typical <https://w3id.org/oc/meta/prov/pa/1> for prov:wasAttributedTo
        if not self.issues_log_fp:
            to_fix = self.detect_issue()
        else:
            if not detection_completed(self.__class__.__name__, 'detection_done', self.checkpoint):
                self.detect_issue() # writes all issues to self.issues_log_fp
            else:
                logging.warning(f"[{self.__class__.__name__}] Issues already detected: reading from file {self.issues_log_fp}")

        # step 1: delete all objects for prov:wasAttributedTo and insert only <https://w3id.org/oc/meta/prov/pa/2>
        if not self.issues_log_fp:
            self.batch_fix_extra_pa(to_fix, checkpoint=checkpoint)
        else:
            self.batch_fix_extra_pa(self.issues_log_fp, checkpoint=checkpoint)
        logging.info(f"[{self.__class__.__name__}] Fixing graphs with multiple processing agents terminated.")


# (5) Correct graphs where at least one snapshots has too many objects for specific properties -> move to daughter class MultiObjectFixer
class MultiObjectFixer(ProvenanceIssueFixer):
    """
    A class to fix issues related to graphs where at least one snapshot has too many objects for specific properties in the OpenCitations Meta graph.
    """
    def __init__(self, endpoint:str, meta_dumps_pub_dates: List[Tuple[str, str]], issues_log_dir:Union[str, None]=None):
        """
        :param meta_dumps_pub_dates: Register of published OpenCitations Meta dumps, in the form: [(<ISO format date 1>, <dump DOI1>), (<ISO format date 2>, <dump DOI2>), ...]
        :type meta_dumps_pub_dates: List[Tuple[str, str]]
        """
        super().__init__(endpoint, issues_log_dir=issues_log_dir)
        self.pa_uri = "https://w3id.org/oc/meta/prov/pa/1" # URI of the processing agent to be used as objects of prov:wasAtttributedTo for newly created snapshots, which is always the default one
        validate_meta_dumps_pub_dates(meta_dumps_pub_dates) # raises errors if something wrong
        self.meta_dumps_pub_dates = sorted([(date.fromisoformat(d), doi) for d, doi in meta_dumps_pub_dates], key=lambda x: x[0])
    
    def detect_issue(self, limit=10000) -> Union[None, List[str]]:
        """
        Fetch graphs containing at least one snapshot with multiple objects for 
        a property that only admits one (e.g. ``oc:hasUpdateQuery``).

        :param limit: The number of results to fetch per page.
        :type limit: int
        :returns: A list of tuples (graph URI, generation time).
        :rtype: Union[None, List[str]]
        """
        output = []
        counter = 0

        template = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX oc: <https://w3id.org/oc/ontology/>

        SELECT ?g ?genTime
        WHERE {
          {
            SELECT DISTINCT ?g
            WHERE {
              GRAPH ?g {
                VALUES ?p { prov:invalidatedAtTime prov:hadPrimarySource oc:hasUpdateQuery }
                ?s ?p ?o ;
                  a prov:Entity .
                FILTER EXISTS {
                  ?s ?p ?o2 .
                  FILTER (?o2 != ?o)
                }
              }
            }
            ORDER BY ?g
          }
          BIND (IRI(CONCAT(str(?g), "se/1")) AS ?target)
          ?target prov:generatedAtTime ?genTime .
        }
        OFFSET %d
        LIMIT %d
        """
        paradata = get_process_paradata(self)
        if self.issues_log_fp:
            res_log = open(self.issues_log_fp, 'w', encoding='utf-8')
            res_log.write(json.dumps(paradata, ensure_ascii=False) + '\n')
        else:
            logging.info(paradata)


        logging.debug("Fetching URIs of graphs containing snapshots with too many objects...")
        try:
            for current_bindings in self._paginate_query(template, limit):
                for b in current_bindings:
                    g = b['g']['value']
                    gen_time = b['genTime']['value']
                    counter += 1
                    out_row = (g, gen_time)
                    if not self.issues_log_fp:
                        output.append(out_row)
                    else:
                        out_row = make_json_safe(out_row)
                        res_log.write(json.dumps(out_row, ensure_ascii=False) + '\n')
            logging.info(f"Found {counter} distinct graphs containing snapshots with too many objects for some properties.")
        finally:
            if self.issues_log_fp:
                res_log.close()

        if self.issues_log_fp:
            save_detection_checkpoint(self.__class__.__name__, 'detection_done', self.checkpoint)
        
        return output if not self.issues_log_fp else None 
    
    def reset_multi_object_graphs(self, graphs:Union[str, list], batch_size=200, checkpoint=None):
        """
        Reset each graph in ``graphs`` by deleting the existing snapshots and creating a new 
        creation snapshot, which will be the only one left for that graph.

        :param graphs: A list of tuples (graph URI, generation time) for graphs that have too many objects for properties that only admit one.
        :type graphs: list
        :returns: None
        """

        # template = Template("""
        # PREFIX prov: <http://www.w3.org/ns/prov#>
        # PREFIX dcterms: <http://purl.org/dc/terms/>
        # PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        # PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        # #WITH GRAPH <$graph> # WITH clause seems not to be supported in rdflib
        # DELETE {
        #   GRAPH <$graph> {
        #     ?s ?p ?o
        #   }
        # }
        # INSERT {
        #   GRAPH <$graph> {
        #     <$creation_snapshot> prov:hadPrimarySource <$primary_source> ;
        #       prov:wasAttributedTo <$processing_agent> ;
        #       prov:specializationOf <$specialization_of> ;
        #       dcterms:description "$description" ;
        #       rdf:type prov:Entity ;
        #       prov:generatedAtTime ?genTime .
        #   }
        # }
        # WHERE {
        #   {
        #     SELECT ?genTime WHERE {
        #       GRAPH <$graph> {
        #         ?_s prov:generatedAtTime ?genTime .
        #         FILTER(strends(str(?_s), "/se/1"))
        #       }
        #     }
        #   }
        #   GRAPH <$graph> {
        #     ?s ?p ?o .
        #   }
        # }
        # """)

        template = Template("""
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        CLEAR GRAPH <$graph> ;
        INSERT DATA {
          GRAPH <$graph> {
            <$creation_snapshot> prov:hadPrimarySource <$primary_source> ;
              prov:wasAttributedTo <$processing_agent> ;
              prov:specializationOf <$specialization_of> ;
              dcterms:description "$description" ;
              rdf:type prov:Entity ;
              prov:generatedAtTime "$gen_time"^^xsd:dateTime .
          }
        }
        """)
        logging.debug("Resetting graphs with too many objects by creating a new single creation snapshot...")

        # batch_size = 500  # updates are executed with individual queries anyway

        for batch_idx, (batch, line_num) in checkpointed_batch(
            graphs, 
            batch_size, 
            fixer_name=self.__class__.__name__, 
            phase="reset_multi_object_graphs", 
            checkpoint=checkpoint
        ):
            try:
                for g, gen_time in batch:
                    creation_se = g + 'se/1'
                    gen_time = gen_time.replace("^^xsd:dateTime", "") 
                    gen_time = gen_time.replace("^^http://www.w3.org/2001/XMLSchema#dateTime", "")
                    prim_source = get_previous_meta_dump_uri(self.meta_dumps_pub_dates, gen_time)
                    processing_agent = self.pa_uri 
                    referent = get_described_res_omid(g)
                    desc = f"The entity '{referent}' has been created."

                    query = template.substitute(
                        graph = g,
                        creation_snapshot = creation_se,
                        primary_source = prim_source, 
                        processing_agent = processing_agent,
                        specialization_of = referent, 
                        description = desc,
                        gen_time = gen_time  # double check
                    )

                    self._update(query)
                logging.info(f"[{self.__class__.__name__}] Batch {batch_idx} completed.")
            except Exception as e:
                logging.error(f"Error while resetting graphs in Batch {batch_idx} for graphs{line_num-batch_size} to {line_num}: {e}")
                print(f"Error while resetting graphs in Batch {batch_idx} for graphs{line_num-batch_size} to {line_num}: {e}")
                raise e
        logging.debug(f"[{self.__class__.__name__}] Graph-resetting process terminated.")
        return None


    def fix_issue(self, checkpoint=None):
        # step 0: detect graphs and snapshots with multiple objects for 1-cardinality properties
        if not self.issues_log_fp:
            to_fix = self.detect_issue()
        else:
            if not detection_completed(self.__class__.__name__, 'detection_done', self.checkpoint):
                self.detect_issue() # writes all issues to self.issues_log_fp
            else:
                logging.warning(f"[{self.__class__.__name__}] Issues already detected: reading from file {self.issues_log_fp}")

        # step 1: reset graphs with a snapshot having extra objects
        if not self.issues_log_fp:
            self.reset_multi_object_graphs(to_fix, checkpoint=checkpoint)
        else:
            self.reset_multi_object_graphs(self.issues_log_fp, checkpoint=checkpoint)
        logging.info(f"[{self.__class__.__name__}] Fixing graphs with multiple objects for 1-cardinality properties terminated.")


def fix_process(
    endpoint: str,
    meta_dumps_pub_dates: List[Tuple[str, str]],
    issues_log_dir: Union[str, None] = None,
    dry_run: bool = False,
    checkpoint='checkpoint.json'
):
    """
    Function wrapping all the single fix operations into a single process,
    with strictly ordered steps, checkpointing, and timing.
    """

    checkpoint = CheckpointManager(checkpoint) # checkpoint file will be deleted if program terminates succesfully

    fixers = (
        FillerFixer(endpoint, issues_log_dir=issues_log_dir),
        DateTimeFixer(endpoint, issues_log_dir=issues_log_dir),
        MissingPrimSourceFixer(endpoint, meta_dumps_pub_dates, issues_log_dir=issues_log_dir),
        MultiPAFixer(endpoint, issues_log_dir=issues_log_dir),
        MultiObjectFixer(endpoint, meta_dumps_pub_dates, issues_log_dir=issues_log_dir),
    )

    timer = TimedProcess(total_phases=len(fixers))
    timer.start()

    for i, fixer in enumerate(fixers):
        logging.info(f"Created instance of {fixer.__class__.__qualname__}.")

        # --- resume logic: skip completed fixer ---
        if checkpoint:
            state = checkpoint.load()
            if (
                state.get("fixer") == fixer.__class__.__name__
                and state.get("phase") == "done"
            ):
                logging.info(f"Skipping {fixer.__class__.__name__} (already completed).")
                continue

        # --- run fixer ---
        timer.start_phase()
        if not dry_run:
            fixer.fix_issue(checkpoint=checkpoint)
        else:
            logging.debug(f"[fix_process]: Would run {fixer.__class__.__name__}")
        phase_time = timer.end_phase()

        # --- log progress ---
        elapsed, remaining = timer.eta(i)
        logging.info(
            f"{fixer.__class__.__name__} finished in {phase_time:.1f}s | "
            f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m"
        )

        # --- mark fixer as done ---
        if checkpoint:
            checkpoint.save(fixer.__class__.__name__, "done", -1)
            logging.info(f"{fixer.__class__.__name__} completed.")

    # clear checkpoint only when the whole pipeline is done
    logging.info("All fixing operations terminated.")
    if checkpoint:
        checkpoint.clear()
    return None
