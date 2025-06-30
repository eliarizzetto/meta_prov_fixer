import logging
import yaml
import time
import argparse
from typing import List, Tuple, Union, Dict, Set, Iterable, Generator, Any
from pathlib import Path
from string import Template

from rdflib import Graph, ConjunctiveGraph, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON, JSONLD, TURTLE, POST
from tqdm import tqdm

from collections import defaultdict
from datetime import date
import warnings

from meta_prov_fixer.utils import get_seq_num, remove_seq_num, get_described_res_omid, get_graph_uri_from_se_uri, normalise_datetime


class ProvenanceFixer:
    def __init__(self, sparql_endpoint: str = 'http://localhost:8890/sparql/', queries_fp: str = 'nuovo_queries.yaml', auth: Union[str, None] = None, dry_run: bool = False):
        self.endpoint = sparql_endpoint
        self.dry_run = dry_run
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)
        if auth:
            self.sparql.addCustomHttpHeader("Authorization", auth)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

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

    def _paginate_query(self, query_template: str, limit: int = 100000, sleep: float = 0.5) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Executes a paginated SPARQL SELECT query and yields result bindings in batches.

        This method is designed to handle pagination by incrementing the OFFSET in the 
        provided SPARQL query template. It assumes that the query returns results in a 
        structure compatible with the SPARQL JSON results format.

        :param query_template: A SPARQL query string with two `%d` placeholders 
                            for offset and limit values (in that order).
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
            logging.info(f"Fetching results {offset} to {offset + limit}...")
            result_batch = self._query(query)

            if result_batch is None:
                break

            bindings = result_batch["results"]["bindings"]
            if not bindings:
                break

            yield bindings
            offset += limit
            time.sleep(sleep)

    # (1) Delete filler snapshots and fix the rest of the graph -> move to daughter class FillerFixer

    def fetch_snapshots_to_delete(self, limit=10000) -> Dict[str, Dict[str, Set[str]]]:
        """
        Fetches snapshots that are fillers and need to be deleted, grouped by their named graph.
        Returns a dictionary where keys are graph URIs and values are dictionaries with 'to_delete' and 'remaining_snapshots' sets,
        storing respectively the URIs of the graph's snapshots that should be deleted and the URIs of the other snapshots.
        :return: A dictionary with graph URIs as keys and dictionaries with 'to_delete' and 'remaining_snapshots' sets as values.
        :rtype: Dict[str, Dict[str, Set[str]]]
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
        # Pagination loop
        logging.info("Fetching filler snapshots (to be deleted) with pagination...")
        for current_bindings in self._paginate_query(template, limit):
            # group query results
            for b in current_bindings:
                g = b['g']['value']
                snapshot = b['snapshot']['value']
                other_se = b['other_se']['value']
                
                grouped_result[g]['to_delete'].add(snapshot)
                grouped_result[g]['remaining_snapshots'].add(other_se)

        logging.info(f"{sum([len(d['to_delete']) for d in grouped_result.values()])} snapshots marked for deletion.")
        return dict(grouped_result)

    def batch_delete_filler_snapshots(self, deletions: Dict[str, Dict[str, Set[str]]], batch_size=500) -> None:
        """
        Deletes snapshots from the triplestore based on the provided deletions dictionary.
        :param deletions: A dictionary where keys are graph URIs and values are dictionaries with 'to_delete' and 'remaining_snapshots' sets.
        :type deletions: Dict[str, Dict[str, Set[str]]]
        """

        template = """
        DELETE WHERE {
            $dels
        }
        """

        logging.info("Deleting filler snapshots in batches...")
        for i in range(0, len(deletions), batch_size):
            batch = deletions.items()[i:i + batch_size]
            dels = []
            for g_uri, values in batch:
                for se_to_delete in values['to_delete']:
                    single_del = f"GRAPH <{g_uri}> {{ <{se_to_delete}> ?p ?o . }}\n"
                    dels.append(single_del)
            dels_str = "    ".join(dels)

            query = template.substitute(dels=dels_str)

            if self.dry_run:
                logging.info(f"[Dry-run] Would delete triples of filler snapshots in graphs {i} to {len(batch)}.")
            else:
                self._query(query)
                logging.info(f"Deleted triples of filler snapshots in graphs {i} to {len(batch)}.")

    def map_se_names(to_delete:set, remaining: set) -> dict:
        """
        For each snapshot in the union of to_delete and remaining (containing snapshots URIs), generates a new URI.
        Values in mapping dictionary are not unique, i.e., multiple old URIs can be mapped to the same new URI.
        If to_delete is empty, the returned dictionary will have identical keys and values, i.e., the URIs will not change.
        Each URI in to_delete will be mapped to the new name of the URI in remaining that immediately precedes it in
        a sequence ordered by sequence number. For example::

            to_delete = {'https://w3id.org/oc/meta/br/06101234191/prov/se/3'}
            remaining = {'https://w3id.org/oc/meta/br/06101234191/prov/se/1', 'https://w3id.org/oc/meta/br/06101234191/prov/se/2', 'https://w3id.org/oc/meta/br/06101234191/prov/se/4'}

            # The returned mapping will be:
            {
                'https://w3id.org/oc/meta/br/06101234191/prov/se/1': 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/2': 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/3': 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/4': 'https://w3id.org/oc/meta/br/06101234191/prov/se/3'
            }

            #-----------
            to_delete = {'https://w3id.org/oc/meta/br/06101234191/prov/se/2', 'https://w3id.org/oc/meta/br/06101234191/prov/se/3'}
            remaining = {'https://w3id.org/oc/meta/br/06101234191/prov/se/1', 'https://w3id.org/oc/meta/br/06101234191/prov/se/4'}

            # The returned mapping will be:
            {
                'https://w3id.org/oc/meta/br/06101234191/prov/se/1': 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/2': 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/3': 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/4': 'https://w3id.org/oc/meta/br/06101234191/prov/se/2'
            }

            # -----------
            to_delete = {'https://w3id.org/oc/meta/br/06101234191/prov/se/2', 'https://w3id.org/oc/meta/br/06101234191/prov/se/4'}
            remaining = {'https://w3id.org/oc/meta/br/06101234191/prov/se/1', 'https://w3id.org/oc/meta/br/06101234191/prov/se/3', 'https://w3id.org/oc/meta/br/06101234191/prov/se/5'}

            # The returned mapping will be:
            {
                'https://w3id.org/oc/meta/br/06101234191/prov/se/1': 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/2': 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/3': 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/4': 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
                'https://w3id.org/oc/meta/br/06101234191/prov/se/5': 'https://w3id.org/oc/meta/br/06101234191/prov/se/3'
            }
        
        :param to_delete: A set of snapshot URIs that should be deleted.
        :type to_delete: set
        :param remaining: A set of snapshot URIs that should remain in the graph (AFTER BEING RENAMED).
        :type remaining: set
        :return: A dictionary mapping old snapshot URIs to their new URIs.
        :rtype: dict
        """

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
        if self.dry_run:
            logging.info(f"[Dry-run] Would rename snapshots in the whole graph according to the following mapping: {mapping}")
            return
        
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

        logging.info("Renaming snapshots in the triplestore...")
        for old_uri, new_uri in mapping.items():
            query = template.substitute(old_uri=old_uri, new_uri=new_uri)
            self._query(query)
        logging.info(f"Snapshot entities were re-named in all named graphs according to the following mapping: {mapping}.")
    
    def adapt_invalidatedAtTime(self, graph_uri: str, snapshots: list) -> None:
        """
        Updates the ``prov:invalidatedAtTime`` property of each snapshot in the provided list to match 
        the value of ``prov:generatedAtTime`` of the following snapshot.
        :param graph_uri: The URI of the named graph containing the snapshots.
        :type graph_uri: str
        :param snapshots: A list of snapshot URIs sorted by their sequence number.
        :type snapshots: list
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
            if self.dry_run:
                logging.info(f"[Dry-run] Would replace the value of invalidatedAtTime for {s} with the value of generatedAtTime for {following_se} in graph {graph_uri}.")
            else:
                self._query(query)
                logging.info(f"Replaced the value of invalidatedAtTime for {s} with the value of generatedAtTime for {following_se} in graph {graph_uri}.")

    
    # (2) DATETIME values correction -> move in daughter class DateTimeFixer

    def fetch_illformed_datetimes(self, limit=1000000):
        """
        Fetches all quads where the datetime object value is not syntactically correct or complete, including cases where
        the timezone is not specified (which would make the datetime impossible to compare with other offset-aware datetimes) 
        and/or where the time value includes microseconds. Querying is paginated.
        """
        result = []

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
        # Pagination loop
        logging.info("Fetching ill-formed datetime values with pagination...")
        for current_bindings in self._paginate_query(template, limit):
            # group query results
            for b in current_bindings:
                g = b['g']['value']
                s = b['s']['value']
                p = b['p']['value']
                dt = b['dt']['value']
                
                result.append((g, s, p, dt))

        logging.info(f"Fetched {len(result)} quads with a badly formed datetime object value.")
        return result
    
    def batch_fix_illformed_datetimes(self, quads: list, batch_size=500):
        """
        Replaces the datetime object of each quad in ``quads`` with its correct version (offset-aware and without microseconds).
        Note that ``xsd:dateTime`` is always made explicit in newly inserted values.
        N.B.: if a snapshot has multiple objects for ``prov:invalidatedAtTime`` or ``prov:generatedAtTime`` (though this should never 
        be the case), they all get deleted and replaced with a single, correct, value.
        """

        template = Template('''
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        DELETE WHERE {
          $to_delete
        } ;
        INSERT DATA {
          $to_insert
        }
        ''')

        for i in range(0, len(quads), batch_size):
            batch = quads[i:i + batch_size]
            to_delete = []
            to_insert = []
            for g, s, p, dt in batch:
                
                to_delete.append(f"GRAPH <{g}> {{ <{s}> <{p}> ?ill_dt . }}\n")
                correct_dt = normalise_datetime(dt)
                to_insert.append(f'GRAPH <{g}> {{ <{s}> <{p}> "{correct_dt}"^^xsd:dateTime . }}\n')
            
            to_delete_str = "  ".join(to_delete)
            to_insert_str = "  ".join(to_insert)
            query = template.substitute(to_delete=to_delete_str, to_insert=to_insert_str)
            if self.dry_run:
                logging.info(f"[Dry-run] Would fix datetime values for quads {i} to {len(batch)}.")
            else:
                self._query(query)
                logging.info(f"Fixed datetime values for quads {i} to {len(batch)}.")

    # (3) correct creation snapshots without primary source -> move to daughter class MissingPrimSourceFixer

    def get_previous_meta_dump_uri(dt:str)-> str:
        """
        Returns the DOI of the OpenCitations Meta dump that was published before the given date.
        :param dt: A date string in ISO format (YYYY-MM-DD).
        :type dt: str
        :return: The DOI of the previous Meta dump.
        :rtype: str
        """
        meta_dumps_pub_dates = [  # TODO: maybe move inside class init
            ('2022-12-19', 'https://doi.org/10.6084/m9.figshare.21747536.v1'),
            ('2022-12-20', 'https://doi.org/10.6084/m9.figshare.21747536.v2'),
            ('2023-02-15', 'https://doi.org/10.6084/m9.figshare.21747536.v3'),
            ('2023-06-28', 'https://doi.org/10.6084/m9.figshare.21747536.v4'),
            ('2023-10-26', 'https://doi.org/10.6084/m9.figshare.21747536.v5'),
            ('2024-04-06', 'https://doi.org/10.6084/m9.figshare.21747536.v6'),
            ('2024-06-17', 'https://doi.org/10.6084/m9.figshare.21747536.v7'),
            ('2025-02-02', 'https://doi.org/10.6084/m9.figshare.21747536.v8')
        ]
        meta_dumps_pub_dates = sorted([(date.fromisoformat(d), doi) for d, doi in meta_dumps_pub_dates], key=lambda x: x[0])
        d = date.fromisoformat(dt.strip()[:10])
        res = None
        for idx, t in enumerate(meta_dumps_pub_dates):
            if d <= t[0]:
                pos = idx-1 if ((idx-1) >= 0) else 0 # if dt predates the publication date of the absolute first Meta dump, assign the first Meta dump
                prim_source = meta_dumps_pub_dates[pos][1]
                res = prim_source
                break
        if res:
            return res
        else:
            warnings.warn(f'get_previous_meta_dump_uri(): {dt} follows the publication date of the latest Meta dump. The register of published dumps might need to be updated!') 
            return meta_dumps_pub_dates[-1][1] # picks latest dump in the register
    

    def fetch_creation_no_primary_source(self, limit=1000000) -> List[Tuple[str, str]]:
        """
        Fetches creation snapshots that do not have a primary source.
        Returns a list of tuples containing the graph URI and the snapshot URI.
        :return: A list of tuples with graph URI and snapshot URI.
        :rtype: List[Tuple[str, str]]
        """
        results = []

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

        # Pagination loop
        logging.info("Fetching creation snapshots without a primary source with pagination...")
        for current_bindings in self._paginate_query(template, limit):
            for b in current_bindings:
                s = b["s"]["value"]
                gen_time = b["genTime"]["value"]
                results.append((s, gen_time))
        
        return results # (<snapshot uri>, <oldest gen. dt in graph>)
    
    def batch_insert_missing_primsource(self, creations_to_fix: List[Tuple[str, str]], batch_size=500):
        """
        Inserts primary sources for creation snapshots that do not have one, in batches. 
        :param creations_to_fix: A list of tuples where each tuple contains the snapshot URI and the generation time, representing all the creation snapshots that must be fixed.
        :param batch_size: The number of snapshots to process in each batch.
        :type creations_to_fix: List[Tuple[str, str]]
        :type batch_size: int
        :return: None
        """
        template = Template("""
        PREFIX prov: <http://www.w3.org/ns/prov#>

        INSERT DATA {
            $quads
        }
        """)
        for i in range(0, len(creations_to_fix), batch_size):
            batch = creations_to_fix[i:i + batch_size]
            quads = []
            for snapshot_uri, gen_time in batch:
                prim_source_uri = self.get_previous_meta_dump_uri(gen_time)
                graph_uri = get_graph_uri_from_se_uri(snapshot_uri)
                quads.append(f"GRAPH <{graph_uri}> {{ <{snapshot_uri}> prov:hadPrimarySource <{prim_source_uri}> . }}\n")
            quads_str = "    ".join(quads)
            query = template.substitute(quads=quads_str)
            if self.dry_run:
                logging.info(f"[Dry-run] Would insert primary sources for creation snapshots {i} to {len(batch)}.")
            else:
                self._query(query)
                logging.info(f"Inserted primary sources for creation snapshots {i} to {len(batch)}.")
                

    # TODO: (4) Correct snapshots with multiple objects for prov:wasAttributedTo -> move in daughter class MultiPAFixer

    def fetch_snapshots_multi_pa(self, limit=10000):
        """
        Fetches graph-snapshot pairs where the snapshot has more than one object for the ``prov:wasAttributedTo`` property.
        """
        result = []

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

        logging.info("Fetching snapshots with multiple objects for prov:wasAttributedTo...")
        for current_bindings in self._paginate_query(template, limit):
            for b in current_bindings:
                g = b['g']['value']
                s = b['s']['value']
                result.append((g, s))
        logging.info(f"Found {len(result)} snapshots with multiple objects for prov:wasAttributedTo.")
        return result

    def batch_delete_extra_pa(self, multi_pa_snapshots:list, batch_size=500):
        """
        Deletes triples where the value of prov:wasAttributedTo is <https://w3id.org/oc/meta/prov/pa/1> if there is at least another processing agent for the same snapshot subject. 
        """
        template = Template("""
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        DELETE DATA {
          $quads_to_delete
        }
        """)

        for i in range(0, len(multi_pa_snapshots), batch_size):
            batch = multi_pa_snapshots[i:i + batch_size]
            to_delete = []
            for g, s in batch:
                to_delete.append(f"GRAPH <{g}> {{ <{s}> prov:wasAttributedTo <https://w3id.org/oc/meta/prov/pa/1> . }}\n")

            to_delete_str = "  ".join(to_delete)
            query = template.substitute(quads_to_delete=to_delete_str)
            if self.dry_run:
                logging.info(f"[Dry-run] Would delete triples with default processing agent from snapshots {i} to {len(batch)}.")
            else:
                self._query(query)
                logging.info(f"Deleted triples with default processing agent from snapshots {i} to {len(batch)}.")


    # (5) Correct graphs where at least one snapshots has too many objects for specific properties -> move to daughter class MultiObjectFixer
        
    def fetch_multiple_objects_graphs(self, limit=10000) -> List[str]:
        """
        Fetches graphs containing at least one snapshot with multiple objects for 
        a property that only admits one (e.g. oc:hasUpdateQuery).

        :return: A list of distinct graph URIs.
        """
        output = []

        template = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX oc: <https://w3id.org/oc/ontology/>

        SELECT ?g
        WHERE {
          {
            SELECT DISTINCT ?g WHERE {
                GRAPH ?g {
                    VALUES ?p {
                    prov:invalidatedAtTime
                    prov:hadPrimarySource
                    oc:hasUpdateQuery
                    }
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
        }
        OFFSET %d
        LIMIT %d
        """

        logging.info("Fetching URIs of graphs containing snapshots with too many objects...")
        for current_bindings in self._paginate_query(template, limit):
            for b in current_bindings:
                g = b['g']['value']
                output.append(g)
        
        logging.info(f"Found {len(output)} distinct graphs containing snapshots with too many objects for some properties.")
        return output
    
    def reset_multi_object_graphs(self, graphs:list, prim_source_uri:str, pa_uri:str):
        """
        Resets each graph in graphs by deleting the existing snapshots and creating a new 
        creation snapshot, which will be the only one left for that graph. 

        :param graphs: A list of distinct graph URIs that have too many objects for properties that only admit one.
        :param prim_source_uri: The URI of the primary source to insert as object of prov:hadPrimarySource in the creation snapshot.
        :param pa_uri: The URI of the processing agent to insert as object of prov:wasAttributedTo in the creation snapshot.
        """

        template = Template("""
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        #WITH GRAPH <$graph> # WITH clause seems not to be supported in rdflib
        DELETE {
          GRAPH <$graph> {
            ?s ?p ?o
          }
        }
        INSERT {
          GRAPH <$graph> {
            <$creation_snapshot> prov:hadPrimarySource <$primary_source> ;
              prov:wasAttributedTo <$processing_agent> ;
              prov:specializationOf <$specialization_of> ;
              dcterms:description "$description"^^xsd:string ;
              rdf:type prov:Entity ;
              prov:generatedAtTime ?minGenTime .
          }
        }
        WHERE {
          {
            SELECT (MIN(?genTime) AS ?minGenTime) WHERE {
              GRAPH <$graph> {
                ?_s prov:generatedAtTime ?genTime .
              }
            }
          }
          GRAPH <$graph> {
            ?s ?p ?o .
          }
        }
        """)

        logging.info("Resetting graphs with too many objects by creating a new single creation snapshot...")

        for g in tqdm(graphs):
            creation_se = g + 'se/1'
            prim_source = prim_source_uri  # TODO: move prim_source_uri to class init instead of this function params
            processing_agent = pa_uri # TODO: move pa_uri to class init instead of this function params
            referent = get_described_res_omid(g)
            desc = f"The entity '{referent}' has been created."

            query = template.substitute(
                graph = g,
                creation_snapshot = creation_se,
                primary_source = prim_source, 
                processing_agent = processing_agent,
                specialization_of = referent, 
                description = desc
            )
            
            if self.dry_run:
                logging.info(f"[Dry-run] Would reset {g} in the triplestore.")
            else:
                self._query(query)
                logging.info(f"Overwritten {g} with new creation snapshot.")
