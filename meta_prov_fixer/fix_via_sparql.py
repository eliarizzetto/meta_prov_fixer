import logging
import yaml
import time
import argparse
from typing import List, Tuple, Union, Dict, Set, Iterable
from pathlib import Path
from string import Template

from rdflib import Graph, ConjunctiveGraph, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON, JSONLD, TURTLE, POST
from tqdm import tqdm

from collections import defaultdict
from datetime import date
import warnings

from meta_prov_fixer.utils import get_seq_num, remove_seq_num, get_described_res_omid, get_graph_uri_from_se_uri


class ProvenanceFixer:
    def __init__(self, sparql_endpoint: str = 'http://localhost:8890/sparql/', queries_fp: str = 'nuovo_queries.yaml', auth: Union[str, None] = None, dry_run: bool = False):
        self.endpoint = sparql_endpoint
        self.dry_run = dry_run
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)
        if auth:
            self.sparql.addCustomHttpHeader("Authorization", auth)

        self._load_queries(queries_fp)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _load_queries(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.queries = yaml.safe_load(f)

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

    def get_snapshots_to_delete(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Fetches snapshots that are fillers and need to be deleted, grouped by their named graph.
        Returns a dictionary where keys are graph URIs and values are dictionaries with 'to_delete' and 'remaining_snapshots' sets,
        storing respectively the URIs of the graph's snapshots that should be deleted and the URIs of the other snapshots.
        :return: A dictionary with graph URIs as keys and dictionaries with 'to_delete' and 'remaining_snapshots' sets as values.
        :rtype: Dict[str, Dict[str, Set[str]]]
        """
        query = self.queries['select_fillers']
        logging.info("Fetching snapshots to delete...")
        query_result = self._query(query)

        if not query_result:
            return {}

        grouped_result = defaultdict(lambda: {'to_delete': set(), 'remaining_snapshots': set()})
        
        # group query results
        for binding in query_result['results']['bindings']:
            g = binding['g']['value']
            snapshot = binding['snapshot']['value']
            other_se = binding['other_se']['value']
            
            grouped_result[g]['to_delete'].add(snapshot)
            grouped_result[g]['remaining_snapshots'].add(other_se)

        logging.info(f"{sum([len(d['to_delete']) for d in grouped_result.values()])} snapshots marked for deletion.")
        return dict(grouped_result)

    def delete_snapshots(self, deletions: Dict[str, Dict[str, Set[str]]]) -> None:
        """
        Deletes snapshots from the triplestore based on the provided deletions dictionary.
        :param deletions: A dictionary where keys are graph URIs and values are dictionaries with 'to_delete' and 'remaining_snapshots' sets.
        :type deletions: Dict[str, Dict[str, Set[str]]]
        """
        template = Template(self.queries['delete_multi_snapshot'])

        for graph_uri, values in tqdm(deletions.items(), desc="Deleting snapshots"):
            to_delete = " ".join([f"<{se_uri}>" for se_uri in sorted(values['to_delete'], key=lambda x: get_seq_num(x))])
            query = template.substitute(graph=graph_uri, snapshots_to_delete=to_delete)
            if self.dry_run:
                logging.info(f"[Dry-run] Would delete the following snapshots from graph {graph_uri}: {to_delete}")
            else:
                self._query(query)
                logging.info(f"Deleted from graph {graph_uri} the following snapshots: {to_delete}")

    def map_se_names(self, to_delete:set, remaining: set) -> dict:
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

        # !IMPORTANT: mapping is sorted ascendingly by the sequence number of old URI (get_sequence_number on mapping's keys)
        # This is required, otherwise newly inserted URIs might be deleted when iterating over mapping's items
        mapping = dict(sorted(mapping.items(), key=lambda i: get_seq_num(i[0])))
        if self.dry_run:
            logging.info(f"[Dry-run] Would rename snapshots in the whole graph according to the following mapping: {mapping}")
            return
        
        template = Template(self.queries['rename_snapshots_global'])

        logging.info("Renaming snapshots in the triplestore...")
        for old_uri, new_uri in mapping.items():
            query = template.substitute(old_uri=old_uri, new_uri=new_uri)
            # self.sparql.setQuery(query)  # is this line necessary?
            self._query(query)
        logging.info(f"Snapshot entities were re-named in all named graphs according to the following mapping: {mapping}.")
    


    # TODO: (2) CORREZIONE DATETIME -> sposta in classe separata

    def adapt_invalidatedAtTime(self, graph_uri: str, snapshots: list) -> None:
        """
        Updates the ``invalidatedAtTime`` property of each snapshot in the provided list to match 
        the value of ``generatedAtTime`` of the following snapshot.
        :param graph_uri: The URI of the named graph containing the snapshots.
        :type graph_uri: str
        :param snapshots: A list of snapshot URIs sorted by their sequence number.
        :type snapshots: list
        """
        
        snapshots = sorted(snapshots, key=lambda x: get_seq_num(x))  # sorting is required!
        template = Template(self.queries['adapt_invalidatedAtTime'])

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


    # TODO: (3) CORREZIONE DEGLI SNAPSHOT DI CREAZIONE SENZA PRIMARY SOURCE -> sposta in classe separata

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
    
    def get_creation_no_primary_source(self, limit=1000000) -> List[Tuple[str, str]]:
        """
        Fetches creation snapshots that do not have a primary source.
        Returns a list of tuples containing the graph URI and the snapshot URI.
        :return: A list of tuples with graph URI and snapshot URI.
        :rtype: List[Tuple[str, str]]
        """
        # query = self.queries['select_creation_without_primsource']
        # logging.info("Fetching creation snapshots without a primary source...")
        # query_result = self._query(query)

        # if not query_result:
        #     logging.info("No creation snapshots without a primary source found.")
        #     return []

        # output = [(binding['s']['value'], binding['genTime']['value']) for binding in query_result['results']['bindings']]
        # logging.info(f"Found {len(output)} creation snapshots without a primary source.")

        # return output

        offset = 0
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
        while True:
            q = template % (offset, limit)
            logging.info(f"Fetching results {offset} to {offset + limit}...")
            result_batch = self._query(q)
            bindings = result_batch["results"]["bindings"]

            if not bindings:
                break

            for binding in bindings:
                s = binding["s"]["value"]
                gen_time = binding["genTime"]["value"]
                results.append((s, gen_time))

            offset += limit
            time.sleep(0.5)
        
        return results # (<snapshot uri>, <oldest gen. dt in graph>)

    def insert_missing_primsource(self, snapshot_uri, prim_source_uri):
        """
        Inserts 'prim_source_uri' as the object of prov:hadPrimarySource for the entity identified by 'snapshot_uri'.
        :param snapshot_uri: The URI of the (creation) snapshot that is missing a primary source.
        :param prim_source_uri: The URI of the primary source to add (should be the DOI of an OpenCitations Meta dump version).
        """
        template = Template(self.queries['insert_prim_source'])
        query = template.substitute(
            snapshot = snapshot_uri,
            source = prim_source_uri
        )
            
        if self.dry_run:
            logging.info(f"[Dry-run] Would assign {prim_source_uri} as primary source of {snapshot_uri} in the triplestore.")
        else:
            self._query(query)
            logging.info(f"Added {prim_source_uri} as primary source of {snapshot_uri}.")
    
    def batch_insert_missing_primsource(self, creations_to_fix: List[Tuple[str, str]], batch_size=500):
        """
        Inserts primary sources for creation snapshots that do not have one, in batches. 
        :param creations_to_fix: A list of tuples where each tuple contains the snapshot URI and the generation time, representing all the creation snapshots that must be fixed.
        :param batch_size: The number of snapshots to process in each batch.
        :type creations_to_fix: List[Tuple[str, str]]
        :type batch_size: int
        :return: None
        """
        # da implementare: usa INSERT DATA senza WHERE. La query che mandi dev'essere del tipo:
        # INSERT DATA { GRAPH <$grafoX> {<$snapshotX1> prov:hadPrimarySource <$sourceX1>} GRAPH <$grafoY> {<$snapshotY1> prov:hadPrimarySource <$sourceY1> } GRAPH {...} }, 
        # in questa funzione devi anche assegnare a `prim_source_uri` il valore di `get_previous_meta_dump_uri(snapshot_uri)` per ogni snapshot che non ha un primary source.
        template = Template("""
        PPREFIX prov: <http://www.w3.org/ns/prov#>

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
                

    # TODO: (4) CORREZIONE DEGLI SNAPSHOT CON PIù VALORI PER wasAttributedTo -> sposta in classe separata


    def update_snapshots_multi_pa(self):
        """
        Deletes triples where the value of prov:AttributedTo is <https://w3id.org/oc/meta/prov/pa/1> if there is at least another processing agent for the same snapshot subject. 
        """
        query = self.queries['delete_default_pa']
        if self.dry_run:
            logging.info("[Dry-run] Would delete triples with default processing agent from snapshots with multiple processing agents.")
        else:
            logging.info("Deleting triples with default processing agent from snapshots with multiple processing agents...")
            self._query(query)
            logging.info("Deleted triples with default processing agent from snapshots with multiple processing agents.")


    # TODO: (5) CORREZIONE DEI GRAFI CON SNAPSHOTS CON TROPPI OGGETTI -> sposta in classe separata

    def get_multiple_objects_graphs(self) -> list:
        """
        Fetches graphs containing at least one snapshot with multiple objects for 
        a property that only admits one (e.g. oc:hasUpdateQuery).

        :return: A list of distinct graph URIs.
        """
        query = self.queries['select_multi_values_graphs']
        logging.info("Fetching URIs of graphs containing snapshots with too many objects...")
        query_result = self._query(query)

        if not query_result:
            logging.info("No graphs fetched.")
            return []
        
        output = [binding['g']['value'] for binding in query_result['results']['bindings']]
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

        template = Template(self.queries['reset_graph_with_new_creation_se'])

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

    
    # TODO: create parent class and rename + restructure this class as specific process for handling filler snapshots

    def process(self):
        start = time.time()

        to_delete = self.get_snapshots_to_delete()
        if not to_delete:
            logging.info("No snapshots to delete. Exiting.")
            return

        self.delete_snapshots(to_delete)

        for graph_uri, d in tqdm(to_delete.items(), desc="Renaming snapshots"):
            mapping = self.map_se_names(d['to_delete'], d['remaining_snapshots'])
            self.rename_snapshots(mapping)

            current_snapshots_sorted = sorted({v for v in mapping.values()}, key=lambda x: get_seq_num(x))
            self.adapt_invalidatedAtTime(graph_uri, current_snapshots_sorted)
            

        elapsed = time.time() - start
        logging.info(f"Process completed in {elapsed:.2f} seconds.")


def cli():
    parser = argparse.ArgumentParser(description="Fix provenance snapshot sequences by deleting and renumbering.")
    parser.add_argument('--endpoint', type=str, required=True, help='SPARQL endpoint URL')
    parser.add_argument('--queries', type=str, default="queries.yaml", help='Path to YAML file with SPARQL queries')
    parser.add_argument('--auth', type=str, default=None, help='Optional Authorization header')
    parser.add_argument('--dry-run', action='store_true', help='Do not modify the triplestore (simulate actions)')

    args = parser.parse_args()

    fixer = ProvenanceFixer(
        sparql_endpoint=args.endpoint,
        queries_fp=args.queries,
        auth=args.auth,
        dry_run=args.dry_run
    )
    fixer.process()


# if __name__ == "__main__":
#     cli()
