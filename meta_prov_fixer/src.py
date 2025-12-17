from rdflib import Graph, Dataset, URIRef, Literal
from rdflib.namespace import XSD, PROV, DCTERMS, RDF
from typing import List, Union, Dict, Tuple, Generator, Iterable, Optional
from meta_prov_fixer.utils import remove_seq_num, get_seq_num, normalise_datetime, get_previous_meta_dump_uri, get_described_res_omid, \
read_rdf_dump, get_rdf_prov_filepaths, batched, get_graph_uri_from_se_uri
import logging
from string import Template
import re
import json
from tqdm import tqdm
from sparqlite import SPARQLClient, QueryError, EndpointError
from typing import TextIO
from datetime import datetime
import os
from pathlib import Path


class FillerFixerFile:

    def __init__(self, endpoint):
        pass
        self.endpoint = endpoint

    def detect(graph:Graph) -> Optional[Tuple[URIRef, Dict[str, List[URIRef]]]]:
        """
        Detects the issues in the input graph. If no filler snapshots are found, return None. 
        Else, a 2-elements tuple is returned, where the first element is the URIRef object of the 
        graph's identifier, and the second element is a dictionary with "to_delete" and 
        "remaining_snapshots" keys, both having as their value a list of URIRef objects, respectively 
        representing the fillers snapshots that must be deleted and the snapshots that should be kept 
        (but must be renamed).

        :param graph: the named graph for the provenance of an entity
        """

        # out = (URIRef(graph.identifier), {"to_delete":[], "remaining_snapshots":[]})

        snapshots = list(graph.subjects(unique=True))
        if len(snapshots) == 1:
            return None
        
        creation_se = URIRef(str(graph.identifier) + 'se/1')
        fillers = set()
        remaining = set()

        for s in snapshots:
            if s == creation_se:
                remaining.add(s)
                continue
            if (s, URIRef('https://w3id.org/oc/ontology/hasUpdateQuery'), None) not in graph:
                if (s, DCTERMS.description, None) not in graph:
                    fillers.add(s)
                else:
                    for desc_val in graph.objects(s, DCTERMS.description, unique=True):
                        if "merged" not in str(desc_val).lower():
                            fillers.add(s)
            if s not in fillers:
                remaining.add(s)

        if not fillers:
            return None

        out = (
            URIRef(graph.identifier), 
            {
                "to_delete":list(fillers), 
                "remaining_snapshots":list(remaining)
            }
        )

        return out


    @staticmethod
    def map_se_names(to_delete:set, remaining: set) -> dict:
        """
        Associates a new URI value to each snapshot URI in the union of ``to_delete`` and ``remaining`` (containing snapshot URIs).

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
        :param remaining: A set of URIs of snapshots that should remain in the graph (AFTER BEING RENAMED).
        :type remaining: set
        :returns: A dictionary mapping old snapshot URIs to their new URIs.
        :rtype: dict
        """
        to_delete :set = {str(el) for el in to_delete}
        remaining :set = {str(el) for el in remaining}
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

    @staticmethod
    def make_global_rename_map(graphs_with_fillers:Iterable):
        """
        Create a dictionary of the form <old_subject_uri>:<new_snapshot_name>.
        
        :param graphs_with_fillers: an Iterable consisting of all the graphs containing filler snapshots,
            the snapshots to delete in that graph, and the other snapshot in that same graph (part of 
            which must be renamed).
        :type graphs_with_fillers: Iterable
        """
        out = dict()

        for g, _dict in graphs_with_fillers:
            mapping = FillerFixerFile.map_se_names(_dict['to_delete'], _dict['remaining_snapshots'])
            for k, v in mapping.items():
                if k != v:
                    out[k] = v
        return out

    
    def fix_local_graph(ds: Dataset, graph:Graph, global_rename_map:dict) -> None:

        # delete all triples where subject is a filler (in local graph)
        for snapshot_node, _, _  in graph.triples((None, None, None)):
            if snapshot_node in global_rename_map:
                ds.remove((URIRef(snapshot_node), None, None))
        
        # replace objects that used to be fillers snapshots (in this graph or in other graphs, using global mapping)
        for subj, pred, obj in graph.triples((None, None, None)):
            if str(obj) in global_rename_map:
                replacement = URIRef(global_rename_map[str(obj)])
                ds.set((subj, pred, replacement))

        # adapt invalidatedAtTime relationships (in local graph only)
        snapshots_strings:list = sorted(list(str(s) for s in graph.subjects(unique=True)), key=lambda x: get_seq_num(x))
        for s, following_se in zip(snapshots_strings, snapshots_strings[1:]):
            new_invaldt:URIRef = min(list(graph.objects(URIRef(following_se), PROV.generatedAtTime, unique=True)))
            ds.set((URIRef(s), PROV.invalidatedAtTime, new_invaldt))

    def build_delete_sparql_query(local_deletions:Union[tuple, list])->str:
        """
        Makes the SPARQL query text for deleting snapshots from the triplestore based on the provided deletions list.

        :param deletions: A tuple or list where the first element is a graph URI, 
            and the second is a dictionary with `'to_delete'` and `'remaining_snapshots'` sets.
        """

        deletion_template = Template("""
            $dels
        """)

        # step 1: delete filler snapshots in the role of subjects
        dels = []
        for g_uri, values in local_deletions:
            for se_to_delete in values['to_delete']:
                single_del = f"DELETE WHERE {{ GRAPH <{str(g_uri)}> {{ <{str(se_to_delete)}> ?p ?o . }}}};\n"
                dels.append(single_del)
        dels_str = "    ".join(dels)
        query_str = deletion_template.substitute(dels=dels_str)
        return query_str
    
    def build_rename_sparql_query(local_mapping:dict) -> str:
        """
        Makes the SPARQL query text to rename snapshots in the triplestore according to the provided mapping.

        :param local_mapping: A dictionary where keys are old snapshot URIs and values are new snapshot URIs.
        :type local_mapping: dict
        """

        mapping = dict(sorted(local_mapping.items(), key=lambda i: get_seq_num(i[0])))
        
        per_snapshot_template = Template("""
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


        per_snapshot_portions = []
        for old_uri, new_uri in mapping.items():
            if old_uri == new_uri:
                continue
            query_portion = per_snapshot_template.substitute(old_uri=old_uri, new_uri=new_uri)
            per_snapshot_portions.append(query_portion)
        
        final_query_str = ";\n".join(per_snapshot_portions)

        return final_query_str


    def build_adapt_invaltime_sparql_query(graph_uri: str, local_snapshots: list) -> str:
        """
        Update the ``prov:invalidatedAtTime`` property of each snapshot in the provided list to match 
        the value of ``prov:generatedAtTime`` of the following snapshot.

        :param graph_uri: The URI of the named graph containing the snapshots.
        :type graph_uri: str
        :param local_snapshots: A list of snapshot URIs.
        :type local_snapshots: list
        :returns: None
        """
        
        snapshots = sorted(local_snapshots, key=lambda x: get_seq_num(x))  # sorting is required!
        per_snaphot_template = Template("""
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

        per_snapshot_portions = []

        for s, following_se in zip(snapshots, snapshots[1:]):
            query_portion = per_snaphot_template.substitute(
                graph=graph_uri,
                snapshot=s,
                following_snapshot=following_se
            )

            per_snapshot_portions.append(query_portion)

        final_query_str = "PREFIX prov: <http://www.w3.org/ns/prov#>\n" + ";\n".join(per_snapshot_portions)

        return final_query_str



class DateTimeFixerFile:

    def __init__(self):
        pass

    def detect(graph:Graph) -> Optional[List[Tuple[URIRef]]]:

        result = []
        pattern_dt = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:(?:\+00:00)|Z)$"

        for s, p, o in graph.triples((None, None, None)):
            if p in (PROV.generatedAtTime, PROV.invalidatedAtTime):
                if not re.match(pattern_dt, str(o)):
                    result.append((graph.identifier, s, p, o))
        return result
    
    def fix_local_graph(ds:Dataset, graph:Graph, to_fix:tuple) -> None:

        for g_uri, subj, prop, obj in to_fix:
            correct_dt_res = Literal(normalise_datetime(str(obj)), datatype=XSD.dateTime)
            ds.set((subj, prop, correct_dt_res))
    
    def build_update_query(to_fix:List[Tuple[URIRef]]):
        template = Template('''
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        DELETE DATA {
        $to_delete
        } ;
        INSERT DATA {
        $to_insert
        }
        ''')

        to_delete = []
        to_insert = []
        for g, s, p, dt in to_fix:
            g = str(g)
            s = str(s)
            p = str(p)
            dt = str(dt)

            to_delete.append(f'GRAPH <{g}> {{ <{s}> <{p}> "{dt}"^^xsd:dateTime . }}\n')
            correct_dt = normalise_datetime(dt)
            to_insert.append(f'GRAPH <{g}> {{ <{s}> <{p}> "{correct_dt}"^^xsd:dateTime . }}\n')
        
        to_delete_str = "  ".join(to_delete)
        to_insert_str = "  ".join(to_insert)
        query = template.substitute(to_delete=to_delete_str, to_insert=to_insert_str)
        return query


class MissingPrimSourceFixerFile:
    
    def __init__(self, meta_dumps_pub_dates):
        self.meta_dumps = meta_dumps_pub_dates

    def detect(graph:Graph) -> Optional[Tuple[URIRef, Literal]]:

        creation_se_uri = URIRef(graph.identifier + 'se/1')
        if ((creation_se_uri, PROV.generatedAtTime, None) in graph) and not ((creation_se_uri, PROV.hadPrimarySource, None) in graph):
            genTime = min(graph.objects(creation_se_uri, PROV.generatedAtTime))
            return (creation_se_uri, genTime)

    def fix_local_graph(ds:Dataset, graph:Graph, to_fix:tuple, meta_dumps) -> None:
        
        primSource_str = get_previous_meta_dump_uri(meta_dumps, str(to_fix[1]))
        primSource_literal = Literal(primSource_str)
        ds.add((to_fix[0], PROV.hadPrimarySource, primSource_literal, graph.identifier))
    

    def build_update_query(to_fix:List[Tuple[URIRef, Literal]], meta_dumps):

        template = Template("""
        PREFIX prov: <http://www.w3.org/ns/prov#>

        INSERT DATA {
            $quads
        }
        """)

        fixes = []
        for snapshot_uri, gen_time in to_fix:
            snapshot_uri = str(snapshot_uri)
            gen_time = str(gen_time)
            prim_source_uri = get_previous_meta_dump_uri(meta_dumps, gen_time)
            graph_uri = get_graph_uri_from_se_uri(snapshot_uri)
            fixes.append(f"GRAPH <{graph_uri}> {{ <{snapshot_uri}> prov:hadPrimarySource <{prim_source_uri}> . }}\n")
        quads_str = "    ".join(fixes)
        query = template.substitute(quads=quads_str)

        return query



class MultiPAFixerFile:

    def __init__(self):
        pass

    def detect(graph:Graph) -> Optional[List[Tuple[URIRef]]]:
        result = []
        for s, _, o in graph.triples((None, PROV.wasAttributedTo, None)):
            processing_agents = list(graph.objects(s, PROV.wasAttributedTo, unique=True))
            if len(processing_agents) > 1 and URIRef('https://w3id.org/oc/meta/prov/pa/1') in processing_agents:
                
                result.append((graph.identifier, s))
        return result
    

    def fix_local_graph(ds:Dataset, graph:Graph, to_fix:List[Tuple[URIRef]]) -> None:
        
        for g_uri, subj in to_fix:
            ds.set((subj, PROV.wasAttributedTo, URIRef('https://w3id.org/oc/meta/prov/pa/2')))
    

    def build_update_query(to_fix:List[Tuple[URIRef]]):

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
        
        to_delete = []
        to_insert = []
        for g, s in to_fix:
            g = str(g)
            s = str(s)
            to_delete.append(f"GRAPH <{g}> {{ <{s}> prov:wasAttributedTo <https://orcid.org/0000-0002-8420-0696> . }}\n")  # deletes Arcangelo's ORCID
            to_delete.append(f"GRAPH <{g}> {{ <{s}> prov:wasAttributedTo <https://w3id.org/oc/meta/prov/pa/1> . }}\n")  # deletes Meta's default processing agent (for ingestions only)
            to_insert.append(f"GRAPH <{g}> {{ <{s}> prov:wasAttributedTo <https://w3id.org/oc/meta/prov/pa/2> . }}\n")  # inserts Meta's processsing agent for modification processes

        to_delete_str = "  ".join(to_delete)
        to_insert_str = "  ".join(to_insert)
        query = template.substitute(quads_to_delete=to_delete_str, quads_to_insert=to_insert_str)
        
        return query
    

class MultiObjectFixerFile:

    def __init__(self):
        pass

    def detect(graph:Graph) -> Optional[Tuple[URIRef, Literal]]:

        creation_se_uri = URIRef(graph.identifier + 'se/1')

        for prop in {PROV.invalidatedAtTime, PROV.hadPrimarySource, URIRef('https://w3id.org/oc/ontology/hasUpdateQuery')}:
            for s in graph.subjects():
                if len(list(graph.objects(s, prop, unique=True))) > 1:
                    creation_gen_time = min(graph.objects(creation_se_uri, PROV.generatedAtTime, unique=True))
                    return (graph.identifier, creation_gen_time)
    

    def fix_local_graph(ds:Dataset, graph:Graph, to_fix:tuple, meta_dumps) -> None:

        creation_se_uri = URIRef(graph.identifier + 'se/1')
        genTime = to_fix[1]
        primSource_str = get_previous_meta_dump_uri(meta_dumps, str(genTime))
        primSource_literal = Literal(primSource_str)
        referent = URIRef(get_described_res_omid(str(creation_se_uri)))
        desc = Literal(f"The entity '{str(referent)}' has been created.")
        triples_to_add = (
            (creation_se_uri, PROV.hadPrimarySource, primSource_literal),
            (creation_se_uri, PROV.wasAttributedTo, URIRef('https://w3id.org/oc/meta/prov/pa/1')),
            (creation_se_uri, PROV.specializationOf, referent),
            (creation_se_uri, DCTERMS.description, desc),
            (creation_se_uri, RDF.type, PROV.Entity),
            (creation_se_uri, PROV.generatedAtTime, genTime)
        )

        ds.remove((None, None, None, graph.identifier))
        for t in triples_to_add:
            quad = t + (graph.identifier, )
            ds.add(quad)
    

    def build_update_query(to_fix, meta_dumps, pa_uri="https://w3id.org/oc/meta/prov/pa/1"):

        prefixes = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n\n
        """
        
        per_graph_template = Template("""
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

        query_parts = []
        for g, gen_time in to_fix:
            g = str(g)
            gen_time = str(gen_time)
            creation_se = g + 'se/1'
            gen_time = gen_time.replace("^^xsd:dateTime", "") 
            gen_time = gen_time.replace("^^http://www.w3.org/2001/XMLSchema#dateTime", "")
            prim_source = get_previous_meta_dump_uri(meta_dumps, gen_time)
            processing_agent = pa_uri 
            referent = get_described_res_omid(g)
            desc = f"The entity '{referent}' has been created."

            per_graph_part = per_graph_template.substitute(
                graph = g,
                creation_snapshot = creation_se,
                primary_source = prim_source, 
                processing_agent = processing_agent,
                specialization_of = referent, 
                description = desc,
                gen_time = gen_time
            )
            query_parts.append(per_graph_part)
        
        query = prefixes + " ; \n\n".join(query_parts)

        return query



def prepare_filler_issues(data_dir)->Tuple[List[tuple], int]:

    result = []
    tot_files = len(get_rdf_prov_filepaths(data_dir))

    for file_data in tqdm(
            read_rdf_dump(data_dir, whole_file=True),
            desc=f'Detecting graphs with fillers for all graphs in {data_dir}',
            total=tot_files
        ):
        stringified_data = json.dumps(file_data)
        d = Dataset(default_union=True)
        d.parse(data=stringified_data, format='json-ld')

        for graph in d.graphs():
            issues_in_graph = FillerFixerFile.detect(graph)
            if issues_in_graph:
                result.append(issues_in_graph)
    return result, tot_files



def sparql_update(
    client: SPARQLClient,
    update_query: str,
    failed_log: TextIO,
) -> bool:
    """
    Execute a SPARQL UPDATE via client.update().

    Uses the client's built-in retry settings. If the update still fails
    after all retries, writes the query to `failed_log`.

    Returns:
        True if the update succeeded, False if it failed and was logged.
    """
    try:
        client.update(update_query)
        return True  # success

    except (QueryError, EndpointError) as exc:
        # log both syntax errors and endpoint errors that weren't recoverable
        failed_log.write(update_query.replace("\n", "\\n") + "\n")
        failed_log.write(f"# Failure: {type(exc).__name__}: {exc}\n\n")
        failed_log.flush()
        return False



def process(
        endpoint, 
        data_dir, 
        meta_dumps_register, 
        out_dir,
        chunk_size=100, 
        failed_queries_fp=f"prov_fix_failed_queries_{datetime.today().strftime('%Y-%m-%d')}.txt",
        overwrite = False
    ):

    os.makedirs(out_dir, exist_ok=True)

    client = SPARQLClient(endpoint)
    failed_queries_log = open(failed_queries_fp, 'a', encoding='utf-8')

    filler_issues, tot_files = prepare_filler_issues(data_dir)  
    # TODO: meccanismo per salvare filler_issues su file e controllarne la presenza in base al file

    rename_mapping = FillerFixerFile.make_global_rename_map(filler_issues)
    graphs_with_fillers = {t[0]:t[1] for t in filler_issues}  # {<URIRef>: {"to_delete":[...], "remaining_snapshots":[...]}}
    meta_dumps = meta_dumps_register # TODO: forse funzione per normalizzare meta_dumps_register?


    for file_data, fp in tqdm(read_rdf_dump(data_dir, whole_file=True, include_fp=True), total=tot_files):

        fixed_fp = os.path.join(out_dir, str(Path(fp).parent)) + '.json'  # destination file of corrected dataset
        os.makedirs(fixed_fp, exist_ok=True)
        if os.path.isfile(fixed_fp) and not overwrite:
            raise FileExistsError(f"The file {fixed_fp} already exists. To overwrite it, set the 'overwrite' parameter to True.")
        
        stringified_data = json.dumps(file_data)
        d = Dataset(default_union=True)
        d.parse(data=stringified_data, format='json-ld')

        ff_issues_in_file = [] # Issues for FillerFixer in current file
        dt_issues_in_file = [] # Issues for DateTimeFixer in current file
        mps_issues_in_file = [] # Issues for MissingPrimSourceFixer in current file
        multi_pa_issues_in_file = [] # Issues for MultiPAFixer in current file
        multi_obj_issues_in_file = [] # Issues for MultiObjectFixer in current file

        # FillerFixer rinomina gli snapshot, quindi fai in modo che gli step successivi leggano il grafo eventualmente modificato
        for graph in d.graphs():

            # ------ FillerFixer ------
            ff_to_fix_val = graphs_with_fillers.get(graph.identifier)
            if ff_to_fix_val:
                ff_issues_in_file.append((graph.identifier, ff_to_fix_val))
                FillerFixerFile.fix_local_graph(d, graph, rename_mapping)
                            
        for graph in d.graphs():
            # ------ DateTimeFixer ------
            local_dt_issues = DateTimeFixerFile.detect(graph)
            if local_dt_issues is not None:
                dt_issues_in_file.append(local_dt_issues)
                DateTimeFixerFile.fix_local_graph(d, graph, local_dt_issues)

            # ------ MissingPrimSourceFixer ------
            local_no_source_issues = MissingPrimSourceFixerFile.detect(graph)
            if local_no_source_issues is not None:
                mps_issues_in_file.append(local_no_source_issues)
                MissingPrimSourceFixerFile.fix_local_graph(d, graph, local_no_source_issues, meta_dumps)
            
            # ------ MultiPAFixer ------
            local_multi_pa_issues = MultiPAFixerFile.detect(graph)
            if local_multi_pa_issues is not None:
                multi_pa_issues_in_file.extend(local_multi_pa_issues)
                MultiPAFixerFile.fix_local_graph(d, graph, local_multi_pa_issues)

            # ------ MultiObjectFixer ------
            local_multi_object_issues = MultiObjectFixerFile.detect(graph)
            if local_multi_object_issues is not None:
                multi_obj_issues_in_file.append(local_multi_object_issues)
                MultiObjectFixerFile.fix_local_graph(d, graph, local_multi_object_issues, meta_dumps)
        
        # Fix triplestore 
        for idx, chunk in enumerate(batched(ff_issues_in_file, chunk_size)):
            for t in chunk: # queries for FillerFixer are executed per-graph
                g_id = str(t[0])
                to_delete = [str(i) for i in t[1]['to_delete']]
                to_rename = [str(i) for i in t[1]['remaining_snapshots']]
                local_mapping = FillerFixerFile.map_se_names(to_delete, to_rename)
                newest_names = list(set(local_mapping.values()))

                del_q = FillerFixerFile.build_delete_sparql_query(to_delete)
                sparql_update(client, del_q, failed_queries_log)  # send deletion query
                rename_q = FillerFixerFile.build_rename_sparql_query(local_mapping)
                sparql_update(client, rename_q, failed_queries_log)  # send query for renaming the rest of the snapshots

                adapt_dt_q = FillerFixerFile.build_adapt_invaltime_sparql_query(g_id, newest_names)
                sparql_update(client, adapt_dt_q, failed_queries_log)  # send query for adapting invalidatedAtTime relations among remaining snapshots

        # N.B.: the steps below are strictly ordered
        for idx, chunk in enumerate(batched(dt_issues_in_file, chunk_size)):
            q = DateTimeFixerFile.build_update_query(chunk)
            sparql_update(client, q, failed_queries_log) 
        
        for idx, chunk in enumerate(batched(mps_issues_in_file, chunk_size)):
            q = MissingPrimSourceFixerFile.build_update_query(chunk, meta_dumps)
            sparql_update(client, q, failed_queries_log) 

        for idx, chunk in enumerate(batched(multi_pa_issues_in_file)):
            q = MultiPAFixerFile.build_update_query(chunk)
            sparql_update(client, q, failed_queries_log) 

        for idx, chunk, in enumerate(batched(multi_obj_issues_in_file)):
            q = MultiObjectFixerFile.build_update_query(chunk, meta_dumps)
            sparql_update(client, q, failed_queries_log)
        
        # Save corrected dataset to file
        out_data = d.serialize(format='json-ld')
        with open(fixed_fp, 'w', encoding='utf-8') as out_file:
            out_file.write(out_data)
        
    
    client.close() # close connection to SPARQL endpoint
    failed_queries_log.close() 






            
            
            
