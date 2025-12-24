import unittest
from unittest.mock import patch, Mock

from rdflib.plugins.sparql.processor import prepareQuery, SPARQLResult
from rdflib import Dataset, URIRef, Literal, PROV, Graph, Literal

import json
import re
from pprint import pprint
from datetime import datetime
from rdflib.compare import isomorphic, graph_diff
import os
import glob
from typing import Union
import logging
from meta_prov_fixer.src import process, FillerFixerFile, DateTimeFixerFile, MissingPrimSourceFixerFile, MultiPAFixerFile, MultiObjectFixerFile
from meta_prov_fixer.utils import get_rdf_prov_filepaths
import meta_prov_fixer.src
from pprint import pprint


meta_dumps_pub_dates = [
    ('2022-12-19', 'https://doi.org/10.6084/m9.figshare.21747536.v1'),
    ('2022-12-20', 'https://doi.org/10.6084/m9.figshare.21747536.v2'),
    ('2023-02-15', 'https://doi.org/10.6084/m9.figshare.21747536.v3'),
    ('2023-06-28', 'https://doi.org/10.6084/m9.figshare.21747536.v4'),
    ('2023-10-26', 'https://doi.org/10.6084/m9.figshare.21747536.v5'),
    ('2024-04-06', 'https://doi.org/10.6084/m9.figshare.21747536.v6'),
    ('2024-06-17', 'https://doi.org/10.6084/m9.figshare.21747536.v7'),
    ('2025-02-02', 'https://doi.org/10.6084/m9.figshare.21747536.v8'),
    ('2025-07-10', 'https://doi.org/10.5281/zenodo.15855112')
]

def convert_query_results(qres: SPARQLResult) -> dict:
    """
    Naively converts the results of a SPARQL query made with rdflib to a local Graph/Dataset object to
     to SPARQLWrapper-like results made to an endpoint (at least for what concerns the bindings['results'] part).
    """
    try:
        return json.loads(qres.serialize(format='json'))
    except Exception as e:
        print(e)
        return None

def rdflib_query(dataset:Dataset, query: str) -> Union[dict, None]:
    """
    Executes a SPARQL query on a given rdflib Dataset and returns the results in a SPARQLWrapper-like JSON format.
    """
    return dataset.query(query)

def _normalize_graph_dt_literals(g):
    XSD_DT = 'http://www.w3.org/2001/XMLSchema#dateTime'
    ng = Graph()
    for s, p, o in g:
        new_o = o
        if isinstance(o, Literal) and (str(o).endswith('+00:00') or str(o).endswith('Z')) and str(o).count(':') >= 2 and o.datatype and str(o.datatype).endswith('dateTime'):
            # Normalize timezone representation to use Z
            new_o = Literal(str(o).replace('+00:00', 'Z'), datatype=o.datatype)
        ng.add((s, p, new_o))
    return ng


# Normalize JSON-LD structures to avoid ordering-related flakes by sorting dict-lists by '@id'
def normalize_jsonld(obj):
    if isinstance(obj, list):
        normalized = [normalize_jsonld(i) for i in obj]
        # If list of dicts with '@id', sort by '@id' for deterministic order
        if all(isinstance(i, dict) and '@id' in i for i in normalized):
            normalized.sort(key=lambda x: x.get('@id', ''))
        return normalized
    if isinstance(obj, dict):
        return {k: normalize_jsonld(v) for k, v in sorted(obj.items())}
    return obj


class BaseTestCase(unittest.TestCase):

    def setUp(self):
        self.endpoint_dataset = Dataset(default_union=True)
        for fp in get_rdf_prov_filepaths(FAKE_DUMP_DIR):
            tmp = Dataset(default_union=True)
            tmp.parse(fp, format='json-ld')
            self.endpoint_dataset += tmp

        def mock_sparql_update(client, query, failed_queries_log=None):
            self.endpoint_dataset.update(query)
            return True

        self.mock_sparql_update = mock_sparql_update
        self.maxDiff = None


TEST_DATA_FP = 'tests/data/data.trig'
CORRECTED_DUMP_DIR = 'tests/data/corrected_dump/'
FAKE_DUMP_DIR = 'tests/data/fake_dump/'
OUT_DIR = 'tests/fixed/'

class TestProcessOnFile(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.sparql_endpoint = "http://example.org/sparql/"
        self.meta_dumps_pub_dates = meta_dumps_pub_dates
        self.dry_run = False
        self.fake_dump_dir = FAKE_DUMP_DIR

        self.expected_dataset = Dataset(default_union=True)
        for fp in get_rdf_prov_filepaths(CORRECTED_DUMP_DIR):
            tmp = Dataset(default_union=True)
            tmp.parse(fp, format='json-ld')
            self.expected_dataset += tmp

    def tearDown(self):
        log_dir = 'tests/data/fix_process_log'
        for f in glob.glob(os.path.join(log_dir, '*')):
            try:
                os.remove(f)
            except Exception:
                pass
        for f in glob.glob(os.path.join(OUT_DIR, '*')):
            try:
                os.remove(f)
            except Exception:
                pass
    
    def normalize_datetime_literals(self, serialized: str) -> str:
        """Normalizes datetime literals by converting +00:00 to Z (for test comparison)."""
        # return re.sub(
        #     r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?\+00:00"(\^\^xsd:dateTime)',
        #     r'"\1Z"\2',
        #     serialized
        # )
        return serialized.replace("+00:00", "Z")

    @patch('meta_prov_fixer.src.sparql_update')
    def test_process(self, mock_updt):

        mock_updt.side_effect = self.mock_sparql_update

        process(
            endpoint=self.sparql_endpoint,
            data_dir=self.fake_dump_dir,
            meta_dumps_register=self.meta_dumps_pub_dates,
            out_dir=OUT_DIR,
            failed_queries_fp="tests/data/fix_process_log/failed_queries",
            chunk_size=100,
            overwrite=True
        )

        # --------- Check fixes ON (simulated) TRIPLESTORE ------------

        self.assertEqual(len(list(self.endpoint_dataset.objects(URIRef('https://w3id.org/oc/meta/br/0610491907/prov/se/1'), None))), 7)

        self.assertEqual(len(set(self.endpoint_dataset.subjects())), 13)

        # Serialize and normalize both datasets
        actual_serialized_endpoint = json.loads(self.normalize_datetime_literals(self.endpoint_dataset.serialize(format='json-ld')))
        expected_serialized = json.loads(self.normalize_datetime_literals(self.expected_dataset.serialize(format='json-ld')))

        actual_norm = normalize_jsonld(actual_serialized_endpoint)
        expected_norm = normalize_jsonld(expected_serialized)

        # Assert normalized serialized forms are equal
        self.assertEqual(actual_norm, expected_norm)

        # Additionally, assert RDF graphs are isomorphic per named graph (semantic equality)
        expected_graphs = {g.identifier: g for g in self.expected_dataset.graphs()}
        actual_graphs = {g.identifier: g for g in self.endpoint_dataset.graphs()}
        self.assertEqual(set(expected_graphs.keys()), set(actual_graphs.keys()), "Graph identifier sets differ between expected and actual datasets")

        # Isomorphic comparison
        for gid, exp_g in expected_graphs.items():
            act_g = actual_graphs[gid]
            # Normalize dateTime literal representations to avoid equivalent-but-different lexical forms
            exp_norm = _normalize_graph_dt_literals(exp_g)
            act_norm = _normalize_graph_dt_literals(act_g)
            if not isomorphic(exp_norm, act_norm):
                common, in_expected, in_actual = graph_diff(exp_norm, act_norm)
                self.fail(
                    f"Graphs for {gid} are not isomorphic after normalizing datetimes. Triples only in expected: {len(list(in_expected))}, only in actual: {len(list(in_actual))}"
                )
        
        # --------- Check fixes ON FILES ------------

        fixed_dumped_files = [f for f in get_rdf_prov_filepaths(OUT_DIR)]
        print(fixed_dumped_files)
        on_file_output = Dataset(default_union=True)
        for f in fixed_dumped_files:
            _tmp = Dataset(default_union=True)
            _tmp.parse(f, format='json-ld')
            on_file_output += _tmp
        
        on_file_out_norm = normalize_jsonld(json.loads(self.normalize_datetime_literals(on_file_output.serialize(format='json-ld'))))

        self.assertEqual(expected_norm, on_file_out_norm)




if __name__ == '__main__':
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s"
    # )
    unittest.main()