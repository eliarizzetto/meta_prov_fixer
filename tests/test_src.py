import unittest
from unittest.mock import patch, Mock

from rdflib.plugins.sparql.processor import prepareQuery, SPARQLResult
from rdflib import Dataset, URIRef, Literal
import json
import re
from meta_prov_fixer.fix_via_sparql import *
from pprint import pprint
from datetime import datetime
from rdflib.compare import isomorphic, graph_diff
import os
import glob
from meta_prov_fixer.src import process


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

def rdflib_update(dataset:Dataset, query: str) -> None:
    """
    Executes a SPARQL update query on a given rdflib Dataset.
    """
    dataset.update(query)


class BaseTestCase(unittest.TestCase):
    """Base test class to reduce boilerplate code."""
    
    def setUp(self):
        self.local_dataset = Dataset(default_union=True)
        self.local_dataset.parse(TEST_DATA_FP, format='trig')

        def local_query(q):
            """Custom function to simulate _query using local Dataset."""
            qres = rdflib_query(self.local_dataset, q)
            return convert_query_results(qres)

        def local_update(q):
            """Custom function to simulate _update using local Dataset."""
            return rdflib_update(self.local_dataset, q)
        
        self.local_query = local_query
        self.local_update = local_update
        self.maxDiff = None


TEST_DATA_FP = 'tests/data/data.trig'
CORRECTED_DATA_FP = 'tests/data/corrected_data.trig'
FAKE_DUMP_DIR = 'tests/data/fake_dump/'



class TestProcessOnFile(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.sparql_endpoint = "http://example.org/sparql/"
        self.meta_dumps_pub_dates = meta_dumps_pub_dates
        self.dry_run = False
        self.fake_dump_dir = FAKE_DUMP_DIR

    def tearDown(self):
        log_dir = 'tests/data/fix_process_log'
        for f in glob.glob(os.path.join(log_dir, '*')):
            try:
                os.remove(f)
            except Exception:
                pass
    
    def normalize_datetime_literals(self, serialized: str) -> str:
        """Normalizes datetime literals by converting +00:00 to Z (for test comparison)."""
        return re.sub(
            r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?\+00:00"(\^\^xsd:dateTime)',
            r'"\1Z"\2',
            serialized
        )

    
    @patch.object(ProvenanceIssueFixer, '_update')
    def test_process(self, mock_update):
        expected_ds = Dataset(default_union=True)
        expected_ds.parse(CORRECTED_DATA_FP, format='trig')

        mock_update.side_effect = self.local_update

        process(
            endpoint=self.sparql_endpoint,
            data_dir=self.fake_dump_dir,
            meta_dumps_register=self.meta_dumps_pub_dates,
            out_dir='tests/process_on_files_out/',
            failed_queries_fp="tests/data/fix_process_log/failed_queries",
            chunk_size=100
        )

        self.assertEqual(
            len(list(self.local_dataset.objects(URIRef('https://w3id.org/oc/meta/br/0610491907/prov/se/1'), None))),
            7
        )
        self.assertEqual(len(set(self.local_dataset.subjects())), 13)

        # Serialize and normalize both datasets
        actual_serialized = self.normalize_datetime_literals(self.local_dataset.serialize(format='trig'))
        expected_serialized = self.normalize_datetime_literals(expected_ds.serialize(format='trig'))

        # Assert serialized forms are equal
        self.assertEqual(actual_serialized.strip(), expected_serialized.strip())


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    unittest.main()