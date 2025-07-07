import unittest
from unittest.mock import patch, Mock

from rdflib.plugins.sparql.processor import prepareQuery, SPARQLResult
from rdflib import Dataset
import json
from meta_prov_fixer.fix_via_sparql import *
from pprint import pprint


meta_dumps_pub_dates = [
    ('2022-12-19', 'https://doi.org/10.6084/m9.figshare.21747536.v1'),
    ('2022-12-20', 'https://doi.org/10.6084/m9.figshare.21747536.v2'),
    ('2023-02-15', 'https://doi.org/10.6084/m9.figshare.21747536.v3'),
    ('2023-06-28', 'https://doi.org/10.6084/m9.figshare.21747536.v4'),
    ('2023-10-26', 'https://doi.org/10.6084/m9.figshare.21747536.v5'),
    ('2024-04-06', 'https://doi.org/10.6084/m9.figshare.21747536.v6'),
    ('2024-06-17', 'https://doi.org/10.6084/m9.figshare.21747536.v7'),
    ('2025-02-02', 'https://doi.org/10.6084/m9.figshare.21747536.v8')
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



TEST_DATA_FP = 'local/data.trig'
CORRECTED_DATA_FP = 'local/corrected_data.trig'



class TestFillerFixer(unittest.TestCase):

    def setUp(self):
        self.fixer = FillerFixer(
            meta_dumps_pub_dates=meta_dumps_pub_dates,
            sparql_endpoint="http://localhost:8890/sparql/",
            auth=None,
            dry_run=False
        )
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

    @patch.object(ProvenanceIssueFixer, '_query')
    def test_detect_issue(self, mock_query):
        mock_query.side_effect = self.local_query
        expected = [
            ('https://w3id.org/oc/meta/br/06101234191/prov/',
            {'remaining_snapshots': {'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                                    'https://w3id.org/oc/meta/br/06101234191/prov/se/4',
                                    'https://w3id.org/oc/meta/br/06101234191/prov/se/5'},
            'to_delete': {'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
                            'https://w3id.org/oc/meta/br/06101234191/prov/se/3'}}),
            ('https://w3id.org/oc/meta/br/0610491907/prov/',
            {'remaining_snapshots': {'https://w3id.org/oc/meta/br/0610491907/prov/se/1',
                                    'https://w3id.org/oc/meta/br/0610491907/prov/se/3'},
            'to_delete': {'https://w3id.org/oc/meta/br/0610491907/prov/se/2'}})]
        
        
        # Run the method that would internally call self.fixer._query(...)
        mocked_res = self.fixer.detect_issue()

        # Assertions based on expected behavior/output
        self.assertIsInstance(mocked_res, list)
        self.assertEqual(len(mocked_res), 2)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 2 for item in mocked_res))
        self.assertEqual(mocked_res, expected)
    
    def test_map_se_names(self):
        to_delete = {
            'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/3'
        }
        remaining_snapshots = {
            'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/4',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/5'
        }
        expected = {
            'https://w3id.org/oc/meta/br/06101234191/prov/se/1' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/2' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/3' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/4' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/5' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/3'
        }
        res = self.fixer.map_se_names(to_delete, remaining_snapshots)
        self.assertIsInstance(res, dict)
        self.assertEqual(
            res,
            expected
        )
        
        to_delete2 = {'https://w3id.org/oc/meta/br/0610491907/prov/se/2'}
        remaining_snapshots2 = {
            'https://w3id.org/oc/meta/br/0610491907/prov/se/1',
            'https://w3id.org/oc/meta/br/0610491907/prov/se/3'
        }

        expected2 = {
            'https://w3id.org/oc/meta/br/0610491907/prov/se/1' : 'https://w3id.org/oc/meta/br/0610491907/prov/se/1',
            'https://w3id.org/oc/meta/br/0610491907/prov/se/2' : 'https://w3id.org/oc/meta/br/0610491907/prov/se/1',
            'https://w3id.org/oc/meta/br/0610491907/prov/se/3' : 'https://w3id.org/oc/meta/br/0610491907/prov/se/2'
        }
        self.assertEqual(
            self.fixer.map_se_names(to_delete2, remaining_snapshots2),
            expected2
        )
        to_delete3 = {'https://w3id.org/oc/meta/br/06101234191/prov/se/2', 
                    'https://w3id.org/oc/meta/br/06101234191/prov/se/4'}
        remaining3 = {'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
                     'https://w3id.org/oc/meta/br/06101234191/prov/se/3',
                     'https://w3id.org/oc/meta/br/06101234191/prov/se/5'}
        expected3 = {
            'https://w3id.org/oc/meta/br/06101234191/prov/se/1' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/2' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/1',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/3' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/4' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/2',
            'https://w3id.org/oc/meta/br/06101234191/prov/se/5' : 'https://w3id.org/oc/meta/br/06101234191/prov/se/3'
        }
        self.assertEqual(
            self.fixer.map_se_names(to_delete3, remaining3),
            expected3
        )
    

    @patch.object(ProvenanceIssueFixer, '_update')
    @patch.object(ProvenanceIssueFixer, '_query')
    def test_fix_issue(self, mock_query, mock_update):
        corrected_ds = Dataset(default_union=True)
        corrected_ds.parse(CORRECTED_DATA_FP, format='trig')

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        self.fixer.fix_issue()

        self.assertEqual(len(set(self.local_dataset.subjects())), 15)

        # TODO: test edge cases (e.g. no data needs fixing, or detection raises errors...)



class TestDateTimeFixer(unittest.TestCase):
    def setUp(self):
        self.fixer = DateTimeFixer(
            meta_dumps_pub_dates=meta_dumps_pub_dates,
            sparql_endpoint="http://localhost:8890/sparql/",
            auth=None,
            dry_run=False
        )
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

    @patch.object(ProvenanceIssueFixer, '_query')
    def test_detect_issue(self, mock_query):
        mock_query.side_effect = self.local_query

        expected = [
            ('https://w3id.org/oc/meta/br/06104437957/prov/', 'https://w3id.org/oc/meta/br/06104437957/prov/se/1', 'http://www.w3.org/ns/prov#generatedAtTime', '2023-12-13T13:56:16.721920+00:00'),
            ('https://w3id.org/oc/meta/br/06103051181/prov/', 'https://w3id.org/oc/meta/br/06103051181/prov/se/1', 'http://www.w3.org/ns/prov#generatedAtTime', '2023-12-13T14:56:02.909172'),
            ('https://w3id.org/oc/meta/br/0610491907/prov/', 'https://w3id.org/oc/meta/br/0610491907/prov/se/1', 'http://www.w3.org/ns/prov#generatedAtTime', '2023-12-13T14:56:13.401731'),
            ('https://w3id.org/oc/meta/br/0610491907/prov/', 'https://w3id.org/oc/meta/br/0610491907/prov/se/2', 'http://www.w3.org/ns/prov#generatedAtTime', '2024-01-01T01:46:42.700865+00:00'),
            ('https://w3id.org/oc/meta/br/0610491907/prov/', 'https://w3id.org/oc/meta/br/0610491907/prov/se/1', 'http://www.w3.org/ns/prov#invalidatedAtTime', '2024-01-01T01:46:42.700865+00:00'),
            ('https://w3id.org/oc/meta/br/0610476324/prov/', 'https://w3id.org/oc/meta/br/0610476324/prov/se/1', 'http://www.w3.org/ns/prov#generatedAtTime', '2023-12-13T14:56:31.016170'),
            ('https://w3id.org/oc/meta/br/06104278913/prov/', 'https://w3id.org/oc/meta/br/06104278913/prov/se/1', 'http://www.w3.org/ns/prov#generatedAtTime', '2023-12-13T13:56:02.871348+00:00'),
            ('https://w3id.org/oc/meta/br/06101234191/prov/', 'https://w3id.org/oc/meta/br/06101234191/prov/se/1', 'http://www.w3.org/ns/prov#generatedAtTime', '2023-12-13T14:56:09.665214'),
            ('https://w3id.org/oc/meta/br/06101234191/prov/', 'https://w3id.org/oc/meta/br/06101234191/prov/se/2', 'http://www.w3.org/ns/prov#generatedAtTime', '2023-12-23T04:24:57.332607+00:00'),
            ('https://w3id.org/oc/meta/br/06101234191/prov/', 'https://w3id.org/oc/meta/br/06101234191/prov/se/1', 'http://www.w3.org/ns/prov#invalidatedAtTime', '2023-12-23T04:24:57.332607+00:00'),
            ('https://w3id.org/oc/meta/br/06104437954/prov/', 'https://w3id.org/oc/meta/br/06104437954/prov/se/1', 'http://www.w3.org/ns/prov#generatedAtTime', '2023-12-13T13:56:16.718737+00:00')
        ]

        mocked_res = self.fixer.detect_issue()

        # Assertions based on expected behavior/output
        self.assertIsInstance(mocked_res, list)
        self.assertEqual(len(mocked_res), 11)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 4 for item in mocked_res))
        self.assertEqual(set(mocked_res), set(expected))
    
    @patch.object(ProvenanceIssueFixer, '_update')
    @patch.object(ProvenanceIssueFixer, '_query')      
    def test_fix_issue(self, mock_query, mock_update):
        # corrected_ds = Dataset(default_union=True)
        # corrected_ds.parse(CORRECTED_DATA_FP, format='trig')

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        original_quad_count = len(list(self.local_dataset.quads())) # should remain unvaried

        self.fixer.fix_issue()

        new_quad_count = len(list(self.local_dataset.quads()))
        print(self.local_dataset.serialize(format='trig'))
        self.assertEqual(original_quad_count, new_quad_count)
        self.assertEqual(len(set(self.local_dataset.subjects())), 18)









if __name__ == '__main__':

    unittest.main()

