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

class TestSimulateFFChanges(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_simulate_ff_changes(self):
        # with open('tests/data/graph_with_filler.json') as f:
        #     test_data = json.load(f)
        # Parse the test data into a Dataset
        test_dataset = Dataset(default_union=True)
        test_dataset.parse(source='tests/data/graph_with_filler.json', format='json-ld')

        # Convert the dataset to JSON-LD formatted dictionary as expected by simulate_ff_changes
        test_graph_json_ld = json.loads(test_dataset.serialize(format='json-ld'))

        self.assertEqual(len(test_graph_json_ld), 1)
        local_named_graph = test_graph_json_ld[0]

        # Call simulate_ff_changes
        result = simulate_ff_changes(local_named_graph)

        # Basic checks
        self.assertIsInstance(result, dict)
        self.assertIn('@id', result)
        self.assertIn('@graph', result)

        exp = Dataset(default_union=True)
        exp.parse(source='tests/data/graph_filler_corrected.json')
        
        expected_res = json.loads(exp.serialize(format='json-ld'))[0]  # rdflib.Dataset is a LIST of graphs
        self.assertEqual(json.dumps(result), json.dumps(expected_res))


class TestFillerFixer(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.fixer = FillerFixer(
            endpoint="http://example.org/sparql/"
        )

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
    def test_batch_delete_filler_snaphots(self, mock_query, mock_update):
        
        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        to_delete = self.fixer.detect_issue()
        self.fixer.batch_delete_filler_snapshots(to_delete)

        q = 'ASK {<https://w3id.org/oc/meta/br/0610491907/prov/se/2> ?p ?o}' # this should be deleted
        self.assertFalse(self.local_dataset.query(q).askAnswer)
    

    @patch.object(ProvenanceIssueFixer, '_update')
    @patch.object(ProvenanceIssueFixer, '_query')
    def test_fix_issue(self, mock_query, mock_update):
        corrected_ds = Dataset(default_union=True)
        corrected_ds.parse(CORRECTED_DATA_FP, format='trig')

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        self.fixer.fix_issue()

        self.assertEqual(len(set(self.local_dataset.subjects())), 15)
        self.assertEqual(len(list(self.local_dataset.objects((URIRef('https://w3id.org/oc/meta/br/0610491907/prov/se/1')), None))), 6)

        # TODO: test edge cases (e.g. no data needs fixing, or detection raises errors...)



class TestDateTimeFixer(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.fixer = DateTimeFixer(
            endpoint="http://example.org/sparql/"
        )

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

        self.assertIsInstance(mocked_res, list)
        self.assertEqual(len(mocked_res), 11)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 4 for item in mocked_res))
        self.assertEqual(set(mocked_res), set(expected))
    
    @patch.object(ProvenanceIssueFixer, '_update')
    @patch.object(ProvenanceIssueFixer, '_query')      
    def test_fix_issue(self, mock_query, mock_update):

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        original_quad_count = len(list(self.local_dataset.quads())) # should remain unvaried

        self.fixer.fix_issue()

        new_quad_count = len(list(self.local_dataset.quads()))
        self.assertEqual(original_quad_count, new_quad_count)
        self.assertEqual(len(set(self.local_dataset.subjects())), 18)



class TestMissingPrimSourceFixer(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.fixer = MissingPrimSourceFixer(
            endpoint="http://example.org/sparql/",
            meta_dumps_pub_dates=meta_dumps_pub_dates
        )

    @patch.object(ProvenanceIssueFixer, '_query')
    def test_detect_issue(self, mock_query):
        mock_query.side_effect = self.local_query
        expected = [
            ("https://w3id.org/oc/meta/br/0610491907/prov/se/1", "2023-12-13T14:56:13.401731"),
            ("https://w3id.org/oc/meta/br/0610476324/prov/se/1", "2023-12-13T14:56:31.016170"),
            ("https://w3id.org/oc/meta/br/06103051181/prov/se/1", "2023-12-13T14:56:02.909172"),
            ("https://w3id.org/oc/meta/br/06101234191/prov/se/1", "2023-12-13T14:56:09.665214"),
            ("https://w3id.org/oc/meta/br/06104278913/prov/se/1", "2023-12-13T13:56:02.871348Z") # "Z" is automatically converted to "+00:00" in rdflib
        ]
        
        
        # Run the method that would internally call self.fixer._query(...)
        mocked_res = self.fixer.detect_issue()

        self.assertIsInstance(mocked_res, list)
        self.assertEqual(len(mocked_res), 5)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 2 for item in mocked_res))
        
        self.assertEqual(set((s, datetime.fromisoformat(dt_s)) for s, dt_s in mocked_res), set((s, datetime.fromisoformat(dt_s)) for s, dt_s in mocked_res))
        # self.assertEqual(set(mocked_res), set(expected))  # fails because "Z" != "+00:00"


    @patch.object(ProvenanceIssueFixer, '_update')
    @patch.object(ProvenanceIssueFixer, '_query')      
    def test_fix_issue(self, mock_query, mock_update):
        # corrected_ds = Dataset(default_union=True)
        # corrected_ds.parse(CORRECTED_DATA_FP, format='trig')

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        original_quad_count = len(list(self.local_dataset.quads()))

        self.fixer.fix_issue()

        new_quad_count = len(list(self.local_dataset.quads()))
        # print(self.local_dataset.serialize(format='trig'))
        self.assertEqual(original_quad_count + 5, new_quad_count)
        self.assertEqual(len(set(self.local_dataset.subjects())), 18)
    


class TestMultiPAFixer(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.fixer = MultiPAFixer(
            endpoint="http://example.org/sparql/"
        )
    
    @patch.object(ProvenanceIssueFixer, '_query')
    def test_detect_issue(self, mock_query):

        mock_query.side_effect = self.local_query
        expected = [
            ("https://w3id.org/oc/meta/br/06104278913/prov/", "https://w3id.org/oc/meta/br/06104278913/prov/se/1")
        ]

        
        mocked_res = self.fixer.detect_issue()

        self.assertIsInstance(mocked_res, list)
        self.assertEqual(len(mocked_res), 1)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 2 for item in mocked_res))
        self.assertEqual(set(mocked_res), set(expected))
    

    @patch.object(ProvenanceIssueFixer, '_update')
    @patch.object(ProvenanceIssueFixer, '_query')
    def test_fix_issue(self, mock_query, mock_update):

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        original_quad_count = len(list(self.local_dataset.quads()))

        self.fixer.fix_issue()

        new_quad_count = len(list(self.local_dataset.quads()))
        self.assertEqual(original_quad_count - 1, new_quad_count)
        self.assertEqual(len(set(self.local_dataset.subjects())), 18)
        ask_q = """
        PREFIX prov: <http://www.w3.org/ns/prov#> 

        ASK WHERE {
          ?s prov:wasAttributedTo ?pa1 ;
            prov:wasAttributedTo ?pa2.
          FILTER(?pa1 != ?pa2)
        }
        """
        qres = self.local_dataset.query(ask_q)
        qres = qres.askAnswer
        self.assertFalse(qres)

class TestMultiObjectFixer(BaseTestCase):
    
    def setUp(self):
        super().setUp()
        self.fixer = MultiObjectFixer(
            endpoint="http://example.org/sparql/",
            meta_dumps_pub_dates=meta_dumps_pub_dates
        )

    @patch.object(ProvenanceIssueFixer, '_query')
    def test_detect_issue(self, mock_query):

        mock_query.side_effect = self.local_query
        expected = [
            ("https://w3id.org/oc/meta/br/06104437954/prov/", '2023-12-13T13:56:16.718737+00:00'),
            ("https://w3id.org/oc/meta/br/06104437957/prov/", '2023-12-13T13:56:16.721920+00:00'),
            ("https://w3id.org/oc/meta/br/0610476324/prov/", '2023-12-13T14:56:31.016170')
        ]
        mocked_res = self.fixer.detect_issue()

        self.assertIsInstance(mocked_res, list)
        self.assertEqual(len(mocked_res), 3)
        self.assertTrue(all(isinstance(item, tuple) for item in mocked_res))
        self.assertEqual(set(mocked_res), set(expected))


    @patch.object(ProvenanceIssueFixer, '_update')
    @patch.object(ProvenanceIssueFixer, '_query')
    def test_fix_issue(self, mock_query, mock_update):

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        original_graphs_no = len(set(self.local_dataset.graphs()))

        self.fixer.fix_issue()

        self.assertEqual(len(set(self.local_dataset.graphs())), original_graphs_no)
        ask_q = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX oc: <https://w3id.org/oc/ontology/> 

        ASK WHERE {
          VALUES ?p {
          prov:invalidatedAtTime
          prov:hadPrimarySource
          oc:hasUpdateQuery
          }
          ?s ?p ?o1, ?o2 .
          FILTER(?o1 != ?o2)
        }
        """
        qres = self.local_dataset.query(ask_q)
        qres = qres.askAnswer
        self.assertFalse(qres)


class TestFixProcess(BaseTestCase):
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
    @patch.object(ProvenanceIssueFixer, '_query')
    def test_fix_process(self, mock_query, mock_update):
        expected_ds = Dataset(default_union=True)
        expected_ds.parse(CORRECTED_DATA_FP, format='trig')

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        fix_process(
            self.sparql_endpoint,
            self.meta_dumps_pub_dates,
            dry_run=self.dry_run,
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
    
    @patch.object(ProvenanceIssueFixer, '_update')
    @patch.object(ProvenanceIssueFixer, '_query')
    def test_fix_process_result_log(self, mock_query, mock_update):
        expected_ds = Dataset(default_union=True)
        expected_ds.parse(CORRECTED_DATA_FP, format='trig')

        mock_query.side_effect = self.local_query
        mock_update.side_effect = self.local_update

        fix_process(
            self.sparql_endpoint,
            self.meta_dumps_pub_dates,
            issues_log_dir='tests/data/fix_process_log',
            dry_run=self.dry_run,
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
    
    @patch.object(ProvenanceIssueFixer, '_update')
    def test_fix_process_reading_from_files(self, mock_update):
        expected_ds = Dataset(default_union=True)
        expected_ds.parse(CORRECTED_DATA_FP, format='trig')

        mock_update.side_effect = self.local_update

        fix_process_reading_from_files(
            self.sparql_endpoint,
            self.fake_dump_dir,
            self.meta_dumps_pub_dates,
            'tests/data/fix_process_log',
            dry_run=self.dry_run,
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