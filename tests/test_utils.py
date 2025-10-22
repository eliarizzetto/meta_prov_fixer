from meta_prov_fixer.utils import *
import unittest

import tempfile
import os

META_DUMPS = [
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

class TestGeneralUtils(unittest.TestCase):
    def test_get_seq_num(self):
        self.assertEqual(get_seq_num("https://w3id.org/oc/meta/br/06104375687/prov/se/123"), 123)

        self.assertRaises(Exception, get_seq_num, "https://w3id.org/oc/meta/br/06104375687/prov/se")
    
    def test_remove_seq_num(self):

        input_uri = "https://w3id.org/oc/meta/br/06104375687/prov/se/1"
        expected_output = "https://w3id.org/oc/meta/br/06104375687/prov/se/"
        self.assertEqual(remove_seq_num(input_uri), expected_output)
    
    def test_get_graph_uri_from_se_uri(self):
        self.assertEqual(
            get_graph_uri_from_se_uri("https://w3id.org/oc/meta/br/06104375687/prov/se/123"),
            'https://w3id.org/oc/meta/br/06104375687/prov/'
        )

        self.assertRaises(Exception, get_graph_uri_from_se_uri, "https://w3id.org/oc/meta/br/06104375687/prov/")
    
    def test_get_previous_meta_dump_uri(self):
        self.meta_dump_pub_dates = sorted([(date.fromisoformat(d), doi) for d, doi in META_DUMPS], key=lambda x: x[0])

        # Test case 1: Date before first dump
        self.assertEqual(
            get_previous_meta_dump_uri(self.meta_dump_pub_dates, '2022-12-18'),
            'https://doi.org/10.6084/m9.figshare.21747536.v1'
        )

        # Test case 2: Date exactly matching a dump
        self.assertEqual(
            get_previous_meta_dump_uri(self.meta_dump_pub_dates, '2023-02-15'),
            'https://doi.org/10.6084/m9.figshare.21747536.v2'
        )

        # Test case 3: Date between two dumps
        self.assertEqual(
            get_previous_meta_dump_uri(self.meta_dump_pub_dates, '2023-04-01'),
            'https://doi.org/10.6084/m9.figshare.21747536.v3'
        )

        # Test case 4: Date after the latest dump
        self.assertEqual(
            get_previous_meta_dump_uri(self.meta_dump_pub_dates, '2026-01-01'),
            'https://doi.org/10.5281/zenodo.15855112'
        )

        # Test case 5: Date matching the latest dump
        self.assertEqual(
            get_previous_meta_dump_uri(self.meta_dump_pub_dates, '2025-07-10'),
            'https://doi.org/10.6084/m9.figshare.21747536.v8'
        )

    def test_normalise_datetime(self):
        # Test case 1: naive datetime string (should assume Europe/Rome)
        self.assertEqual(
            normalise_datetime("2023-01-01T12:00:00"),
            "2023-01-01T11:00:00Z"
        )

        # Test case 2: datetime with microseconds (should strip them)
        self.assertEqual(
            normalise_datetime("2023-01-01T12:00:00.123456"),
            "2023-01-01T11:00:00Z"
        )

        # Test case 3: datetime with timezone offset (should convert to UTC)
        self.assertEqual(
            normalise_datetime("2023-01-01T12:00:00+01:00"),
            "2023-01-01T11:00:00Z"
        )

        # Test case 4: datetime with explicit xsd:dateTime datatype
        self.assertEqual(
            normalise_datetime("2023-01-01T12:00:00^^xsd:dateTime"),
            "2023-01-01T11:00:00Z"
        )

        # Test case 5: datetime with explicit xsd:string datatype
        self.assertEqual(
            normalise_datetime("2023-01-01T12:00:00^^xsd:string"),
            "2023-01-01T11:00:00Z"
        )

        # Test case 6: datetime with explicit full xsd namespace
        self.assertEqual(
            normalise_datetime("2023-01-01T12:00:00^^http://www.w3.org/2001/XMLSchema#dateTime"),
            "2023-01-01T11:00:00Z"
        )

        # Test case 7: datetime with explicit full string namespace
        self.assertEqual(
            normalise_datetime("2023-01-01T12:00:00^^http://www.w3.org/2001/XMLSchema#string"),
            "2023-01-01T11:00:00Z"
        )

        # Test case 8: datetime with microseconds and explicit offset
        self.assertEqual(normalise_datetime("2023-01-01T12:00:00.123456Z^^xsd:dateTime"),
            "2023-01-01T12:00:00Z")

    def test_get_described_res_omid(self):
        # Test case 1: Snapshot URI ending with '/prov/se/<counter>'
        self.assertEqual(
            get_described_res_omid("https://w3id.org/oc/meta/br/06104375687/prov/se/123"),
            "https://w3id.org/oc/meta/br/06104375687"
        )

        # Test case 2: Graph URI ending with '/prov/'
        self.assertEqual(
            get_described_res_omid("https://w3id.org/oc/meta/br/06104375687/prov/"),
            "https://w3id.org/oc/meta/br/06104375687"
        )

        # Test case 3: URI without '/prov/' or '/prov/se/' suffix
        self.assertRaises(Exception, 
            get_described_res_omid, "https://w3id.org/oc/meta/br/06104375687")


class TestReadRDFDump(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_read_rdf_dump(self):
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a prov directory
            prov_dir = os.path.join(tmpdir, 'prov')
            os.makedirs(prov_dir)

            # Create a sample JSON file
            sample_data = [
                {"@id": "http://example.org/graph1", "name": "Graph 1"},
                {"@id": "http://example.org/graph2", "name": "Graph 2"}
            ]

            # Write to a .json file
            json_file = os.path.join(prov_dir, 'sample.json')
            with open(json_file, 'w') as f:
                json.dump(sample_data, f)

            # Test the function
            results = list(read_rdf_dump(tmpdir))
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["@id"], "http://example.org/graph1")
            self.assertEqual(results[1]["@id"], "http://example.org/graph2")

            # Test with a .json.xz file
            xz_file = os.path.join(prov_dir, 'sample.json.xz')
            with lzma.open(xz_file, 'wt', encoding='utf-8') as f:
                json.dump(sample_data, f)

            results = list(read_rdf_dump(tmpdir))
            self.assertEqual(len(results), 2)  # (the decompressed .json files are skipped, as they are in the same folder)

            # Test with a .zip file containing a .json
            zip_file = os.path.join(prov_dir, 'sample.zip')
            with ZipFile(zip_file, 'w') as zf:
                zf.writestr('data.json', json.dumps(sample_data))

            results = list(read_rdf_dump(tmpdir))
            # (the decompressed .json files are skipped, as they are in the same folder, but there are .json.xz files which are counted)
            self.assertEqual(len(results), 4)  


class TestChunker(unittest.TestCase):
    
    def test_chunker_from_list(self):
        # Test with a list
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        batches = list(chunker(data, batch_size=3))
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0][0], [1, 2, 3])
        self.assertEqual(batches[0][1], 3)
        self.assertEqual(batches[1][0], [4, 5, 6])
        self.assertEqual(batches[1][1], 6)
        self.assertEqual(batches[2][0], [7, 8, 9])
        self.assertEqual(batches[2][1], 9)
        self.assertEqual(batches[3][0], [10])
        self.assertEqual(batches[3][1], 10)

    def test_chunker_from_file(self):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            f.write('{"process": "paradata}\n{"id": 1}\n{"id": 2}\n{"id": 3}\n{"id": 4}\n')
            temp_file = f.name

        try:
            # Test with file path
            batches = list(chunker(temp_file, batch_size=2))
            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[0][0], [{"id": 1}, {"id": 2}])
            self.assertEqual(batches[1][0], [{"id": 3}, {"id": 4}])

        finally:
            os.unlink(temp_file)

    def test_chunker_no_skip_first_line(self):
        # Create a temporary file with a header line
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            f.write('{"id": 0}\n{"id": 1}\n{"id": 2}\n{"id": 3}\n')
            temp_file = f.name

        try:
            # Test with skip_first_line=False
            batches = list(chunker(temp_file, batch_size=2, skip_first_line=False))
            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[0][0], [{"id": 0}, {"id": 1}])
            self.assertEqual(batches[0][1], 2)
            self.assertEqual(batches[1][0], [{"id": 2}, {"id": 3}])
            self.assertEqual(batches[1][1], 4)
        finally:
            os.unlink(temp_file)

class TestCheckpointing(unittest.TestCase):
    
    def test_save_and_load(self):
        # Create a temporary file for the checkpoint
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            # Create a CheckpointManager instance
            cm = CheckpointManager(path=temp_file)

            # Save a checkpoint
            cm.save("test_fixer", "test_phase", 5)

            # Load the checkpoint
            loaded = cm.load()

            # Check if the loaded data matches the saved data
            self.assertEqual(loaded["fixer"], "test_fixer")
            self.assertEqual(loaded["phase"], "test_phase")
            self.assertEqual(loaded["batch_idx"], 5)

        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_load_nonexistent(self):
        # Create a CheckpointManager with a non-existent file
        cm = CheckpointManager(path="nonexistent.json")

        # Load should return an empty dict
        loaded = cm.load()
        self.assertEqual(loaded, {})

    def test_clear(self):
        # Create a temporary file for the checkpoint
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            # Create a CheckpointManager instance
            cm = CheckpointManager(path=temp_file)

            # Save a checkpoint
            cm.save("test_fixer", "test_phase", 5)

            # Check that the file exists
            self.assertTrue(os.path.exists(temp_file))

            # Clear the checkpoint
            cm.clear()

            # Check that the file no longer exists
            self.assertFalse(os.path.exists(temp_file))

        finally:
            # Clean up the temporary file if it still exists
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_checkpointed_batch(self):
        # Create a temporary file for the checkpoint
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            json.dump({}, f)

        try:
            # Create a CheckpointManager instance
            cm = CheckpointManager(path=temp_file)

            # Test checkpointed_batch with a function that returns a list
            def sample_func():
                return [1, 2, 3, 4, 5]

            # Call checkpointed_batch with a batch size of 2
            result = list(checkpointed_batch(sample_func(), batch_size=2, fixer_name='sample_func', phase='done', checkpoint=cm))

            # Check that the result is correct
            self.assertEqual(result, [(0, ([1, 2], 2)), (1, ([3, 4], 4)), (2, ([5], 5))])

            # Check that the checkpoint was saved
            loaded = cm.load()
            self.assertEqual(loaded["fixer"], "sample_func")
            self.assertEqual(loaded["phase"], "done")
            self.assertEqual(loaded["batch_idx"], 2)

        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

if __name__ == '__main__':
    unittest.main()