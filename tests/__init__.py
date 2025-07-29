import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from ..meta_prov_fixer.utils import normalise_datetime, get_seq_num
from ..meta_prov_fixer.detectors import SnapshotEntity

# filepath: tests/test___init__.py

# Mock data for SnapshotEntity
mock_prov_graph = {
    "@graph": [
        {
            "@id": "https://example.org/prov/se/1",
            "http://purl.org/dc/terms/description": [{"@value": "Creation snapshot"}],
            "http://www.w3.org/ns/prov#generatedAtTime": [{"@value": "2023-10-01T12:00:00+02:00"}],
            "http://www.w3.org/ns/prov#hadPrimarySource": [{"@id": "https://example.org/source"}],
        },
        {
            "@id": "https://example.org/prov/se/2",
            "http://www.w3.org/ns/prov#generatedAtTime": [{"@value": "2023-10-02T12:00:00+02:00"}],
        },
    ]
}

mock_se = {
    "@id": "https://example.org/prov/se/1",
    "http://purl.org/dc/terms/description": [{"@value": "Creation snapshot"}],
    "http://www.w3.org/ns/prov#generatedAtTime": [{"@value": "2023-10-01T12:00:00+02:00"}],
    "http://www.w3.org/ns/prov#hadPrimarySource": [{"@id": "https://example.org/source"}],
}

# Test normalise_datetime
def test_normalise_datetime():
    assert normalise_datetime("2023-10-01T12:00:00+02:00") == "2023-10-01T10:00:00+00:00"
    assert normalise_datetime("2023-10-01T12:00:00") == "2023-10-01T10:00:00+00:00"
    assert normalise_datetime("2023-10-01T12:00:00Z") == "2023-10-01T12:00:00+00:00"

# Test get_seq_num
def test_get_seq_num():
    assert get_seq_num("https://example.org/prov/se/1") == 1
    assert get_seq_num("https://example.org/prov/se/123") == 123
    assert get_seq_num("https://example.org/prov/se/") is None

# Test SnapshotEntity methods
def test_snapshot_entity_missing_description():
    se = SnapshotEntity(mock_se, mock_prov_graph)
    assert not se.missing_description()

def test_snapshot_entity_missing_generatedAtTime():
    se = SnapshotEntity(mock_se, mock_prov_graph)
    assert not se.missing_generatedAtTime()

def test_snapshot_entity_missing_hadPrimarySource():
    se = SnapshotEntity(mock_se, mock_prov_graph)
    assert not se.missing_hadPrimarySource()

def test_snapshot_entity_multi_description():
    se = SnapshotEntity(mock_se, mock_prov_graph)
    assert not se.multi_description()

def test_snapshot_entity_multi_generatedAtTime():
    se = SnapshotEntity(mock_se, mock_prov_graph)
    assert not se.multi_generatedAtTime()