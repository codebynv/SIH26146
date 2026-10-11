"""Step 2 - JSON parsing, format detection and controlled errors."""

from __future__ import annotations

import pytest

from backend.app.domain.enums import SourceFormat
from backend.app.ingestion.parsers import IngestionError, detect_format, parse_json

from tests.conftest import ADDR_1, ADDR_2, TXID_A, json_bytes

RECORD = {
    "timestamp": "2024-05-01T12:00:00Z",
    "txid": TXID_A,
    "input_addresses": [ADDR_1, ADDR_2],
    "fee": "0.00012",
    "script_type": "P2SH",
}


def test_detects_json_from_content():
    assert detect_format(json_bytes([RECORD])) is SourceFormat.JSON


def test_parses_top_level_array(service):
    result = service.ingest_bytes(json_bytes([RECORD, {**RECORD, "fee": "0.5"}]))
    assert result.raw_record_count == 2


def test_parses_single_top_level_object(service):
    result = service.ingest_bytes(json_bytes(RECORD))
    assert result.raw_record_count == 1
    assert result.records[0].txid == TXID_A


@pytest.mark.parametrize("key", ["data", "records", "rows", "items", "results", "transactions"])
def test_parses_wrapped_list_under_common_keys(service, key):
    result = service.ingest_bytes(json_bytes({key: [RECORD]}))
    assert result.raw_record_count == 1
    assert result.records[0].txid == TXID_A


def test_locator_points_at_the_json_index(service):
    result = service.ingest_bytes(
        json_bytes([RECORD, {**RECORD, "fee": "0.5"}, {**RECORD, "fee": "0.6"}])
    )
    assert [r.provenance.source_record for r in result.records] == [
        "$[0]",
        "$[1]",
        "$[2]",
    ]


def test_locator_includes_the_container_key(service):
    result = service.ingest_bytes(json_bytes({"records": [RECORD]}))
    assert result.records[0].provenance.source_record == "$['records'][0]"


def test_nested_lists_are_read(service):
    record = service.ingest_bytes(json_bytes([RECORD])).records[0]
    assert record.input_addresses == [ADDR_1, ADDR_2]
    assert float(record.fee) == pytest.approx(0.00012)


def test_empty_array_is_an_empty_dataset(service):
    result = service.ingest_bytes(json_bytes([]))
    assert result.raw_record_count == 0
    assert result.records == []


def test_row_by_row_duplicate_records_are_deduplicated(service):
    """Identical records collapse, so only one locator survives."""
    result = service.ingest_bytes(json_bytes([RECORD, RECORD]))
    assert result.raw_record_count == 2
    assert result.record_count == 1


def test_empty_object_is_reported_as_a_no_data_record(service):
    """An empty object is a single empty record, and is reported as such."""
    result = service.ingest_bytes(json_bytes({}))
    assert result.raw_record_count == 1
    assert "NO_DATA" in [issue.code for issue in result.issues]


def test_empty_body_is_an_empty_dataset(service):
    result = service.ingest_bytes(b"")
    assert result.raw_record_count == 0


def test_malformed_json_raises_controlled_error(service):
    with pytest.raises(IngestionError) as excinfo:
        service.ingest_bytes(b'{"txid": "abc",}')
    assert excinfo.value.code == "MALFORMED_JSON"


def test_truncated_json_raises_controlled_error(service):
    with pytest.raises(IngestionError) as excinfo:
        service.ingest_bytes(b'[{"txid": "abc"')
    assert excinfo.value.code == "MALFORMED_JSON"


def test_json_shaped_content_is_never_fall_back_to_csv(service):
    """Malformed JSON must not be silently ingested as CSV rows."""
    with pytest.raises(IngestionError):
        service.ingest_bytes(b"{ this is not json at all")


def test_parser_rejects_top_level_scalar():
    with pytest.raises(IngestionError) as excinfo:
        list(parse_json(b"42"))
    assert excinfo.value.code == "UNEXPECTED_JSON_SHAPE"


def test_parser_rejects_top_level_string():
    with pytest.raises(IngestionError) as excinfo:
        list(parse_json(b'"just a string"'))
    assert excinfo.value.code == "UNEXPECTED_JSON_SHAPE"


def test_non_object_array_element_is_rejected(service):
    with pytest.raises(IngestionError) as excinfo:
        service.ingest_bytes(b'[{"txid": "a"}, 5]')
    assert excinfo.value.code == "UNEXPECTED_JSON_ELEMENT"


def test_null_container_is_reported_rather_than_silently_accepted(service):
    result = service.ingest_bytes(json_bytes({"records": None}))
    assert "NO_DATA" in [issue.code for issue in result.issues]


def test_parser_is_lazy_and_returns_raw_records():
    records = list(parse_json(json_bytes([RECORD])))
    assert records[0].data["script_type"] == "P2SH"
