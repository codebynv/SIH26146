"""Step 2 - CSV parsing, format detection and controlled errors."""

from __future__ import annotations

import pytest

from backend.app.domain.enums import SourceFormat
from backend.app.ingestion.parsers import detect_format, parse_csv
from backend.app.ingestion.service import IngestionService
from tests.conftest import ADDR_1, ADDR_3, TXID_A, csv_bytes, csv_row

ROW_COMBINED = csv_row(
    timestamp="2024-05-01T12:00:00Z",
    src_ip="203.0.113.5",
    dst_ip="198.51.100.9",
    src_port=51000,
    dst_port=443,
    txid=TXID_A,
    input_addresses=ADDR_1,
    output_addresses=ADDR_3,
    input_amounts="1.5",
    output_amounts="1.49",
    fee="0.00005",
    script_type="P2WPKH",
    geo_country="IN",
    asn="AS15169",
)


def test_detects_csv_from_header_row():
    assert detect_format(csv_bytes(ROW_COMBINED)) is SourceFormat.CSV


def test_parses_header_and_rows():
    records = list(parse_csv(csv_bytes(ROW_COMBINED)))
    assert len(records) == 1
    assert records[0].data["txid"] == TXID_A


def test_row_locator_is_one_based_and_counts_header():
    records = list(parse_csv(csv_bytes(ROW_COMBINED, ROW_COMBINED)))
    assert [record.locator for record in records] == ["row 2", "row 3"]


def test_canonical_fields_are_read():
    result = IngestionService().ingest_bytes(csv_bytes(ROW_COMBINED))
    record = result.records[0]
    assert record.src_ip is not None and str(record.src_ip) == "203.0.113.5"
    assert record.dst_port == 443
    assert record.txid == TXID_A
    assert record.input_addresses == [ADDR_1]
    assert float(record.fee) == pytest.approx(0.00005)


def test_empty_csv_yields_no_records_without_error(service):
    result = service.ingest_bytes(b"")
    assert result.raw_record_count == 0
    assert result.records == []
    assert "no records were found in the input" in result.warnings


def test_header_only_csv_is_an_empty_dataset(service):
    content = b"timestamp,src_ip,dst_ip\n"
    result = service.ingest_bytes(content)
    assert result.raw_record_count == 0
    assert result.records == []


def test_blank_lines_are_skipped(service):
    content = b"timestamp,src_ip\n2024-05-01T00:00:00Z,1.2.3.4\n\n\n2024-05-02T00:00:00Z,1.2.3.5\n"
    result = service.ingest_bytes(content)
    assert result.raw_record_count == 2


def test_semicolon_delimiter_is_detected(service):
    content = b"timestamp;src_ip;dst_port\n2024-05-01T00:00:00Z;1.2.3.4;443\n"
    result = service.ingest_bytes(content)
    record = result.records[0]
    assert str(record.src_ip) == "1.2.3.4"
    assert record.dst_port == 443


def test_tab_delimiter_is_detected(service):
    content = b"timestamp\tsrc_ip\tdst_port\n2024-05-01T00:00:00Z\t1.2.3.4\t443\n"
    record = service.ingest_bytes(content).records[0]
    assert record.dst_port == 443


def test_utf8_bom_is_stripped_from_header(service):
    content = b"\xef\xbb\xbftimestamp,src_ip\n2024-05-01T00:00:00Z,1.2.3.4\n"
    result = service.ingest_bytes(content)
    assert result.records[0].timestamp is not None


def test_ragged_row_with_extra_values_is_reported(service):
    content = b"timestamp,src_ip\n2024-05-01T00:00:00Z,1.2.3.4,surplus,more\n"
    result = service.ingest_bytes(content)
    codes = [issue.code for issue in result.issues]
    assert "RAGGED_RECORD" in codes


def test_ragged_row_with_missing_values_is_reported(service):
    content = b"timestamp,src_ip,dst_ip,dst_port\n2024-05-01T00:00:00Z,1.2.3.4\n"
    result = service.ingest_bytes(content)
    assert "RAGGED_RECORD" in [issue.code for issue in result.issues]


def test_valid_prefix_of_ragged_row_is_retained(service):
    """A ragged row must not discard the data that was readable."""
    content = b"timestamp,src_ip,dst_port\n2024-05-01T00:00:00Z,1.2.3.4,443,extra\n"
    record = service.ingest_bytes(content).records[0]
    assert str(record.src_ip) == "1.2.3.4"
    assert record.dst_port == 443


def test_json_encoded_list_inside_a_csv_cell(service):
    content = (
        'timestamp,txid,input_addresses,input_amounts\n'
        '2024-05-01T00:00:00Z,%s,"[""%s"",""%s""]","[1.5,0.25]"\n'
        % (TXID_A, ADDR_1, ADDR_3)
    ).encode()
    record = service.ingest_bytes(content).records[0]
    assert record.input_addresses == [ADDR_1, ADDR_3]
    assert [float(v) for v in record.input_amounts] == [1.5, 0.25]


def test_malformed_json_inside_a_csv_cell_is_reported(service):
    content = b'timestamp,txid,input_addresses\n2024-05-01T00:00:00Z,x,"[not json]"\n'
    result = service.ingest_bytes(content)
    assert "INVALID_FIELD" in [issue.code for issue in result.issues]
    assert result.records[0].input_addresses == []


def test_long_cells_do_not_crash_the_parser(service):
    payload = "x" * 200_000
    content = ("timestamp,geo_country\n2024-05-01T00:00:00Z,%s\n" % payload).encode()
    record = service.ingest_bytes(content).records[0]
    assert record.geo_country == payload


def test_unbalanced_quote_does_not_crash_and_is_reported(service):
    """Python's csv reader is lenient, so an unterminated quote must still be
    surfaced as a problem record rather than silently ingested as valid."""
    content = b'timestamp,txid\n"2024-05-01T00:00:00Z,%s\n' % TXID_A.encode()
    result = service.ingest_bytes(content)
    codes = [issue.code for issue in result.issues]
    assert "INVALID_TIMESTAMP" in codes
    assert "NO_DATA" in codes


def test_embedded_newline_in_a_quoted_field_is_preserved(service):
    """A quoted multi-line value must not be silently joined."""
    content = b'timestamp,geo_country\n"2024-05-01T00:00:00Z","line1\nline2"\n'
    record = service.ingest_bytes(content).records[0]
    assert record.geo_country == "line1\nline2"


def test_null_byte_in_a_value_is_reported_not_silently_dropped(service):
    content = b"timestamp,src_ip\n2024-05-01T00:00:00Z,1.2.3.4\x00\n"
    result = service.ingest_bytes(content)
    assert "INVALID_FIELD" in [issue.code for issue in result.issues]
