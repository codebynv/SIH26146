"""Step 2 - validation of timestamps, addresses, ports, amounts and lists."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from backend.app.domain.validation import (
    MAX_PORT,
    parse_timestamp_detailed,
)
from tests.conftest import ADDR_1, TXID_A, make_record


# ---------------------------------------------------------------- timestamps
def test_utc_timestamp_is_normalised_to_utc(provenance):
    record = make_record(provenance, timestamp="2024-05-01T12:00:00Z")
    assert record.timestamp.tzinfo is not None
    assert record.timestamp.utcoffset() == timedelta(0)


def test_offset_timestamp_is_converted_to_utc(provenance):
    record = make_record(provenance, timestamp="2024-05-01T14:00:00+02:00")
    assert record.timestamp.hour == 12
    assert record.timestamp.utcoffset() == timedelta(0)


def test_naive_timestamp_is_never_localised(provenance):
    """A naive timestamp must not be stamped with the host's local timezone."""
    record = make_record(provenance, timestamp="2024-05-01T12:00:00")
    assert record.timestamp.tzinfo is None
    assert record.timestamp.hour == 12


def test_naive_timestamp_is_reported_as_a_warning(provenance):
    record = make_record(provenance, timestamp="2024-05-01T12:00:00")
    codes = [issue.code for issue in record.issues]
    assert "NAIVE_TIMESTAMP" in codes
    assert record.warnings


def test_aware_timestamp_produces_no_naive_warning(provenance):
    record = make_record(provenance, timestamp="2024-05-01T12:00:00Z")
    assert "NAIVE_TIMESTAMP" not in [issue.code for issue in record.issues]


def test_naive_detection_reports_naivety():
    _, was_naive = parse_timestamp_detailed("2024-05-01T12:00:00")
    assert was_naive is True
    _, was_naive = parse_timestamp_detailed("2024-05-01T12:00:00Z")
    assert was_naive is False


def test_epoch_numeric_timestamp_is_refused_not_guessed():
    """Seconds-vs-milliseconds is undecidable, so it must be an error."""
    with pytest.raises(ValueError) as excinfo:
        parse_timestamp_detailed(1714567890)
    assert "ambiguous" in str(excinfo.value)


def test_malformed_timestamp_is_reported(provenance):
    record = make_record(provenance, timestamp="not-a-timestamp")
    assert record.timestamp is None
    assert "INVALID_TIMESTAMP" in [issue.code for issue in record.issues]


def test_empty_timestamp_becomes_none_without_issue(provenance):
    record = make_record(provenance, timestamp="")
    assert record.timestamp is None
    assert "INVALID_TIMESTAMP" not in [issue.code for issue in record.issues]


def test_date_only_timestamp_is_accepted(provenance):
    record = make_record(provenance, timestamp="2024-05-01")
    assert record.timestamp == datetime(2024, 5, 1)


# ---------------------------------------------------------------- addresses
@pytest.mark.parametrize("value", ["203.0.113.5", "198.51.100.9", "2001:db8::1", "::1"])
def test_valid_ip_addresses_are_accepted(provenance, value):
    record = make_record(provenance, src_ip=value)
    assert str(record.src_ip) == value


@pytest.mark.parametrize(
    "value", ["999.1.1.1", "1.2.3", "1.2.3.4.5", "example.com", "203.0.113.5/24", "  "]
)
def test_invalid_ip_addresses_are_reported(provenance, value):
    record = make_record(provenance, src_ip=value)
    assert record.src_ip is None
    if value.strip():
        assert "INVALID_FIELD" in [issue.code for issue in record.issues]


def test_bad_ip_does_not_discard_the_rest_of_the_record(provenance):
    record = make_record(provenance, src_ip="nope", dst_ip="203.0.113.9", dst_port=443)
    assert record.src_ip is None
    assert str(record.dst_ip) == "203.0.113.9"
    assert record.dst_port == 443


# ---------------------------------------------------------------- ports
@pytest.mark.parametrize("value", [0, 80, 443, 65535])
def test_valid_ports_are_accepted(provenance, value):
    record = make_record(provenance, dst_port=value)
    assert record.dst_port == value


@pytest.mark.parametrize("value", [-1, 65536, 99999, 100000])
def test_out_of_range_ports_are_reported(provenance, value):
    record = make_record(provenance, dst_port=value)
    assert record.dst_port is None
    assert "INVALID_FIELD" in [issue.code for issue in record.issues]


def test_port_upper_bound_is_inclusive():
    record_fields = {"dst_port": MAX_PORT}
    assert record_fields["dst_port"] == 65535


def test_port_given_as_string_is_coerced(provenance):
    record = make_record(provenance, src_port="51000")
    assert record.src_port == 51000


def test_non_integer_port_is_reported(provenance):
    record = make_record(provenance, src_port="eighty")
    assert record.src_port is None
    assert "INVALID_FIELD" in [issue.code for issue in record.issues]


# ---------------------------------------------------------------- amounts/fee
@pytest.mark.parametrize("value", ["0", "0.00000001", "1.5", 2, 0.25])
def test_valid_amounts_are_accepted(provenance, value):
    record = make_record(provenance, fee=value)
    assert record.fee is not None
    assert record.fee >= 0


def test_negative_fee_is_reported(provenance):
    record = make_record(provenance, fee="-0.5")
    assert record.fee is None
    assert "INVALID_FIELD" in [issue.code for issue in record.issues]


def test_non_numeric_fee_is_reported(provenance):
    record = make_record(provenance, fee="free")
    assert record.fee is None
    assert "INVALID_FIELD" in [issue.code for issue in record.issues]


def test_fee_is_kept_as_exact_decimal(provenance):
    """Float cannot represent every BTC amount exactly; Decimal can."""
    record = make_record(provenance, fee="0.1")
    assert isinstance(record.fee, Decimal)
    assert record.fee == Decimal("0.1")
    assert record.fee + Decimal("0.2") == Decimal("0.3")


def test_satoshi_scale_amounts_survive(provenance):
    record = make_record(provenance, fee="0.00000001")
    assert record.fee == Decimal("1E-8")


def test_zero_fee_is_valid(provenance):
    record = make_record(provenance, fee=0)
    assert record.fee == Decimal(0)


# ---------------------------------------------------------------- lists
def test_address_list_from_json_string(provenance):
    record = make_record(provenance, input_addresses='["%s","%s"]' % (ADDR_1, TXID_A))
    assert record.input_addresses == [ADDR_1, TXID_A]


def test_address_list_from_semicolon_delimited_string(provenance):
    record = make_record(provenance, input_addresses="%s;%s" % (ADDR_1, TXID_A))
    assert record.input_addresses == [ADDR_1, TXID_A]


def test_address_list_from_pipe_delimited_string(provenance):
    record = make_record(provenance, output_addresses="%s|%s" % (ADDR_1, TXID_A))
    assert record.output_addresses == [ADDR_1, TXID_A]


def test_single_scalar_becomes_a_single_item_list(provenance):
    record = make_record(provenance, input_addresses=ADDR_1)
    assert record.input_addresses == [ADDR_1]


def test_empty_list_fields_default_to_empty(provenance):
    record = make_record(provenance, timestamp="2024-05-01T00:00:00Z", src_ip="1.2.3.4")
    assert record.input_addresses == []
    assert record.output_addresses == []
    assert record.input_amounts == []
    assert record.output_amounts == []


def test_invalid_amount_in_list_is_reported_with_position(provenance):
    record = make_record(provenance, input_amounts="1.5;notanumber;2.5")
    assert record.input_amounts == []
    issues = [i for i in record.issues if i.field == "input_amounts"]
    assert issues
    assert "item 1" in issues[0].message


def test_comma_is_not_a_list_delimiter(provenance):
    """A comma cannot delimit a list value: it is the CSV field separator."""
    record = make_record(provenance, input_addresses="a,b")
    assert record.input_addresses == ["a,b"]


def test_nested_collection_in_list_is_reported(provenance):
    record = make_record(provenance, input_addresses='[["a"],["b"]]')
    assert record.input_addresses == []
    assert "INVALID_FIELD" in [issue.code for issue in record.issues]


def test_malformed_json_list_is_reported(provenance):
    """A '[' prefix declares list intent, so a parse failure must be an error."""
    record = make_record(provenance, input_addresses="[unterminated")
    assert record.input_addresses == []
    assert "INVALID_FIELD" in [issue.code for issue in record.issues]


# ---------------------------------------------------------------- asn
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("15169", 15169), ("AS15169", 15169), ("as15169", 15169), (64500, 64500), (0, 0)],
)
def test_valid_asn_forms(provenance, raw, expected):
    record = make_record(provenance, asn=raw)
    assert record.asn == expected


def test_asn_with_trailing_label_is_accepted(provenance):
    record = make_record(provenance, asn="AS15169 Google")
    assert record.asn == 15169


def test_unparseable_asn_is_reported(provenance):
    record = make_record(provenance, asn="unknown")
    assert record.asn is None
    assert "INVALID_FIELD" in [issue.code for issue in record.issues]


# ---------------------------------------------------------------- record-level
def test_record_with_no_canonical_data_is_reported(provenance):
    from backend.app.ingestion.models import NormalizedRecord

    record = NormalizedRecord.from_raw({"unknown_column": "value"}, provenance)
    assert "NO_DATA" in [issue.code for issue in record.issues]
    # NO_DATA is a warning, so the record itself is not considered invalid.
    assert record.is_valid


def test_strict_empty_rejects_a_record_with_no_canonical_data(provenance):
    from backend.app.ingestion.models import NormalizedRecord

    with pytest.raises(ValueError):
        NormalizedRecord.from_raw({"unknown": "x"}, provenance, strict_empty=True)


def test_is_valid_reflects_only_errors(provenance):
    clean = make_record(provenance, timestamp="2024-05-01T12:00:00Z", src_ip="1.2.3.4")
    assert clean.is_valid
    dirty = make_record(provenance, timestamp="2024-05-01T12:00:00Z", src_ip="bad")
    assert not dirty.is_valid
