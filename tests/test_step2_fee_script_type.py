"""Step 2 - fee and script_type as first-class, preserved fields."""

from __future__ import annotations

from decimal import Decimal

import pytest

from backend.app.ingestion.models import NormalizedRecord
from tests.conftest import ADDR_1, TXID_A, csv_bytes, csv_row, json_bytes, make_record

SCRIPT_TYPES = ["P2PKH", "P2SH", "P2WPKH", "P2WSH", "P2TR", "NULL_DATA", "MULTISIG", "ANYONECANPAY"]


@pytest.mark.parametrize("script_type", SCRIPT_TYPES)
def test_known_script_types_are_preserved_verbatim(provenance, script_type):
    record = make_record(provenance, txid=TXID_A, script_type=script_type)
    assert record.script_type == script_type


def test_unknown_future_script_type_is_preserved_not_rejected(provenance):
    """Refusing an unrecognised script type would lose real evidence."""
    record = make_record(provenance, txid=TXID_A, script_type="P2XYZ-FUTURE")
    assert record.script_type == "P2XYZ-FUTURE"
    assert record.is_valid


def test_script_type_case_is_preserved(provenance):
    record = make_record(provenance, txid=TXID_A, script_type="p2wpkh")
    assert record.script_type == "p2wpkh"


def test_script_type_survives_csv_ingestion(service):
    content = csv_bytes(
        csv_row(
            timestamp="2024-05-01T12:00:00Z",
            txid=TXID_A,
            input_addresses=ADDR_1,
            input_amounts="1.5",
            output_amounts="1.49",
            fee="0.0001",
            script_type="P2TR",
        )
    )
    record = service.ingest_bytes(content).records[0]
    assert record.script_type == "P2TR"
    assert record.fee == Decimal("0.0001")


def test_script_type_survives_json_ingestion(service):
    payload = [{"timestamp": "2024-05-01T12:00:00Z", "txid": TXID_A, "script_type": "P2WSH"}]
    assert service.ingest_bytes(json_bytes(payload)).records[0].script_type == "P2WSH"


def test_script_type_survives_xml_ingestion(service):
    content = (
        "<records><record><timestamp>2024-05-01T00:00:00Z</timestamp>"
        "<txid>%s</txid><script_type>P2TR</script_type></record></records>" % TXID_A
    ).encode()
    assert service.ingest_bytes(content).records[0].script_type == "P2TR"


def test_missing_script_type_is_none_not_empty_string(provenance):
    record = make_record(provenance, txid=TXID_A)
    assert record.script_type is None


def test_blank_script_type_is_treated_as_absent(provenance):
    record = make_record(provenance, txid=TXID_A, script_type="   ")
    assert record.script_type is None


def test_script_type_is_surrounded_by_whitespace_is_stripped(provenance):
    record = make_record(provenance, txid=TXID_A, script_type="  P2PKH  ")
    assert record.script_type == "P2PKH"


def test_fee_and_script_type_are_independent(provenance):
    record = make_record(provenance, txid=TXID_A, fee="0.001", script_type="P2SH")
    assert record.fee == Decimal("0.001")
    assert record.script_type == "P2SH"


def test_fee_without_script_type(provenance):
    record = make_record(provenance, txid=TXID_A, fee="0.001")
    assert record.fee == Decimal("0.001")
    assert record.script_type is None


def test_script_type_without_fee(provenance):
    record = make_record(provenance, txid=TXID_A, script_type="P2SH")
    assert record.fee is None
    assert record.script_type == "P2SH"


def test_satoshi_fee_is_accepted(provenance):
    record = make_record(provenance, txid=TXID_A, fee="0.00000001")
    assert record.fee == Decimal("1E-8")


def test_large_fee_is_accepted_exactly(provenance):
    record = make_record(provenance, txid=TXID_A, fee="1234.56789012345678")
    assert record.fee == Decimal("1234.56789012345678")


def test_invalid_fee_does_not_take_script_type_with_it(provenance):
    record = make_record(provenance, txid=TXID_A, fee="n/a", script_type="P2SH")
    assert record.fee is None
    assert record.script_type == "P2SH"


def test_fee_aliases_normalise_to_canonical_field(provenance):
    for alias in ("fee_btc", "tx_fee", "txn_fee", "fees", "miner_fee", "transaction_fee"):
        record = NormalizedRecord.from_raw(
            {"txid": TXID_A, alias: "0.0002"}, provenance
        )
        assert record.fee == Decimal("0.0002"), alias


def test_script_type_aliases_normalise_to_canonical_field(provenance):
    for alias in ("script", "script_pubkey_type", "tx_type", "lockup_type"):
        record = NormalizedRecord.from_raw(
            {"txid": TXID_A, alias: "P2PKH"}, provenance
        )
        assert record.script_type == "P2PKH", alias


def test_empty_fee_is_treated_as_absent(provenance):
    record = make_record(provenance, txid=TXID_A, fee="")
    assert record.fee is None
    assert "INVALID_FIELD" not in [issue.code for issue in record.issues]


def test_amounts_serialize_as_json_numbers_not_strings(provenance):
    """Decimal is kept internally but must cross the JSON boundary as a number."""
    record = make_record(
        provenance, txid=TXID_A, fee="0.0001", input_amounts=["1.5", "0.25"]
    )
    payload = record.model_dump(mode="json")
    assert isinstance(payload["fee"], float)
    assert payload["fee"] == pytest.approx(0.0001)
    assert payload["input_amounts"] == [pytest.approx(1.5), pytest.approx(0.25)]


def test_json_dump_keeps_amount_precision(provenance):
    import json

    record = make_record(provenance, txid=TXID_A, fee="0.00000001")
    payload = json.loads(record.model_dump_json())
    assert payload["fee"] == pytest.approx(1e-8)


def test_none_amounts_serialise_as_null(provenance):
    record = make_record(provenance, txid=TXID_A)
    assert record.model_dump(mode="json")["fee"] is None
