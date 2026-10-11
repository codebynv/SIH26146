"""Step 2 - input field aliases normalised onto canonical SIH names."""

from __future__ import annotations

import pytest

from backend.app.domain.validation import CANONICAL_FIELDS, canonicalize_key
from backend.app.ingestion.aliases import FIELD_ALIASES, normalize_keys, resolve_key
from backend.app.ingestion.models import NormalizedRecord
from tests.conftest import ADDR_1, TXID_A


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Src IP", "src_ip"),
        ("src-ip", "src_ip"),
        ("SRC_IP", "src_ip"),
        ("  src ip  ", "src_ip"),
        ("Source-IP", "source_ip"),
        ("Timestamp", "timestamp"),
        ("TXID", "txid"),
        ("Geo Country", "geo_country"),
    ],
)
def test_key_canonicalisation(raw, expected):
    assert canonicalize_key(raw) == expected


@pytest.mark.parametrize("raw", ["Source-IP", "source_ip", "SOURCE IP"])
def test_source_ip_variants_resolve_to_src_ip(raw):
    assert resolve_key(raw) == "src_ip"


@pytest.mark.parametrize("field", CANONICAL_FIELDS)
def test_every_canonical_field_resolves_to_itself(field):
    assert resolve_key(field) == field


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("ts", "timestamp"),
        ("datetime", "timestamp"),
        ("event_time", "timestamp"),
        ("source_ip", "src_ip"),
        ("client_ip", "src_ip"),
        ("dest_ip", "dst_ip"),
        ("destination_port", "dst_port"),
        ("dport", "dst_port"),
        ("sport", "src_port"),
        ("tx_id", "txid"),
        ("transaction_hash", "txid"),
        ("tx_hash", "txid"),
        ("hash", "txid"),
        ("inputs", "input_addresses"),
        ("from_addresses", "input_addresses"),
        ("outputs", "output_addresses"),
        ("to_addresses", "output_addresses"),
        ("fee_btc", "fee"),
        ("txn_fee", "fee"),
        ("fee_sat", "fee"),
        ("script", "script_type"),
        ("script_pubkey_type", "script_type"),
        ("tx_type", "script_type"),
        ("country", "geo_country"),
        ("country_code", "geo_country"),
        ("as_number", "asn"),
        ("autonomous_system", "asn"),
    ],
)
def test_alias_resolves_to_canonical_field(raw, canonical):
    assert resolve_key(raw) == canonical


def test_alias_table_covers_every_canonical_field():
    assert set(FIELD_ALIASES) == set(CANONICAL_FIELDS)


def test_aliases_are_applied_end_to_end(provenance):
    raw = {
        "ts": "2024-05-01T12:00:00Z",
        "client_ip": "203.0.113.5",
        "server_port": "443",
        "tx_hash": TXID_A,
        "from_addresses": [ADDR_1],
        "txn_fee": "0.0001",
        "country_code": "IN",
        "as_number": "AS15169",
    }
    record = NormalizedRecord.from_raw(raw, provenance)
    assert record.timestamp is not None
    assert str(record.src_ip) == "203.0.113.5"
    assert record.dst_port == 443
    assert record.txid == TXID_A
    assert record.input_addresses == [ADDR_1]
    assert float(record.fee) == pytest.approx(0.0001)
    assert record.geo_country == "IN"
    assert record.asn == 15169


def test_canonical_key_wins_over_alias(provenance):
    """A real canonical column must not be shadowed by a looser alias."""
    raw = {"txid": "canonical", "hash": "alias"}
    record = NormalizedRecord.from_raw(raw, provenance)
    assert record.txid == "canonical"


def test_normalize_keys_splits_canonical_from_extra():
    canonical, extra = normalize_keys({"ts": "1", "unknown_col": "kept", "asn": 5})
    assert canonical == {"timestamp": "1", "asn": 5}
    assert extra == {"unknown_col": "kept"}


def test_unmapped_fields_are_preserved_not_discarded(provenance):
    raw = {"timestamp": "2024-05-01T00:00:00Z", "block_height": 840000, "confirmations": 6}
    record = NormalizedRecord.from_raw(raw, provenance)
    assert record.extra == {"block_height": 840000, "confirmations": 6}


def test_alias_lookup_is_case_insensitive():
    assert resolve_key("TX_ID") == "txid"
    assert resolve_key("Tx_Id") == "txid"


def test_unmapped_aliases_still_reach_extra():
    """An alias that is also a shadowed key is kept rather than dropped."""
    canonical, extra = normalize_keys({"txid": "a", "hash": "b"})
    assert canonical["txid"] == "a"
    assert extra == {"hash": "b"}


def test_unknown_key_does_not_resolve():
    assert resolve_key("totally_unknown") is None
