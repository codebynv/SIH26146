"""Step 3 - observation classification from meaningful field presence."""

from __future__ import annotations



from backend.app.domain.adapters import classify_observation_type
from backend.app.domain.enums import ObservationType
from tests.conftest import ADDR_1, ADDR_3, TXID_A, make_record


# ---------------------------------------------------------------- network
def test_ip_and_ports_classify_as_network(provenance):
    record = make_record(
        provenance, src_ip="203.0.113.5", dst_ip="198.51.100.9", src_port=1, dst_port=443
    )
    assert classify_observation_type(record) is ObservationType.NETWORK


def test_single_ip_classifies_as_network(provenance):
    record = make_record(provenance, src_ip="203.0.113.5")
    assert classify_observation_type(record) is ObservationType.NETWORK


def test_ipv6_classifies_as_network(provenance):
    record = make_record(provenance, dst_ip="2001:db8::1")
    assert classify_observation_type(record) is ObservationType.NETWORK


def test_ip_with_geo_and_asn_classifies_as_network(provenance):
    record = make_record(
        provenance, src_ip="203.0.113.5", geo_country="IN", asn="AS15169"
    )
    assert classify_observation_type(record) is ObservationType.NETWORK


def test_ports_without_an_ip_are_not_a_network_observation(provenance):
    """Ports alone do not identify an endpoint."""
    record = make_record(provenance, src_port=51000, dst_port=443)
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_geo_and_asn_without_an_ip_are_not_network(provenance):
    record = make_record(provenance, geo_country="IN", asn="AS15169")
    assert classify_observation_type(record) is ObservationType.UNKNOWN


# ---------------------------------------------------------------- blockchain
def test_txid_classifies_as_blockchain(provenance):
    record = make_record(provenance, txid=TXID_A)
    assert classify_observation_type(record) is ObservationType.BLOCKCHAIN


def test_input_addresses_classify_as_blockchain(provenance):
    record = make_record(provenance, input_addresses=[ADDR_1])
    assert classify_observation_type(record) is ObservationType.BLOCKCHAIN


def test_output_addresses_classify_as_blockchain(provenance):
    record = make_record(provenance, output_addresses=[ADDR_3])
    assert classify_observation_type(record) is ObservationType.BLOCKCHAIN


def test_input_amounts_classify_as_blockchain(provenance):
    record = make_record(provenance, input_amounts=["1.5"])
    assert classify_observation_type(record) is ObservationType.BLOCKCHAIN


def test_output_amounts_classify_as_blockchain(provenance):
    record = make_record(provenance, output_amounts=["1.49"])
    assert classify_observation_type(record) is ObservationType.BLOCKCHAIN


# ------------------------------------------- incidental fields must not count
def test_fee_alone_is_not_blockchain(provenance):
    """A generic 'fee' column is common in unrelated data."""
    record = make_record(provenance, fee="0.0001")
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_script_type_alone_is_not_blockchain(provenance):
    record = make_record(provenance, script_type="P2WPKH")
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_fee_and_script_type_together_are_still_not_blockchain(provenance):
    record = make_record(provenance, fee="0.0001", script_type="P2WPKH")
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_txid_with_fee_is_blockchain(provenance):
    record = make_record(provenance, txid=TXID_A, fee="0.0001")
    assert classify_observation_type(record) is ObservationType.BLOCKCHAIN


def test_addresses_with_script_type_is_blockchain(provenance):
    record = make_record(provenance, input_addresses=[ADDR_1], script_type="P2SH")
    assert classify_observation_type(record) is ObservationType.BLOCKCHAIN


# ---------------------------------------------------------------- combined
def test_ip_plus_txid_is_combined(provenance):
    record = make_record(provenance, src_ip="203.0.113.5", txid=TXID_A)
    assert classify_observation_type(record) is ObservationType.COMBINED


def test_ip_plus_addresses_is_combined(provenance):
    record = make_record(provenance, dst_ip="198.51.100.9", output_addresses=[ADDR_1])
    assert classify_observation_type(record) is ObservationType.COMBINED


def test_ip_plus_amounts_is_combined(provenance):
    record = make_record(provenance, src_ip="203.0.113.5", input_amounts=["1.0"])
    assert classify_observation_type(record) is ObservationType.COMBINED


def test_full_combined_record(provenance):
    record = make_record(
        provenance,
        src_ip="203.0.113.5",
        dst_ip="198.51.100.9",
        src_port=51000,
        dst_port=443,
        txid=TXID_A,
        input_addresses=[ADDR_1],
        output_addresses=[ADDR_3],
        input_amounts=["1.5"],
        output_amounts=["1.49"],
        fee="0.00005",
        script_type="P2WPKH",
        geo_country="IN",
        asn="AS15169",
    )
    assert classify_observation_type(record) is ObservationType.COMBINED


# ---------------------------------------------------------------- unknown
def test_timestamp_only_is_unknown(provenance):
    record = make_record(provenance)
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_empty_record_is_unknown(provenance):
    record = make_record(provenance, timestamp="2024-05-01T00:00:00Z")
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_geo_country_only_is_unknown(provenance):
    record = make_record(provenance, geo_country="IN")
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_asn_only_is_unknown(provenance):
    record = make_record(provenance, asn="AS15169")
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_blank_string_fields_are_not_evidence(provenance):
    record = make_record(provenance, src_ip="   ", txid="")
    assert classify_observation_type(record) is ObservationType.UNKNOWN


def test_empty_lists_are_not_evidence(provenance):
    record = make_record(provenance, input_addresses=[], output_amounts=[])
    assert classify_observation_type(record) is ObservationType.UNKNOWN
