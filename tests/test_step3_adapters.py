"""Step 3 - the normalized_to_unified adapter."""

from __future__ import annotations

from decimal import Decimal



from backend.app.domain.adapters import (
    classify_observation_type,
    normalized_to_unified,
    records_to_unified,
)
from backend.app.domain.enums import ObservationType
from backend.app.domain.models import Provenance
from backend.app.ingestion.models import NormalizedRecord
from tests.conftest import ADDR_1, ADDR_3, TXID_A, TXID_B, make_record


def _prov(locator: str = "row 2") -> Provenance:
    """Provenance for records built directly from raw mappings."""
    return Provenance(
        ingestion_id="step3-extra",
        source_file="extra.csv",
        source_format="csv",
        source_record=locator,
        context={},
    )


# ---------------------------------------------------------------- network
def test_network_only_record_maps_to_network_subobject(provenance):
    record = make_record(
        provenance,
        src_ip="203.0.113.5",
        dst_ip="198.51.100.9",
        src_port=51000,
        dst_port=443,
        geo_country="IN",
        asn="AS15169",
    )
    observation = normalized_to_unified(record)
    assert observation.observation_type is ObservationType.NETWORK
    assert observation.blockchain is None
    assert str(observation.network.src_ip) == "203.0.113.5"
    assert observation.network.dst_port == 443
    assert observation.network.asn == 15169


# ---------------------------------------------------------------- blockchain
def test_blockchain_only_record_maps_to_blockchain_subobject(provenance):
    record = make_record(
        provenance,
        txid=TXID_A,
        input_addresses=[ADDR_1],
        output_addresses=[ADDR_3],
        input_amounts=["1.5"],
        output_amounts=["1.49"],
        fee="0.00005",
        script_type="P2WPKH",
    )
    observation = normalized_to_unified(record)
    assert observation.observation_type is ObservationType.BLOCKCHAIN
    assert observation.network is None
    assert observation.blockchain.txid == TXID_A
    assert observation.blockchain.fee == Decimal("0.00005")
    assert observation.blockchain.script_type == "P2WPKH"
    assert observation.blockchain.input_amounts == [Decimal("1.5")]


# ---------------------------------------------------------------- combined
def test_combined_record_keeps_layers_separate(provenance):
    record = make_record(
        provenance, src_ip="203.0.113.5", txid=TXID_A, fee="0.0001", script_type="P2TR"
    )
    observation = normalized_to_unified(record)
    assert observation.observation_type is ObservationType.COMBINED
    assert observation.network is not None
    assert observation.blockchain is not None
    # No inferred linkage: the network side carries no wallet/address info.
    assert not hasattr(observation.network, "txid")
    assert not hasattr(observation.network, "input_addresses")


def test_combined_does_not_claim_ip_to_wallet_attribution(provenance):
    """Co-occurrence of an IP and a TXID is not proof of identity."""
    record = make_record(provenance, src_ip="203.0.113.5", txid=TXID_A)
    observation = normalized_to_unified(record)
    payload = observation.model_dump(mode="json")
    # Nothing anywhere asserts the IP belongs to the transaction's wallets.
    network_payload = payload["network"]
    for forbidden in ("wallet", "owner", "attributed_to", "belongs_to", "identity"):
        assert forbidden not in network_payload


# ---------------------------------------------------------------- unknown
def test_unknown_record_has_neither_subobject(provenance):
    observation = normalized_to_unified(make_record(provenance))
    assert observation.observation_type is ObservationType.UNKNOWN
    assert observation.network is None
    assert observation.blockchain is None


def test_fee_only_record_stays_unknown(provenance):
    observation = normalized_to_unified(
        make_record(provenance, src_ip="1.2.3.4", fee="0.0001", script_type="P2PKH")
    )
    # It has an IP, so it is NETWORK; fee/script_type do not make it blockchain.
    assert observation.observation_type is ObservationType.NETWORK
    assert observation.blockchain is None
    assert observation.network is not None


# ---------------------------------------------------------------- identifiers
def test_observation_id_is_not_the_txid(provenance):
    observation = normalized_to_unified(
        make_record(provenance, txid=TXID_A, input_addresses=[ADDR_1])
    )
    assert observation.observation_id != TXID_A
    assert observation.txid == TXID_A


def test_same_txid_yields_different_observation_ids(provenance):
    """One transaction can legitimately appear in several observations."""
    first = normalized_to_unified(
        make_record(provenance, timestamp="2024-05-01T00:00:00Z", txid=TXID_A)
    )
    second = normalized_to_unified(
        make_record(
            provenance,
            timestamp="2024-05-01T00:00:00Z",
            txid=TXID_A,
            src_ip="203.0.113.5",
        )
    )
    assert first.txid == second.txid == TXID_A
    assert first.observation_id != second.observation_id


def test_observation_id_is_deterministic_for_identical_content(provenance):
    from backend.app.domain.models import Provenance

    first = normalized_to_unified(make_record(provenance, src_ip="1.2.3.4", dst_port=443))
    other = Provenance(
        ingestion_id="different-run",
        source_file="other.csv",
        source_format="json",
        source_record="$[7]",
        context={},
    )
    second = normalized_to_unified(make_record(other, src_ip="1.2.3.4", dst_port=443))
    # Same canonical content, different provenance -> same observation identity.
    assert first.observation_id == second.observation_id


def test_observation_id_differs_when_content_differs(provenance):
    first = normalized_to_unified(make_record(provenance, src_ip="1.2.3.4"))
    second = normalized_to_unified(make_record(provenance, src_ip="1.2.3.5"))
    assert first.observation_id != second.observation_id


def test_observation_id_can_be_overridden(provenance):
    observation = normalized_to_unified(
        make_record(provenance, src_ip="1.2.3.4"), observation_id="forced-id"
    )
    assert observation.observation_id == "forced-id"


# ---------------------------------------------------------------- preservation
def test_all_sih_fields_are_preserved_on_a_full_record(provenance):
    record = make_record(
        provenance,
        src_ip="203.0.113.5",
        dst_ip="198.51.100.9",
        src_port=51000,
        dst_port=443,
        txid=TXID_A,
        input_addresses=[ADDR_1],
        output_addresses=[ADDR_3],
        input_amounts=["1.5", "0.25"],
        output_amounts=["1.7"],
        fee="0.00005",
        script_type="P2WPKH",
        geo_country="IN",
        asn="AS15169",
    )
    observation = normalized_to_unified(record)
    assert str(observation.network.src_ip) == "203.0.113.5"
    assert str(observation.network.dst_ip) == "198.51.100.9"
    assert observation.network.src_port == 51000
    assert observation.network.dst_port == 443
    assert observation.network.geo_country == "IN"
    assert observation.network.asn == 15169
    assert observation.blockchain.txid == TXID_A
    assert observation.blockchain.input_addresses == [ADDR_1]
    assert observation.blockchain.output_addresses == [ADDR_3]
    assert observation.blockchain.input_amounts == [Decimal("1.5"), Decimal("0.25")]
    assert observation.blockchain.output_amounts == [Decimal("1.7")]
    assert observation.blockchain.fee == Decimal("0.00005")
    assert observation.blockchain.script_type == "P2WPKH"
    assert observation.timestamp == record.timestamp


def test_unmapped_fields_are_preserved_in_extra(provenance):
    record = make_record(provenance, txid=TXID_A, input_addresses=[ADDR_1])
    record = record.model_copy(update={"extra": {"block_height": 840000, "raw_hex": "0xdead"}})
    observation = normalized_to_unified(record)
    assert observation.extra == {"block_height": 840000, "raw_hex": "0xdead"}


def test_extra_defaults_to_empty(provenance):
    observation = normalized_to_unified(make_record(provenance, src_ip="1.2.3.4"))
    assert observation.extra == {}


# ---------------------------------------------------------------- provenance
def test_provenance_is_carried_through_verbatim(provenance):
    record = make_record(provenance, src_ip="1.2.3.4")
    observation = normalized_to_unified(record)
    assert observation.provenance == provenance
    assert observation.provenance.source_file == "unit-test.csv"
    assert observation.provenance.source_record == "row 2"
    assert observation.provenance.ingestion_id == "test-ingestion"


def test_record_issues_become_observation_warnings(provenance):
    record = make_record(provenance, timestamp="2024-05-01T12:00:00", src_ip="1.2.3.4")
    observation = normalized_to_unified(record)
    assert [issue.code for issue in observation.warnings] == ["NAIVE_TIMESTAMP"]
    assert observation.is_valid


def test_record_errors_become_observation_errors(provenance):
    record = make_record(provenance, src_ip="not-an-ip", dst_ip="1.2.3.4")
    observation = normalized_to_unified(record)
    assert [issue.code for issue in observation.errors] == ["INVALID_FIELD"]
    assert not observation.is_valid


def test_warnings_and_errors_are_separated(provenance):
    record = make_record(
        provenance, timestamp="2024-05-01T12:00:00", src_ip="bad", dst_port="bad2"
    )
    observation = normalized_to_unified(record)
    assert observation.warnings
    assert len(observation.errors) == 2


def test_issue_source_locator_survives_into_the_observation(provenance):
    record = make_record(provenance, src_ip="bad")
    observation = normalized_to_unified(record)
    assert observation.errors[0].source_record == "row 2"


# ---------------------------------------------------------------- batch
def test_records_to_unified_preserves_order(provenance):
    records = [
        make_record(provenance, src_ip="1.1.1.1"),
        make_record(provenance, src_ip="2.2.2.2"),
        make_record(provenance, txid=TXID_B),
    ]
    observations = records_to_unified(records)
    assert [o.observation_type for o in observations] == [
        ObservationType.NETWORK,
        ObservationType.NETWORK,
        ObservationType.BLOCKCHAIN,
    ]


def test_records_to_unified_on_empty_input():
    assert records_to_unified([]) == []


def test_classify_accepts_any_structural_record(provenance):
    """Classification depends on field presence, not the concrete class."""

    class Duck:
        txid = None
        input_addresses = []
        output_addresses = []
        input_amounts = []
        output_amounts = []
        src_ip = "1.2.3.4"
        dst_ip = None

    assert classify_observation_type(Duck()) is ObservationType.NETWORK


def test_adapter_does_not_mutate_the_source_record(provenance):
    record = make_record(provenance, src_ip="1.2.3.4", fee="0.0001")
    before = record.model_dump()
    normalized_to_unified(record)
    assert record.model_dump() == before


# ------------------------------- canonical network data without an IP address
def test_geo_country_without_an_ip_is_preserved(provenance):
    """Supplied canonical data is kept even when it identifies nothing.

    A ``geo_country`` with no IP address cannot identify an endpoint, but it
    is a canonical SIH26146 field the source actually provided.  Discarding
    it would destroy evidence, so it is preserved on the sub-object.
    """
    observation = normalized_to_unified(make_record(provenance, geo_country="IN"))
    assert observation.network is not None
    assert observation.network.geo_country == "IN"
    # Preserved, not demoted to an unmapped field.
    assert observation.extra == {}


def test_asn_without_an_ip_is_preserved(provenance):
    observation = normalized_to_unified(make_record(provenance, asn="AS15169"))
    assert observation.network is not None
    assert observation.network.asn == 15169
    assert observation.extra == {}


def test_geo_country_and_asn_without_an_ip_are_both_preserved(provenance):
    observation = normalized_to_unified(
        make_record(provenance, geo_country="IN", asn="AS15169")
    )
    assert observation.network.geo_country == "IN"
    assert observation.network.asn == 15169
    assert observation.extra == {}


def test_preserved_network_data_does_not_invent_an_ip(provenance):
    """Preserving a field must never manufacture identity."""
    observation = normalized_to_unified(
        make_record(provenance, geo_country="IN", asn="AS15169")
    )
    assert observation.network.src_ip is None
    assert observation.network.dst_ip is None
    assert observation.network.has_network_identity is False


def test_ports_without_an_ip_are_preserved_but_not_identifying(provenance):
    observation = normalized_to_unified(make_record(provenance, src_port=443))
    assert observation.network is not None
    assert observation.network.src_port == 443
    assert observation.network.has_network_identity is False


def test_geo_without_ip_stays_unknown(provenance):
    observation = normalized_to_unified(make_record(provenance, geo_country="IN"))
    assert observation.observation_type is ObservationType.UNKNOWN


def test_geo_with_txid_stays_blockchain_not_combined(provenance):
    """Preserved network data must not promote a record to COMBINED."""
    observation = normalized_to_unified(
        make_record(provenance, txid=TXID_A, geo_country="IN", asn="AS15169")
    )
    assert observation.observation_type is ObservationType.BLOCKCHAIN
    assert observation.network.geo_country == "IN"
    assert observation.blockchain.txid == TXID_A


def test_geo_with_ip_still_classifies_as_network(provenance):
    observation = normalized_to_unified(
        make_record(provenance, src_ip="203.0.113.5", geo_country="IN")
    )
    assert observation.observation_type is ObservationType.NETWORK
    assert observation.network.has_network_identity is True


def test_geo_with_ip_and_txid_still_classifies_as_combined(provenance):
    observation = normalized_to_unified(
        make_record(provenance, src_ip="203.0.113.5", txid=TXID_A, geo_country="IN")
    )
    assert observation.observation_type is ObservationType.COMBINED


def test_blockchain_only_record_has_no_network_subobject(provenance):
    observation = normalized_to_unified(
        make_record(provenance, txid=TXID_A, input_amounts=["1.5"])
    )
    assert observation.observation_type is ObservationType.BLOCKCHAIN
    assert observation.network is None


def test_empty_record_stays_unknown_with_no_subobjects(provenance):
    observation = normalized_to_unified(make_record(provenance))
    assert observation.observation_type is ObservationType.UNKNOWN
    assert observation.network is None
    assert observation.blockchain is None


# --------------------------------- identity follows the preserved data
def test_records_differing_only_in_geo_country_get_different_ids(provenance):
    a = normalized_to_unified(make_record(provenance, geo_country="IN"))
    b = normalized_to_unified(make_record(provenance, geo_country="US"))
    assert a.observation_id != b.observation_id


def test_records_differing_only_in_asn_get_different_ids(provenance):
    a = normalized_to_unified(make_record(provenance, asn="AS15169"))
    b = normalized_to_unified(make_record(provenance, asn="AS16509"))
    assert a.observation_id != b.observation_id


def test_identical_geo_without_ip_is_reproducible(provenance):
    a = normalized_to_unified(make_record(provenance, geo_country="IN"))
    b = normalized_to_unified(make_record(provenance, geo_country="IN"))
    assert a.observation_id == b.observation_id


def test_preserved_network_data_never_becomes_the_txid(provenance):
    observation = normalized_to_unified(
        make_record(provenance, txid=TXID_A, geo_country="IN")
    )
    assert observation.observation_id != TXID_A


def test_preserved_network_data_keeps_provenance(provenance):
    observation = normalized_to_unified(make_record(provenance, geo_country="IN"))
    assert observation.provenance is provenance
    assert observation.provenance.source_record == provenance.source_record


# ------------------------------------------- observation identity vs. extra
def test_differing_unmapped_evidence_yields_different_observation_ids():
    """Unmapped evidence must be part of the observation identity.

    Two records agreeing on every canonical SIH field but carrying different
    unmapped values are two observations.  If they shared an observation_id,
    anything downstream that keys on it would silently merge them again,
    reintroducing the evidence loss that deduplication now prevents.
    """
    a = NormalizedRecord.from_raw(
        {"txid": TXID_A, "block_height": 840000}, _prov("row 2")
    )
    b = NormalizedRecord.from_raw(
        {"txid": TXID_A, "block_height": 840001}, _prov("row 3")
    )
    assert a.txid == b.txid
    first = normalized_to_unified(a)
    second = normalized_to_unified(b)
    assert first.observation_id != second.observation_id
    assert first.extra == {"block_height": 840000}
    assert second.extra == {"block_height": 840001}


def test_identical_records_still_share_an_observation_id():
    """Determinism is preserved: the same evidence keeps the same identity."""
    a = NormalizedRecord.from_raw(
        {"txid": TXID_A, "block_height": 840000}, _prov("row 2")
    )
    b = NormalizedRecord.from_raw(
        {"txid": TXID_A, "block_height": 840000}, _prov("row 9")
    )
    assert normalized_to_unified(a).observation_id == normalized_to_unified(b).observation_id


def test_unmapped_key_order_does_not_change_the_observation_id():
    a = NormalizedRecord.from_raw(
        {"txid": TXID_A, "meta": {"h": 1, "c": 2}}, _prov("row 2")
    )
    b = NormalizedRecord.from_raw(
        {"txid": TXID_A, "meta": {"c": 2, "h": 1}}, _prov("row 3")
    )
    assert normalized_to_unified(a).observation_id == normalized_to_unified(b).observation_id


def test_observation_id_stays_distinct_from_txid_with_extra_present():
    a = NormalizedRecord.from_raw({"txid": TXID_A, "k": "v"}, _prov("row 2"))
    observation = normalized_to_unified(a)
    assert observation.observation_id != TXID_A
    assert observation.txid == TXID_A
