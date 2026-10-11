"""Step 3 - unified domain model construction and validation."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.app.domain.enums import ObservationType, SourceFormat
from backend.app.domain.models import (
    BlockchainTransaction,
    NetworkObservation,
    Provenance,
    UnifiedObservation,
    ValidationIssue,
    derive_observation_id,
)
from tests.conftest import ADDR_1, ADDR_2, TXID_A


def test_network_observation_construction():
    observation = NetworkObservation(
        timestamp="2024-05-01T12:00:00Z",
        src_ip="203.0.113.5",
        dst_ip="198.51.100.9",
        src_port=51000,
        dst_port=443,
        geo_country="IN",
        asn="AS15169",
    )
    assert str(observation.src_ip) == "203.0.113.5"
    assert observation.dst_port == 443
    assert observation.asn == 15169
    assert observation.geo_country == "IN"


def test_network_observation_rejects_bad_ip():
    with pytest.raises(ValidationError):
        NetworkObservation(src_ip="999.1.1.1")


def test_network_observation_rejects_out_of_range_port():
    with pytest.raises(ValidationError):
        NetworkObservation(dst_port=70000)


def test_network_observation_normalises_timestamp_to_utc():
    observation = NetworkObservation(timestamp="2024-05-01T14:00:00+02:00")
    assert observation.timestamp.hour == 12
    assert observation.timestamp.utcoffset().total_seconds() == 0


def test_network_observation_keeps_naive_timestamp_naive():
    observation = NetworkObservation(timestamp="2024-05-01T14:00:00")
    assert observation.timestamp.tzinfo is None


def test_blockchain_transaction_construction():
    transaction = BlockchainTransaction(
        timestamp="2024-05-01T12:00:00Z",
        txid=TXID_A,
        input_addresses=[ADDR_1, ADDR_2],
        output_addresses=[ADDR_2],
        input_amounts=["1.5", "0.25"],
        output_amounts=["1.7"],
        fee="0.00005",
        script_type="P2WPKH",
    )
    assert transaction.txid == TXID_A
    assert transaction.input_addresses == [ADDR_1, ADDR_2]
    assert transaction.fee == Decimal("0.00005")
    assert transaction.script_type == "P2WPKH"
    assert transaction.input_amounts == [Decimal("1.5"), Decimal("0.25")]


def test_blockchain_transaction_rejects_negative_fee():
    with pytest.raises(ValidationError):
        BlockchainTransaction(fee="-1")


def test_blockchain_transaction_preserves_unknown_script_type():
    assert BlockchainTransaction(script_type="P2XYZ-FUTURE").script_type == "P2XYZ-FUTURE"


def test_blockchain_transaction_rejects_bad_amount():
    with pytest.raises(ValidationError):
        BlockchainTransaction(input_amounts=["1.5", "abc"])


def test_blockchain_transaction_rejects_bad_address():
    with pytest.raises(ValidationError):
        BlockchainTransaction(input_addresses=[["nested"]])


def test_observation_is_immutable(provenance):
    observation = UnifiedObservation(
        observation_id="obs-1", observation_type=ObservationType.NETWORK, provenance=provenance
    )
    with pytest.raises(ValidationError):
        observation.observation_type = ObservationType.BLOCKCHAIN


def test_observation_rejects_unknown_field(provenance):
    """A misspelled field must be an error, not silently ignored."""
    with pytest.raises(ValidationError):
        UnifiedObservation(
            observation_id="obs-1",
            observation_type=ObservationType.NETWORK,
            provenance=provenance,
            src_ip="1.2.3.4",
        )


def test_observation_id_is_required(provenance):
    with pytest.raises(ValidationError):
        UnifiedObservation(observation_type=ObservationType.NETWORK, provenance=provenance)


def test_observation_defaults_are_empty(provenance):
    observation = UnifiedObservation(
        observation_id="obs-1",
        observation_type=ObservationType.UNKNOWN,
        provenance=provenance,
    )
    assert observation.network is None
    assert observation.blockchain is None
    assert observation.extra == {}
    assert observation.warnings == ()
    assert observation.errors == ()
    assert observation.is_valid


def test_observation_is_invalid_when_it_has_errors(provenance):
    observation = UnifiedObservation(
        observation_id="obs-1",
        observation_type=ObservationType.NETWORK,
        provenance=provenance,
        errors=(ValidationIssue(code="INVALID_FIELD", message="bad"),),
    )
    assert not observation.is_valid


def test_observation_with_warnings_is_still_valid(provenance):
    observation = UnifiedObservation(
        observation_id="obs-1",
        observation_type=ObservationType.NETWORK,
        provenance=provenance,
        warnings=(ValidationIssue(code="NAIVE_TIMESTAMP", message="naive"),),
    )
    assert observation.is_valid


def test_observation_all_issues_combines_both(provenance):
    observation = UnifiedObservation(
        observation_id="obs-1",
        observation_type=ObservationType.NETWORK,
        provenance=provenance,
        warnings=(ValidationIssue(code="W", message="w"),),
        errors=(ValidationIssue(code="E", message="e"),),
    )
    assert len(observation.all_issues()) == 2


def test_observation_txid_accessor_is_none_without_blockchain(provenance):
    observation = UnifiedObservation(
        observation_id="obs-1", observation_type=ObservationType.NETWORK, provenance=provenance
    )
    assert observation.txid is None


def test_observation_txid_accessor_reads_blockchain(provenance):
    observation = UnifiedObservation(
        observation_id="obs-1",
        observation_type=ObservationType.BLOCKCHAIN,
        provenance=provenance,
        blockchain=BlockchainTransaction(txid=TXID_A),
    )
    assert observation.txid == TXID_A


def test_observation_serialises_to_json(provenance):
    observation = UnifiedObservation(
        observation_id="obs-1",
        observation_type=ObservationType.BLOCKCHAIN,
        provenance=provenance,
        blockchain=BlockchainTransaction(txid=TXID_A, fee="0.0001"),
    )
    payload = observation.model_dump(mode="json")
    assert payload["blockchain"]["fee"] == pytest.approx(0.0001)
    assert payload["observation_type"] == "BLOCKCHAIN"


def test_source_format_enum_values():
    assert SourceFormat.CSV.value == "csv"
    assert SourceFormat.JSON.value == "json"
    assert SourceFormat.XML.value == "xml"


@pytest.mark.parametrize(
    "value", ["NETWORK", "BLOCKCHAIN", "COMBINED", "UNKNOWN"]
)
def test_observation_type_members(value):
    assert ObservationType(value).value == value


def test_observation_type_has_exactly_four_members():
    assert {member.name for member in ObservationType} == {
        "NETWORK",
        "BLOCKCHAIN",
        "COMBINED",
        "UNKNOWN",
    }


def test_provenance_requires_an_ingestion_id():
    with pytest.raises(ValidationError):
        Provenance(source_file="a.csv")


def test_validation_issue_renders_readably():
    issue = ValidationIssue(field="fee", code="INVALID_FIELD", message="must be numeric")
    assert "fee" in str(issue)
    assert "must be numeric" in str(issue)


def test_derived_observation_id_is_deterministic():
    payload = {"type": "NETWORK", "timestamp": "2024-05-01T00:00:00+00:00"}
    assert derive_observation_id(payload) == derive_observation_id(payload)


def test_derived_observation_id_differs_by_payload():
    a = derive_observation_id({"type": "NETWORK"})
    b = derive_observation_id({"type": "BLOCKCHAIN"})
    assert a != b


def test_derived_observation_id_is_a_uuid():
    value = derive_observation_id({"type": "NETWORK"})
    assert len(value) == 36
    assert value.count("-") == 4


def test_derived_observation_id_is_never_a_txid():
    value = derive_observation_id({"type": "BLOCKCHAIN", "txid": TXID_A})
    assert value != TXID_A
    assert TXID_A not in value
