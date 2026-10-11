"""Adapters from normalised ingestion records to unified observations
(SIH26146 - Step 3).

This module is the seam between Steps 2 and 3.  It is intentionally the
*only* place where an ingestion record becomes a domain observation, and it
performs no inference beyond that mechanical mapping.

It depends on a structural :class:`NormalizedRecordProtocol` rather than
importing :class:`~backend.app.ingestion.models.NormalizedRecord`, so the
domain package never depends on the ingestion package and the layering stays
one-directional.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from backend.app.domain.enums import ObservationType
from backend.app.domain.models import (
    BlockchainTransaction,
    NetworkObservation,
    Provenance,
    UnifiedObservation,
    ValidationIssue,
    derive_observation_id,
)
from backend.app.domain.validation import (
    AsNumber,
    BtcAddressList,
    BtcAmount,
    BtcAmountList,
    IpAddress,
    PortNumber,
    UtcTimestamp,
    canonicalize_evidence,
)
from pydantic import TypeAdapter

__all__ = [
    "NormalizedRecordProtocol",
    "classify_observation_type",
    "normalized_to_unified",
    "records_to_unified",
]


@runtime_checkable
class NormalizedRecordProtocol(Protocol):
    """The structural surface the adapter needs from an ingestion record."""

    timestamp: Optional[Any]
    src_ip: Optional[Any]
    dst_ip: Optional[Any]
    src_port: Optional[Any]
    dst_port: Optional[Any]
    txid: Optional[str]
    input_addresses: list[str]
    output_addresses: list[str]
    input_amounts: list[Any]
    output_amounts: list[Any]
    fee: Optional[Any]
    script_type: Optional[str]
    geo_country: Optional[str]
    asn: Optional[int]
    extra: Mapping[str, Any]
    provenance: Provenance
    issues: Any


_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    "src_ip": TypeAdapter(Optional[IpAddress]),
    "dst_ip": TypeAdapter(Optional[IpAddress]),
    "src_port": TypeAdapter(Optional[PortNumber]),
    "dst_port": TypeAdapter(Optional[PortNumber]),
    "txid": TypeAdapter(Optional[str]),
    "input_addresses": TypeAdapter(BtcAddressList),
    "output_addresses": TypeAdapter(BtcAddressList),
    "input_amounts": TypeAdapter(BtcAmountList),
    "output_amounts": TypeAdapter(BtcAmountList),
    "fee": TypeAdapter(Optional[BtcAmount]),
    "script_type": TypeAdapter(Optional[str]),
    "geo_country": TypeAdapter(Optional[str]),
    "asn": TypeAdapter(Optional[AsNumber]),
    "timestamp": TypeAdapter(Optional[UtcTimestamp]),
}


def _build_network(record: Any, timestamp: Any) -> NetworkObservation | None:
    """Build the network sub-object, or ``None`` if it carries no data.

    Presence is decided by :attr:`NetworkObservation.has_network_data`, so
    canonical network-layer fields supplied without an IP address -- a
    ``geo_country`` or an ``asn``, say -- are preserved rather than discarded.
    Whether the record actually *identifies* a network endpoint is a separate
    question, answered by :attr:`NetworkObservation.has_network_identity` and
    consulted only by :func:`_classify_layers`.
    """
    candidate = NetworkObservation(
        timestamp=timestamp,
        src_ip=_ADAPTERS["src_ip"].validate_python(getattr(record, "src_ip", None)),
        dst_ip=_ADAPTERS["dst_ip"].validate_python(getattr(record, "dst_ip", None)),
        src_port=_ADAPTERS["src_port"].validate_python(
            getattr(record, "src_port", None)
        ),
        dst_port=_ADAPTERS["dst_port"].validate_python(
            getattr(record, "dst_port", None)
        ),
        geo_country=_ADAPTERS["geo_country"].validate_python(
            getattr(record, "geo_country", None)
        ),
        asn=_ADAPTERS["asn"].validate_python(getattr(record, "asn", None)),
    )
    return candidate if candidate.has_network_data else None


def _build_blockchain(record: Any, timestamp: Any) -> BlockchainTransaction | None:
    """Build the blockchain sub-object, or ``None`` if it carries no evidence.

    Evidence is decided by
    :attr:`BlockchainTransaction.has_structured_transaction_data`, which
    deliberately ignores a lone ``fee`` or ``script_type``.
    """
    candidate = BlockchainTransaction(
        timestamp=timestamp,
        txid=_ADAPTERS["txid"].validate_python(getattr(record, "txid", None)),
        input_addresses=_ADAPTERS["input_addresses"].validate_python(
            getattr(record, "input_addresses", None) or []
        ),
        output_addresses=_ADAPTERS["output_addresses"].validate_python(
            getattr(record, "output_addresses", None) or []
        ),
        input_amounts=_ADAPTERS["input_amounts"].validate_python(
            getattr(record, "input_amounts", None) or []
        ),
        output_amounts=_ADAPTERS["output_amounts"].validate_python(
            getattr(record, "output_amounts", None) or []
        ),
        fee=_ADAPTERS["fee"].validate_python(getattr(record, "fee", None)),
        script_type=_ADAPTERS["script_type"].validate_python(
            getattr(record, "script_type", None)
        ),
    )
    return candidate if candidate.has_structured_transaction_data else None


def _has_network_evidence(record: Any) -> bool:
    """True when a record carries *identifying* network data.

    Requires at least one IP address.  A port number on its own does not
    identify an endpoint and is not sufficient, and neither does a
    ``geo_country`` or ``asn`` on its own -- those are preserved, but they
    identify nothing.
    """
    candidate = _build_network(record, getattr(record, "timestamp", None))
    return candidate is not None and candidate.has_network_identity


def _has_blockchain_evidence(record: Any) -> bool:
    """True when a record carries real blockchain structure.

    Deliberately *excludes* ``fee`` and ``script_type``.  Those columns are
    common in unrelated network and log data, so on their own they are
    incidental and must not promote a record to a blockchain observation.
    """
    return _build_blockchain(record, getattr(record, "timestamp", None)) is not None


def classify_observation_type(record: Any) -> ObservationType:
    """Classify a normalised record by the layers it actually carries.

    ``COMBINED`` means one source record mentioned both an IP address and
    blockchain structure.  It is a statement about the *record's contents*
    only -- it is never a claim that the IP belongs to the wallet, and no
    such claim is derivable from co-occurrence.
    """
    network = _build_network(record, getattr(record, "timestamp", None))
    blockchain = _build_blockchain(record, getattr(record, "timestamp", None))
    return _classify_layers(network, blockchain)


def _classify_layers(
    network: NetworkObservation | None,
    blockchain: BlockchainTransaction | None,
) -> ObservationType:
    """Classify from already-built layer objects.

    Classification depends on network **identity**, never on the mere presence
    of a ``NetworkObservation``.  A record may legitimately carry canonical
    network-layer data such as a ``geo_country`` with no IP address at all;
    that data is preserved on the sub-object, but it does not identify an
    endpoint, so it does not on its own make the observation ``NETWORK`` or
    ``COMBINED``.

    Keeping the rule in one place is deliberate: the "is the sub-object
    present" question (preservation) and the "is it identifying" question
    (classification) are easy to conflate, and conflating them either
    discards supplied evidence or over-claims a network identity.
    """
    network_evidence = network is not None and network.has_network_identity
    if network_evidence and blockchain is not None:
        return ObservationType.COMBINED
    if network_evidence:
        return ObservationType.NETWORK
    if blockchain is not None:
        return ObservationType.BLOCKCHAIN
    return ObservationType.UNKNOWN


def _split_issues(issues: Any) -> tuple[tuple[ValidationIssue, ...], tuple[ValidationIssue, ...]]:
    from backend.app.domain.enums import IssueSeverity

    warnings: list[ValidationIssue] = []
    errors: list[ValidationIssue] = []
    for issue in issues or ():
        if issue.severity is IssueSeverity.ERROR:
            errors.append(issue)
        else:
            warnings.append(issue)
    return tuple(warnings), tuple(errors)


def normalized_to_unified(
    record: Any,
    *,
    observation_id: Optional[str] = None,
) -> UnifiedObservation:
    """Adapt a normalised ingestion record into a :class:`UnifiedObservation`.

    The mapping is mechanical: canonical fields land in the layer sub-object
    they belong to, provenance is carried through untouched, unmapped source
    fields are preserved in :attr:`UnifiedObservation.extra`, and validation
    issues are split into warnings and errors.

    :param observation_id: override for the derived identifier. The default is
        a deterministic UUIDv5 over the observation's own content, which is
        always distinct from any TXID.
    """
    warnings, errors = _split_issues(getattr(record, "issues", ()))

    timestamp = _ADAPTERS["timestamp"].validate_python(
        getattr(record, "timestamp", None)
    )

    # Build each layer once and let the model decide whether it is present.
    # A layer may be present for preservation (canonical data was supplied)
    # without being identifying; _classify_layers keeps those apart.
    network = _build_network(record, timestamp)
    blockchain = _build_blockchain(record, timestamp)

    observation_type = _classify_layers(network, blockchain)

    if observation_id is None:
        # ``extra`` takes part in the derived identity.  Two records that agree
        # on every canonical SIH field but carry different unmapped evidence
        # are two observations, and giving them the same observation_id would
        # silently merge them again for anything downstream that keys on it.
        observation_id = derive_observation_id(
            {
                "type": observation_type.value,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "network": network.model_dump(mode="json") if network else None,
                "blockchain": blockchain.model_dump(mode="json") if blockchain else None,
                "extra": canonicalize_evidence(dict(getattr(record, "extra", {}) or {})),
            }
        )

    return UnifiedObservation(
        observation_id=observation_id,
        observation_type=observation_type,
        timestamp=timestamp,
        network=network,
        blockchain=blockchain,
        provenance=record.provenance,
        extra=dict(getattr(record, "extra", {}) or {}),
        warnings=warnings,
        errors=errors,
    )


def records_to_unified(records: Any) -> list[UnifiedObservation]:
    """Adapt a sequence of normalised records, preserving order."""
    return [normalized_to_unified(record) for record in records]
