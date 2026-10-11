"""Unified domain models for BitTrace AI (SIH26146 - Step 3).

The domain model is intentionally *layered but not linked*.  A
:class:`UnifiedObservation` can carry network data, blockchain data, or
both, but the two payloads are kept in **separate** sub-objects and are
never joined by inference.

Explicit non-goals for this step
--------------------------------
* No IP-to-wallet attribution.  An IP address and a TXID appearing on the
  same source record is *not* evidence that a wallet is controlled by that
  address; it is only evidence that one record mentioned both.
* No cross-layer correlation, evidence graph, risk scoring, clustering or
  ML.  Those belong to Step 4 and later.
* ``txid`` is **not** an identifier for an observation.  One transaction
  can appear in many observations, one observation can contain many
  transactions, and a network-only observation has no transaction at all.
  :attr:`UnifiedObservation.observation_id` is therefore a separate,
  deterministic identifier.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.enums import IssueSeverity, ObservationType, SourceFormat
from backend.app.domain.validation import (
    AsNumber,
    BtcAddressList,
    BtcAmount,
    BtcAmountList,
    IpAddress,
    PortNumber,
    UTC,
    UtcTimestamp,
)

__all__ = [
    "ValidationIssue",
    "Provenance",
    "NetworkObservation",
    "BlockchainTransaction",
    "UnifiedObservation",
    "OBSERVATION_NAMESPACE",
    "derive_observation_id",
]

#: Stable UUIDv5 namespace for deriving deterministic observation identifiers.
OBSERVATION_NAMESPACE = uuid.UUID("6f9b1f2e-3a5c-5d7e-9b1a-0c2d4e6f8a1b")


class ValidationIssue(BaseModel):
    """A single validation problem found while normalising one record.

    Issues are the mechanism by which malformed input is *reported* instead
    of being silently dropped or silently coerced.
    """

    model_config = ConfigDict(frozen=True)

    field: Optional[str] = Field(
        default=None, description="Canonical field the issue relates to."
    )
    code: str = Field(description="Stable machine-readable issue code.")
    message: str = Field(description="Human-readable description.")
    severity: IssueSeverity = Field(
        default=IssueSeverity.ERROR,
        description="ERROR means data was discarded; WARNING means data was kept "
        "but is ambiguous or lossy.",
    )
    raw_value: Optional[Any] = Field(
        default=None, description="The offending input value, when representable."
    )
    source_record: Optional[str] = Field(
        default=None, description="Where in the source the record came from."
    )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        target = self.field or "<record>"
        return "[%s] %s: %s" % (self.severity.value, target, self.message)


class Provenance(BaseModel):
    """Where a record came from, preserved for the life of the record.

    Sufficient to answer: which file, which format, which row/element, and
    which ingestion run produced this record.
    """

    model_config = ConfigDict(frozen=True)

    ingestion_id: str = Field(description="Identifier of the ingestion run.")
    source_file: Optional[str] = Field(
        default=None,
        description="Display name of the source. Never a filesystem path derived "
        "from untrusted client input.",
    )
    source_format: Optional[SourceFormat] = Field(
        default=None, description="Detected wire format the record was parsed from."
    )
    source_record: Optional[str] = Field(
        default=None,
        description="Record locator within the source, e.g. 'row 4', "
        "'$[2]' or 'transaction[1]'.",
    )
    ingested_at: Optional[datetime] = Field(
        default=None, description="UTC time at which ingestion occurred."
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional ingestion context, e.g. declared format, "
        "size, or notes.",
    )


class NetworkObservation(BaseModel):
    """Network-layer facts about a single source record.

    These are observations about a *connection*, not about a person or a
    wallet.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: Optional[UtcTimestamp] = Field(
        default=None,
        description="UTC when the source supplied a timezone; naive (and flagged) "
        "when it did not. Never localised implicitly.",
    )
    src_ip: Optional[IpAddress] = None
    dst_ip: Optional[IpAddress] = None
    src_port: Optional[PortNumber] = None
    dst_port: Optional[PortNumber] = None
    geo_country: Optional[str] = Field(
        default=None, description="Country attribution as supplied by the source."
    )
    asn: Optional[AsNumber] = None

    @property
    def has_network_identity(self) -> bool:
        """True when at least one IP address is present.

        Ports without an IP address are not treated as identifying network
        data on their own.

        This is the **identity** test, and it is the only one classification
        ever consults.  It is deliberately narrower than
        :attr:`has_network_data`: preserved data is not identity.
        """
        return self.src_ip is not None or self.dst_ip is not None

    @property
    def has_network_data(self) -> bool:
        """True when at least one canonical network-layer field is present.

        This is a **preservation** predicate, deliberately distinct from
        :attr:`has_network_identity`.  A source may supply ``geo_country`` or
        ``asn`` with no IP address anywhere in the record.  Those are
        canonical SIH26146 fields, and silently dropping them would destroy
        evidence the source actually provided, so this sub-object is kept.

        ``timestamp`` is excluded on purpose: it belongs to both layers, so
        counting it would attach a network sub-object to every record.

        Returning True here does **not** mean the record identifies a network
        endpoint, and it must never be used to infer one.  No address, owner
        or location is inferred from a preserved ``geo_country`` or ``asn``;
        they are reproduced exactly as supplied.  :attr:`has_network_identity`
        remains the identity test.
        """
        return (
            self.src_ip is not None
            or self.dst_ip is not None
            or self.src_port is not None
            or self.dst_port is not None
            or self.geo_country is not None
            or self.asn is not None
        )


class BlockchainTransaction(BaseModel):
    """Blockchain-layer facts about a single source record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: Optional[UtcTimestamp] = None
    txid: Optional[str] = Field(
        default=None,
        description="Bitcoin transaction id as reported by the source. This is a "
        "payload field, not the identity of the observation.",
    )
    input_addresses: BtcAddressList = Field(default_factory=list)
    output_addresses: BtcAddressList = Field(default_factory=list)
    input_amounts: BtcAmountList = Field(default_factory=list)
    output_amounts: BtcAmountList = Field(default_factory=list)
    fee: Optional[BtcAmount] = Field(
        default=None, description="Transaction fee in BTC, preserved verbatim."
    )
    script_type: Optional[str] = Field(
        default=None,
        description="Script/lockup type as reported by the source, e.g. P2WPKH. "
        "Preserved verbatim: unknown future types are carried, not rejected.",
    )

    @property
    def has_txid(self) -> bool:
        return bool(self.txid and self.txid.strip())

    @property
    def has_addresses(self) -> bool:
        return bool(self.input_addresses or self.output_addresses)

    @property
    def has_amounts(self) -> bool:
        return bool(self.input_amounts or self.output_amounts)

    @property
    def has_structured_transaction_data(self) -> bool:
        """True when there is blockchain evidence beyond a lone fee/script_type.

        A ``fee`` or ``script_type`` on its own is *incidental*: plenty of
        non-blockchain records carry a generic ``fee`` or a ``type`` column.
        Requiring a txid, addresses or amounts prevents those records from
        being mislabelled as blockchain observations.
        """
        return self.has_txid or self.has_addresses or self.has_amounts


class UnifiedObservation(BaseModel):
    """A single normalised record in the canonical BitTrace AI shape.

    One source record maps to exactly one observation.  The observation
    holds the SIH26146 payload split by layer, the provenance needed to
    trace it back, and any unmapped fields so that nothing useful is lost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str = Field(
        description="Stable identifier for this observation, derived from its "
        "content. Deliberately distinct from txid."
    )
    observation_type: ObservationType = Field(
        description="Which layers this record carries meaningful data for."
    )
    timestamp: Optional[UtcTimestamp] = Field(
        default=None,
        description="The record's own timestamp, carried at the top level for "
        "convenience. It is the same value present in the sub-objects; it is "
        "not derived from, or matched against, any other record.",
    )
    network: Optional[NetworkObservation] = None
    blockchain: Optional[BlockchainTransaction] = None
    provenance: Provenance
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields present in the source that map to no canonical SIH "
        "field. Retained so no potentially useful evidence is discarded.",
    )
    warnings: tuple[ValidationIssue, ...] = Field(
        default=(),
        description="Non-fatal issues: lossy or ambiguous normalisation.",
    )
    errors: tuple[ValidationIssue, ...] = Field(
        default=(),
        description="Fatal issues: data was rejected and the field dropped.",
    )

    @property
    def is_valid(self) -> bool:
        """True when no field was discarded during normalisation."""
        return not self.errors

    @property
    def txid(self) -> Optional[str]:
        """Convenience accessor; ``None`` for non-blockchain observations."""
        return self.blockchain.txid if self.blockchain else None

    def all_issues(self) -> tuple[ValidationIssue, ...]:
        return self.warnings + self.errors


def derive_observation_id(payload: dict[str, Any]) -> str:
    """Derive a deterministic observation identifier from *payload*.

    A UUIDv5 over a canonical JSON rendering of the observation means the
    same record always yields the same id (reproducible across runs and
    machines) while remaining completely distinct from any TXID, which is a
    64-character hex string with its own semantics.
    """
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    return str(uuid.uuid5(OBSERVATION_NAMESPACE, canonical))
