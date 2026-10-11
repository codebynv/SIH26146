"""Normalised ingestion record (SIH26146 - Step 2).

A :class:`NormalizedRecord` is the boundary object of Step 2: parsers turn
bytes into raw mappings, and this module turns raw mappings into validated,
canonical records.  Records carry their own provenance and any validation
issues, so nothing is lost between the file on disk and the domain model.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_serializer,
)

from backend.app.domain.enums import IssueSeverity
from backend.app.domain.models import Provenance, ValidationIssue
from backend.app.domain.validation import (
    AsNumber,
    BtcAddressList,
    BtcAmount,
    BtcAmountList,
    CANONICAL_FIELDS,
    IpAddress,
    PortNumber,
    TextValue,
    canonicalize_evidence,
    parse_timestamp_detailed,
)
from backend.app.ingestion.aliases import normalize_keys

__all__ = [
    "NormalizedRecord",
    "FIELD_ADAPTERS",
    "canonicalize_evidence",
    "new_ingestion_id",
    "new_provenance",
]

# Per-field adapters.  Validating one field at a time is deliberate: a single
# bad value must not discard the rest of a record, and each rejection must be
# attributable to a named field.
FIELD_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    "src_ip": TypeAdapter(Optional[IpAddress]),
    "dst_ip": TypeAdapter(Optional[IpAddress]),
    "src_port": TypeAdapter(Optional[PortNumber]),
    "dst_port": TypeAdapter(Optional[PortNumber]),
    "txid": TypeAdapter(Optional[TextValue]),
    "input_addresses": TypeAdapter(BtcAddressList),
    "output_addresses": TypeAdapter(BtcAddressList),
    "input_amounts": TypeAdapter(BtcAmountList),
    "output_amounts": TypeAdapter(BtcAmountList),
    "fee": TypeAdapter(Optional[BtcAmount]),
    "script_type": TypeAdapter(Optional[TextValue]),
    "geo_country": TypeAdapter(Optional[TextValue]),
    "asn": TypeAdapter(Optional[AsNumber]),
}

_LIST_FIELDS = frozenset(
    {"input_addresses", "output_addresses", "input_amounts", "output_amounts"}
)


def new_ingestion_id() -> str:
    """Generate an identifier for a single ingestion run."""
    return uuid.uuid4().hex


def new_provenance(
    *,
    ingestion_id: str,
    source_file: Optional[str] = None,
    source_format: Optional[Any] = None,
    source_record: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Provenance:
    """Build provenance stamped with the current UTC time."""
    return Provenance(
        ingestion_id=ingestion_id,
        source_file=source_file,
        source_format=source_format,
        source_record=source_record,
        ingested_at=datetime.now(timezone.utc),
        context=dict(context or {}),
    )


class NormalizedRecord(BaseModel):
    """One source record normalised onto the canonical SIH26146 schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: Optional[Any] = Field(
        default=None,
        description="UTC when the source supplied a timezone; still-naive (and "
        "flagged) when it did not. Never implicitly localised.",
    )
    src_ip: Optional[Any] = None
    dst_ip: Optional[Any] = None
    src_port: Optional[Any] = None
    dst_port: Optional[Any] = None
    txid: Optional[str] = None
    input_addresses: list[str] = Field(default_factory=list)
    output_addresses: list[str] = Field(default_factory=list)
    input_amounts: list[Any] = Field(default_factory=list)
    output_amounts: list[Any] = Field(default_factory=list)
    fee: Optional[Any] = None
    script_type: Optional[str] = None
    geo_country: Optional[str] = None
    asn: Optional[int] = None

    provenance: Provenance
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Source fields that map to no canonical SIH field."
    )
    issues: tuple[ValidationIssue, ...] = Field(default=())

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IssueSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IssueSeverity.WARNING)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @field_serializer("fee", "input_amounts", "output_amounts", when_used="json")
    def _serialize_amounts(self, value: Any) -> Any:
        """Emit amounts as JSON numbers, matching the domain models.

        Amounts are held as :class:`~decimal.Decimal` so no precision is lost
        internally; the JSON boundary is the only place a float appears.
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        return float(value)

    @property
    def has_any_data(self) -> bool:
        """True when at least one canonical SIH field carries a value.

        An entirely empty record is not an observation; it is an artefact of
        a ragged file and should be reported rather than classified.
        """
        for field in CANONICAL_FIELDS:
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, (list, tuple, dict, str)) and not value:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return True
        return False

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical SIH field values only.

        Provenance is deliberately excluded: the same observation arriving in
        two files must deduplicate against the other.  Unmapped ``extra`` is
        excluded here too, but only because it is folded in separately by
        :meth:`identity_payload`; it is *not* ignored.
        """
        payload: dict[str, Any] = {}
        for field in CANONICAL_FIELDS:
            value = getattr(self, field)
            if field in _LIST_FIELDS:
                payload[field] = [str(v) for v in (value or [])]
            elif isinstance(value, datetime):
                payload[field] = value.isoformat()
            elif value is None:
                payload[field] = None
            else:
                payload[field] = str(value)
        return payload

    def identity_payload(self) -> dict[str, Any]:
        """Everything that makes this record *this* record, for deduplication.

        Two records may be merged only when this payload matches exactly.  It
        covers all three things that distinguish one piece of evidence from
        another:

        1. every canonical SIH field (:meth:`canonical_payload`);
        2. every unmapped source field, normalised deterministically;
        3. the set of issue codes raised while normalising the record.

        Including (2) is what stops two rows that agree on every SIH column but
        carry different unmapped evidence (``block_height`` 840000 vs 840001,
        say) from collapsing into one and losing a row.  Including (3) stops a
        merge from discarding a normalisation warning -- a ``RAGGED_RECORD``
        notice depends on the source row's *shape*, which no field value
        captures.

        When two records share an identity payload, merging them is provably
        lossless: every field, every unmapped value and every reported issue
        is identical, so only redundant provenance differs.
        """
        payload = self.canonical_payload()
        # Reserved keys cannot collide with a canonical field name.
        payload["_extra"] = canonicalize_evidence(dict(self.extra or {}))
        payload["_issues"] = sorted(
            "%s:%s" % (issue.code, issue.field or "") for issue in self.issues
        )
        return payload

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_raw(
        cls,
        raw: Mapping[str, Any],
        provenance: Provenance,
        *,
        strict_empty: bool = False,
    ) -> "NormalizedRecord":
        """Normalise a raw field mapping into a validated record.

        Each canonical field is validated independently.  A field that fails
        validation is set to its empty value and reported as an
        ``ERROR`` issue carrying the original input, so the record survives
        with everything that *was* valid and nothing is silently corrupted.

        :param strict_empty: when ``True`` a record with no canonical data at
            all raises :class:`ValueError` instead of being returned with a
            ``NO_DATA`` issue.
        """
        canonical, extra = normalize_keys(raw)
        issues: list[ValidationIssue] = []
        values: dict[str, Any] = {}

        # -- timestamp (handled separately: naive-ness must be reported) --
        raw_timestamp = canonical.get("timestamp")
        timestamp: Optional[datetime] = None
        if raw_timestamp is not None and not _is_blank(raw_timestamp):
            try:
                timestamp, was_naive = parse_timestamp_detailed(raw_timestamp)
            except ValueError as exc:
                issues.append(
                    ValidationIssue(
                        field="timestamp",
                        code="INVALID_TIMESTAMP",
                        message=str(exc),
                        severity=IssueSeverity.ERROR,
                        raw_value=_safe_raw(raw_timestamp),
                        source_record=provenance.source_record,
                    )
                )
            else:
                if was_naive:
                    issues.append(
                        ValidationIssue(
                            field="timestamp",
                            code="NAIVE_TIMESTAMP",
                            message=(
                                "timestamp %r carries no timezone; it was kept as "
                                "naive rather than being assigned the host's local "
                                "timezone. Supply an offset (e.g. 'Z') for an "
                                "unambiguous UTC instant."
                                % (raw_timestamp,)
                            ),
                            severity=IssueSeverity.WARNING,
                            raw_value=_safe_raw(raw_timestamp),
                            source_record=provenance.source_record,
                        )
                    )
        values["timestamp"] = timestamp

        # -- every other canonical field --
        for field, adapter in FIELD_ADAPTERS.items():
            if field not in canonical:
                values[field] = _empty_for(field)
                continue
            raw_value = canonical[field]
            if _is_blank(raw_value):
                values[field] = _empty_for(field)
                continue
            try:
                values[field] = adapter.validate_python(raw_value)
            except ValidationError as exc:
                detail = _first_message(exc)
                issues.append(
                    ValidationIssue(
                        field=field,
                        code="INVALID_FIELD",
                        message=detail,
                        severity=IssueSeverity.ERROR,
                        raw_value=_safe_raw(raw_value),
                        source_record=provenance.source_record,
                    )
                )
                values[field] = _empty_for(field)

        record = cls(provenance=provenance, extra=extra, issues=tuple(issues), **values)

        if not record.has_any_data:
            if strict_empty:
                raise ValueError("record contains no canonical SIH fields")
            record = record.model_copy(
                update={
                    "issues": record.issues
                    + (
                        ValidationIssue(
                            field=None,
                            code="NO_DATA",
                            message=(
                                "record carried none of the canonical SIH fields; "
                                "it may be a blank or ragged source row."
                            ),
                            severity=IssueSeverity.WARNING,
                            source_record=provenance.source_record,
                        ),
                    )
                }
            )
        return record


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _empty_for(field: str) -> Any:
    if field in _LIST_FIELDS:
        return []
    return None


def _safe_raw(value: Any) -> Any:
    """Render an offending value for an issue without risking serialisation."""
    if value is None or isinstance(value, (str, int, float, bool, Decimal)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value][:20]
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in list(value.items())[:20]}
    return str(value)


def _first_message(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:  # pragma: no cover - pydantic always reports at least one
        return "invalid value"
    return str(errors[0].get("msg", "invalid value"))
