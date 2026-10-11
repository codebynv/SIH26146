"""Typed, immutable behavioral feature models (SIH26146 - Step 5).

Step 5 exists to *describe* observable behaviour.  It turns observations and
recorded evidence relationships into counts, spans, sums and distinct-value
lists, and it stops there.  Nothing in this module decides what any of that
means: no verdict, no score, no grade, no grouping judgement, no claim about
who or what stands behind a record.  Deciding meaning is Step 6 and later, and
the boundary is enforced structurally -- there is simply no field here that
could hold a judgement.

The vocabulary, in order
------------------------
:class:`BehavioralFeature`
    One named measurement.  A name, a category, a scope, a *typed* value, the
    records that contributed to it, and a content-derived identifier.

:class:`MeasurementStatus`
    Whether the value was measured, could not be measured, or was measured
    over a subset chosen by an explicit policy.

    The empty-set policy this module commits to
    ------------------------------------------
    ``ABSENT`` means "the data this scalar needs was not present", and the
    value is ``None``.  It is used for scalar measurements only: totals,
    spans, averages and instants.

    A count of zero and an empty distinct-value list are *not* ``ABSENT``.
    Zero is a real measurement -- the extractor looked for transactions and
    found none -- and reporting it as "not applicable" would make an empty
    result indistinguishable from an unexamined one.  So:

    * ``CountValue(0, OBSERVED)`` means "zero were present";
    * ``LabelSetValue((), OBSERVED)`` means "none were present";
    * ``DecimalValue(None, ABSENT)`` means "no amount data exists at all".

    A zero *amount* is a third, separate thing: it is a present amount whose
    value is zero, and it is reported as ``DecimalValue(Decimal("0"),
    OBSERVED)``.  Absence is never dressed up as zero.

:class:`FeatureOrigin`
    Which observations and which recorded relationships fed a feature.  It is
    how a reader audits a number without re-running the extraction.

Determinism
-----------
Everything is frozen, ordered by content, and identified by a UUIDv5 over the
identifier of the thing being described.  No clock, no randomness, no ambient
:mod:`decimal` context: :data:`AMOUNT_STATISTICS_CONTEXT` pins precision and
rounding so a mean cannot change because a caller left a different decimal
context in place.
"""

from __future__ import annotations

import decimal
import json
import re
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AMOUNT_STATISTICS_CONTEXT",
    "AmountKind",
    "BEHAVIOR_FEATURE_NAMESPACE",
    "BEHAVIOR_PROFILE_NAMESPACE",
    "BehavioralFeature",
    "CountValue",
    "DecimalValue",
    "FeatureCategory",
    "FeatureOrigin",
    "FeatureScope",
    "FeatureValue",
    "LabelSetValue",
    "LabelValue",
    "MeasurementStatus",
    "TimestampKind",
    "TimestampState",
    "TimestampValue",
    "derive_feature_id",
    "divide_exactly",
]

#: Stable UUIDv5 namespace for behavioral feature identifiers.
BEHAVIOR_FEATURE_NAMESPACE = uuid.UUID("3d5a7c18-6e42-5b90-a1f7-8c04d9e2b6a3")

#: Stable UUIDv5 namespace for behavioral profile identifiers.
BEHAVIOR_PROFILE_NAMESPACE = uuid.UUID("9b6f2a05-84d7-5e31-9c66-0b1e3f7a2d58")

#: The decimal context used for every amount statistic.  Pinned rather than
#: inherited so that a mean computed here cannot change because the calling
#: process happens to have a different precision or rounding mode installed.
AMOUNT_STATISTICS_CONTEXT = decimal.Context(
    prec=34,
    rounding=decimal.ROUND_HALF_EVEN,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[decimal.InvalidOperation, decimal.DivisionByZero],
)

_FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_IDENTIFIER_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def divide_exactly(total: Decimal, divisor: Union[int, Decimal]) -> Decimal:
    """Divide *total* by *divisor* under :data:`AMOUNT_STATISTICS_CONTEXT`.

    The context is entered explicitly rather than inherited, because decimal
    precision is process-global state.  A caller who lowered it to 3 would
    otherwise silently change every mean, median and rate this layer reports,
    which would make results depend on import order.  *divisor* is required to
    be positive: a zero or negative divisor is a programming error, not a
    measurement.
    """
    if divisor <= 0:
        raise ValueError("divisor must be positive, got %r" % (divisor,))
    with decimal.localcontext(AMOUNT_STATISTICS_CONTEXT):
        return +(total / decimal.Decimal(divisor))


def _render_decimal(value: Decimal) -> str:
    """Render a monetary value exactly, never as a float.

    Bitcoin amounts are :class:`~decimal.Decimal` precisely because binary
    floating point cannot represent them exactly.  Converting back to ``float``
    here would undo that guarantee at the one boundary a downstream consumer is
    guaranteed to read, so the exact decimal string is what gets serialised.
    """
    return str(value)


def _render_timestamp(value: datetime) -> str:
    """Render a timestamp as ISO-8601 without attaching or changing an offset."""
    return value.isoformat()


class FeatureCategory(str, Enum):
    """Which family of description a feature belongs to.

    The categories name *kinds of measurement*, not kinds of behaviour, and
    never kinds of behaviour that matters: there is no "suspicious activity"
    bucket here to fill, because this layer has no opinion about which bucket
    anything belongs in.
    """

    ACTIVITY_VOLUME = "ACTIVITY_VOLUME"
    TEMPORAL_ACTIVITY = "TEMPORAL_ACTIVITY"
    TRANSACTION_AMOUNTS = "TRANSACTION_AMOUNTS"
    NETWORK_BEHAVIOR = "NETWORK_BEHAVIOR"
    ADDRESS_PARTICIPATION = "ADDRESS_PARTICIPATION"
    EVIDENCE_GRAPH = "EVIDENCE_GRAPH"
    PROVENANCE_DIVERSITY = "PROVENANCE_DIVERSITY"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class FeatureScope(str, Enum):
    """What a feature was measured over."""

    OBSERVATION = "OBSERVATION"
    PROFILE = "PROFILE"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class MeasurementStatus(str, Enum):
    """How a value came to be, when there is a value to describe.

    ``OBSERVED``
        Measured from every record in scope that carries the needed data.
        A count of zero, an empty distinct-value list and a present amount of
        zero are all ``OBSERVED``: the data was looked for and this is what it
        turned out to be.

    ``ABSENT``
        There is no value to report: the data needed to measure it does not
        exist, or the thing being counted was never supplied.  The value is
        ``None``.  This is the *explicit absence* the layer promises instead
        of a fabricated zero -- a total amount with no amount data is
        ``ABSENT``, while a total that genuinely came to zero is
        ``OBSERVED`` with a zero value, and a relationship count with no
        evidence graph to count in is ``ABSENT`` while one over an empty graph
        is ``OBSERVED`` with zero.

    ``PARTIAL``
        Measured over a subset chosen by an explicit, documented policy, with
        :attr:`~DecimalValue.excluded_count` recording how many records were
        left out and why.  Timezone policy is the one case: a naive timestamp
        is never placed on a UTC timeline, so it is excluded from the
        timezone-aware span features and counted instead.
    """

    OBSERVED = "OBSERVED"
    ABSENT = "ABSENT"
    PARTIAL = "PARTIAL"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class TimestampKind(str, Enum):
    """Whether a timestamp carries a usable timezone."""

    AWARE = "AWARE"
    NAIVE = "NAIVE"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class AmountKind(str, Enum):
    """Which statistic a monetary value is.

    Present so a consumer can tell a total from a mean from a median without
    parsing the feature name.  A total and a median are both "a number of
    bitcoin", and a consumer that has to string-match a name to tell them
    apart is a consumer one rename away from being wrong.
    """

    TOTAL_INPUT = "TOTAL_INPUT"
    TOTAL_OUTPUT = "TOTAL_OUTPUT"
    TOTAL_FEE = "TOTAL_FEE"
    MINIMUM = "MINIMUM"
    MAXIMUM = "MAXIMUM"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class TimestampState(str, Enum):
    """The timezone composition of a set of timestamps.

    A label rather than a number, because the answer "what is the earliest
    event here?" is genuinely different in each of these four cases, and
    collapsing them would hide the difference.
    """

    NONE = "NONE"
    AWARE_ONLY = "AWARE_ONLY"
    NAIVE_ONLY = "NAIVE_ONLY"
    MIXED = "MIXED"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


def _require_non_blank(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty, non-blank string" % field)
    return value


def _require_sorted_unique(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    """Reject an identifier tuple that is not sorted and duplicate-free.

    Enforced rather than repaired.  An ordering that only holds "because the
    producer happened to sort it" is not a guarantee, and a determinism claim
    is only worth making if a violation is visible rather than invisible.
    """
    for item in values:
        _require_non_blank(item, "%s entry" % field)
    if tuple(sorted(set(values))) != values:
        raise ValueError(
            "%s must be sorted and free of duplicates" % field
        )
    return values


class CountValue(BaseModel):
    """A counted number of records, fields or relationships.

    Zero and ``ABSENT`` are different answers and both are expressible.  A
    count of zero means the extractor looked for the thing and there were
    none -- a real result, and always ``OBSERVED``.  ``ABSENT`` means the
    thing being counted was never supplied at all, for instance when no
    evidence graph was given to measure against.  Reporting the second case as
    zero would claim "there are no recorded relationships" when the truth is
    "no graph was provided to look in".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["COUNT"] = "COUNT"
    status: MeasurementStatus = MeasurementStatus.OBSERVED
    value: Optional[int] = Field(
        default=None,
        strict=True,
        description="The counted number, or None when ABSENT.",
    )
    excluded_count: int = Field(
        default=0, ge=0, description="Records left out by an explicit policy."
    )
    unit: Optional[str] = Field(default=None, description="Unit label, if any.")

    @field_validator("value")
    @classmethod
    def _check_value(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("a count must be an integer, not %r" % (value,))
        if value < 0:
            raise ValueError("a count must not be negative, got %d" % value)
        return value

    @model_validator(mode="after")
    def _check_status(self) -> "CountValue":
        if (self.status is MeasurementStatus.ABSENT) != (self.value is None):
            raise ValueError(
                "ABSENT means no count and a present status means a count: "
                "status %s does not match value %r" % (self.status.value, self.value)
            )
        if self.status is MeasurementStatus.PARTIAL and self.excluded_count < 1:
            raise ValueError(
                "a PARTIAL count must say how many records were excluded"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status.value,
            "value": self.value,
            "excluded_count": self.excluded_count,
            "unit": self.unit,
        }


class DecimalValue(BaseModel):
    """An exact decimal measurement: an amount, a span or a rate.

    ``ABSENT`` is allowed and means "no data of this kind exists"; the value is
    then ``None``.  A present value of zero is ``OBSERVED`` with a zero
    :class:`~decimal.Decimal`, and is never reported as ``ABSENT``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["DECIMAL"] = "DECIMAL"
    status: MeasurementStatus = MeasurementStatus.OBSERVED
    value: Optional[Decimal] = Field(
        default=None,
        strict=True,
        description=(
            "The exact value, or None when ABSENT. Strict on purpose: a float, "
            "an int or a numeric string would all be silently coerced to a "
            "Decimal here, which is the exact loss of precision this type "
            "exists to prevent."
        ),
    )
    excluded_count: int = Field(default=0, ge=0)
    unit: Optional[str] = Field(
        default=None, description="e.g. 'BTC', 'seconds' or 'events/day'."
    )
    amount_kind: Optional[AmountKind] = Field(
        default=None,
        description="Which statistic a monetary value is; null for spans and rates.",
    )

    @field_validator("value")
    @classmethod
    def _check_value(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Decimal):
            raise ValueError("a decimal measurement must be a Decimal, not %r" % (value,))
        if not value.is_finite():
            raise ValueError("a decimal measurement must be finite, got %s" % value)
        return value

    @model_validator(mode="after")
    def _check_status(self) -> "DecimalValue":
        if (self.status is MeasurementStatus.ABSENT) != (self.value is None):
            raise ValueError(
                "ABSENT means no value and a present status means a value: "
                "status %s does not match value %r" % (self.status.value, self.value)
            )
        if self.status is MeasurementStatus.PARTIAL and self.excluded_count < 1:
            raise ValueError("a PARTIAL measurement must record its exclusions")
        if self.amount_kind is not None and self.unit != "BTC":
            raise ValueError(
                "amount_kind applies to monetary values only, but unit is %r"
                % (self.unit,)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status.value,
            "value": None if self.value is None else _render_decimal(self.value),
            "excluded_count": self.excluded_count,
            "unit": self.unit,
            "amount_kind": None if self.amount_kind is None else self.amount_kind.value,
        }


class TimestampValue(BaseModel):
    """An instant on a timeline, explicitly tagged aware or naive.

    The tag is not decoration.  A naive timestamp has no known offset, so
    placing it next to a UTC one would be an assumption, not a measurement; a
    consumer that can see the tag can decide what to do without being misled
    by a value that merely looks like a UTC instant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["TIMESTAMP"] = "TIMESTAMP"
    status: MeasurementStatus = MeasurementStatus.OBSERVED
    value: Optional[datetime] = Field(default=None, strict=True)
    excluded_count: int = Field(default=0, ge=0)
    timestamp_kind: TimestampKind = TimestampKind.AWARE
    unit: Optional[str] = Field(default=None)

    @field_validator("value")
    @classmethod
    def _check_value(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ValueError("a timestamp measurement must be a datetime")
        return value

    @model_validator(mode="after")
    def _check_status(self) -> "TimestampValue":
        if (self.status is MeasurementStatus.ABSENT) != (self.value is None):
            raise ValueError(
                "ABSENT means no instant and a present status means an instant: "
                "status %s does not match value %r" % (self.status.value, self.value)
            )
        if self.status is MeasurementStatus.PARTIAL and self.excluded_count < 1:
            raise ValueError("a PARTIAL measurement must record its exclusions")
        if self.value is not None:
            is_naive = self.value.tzinfo is None or self.value.tzinfo.utcoffset(
                self.value
            ) is None
            if is_naive != (self.timestamp_kind is TimestampKind.NAIVE):
                raise ValueError(
                    "timestamp_kind %s contradicts the value's timezone: %r"
                    % (self.timestamp_kind.value, self.value)
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status.value,
            "value": None if self.value is None else _render_timestamp(self.value),
            "excluded_count": self.excluded_count,
            "timestamp_kind": self.timestamp_kind.value,
            "unit": self.unit,
        }


class LabelValue(BaseModel):
    """A single descriptive label, such as a timezone composition.

    ``ABSENT`` means the label does not exist for this data set; a label is
    never ``ABSENT`` merely for being uninformative.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["LABEL"] = "LABEL"
    status: MeasurementStatus = MeasurementStatus.OBSERVED
    value: Optional[str] = Field(default=None, strict=True)
    excluded_count: int = Field(default=0, ge=0)
    unit: Optional[str] = Field(default=None)

    @field_validator("value")
    @classmethod
    def _check_value(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        _require_non_blank(value, "a label value")
        return value

    @model_validator(mode="after")
    def _check_status(self) -> "LabelValue":
        if (self.status is MeasurementStatus.ABSENT) != (self.value is None):
            raise ValueError(
                "ABSENT means no label and a present status means a label: "
                "status %s does not match value %r" % (self.status.value, self.value)
            )
        if self.status is MeasurementStatus.PARTIAL and self.excluded_count < 1:
            raise ValueError("a PARTIAL measurement must record its exclusions")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status.value,
            "value": self.value,
            "excluded_count": self.excluded_count,
            "unit": self.unit,
        }


class LabelSetValue(BaseModel):
    """The sorted, duplicate-free set of distinct values actually present.

    An empty tuple is ``OBSERVED``: it says the extractor looked and there were
    none.  ``ABSENT`` is not available here, and that is the point -- "no
    distinct source IP address appeared" is an answer, not a missing
    measurement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["LABEL_SET"] = "LABEL_SET"
    status: MeasurementStatus = MeasurementStatus.OBSERVED
    value: tuple[str, ...] = Field(
        default=(), strict=True, description="Sorted, duplicate-free distinct values."
    )
    excluded_count: int = Field(default=0, ge=0)
    unit: Optional[str] = Field(default=None)

    @field_validator("value")
    @classmethod
    def _check_value(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_sorted_unique(tuple(value), "a distinct-value set")

    @model_validator(mode="after")
    def _check_status(self) -> "LabelSetValue":
        if self.status is MeasurementStatus.ABSENT:
            raise ValueError(
                "a distinct-value set cannot be ABSENT: an empty set is the "
                "measured answer that no distinct value was present"
            )
        if self.status is MeasurementStatus.PARTIAL and self.excluded_count < 1:
            raise ValueError("a PARTIAL measurement must record its exclusions")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status.value,
            "value": list(self.value),
            "excluded_count": self.excluded_count,
            "unit": self.unit,
        }


#: The typed feature values.  A discriminated union rather than a
#: ``dict[str, Any]``, so a consumer gets a checked shape and the extractor
#: cannot invent a feature that holds something the model has no name for.
FeatureValue = Annotated[
    Union[CountValue, DecimalValue, TimestampValue, LabelValue, LabelSetValue],
    Field(discriminator="kind"),
]


class FeatureOrigin(BaseModel):
    """The records and relationships a feature was measured from.

    This is what makes a number auditable without re-running anything: a
    reader can see which observations carried a source IP address, which
    ingestion runs were involved, and which recorded relationships were
    consulted.  Every tuple is sorted and duplicate-free.

    Absence of a field on an observation is *not* recorded here.  An
    observation with no source IP simply is not among the contributors to the
    source-IP features, which is exactly what a contributor list should say.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_ids: tuple[str, ...] = Field(
        default=(), description="Observations that carried data for this feature."
    )
    ingestion_ids: tuple[str, ...] = Field(default=())
    source_files: tuple[str, ...] = Field(default=())
    source_records: tuple[str, ...] = Field(default=())
    relationship_ids: tuple[str, ...] = Field(
        default=(), description="Recorded relationships consulted for this feature."
    )

    @field_validator(
        "observation_ids", "ingestion_ids", "source_files", "source_records",
        "relationship_ids",
    )
    @classmethod
    def _check_sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_sorted_unique(tuple(value), "a feature origin")

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping with a fixed key order."""
        return {
            "observation_ids": list(self.observation_ids),
            "ingestion_ids": list(self.ingestion_ids),
            "source_files": list(self.source_files),
            "source_records": list(self.source_records),
            "relationship_ids": list(self.relationship_ids),
        }


def derive_feature_id(
    name: str,
    scope: FeatureScope,
    observation_id: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> str:
    """Derive a feature identifier from *what* is described, not from ordering.

    The UUIDv5 payload is the scope, the feature name and the identifier of the
    subject, so the same measurement of the same subject always has the same
    identifier, in any process, on any machine, in any input order.  Two
    features cannot collide by accident: they would need the same name *and*
    the same subject.
    """
    payload = json.dumps(
        {
            "scope": FeatureScope(scope).value,
            "name": name,
            "observation_id": observation_id,
            "profile_id": profile_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(BEHAVIOR_FEATURE_NAMESPACE, payload))


class BehavioralFeature(BaseModel):
    """One named, immutable, typed behavioural measurement.

    A feature answers a question of the form "how much / how many / which
    distinct values / over what span", about a stated subject, from stated
    records.  It is not an opinion, and the model has no field in which one
    could be recorded: no score, no severity, no label describing behaviour as
    good or bad, and no field naming a person, organisation or system as the
    subject of a record.

    Identity is derived, not assigned.  :attr:`feature_id` must equal
    :func:`derive_feature_id` of this feature's own name and subject, so a
    feature cannot claim an identity that does not match what it measures.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_id: str = Field(description="Derived from name, scope and subject.")
    name: str = Field(description="Stable snake_case feature name.")
    category: FeatureCategory
    scope: FeatureScope
    value: FeatureValue
    observation_id: Optional[str] = Field(
        default=None, description="Set for an observation-scoped feature."
    )
    profile_id: Optional[str] = Field(
        default=None, description="Set for a profile-scoped feature."
    )
    origin: FeatureOrigin = Field(default_factory=FeatureOrigin)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        _require_non_blank(value, "a feature name")
        if not _FEATURE_NAME_RE.match(value):
            raise ValueError(
                "feature name %r must be lower snake_case, matching %s"
                % (value, _FEATURE_NAME_RE.pattern)
            )
        return value

    @field_validator("feature_id")
    @classmethod
    def _check_feature_id(cls, value: str) -> str:
        if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
            raise ValueError("feature_id must be a UUID, got %r" % (value,))
        return value

    @field_validator("observation_id", "profile_id")
    @classmethod
    def _check_subject(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _require_non_blank(value, "a feature subject")

    @model_validator(mode="after")
    def _check_consistency(self) -> "BehavioralFeature":
        if self.scope is FeatureScope.OBSERVATION:
            if self.observation_id is None:
                raise ValueError("an OBSERVATION-scoped feature needs an observation_id")
            if self.profile_id is not None:
                raise ValueError(
                    "an OBSERVATION-scoped feature must not also claim a profile_id"
                )
        else:
            if self.observation_id is not None:
                raise ValueError(
                    "a PROFILE-scoped feature must not also claim an observation_id"
                )
        expected = derive_feature_id(
            self.name, self.scope, self.observation_id, self.profile_id
        )
        if self.feature_id != expected:
            raise ValueError(
                "feature_id %s does not match the identifier derived from name %r, "
                "scope %s and subject (expected %s)"
                % (self.feature_id, self.name, self.scope.value, expected)
            )
        return self

    @property
    def status(self) -> MeasurementStatus:
        """The status of the underlying value."""
        return self.value.status

    @property
    def unit(self) -> Optional[str]:
        """The unit of the underlying value, if it has one."""
        return self.value.unit

    def describes_attribution(self) -> bool:
        """Always ``False``.

        Step 5 measures behaviour; it never states what produced a record, who
        held an address, or which party an observation belongs to.  This method
        exists so the absence is checkable by a test rather than merely
        documented: there is no field on this model that could hold such a
        claim, and the one question a reader might ask is answered in the
        negative by construction.
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping with a fixed key order."""
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "category": self.category.value,
            "scope": self.scope.value,
            "observation_id": self.observation_id,
            "profile_id": self.profile_id,
            "value": self.value.to_dict(),
            "origin": self.origin.to_dict(),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "%s=%s (%s)" % (self.name, _value_repr(self.value), self.status.value)


def _value_repr(value: FeatureValue) -> str:
    if isinstance(value, LabelSetValue):
        return "[%s]" % ", ".join(value.value)
    if isinstance(value, DecimalValue) and value.value is not None:
        return _render_decimal(value.value)
    if isinstance(value, TimestampValue) and value.value is not None:
        return "%s %s" % (_render_timestamp(value.value), value.timestamp_kind.value)
    return str(value.value)
