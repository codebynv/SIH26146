"""Evidence relationship models (SIH26146 - Step 4A).

This module defines the *semantics* of a derived relationship between
observations.  It deliberately contains no engine: nothing here finds a
relationship, scores it, stores it, or looks one up.  A rule that produces
relationships does not exist yet, and a relationship can currently only be
constructed explicitly, by hand, by whoever understands the evidence.

The non-negotiable property
---------------------------
An :class:`EvidenceRelationship` records *that a match exists* and *how strong
the evidence for it is*.  It never records *who or what the match implies*.

There is no ownership field, no person, no entity, no wallet-control
assertion, and no attribution anywhere on this model -- and because the model
is ``extra="forbid"``, none can be attached later by accident.  A
``SAME_IP`` + ``CORRELATED`` relationship means:

    "Two observations share the same IP address according to a deterministic
     rule, and here is the rule, its version, the reasoning and the source
     records."

It does **not** mean the same person, device, organisation or wallet was
involved, and it must never be rendered or summarised as though it did.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.app.correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    InferenceLevel,
    TemporalRelation,
)
from backend.app.correlation.provenance import ProvenanceRef

__all__ = [
    "RELATIONSHIP_NAMESPACE",
    "Confidence",
    "TemporalWindow",
    "EvidenceRelationship",
    "derive_relationship_id",
]

#: Stable UUIDv5 namespace for deriving deterministic relationship
#: identifiers.  Deliberately different from the observation namespace in
#: :mod:`backend.app.domain.models`, so a relationship id can never collide
#: with an observation id even if the two payloads were ever identical.
RELATIONSHIP_NAMESPACE = uuid.UUID("b3d7c1a4-6e28-5f3b-8c1d-7a4b9e2f5d63")


def _require_non_blank(value: str, field: str) -> str:
    """Reject empty or whitespace-only text for a field that must be present.

    A blank ``rule_id`` or ``rationale`` is not a harmless default.  It reads
    as though the link were explained and reproducible when it is neither, so
    these fields are refused rather than defaulted.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "%s must be a non-empty, non-blank string; a relationship without it "
            "cannot be explained or reproduced" % field
        )
    return value


#: A confidence value constrained to the closed unit interval, with NaN and
#: both infinities refused.  ``allow_inf_nan=False`` is the substantive part:
#: a NaN would propagate silently through comparisons and a bare infinity
#: would look like maximal confidence.
class Confidence(BaseModel):
    """A typed, bounded, explained confidence figure.

    A bare ``float`` is not used for this.  A float says how strong the
    evidence is but not *where the number came from*, so it cannot be audited
    or later reproduced.  :attr:`basis` records the origin, which is what
    makes the value reviewable.

    The number is the strength of the *evidence for the stated relationship*.
    It is never the probability that a real-world identity, ownership or
    attribution claim is true, because no such claim is made.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Strength of evidence for the stated relationship, in the "
        "closed interval [0, 1]. Rejects NaN and infinities.",
    )
    basis: ConfidenceBasis = Field(
        description="Documented origin of this figure.  There is no "
        "machine-learned basis in Step 4A because no model produces one."
    )

    @field_validator("value")
    @classmethod
    def _reject_non_finite(cls, value: float) -> float:
        # Defence in depth: the Field constraints already refuse these, but an
        # explicit check keeps the guarantee local and obvious.
        if value != value:  # NaN is the only value not equal to itself.
            raise ValueError("confidence must be a real number, not NaN")
        if value in (float("inf"), float("-inf")):
            raise ValueError("confidence must be finite, not infinite")
        return value

    @classmethod
    def exact_match(cls) -> "Confidence":
        """Confidence for a value equality, where the comparison is certain.

        Certainty is asserted about the *comparison* only.  Two records
        carrying the same IP really do carry the same IP; whether that implies
        anything about a person or a wallet remains unestablished.
        """
        return cls(value=1.0, basis=ConfidenceBasis.EXACT_MATCH)

    @classmethod
    def from_rule(cls, value: float) -> "Confidence":
        """Confidence assigned by a named rule under a documented policy."""
        return cls(value=value, basis=ConfidenceBasis.DETERMINISTIC_RULE)

    @classmethod
    def from_assessment(cls, value: float) -> "Confidence":
        """Confidence supplied by an analyst, attributable to that analyst."""
        return cls(value=value, basis=ConfidenceBasis.HUMAN_ASSESSMENT)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "%.3f (%s)" % (self.value, self.basis.value)


class TemporalWindow(BaseModel):
    """An explicit, timezone-aware span of time.

    Both bounds are required and must be timezone-aware.  A naive datetime is
    refused rather than localised, because a derived comparison window is
    never "local time by default" -- assuming so would silently shift
    relationships by the host's offset and produce confidently wrong temporal
    conclusions.  This mirrors the Step 2 rule that a naive source timestamp is
    preserved as naive and never stamped with a local zone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: datetime = Field(description="Inclusive start of the window, tz-aware.")
    end: datetime = Field(description="Inclusive end of the window, tz-aware.")

    @field_validator("start", "end")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(
                "temporal window bounds must be timezone-aware; a naive datetime "
                "would be interpreted against the host's local zone and would "
                "silently shift the window"
            )
        return value

    @model_validator(mode="after")
    def _check_order(self) -> "TemporalWindow":
        if self.start > self.end:
            raise ValueError("temporal window start must not be after its end")
        return self

    @property
    def duration_seconds(self) -> float:
        """Length of the window in seconds."""
        return (self.end - self.start).total_seconds()

    def contains(self, moment: datetime) -> bool:
        """True when *moment* falls inside this window (bounds inclusive)."""
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            raise ValueError("cannot compare against a timezone-naive moment")
        return self.start <= moment <= self.end

    def overlaps(self, other: "TemporalWindow") -> bool:
        """True when the two windows share at least one instant.

        A definitional property of two intervals, provided so the semantics of
        :class:`~backend.app.correlation.enums.TemporalRelation` are testable.
        This is not a temporal index or a matching pass; nothing is searched.
        """
        return self.start <= other.end and other.start <= self.end


class EvidenceRelationship(BaseModel):
    """A derived, explainable relationship between observations.

    A relationship is a *claim about the evidence*: that two groups of
    observations match in some stated way, under a named and versioned rule,
    with a confidence figure and a human-readable reason.

    It is emphatically **not** a claim about the world.  There is no field
    here for an owner, a person, an organisation, a wallet or any attribution
    conclusion, and ``extra="forbid"`` means one cannot be smuggled in as an
    extra field either.

    Every relationship is derived, so :attr:`rule_id` and :attr:`rule_version`
    are mandatory.  A relationship that cannot name the rule that produced it
    cannot be reproduced, and one that cannot be reproduced cannot be defended
    in an investigation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relationship_id: str = Field(
        description="Deterministic identifier for this relationship, derived "
        "from its semantic content."
    )
    source_observation_ids: tuple[str, ...] = Field(
        description="Observations on the source side. Non-empty; every entry "
        "must be a non-blank observation identifier."
    )
    target_observation_ids: tuple[str, ...] = Field(
        description="Observations on the target side. Non-empty; every entry "
        "must be a non-blank observation identifier."
    )
    evidence_type: EvidenceType = Field(
        description="What kind of match this is. Answers 'what evidence is "
        "this?', never 'what does it imply?'."
    )
    inference_level: InferenceLevel = Field(
        description="How the relationship may be characterised. Answers 'how "
        "should this be read?', and is a separate axis from evidence_type."
    )
    confidence: Confidence = Field(
        description="Strength of evidence for the stated match."
    )
    rationale: str = Field(
        description="Human-readable explanation of why this relationship was "
        "recorded. Must not be blank."
    )
    rule_id: str = Field(
        description="Stable identifier of the rule that produced this "
        "relationship. Mandatory: a relationship without it cannot be reproduced."
    )
    rule_version: str = Field(
        description="Version of the rule that produced this relationship, so a "
        "later result can be reproduced or deliberately not reproduced."
    )
    provenance_refs: tuple[ProvenanceRef, ...] = Field(
        default=(),
        description="Citations back to the source evidence this relationship "
        "rests on. Retained, never dropped, so a reader can always walk back to "
        "the original records.",
    )
    created_at: datetime = Field(
        description="When the relationship was recorded. Timezone-aware. "
        "Excluded from the derived identifier, so re-deriving the same "
        "relationship yields the same id."
    )
    temporal_window: Optional[TemporalWindow] = Field(
        default=None,
        description="Optional explicit span the relationship is stated over.",
    )
    temporal_relation: Optional[TemporalRelation] = Field(
        default=None,
        description="Optional relation to the other side's window. Meaningful "
        "only alongside temporal_window.",
    )

    @field_validator("created_at")
    @classmethod
    def _require_created_at_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(
                "created_at must be timezone-aware; a naive timestamp would be "
                "interpreted against the host's local zone"
            )
        return value

    @field_validator("source_observation_ids", "target_observation_ids")
    @classmethod
    def _check_observation_ids(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        label = "source" if "source" in str(info.field_name) else "target"
        if not value:
            raise ValueError(
                "%s_observation_ids must name at least one observation; a "
                "relationship with no %s is meaningless" % (label, label)
            )
        for observation_id in value:
            _require_non_blank(observation_id, "%s_observation_ids entry" % label)
        if len(set(value)) != len(value):
            raise ValueError(
                "%s_observation_ids must not repeat an observation; duplicate "
                "entries do not add evidence" % label
            )
        return value

    @field_validator("rationale", "rule_id", "rule_version")
    @classmethod
    def _check_explanatory_text(cls, value: str, info: Any) -> str:
        return _require_non_blank(value, info.field_name)

    @field_validator("rationale")
    @classmethod
    def _check_rationale_is_explanatory(cls, value: str) -> str:
        if len(value.strip()) < 10:
            raise ValueError(
                "rationale must actually explain the relationship, not merely "
                "be non-empty"
            )
        return value

    @model_validator(mode="after")
    def _check_temporal_pair(self) -> "EvidenceRelationship":
        if self.temporal_relation is not None and self.temporal_window is None:
            raise ValueError(
                "temporal_relation requires temporal_window; a relation with no "
                "window states an ordering against nothing"
            )
        return self

    @property
    def is_self_referential(self) -> bool:
        """True when the same observations appear on both sides.

        Reported, not rejected.  Whether a self-referential relationship is
        meaningful depends on the rule that produced it, and that rule does
        not exist yet; encoding a guess here would either ban a legitimate
        future case or endorse a meaningless one.  A rule is the right place to
        decide, and Step 4B will have to decide it explicitly.
        """
        return bool(set(self.source_observation_ids) & set(self.target_observation_ids))

    @property
    def all_observation_ids(self) -> tuple[str, ...]:
        """Every observation this relationship refers to, de-duplicated."""
        seen: dict[str, None] = {}
        for observation_id in self.source_observation_ids + self.target_observation_ids:
            seen.setdefault(observation_id, None)
        return tuple(seen)

    def explains_attribution(self) -> bool:
        """Always ``False``.

        Present as an explicit, testable statement of the model's purpose: no
        relationship produced by this model ever asserts ownership, identity or
        attribution.  It exists so that a caller can assert the guarantee
        rather than infer it from absence, and so that a future change which
        quietly introduced attribution would fail loudly.
        """
        return False

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "%s: %s/%s %s %s" % (
            self.relationship_id,
            self.evidence_type.value,
            self.inference_level.value,
            len(self.source_observation_ids),
            len(self.target_observation_ids),
        )


def derive_relationship_id(payload: dict[str, Any]) -> str:
    """Derive a deterministic relationship identifier from *payload*.

    A UUIDv5 over a canonical JSON rendering, so the same relationship derived
    twice yields the same id.  Callers should pass the *semantic* content only
    -- evidence type, inference level, rule identity, observation ids,
    confidence, rationale.  Wall-clock fields such as ``created_at`` are
    deliberately excluded by the caller, because a relationship re-derived
    tomorrow is the same relationship and must not acquire a new identity just
    because it was computed at a different moment.
    """
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    return str(uuid.uuid5(RELATIONSHIP_NAMESPACE, canonical))
