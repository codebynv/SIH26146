"""Typed, immutable evidence-fusion models (SIH26146 - Step 8).

Step 8 is where distinct, already-established layers - Step 4C recorded-evidence
graph, Step 5 behavioural profiles, Step 6 anomaly and cluster results, Step 7
investigation paths with their temporal structure - are brought onto one
explicit table and combined into a single deterministic priority figure.  This
module is the *vocabulary* of that combination and nothing else: the adapters
that read earlier layers live in :mod:`..fusion.signals`, the weighted
aggregation lives in :mod:`..fusion.fusion`, and the explanations live in
:mod:`..fusion.explain`.

What a signal is
----------------
A :class:`FusionSignal` is one typed number about the assessed subject, carrying
its own status, the step that produced it, the identifier of the input object it
came from, the recorded evidence it rests on, the inference levels of that
evidence (verbatim, never upgraded), the provenance citations, a content
derivation trail (``lineage``) and a human-readable explanation.  Every signal
has a content-derived deterministic identifier.

Missing is not zero
-------------------
A signal is ``PRESENT`` only when the number was actually measured.  ``ABSENT``
means the producer object was not supplied, ``UNAVAILABLE`` means it was
supplied but produced nothing for this subject (no model fitted, subject not
scored), and ``NOT_APPLICABLE`` is an explicit "this question has no answer for
this subject".  None of the three contributes a value to the fused score and
none is silently replaced by zero.

What the fused score is (and is not)
------------------------------------
The fused score is a deterministic priority-ranking number in ``[0, 1]``
computed by a disclosed weighted formula.  It is **not** a probability, it is
not an accuracy or detection figure, and it asserts no attribution, ownership,
identity or guilt.  Confidence is retained as explainability context and is
never multiplied into the score.  Inference levels recorded by earlier steps are
carried verbatim and never raised because signals agree.

Determinism
-----------
Every identifier here derives from content with UUIDv5 in a namespace of its
own; there is no clock and no :func:`uuid.uuid4`.  Arithmetic is pinned to
:data:`FUSION_DECIMAL_CONTEXT`, a fixed 34-digit decimal context, so results do
not depend on ambient decimal state or floating-point order.
"""

from __future__ import annotations

import decimal
import json
import re
import uuid
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..correlation.enums import ConfidenceBasis, InferenceLevel
from ..correlation.provenance import ProvenanceRef

__all__ = [
    "ConfidenceContext",
    "EVIDENCE_COUNT_CAP_DEFAULT",
    "ExplainabilityKind",
    "FUSION_DECIMAL_CONTEXT",
    "FUSION_NAMESPACE",
    "FUSION_SIGNAL_NAMESPACE",
    "FusionAssessment",
    "FusionConfig",
    "FusionError",
    "FusionExplanation",
    "FusionSignal",
    "FusionStatus",
    "MissingSignal",
    "PATH_COUNT_CAP_DEFAULT",
    "PROVENANCE_FILE_CAP_DEFAULT",
    "RiskBand",
    "SignalContribution",
    "SignalFamily",
    "SignalRole",
    "SignalStatus",
    "band_for_score",
    "derive_assessment_id",
    "derive_signal_id",
]

#: Stable UUIDv5 namespace for fusion-assessment identifiers.  Deliberately
#: different from every namespace used by Steps 3-7 so an assessment id can
#: never be mistaken for the observation, relationship, node, edge, path or
#: profile identifiers it is built from.
FUSION_NAMESPACE = uuid.UUID("f0a1b2c3-4d5e-6f78-9abc-01d2e3f4a5b6")

#: Stable UUIDv5 namespace for fusion-signal identifiers.
FUSION_SIGNAL_NAMESPACE = uuid.UUID("7c8d9e0f-1a2b-3c4d-5e6f-708192a3b4c5")

#: The decimal context used for every fused-score computation.  Pinned rather
#: than inherited so a mean or a weighted sum cannot change because the calling
#: process happens to have a different precision or rounding mode installed.
FUSION_DECIMAL_CONTEXT = decimal.Context(
    prec=34,
    rounding=decimal.ROUND_HALF_EVEN,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[decimal.InvalidOperation, decimal.DivisionByZero],
)

#: Default caps for the descriptive count signals.  A count is normalised by
#: ``min(1, count / cap)``; the cap is disclosed in the configuration and in
#: every signal that uses it, never hidden.
EVIDENCE_COUNT_CAP_DEFAULT = 20
PATH_COUNT_CAP_DEFAULT = 10
PROVENANCE_FILE_CAP_DEFAULT = 10

_SIGNAL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class FusionError(ValueError):
    """Raised when a fusion assessment cannot be built faithful to the inputs.

    A :class:`ValueError`, matching the pattern of earlier steps, so ordinary
    ``except ValueError`` still works while a caller can tell a fusion-layer
    refusal apart from any other bad value.
    """


class SignalFamily(str, Enum):
    """Which layer of recorded evidence a signal summarises.

    Each family is one already-established kind of measurement: the anomaly
    figure (Step 6), the behavioural profile (Step 5), geometric cluster
    context (Step 6), the recorded-evidence graph (Steps 4A/4C), provenance
    citations (Steps 2/4A), temporal structure of investigation paths
    (Step 7), and path structure (Step 7).
    """

    ANOMALY = "ANOMALY"
    BEHAVIORAL = "BEHAVIORAL"
    CLUSTER_CONTEXT = "CLUSTER_CONTEXT"
    EVIDENCE_GRAPH = "EVIDENCE_GRAPH"
    PATH_STRUCTURE = "PATH_STRUCTURE"
    PROVENANCE = "PROVENANCE"
    TEMPORAL = "TEMPORAL"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class SignalStatus(str, Enum):
    """Whether a signal carries a measured number, and if not, why not.

    ``PRESENT``
        The number was actually measured, including a measured zero where the
        underlying count genuinely is zero.
    ``ABSENT``
        The producer object (graph, run result, path set) was not supplied, so
        the number could not be measured at all.  Missing data, not an answer.
    ``UNAVAILABLE``
        The producer was supplied but produced nothing for this subject: the
        run never fitted, or the subject was not part of the population.
    ``NOT_APPLICABLE``
        The question this signal asks has no answer for this subject by the
        signal's own stated contract (for example, a forward-flow ratio over
        zero comparable transitions).
    """

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class SignalRole(str, Enum):
    """Semantic role of a signal in investigative fusion.

    ``PRIORITY_SIGNAL``
        A direct indicator of investigative priority. Higher normalized value
        indicates higher triage priority (e.g. behavioral anomaly, connection
        density, path reachability).
    ``EVIDENCE_STRENGTH``
        Measures how well-supported, corroborated, or actionable the evidence is
        (e.g. feature coverage, evidence type diversity, provenance citation
        ratio, source file diversity, temporal comparability).
    ``CONTEXT_ONLY``
        Descriptive or geometric context useful for human explanation and path
        validation, but does not contribute to the fused priority score
        (e.g. normal chronological forward flow).
    """

    PRIORITY_SIGNAL = "PRIORITY_SIGNAL"
    EVIDENCE_STRENGTH = "EVIDENCE_STRENGTH"
    CONTEXT_ONLY = "CONTEXT_ONLY"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class RiskBand(str, Enum):
    """A fixed operating point on the fused priority score.

    The bands are *thresholds chosen in advance*, not fitted cut points and
    not judgements about guilt, intent or identity.  ``LOW`` means "the
    recorded evidence, at these weights, ranks low"; ``CRITICAL`` means "the
    recorded evidence, at these weights, ranks highest".  Neither names what
    the subject is or what ought to be done about it.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class FusionStatus(str, Enum):
    """Whether a fused score was produced.

    ``COMPUTED``
        At least one weighted signal was available, so the fused score and its
        band exist.  ``INSUFFICIENT_SIGNALS``
        The weighted availability denominator was zero - every family that
        carries weight produced no PRESENT signal - so no score is invented.
    """

    COMPUTED = "COMPUTED"
    INSUFFICIENT_SIGNALS = "INSUFFICIENT_SIGNALS"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class ExplainabilityKind(str, Enum):
    """The kind of a deterministic explanation item.

    ``TOP_CONTRIBUTOR``
        The family contributing most to the fused score.
    ``SIGNAL_CONTRIBUTION``
        A ranked explanation of one family's contribution.
    ``MISSING_SIGNAL``
        A non-PRESENT signal and why it contributed nothing.
    ``CLUSTER_CONTEXT_NOTE``
        The standing statement that cluster membership is descriptive context
        only.
    ``INFERENCE_LEVEL_NOTE``
        The standing statement that inference levels are carried verbatim.
    ``OVERLAP_NOTE``
        The standing statement that a relationship id may appear in more than
        one family's lineage and that independence is not assumed.
    ``INTERPRETATION_NOTE``
        The standing statement of what the fused score is and is not.
    """

    TOP_CONTRIBUTOR = "TOP_CONTRIBUTOR"
    SIGNAL_CONTRIBUTION = "SIGNAL_CONTRIBUTION"
    MISSING_SIGNAL = "MISSING_SIGNAL"
    CLUSTER_CONTEXT_NOTE = "CLUSTER_CONTEXT_NOTE"
    INFERENCE_LEVEL_NOTE = "INFERENCE_LEVEL_NOTE"
    OVERLAP_NOTE = "OVERLAP_NOTE"
    INTERPRETATION_NOTE = "INTERPRETATION_NOTE"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


def _require_non_blank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty, non-blank string" % field)
    return value


def _require_sorted_unique(
    values: Sequence[str], field: str
) -> Tuple[str, ...]:
    candidate = tuple(values)
    for item in candidate:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("%s entries must be non-blank strings" % field)
    if tuple(sorted(set(candidate))) != candidate:
        raise ValueError("%s must be sorted and free of duplicates" % field)
    return candidate


def _as_decimal(value: Any, field: str) -> decimal.Decimal:
    with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
        number = decimal.Decimal(str(value))
        if not number.is_finite():
            raise ValueError("%s must be a finite decimal, got %s" % (field, number))
        return +number


def canonical_json(payload: Any) -> str:
    """Serialize *payload* to a stable string for derived identifiers.

    Sorted keys, no insignificant whitespace, no locale formatting, and a
    ``str`` fallback so enums and decimals serialize without drifting.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True
    )


def derive_signal_id(
    subject_id: str, signal_type: str, source_step: int, source_id: str
) -> str:
    """Derive the deterministic identifier of a signal from its contents.

    The payload is the assessed subject, the signal type, the step that
    produced it and the identifier of the input object it came from.  No clock
    and no UUID4: the same subject, type, step and source always produce the
    same id, so a duplicate signal is a detectable collision rather than
    something silently added twice.
    """
    if not subject_id.strip():
        raise FusionError("subject_id must be a non-blank string")
    if not source_id.strip():
        raise FusionError("source_id must be a non-blank string")
    if not _SIGNAL_TYPE_RE.match(signal_type or ""):
        raise FusionError(
            "signal_type %r must match %s" % (signal_type, _SIGNAL_TYPE_RE.pattern)
        )
    if isinstance(source_step, bool) or not isinstance(source_step, int):
        raise FusionError("source_step must be an integer")
    payload = {
        "subject_id": subject_id,
        "signal_type": signal_type,
        "source_step": source_step,
        "source_id": source_id,
    }
    return str(uuid.uuid5(FUSION_SIGNAL_NAMESPACE, canonical_json(payload)))


def derive_assessment_id(
    subject_profile_id: str, config_json: str, signal_ids: Sequence[str]
) -> str:
    """Derive the deterministic identifier of an assessment.

    A UUIDv5 over the assessed profile, the *configuration* that produced the
    numbers (so a different weighting policy or threshold set yields a
    different assessment id even for the same signals) and the ordered signal
    ids that entered the arithmetic.
    """
    _require_non_blank(subject_profile_id, "subject_profile_id")
    _require_non_blank(config_json, "config_json")
    for signal_id in signal_ids:
        _require_non_blank(signal_id, "signal_ids entry")
    payload = {
        "subject_profile_id": subject_profile_id,
        "config_json": config_json,
        "signal_ids": sorted(set(signal_ids)),
    }
    return str(uuid.uuid5(FUSION_NAMESPACE, canonical_json(payload)))


def band_for_score(
    score: decimal.Decimal, thresholds: Tuple[decimal.Decimal, decimal.Decimal, decimal.Decimal]
) -> RiskBand:
    """Return the risk band for *score* under ascending *thresholds*.

    ``LOW`` below the first threshold, ``MEDIUM`` up to the second,
    ``HIGH`` up to the third, ``CRITICAL`` at or above it.  The thresholds are
    chosen operating points, disclosed in the configuration, and are not
    statements about the subject.
    """
    if len(thresholds) != 3:
        raise ValueError("thresholds must be exactly (low, medium, high)")
    low, medium, high = thresholds
    if not (decimal.Decimal("0") < low < medium < high < decimal.Decimal("1")):
        raise ValueError("thresholds must be strictly ascending inside (0, 1)")
    if score < low:
        return RiskBand.LOW
    if score < medium:
        return RiskBand.MEDIUM
    if score < high:
        return RiskBand.HIGH
    return RiskBand.CRITICAL


class FusionSignal(BaseModel):
    """One typed, deterministic number about the assessed subject.

    The signal carries everything needed to audit the number without re-running
    anything: its status, the step that produced it, the input object it came
    from, the recorded relationships it rests on, the inference levels of that
    evidence carried verbatim, the provenance citations, a content lineage, and
    a human-readable explanation.  Its identifier is derived from its content,
    so the same signal cannot be added twice by accident.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(description="Derived from subject, type, step and source.")
    signal_type: str = Field(description="Uppercase snake-case signal name.")
    family: SignalFamily
    role: SignalRole = Field(
        default=SignalRole.PRIORITY_SIGNAL,
        description="Semantic role: PRIORITY_SIGNAL, EVIDENCE_STRENGTH, or CONTEXT_ONLY.",
    )
    status: SignalStatus
    source_step: int = Field(
        ge=1, description="Step of the pipeline that produced this signal."
    )
    source_id: str = Field(
        description="Identifier of the input object that produced this signal."
    )
    subject_id: str = Field(
        description="Identifier of the assessed profile this signal describes."
    )
    raw_value: Optional[float] = Field(
        default=None, description="Model- or layer-native value; PRESENT only."
    )
    normalized_value: Optional[float] = Field(
        default=None,
        description="Unit-interval value used in the weighted sum; PRESENT only.",
    )
    explanation: str = Field(
        description="Stated normalization contract and provenance of the number."
    )
    evidence_refs: Tuple[str, ...] = Field(
        default=(),
        description="Relationship ids this signal's evidence rests on, sorted.",
    )
    inference_levels: Tuple[InferenceLevel, ...] = Field(
        default=(),
        description="Inference levels of that evidence, verbatim, sorted.",
    )
    provenance_refs: Tuple[ProvenanceRef, ...] = Field(
        default=(), description="Citations back to source evidence."
    )
    lineage: Tuple[str, ...] = Field(
        default=(),
        description="Content ids (profiles, relationships, paths) this signal "
        "was derived from, sorted.",
    )
    context: Tuple[Tuple[str, str], ...] = Field(
        default=(), description="Key/value descriptors, sorted by key."
    )

    @field_validator("signal_id", "signal_type", "source_id", "subject_id")
    @classmethod
    def _check_text(cls, value: str, info: Any) -> str:
        return _require_non_blank(value, str(info.field_name))

    @field_validator("signal_type")
    @classmethod
    def _check_type_shape(cls, value: str) -> str:
        if not _SIGNAL_TYPE_RE.match(value):
            raise ValueError(
                "signal_type %r must match %s" % (value, _SIGNAL_TYPE_RE.pattern)
            )
        return value

    @field_validator("explanation")
    @classmethod
    def _check_explanation(cls, value: str) -> str:
        _require_non_blank(value, "explanation")
        if len(value.strip()) < 10:
            raise ValueError(
                "explanation must actually explain the signal's normalization "
                "contract, not merely be non-empty"
            )
        return value

    @field_validator("raw_value", "normalized_value")
    @classmethod
    def _check_finite(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not isinstance(value, float) or (value != value) or value in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError("signal values must be finite floats")
        return value

    @field_validator("normalized_value")
    @classmethod
    def _check_normalized_range(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError("normalized_value must lie inside [0, 1]")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def _check_refs(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return _require_sorted_unique(value, "evidence_refs")

    @field_validator("lineage")
    @classmethod
    def _check_lineage(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return _require_sorted_unique(value, "lineage")

    @field_validator("inference_levels")
    @classmethod
    def _check_inference_levels(
        cls, value: Tuple[InferenceLevel, ...]
    ) -> Tuple[InferenceLevel, ...]:
        shown = [item.value for item in value]
        if shown != sorted(shown) or len(shown) != len(set(shown)):
            raise ValueError("inference_levels must be ascending and unique")
        return value

    @field_validator("context")
    @classmethod
    def _check_context(
        cls, value: Tuple[Tuple[str, str], ...]
    ) -> Tuple[Tuple[str, str], ...]:
        produced: list[Tuple[str, str]] = []
        seen: set[str] = set()
        for key, val in value:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("context keys must be non-blank strings")
            if not isinstance(val, str):
                raise ValueError("context values must be strings")
            if key in seen:
                raise ValueError("context keys must be unique")
            seen.add(key)
            produced.append((key, val))
        if [key for key, _ in produced] != sorted(key for key, _ in produced):
            raise ValueError("context must be ascending by key")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "FusionSignal":
        present = self.status is SignalStatus.PRESENT
        has_numbers = self.normalized_value is not None or self.raw_value is not None
        if present and not has_numbers:
            raise ValueError("a PRESENT signal must carry a normalized value")
        if not present and has_numbers:
            raise ValueError(
                "a non-PRESENT signal must carry no numeric value; a missing "
                "signal is never replaced by a zero"
            )
        expected = derive_signal_id(
            self.subject_id, self.signal_type, self.source_step, self.source_id
        )
        if self.signal_id != expected:
            raise ValueError(
                "signal_id %s does not match the identifier derived from "
                "subject %s, type %s, step %d and source %s (expected %s)"
                % (
                    self.signal_id,
                    self.subject_id,
                    self.signal_type,
                    self.source_step,
                    self.source_id,
                    expected,
                )
            )
        return self

    def explains_attribution(self) -> bool:
        """Always ``False``.

        A signal is a number about evidence.  It never says who owns, controls
        or is behind anything, and it has no field that could hold such a
        statement - the method makes the absence checkable.
        """
        return False

    def is_priority_signal(self) -> bool:
        """Whether this signal directly indicates investigative priority."""
        return self.role is SignalRole.PRIORITY_SIGNAL

    def is_evidence_strength(self) -> bool:
        """Whether this signal measures evidence corroboration/actionability."""
        return self.role is SignalRole.EVIDENCE_STRENGTH

    def is_context_only(self) -> bool:
        """Whether this signal is explanatory context that does not alter priority."""
        return self.role is SignalRole.CONTEXT_ONLY

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "family": self.family.value,
            "role": self.role.value,
            "status": self.status.value,
            "source_step": self.source_step,
            "source_id": self.source_id,
            "subject_id": self.subject_id,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "explanation": self.explanation,
            "evidence_refs": list(self.evidence_refs),
            "inference_levels": [item.value for item in self.inference_levels],
            "provenance_refs": [
                {
                    "ingestion_id": ref.ingestion_id,
                    "source_file": ref.source_file,
                    "source_record": ref.source_record,
                }
                for ref in self.provenance_refs
            ],
            "lineage": list(self.lineage),
            "context": dict(self.context),
        }


class SignalContribution(BaseModel):
    """One family's share of the fused score, fully broken down.

    ``value`` is the family's aggregate normalized figure (the mean of its
    PRESENT normalized values), ``availability`` is the share of the family's
    expected signals that were PRESENT, and ``contribution`` is
    ``weight * value * availability``.  ``signal_ids`` names exactly which
    signals entered the family mean, so the figure can be recomputed by a
    reader.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: SignalFamily
    weight: decimal.Decimal = Field(ge=0, description="Configured family weight.")
    normalized_value: decimal.Decimal = Field(ge=0, le=1)
    availability: decimal.Decimal = Field(ge=0, le=1)
    contribution: decimal.Decimal = Field(ge=0)
    present_count: int = Field(ge=0)
    total_count: int = Field(ge=1)
    signal_ids: Tuple[str, ...] = Field(
        default=(), description="Ids of the family's PRESENT signals, sorted."
    )
    explanation: str = Field(
        description="The stated contribution arithmetic in words."
    )

    @field_validator("weight", "normalized_value", "availability", "contribution")
    @classmethod
    def _check_decimals(cls, value: decimal.Decimal, info: Any) -> decimal.Decimal:
        produced = _as_decimal(value, str(info.field_name))
        if produced < 0:
            raise ValueError("%s must be non-negative" % info.field_name)
        return produced

    @field_validator("normalized_value", "availability")
    @classmethod
    def _check_unit(cls, value: decimal.Decimal, info: Any) -> decimal.Decimal:
        if value > 1:
            raise ValueError("%s must not exceed 1" % info.field_name)
        return value

    @field_validator("signal_ids")
    @classmethod
    def _check_signal_ids(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return _require_sorted_unique(value, "signal_ids")

    @field_validator("explanation")
    @classmethod
    def _check_explanation(cls, value: str) -> str:
        _require_non_blank(value, "explanation")
        if len(value.strip()) < 10:
            raise ValueError("explanation must be a real sentence")
        return value

    @model_validator(mode="after")
    def _check_numbers(self) -> "SignalContribution":
        if self.present_count > self.total_count:
            raise ValueError("present_count must not exceed total_count")
        if self.present_count != len(self.signal_ids):
            raise ValueError("signal_ids must name exactly the present signals")
        with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
            expected = self.weight * self.normalized_value * self.availability
        if self.contribution != expected:
            raise ValueError(
                "contribution is not rejection-susceptible: weight * value * "
                "availability = %s, reported %s"
                % (expected, self.contribution)
            )
        return self

    def to_dict(self) -> dict:
        return {
            "family": self.family.value,
            "weight": str(self.weight),
            "normalized_value": str(self.normalized_value),
            "availability": str(self.availability),
            "contribution": str(self.contribution),
            "present_count": self.present_count,
            "total_count": self.total_count,
            "signal_ids": list(self.signal_ids),
            "explanation": self.explanation,
        }


class MissingSignal(BaseModel):
    """A non-PRESENT signal and why it contributed nothing.

    Missing signals are reported, never replaced by a zero.  This object is
    the record of that refusal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str
    signal_type: str
    family: SignalFamily
    status: SignalStatus
    source_step: int = Field(ge=1)
    explanation: str = Field(
        description="Why no number was produced for this subject."
    )

    @field_validator("signal_id", "signal_type")
    @classmethod
    def _check_text(cls, value: str, info: Any) -> str:
        return _require_non_blank(value, str(info.field_name))

    @field_validator("explanation")
    @classmethod
    def _check_explanation(cls, value: str) -> str:
        _require_non_blank(value, "explanation")
        if len(value.strip()) < 10:
            raise ValueError("explanation must be a real sentence")
        return value

    @model_validator(mode="after")
    def _check_status(self) -> "MissingSignal":
        if self.status is SignalStatus.PRESENT:
            raise ValueError("a missing-signal record must not be PRESENT")
        return self

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "family": self.family.value,
            "status": self.status.value,
            "source_step": self.source_step,
            "explanation": self.explanation,
        }


class ConfidenceContext(BaseModel):
    """Edge confidence retained as explainability context only.

    The bounds are the *edges' own* confidence values over the subject's
    incident relationships.  They never enter the fused-score arithmetic: a
    reader sees them, and the standing note says what they are and are not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_edge_confidence: Optional[decimal.Decimal] = Field(
        default=None, description="Lowest edge confidence among incident edges."
    )
    max_edge_confidence: Optional[decimal.Decimal] = Field(
        default=None, description="Highest edge confidence among incident edges."
    )
    considered_edge_count: int = Field(
        default=0, ge=0, description="Incident edges the bounds were computed over."
    )
    bases: Tuple[ConfidenceBasis, ...] = Field(
        default=(), description="Confidence bases seen, ascending."
    )
    note: str = Field(
        default=(
            "Edge confidence is retained as explainability context; it is "
            "never multiplied into the fused score, because confidence is the "
            "strength of each stated relationship, not a weight over the "
            "subject."
        ),
        description="The standing statement of what confidence is here.",
    )

    @field_validator("min_edge_confidence", "max_edge_confidence")
    @classmethod
    def _check_decimals(
        cls, value: Optional[decimal.Decimal]
    ) -> Optional[decimal.Decimal]:
        if value is None:
            return None
        produced = _as_decimal(value, "confidence")
        if not 0 <= produced <= 1:
            raise ValueError("confidence must lie inside [0, 1]")
        return produced

    @field_validator("bases")
    @classmethod
    def _check_bases(
        cls, value: Tuple[ConfidenceBasis, ...]
    ) -> Tuple[ConfidenceBasis, ...]:
        shown = [item.value for item in value]
        if shown != sorted(shown) or len(shown) != len(set(shown)):
            raise ValueError("bases must be ascending and unique")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "ConfidenceContext":
        has_low = self.min_edge_confidence is not None
        has_high = self.max_edge_confidence is not None
        if self.considered_edge_count == 0:
            if has_low or has_high:
                raise ValueError(
                    "no incident edges means no confidence bounds to record"
                )
            return self
        if not has_low or not has_high:
            raise ValueError(
                "a non-empty edge set requires both confidence bounds"
            )
        if self.min_edge_confidence > self.max_edge_confidence:
            raise ValueError(
                "min_edge_confidence must not exceed max_edge_confidence"
            )
        return self

    def to_dict(self) -> dict:
        return {
            "min_edge_confidence": (
                None if self.min_edge_confidence is None else str(self.min_edge_confidence)
            ),
            "max_edge_confidence": (
                None if self.max_edge_confidence is None else str(self.max_edge_confidence)
            ),
            "considered_edge_count": self.considered_edge_count,
            "bases": [item.value for item in self.bases],
            "note": self.note,
        }


class FusionExplanation(BaseModel):
    """One deterministic, machine-reported explanation item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ExplainabilityKind
    text: str = Field(description="The explanation sentence or two.")
    evidence: Tuple[str, ...] = Field(
        default=(), description="Ids cited by this item, sorted; may be empty."
    )

    @field_validator("text")
    @classmethod
    def _check_text(cls, value: str) -> str:
        _require_non_blank(value, "text")
        if len(value.strip()) < 10:
            raise ValueError("explanation text must be a real sentence")
        return value

    @field_validator("evidence")
    @classmethod
    def _check_evidence(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return _require_sorted_unique(value, "evidence")

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "text": self.text, "evidence": list(self.evidence)}


_DEFAULT_WEIGHTS = (
    (SignalFamily.ANOMALY, decimal.Decimal("0.30")),
    (SignalFamily.BEHAVIORAL, decimal.Decimal("0.15")),
    (SignalFamily.CLUSTER_CONTEXT, decimal.Decimal("0.05")),
    (SignalFamily.EVIDENCE_GRAPH, decimal.Decimal("0.20")),
    (SignalFamily.PATH_STRUCTURE, decimal.Decimal("0.10")),
    (SignalFamily.PROVENANCE, decimal.Decimal("0.10")),
    (SignalFamily.TEMPORAL, decimal.Decimal("0.10")),
)

_FORMULA_STATEMENT = (
    "fused = sum over families of (weight * family_value * availability) / "
    "sum over families of (weight * availability); family_value is the mean of "
    "the family's PRESENT scoring signals (PRIORITY_SIGNAL and EVIDENCE_STRENGTH); "
    "availability is the family's PRESENT scoring signal count divided by its "
    "expected scoring signal count.  CONTEXT_ONLY signals provide explanation "
    "context and do not alter the fused priority score."
)

_AVAILABILITY_STATEMENT = (
    "availability is per family: the number of PRESENT scoring signals divided "
    "by the total scoring signals emitted for that family.  Missing signals "
    "(ABSENT / UNAVAILABLE / NOT_APPLICABLE) contribute zero and are never "
    "replaced by a zero value.  CONTEXT_ONLY signals do not alter availability."
)

_INTERPRETATION_STATEMENT = (
    "The fused score is a deterministic priority-ranking number in [0, 1] "
    "produced by the disclosed weighted formula.  It is not a probability, it "
    "is not an accuracy or detection figure, and it asserts no attribution, "
    "ownership, identity or guilt."
)


class FusionConfig(BaseModel):
    """Every choice that can change a fused score.

    Weights, band thresholds, the three disclosed normalisation caps, and the
    standing statements of the formula, the availability rule and the score's
    interpretation.  Frozen and self-contained: an assessment records this
    configuration's canonical JSON, so the exact policy behind its number can
    be replayed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(default="FUSION/1", description="Policy version tag.")
    weights: Tuple[Tuple[SignalFamily, decimal.Decimal], ...] = Field(
        default=_DEFAULT_WEIGHTS,
        description="One non-negative weight per family, ascending by family.",
    )
    band_low: decimal.Decimal = Field(default=decimal.Decimal("0.40"))
    band_medium: decimal.Decimal = Field(default=decimal.Decimal("0.60"))
    band_high: decimal.Decimal = Field(default=decimal.Decimal("0.80"))
    evidence_count_cap: int = Field(default=EVIDENCE_COUNT_CAP_DEFAULT, ge=1)
    path_count_cap: int = Field(default=PATH_COUNT_CAP_DEFAULT, ge=1)
    provenance_file_cap: int = Field(default=PROVENANCE_FILE_CAP_DEFAULT, ge=1)
    formula: str = Field(default=_FORMULA_STATEMENT)
    availability_rule: str = Field(default=_AVAILABILITY_STATEMENT)
    interpretation: str = Field(default=_INTERPRETATION_STATEMENT)

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        return _require_non_blank(value, "version")

    @field_validator("formula", "availability_rule", "interpretation")
    @classmethod
    def _check_statements(cls, value: str, info: Any) -> str:
        _require_non_blank(value, str(info.field_name))
        if len(value.strip()) < 10:
            raise ValueError("%s must be a real statement" % info.field_name)
        return value

    @field_validator("band_low", "band_medium", "band_high")
    @classmethod
    def _check_band(cls, value: decimal.Decimal, info: Any) -> decimal.Decimal:
        produced = _as_decimal(value, str(info.field_name))
        if not decimal.Decimal("0") < produced < decimal.Decimal("1"):
            raise ValueError("%s must lie strictly inside (0, 1)" % info.field_name)
        return produced

    @field_validator("weights")
    @classmethod
    def _check_weights(
        cls, value: Tuple[Tuple[SignalFamily, decimal.Decimal], ...]
    ) -> Tuple[Tuple[SignalFamily, decimal.Decimal], ...]:
        seen: set[SignalFamily] = set()
        produced: list[Tuple[SignalFamily, decimal.Decimal]] = []
        for family, weight in value:
            family = SignalFamily(family)
            if family in seen:
                raise ValueError("a weight must be supplied for each family once")
            seen.add(family)
            number = _as_decimal(weight, "weight for %s" % family.value)
            if number < 0:
                raise ValueError("weights must be non-negative")
            produced.append((family, number))
        if seen != set(SignalFamily):
            raise ValueError(
                "weights must cover every family exactly; missing: %s"
                % sorted(set(SignalFamily) - seen, key=lambda item: item.value)
            )
        if all(number == 0 for _, number in produced):
            raise ValueError("at least one family weight must be non-zero")
        return tuple(sorted(produced, key=lambda pair: pair[0].value))

    @model_validator(mode="after")
    def _check_band_order(self) -> "FusionConfig":
        if not (
            self.band_low < self.band_medium < self.band_high
        ):
            raise ValueError(
                "band thresholds must be strictly ascending: low < medium < high"
            )
        return self

    @property
    def weight_of(self) -> dict[SignalFamily, decimal.Decimal]:
        return dict(self.weights)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "weights": [
                {"family": family.value, "weight": str(weight)}
                for family, weight in self.weights
            ],
            "band": {
                "low": str(self.band_low),
                "medium": str(self.band_medium),
                "high": str(self.band_high),
            },
            "evidence_count_cap": self.evidence_count_cap,
            "path_count_cap": self.path_count_cap,
            "provenance_file_cap": self.provenance_file_cap,
            "formula": self.formula,
            "availability_rule": self.availability_rule,
            "interpretation": self.interpretation,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


class FusionAssessment(BaseModel):
    """The complete, immutable result of one fusion assessment.

    Everything a reader needs to re-play the number is on the object: the
    signals that entered (or were refused), the per-family contributions, the
    missing signals with reasons, the inference levels carried verbatim, the
    edge-confidence context, the strong overlap note, the deterministic
    explanation items, and the canonical configuration JSON that shaped the
    arithmetic.  The object is self-verifying: its own validator recomputes the
    contributions and the fused score from the recorded signals and the
    recorded configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: str
    subject_profile_id: str
    subject_observation_ids: Tuple[str, ...] = ()
    config_version: str
    config_json: str
    status: FusionStatus
    fused_score: Optional[decimal.Decimal] = Field(default=None)
    risk_band: Optional[RiskBand] = Field(default=None)
    interpretation: str
    signal_count: int = Field(ge=0)
    present_signal_count: int = Field(ge=0)
    signals: Tuple[FusionSignal, ...] = ()
    contributions: Tuple[SignalContribution, ...] = ()
    missing_signals: Tuple[MissingSignal, ...] = ()
    inference_levels: Tuple[InferenceLevel, ...] = ()
    confidence: ConfidenceContext = Field(default_factory=ConfidenceContext)
    distinct_relationship_ids: Tuple[str, ...] = ()
    explanation: Tuple[FusionExplanation, ...] = ()

    @field_validator("assessment_id", "subject_profile_id", "config_version")
    @classmethod
    def _check_text(cls, value: str, info: Any) -> str:
        return _require_non_blank(value, str(info.field_name))

    @field_validator("config_json")
    @classmethod
    def _check_config_json(cls, value: str) -> str:
        _require_non_blank(value, "config_json")
        try:
            json.loads(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("config_json must be valid JSON: %s" % exc) from exc
        return value

    @field_validator("interpretation")
    @classmethod
    def _check_interpretation(cls, value: str) -> str:
        _require_non_blank(value, "interpretation")
        if len(value.strip()) < 10:
            raise ValueError("interpretation must be a real statement")
        return value

    @field_validator("subject_observation_ids")
    @classmethod
    def _check_subject_ids(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return _require_sorted_unique(value, "subject_observation_ids")

    @field_validator("distinct_relationship_ids")
    @classmethod
    def _check_distinct(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return _require_sorted_unique(value, "distinct_relationship_ids")

    @field_validator("inference_levels")
    @classmethod
    def _check_inference_levels(
        cls, value: Tuple[InferenceLevel, ...]
    ) -> Tuple[InferenceLevel, ...]:
        shown = [item.value for item in value]
        if shown != sorted(shown) or len(shown) != len(set(shown)):
            raise ValueError("inference_levels must be ascending and unique")
        return value

    @field_validator("fused_score")
    @classmethod
    def _check_fused(cls, value: Optional[decimal.Decimal]) -> Optional[decimal.Decimal]:
        if value is None:
            return None
        produced = _as_decimal(value, "fused_score")
        if not 0 <= produced <= 1:
            raise ValueError("fused_score must lie inside [0, 1]")
        return produced

    @model_validator(mode="after")
    def _check_consistency(self) -> "FusionAssessment":
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count must equal the number of signals")
        if self.present_signal_count != sum(
            1 for signal in self.signals if signal.status is SignalStatus.PRESENT
        ):
            raise ValueError(
                "present_signal_count must equal the number of PRESENT signals"
            )
        ids = [signal.signal_id for signal in self.signals]
        if ids != sorted(ids):
            raise ValueError("signals must be ascending by signal_id")
        if len(ids) != len(set(ids)):
            raise ValueError("signals must not repeat a signal_id")
        missing_ids = [item.signal_id for item in self.missing_signals]
        if missing_ids != sorted(missing_ids):
            raise ValueError("missing_signals must be ascending by signal_id")
        expected_missing = {
            signal.signal_id
            for signal in self.signals
            if signal.status is not SignalStatus.PRESENT
        }
        if set(missing_ids) != expected_missing:
            raise ValueError("missing_signals must exactly list the non-PRESENT signals")
        if [c.family.value for c in self.contributions] != sorted(
            c.family.value for c in self.contributions
        ):
            raise ValueError("contributions must be ascending by family")

        expected_levels = tuple(
            sorted(
                {
                    level
                    for signal in self.signals
                    if signal.status is SignalStatus.PRESENT
                    for level in signal.inference_levels
                },
                key=lambda item: item.value,
            )
        )
        if self.inference_levels != expected_levels:
            raise ValueError(
                "inference_levels must be the ascending union over PRESENT signals"
            )
        expected_distinct = tuple(
            sorted(
                {
                    relationship_id
                    for signal in self.signals
                    if signal.status is SignalStatus.PRESENT
                    for relationship_id in signal.evidence_refs
                }
            )
        )
        if self.distinct_relationship_ids != expected_distinct:
            raise ValueError(
                "distinct_relationship_ids must be the ascending union of "
                "PRESENT signals' evidence refs"
            )

        payload = json.loads(self.config_json)
        weights = {
            entry["family"]: decimal.Decimal(str(entry["weight"]))
            for entry in payload["weights"]
        }
        thresholds = (
            decimal.Decimal(str(payload["band"]["low"])),
            decimal.Decimal(str(payload["band"]["medium"])),
            decimal.Decimal(str(payload["band"]["high"])),
        )
        if (
            self.config_version != payload["version"]
            or payload["interpretation"] != self.interpretation
        ):
            raise ValueError(
                "config_json must match the recorded version and interpretation"
            )

        with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
            numerator = decimal.Decimal("0")
            denominator = decimal.Decimal("0")
        expected_ids = set()
        for contribution in self.contributions:
            family_signals = [
                signal
                for signal in self.signals
                if signal.family is contribution.family
            ]
            scoring = [
                signal
                for signal in family_signals
                if signal.role is not SignalRole.CONTEXT_ONLY
            ]
            if contribution.total_count != len(scoring):
                raise ValueError(
                    "contribution for %s must count every scoring signal of that family"
                    % contribution.family.value
                )
            present = [
                signal for signal in scoring
                if signal.status is SignalStatus.PRESENT
            ]
            if contribution.present_count != len(present):
                raise ValueError(
                    "contribution for %s miscounts PRESENT scoring signals"
                    % contribution.family.value
                )
            present_ids = tuple(sorted(signal.signal_id for signal in present))
            if contribution.signal_ids != present_ids:
                raise ValueError(
                    "contribution for %s must name exactly its PRESENT scoring signals"
                    % contribution.family.value
                )
            expected_ids.update(present_ids)
            with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
                if present:
                    value = sum(
                        (decimal.Decimal(str(signal.normalized_value)) for signal in present),
                        decimal.Decimal("0"),
                    ) / decimal.Decimal(len(present))
                else:
                    value = decimal.Decimal("0")
                availability = decimal.Decimal(
                    contribution.present_count
                ) / decimal.Decimal(contribution.total_count)
                weight = weights[contribution.family.value]
                check = weight * value * availability
            if contribution.normalized_value != value:
                raise ValueError(
                    "contribution for %s has the wrong family value"
                    % contribution.family.value
                )
            if contribution.availability != availability:
                raise ValueError(
                    "contribution for %s has the wrong availability"
                    % contribution.family.value
                )
            if contribution.weight != weight:
                raise ValueError(
                    "contribution for %s has a weight the configuration does not state"
                    % contribution.family.value
                )
            if contribution.contribution != check:
                raise ValueError(
                    "contribution for %s is not the configured weight times its "
                    "value times its availability" % contribution.family.value
                )
            numerator += contribution.contribution
            denominator += weight * availability

        if self.signals:
            families_seen = {signal.family for signal in self.signals}
            if families_seen != set(SignalFamily):
                raise ValueError(
                    "an assessment must carry at least one signal per family so "
                    "availability is defined"
                )
        actually_declared = [signal.signal_id for signal in self.signals]
        present_scoring_declared = [
            signal.signal_id
            for signal in self.signals
            if signal.status is SignalStatus.PRESENT and signal.role is not SignalRole.CONTEXT_ONLY
        ]
        if expected_ids != set(present_scoring_declared):
            raise ValueError("contributions must cover every PRESENT scoring signal")

        computed = self.status is FusionStatus.COMPUTED
        if denominator <= 0:
            if computed:
                raise ValueError(
                    "a zero weighted-availability denominator cannot produce a "
                    "fused score"
                )
            if self.fused_score is not None or self.risk_band is not None:
                raise ValueError(
                    "INSUFFICIENT_SIGNALS must carry no score and no band"
                )
        else:
            if not computed:
                raise ValueError(
                    "a non-zero weighted-availability denominator must produce "
                    "a fused score"
                )
            if self.fused_score is None or self.risk_band is None:
                raise ValueError("a COMPUTED assessment must carry a score and a band")
            with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
                expected_score = numerator / denominator
            if self.fused_score != expected_score:
                raise ValueError(
                    "fused_score is not the configured weighted ratio: expected "
                    "%s, recorded %s" % (expected_score, self.fused_score)
                )
            expected_band = band_for_score(self.fused_score, thresholds)
            if self.risk_band is not expected_band:
                raise ValueError(
                    "risk_band %s does not match the recorded thresholds"
                    % self.risk_band.value
                )

        expected_id = derive_assessment_id(
            self.subject_profile_id, self.config_json, actually_declared
        )
        if self.assessment_id != expected_id:
            raise ValueError(
                "assessment_id %s does not match the identifier derived from "
                "the subject, configuration and signals (expected %s)"
                % (self.assessment_id, expected_id)
            )
        return self

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "FusionAssessment":
        copied = super().model_copy(update=update, deep=deep)
        return self.model_validate(copied.model_dump())

    def explains_attribution(self) -> bool:
        """Always ``False``.

        An assessment combines recorded evidence into a priority number; it
        never says who owns, controls or is behind anything, and it has no
        field that could hold such a statement.
        """
        return False

    @property
    def priority_signals(self) -> Tuple[FusionSignal, ...]:
        """Signals that directly indicate investigative priority."""
        return tuple(s for s in self.signals if s.role is SignalRole.PRIORITY_SIGNAL)

    @property
    def evidence_strength_signals(self) -> Tuple[FusionSignal, ...]:
        """Signals measuring evidence corroboration and actionability."""
        return tuple(s for s in self.signals if s.role is SignalRole.EVIDENCE_STRENGTH)

    @property
    def context_signals(self) -> Tuple[FusionSignal, ...]:
        """Signals providing descriptive context without altering priority."""
        return tuple(s for s in self.signals if s.role is SignalRole.CONTEXT_ONLY)

    def to_dict(self) -> dict:
        return {
            "assessment_id": self.assessment_id,
            "subject_profile_id": self.subject_profile_id,
            "subject_observation_ids": list(self.subject_observation_ids),
            "config_version": self.config_version,
            "config_json": self.config_json,
            "status": self.status.value,
            "fused_score": None if self.fused_score is None else str(self.fused_score),
            "risk_band": None if self.risk_band is None else self.risk_band.value,
            "interpretation": self.interpretation,
            "signal_count": self.signal_count,
            "present_signal_count": self.present_signal_count,
            "signals": [signal.to_dict() for signal in self.signals],
            "contributions": [item.to_dict() for item in self.contributions],
            "missing_signals": [item.to_dict() for item in self.missing_signals],
            "inference_levels": [item.value for item in self.inference_levels],
            "confidence": self.confidence.to_dict(),
            "distinct_relationship_ids": list(self.distinct_relationship_ids),
            "explanation": [item.to_dict() for item in self.explanation],
        }