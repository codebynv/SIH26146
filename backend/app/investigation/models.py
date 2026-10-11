"""Temporal-flow and investigation-path models (SIH26146 - Step 7).

What this module is
-------------------
Step 7 turns already-established evidence -- the recorded relationships that
Step 4C drew as an :class:`~..graph.models.EvidenceGraph` -- into *investigation
structures*: ordered, evidence-supported trails through observations.  This
module defines the vocabulary of those structures and nothing else.  There is no
search and no traversal here; the algorithms live in :mod:`...paths` and
:mod:`...engine`.

A path is a claim about recorded evidence, and only that
------------------------------------------------------
An :class:`InvestigationPath` answers one question: "which observations are
connected, in which order, through which recorded evidence edges?".  It does
not answer "is this a person, a wallet, an attacker, fraud, or a crime".  This
whole package is named *investigation* because it produces materials an
investigator can read, not because it produces conclusions.

Path representation rules
-------------------------
* Every transition is a copy of one graph edge.  No synthetic edge is invented,
  and no two observations are joined merely because they look similar or sit in
  the same Step 6 cluster.
* Direction is explicit.  A transition records which side of the evidence edge
  was walked from (``FORWARD`` when following ``source -> target``, ``REVERSE``
  otherwise) and the engine only ever walks what a caller asked it to walk.
* Inference levels are never upgraded.  A path through several ``CORRELATED``
  edges is a path through correlated evidence; connectivity is not identity and
  cluster membership is not a link.
* Every transition preserves the evidence that produced it: relationship id,
  edge id, evidence type, rule id and version, inference level, confidence with
  its basis, the rationale, the provenance citations, and the relationship's own
  temporal window and relation.

Path identifiers
----------------
A path id is a UUIDv5 over its ordered observation ids, its ordered relationship
ids and the search scope that produced it.  It contains no clock and no UUID4,
so repeated execution produces identical ids.  Scope is included so two queries
that reached the same node/edge sequence under different filters are not
mistaken for one another.

Anti-attribution
----------------
``extra="forbid"`` on every model means a field for an owner, a person, an actor
or an identity can never be attached by accident, and
:meth:`InvestigationPath.explains_attribution` returns ``False``
unconditionally so the guarantee can be asserted rather than assumed.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    InferenceLevel,
    TemporalRelation,
)
from ..correlation.models import Confidence, TemporalWindow
from ..correlation.provenance import ProvenanceRef
from ..graph.models import derive_node_id
from ..ml.models import UnusualnessBand

__all__ = [
    "EdgeOrientation",
    "EvidenceSummary",
    "IncomparablePolicy",
    "InvestigationError",
    "InvestigationPath",
    "PATH_NAMESPACE",
    "PathNode",
    "PathSearchRequest",
    "PathSearchResult",
    "PathTemporalStatus",
    "PathTransition",
    "ProfileContext",
    "SearchStatus",
    "TemporalFlowStep",
    "TemporalOrder",
    "TemporalTransition",
    "TemporalWindowConfig",
    "TimestampAwareness",
    "derive_path_id",
]


#: Stable UUIDv5 namespace for path identifiers.  Deliberately different from
#: the observation namespace (Step 3), the relationship namespace (Step 4A) and
#: the graph node/edge namespaces (Step 4C): a path id must never be mistaken
#: for any of the ids it is built from.
PATH_NAMESPACE = uuid.UUID("e4a1c7b2-0f3d-5a8e-9b6c-2d4f5a6b7c8d")


class InvestigationError(ValueError):
    """Raised when the engine is asked to do something malformed.

    A :class:`ValueError`, so ordinary ``except ValueError`` still works, but the
    distinct type lets a caller tell a programming error in this layer apart
    from the normal no-result cases, which are returned as typed
    :class:`SearchStatus` values rather than exceptions.
    """


class SearchStatus(str, Enum):
    """The outcome of a search, as a value rather than an exception.

    Normal no-result cases return a status; only a malformed request or a
    malformed input object raises :class:`InvestigationError`.
    """

    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    INVALID_REQUEST = "INVALID_REQUEST"
    NO_TEMPORAL_COMPARISON = "NO_TEMPORAL_COMPARISON"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class TemporalOrder(str, Enum):
    """How one observation's timestamp relates to another's along a path.

    ``BEFORE`` / ``SAME_TIME`` / ``AFTER`` express an ordering between two
    comparable timestamps.  ``NOT_COMPARABLE`` means both sides carry a
    timestamp but they cannot be compared (one aware and one naive).
    ``UNKNOWN`` means at least one side carries no timestamp at all.

    A plain ordering is recorded with no judgement attached: a short gap and a
    long gap are both just durations, and neither is called suspicious here.
    """

    BEFORE = "BEFORE"
    SAME_TIME = "SAME_TIME"
    AFTER = "AFTER"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class PathTemporalStatus(str, Enum):
    """How much temporal structure a path actually carries."""

    COMPARABLE = "COMPARABLE"
    PARTIALLY_COMPARABLE = "PARTIALLY_COMPARABLE"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class EdgeOrientation(str, Enum):
    """Which direction of an evidence edge a transition walked."""

    FORWARD = "FORWARD"
    REVERSE = "REVERSE"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class TimestampAwareness(str, Enum):
    """Whether a carried timestamp is usable for timezone-aware comparison."""

    AWARE = "AWARE"
    NAIVE = "NAIVE"
    ABSENT = "ABSENT"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class IncomparablePolicy(str, Enum):
    """What a temporal-window filter does with a non-comparable transition.

    ``EXCLUDE`` refuses the transition because the gap cannot be verified
    against the requested bounds; ``INCLUDE`` lets it through unverified.  The
    choice is explicit because silently guessing at a gap would be exactly the
    sort of UTC assumption this layer refuses to make.
    """

    EXCLUDE = "EXCLUDE"
    INCLUDE = "INCLUDE"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


def _require_non_blank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty, non-blank string" % field)
    return value


def _canonical_json(payload: Any) -> str:
    """Deterministic JSON for derived identifiers.

    Sorted keys, no insignificant whitespace, no locale formatting, and a
    ``str`` fallback so enums and datetimes serialize without drifting.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True
    )


def derive_path_id(
    observation_ids: Sequence[str],
    relationship_ids: Sequence[str],
    scope: Optional[Any] = None,
) -> str:
    """Derive the deterministic identifier of a path from its contents.

    A UUIDv5 over the *ordered* node sequence, the *ordered* edge sequence and
    the search scope that produced the path.  Order is significant: ``A -> B ->
    C`` and ``C -> B -> A`` are different trails.  ``scope`` is canonicalized
    before hashing, so equal filters always produce equal scope text.

    There is no clock and no UUID4 here, so the same inputs always produce the
    same id on every run and every machine.
    """
    nodes = tuple(observation_ids)
    edges = tuple(relationship_ids)
    if not nodes:
        raise InvestigationError(
            "a path id requires at least one observation; an empty path ids "
            "nothing"
        )
    def _blank_entry(entries: Sequence[str], field: str) -> None:
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                raise InvestigationError(
                    "%s must hold only non-empty, non-blank strings" % field
                )

    _blank_entry(nodes, "observation_ids")
    _blank_entry(edges, "relationship_ids")
    payload = {
        "observation_ids": nodes,
        "relationship_ids": edges,
        "scope": scope,
    }
    return str(uuid.uuid5(PATH_NAMESPACE, _canonical_json(payload)))


class ProfileContext(BaseModel):
    """Step 5/6 metadata carried on a path node as *context only*.

    A node may carry the Step 5 profile it belongs to, the Step 6 cluster that
    profile was placed in, and the Step 6 anomaly band and normalized score
    reported for it.  These are contextual ML outputs attached for the
    investigator's convenience.  They are never used as evidence: profile
    clustering does not create graph edges, cluster membership is not a link
    between observations, and the values are never combined into a path-level
    figure here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(
        description="Step 5 profile identifier whose features produced the "
        "Step 6 outputs below."
    )
    observation_ids: Tuple[str, ...] = Field(
        description="Observations that profile was built over, ascending."
    )
    cluster_id: Optional[str] = Field(
        default=None, description="Step 6 cluster the profile was placed in, if any."
    )
    normalized_score: Optional[float] = Field(
        default=None,
        description="Step 6 normalized anomaly score, carried verbatim as "
        "context. Never combined into a path figure.",
    )
    band: Optional[UnusualnessBand] = Field(
        default=None, description="Step 6 unusualness band, carried verbatim."
    )

    @field_validator("profile_id")
    @classmethod
    def _check_profile_id(cls, value: str) -> str:
        return _require_non_blank(value, "profile_id")

    @field_validator("cluster_id")
    @classmethod
    def _check_cluster_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _require_non_blank(value, "cluster_id")

    @field_validator("observation_ids")
    @classmethod
    def _check_observation_ids(
        cls, value: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        if not value:
            raise ValueError("observation_ids must name at least one observation")
        for observation_id in value:
            _require_non_blank(observation_id, "observation_ids entry")
        if len(set(value)) != len(value):
            raise ValueError("observation_ids entries must be unique")
        if list(value) != sorted(value):
            raise ValueError("observation_ids entries must be in ascending order")
        return value

    @field_validator("normalized_score")
    @classmethod
    def _check_normalized_figure(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("normalized_score must lie inside (0, 1]")
        return value

    @model_validator(mode="after")
    def _check_band_needs_figure(self) -> "ProfileContext":
        if self.band is not None and self.normalized_score is None:
            raise ValueError("band requires the normalized_score that produced it")
        return self

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "observation_ids": list(self.observation_ids),
            "cluster_id": self.cluster_id,
            "normalized_score": self.normalized_score,
            "band": self.band.value if self.band else None,
        }


class TemporalWindowConfig(BaseModel):
    """Configurable temporal constraints for a path search.

    All bounds are :class:`datetime.timedelta` values, so they are
    timezone-agnostic by construction: a duration compares against an elapsed
    time only when that elapsed time was actually computed from two comparable
    (same-awareness) timestamps.  Nothing here hard-codes a window; every
    bound is optional and explicit.

    ``gap_boundary`` is *descriptive*, not investigative.  It merely lets a
    transition report whether its gap exceeds a caller-chosen magnitude; it
    does not mark gaps, however large, as suspicious.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_transition_gap: Optional[timedelta] = Field(
        default=None, description="Smallest allowed elapsed between adjacent nodes."
    )
    max_transition_gap: Optional[timedelta] = Field(
        default=None, description="Largest allowed elapsed between adjacent nodes."
    )
    max_path_duration: Optional[timedelta] = Field(
        default=None, description="Largest allowed span from first to last node."
    )
    gap_boundary: Optional[timedelta] = Field(
        default=None,
        description="Descriptive magnitude threshold reported as a boundary "
        "crossing. Not a judgement.",
    )
    on_incomparable: IncomparablePolicy = IncomparablePolicy.EXCLUDE

    @model_validator(mode="after")
    def _check_bounds(self) -> "TemporalWindowConfig":
        for name in (
            "min_transition_gap",
            "max_transition_gap",
            "max_path_duration",
            "gap_boundary",
        ):
            value = getattr(self, name)
            if value is not None and value < timedelta(0):
                raise ValueError("%s must not be negative" % name)
        if (
            self.min_transition_gap is not None
            and self.max_transition_gap is not None
            and self.min_transition_gap > self.max_transition_gap
        ):
            raise ValueError(
                "min_transition_gap must not exceed max_transition_gap"
            )
        return self

    @property
    def restricts_transitions(self) -> bool:
        """True when at least one gap bound is configured."""
        return (
            self.min_transition_gap is not None
            or self.max_transition_gap is not None
        )

    @property
    def restricts_path_duration(self) -> bool:
        """True when an overall path window is configured."""
        return self.max_path_duration is not None

    def to_dict(self) -> dict:
        return {
            "min_transition_gap_seconds": (
                None
                if self.min_transition_gap is None
                else self.min_transition_gap.total_seconds()
            ),
            "max_transition_gap_seconds": (
                None
                if self.max_transition_gap is None
                else self.max_transition_gap.total_seconds()
            ),
            "max_path_duration_seconds": (
                None
                if self.max_path_duration is None
                else self.max_path_duration.total_seconds()
            ),
            "gap_boundary_seconds": (
                None if self.gap_boundary is None else self.gap_boundary.total_seconds()
            ),
            "on_incomparable": self.on_incomparable.value,
        }


class TemporalTransition(BaseModel):
    """Temporal structure of one adjacent pair in a path.

    ``relation`` describes how the *target* timestamp relates to the *source*
    timestamp: ``AFTER`` when the target is later, ``BEFORE`` when it is
    earlier, ``SAME_TIME`` when the two instants coincide.
    ``elapsed_seconds`` is ``target - source`` in seconds, so it is positive
    exactly when the relation is ``AFTER``.  Both timestamps are carried
    verbatim -- nothing is localised, normalised away, or rewritten.

    ``comparable`` is ``False`` for a mixed aware/naive pair or when either
    side is missing, and then no elapsed time is invented.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_observation_id: str
    target_observation_id: str
    relation: TemporalOrder
    comparable: bool
    source_timestamp: Optional[datetime]
    target_timestamp: Optional[datetime]
    source_awareness: TimestampAwareness
    target_awareness: TimestampAwareness
    elapsed_seconds: Optional[float]
    crosses_boundary: Optional[bool]

    @field_validator("source_observation_id", "target_observation_id")
    @classmethod
    def _check_identifiers(cls, value: str, info: Any) -> str:
        return _require_non_blank(value, str(info.field_name))

    @model_validator(mode="after")
    def _check_consistency(self) -> "TemporalTransition":
        if self.source_observation_id == self.target_observation_id:
            raise ValueError(
                "a temporal transition must join two distinct observations"
            )
        for name, awareness, present in (
            (
                "source",
                self.source_awareness,
                self.source_timestamp is not None,
            ),
            (
                "target",
                self.target_awareness,
                self.target_timestamp is not None,
            ),
        ):
            ts = getattr(self, "%s_timestamp" % name)
            if not present:
                if awareness is not TimestampAwareness.ABSENT:
                    raise ValueError(
                        "%s_awareness must be ABSENT when no timestamp is "
                        "carried" % name
                    )
            elif ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
                if awareness is not TimestampAwareness.NAIVE:
                    raise ValueError(
                        "%s_awareness must be NAIVE for a naive timestamp" % name
                    )
            else:
                if awareness is not TimestampAwareness.AWARE:
                    raise ValueError(
                        "%s_awareness must be AWARE for a timezone-aware "
                        "timestamp" % name
                    )

        if not self.comparable:
            if self.elapsed_seconds is not None:
                raise ValueError(
                    "elapsed_seconds must be None when the pair is not "
                    "comparable"
                )
            if self.crosses_boundary is not None:
                raise ValueError(
                    "crosses_boundary must be None when the pair is not "
                    "comparable"
                )
            if self.relation not in (TemporalOrder.UNKNOWN, TemporalOrder.NOT_COMPARABLE):
                raise ValueError(
                    "a non-comparable pair must report UNKNOWN or "
                    "NOT_COMPARABLE, not %s" % self.relation.value
                )
            if (
                self.source_timestamp is None or self.target_timestamp is None
            ) and self.relation is TemporalOrder.NOT_COMPARABLE:
                raise ValueError(
                    "NOT_COMPARABLE requires a timestamp on both sides; "
                    "otherwise the relation is UNKNOWN"
                )
            if (
                self.source_timestamp is not None
                and self.target_timestamp is not None
                and self.relation is TemporalOrder.NOT_COMPARABLE
                and self.source_awareness is self.target_awareness
            ):
                raise ValueError(
                    "same-awareness timestamps are comparable; the relation "
                    "cannot be NOT_COMPARABLE"
                )
            return self

        if self.source_timestamp is None or self.target_timestamp is None:
            raise ValueError("a comparable transition requires both timestamps")
        if self.source_awareness is not self.target_awareness:
            raise ValueError(
                "a comparable transition requires matching timestamp "
                "awareness"
            )
        delta = self.target_timestamp - self.source_timestamp
        elapsed = delta.total_seconds()
        if self.elapsed_seconds is None or not math.isclose(
            self.elapsed_seconds, elapsed, abs_tol=1e-9
        ):
            raise ValueError(
                "elapsed_seconds must equal target - source in seconds"
            )
        if self.relation is TemporalOrder.AFTER:
            if not (self.elapsed_seconds > 0):
                raise ValueError("AFTER requires a positive elapsed time")
        elif self.relation is TemporalOrder.BEFORE:
            if not (self.elapsed_seconds < 0):
                raise ValueError("BEFORE requires a negative elapsed time")
        elif self.relation is TemporalOrder.SAME_TIME:
            if not math.isclose(self.elapsed_seconds, 0.0, abs_tol=1e-9):
                raise ValueError("SAME_TIME requires zero elapsed time")
        else:
            raise ValueError(
                "a comparable transition must report BEFORE, SAME_TIME or "
                "AFTER"
            )
        return self

    def to_dict(self) -> dict:
        return {
            "source_observation_id": self.source_observation_id,
            "target_observation_id": self.target_observation_id,
            "relation": self.relation.value,
            "comparable": self.comparable,
            "source_timestamp": (
                self.source_timestamp.isoformat()
                if self.source_timestamp is not None
                else None
            ),
            "target_timestamp": (
                self.target_timestamp.isoformat()
                if self.target_timestamp is not None
                else None
            ),
            "source_awareness": self.source_awareness.value,
            "target_awareness": self.target_awareness.value,
            "elapsed_seconds": self.elapsed_seconds,
            "crosses_boundary": self.crosses_boundary,
        }


class PathTransition(BaseModel):
    """One evidence-supported step of a path.

    Every field that describes evidence is copied unchanged from the graph edge
    it preserves: the relationship id, the edge id, the evidence type, the
    inference level, the confidence and its basis, the rule id and version, the
    rationale, the provenance citations, and the relationship's own temporal
    window and relation.  Nothing is recomputed, and nothing is strengthened:
    a ``CORRELATED`` edge stays a ``CORRELATED`` transition whatever the path
    looks like.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_observation_id: str
    to_observation_id: str
    edge_id: str
    relationship_id: str
    orientation: EdgeOrientation
    evidence_type: EvidenceType
    inference_level: InferenceLevel
    confidence: Confidence
    rule_id: str
    rule_version: str
    rationale: str
    provenance_refs: Tuple[ProvenanceRef, ...] = ()
    temporal: Optional[TemporalTransition] = None
    temporal_window: Optional[TemporalWindow] = None
    temporal_relation: Optional[TemporalRelation] = None

    @field_validator(
        "from_observation_id",
        "to_observation_id",
        "edge_id",
        "relationship_id",
        "rule_id",
        "rule_version",
        "rationale",
    )
    @classmethod
    def _check_text(cls, value: str, info: Any) -> str:
        return _require_non_blank(value, str(info.field_name))

    @model_validator(mode="after")
    def _check_consistency(self) -> "PathTransition":
        if self.from_observation_id == self.to_observation_id:
            raise ValueError("a transition must join two distinct observations")
        if self.temporal_relation is not None and self.temporal_window is None:
            raise ValueError(
                "temporal_relation requires temporal_window, matching the "
                "relationship contract"
            )
        if self.temporal is not None:
            if self.temporal.source_observation_id != self.from_observation_id:
                raise ValueError(
                    "the temporal transition must start at from_observation_id"
                )
            if self.temporal.target_observation_id != self.to_observation_id:
                raise ValueError(
                    "the temporal transition must end at to_observation_id"
                )
        return self

    def to_dict(self) -> dict:
        return {
            "from_observation_id": self.from_observation_id,
            "to_observation_id": self.to_observation_id,
            "edge_id": self.edge_id,
            "relationship_id": self.relationship_id,
            "orientation": self.orientation.value,
            "evidence_type": self.evidence_type.value,
            "inference_level": self.inference_level.value,
            "confidence": {
                "value": self.confidence.value,
                "basis": self.confidence.basis.value,
            },
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "rationale": self.rationale,
            "provenance_refs": [
                {
                    "ingestion_id": reference.ingestion_id,
                    "source_file": reference.source_file,
                    "source_record": reference.source_record,
                }
                for reference in self.provenance_refs
            ],
            "temporal": (
                self.temporal.to_dict() if self.temporal is not None else None
            ),
            "temporal_window": (
                None
                if self.temporal_window is None
                else {
                    "start": self.temporal_window.start.isoformat(),
                    "end": self.temporal_window.end.isoformat(),
                }
            ),
            "temporal_relation": (
                None if self.temporal_relation is None else self.temporal_relation.value
            ),
        }


class PathNode(BaseModel):
    """One observation as it appears in a path.

    A node is an *observation*, not a person or a wallet.  It records the
    observation id, the graph node id it corresponds to, the observation's own
    timestamp when one was supplied, and any Step 5/6 profile, cluster and
    anomaly metadata the caller attached as context.  Cluster membership and
    anomaly figures are descriptive context only and are never used as path
    evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str
    graph_node_id: str
    timestamp: Optional[datetime]
    timestamp_awareness: TimestampAwareness
    ml_context: Tuple[ProfileContext, ...] = ()

    @field_validator("observation_id")
    @classmethod
    def _check_observation_id(cls, value: str) -> str:
        return _require_non_blank(value, "observation_id")

    @field_validator("graph_node_id")
    @classmethod
    def _check_graph_node_id(cls, value: str) -> str:
        return _require_non_blank(value, "graph_node_id")

    @model_validator(mode="after")
    def _check_consistency(self) -> "PathNode":
        expected = derive_node_id(self.observation_id)
        if self.graph_node_id != expected:
            raise ValueError(
                "graph_node_id must be the node id the graph layer derives "
                "for this observation"
            )
        ts = self.timestamp
        if ts is None:
            if self.timestamp_awareness is not TimestampAwareness.ABSENT:
                raise ValueError(
                    "timestamp_awareness must be ABSENT when no timestamp is "
                    "carried"
                )
        elif ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            if self.timestamp_awareness is not TimestampAwareness.NAIVE:
                raise ValueError(
                    "timestamp_awareness must be NAIVE for a naive timestamp"
                )
        else:
            if self.timestamp_awareness is not TimestampAwareness.AWARE:
                raise ValueError(
                    "timestamp_awareness must be AWARE for a timezone-aware "
                    "timestamp"
                )
        ids = [context.profile_id for context in self.ml_context]
        if ids != sorted(ids):
            raise ValueError("ml_context entries must be ascending by profile_id")
        if len(ids) != len(set(ids)):
            raise ValueError("ml_context entries must be unique by profile_id")
        return self

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "graph_node_id": self.graph_node_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timestamp_awareness": self.timestamp_awareness.value,
            "ml_context": [context.to_dict() for context in self.ml_context],
        }


class EvidenceSummary(BaseModel):
    """Structural summary of a path's evidence.

    Only counts and bounds over the evidence the path already carries: how many
    edges, which distinct evidence types and inference levels and confidence
    bases appear, and the minimum and maximum *edge* confidence.  The bounds are
    edges' own numbers framed as a range; no new path-level confidence, score or
    risk figure is produced, because combining the edges' figures is exactly
    what Step 8 will do.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_count: int = Field(ge=0)
    evidence_types: Tuple[EvidenceType, ...] = ()
    inference_levels: Tuple[InferenceLevel, ...] = ()
    confidence_bases: Tuple[ConfidenceBasis, ...] = ()
    relationship_ids: Tuple[str, ...] = ()
    min_edge_confidence: Optional[float] = None
    max_edge_confidence: Optional[float] = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "EvidenceSummary":
        if self.edge_count == 0:
            if self.evidence_types or self.inference_levels or self.relationship_ids:
                raise ValueError(
                    "an empty path must carry no evidence-derived fields"
                )
            if (
                self.min_edge_confidence is not None
                or self.max_edge_confidence is not None
            ):
                raise ValueError("an empty path has no edge confidence bounds")
            return self
        if len(self.relationship_ids) != self.edge_count:
            raise ValueError(
                "relationship_ids must name exactly one relationship per edge"
            )
        if len(self.relationship_ids) != len(set(self.relationship_ids)):
            raise ValueError("relationship_ids must be unique along a simple path")
        if (
            self.min_edge_confidence is None
            or self.max_edge_confidence is None
            or not (
                0.0 <= self.min_edge_confidence <= self.max_edge_confidence <= 1.0
            )
        ):
            raise ValueError(
                "min_edge_confidence and max_edge_confidence must both be "
                "present and ordered for a non-empty path"
            )
        shown = [item.value for item in self.evidence_types]
        if shown != sorted(shown):
            raise ValueError("evidence_types must be ascending")
        shown = [item.value for item in self.inference_levels]
        if shown != sorted(shown):
            raise ValueError("inference_levels must be ascending")
        shown = [item.value for item in self.confidence_bases]
        if shown != sorted(shown):
            raise ValueError("confidence_bases must be ascending")
        return self

    def to_dict(self) -> dict:
        return {
            "edge_count": self.edge_count,
            "evidence_types": [item.value for item in self.evidence_types],
            "inference_levels": [item.value for item in self.inference_levels],
            "confidence_bases": [item.value for item in self.confidence_bases],
            "relationship_ids": list(self.relationship_ids),
            "min_edge_confidence": self.min_edge_confidence,
            "max_edge_confidence": self.max_edge_confidence,
        }


class TemporalFlowStep(BaseModel):
    """One machine-readable line of a temporal flow.

    Rendered form::

        observation A  --evidence edge-->  observation B
          elapsed +4 minutes, relation AFTER [comparable]

    Carried as typed data rather than prose so an investigator (or a later
    presentation layer) can render it without re-deriving anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_observation_id: str
    to_observation_id: str
    relationship_id: str
    evidence_type: EvidenceType
    source_timestamp: Optional[datetime]
    target_timestamp: Optional[datetime]
    elapsed_seconds: Optional[float]
    temporal_relation: TemporalOrder
    comparable: bool

    def to_dict(self) -> dict:
        return {
            "from_observation_id": self.from_observation_id,
            "to_observation_id": self.to_observation_id,
            "relationship_id": self.relationship_id,
            "evidence_type": self.evidence_type.value,
            "source_timestamp": (
                self.source_timestamp.isoformat()
                if self.source_timestamp is not None
                else None
            ),
            "target_timestamp": (
                self.target_timestamp.isoformat()
                if self.target_timestamp is not None
                else None
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "temporal_relation": self.temporal_relation.value,
            "comparable": self.comparable,
        }


class InvestigationPath(BaseModel):
    """An ordered, evidence-supported trail through the recorded graph.

    The path is ``node -> edge -> node -> edge -> node``.  Every transition is a
    copy of an existing graph edge, so removing a transition would remove the
    evidence that supports the step and nothing else would change.

    ``first_timestamp`` / ``last_timestamp`` are carried verbatim from the first
    and last nodes.  ``duration_seconds`` is ``last - first`` and is present
    only when those two timestamps are comparable (same awareness); otherwise
    no duration is invented.  ``temporal_status`` summarises how much of the
    path is temporally comparable.

    This object describes evidence.  It does not attribute it to anyone, does
    not call it malicious, and produces no risk figure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path_id: str
    nodes: Tuple[PathNode, ...]
    transitions: Tuple[PathTransition, ...]
    hop_count: int = Field(ge=0)
    first_timestamp: Optional[datetime]
    last_timestamp: Optional[datetime]
    duration_seconds: Optional[float] = None
    temporal_status: PathTemporalStatus
    evidence_summary: EvidenceSummary

    @field_validator("path_id")
    @classmethod
    def _check_path_id(cls, value: str) -> str:
        return _require_non_blank(value, "path_id")

    @model_validator(mode="after")
    def _check_consistency(self) -> "InvestigationPath":
        if not self.nodes:
            raise ValueError("a path must carry at least one node")
        if self.hop_count != len(self.transitions):
            raise ValueError("hop_count must equal the number of transitions")
        if len(self.nodes) != len(self.transitions) + 1:
            raise ValueError(
                "a path with %d transition(s) must carry %d node(s)"
                % (len(self.transitions), len(self.transitions) + 1)
            )
        observation_ids = [node.observation_id for node in self.nodes]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("a simple path must not revisit an observation")
        relationship_ids = [
            transition.relationship_id for transition in self.transitions
        ]
        if len(relationship_ids) != len(set(relationship_ids)):
            raise ValueError("a simple path must not repeat an evidence edge")
        for index, transition in enumerate(self.transitions):
            if transition.from_observation_id != observation_ids[index]:
                raise ValueError(
                    "transition %d must start at the path's node %d"
                    % (index, index)
                )
            if transition.to_observation_id != observation_ids[index + 1]:
                raise ValueError(
                    "transition %d must end at the path's node %d"
                    % (index, index + 1)
                )

        first = self.first_timestamp
        last = self.last_timestamp
        if first != self.nodes[0].timestamp:
            raise ValueError("first_timestamp must be the first node's timestamp")
        if last != self.nodes[-1].timestamp:
            raise ValueError("last_timestamp must be the last node's timestamp")

        if first is not None and last is not None:
            same_kind = (first.tzinfo is None) == (last.tzinfo is None)
            if same_kind and (
                first.tzinfo is None or first.tzinfo.utcoffset(first) is not None
            ):
                expected = (last - first).total_seconds()
                if self.duration_seconds is None or not math.isclose(
                    self.duration_seconds, expected, abs_tol=1e-9
                ):
                    raise ValueError(
                        "duration_seconds must equal last - first for "
                        "comparable timestamps"
                    )
            elif self.duration_seconds is not None:
                raise ValueError(
                    "duration_seconds must be None when first and last are not "
                    "comparable"
                )
        elif self.duration_seconds is not None:
            raise ValueError(
                "duration_seconds must be None when a timestamp is missing"
            )

        structural = self._derive_temporal_status()
        if structural is not self.temporal_status:
            raise ValueError(
                "temporal_status %s does not match the transitions"
                % self.temporal_status.value
            )

        if self.evidence_summary.edge_count != self.hop_count:
            raise ValueError(
                "evidence_summary.edge_count must equal the hop count"
            )
        if self.evidence_summary.relationship_ids != tuple(relationship_ids):
            raise ValueError(
                "evidence_summary must list relationship ids in path order"
            )
        return self

    def _derive_temporal_status(self) -> PathTemporalStatus:
        transitions = self.transitions
        if not transitions:
            return PathTemporalStatus.UNKNOWN
        temporals = [transition.temporal for transition in transitions]
        present = [item is not None for item in temporals]
        if not any(present):
            return PathTemporalStatus.UNKNOWN
        comparable = [
            item is not None and item.comparable for item in temporals
        ]
        if all(comparable):
            return PathTemporalStatus.COMPARABLE
        if not any(comparable):
            return PathTemporalStatus.NOT_COMPARABLE
        return PathTemporalStatus.PARTIALLY_COMPARABLE

    @property
    def observation_ids(self) -> Tuple[str, ...]:
        """Ordered observation ids along the path, start to end."""
        return tuple(node.observation_id for node in self.nodes)

    @property
    def relationship_ids(self) -> Tuple[str, ...]:
        """Ordered relationship ids along the path, start to end."""
        return tuple(transition.relationship_id for transition in self.transitions)

    def temporal_flow(self) -> Tuple[TemporalFlowStep, ...]:
        """The path as a machine-readable temporal-flow sequence."""
        steps = []
        for index, transition in enumerate(self.transitions):
            temporal = transition.temporal
            steps.append(
                TemporalFlowStep(
                    from_observation_id=transition.from_observation_id,
                    to_observation_id=transition.to_observation_id,
                    relationship_id=transition.relationship_id,
                    evidence_type=transition.evidence_type,
                    source_timestamp=(
                        None
                        if temporal is None
                        else temporal.source_timestamp
                    ),
                    target_timestamp=(
                        None if temporal is None else temporal.target_timestamp
                    ),
                    elapsed_seconds=(
                        None if temporal is None else temporal.elapsed_seconds
                    ),
                    temporal_relation=(
                        TemporalOrder.UNKNOWN
                        if temporal is None
                        else temporal.relation
                    ),
                    comparable=(temporal is not None and temporal.comparable),
                )
            )
        return tuple(steps)

    def explains_attribution(self) -> bool:
        """Always ``False``.

        A path is a structure over recorded evidence; it never says who owns,
        controls or is behind anything.  The method exists so the guarantee can
        be asserted rather than inferred from an absence.
        """
        return False

    def to_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "observation_ids": list(self.observation_ids),
            "relationship_ids": list(self.relationship_ids),
            "hop_count": self.hop_count,
            "nodes": [node.to_dict() for node in self.nodes],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "first_timestamp": (
                self.first_timestamp.isoformat()
                if self.first_timestamp is not None
                else None
            ),
            "last_timestamp": (
                self.last_timestamp.isoformat()
                if self.last_timestamp is not None
                else None
            ),
            "duration_seconds": self.duration_seconds,
            "temporal_status": self.temporal_status.value,
            "evidence_summary": self.evidence_summary.to_dict(),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "path(%d nodes, %d edges)" % (
            len(self.nodes),
            len(self.transitions),
        )


class PathSearchRequest(BaseModel):
    """An explicit, bounded request to search recorded evidence.

    Every search must state its bounds.  ``max_depth`` is the largest number of
    evidence edges a path may contain and ``max_paths`` the largest number of
    paths a search may return; there is no default, so an unbounded search
    cannot be requested by accident.

    ``allowed_evidence_types`` and ``allowed_inference_levels``, when non-empty,
    restrict traversal to edges of those kinds.  ``temporal`` carries the
    optional temporal constraints.  ``directed`` states whether traversal
    follows only ``source -> target`` edges or may also walk ``target ->
    source``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_observation_id: str
    target_observation_id: Optional[str] = None
    max_depth: int = Field(ge=1, description="Upper bound on evidence edges.")
    max_paths: int = Field(ge=1, description="Upper bound on returned paths.")
    directed: bool = True
    allowed_evidence_types: Tuple[EvidenceType, ...] = ()
    allowed_inference_levels: Tuple[InferenceLevel, ...] = ()
    temporal: Optional[TemporalWindowConfig] = None

    @field_validator("start_observation_id", "target_observation_id")
    @classmethod
    def _check_identifiers(cls, value: Optional[str], info: Any) -> Optional[str]:
        if value is None:
            return None
        return _require_non_blank(value, str(info.field_name))

    @field_validator("allowed_evidence_types")
    @classmethod
    def _check_evidence_types(
        cls, value: Tuple[EvidenceType, ...]
    ) -> Tuple[EvidenceType, ...]:
        if len(set(value)) != len(value):
            raise ValueError("allowed_evidence_types must not repeat a type")
        return tuple(sorted(value, key=lambda item: item.value))

    @field_validator("allowed_inference_levels")
    @classmethod
    def _check_inference_levels(
        cls, value: Tuple[InferenceLevel, ...]
    ) -> Tuple[InferenceLevel, ...]:
        if len(set(value)) != len(value):
            raise ValueError("allowed_inference_levels must not repeat a level")
        return tuple(sorted(value, key=lambda item: item.value))

    def scope_payload(self, variant: str, config: Optional[TemporalWindowConfig]) -> dict:
        """The search configuration that shapes path identities.

        ``max_depth`` and ``max_paths`` are search *limits* and are excluded on
        purpose: they bound how hard the search looks, not what a path *is*.  A
        path that would have been found at depth 5 is the same trail whether the
        caller looked 5 or 50 edges deep.
        """
        temporal_payload = (
            None if config is None else config.to_dict()
        )
        return {
            "variant": variant,
            "directed": self.directed,
            "allowed_evidence_types": (
                None
                if not self.allowed_evidence_types
                else [item.value for item in self.allowed_evidence_types]
            ),
            "allowed_inference_levels": (
                None
                if not self.allowed_inference_levels
                else [item.value for item in self.allowed_inference_levels]
            ),
            "temporal": temporal_payload,
        }


class PathSearchResult(BaseModel):
    """The outcome of a search, with every non-result case a typed status.

    ``temporal_exclusions`` counts candidate transitions rejected because a
    comparable gap fell outside requested bounds, and
    ``incomparable_exclusions`` counts candidates rejected because their
    timestamps could not be compared at all under an ``EXCLUDE`` policy.  The
    two counters let the exact reason a search came back empty be checked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: SearchStatus
    detail: str
    request: PathSearchRequest
    paths: Tuple[InvestigationPath, ...] = ()
    path_count: int = Field(ge=0)
    shortest_hops: Optional[int] = None
    temporal_exclusions: int = Field(ge=0, default=0)
    incomparable_exclusions: int = Field(ge=0, default=0)
    search_scope: str

    @field_validator("detail")
    @classmethod
    def _check_detail(cls, value: str) -> str:
        return _require_non_blank(value, "detail")

    @field_validator("search_scope")
    @classmethod
    def _check_scope(cls, value: str) -> str:
        return _require_non_blank(value, "search_scope")

    @model_validator(mode="after")
    def _check_consistency(self) -> "PathSearchResult":
        if self.path_count != len(self.paths):
            raise ValueError("path_count must equal the number of paths")
        if self.paths:
            if self.status is not SearchStatus.FOUND:
                raise ValueError("a result carrying paths must be FOUND")
            if self.shortest_hops is None:
                raise ValueError("a FOUND result must report shortest_hops")
        elif self.status is SearchStatus.FOUND:
            raise ValueError("a FOUND result must carry at least one path")
        if self.paths and self.status is SearchStatus.FOUND:
            hops = [path.hop_count for path in self.paths]
            if self.shortest_hops != min(hops):
                raise ValueError(
                    "shortest_hops must be the minimum hop count among paths"
                )
        return self

    def explains_attribution(self) -> bool:
        """Always ``False``.  A search result never attributes anything."""
        return False

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "detail": self.detail,
            "path_count": self.path_count,
            "shortest_hops": self.shortest_hops,
            "temporal_exclusions": self.temporal_exclusions,
            "incomparable_exclusions": self.incomparable_exclusions,
            "search_scope": self.search_scope,
            "paths": [path.to_dict() for path in self.paths],
        }