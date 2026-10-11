"""Path representation and ordering (SIH26146 - Step 7).

This module turns the engine's raw walk results into validated
:class:`~..models.InvestigationPath` objects and defines the deterministic
ordering the whole package uses to compare paths.

What a path is
--------------
A walk is an alternating sequence ``node -> edge -> node -> ...``.  Every step
names the observation walked from, the observation walked to, the graph edge
that was traversed, the direction in which it was traversed and the temporal
transition between the two observations.  :func:`assemble_path` copies each
edge's evidence into a :class:`~..models.PathTransition` verbatim and computes
the structural path summary.  Nothing is inferred: a path is only ever as
strong as the edges it preserves.

Ordering
--------
The spec ordering for path candidates is:

1. timestamp where comparable
2. observation id
3. relationship id

:func:`compare_paths` implements exactly that.  Between two aware instants it
compares the UTC-normalised instants; between two naive instants it compares
them directly; anywhere else it falls through to the observation id.  Equal
hop counts and equal prefixes compare node by node, so a shuffled graph or a
shuffled input order can never change the ordering of results.

Deduplication
-------------
:func:`deduplicate_paths` collapses two paths only when their ordered node and
edge sequences are identical.  Two parallel evidence edges between the same
pair of observations yield two distinct paths, because their relationship ids
differ; collapsing them would discard evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from ..graph.models import EvidenceEdge, derive_node_id

from .models import (
    EvidenceSummary,
    EdgeOrientation,
    InvestigationError,
    InvestigationPath,
    PathNode,
    PathTransition,
    PathTemporalStatus,
    ProfileContext,
    TemporalTransition,
    derive_path_id,
)
from .temporal import is_aware, timestamp_awareness

__all__ = [
    "WalkStep",
    "assemble_path",
    "compare_paths",
    "deduplicate_paths",
]


@dataclass(frozen=True)
class WalkStep:
    """One traversed edge, with the nodes it joins and its temporal descriptor."""

    from_observation_id: str
    to_observation_id: str
    edge: EvidenceEdge
    orientation: EdgeOrientation
    temporal: Optional[TemporalTransition] = None


def _comparable(first: Optional[datetime], last: Optional[datetime]) -> bool:
    """True when two timestamps can be compared under the temporal rules."""
    if first is None or last is None:
        return False
    return is_aware(first) == is_aware(last)


def _duration_seconds(
    first: Optional[datetime], last: Optional[datetime]
) -> Optional[float]:
    if not _comparable(first, last):
        return None
    assert first is not None and last is not None
    return (last - first).total_seconds()


def _temporal_status(transitions: Sequence[PathTransition]) -> PathTemporalStatus:
    if not transitions:
        return PathTemporalStatus.UNKNOWN
    if not any(transition.temporal is not None for transition in transitions):
        return PathTemporalStatus.UNKNOWN
    comparable = [
        transition.temporal is not None and transition.temporal.comparable
        for transition in transitions
    ]
    if all(comparable):
        return PathTemporalStatus.COMPARABLE
    if not any(comparable):
        return PathTemporalStatus.NOT_COMPARABLE
    return PathTemporalStatus.PARTIALLY_COMPARABLE


def assemble_path(
    steps: Sequence[WalkStep],
    *,
    temporal_index: Mapping[str, Optional[datetime]],
    node_contexts: Mapping[str, Sequence[ProfileContext]] = {},
    scope: Any = None,
) -> InvestigationPath:
    """Build a validated :class:`InvestigationPath` from walked steps.

    ``steps`` must form one continuous trail: the first step must begin
    somewhere, and every later step must begin where its predecessor ended.
    Continuity is checked here so that a caller assembling a path by hand cannot
    produce one that jumps.  Edge evidence is copied verbatim and the summary,
    timestamps, duration and temporal status are derived structurally.
    """
    if not steps:
        raise InvestigationError("a path requires at least one evidence step")
    steps = list(steps)

    observation_sequence: list[str] = [steps[0].from_observation_id]
    for previous, current in zip(steps, steps[1:]):
        if current.from_observation_id != previous.to_observation_id:
            raise InvestigationError(
                "discontinuous step: %s follows %s but must follow %s"
                % (current.from_observation_id, previous.to_observation_id, previous.to_observation_id)
            )
    observation_sequence.extend(step.to_observation_id for step in steps)
    if len(observation_sequence) != len(set(observation_sequence)):
        raise InvestigationError(
            "a simple path must not revisit an observation; refusing a cycle"
        )

    nodes = tuple(
        PathNode(
            observation_id=observation_id,
            graph_node_id=derive_node_id(observation_id),
            timestamp=temporal_index.get(observation_id),
            timestamp_awareness=timestamp_awareness(
                temporal_index.get(observation_id)
            ),
            ml_context=tuple(
                sorted(node_contexts.get(observation_id, ()), key=lambda c: c.profile_id)
            ),
        )
        for observation_id in observation_sequence
    )

    transitions = tuple(
        PathTransition(
            from_observation_id=step.from_observation_id,
            to_observation_id=step.to_observation_id,
            edge_id=step.edge.edge_id,
            relationship_id=step.edge.relationship_id,
            orientation=step.orientation,
            evidence_type=step.edge.evidence_type,
            inference_level=step.edge.inference_level,
            confidence=step.edge.confidence,
            rule_id=step.edge.rule_id,
            rule_version=step.edge.rule_version,
            rationale=step.edge.rationale,
            provenance_refs=step.edge.provenance_refs,
            temporal=step.temporal,
            temporal_window=step.edge.temporal_window,
            temporal_relation=step.edge.temporal_relation,
        )
        for step in steps
    )

    edge_values = [transition.confidence.value for transition in transitions]
    min_edge_confidence = min(edge_values)
    max_edge_confidence = max(edge_values)

    relationship_ids = tuple(
        transition.relationship_id for transition in transitions
    )
    evidence_summary = EvidenceSummary(
        edge_count=len(transitions),
        relationship_ids=relationship_ids,
        evidence_types=tuple(
            sorted({transition.evidence_type for transition in transitions}, key=lambda item: item.value)
        ),
        inference_levels=tuple(
            sorted({transition.inference_level for transition in transitions}, key=lambda item: item.value)
        ),
        confidence_bases=tuple(
            sorted(
                {transition.confidence.basis for transition in transitions},
                key=lambda item: item.value,
            )
        ),
        min_edge_confidence=min_edge_confidence,
        max_edge_confidence=max_edge_confidence,
    )

    first_timestamp = nodes[0].timestamp
    last_timestamp = nodes[-1].timestamp
    duration_seconds = _duration_seconds(first_timestamp, last_timestamp)

    path_id = derive_path_id(
        tuple(node.observation_id for node in nodes),
        relationship_ids,
        scope=scope,
    )

    return InvestigationPath(
        path_id=path_id,
        nodes=nodes,
        transitions=transitions,
        hop_count=len(transitions),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        duration_seconds=duration_seconds,
        temporal_status=_temporal_status(transitions),
        evidence_summary=evidence_summary,
    )


def _compare_instants(
    left: Optional[datetime], right: Optional[datetime]
) -> Optional[int]:
    """Order two timestamps, or report they cannot be compared.

    Returns ``-1``/``0``/``1`` when the pair is comparable (both aware or both
    naive), else ``None``.
    """
    if left is None or right is None:
        return None
    if is_aware(left) != is_aware(right):
        return None
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def compare_paths(left: InvestigationPath, right: InvestigationPath) -> int:
    """Compare two paths for deterministic ordering.

    Shortest hop count first; then, position by position, the timestamp when the
    two timestamps are comparable, then the observation id, then the edge's
    relationship id.

    The comparator is total and independent of input order, so enumeration
    results sort identically however the graph or the request was shuffled.
    """
    if left.hop_count != right.hop_count:
        return -1 if left.hop_count < right.hop_count else 1
    left_nodes = left.nodes
    right_nodes = right.nodes
    for index in range(min(len(left_nodes), len(right_nodes))):
        lnode = left_nodes[index]
        rnode = right_nodes[index]
        instant_order = _compare_instants(lnode.timestamp, rnode.timestamp)
        if instant_order is not None and instant_order != 0:
            return instant_order
        if lnode.observation_id != rnode.observation_id:
            return -1 if lnode.observation_id < rnode.observation_id else 1
        if index < len(left.transitions) and index < len(right.transitions):
            left_relationship = left.transitions[index].relationship_id
            right_relationship = right.transitions[index].relationship_id
            if left_relationship != right_relationship:
                return (
                    -1
                    if left_relationship < right_relationship
                    else 1
                )
    if len(left_nodes) != len(right_nodes):
        return -1 if len(left_nodes) < len(right_nodes) else 1
    return 0


def deduplicate_paths(
    paths: Sequence[InvestigationPath],
) -> tuple[InvestigationPath, ...]:
    """Collapse duplicates, keeping the first occurrence of each trail.

    Two paths are duplicates only when their ordered node sequence and ordered
    relationship sequence are identical.  Distinct evidence edges keep a pair
    of observations apart as two paths no matter how similar they look.
    """
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    unique: list[InvestigationPath] = []
    for path in paths:
        key = (path.observation_ids, path.relationship_ids)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)