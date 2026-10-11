"""Bounded investigation-path search over the recorded evidence graph (SIH26146 - Step 7).

The engine turns a caller-selected starting observation into ordered,
evidence-supported trails through an existing :class:`~..graph.models.EvidenceGraph`.
It is the *search* component of the investigation layer: everything it returns is
a structure over recorded evidence, and it decides nothing beyond the structure.

Bounded search
--------------
Every search must name its bounds (``max_depth`` edges and ``max_paths``
results) and the engine additionally refuses to walk more than an engine-level
maximum depth, so an unbounded search cannot be requested by accident.  No
unrestricted all-pairs exploration is implemented anywhere.

Algorithms
----------
* **BFS** (``shortest_path``) for a shortest-hop path: it finds every
  shortest path from start to target by tracking, for each node, the recorded
  edges that achieve its shortest distance, then reconstructs them in
  deterministic order.
* **Bounded DFS** (``enumerate_paths``) for enumeration: it walks the graph
  depth-first with an explicit stack, never revisiting an observation, and stops
  once ``max_paths`` candidate trails have been produced.  Simple paths only,
  so cycles cannot be walked forever.

Neither algorithm uses confidence, anomaly figures or cluster membership as a
cost or an input beyond the filters the caller asks for, because weighting
evidence is Step 8's job.  Neighbour ordering and result ordering are
deterministic everywhere, so shuffled input cannot change the answer.

What the engine will not do
---------------------------
It does not combine signals into a score, does not rank trails as important,
does not label a trail, and does not attribute it to anyone.  Step 6 anomaly and
cluster values may be attached to path nodes as *context* supplied by the
caller; they are carried, never combined.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import timedelta
from functools import cmp_to_key
from typing import Any, Mapping, Optional, Sequence

from ..graph.models import EvidenceEdge, EvidenceGraph

from .models import (
    EdgeOrientation,
    IncomparablePolicy,
    InvestigationError,
    InvestigationPath,
    PathSearchRequest,
    PathSearchResult,
    ProfileContext,
    SearchStatus,
    TemporalWindowConfig,
)
from .paths import WalkStep, assemble_path, compare_paths, deduplicate_paths
from .temporal import build_temporal_index, build_temporal_transition

__all__ = [
    "InvestigationEngine",
]

_SHORTEST = "SHORTEST"
_ENUMERATE = "ENUMERATE"

_NO_PATH_DETAIL = {
    "start_absent": "start observation %s is not present in the recorded graph",
    "target_absent": "target observation %s is not present in the recorded graph",
    "no_edges": "observation %s has no incident evidence edges in the recorded graph",
    "no_path": "no bounded path connects %s to %s within %d edge(s) under the given filters",
}


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True
    )


class InvestigationEngine:
    """Stateless, deterministic investigation-path search.

    One instance may be shared freely: every call depends only on its
    arguments, and two calls with equal arguments return equal results.
    """

    def __init__(self, max_depth_cap: int = 128) -> None:
        """Create an engine whose searches may not exceed *max_depth_cap* edges."""
        if isinstance(max_depth_cap, bool) or not isinstance(max_depth_cap, int):
            raise InvestigationError("max_depth_cap must be an integer")
        if max_depth_cap < 1:
            raise InvestigationError("max_depth_cap must be at least 1")
        self._max_depth_cap = max_depth_cap

    @property
    def max_depth_cap(self) -> int:
        """The engine-level upper bound on walk depth."""
        return self._max_depth_cap

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    def shortest_path(
        self,
        graph: EvidenceGraph,
        request: PathSearchRequest,
        *,
        observations: Optional[Any] = None,
        profile_contexts: Optional[Sequence[ProfileContext]] = None,
    ) -> PathSearchResult:
        """Return every shortest-hop path from the request's start to its target.

        A target is required for a shortest-path query; without one the query is
        returned as ``INVALID_REQUEST``.  ``observations`` may be a mapping of
        ``observation_id`` to a timestamp, or a sequence of objects carrying
        ``observation_id`` and ``timestamp`` attributes.  ``profile_contexts``
        carries Step 5/6 metadata to attach to nodes as context only.
        """
        return self._search(
            _SHORTEST,
            graph,
            request,
            observations=observations,
            profile_contexts=profile_contexts,
        )

    def enumerate_paths(
        self,
        graph: EvidenceGraph,
        request: PathSearchRequest,
        *,
        observations: Optional[Any] = None,
        profile_contexts: Optional[Sequence[ProfileContext]] = None,
    ) -> PathSearchResult:
        """Return every bounded simple trail from the request's start.

        When a target is given, only trails reaching it are returned; otherwise
        every simple trail of up to ``max_depth`` edges is returned, up to
        ``max_paths`` of them.
        """
        return self._search(
            _ENUMERATE,
            graph,
            request,
            observations=observations,
            profile_contexts=profile_contexts,
        )

    # ------------------------------------------------------------------ #
    # Shared search pipeline
    # ------------------------------------------------------------------ #

    def _search(
        self,
        variant: str,
        graph: Any,
        request: Any,
        *,
        observations: Any,
        profile_contexts: Optional[Sequence[ProfileContext]],
    ) -> PathSearchResult:
        if not isinstance(graph, EvidenceGraph):
            raise InvestigationError(
                "the engine searches an EvidenceGraph, not %s"
                % type(graph).__name__
            )
        if not isinstance(request, PathSearchRequest):
            raise InvestigationError(
                "the engine searches with a PathSearchRequest, not %s"
                % type(request).__name__
            )
        if request.max_depth > self._max_depth_cap:
            raise InvestigationError(
                "max_depth %d exceeds the engine cap of %d; raise the cap "
                "explicitly rather than searching unboundedly"
                % (request.max_depth, self._max_depth_cap)
            )
        if variant == _SHORTEST and request.target_observation_id is None:
            return self._no_result(
                SearchStatus.INVALID_REQUEST,
                "a shortest-path query requires a target observation",
                request,
                variant,
            )
        if graph.node_count() == 0:
            return self._no_result(
                SearchStatus.INSUFFICIENT_DATA,
                "the recorded graph holds no observations to search",
                request,
                variant,
            )

        start = request.start_observation_id
        target = request.target_observation_id
        if not graph.has_node(start):
            return self._no_result(
                SearchStatus.NOT_FOUND,
                _NO_PATH_DETAIL["start_absent"] % start,
                request,
                variant,
            )
        if target == start:
            return self._no_result(
                SearchStatus.INVALID_REQUEST,
                "start observation equals target observation; the recorded "
                "graph forbids self-edges, so no simple path can join a node "
                "to itself",
                request,
                variant,
            )
        if target is not None and not graph.has_node(target):
            return self._no_result(
                SearchStatus.NOT_FOUND,
                _NO_PATH_DETAIL["target_absent"] % target,
                request,
                variant,
            )

        config = request.temporal
        temporal_index = build_temporal_index(observations)
        node_contexts = self._index_node_contexts(profile_contexts)
        boundary = config.gap_boundary if config is not None else None

        if variant == _SHORTEST:
            assert target is not None
            candidate_paths, excl = self._bfs_shortest(
                graph, request, temporal_index, config, boundary, start, target
            )
        else:
            candidate_paths, excl = self._dfs_enumerate(
                graph, request, temporal_index, config, boundary, start, target
            )
        temporal_exclusions, incomparable_exclusions = excl

        scope_payload = request.scope_payload(variant, config)
        search_scope = _canonical_json(scope_payload)

        assembled = [
            assemble_path(
                steps,
                temporal_index=temporal_index,
                node_contexts=node_contexts,
                scope=scope_payload,
            )
            for steps in candidate_paths
        ]
        assembled = deduplicate_paths(assembled)
        assembled = tuple(
            sorted(assembled, key=cmp_to_key(compare_paths))
        )

        filtered, duration_temporal, duration_incomparable = self._apply_duration(
            assembled, config
        )
        temporal_exclusions += duration_temporal
        incomparable_exclusions += duration_incomparable
        paths = filtered[: request.max_paths]

        if paths:
            shortest_hops = min(path.hop_count for path in paths)
            return PathSearchResult(
                status=SearchStatus.FOUND,
                detail="found %d bounded path(s) connecting %s%s"
                % (
                    len(paths),
                    start,
                    "" if target is None else " to %s" % target,
                ),
                request=request,
                paths=paths,
                path_count=len(paths),
                shortest_hops=shortest_hops,
                temporal_exclusions=temporal_exclusions,
                incomparable_exclusions=incomparable_exclusions,
                search_scope=search_scope,
            )

        if config is not None and (
            config.restricts_transitions or config.restricts_path_duration
        ):
            if temporal_exclusions == 0 and incomparable_exclusions > 0:
                return self._no_result(
                    SearchStatus.NO_TEMPORAL_COMPARISON,
                    "temporal constraints were requested but every candidate "
                    "transition's timestamps were incomparable or missing under "
                    "policy %s"
                    % config.on_incomparable.value,
                    request,
                    variant,
                    temporal_exclusions=temporal_exclusions,
                    incomparable_exclusions=incomparable_exclusions,
                )
        return self._no_result(
            SearchStatus.NOT_FOUND,
            _NO_PATH_DETAIL["no_path"]
            % (start, target if target is not None else "any observation", request.max_depth),
            request,
            variant,
            temporal_exclusions=temporal_exclusions,
            incomparable_exclusions=incomparable_exclusions,
        )

    def _no_result(
        self,
        status: SearchStatus,
        detail: str,
        request: PathSearchRequest,
        variant: str,
        *,
        temporal_exclusions: int = 0,
        incomparable_exclusions: int = 0,
    ) -> PathSearchResult:
        scope_payload = request.scope_payload(variant, request.temporal)
        return PathSearchResult(
            status=status,
            detail=detail,
            request=request,
            paths=(),
            path_count=0,
            shortest_hops=None,
            temporal_exclusions=temporal_exclusions,
            incomparable_exclusions=incomparable_exclusions,
            search_scope=_canonical_json(scope_payload),
        )

    # ------------------------------------------------------------------ #
    # Candidates and filters
    # ------------------------------------------------------------------ #

    def _edge_allowed(
        self, edge: EvidenceEdge, request: PathSearchRequest
    ) -> bool:
        if (
            request.allowed_evidence_types
            and edge.evidence_type not in request.allowed_evidence_types
        ):
            return False
        if (
            request.allowed_inference_levels
            and edge.inference_level not in request.allowed_inference_levels
        ):
            return False
        return True

    def _walk(
        self,
        graph: EvidenceGraph,
        node: str,
        request: PathSearchRequest,
        *,
        reverse: bool = False,
    ) -> list[WalkStep]:
        """Walkable steps from *node*, deterministically ordered.

        ``reverse=False`` enumerates steps the engine may take from *node*.
        ``reverse=True`` enumerates steps that may arrive at *node* -- the
        predecessor view used for reachability pruning.  In both cases the
        caller-supplied evidence-type and inference-level filters are applied
        before a step is even offered.
        """
        candidates: list[WalkStep] = []
        if not reverse:
            for edge in graph.outgoing_edges(node):
                if not self._edge_allowed(edge, request):
                    continue
                for to in edge.target_observation_ids:
                    if to != node:
                        candidates.append(
                            WalkStep(
                                from_observation_id=node,
                                to_observation_id=to,
                                edge=edge,
                                orientation=EdgeOrientation.FORWARD,
                            )
                        )
            if not request.directed:
                for edge in graph.incoming_edges(node):
                    if not self._edge_allowed(edge, request):
                        continue
                    for to in edge.source_observation_ids:
                        if to != node:
                            candidates.append(
                                WalkStep(
                                    from_observation_id=node,
                                    to_observation_id=to,
                                    edge=edge,
                                    orientation=EdgeOrientation.REVERSE,
                                )
                            )
        else:
            for edge in graph.incoming_edges(node):
                if not self._edge_allowed(edge, request):
                    continue
                for source in edge.source_observation_ids:
                    if source != node:
                        candidates.append(
                            WalkStep(
                                from_observation_id=source,
                                to_observation_id=node,
                                edge=edge,
                                orientation=EdgeOrientation.FORWARD,
                            )
                        )
            if not request.directed:
                for edge in graph.outgoing_edges(node):
                    if not self._edge_allowed(edge, request):
                        continue
                    for source in edge.target_observation_ids:
                        if source != node:
                            candidates.append(
                                WalkStep(
                                    from_observation_id=source,
                                    to_observation_id=node,
                                    edge=edge,
                                    orientation=EdgeOrientation.REVERSE,
                                )
                            )
        candidates.sort(key=lambda step: (step.to_observation_id, step.edge.relationship_id))
        return candidates

    def _temporal_transition(
        self,
        step: WalkStep,
        temporal_index: Mapping[str, Optional[object]],
        boundary: Optional[timedelta],
    ) -> Optional[object]:
        if not temporal_index:
            return None
        return build_temporal_transition(
            step.from_observation_id,
            step.to_observation_id,
            temporal_index.get(step.from_observation_id),
            temporal_index.get(step.to_observation_id),
            boundary,
        )

    def _transition_allowed(
        self,
        step: WalkStep,
        config: Optional[TemporalWindowConfig],
    ) -> tuple[bool, Optional[str]]:
        """Apply transition-gap constraints to one step.

        Returns ``(allowed, reason)`` where ``reason`` is ``"incomparable"``
        when the pair could not be compared (relevant to the
        ``NO_TEMPORAL_COMPARISON`` status) or ``"bound"`` when a comparable gap
        fell outside the requested bounds.
        """
        if config is None or not config.restricts_transitions:
            return True, None
        temporal = step.temporal
        if temporal is None or not temporal.comparable:
            if config.on_incomparable is IncomparablePolicy.EXCLUDE:
                return False, "incomparable"
            return True, None
        elapsed = timedelta(seconds=temporal.elapsed_seconds if temporal.elapsed_seconds is not None else 0.0)
        if config.min_transition_gap is not None and elapsed < config.min_transition_gap:
            return False, "bound"
        if config.max_transition_gap is not None and elapsed > config.max_transition_gap:
            return False, "bound"
        return True, None

    def _apply_duration(
        self,
        paths: Sequence[InvestigationPath],
        config: Optional[TemporalWindowConfig],
    ) -> tuple[tuple[InvestigationPath, ...], int, int]:
        """Filter assembled paths by the overall path window, if requested."""
        if config is None or not config.restricts_path_duration:
            return tuple(paths), 0, 0
        max_seconds = config.max_path_duration.total_seconds()
        kept: list[InvestigationPath] = []
        temporal_exclusions = 0
        incomparable_exclusions = 0
        for path in paths:
            if path.duration_seconds is None:
                if config.on_incomparable is IncomparablePolicy.INCLUDE:
                    kept.append(path)
                else:
                    incomparable_exclusions += 1
            elif path.duration_seconds > max_seconds:
                temporal_exclusions += 1
            else:
                kept.append(path)
        return tuple(kept), temporal_exclusions, incomparable_exclusions

    def _index_node_contexts(
        self, profile_contexts: Optional[Sequence[ProfileContext]],
    ) -> Mapping[str, tuple[ProfileContext, ...]]:
        """Map observation ids to the profile contexts attached to them.

        Context is never combined, weighted or used as evidence; it is only
        filed next to the observations it describes, ordered by profile id.
        """
        by_observation: dict[str, dict[str, ProfileContext]] = {}
        for context in profile_contexts or ():
            if not isinstance(context, ProfileContext):
                raise InvestigationError(
                    "profile_contexts must be ProfileContext objects, not %s"
                    % type(context).__name__
                )
            for observation_id in context.observation_ids:
                by_observation.setdefault(observation_id, {})[
                    context.profile_id
                ] = context
        return {
            observation_id: tuple(
                sorted(contexts.values(), key=lambda item: item.profile_id)
            )
            for observation_id, contexts in sorted(by_observation.items())
        }

    # ------------------------------------------------------------------ #
    # BFS shortest-hop search
    # ------------------------------------------------------------------ #

    def _bfs_shortest(
        self,
        graph: EvidenceGraph,
        request: PathSearchRequest,
        temporal_index: Mapping[str, Optional[object]],
        config: Optional[TemporalWindowConfig],
        boundary: Optional[timedelta],
        start: str,
        target: str,
    ) -> tuple[list[list[WalkStep]], tuple[int, int]]:
        """Collect every shortest-hop walk from *start* to *target*, capped."""
        dist: dict[str, int] = {start: 0}
        preds: dict[str, list[WalkStep]] = {start: []}
        queue: deque[str] = deque([start])
        temporal_exclusions = 0
        incomparable_exclusions = 0

        while queue:
            node = queue.popleft()
            depth = dist[node]
            if depth >= request.max_depth:
                continue
            for step in self._walk(graph, node, request):
                step = WalkStep(
                    from_observation_id=step.from_observation_id,
                    to_observation_id=step.to_observation_id,
                    edge=step.edge,
                    orientation=step.orientation,
                    temporal=self._temporal_transition(step, temporal_index, boundary),
                )
                allowed, reason = self._transition_allowed(step, config)
                if not allowed:
                    if reason == "incomparable":
                        incomparable_exclusions += 1
                    else:
                        temporal_exclusions += 1
                    continue
                neighbour = step.to_observation_id
                next_depth = depth + 1
                if neighbour not in dist:
                    dist[neighbour] = next_depth
                    preds[neighbour] = [step]
                    queue.append(neighbour)
                elif dist[neighbour] == next_depth:
                    preds[neighbour].append(step)

        if target not in dist:
            return [], (temporal_exclusions, incomparable_exclusions)

        walks: list[list[WalkStep]] = []

        def collect(
            node: str, nodes_rev: tuple[str, ...], steps_rev: tuple[WalkStep, ...], remaining: int
        ) -> None:
            if len(walks) >= request.max_paths:
                return
            if remaining == 0:
                walks.append(list(reversed(steps_rev)))
                return
            for step in preds.get(node, ()):
                collect(
                    step.from_observation_id,
                    nodes_rev + (node,),
                    steps_rev + (step,),
                    remaining - 1,
                )

        collect(target, (target,), (), dist[target])
        return walks, (temporal_exclusions, incomparable_exclusions)

    # ------------------------------------------------------------------ #
    # Bounded DFS enumeration
    # ------------------------------------------------------------------ #

    def _dfs_enumerate(
        self,
        graph: EvidenceGraph,
        request: PathSearchRequest,
        temporal_index: Mapping[str, Optional[object]],
        config: Optional[TemporalWindowConfig],
        boundary: Optional[timedelta],
        start: str,
        target: Optional[str],
    ) -> tuple[list[list[WalkStep]], tuple[int, int]]:
        """Enumerate simple walks by bounded depth-first search.

        Simple paths only (an observation is never revisited), an explicit
        stack keeps the walk free of recursion limits, neighbours come out in a
        deterministic order, and the whole walk stops once ``max_paths``
        candidate trails have been produced.
        """
        temporal_exclusions = 0
        incomparable_exclusions = 0

        if target is not None:
            target_dist = self._reverse_distances(
                graph, request, target, request.max_depth
            )
            if target not in target_dist:
                return [], (temporal_exclusions, incomparable_exclusions)
            if target_dist[start] is None or target_dist[start] > request.max_depth:
                return [], (temporal_exclusions, incomparable_exclusions)

        walks: list[list[WalkStep]] = []
        stack: list[tuple[str, int, tuple[WalkStep, ...], frozenset]] = [
            (start, 0, (), frozenset({start}))
        ]
        while stack and len(walks) < request.max_paths:
            node, depth, steps, visited = stack.pop()

            if depth >= 1:
                if target is None or node == target:
                    walks.append(list(steps))
                    if target is not None:
                        continue
                if len(walks) >= request.max_paths:
                    break
            if depth >= request.max_depth:
                continue

            candidates: list[tuple[str, WalkStep]] = []
            for step in self._walk(graph, node, request):
                if step.to_observation_id in visited:
                    continue
                enriched = WalkStep(
                    from_observation_id=step.from_observation_id,
                    to_observation_id=step.to_observation_id,
                    edge=step.edge,
                    orientation=step.orientation,
                    temporal=self._temporal_transition(step, temporal_index, boundary),
                )
                allowed, reason = self._transition_allowed(enriched, config)
                if not allowed:
                    if reason == "incomparable":
                        incomparable_exclusions += 1
                    else:
                        temporal_exclusions += 1
                    continue
                candidates.append((step.to_observation_id, enriched))
            candidates.sort(key=lambda item: (item[0], item[1].edge.relationship_id))

            if target is not None:
                remaining_budget = request.max_depth - (depth + 1)
                pruned = [
                    (to, enriched)
                    for to, enriched in candidates
                    if target_dist.get(to) is not None
                    and to != target
                    and target_dist[to] <= remaining_budget
                ]
                near_target = [
                    (to, enriched)
                    for to, enriched in candidates
                    if to == target
                ]
                candidates = pruned + near_target

            for to, enriched in reversed(candidates):
                stack.append(
                    (to, depth + 1, steps + (enriched,), visited | frozenset({to}))
                )

        return walks, (temporal_exclusions, incomparable_exclusions)

    def _reverse_distances(
        self,
        graph: EvidenceGraph,
        request: PathSearchRequest,
        source: str,
        max_depth: int,
    ) -> dict[str, Optional[int]]:
        """Shortest step distances *into* *source*, for reachability pruning."""
        dist: dict[str, Optional[int]] = {source: 0}
        queue: deque[str] = deque([source])
        seen: set[str] = {source}
        while queue:
            node = queue.popleft()
            depth = dist[node]
            if depth is None or depth >= max_depth:
                continue
            for step in self._walk(graph, node, request, reverse=True):
                predecessor = step.from_observation_id
                if predecessor in seen:
                    continue
                seen.add(predecessor)
                dist[predecessor] = depth + 1
                queue.append(predecessor)
        return {node: dist.get(node) for node in dist}

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "InvestigationEngine(max_depth_cap=%d)" % self._max_depth_cap