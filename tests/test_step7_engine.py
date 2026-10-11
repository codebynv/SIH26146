"""Tests for the Step 7 investigation engine (SIH26146 - Step 7).

``backend.app.investigation.engine`` searches the recorded-evidence graph with
bounded BFS (shortest) and bounded DFS (enumeration).  Under test: the typed
precondition statuses, artificial shortest-hop guarantees, cycle-safe
enumeration, evidence and temporal filtering, determinism under shuffling, and
the boundary that Step 6 anomaly/cluster figures are *carried* context, never
combined.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from backend.app.correlation.enums import EvidenceType, InferenceLevel
from backend.app.correlation.models import (
    Confidence,
    EvidenceRelationship,
    derive_relationship_id,
)
from backend.app.correlation.provenance import ProvenanceRef
from backend.app.graph import build_graph
from backend.app.investigation import (
    IncomparablePolicy,
    InvestigationEngine,
    InvestigationError,
    ProfileContext,
    SearchStatus,
    TemporalWindowConfig,
    derive_path_id,
)
from backend.app.investigation.models import (
    EdgeOrientation,
    EvidenceSummary,
    PathSearchRequest,
    TemporalFlowStep,
)
from backend.app.ml.models import UnusualnessBand

UTC = timezone.utc


def _ts(day: int, minute: int) -> datetime:
    return datetime(2024, 5, day, 12, minute, tzinfo=UTC)


def _ref(ingestion_id: str = "ing-1", record: str = "row 2") -> ProvenanceRef:
    return ProvenanceRef(
        ingestion_id=ingestion_id, source_file="evidence.csv", source_record=record
    )


def _relationship(source: str, target: str, **overrides) -> EvidenceRelationship:
    payload = {
        "relationship_id": derive_relationship_id(
            {"engine": "%s->%s" % (source, target)}
        ),
        "source_observation_ids": (source,),
        "target_observation_ids": (target,),
        "evidence_type": EvidenceType.SAME_IP,
        "inference_level": InferenceLevel.OBSERVED,
        "confidence": Confidence.exact_match(),
        "rationale": "Both observations carry src_ip 203.0.113.5 as recorded.",
        "rule_id": "rule.same_ip",
        "rule_version": "1.0.0",
        "provenance_refs": (_ref(),),
        "created_at": _ts(1, 0),
    }
    payload.update(overrides)
    return EvidenceRelationship(**payload)


def _chain() -> list[EvidenceRelationship]:
    return [
        _relationship("a", "b"),
        _relationship("b", "c"),
        _relationship("c", "d"),
    ]


def _diamond() -> list[EvidenceRelationship]:
    return [
        _relationship("a", "b"),
        _relationship("a", "c"),
        _relationship("b", "d"),
        _relationship("c", "d"),
    ]


def _cycle() -> list[EvidenceRelationship]:
    return [
        _relationship("a", "b"),
        _relationship("b", "a"),
        _relationship("b", "c"),
        _relationship("c", "b"),
        _relationship("c", "d"),
    ]


def _request(start: str, **overrides) -> PathSearchRequest:
    payload = {"start_observation_id": start, "max_depth": 5, "max_paths": 20}
    payload.update(overrides)
    return PathSearchRequest(**payload)


def _profile(profile_id: str, observation_ids, **overrides) -> ProfileContext:
    payload = {"profile_id": profile_id, "observation_ids": tuple(observation_ids)}
    payload.update(overrides)
    return ProfileContext(**payload)


def _ts_index_simple() -> dict[str, datetime]:
    minutes = {"a": 0, "b": 4, "c": 9, "d": 15}
    return {key: _ts(1, value) for key, value in minutes.items()}


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------- #
# Engine construction
# --------------------------------------------------------------------------- #


class TestEngineConstruction:
    def test_default_engine_has_a_cap(self):
        engine = InvestigationEngine()
        assert engine.max_depth_cap == 128

    def test_cap_must_be_a_positive_integer(self):
        with pytest.raises(InvestigationError):
            InvestigationEngine(max_depth_cap=0)
        with pytest.raises(InvestigationError):
            InvestigationEngine(max_depth_cap="many")


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #


class TestPreconditions:
    def test_non_graph_input_is_refused(self):
        engine = InvestigationEngine()
        with pytest.raises(InvestigationError):
            engine.shortest_path([], _request("a", target_observation_id="b"))

    def test_non_request_input_is_refused(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        with pytest.raises(InvestigationError):
            engine.enumerate_paths(graph, {"start": "a"})

    def test_depth_above_the_engine_cap_is_refused(self):
        engine = InvestigationEngine(max_depth_cap=4)
        graph = build_graph(_chain())
        with pytest.raises(InvestigationError):
            engine.enumerate_paths(graph, _request("a", max_depth=8))

    def test_empty_graph_is_insufficient_data(self):
        engine = InvestigationEngine()
        graph = build_graph([])
        result = engine.enumerate_paths(graph, _request("a"))
        assert result.status is SearchStatus.INSUFFICIENT_DATA
        assert result.paths == ()

    def test_absent_start_is_not_found(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.enumerate_paths(graph, _request("zz"))
        assert result.status is SearchStatus.NOT_FOUND

    def test_absent_target_is_not_found(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="zz")
        )
        assert result.status is SearchStatus.NOT_FOUND

    def test_self_target_is_invalid_request(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="a")
        )
        assert result.status is SearchStatus.INVALID_REQUEST

    def test_isolated_start_finds_nothing(self):
        engine = InvestigationEngine()
        relationship = _relationship("a", "b")
        graph = build_graph((relationship,))
        assert graph.has_node("a")
        result = engine.enumerate_paths(graph, _request("b"))
        assert result.status is SearchStatus.NOT_FOUND


# --------------------------------------------------------------------------- #
# Shortest-path (BFS)
# --------------------------------------------------------------------------- #


class TestShortestPath:
    def test_direct_edge_is_a_one_hop_path(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="b", max_paths=10)
        )
        assert result.status is SearchStatus.FOUND
        assert result.path_count == 1
        assert result.shortest_hops == 1
        assert result.paths[0].observation_ids == ("a", "b")

    def test_chain_gives_the_real_shortest_hop_count(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        assert result.shortest_hops == 3
        assert result.paths[0].observation_ids == ("a", "b", "c", "d")

    def test_diamond_returns_every_shortest_path(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        assert result.status is SearchStatus.FOUND
        assert result.path_count == 2
        assert result.shortest_hops == 2
        trails = {path.observation_ids for path in result.paths}
        assert trails == {("a", "b", "d"), ("a", "c", "d")}

    def test_direct_edge_beats_the_long_route(self):
        engine = InvestigationEngine()
        relationships = _diamond() + [_relationship("a", "d")]
        graph = build_graph(relationships)
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        assert result.shortest_hops == 1
        assert result.path_count == 1
        assert result.paths[0].observation_ids == ("a", "d")

    def test_max_depth_below_the_shortest_distance_is_not_found(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d", max_depth=2, max_paths=10),
        )
        assert result.status is SearchStatus.NOT_FOUND

    def test_max_paths_caps_the_returned_set(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=1)
        )
        assert result.path_count == 1

    def test_reverse_walking_is_refused_when_directed(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph, _request("b", target_observation_id="a", directed=True)
        )
        assert result.status is SearchStatus.NOT_FOUND

    def test_reverse_walking_is_allowed_when_undirected(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph, _request("b", target_observation_id="a", directed=False)
        )
        assert result.status is SearchStatus.FOUND
        assert result.paths[0].observation_ids == ("b", "a")
        assert result.paths[0].transitions[0].orientation is EdgeOrientation.REVERSE

    def test_both_directions_found_when_undirected(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        result = engine.enumerate_paths(
            graph, _request("d", max_depth=1, directed=False, max_paths=10)
        )
        trails = {path.observation_ids for path in result.paths}
        assert trails == {("d", "b"), ("d", "c")}

    def test_path_ids_are_deterministic_across_runs(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        first = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        second = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        assert [path.path_id for path in first.paths] == [
            path.path_id for path in second.paths
        ]

    def test_shuffled_input_changes_nothing(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        seeded = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        shuffled = build_graph(list(reversed(_diamond())))
        again = engine.shortest_path(
            shuffled, _request("a", target_observation_id="d", max_paths=10)
        )
        assert [(p.observation_ids, p.path_id) for p in seeded.paths] == [
            (p.observation_ids, p.path_id) for p in again.paths
        ]

    def test_evidence_type_filter_prunes_edges(self):
        engine = InvestigationEngine()
        relationships = [
            _relationship("a", "b", evidence_type=EvidenceType.SAME_IP),
            _relationship("b", "c", evidence_type=EvidenceType.SAME_TXID),
            _relationship("c", "d", evidence_type=EvidenceType.SAME_IP),
        ]
        graph = build_graph(relationships)
        filtered = engine.shortest_path(
            graph,
            _request(
                "a",
                target_observation_id="d",
                allowed_evidence_types=(EvidenceType.SAME_IP,),
            ),
        )
        assert filtered.status is SearchStatus.NOT_FOUND
        unfiltered = engine.shortest_path(
            graph, _request("a", target_observation_id="d")
        )
        assert unfiltered.status is SearchStatus.FOUND

    def test_temporal_descriptors_travel_on_transitions(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d"),
            observations=_ts_index_simple(),
        )
        assert result.paths[0].transitions[0].temporal is not None
        assert result.paths[0].transitions[0].temporal.relation.value == "AFTER"

    def test_scope_payload_identifies_the_query_shape(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        payload = json.loads(result.search_scope)
        assert payload["variant"] == "SHORTEST"
        assert json.loads(result.search_scope) is not None


# --------------------------------------------------------------------------- #
# Enumeration (DFS)
# --------------------------------------------------------------------------- #


class TestEnumeration:
    def test_chain_emits_the_full_trail(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.enumerate_paths(graph, _request("a", max_depth=3))
        assert result.status is SearchStatus.FOUND
        assert ("a", "b", "c", "d") in {p.observation_ids for p in result.paths}

    def test_prefixes_are_emitted_up_to_max_depth(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.enumerate_paths(graph, _request("a", max_depth=2))
        trails = {p.observation_ids for p in result.paths}
        assert trails == {("a", "b"), ("a", "b", "c")}

    def test_cycle_heavy_graph_terminates_and_revisits_nothing(self):
        engine = InvestigationEngine()
        graph = build_graph(_cycle())
        result = engine.enumerate_paths(graph, _request("a", max_depth=6, max_paths=200))
        assert result.status is SearchStatus.FOUND
        assert len(result.paths) > 0
        for path in result.paths:
            assert len(path.observation_ids) == len(set(path.observation_ids))

    def test_enumeration_with_a_target_stops_at_the_target(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        result = engine.enumerate_paths(
            graph, _request("a", target_observation_id="d", max_depth=5, max_paths=20)
        )
        for path in result.paths:
            assert path.observation_ids[-1] == "d"
        trails = {p.observation_ids for p in result.paths}
        assert trails == {("a", "b", "d"), ("a", "c", "d")}

    def test_max_paths_caps_enumeration(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.enumerate_paths(graph, _request("a", max_depth=3, max_paths=1))
        assert result.path_count == 1

    def test_results_are_sorted_by_hop_then_content(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        result = engine.enumerate_paths(
            graph, _request("a", target_observation_id="d", max_depth=5, max_paths=20)
        )
        hops = [path.hop_count for path in result.paths]
        assert hops == sorted(hops)

    def test_undirected_enumeration_reaches_the_source_side(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.enumerate_paths(
            graph, _request("b", max_depth=2, directed=False)
        )
        trails = {p.observation_ids for p in result.paths}
        assert ("b", "a") in trails
        assert ("b", "c", "d") in trails

    def test_parallel_edges_produce_distinct_paths(self):
        engine = InvestigationEngine()
        relationships = [
            _relationship("a", "b", relationship_id="rel-p-1"),
            _relationship("a", "b", relationship_id="rel-p-2"),
        ]
        graph = build_graph(relationships)
        result = engine.enumerate_paths(
            graph, _request("a", target_observation_id="b", max_depth=1)
        )
        assert result.path_count == 2
        assert len({p.relationship_ids for p in result.paths}) == 2

    def test_search_scope_distinguishes_the_variants(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        shortest = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        enumerated = engine.enumerate_paths(
            graph, _request("a", target_observation_id="d", max_depth=5, max_paths=10)
        )
        assert shortest.search_scope != enumerated.search_scope


# --------------------------------------------------------------------------- #
# Temporal windows
# --------------------------------------------------------------------------- #


class TestTemporalWindows:
    def test_min_transition_gap_excludes_fast_transitions(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        config = TemporalWindowConfig(min_transition_gap=timedelta(minutes=5))
        result = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d", temporal=config),
            observations=_ts_index_simple(),
        )
        assert result.status is SearchStatus.NOT_FOUND
        assert result.temporal_exclusions > 0

    def test_max_transition_gap_keeps_a_strict_window(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        config = TemporalWindowConfig(max_transition_gap=timedelta(minutes=10))
        result = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d", temporal=config),
            observations=_ts_index_simple(),
        )
        assert result.status is SearchStatus.FOUND
        assert result.temporal_exclusions == 0

    def test_all_incomparable_exclusions_report_the_dedicated_status(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        config = TemporalWindowConfig(max_transition_gap=timedelta(minutes=10))
        naive_index = {
            "a": datetime(2024, 5, 1, 12, 0),
            "b": datetime(2024, 5, 1, 12, 0),
            "c": datetime(2024, 5, 1, 12, 0),
            "d": datetime(2024, 5, 1, 12, 0),
        }
        mixed = dict(naive_index)
        mixed["a"] = _ts(1, 0)
        result = engine.enumerate_paths(
            graph, _request("a", target_observation_id="d", temporal=config, max_paths=20),
            observations=mixed,
        )
        assert result.status is SearchStatus.NO_TEMPORAL_COMPARISON
        assert result.temporal_exclusions == 0
        assert result.incomparable_exclusions > 0

    def test_include_policy_lets_incomparable_pass_through(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        config = TemporalWindowConfig(
            max_transition_gap=timedelta(minutes=10),
            on_incomparable=IncomparablePolicy.INCLUDE,
        )
        mixed = _ts_index_simple()
        mixed["b"] = datetime(2024, 5, 1, 12, 4)  # one naive, one aware pair
        result = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d", temporal=config),
            observations=mixed,
        )
        assert result.status is SearchStatus.FOUND

    def test_max_path_duration_filters_the_overall_window(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        config = TemporalWindowConfig(max_path_duration=timedelta(minutes=10))
        result = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d", temporal=config),
            observations=_ts_index_simple(),
        )
        assert result.status is SearchStatus.NOT_FOUND
        assert result.temporal_exclusions > 0


# --------------------------------------------------------------------------- #
# Step 6 context is carried, never combined
# --------------------------------------------------------------------------- #


class TestStep6Context:
    def test_profile_and_cluster_contexts_are_carried_on_nodes(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        profile_a = _profile(
            "profile-a",
            ("a",),
            cluster_id="cluster-1",
            normalized_score=0.8,
            band=UnusualnessBand.HIGH,
        )
        profile_d = _profile(
            "profile-d",
            ("d",),
            normalized_score=0.55,
            band=UnusualnessBand.LOW,
        )
        result = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d", max_paths=10),
            profile_contexts=(profile_a, profile_d),
        )
        node_a = result.paths[0].nodes[0]
        assert [c.profile_id for c in node_a.ml_context] == ["profile-a"]
        assert node_a.ml_context[0].normalized_score == 0.8
        node_d = result.paths[0].nodes[-1]
        assert [c.profile_id for c in node_d.ml_context] == ["profile-d"]

    def test_context_does_not_change_the_path_id(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        with_context = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d", max_paths=10),
            profile_contexts=(_profile("p", ("a",)),),
        )
        without_context = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        assert [p.path_id for p in with_context.paths] == [
            p.path_id for p in without_context.paths
        ]

    def test_malformed_context_is_refused(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        with pytest.raises(InvestigationError):
            engine.enumerate_paths(
                graph,
                _request("a"),
                profile_contexts=[{"not": "a profile"}],
            )

    def test_the_engine_produces_no_path_level_figure(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        result = engine.shortest_path(
            graph, _request("a", target_observation_id="d", max_paths=10)
        )
        for path in result.paths:
            assert isinstance(path.evidence_summary, EvidenceSummary)
            assert path.evidence_summary.max_edge_confidence <= 1.0
            assert "score" not in path.evidence_summary.model_dump_json().lower()
            assert "risk" not in path.evidence_summary.model_dump_json().lower()


# --------------------------------------------------------------------------- #
# The engine mutates nothing
# --------------------------------------------------------------------------- #


class TestNoMutation:
    def test_the_graph_is_unchanged_by_a_search(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        before_edges = [e.relationship_id for e in graph.edges()]
        before_nodes = [n.observation_id for n in graph.nodes()]
        result = engine.enumerate_paths(graph, _request("a", max_depth=3))
        assert result.paths
        assert [e.relationship_id for e in graph.edges()] == before_edges
        assert [n.observation_id for n in graph.nodes()] == before_nodes

    def test_the_observations_mapping_is_unchanged(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        observations = _ts_index_simple()
        snapshot = dict(observations)
        engine.enumerate_paths(graph, _request("a", max_depth=3), observations=observations)
        assert observations == snapshot

    def test_the_profile_contexts_are_unchanged(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        profile = _profile("profile-a", ("a",), normalized_score=0.8, band=UnusualnessBand.HIGH)
        engine.enumerate_paths(
            graph, _request("a", max_depth=3), profile_contexts=(profile,)
        )
        assert profile.normalized_score == 0.8


# --------------------------------------------------------------------------- #
# Result structure (both variants)
# --------------------------------------------------------------------------- #


class TestResultStructure:
    def test_found_results_carry_only_bounded_paths(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        for result in (
            engine.shortest_path(graph, _request("a", target_observation_id="d")),
            engine.enumerate_paths(graph, _request("a", max_depth=2)),
        ):
            assert result.status is SearchStatus.FOUND
            assert result.path_count == len(result.paths)
            for path in result.paths:
                assert path.hop_count <= 2
                assert path.nodes[0].observation_id == "a"
                assert path.path_id == derive_path_id(
                    path.observation_ids, path.relationship_ids, scope=json.loads(result.search_scope)
                )

    def test_no_result_cases_are_typed_and_quiet(self):
        engine = InvestigationEngine()
        graph = build_graph([])
        result = engine.enumerate_paths(graph, _request("a"))
        assert isinstance(result.status, SearchStatus)
        assert result.paths == ()
        assert result.detail

    def test_shortest_hops_is_the_minimum(self):
        engine = InvestigationEngine()
        graph = build_graph(_diamond())
        result = engine.enumerate_paths(
            graph, _request("a", target_observation_id="d", max_depth=4, max_paths=20)
        )
        assert result.paths
        assert result.shortest_hops == min(p.hop_count for p in result.paths)

    def test_temporal_flow_steps_are_typed(self):
        engine = InvestigationEngine()
        graph = build_graph(_chain())
        result = engine.shortest_path(
            graph,
            _request("a", target_observation_id="d"),
            observations=_ts_index_simple(),
        )
        for path in result.paths:
            for step in path.temporal_flow():
                assert isinstance(step, TemporalFlowStep)
                assert step.from_observation_id != step.to_observation_id


def test_the_engine_surface_exposes_only_the_search_entry_points():
    engine = InvestigationEngine()
    public = {name for name in dir(engine) if not name.startswith("_")}
    assert {"shortest_path", "enumerate_paths", "max_depth_cap"} <= public