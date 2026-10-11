"""Tests for the Step 7 path representation (SIH26146 - Step 7).

``backend.app.investigation.paths`` turns walked steps into validated paths,
orders them, and collapses duplicates.  The guarantees under test: continuity
is enforced, evidence travels verbatim, ordering is a total order over content
whatever the input order, and deduplication collapses only identical trails.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from backend.app.correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    InferenceLevel,
)
from backend.app.correlation.models import (
    Confidence,
    EvidenceRelationship,
    derive_relationship_id,
)
from backend.app.correlation.provenance import ProvenanceRef
from backend.app.graph import EvidenceEdge, build_graph
from backend.app.investigation import paths as paths_module
from backend.app.investigation.models import (
    EdgeOrientation,
    InvestigationError,
    InvestigationPath,
    PathTemporalStatus,
    ProfileContext,
    TemporalTransition,
    TimestampAwareness,
)
from backend.app.investigation.paths import (
    WalkStep,
    assemble_path,
    compare_paths,
    deduplicate_paths,
)
from backend.app.investigation.temporal import build_temporal_transition

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2024, 5, 1, 12, 4, tzinfo=UTC)
T2 = datetime(2024, 5, 1, 12, 9, tzinfo=UTC)

OBS_A = "obs-a"
OBS_B = "obs-b"
OBS_C = "obs-c"


def _ref(ingestion_id: str = "ing-1", record: str = "row 2") -> ProvenanceRef:
    return ProvenanceRef(
        ingestion_id=ingestion_id, source_file="evidence.csv", source_record=record
    )


def _relationship(**overrides) -> EvidenceRelationship:
    payload = {
        "relationship_id": derive_relationship_id({"paths-test": 1}),
        "source_observation_ids": (OBS_A,),
        "target_observation_ids": (OBS_B,),
        "evidence_type": EvidenceType.SAME_IP,
        "inference_level": InferenceLevel.OBSERVED,
        "confidence": Confidence.exact_match(),
        "rationale": "Both observations carry src_ip 203.0.113.5 as recorded.",
        "rule_id": "rule.same_ip",
        "rule_version": "1.0.0",
        "provenance_refs": (_ref(),),
        "created_at": T0,
    }
    payload.update(overrides)
    return EvidenceRelationship(**payload)


def _step(
    from_observation_id: str = OBS_A,
    to_observation_id: str = OBS_B,
    **overrides,
) -> WalkStep:
    edge = EvidenceEdge.from_relationship(_relationship(**overrides))
    return WalkStep(
        from_observation_id=from_observation_id,
        to_observation_id=to_observation_id,
        edge=edge,
        orientation=EdgeOrientation.FORWARD,
    )


def _edge_between(source: str, target: str, **overrides) -> EvidenceEdge:
    relationship = _relationship(
        relationship_id=derive_relationship_id(
            {"path-edge": "%s->%s" % (source, target)}
        ),
        source_observation_ids=(source,),
        target_observation_ids=(target,),
    )
    return EvidenceEdge.from_relationship(relationship)


def _index(timestamps: dict[str, datetime]) -> dict[str, Any]:
    return dict(timestamps)


# --------------------------------------------------------------------------- #
# The public surface
# --------------------------------------------------------------------------- #


class TestPublicSurface:
    def test_the_module_exports_a_sorted_complete_list(self):
        assert list(paths_module.__all__) == sorted(paths_module.__all__)
        assert set(paths_module.__all__) <= set(dir(paths_module))
        for name in paths_module.__all__:
            assert getattr(paths_module, name) is not None, name

    def test_a_step_is_frozen(self):
        step = _step()
        with pytest.raises(Exception):
            step.from_observation_id = OBS_B


# --------------------------------------------------------------------------- #
# assemble_path
# --------------------------------------------------------------------------- #


class TestAssemblePath:
    def test_empty_steps_are_refused(self):
        with pytest.raises(InvestigationError):
            assemble_path([], temporal_index=_index({}))

    def test_discontinuous_steps_are_refused(self):
        step_a = _step(OBS_A, OBS_B)
        step_b = _step(OBS_A, OBS_C)  # must start at B, not A
        with pytest.raises(InvestigationError):
            assemble_path([step_a, step_b], temporal_index=_index({}))

    def test_a_cycle_is_refused(self):
        step_a = _step(OBS_A, OBS_B)
        step_b = _step(OBS_B, OBS_A)
        with pytest.raises(InvestigationError):
            assemble_path([step_a, step_b], temporal_index=_index({}))

    def test_single_step_assembles_a_valid_path(self):
        step = _step()
        path = assemble_path([step], temporal_index=_index({}))
        assert isinstance(path, InvestigationPath)
        assert path.observation_ids == (OBS_A, OBS_B)
        assert path.hop_count == 1
        assert path.temporal_status is PathTemporalStatus.UNKNOWN
        assert path.duration_seconds is None

    def test_evidence_is_copied_verbatim(self):
        step = _step(
            evidence_type=EvidenceType.SAME_TXID,
            inference_level=InferenceLevel.CORRELATED,
            confidence=Confidence.from_rule(0.7),
            rule_id="rule.same_txid",
            rule_version="2.0.0",
            rationale="A rationale that must survive verbatim.",
        )
        path = assemble_path([step], temporal_index=_index({}))
        transition = path.transitions[0]
        assert transition.evidence_type is EvidenceType.SAME_TXID
        assert transition.inference_level is InferenceLevel.CORRELATED
        assert transition.confidence.value == 0.7
        assert transition.confidence.basis is ConfidenceBasis.DETERMINISTIC_RULE
        assert transition.rule_id == "rule.same_txid"
        assert transition.rule_version == "2.0.0"
        assert transition.rationale == "A rationale that must survive verbatim."
        assert transition.provenance_refs == step.edge.provenance_refs

    def test_nodes_carry_timestamps_from_the_index(self):
        step = _step()
        path = assemble_path(
            [step],
            temporal_index=_index({OBS_A: T0, OBS_B: T1}),
        )
        assert path.first_timestamp == T0
        assert path.last_timestamp == T1
        assert path.duration_seconds == 240.0
        assert path.nodes[0].timestamp_awareness is TimestampAwareness.AWARE

    def test_temporal_transitions_are_built_and_carried(self):
        step = _step()
        enriched = WalkStep(
            from_observation_id=OBS_A,
            to_observation_id=OBS_B,
            edge=step.edge,
            orientation=EdgeOrientation.FORWARD,
            temporal=build_temporal_transition(OBS_A, OBS_B, T0, T1),
        )
        path = assemble_path(
            [enriched],
            temporal_index=_index({OBS_A: T0, OBS_B: T1}),
        )
        assert path.temporal_status is PathTemporalStatus.COMPARABLE
        assert path.transitions[0].temporal.elapsed_seconds == 240.0

    def test_ml_context_is_carried_sorted(self):
        context_a = ProfileContext(
            profile_id="profile-2", observation_ids=(OBS_A,)
        )
        context_b = ProfileContext(
            profile_id="profile-1", observation_ids=(OBS_A,)
        )
        step = _step()
        path = assemble_path(
            [step],
            temporal_index=_index({}),
            node_contexts={OBS_A: (context_a, context_b)},
        )
        assert [item.profile_id for item in path.nodes[0].ml_context] == [
            "profile-1",
            "profile-2",
        ]

    def test_ml_context_is_never_used_for_the_path_id(self):
        context = ProfileContext(
            profile_id="profile-1", observation_ids=(OBS_A,)
        )
        step = _step()
        with_context = assemble_path(
            [step],
            temporal_index=_index({}),
            node_contexts={OBS_A: (context,)},
        )
        without_context = assemble_path(
            [step], temporal_index=_index({})
        )
        assert with_context.path_id == without_context.path_id

    def test_two_step_path_summary(self):
        edge_a = _edge_between(OBS_A, OBS_B)
        edge_b = _edge_between(OBS_B, OBS_C)
        steps = [
            WalkStep(
                from_observation_id=OBS_A,
                to_observation_id=OBS_B,
                edge=edge_a,
                orientation=EdgeOrientation.FORWARD,
            ),
            WalkStep(
                from_observation_id=OBS_B,
                to_observation_id=OBS_C,
                edge=edge_b,
                orientation=EdgeOrientation.FORWARD,
            ),
        ]
        path = assemble_path(steps, temporal_index=_index({}))
        assert path.hop_count == 2
        assert path.observation_ids == (OBS_A, OBS_B, OBS_C)
        assert path.evidence_summary.edge_count == 2
        assert path.evidence_summary.min_edge_confidence <= 1.0
        assert path.evidence_summary.max_edge_confidence == 1.0

    def test_scope_shapes_the_path_id(self):
        step = _step()
        one = assemble_path([step], temporal_index=_index({}), scope={"variant": "X"})
        two = assemble_path([step], temporal_index=_index({}), scope={"variant": "Y"})
        assert one.path_id != two.path_id


# --------------------------------------------------------------------------- #
# compare_paths
# --------------------------------------------------------------------------- #


class TestComparePaths:
    def test_shorter_paths_come_first(self):
        short = assemble_path([_step()], temporal_index=_index({}))
        long = assemble_path(
            [
                _step(OBS_A, OBS_B),
                _step(OBS_B, OBS_C, relationship_id=derive_relationship_id({"p": 2})),
            ],
            temporal_index=_index({}),
        )
        assert compare_paths(short, long) < 0
        assert compare_paths(long, short) > 0

    def test_equal_paths_compare_zero(self):
        path = assemble_path([_step()], temporal_index=_index({}))
        assert compare_paths(path, path) == 0

    def test_timestamp_breaks_the_tie_where_comparable(self):
        early = assemble_path([_step()], temporal_index=_index({OBS_A: T0, OBS_B: T1}))
        late = assemble_path([_step()], temporal_index=_index({OBS_A: T1, OBS_B: T2}))
        assert compare_paths(early, late) < 0
        assert compare_paths(late, early) > 0

    def test_incomparable_timestamps_fall_through_to_observation_id(self):
        aware = assemble_path(
            [_step(relationship_id="rel-aaaa")],
            temporal_index=_index({OBS_A: T0, OBS_B: T1}),
        )
        naive = assemble_path(
            [_step(relationship_id="rel-zzzz")],
            temporal_index=_index(
                {
                    OBS_A: datetime(2024, 5, 1, 12, 0),
                    OBS_B: datetime(2024, 5, 1, 12, 4),
                }
            ),
        )
        assert compare_paths(aware, naive) < 0  # identical ids, so rel id decides

    def test_relationship_id_breaks_the_last_tie(self):
        first = assemble_path(
            [
                _step(
                    relationship_id=derive_relationship_id({"tie": "a"})
                )
            ],
            temporal_index=_index({}),
        )
        second = assemble_path(
            [
                _step(
                    relationship_id=derive_relationship_id({"tie": "b"})
                )
            ],
            temporal_index=_index({}),
        )
        assert (compare_paths(first, second) < 0) == (
            first.transitions[0].relationship_id
            < second.transitions[0].relationship_id
        )

    def test_missing_timestamps_compare_by_observation_id(self):
        with_ts = assemble_path(
            [_step()], temporal_index=_index({OBS_A: T0, OBS_B: T1})
        )
        without_ts = assemble_path([_step()], temporal_index=_index({}))
        assert compare_paths(with_ts, without_ts) == 0

    def test_ordering_is_independent_of_input_order(self):
        step = _step()
        path = assemble_path([step], temporal_index=_index({}))
        assert compare_paths(path, path) == 0


# --------------------------------------------------------------------------- #
# deduplicate_paths
# --------------------------------------------------------------------------- #


class TestDeduplicatePaths:
    def test_identical_trails_collapse(self):
        first = assemble_path([_step()], temporal_index=_index({}))
        second = assemble_path([_step()], temporal_index=_index({}))
        unique = deduplicate_paths([first, second])
        assert len(unique) == 1

    def test_parallel_edges_stay_distinct(self):
        step_one = _step(
            relationship_id=derive_relationship_id({"parallel": 1})
        )
        step_two = _step(
            relationship_id=derive_relationship_id({"parallel": 2})
        )
        first = assemble_path([step_one], temporal_index=_index({}))
        second = assemble_path([step_two], temporal_index=_index({}))
        unique = deduplicate_paths([first, second])
        assert len(unique) == 2

    def test_different_directions_stay_distinct(self):
        forward = assemble_path([_step()], temporal_index=_index({}))
        reverse = WalkStep(
            from_observation_id=OBS_B,
            to_observation_id=OBS_A,
            edge=_edge_between(OBS_A, OBS_B),
            orientation=EdgeOrientation.REVERSE,
        )
        backward_path = assemble_path([reverse], temporal_index=_index({}))
        unique = deduplicate_paths([forward, backward_path])
        assert len(unique) == 2

    def test_first_occurrence_is_kept(self):
        step = _step()
        first = assemble_path([step], temporal_index=_index({}), scope={"v": 1})
        second = assemble_path([step], temporal_index=_index({}), scope={"v": 2})
        unique = deduplicate_paths([first, second])
        assert len(unique) == 1
        assert unique[0] is first


def test_a_walk_through_a_real_graph_round_trips():
    """A step drawn straight from a built graph assembles against the graph."""
    relationship = _relationship(
        relationship_id=derive_relationship_id({"roundtrip": 1}),
        source_observation_ids=(OBS_A,),
        target_observation_ids=(OBS_B,),
    )
    graph = build_graph((relationship,))
    edge = graph.edges()[0]
    step = WalkStep(
        from_observation_id=edge.source_observation_ids[0],
        to_observation_id=edge.target_observation_ids[0],
        edge=edge,
        orientation=EdgeOrientation.FORWARD,
        temporal=build_temporal_transition(
            edge.source_observation_ids[0], edge.target_observation_ids[0], T0, T1
        ),
    )
    path = assemble_path(
        [step], temporal_index=_index({OBS_A: T0, OBS_B: T1})
    )
    assert path.transitions[0].edge_id == edge.edge_id
    from backend.app.investigation.models import derive_path_id

    assert path.path_id == derive_path_id(
        path.observation_ids, path.relationship_ids
    )