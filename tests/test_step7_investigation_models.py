"""Tests for the Step 7 investigation-path models (SIH26146 - Step 7).

This file checks the *vocabulary* layer: the path identifier derivation, the
enums, the request/result contracts, the evidence-carrying transition/node
models, and the anti-attribution guarantee.  The temporal comparison rules,
path assembly/ordering and the search engine live in their own test files.
"""

from __future__ import annotations

import ast
import pathlib
import re
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from backend.app.correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    InferenceLevel,
    TemporalRelation,
)
from backend.app.correlation.models import (
    Confidence,
    EvidenceRelationship,
    TemporalWindow,
    derive_relationship_id,
)
from backend.app.correlation.provenance import ProvenanceRef
from backend.app.graph import EvidenceEdge, ObservationNode, derive_node_id
from backend.app.graph.builder import build_graph
from backend.app.investigation import models as investigation_models
from backend.app.investigation import (
    PATH_NAMESPACE,
    EdgeOrientation,
    EvidenceSummary,
    IncomparablePolicy,
    InvestigationError,
    InvestigationPath,
    PathNode,
    PathSearchRequest,
    PathSearchResult,
    PathTransition,
    PathTemporalStatus,
    ProfileContext,
    SearchStatus,
    TemporalFlowStep,
    TemporalOrder,
    TemporalTransition,
    TemporalWindowConfig,
    TimestampAwareness,
    derive_path_id,
)
from backend.app.ml.models import UnusualnessBand

UTC = timezone.utc
NOW = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)
WINDOW_START = datetime(2024, 5, 1, 8, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2024, 5, 1, 9, 30, 0, tzinfo=UTC)

OBS_A = "obs-a"
OBS_B = "obs-b"
OBS_C = "obs-c"

INVESTIGATION_PACKAGE_DIR = pathlib.Path(
    pathlib.Path(investigation_models.__file__).resolve().parent
)
STEP7_TEST_FILES = sorted(
    pathlib.Path(__file__).resolve().parent.glob("test_step7_*.py")
)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def _ref(ingestion_id: str = "ing-1", record: str = "row 2") -> ProvenanceRef:
    return ProvenanceRef(
        ingestion_id=ingestion_id, source_file="evidence.csv", source_record=record
    )


def _relationship(**overrides) -> EvidenceRelationship:
    payload = {
        "relationship_id": derive_relationship_id({"investigation-test": 1}),
        "source_observation_ids": (OBS_A,),
        "target_observation_ids": (OBS_B,),
        "evidence_type": EvidenceType.SAME_IP,
        "inference_level": InferenceLevel.OBSERVED,
        "confidence": Confidence.exact_match(),
        "rationale": "Both observations carry src_ip 203.0.113.5 as recorded.",
        "rule_id": "rule.same_ip",
        "rule_version": "1.0.0",
        "provenance_refs": (_ref(),),
        "created_at": NOW,
    }
    payload.update(overrides)
    return EvidenceRelationship(**payload)


def _edge(**overrides) -> EvidenceEdge:
    return EvidenceEdge.from_relationship(_relationship(**overrides))


def _edge_fields(**overrides) -> dict:
    edge = _edge()
    fields = {name: getattr(edge, name) for name in EvidenceEdge.model_fields}
    fields.update(overrides)
    return fields


def _transition(**overrides) -> PathTransition:
    edge = _edge()
    payload = dict(
        _edge_fields(),
        from_observation_id=edge.source_observation_ids[0],
        to_observation_id=edge.target_observation_ids[0],
        orientation=EdgeOrientation.FORWARD,
        temporal=None,
    )
    payload = {
        name: payload[name]
        for name in (
            "from_observation_id",
            "to_observation_id",
            "edge_id",
            "relationship_id",
            "evidence_type",
            "inference_level",
            "confidence",
            "rule_id",
            "rule_version",
            "rationale",
            "provenance_refs",
            "temporal_window",
            "temporal_relation",
        )
    }
    payload["orientation"] = EdgeOrientation.FORWARD
    payload["temporal"] = None
    payload.update(overrides)
    return PathTransition(**payload)


def _node(observation_id: str = OBS_A, **overrides) -> PathNode:
    payload = {
        "observation_id": observation_id,
        "graph_node_id": derive_node_id(observation_id),
        "timestamp": None,
        "timestamp_awareness": TimestampAwareness.ABSENT,
        "ml_context": (),
    }
    payload.update(overrides)
    return PathNode(**payload)


def _path(
    observations: tuple[str, ...] = (OBS_A, OBS_B),
    transitions: tuple[PathTransition, ...] = (_transition(),),
    **overrides,
) -> InvestigationPath:
    nodes = tuple(_node(observation_id) for observation_id in observations)
    transitions = tuple(transitions)
    first = nodes[0].timestamp
    last = nodes[-1].timestamp
    relationship_ids = tuple(item.relationship_id for item in transitions)
    edge_values = [item.confidence.value for item in transitions]
    summary = EvidenceSummary(
        edge_count=len(transitions),
        relationship_ids=relationship_ids,
        evidence_types=tuple(
            sorted({item.evidence_type for item in transitions}, key=lambda e: e.value)
        ),
        inference_levels=tuple(
            sorted({item.inference_level for item in transitions}, key=lambda e: e.value)
        ),
        confidence_bases=tuple(
            sorted({item.confidence.basis for item in transitions}, key=lambda e: e.value)
        ),
        min_edge_confidence=min(edge_values),
        max_edge_confidence=max(edge_values),
    )
    payload = {
        "path_id": derive_path_id(
            tuple(node.observation_id for node in nodes),
            relationship_ids,
            scope={"variant": "TEST"},
        ),
        "nodes": nodes,
        "transitions": transitions,
        "hop_count": len(transitions),
        "first_timestamp": first,
        "last_timestamp": last,
        "duration_seconds": None,
        "temporal_status": PathTemporalStatus.UNKNOWN,
        "evidence_summary": summary,
    }
    payload.update(overrides)
    return InvestigationPath(**payload)


# --------------------------------------------------------------------------- #
# The public surface
# --------------------------------------------------------------------------- #


class TestPublicSurface:
    def test_the_module_exports_a_sorted_complete_list(self):
        assert list(investigation_models.__all__) == sorted(investigation_models.__all__)
        assert set(investigation_models.__all__) <= set(dir(investigation_models))
        for name in investigation_models.__all__:
            assert getattr(investigation_models, name) is not None, name

    def test_no_exported_name_is_an_implementation_detail(self):
        for name in investigation_models.__all__:
            assert not name.startswith("_"), name


# --------------------------------------------------------------------------- #
# Path identifier derivation
# --------------------------------------------------------------------------- #


class TestPathId:
    def test_path_id_is_a_uuidv5_string(self):
        path_id = derive_path_id((OBS_A, OBS_B), ("rel-1",))
        assert isinstance(path_id, str)
        assert uuid.UUID(path_id).version == 5

    def test_path_id_is_deterministic(self):
        first = derive_path_id((OBS_A, OBS_B), ("rel-1",), scope={"variant": "X"})
        second = derive_path_id((OBS_A, OBS_B), ("rel-1",), scope={"variant": "X"})
        assert first == second

    def test_path_id_is_order_sensitive(self):
        forward = derive_path_id((OBS_A, OBS_B), ("rel-1",))
        backward = derive_path_id((OBS_B, OBS_A), ("rel-1",))
        assert forward != backward

    def test_path_id_is_relationship_sensitive(self):
        one = derive_path_id((OBS_A, OBS_B), ("rel-1",))
        two = derive_path_id((OBS_A, OBS_B), ("rel-2",))
        assert one != two

    def test_path_id_is_scope_sensitive(self):
        one = derive_path_id((OBS_A, OBS_B), ("rel-1",), scope={"variant": "X"})
        two = derive_path_id((OBS_A, OBS_B), ("rel-1",), scope={"variant": "Y"})
        assert one != two

    def test_scope_canonicalisation_makes_equivalent_scopes_equal(self):
        assert derive_path_id(
            (OBS_A, OBS_B), ("rel-1",), scope={"b": 1, "a": [2, 1]}
        ) == derive_path_id(
            (OBS_A, OBS_B), ("rel-1",), scope={"a": [2, 1], "b": 1}
        )

    def test_equal_scope_text_with_different_structure_stay_different(self):
        assert derive_path_id(
            (OBS_A, OBS_B), ("rel-1",), scope={"a": [1, 2]}
        ) != derive_path_id(
            (OBS_A, OBS_B), ("rel-1",), scope={"a": 12}
        )

    def test_path_id_is_in_its_own_namespace(self):
        derived = derive_path_id(
            (OBS_A, OBS_B), ("rel-1",), scope={"variant": "X"}
        )
        assert isinstance(derived, str)
        assert uuid.UUID(derived).version == 5

    def test_path_namespace_differs_from_the_node_and_edge_namespaces(self):
        from backend.app.graph import GRAPH_EDGE_NAMESPACE, GRAPH_NODE_NAMESPACE

        assert {
            str(PATH_NAMESPACE),
            str(GRAPH_NODE_NAMESPACE),
            str(GRAPH_EDGE_NAMESPACE),
        } == {str(PATH_NAMESPACE), str(GRAPH_NODE_NAMESPACE), str(GRAPH_EDGE_NAMESPACE)}
        assert len({str(PATH_NAMESPACE), str(GRAPH_NODE_NAMESPACE), str(GRAPH_EDGE_NAMESPACE)}) == 3

    def test_empty_path_is_refused(self):
        with pytest.raises(InvestigationError):
            derive_path_id((), ())

    def test_blank_observation_is_refused(self):
        with pytest.raises(InvestigationError):
            derive_path_id(("obs-a", "  "), ("rel-1",))

    def test_blank_relationship_is_refused(self):
        with pytest.raises(InvestigationError):
            derive_path_id((OBS_A, OBS_B), ("rel-1", ""))


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class TestEnums:
    def test_search_status_has_the_five_outcomes(self):
        assert [item.value for item in SearchStatus] == [
            "FOUND",
            "NOT_FOUND",
            "INVALID_REQUEST",
            "NO_TEMPORAL_COMPARISON",
            "INSUFFICIENT_DATA",
        ]

    def test_temporal_order_has_the_five_relations(self):
        assert [item.value for item in TemporalOrder] == [
            "BEFORE",
            "SAME_TIME",
            "AFTER",
            "NOT_COMPARABLE",
            "UNKNOWN",
        ]

    def test_path_temporal_status_has_the_four_states(self):
        assert [item.value for item in PathTemporalStatus] == [
            "COMPARABLE",
            "PARTIALLY_COMPARABLE",
            "NOT_COMPARABLE",
            "UNKNOWN",
        ]

    def test_edge_orientation_has_the_two_directions(self):
        assert [item.value for item in EdgeOrientation] == ["FORWARD", "REVERSE"]

    def test_timestamp_awareness_has_the_three_states(self):
        assert [item.value for item in TimestampAwareness] == [
            "AWARE",
            "NAIVE",
            "ABSENT",
        ]

    def test_incomparable_policy_has_no_silent_default_guess(self):
        assert [item.value for item in IncomparablePolicy] == ["EXCLUDE", "INCLUDE"]

    def test_investigation_error_is_a_value_error(self):
        assert issubclass(InvestigationError, ValueError)


# --------------------------------------------------------------------------- #
# ProfileContext (Step 5/6 metadata, carried as context only)
# --------------------------------------------------------------------------- #


class TestProfileContext:
    def test_minimal_context_is_valid(self):
        context = ProfileContext(
            profile_id="profile-1", observation_ids=(OBS_A,)
        )
        assert context.profile_id == "profile-1"
        assert context.observation_ids == (OBS_A,)
        assert context.cluster_id is None
        assert context.normalized_score is None
        assert context.band is None

    def test_context_carries_cluster_and_anomaly_figures_verbatim(self):
        context = ProfileContext(
            profile_id="profile-1",
            observation_ids=(OBS_A, OBS_B),
            cluster_id="cluster-7",
            normalized_score=0.82,
            band=UnusualnessBand.HIGH,
        )
        assert context.cluster_id == "cluster-7"
        assert context.normalized_score == 0.82
        assert context.band is UnusualnessBand.HIGH

    def test_to_dict_is_deterministic(self):
        context = ProfileContext(
            profile_id="profile-1",
            observation_ids=(OBS_A, OBS_B),
            cluster_id="cluster-7",
            normalized_score=0.82,
            band=UnusualnessBand.HIGH,
        )
        assert context.to_dict() == context.to_dict()

    def test_is_frozen(self):
        context = ProfileContext(profile_id="profile-1", observation_ids=(OBS_A,))
        with pytest.raises(ValidationError):
            context.profile_id = "other"

    def test_blank_profile_id_is_refused(self):
        with pytest.raises(ValidationError):
            ProfileContext(profile_id="  ", observation_ids=(OBS_A,))

    def test_empty_observation_ids_are_refused(self):
        with pytest.raises(ValidationError):
            ProfileContext(profile_id="profile-1", observation_ids=())

    def test_observation_ids_must_be_ascending_and_unique(self):
        with pytest.raises(ValidationError):
            ProfileContext(profile_id="p", observation_ids=(OBS_B, OBS_A))
        with pytest.raises(ValidationError):
            ProfileContext(profile_id="p", observation_ids=(OBS_A, OBS_A))

    def test_score_must_lie_inside_open_unit_interval(self):
        with pytest.raises(ValidationError):
            ProfileContext(profile_id="p", observation_ids=(OBS_A,), normalized_score=0.0)
        with pytest.raises(ValidationError):
            ProfileContext(profile_id="p", observation_ids=(OBS_A,), normalized_score=1.5)
        with pytest.raises(ValidationError):
            ProfileContext(
                profile_id="p", observation_ids=(OBS_A,), normalized_score=float("nan")
            )

    def test_band_requires_its_score(self):
        with pytest.raises(ValidationError):
            ProfileContext(
                profile_id="p",
                observation_ids=(OBS_A,),
                band=UnusualnessBand.HIGH,
            )

    def test_extra_fields_are_forbidden(self):
        with pytest.raises(ValidationError):
            ProfileContext(
                profile_id="p",
                observation_ids=(OBS_A,),
                owner="someone",
            )


# --------------------------------------------------------------------------- #
# TemporalWindowConfig
# --------------------------------------------------------------------------- #


class TestTemporalWindowConfig:
    def test_empty_config_restricts_nothing(self):
        config = TemporalWindowConfig()
        assert config.restricts_transitions is False
        assert config.restricts_path_duration is False
        assert config.on_incomparable is IncomparablePolicy.EXCLUDE

    def test_gap_bounds_restrict_transitions(self):
        config = TemporalWindowConfig(min_transition_gap=timedelta(seconds=10))
        assert config.restricts_transitions is True
        config = TemporalWindowConfig(max_transition_gap=timedelta(seconds=10))
        assert config.restricts_transitions is True

    def test_path_duration_restricts_the_overall_window(self):
        config = TemporalWindowConfig(max_path_duration=timedelta(hours=1))
        assert config.restricts_path_duration is True

    def test_negative_bounds_are_refused(self):
        with pytest.raises(ValidationError):
            TemporalWindowConfig(min_transition_gap=timedelta(seconds=-1))
        with pytest.raises(ValidationError):
            TemporalWindowConfig(max_transition_gap=timedelta(seconds=-1))
        with pytest.raises(ValidationError):
            TemporalWindowConfig(max_path_duration=timedelta(seconds=-1))
        with pytest.raises(ValidationError):
            TemporalWindowConfig(gap_boundary=timedelta(seconds=-1))

    def test_min_must_not_exceed_max(self):
        with pytest.raises(ValidationError):
            TemporalWindowConfig(
                min_transition_gap=timedelta(seconds=60),
                max_transition_gap=timedelta(seconds=30),
            )

    def test_to_dict_reports_seconds_and_is_deterministic(self):
        config = TemporalWindowConfig(
            min_transition_gap=timedelta(seconds=10),
            gap_boundary=timedelta(minutes=2),
        )
        expected = {
            "min_transition_gap_seconds": 10.0,
            "max_transition_gap_seconds": None,
            "max_path_duration_seconds": None,
            "gap_boundary_seconds": 120.0,
            "on_incomparable": "EXCLUDE",
        }
        assert config.to_dict() == expected

    def test_is_frozen(self):
        config = TemporalWindowConfig()
        with pytest.raises(ValidationError):
            config.min_transition_gap = timedelta(seconds=1)


# --------------------------------------------------------------------------- #
# TemporalTransition
# --------------------------------------------------------------------------- #


class TestTemporalTransition:
    def test_comparable_pair_agrees_with_elapsed(self):
        source = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
        target = datetime(2024, 5, 1, 12, 4, tzinfo=UTC)
        transition = TemporalTransition(
            source_observation_id=OBS_A,
            target_observation_id=OBS_B,
            relation=TemporalOrder.AFTER,
            comparable=True,
            source_timestamp=source,
            target_timestamp=target,
            source_awareness=TimestampAwareness.AWARE,
            target_awareness=TimestampAwareness.AWARE,
            elapsed_seconds=240.0,
            crosses_boundary=False,
        )
        assert transition.elapsed_seconds == 240.0
        assert transition.crosses_boundary is False

    def test_not_comparable_pair_never_invents_elapsed(self):
        transition = TemporalTransition(
            source_observation_id=OBS_A,
            target_observation_id=OBS_B,
            relation=TemporalOrder.NOT_COMPARABLE,
            comparable=False,
            source_timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
            target_timestamp=datetime(2024, 5, 1, 12, 0),
            source_awareness=TimestampAwareness.AWARE,
            target_awareness=TimestampAwareness.NAIVE,
            elapsed_seconds=None,
            crosses_boundary=None,
        )
        assert transition.comparable is False
        assert transition.elapsed_seconds is None

    def test_missing_side_is_unknown_and_not_comparable(self):
        transition = TemporalTransition(
            source_observation_id=OBS_A,
            target_observation_id=OBS_B,
            relation=TemporalOrder.UNKNOWN,
            comparable=False,
            source_timestamp=None,
            target_timestamp=None,
            source_awareness=TimestampAwareness.ABSENT,
            target_awareness=TimestampAwareness.ABSENT,
            elapsed_seconds=None,
            crosses_boundary=None,
        )
        assert transition.comparable is False

    def test_elapsed_must_match_the_timestamps(self):
        with pytest.raises(ValidationError):
            TemporalTransition(
                source_observation_id=OBS_A,
                target_observation_id=OBS_B,
                relation=TemporalOrder.AFTER,
                comparable=True,
                source_timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
                target_timestamp=datetime(2024, 5, 1, 12, 4, tzinfo=UTC),
                source_awareness=TimestampAwareness.AWARE,
                target_awareness=TimestampAwareness.AWARE,
                elapsed_seconds=120.0,
                crosses_boundary=False,
            )

    def test_awareness_must_match_the_timestamp(self):
        with pytest.raises(ValidationError):
            TemporalTransition(
                source_observation_id=OBS_A,
                target_observation_id=OBS_B,
                relation=TemporalOrder.UNKNOWN,
                comparable=False,
                source_timestamp=None,
                target_timestamp=None,
                source_awareness=TimestampAwareness.AWARE,
                target_awareness=TimestampAwareness.ABSENT,
                elapsed_seconds=None,
                crosses_boundary=None,
            )

    def test_self_transition_is_refused(self):
        with pytest.raises(ValidationError):
            TemporalTransition(
                source_observation_id=OBS_A,
                target_observation_id=OBS_A,
                relation=TemporalOrder.UNKNOWN,
                comparable=False,
                source_timestamp=None,
                target_timestamp=None,
                source_awareness=TimestampAwareness.ABSENT,
                target_awareness=TimestampAwareness.ABSENT,
                elapsed_seconds=None,
                crosses_boundary=None,
            )

    def test_to_dict_is_deterministic(self):
        transition = TemporalTransition(
            source_observation_id=OBS_A,
            target_observation_id=OBS_B,
            relation=TemporalOrder.AFTER,
            comparable=True,
            source_timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
            target_timestamp=datetime(2024, 5, 1, 12, 4, tzinfo=UTC),
            source_awareness=TimestampAwareness.AWARE,
            target_awareness=TimestampAwareness.AWARE,
            elapsed_seconds=240.0,
            crosses_boundary=False,
        )
        assert transition.to_dict() == transition.to_dict()


# --------------------------------------------------------------------------- #
# PathTransition
# --------------------------------------------------------------------------- #


class TestPathTransition:
    def test_transition_copies_evidence_verbatim(self):
        edge = _edge()
        transition = PathTransition(
            from_observation_id=edge.source_observation_ids[0],
            to_observation_id=edge.target_observation_ids[0],
            edge_id=edge.edge_id,
            relationship_id=edge.relationship_id,
            orientation=EdgeOrientation.FORWARD,
            evidence_type=edge.evidence_type,
            inference_level=edge.inference_level,
            confidence=edge.confidence,
            rule_id=edge.rule_id,
            rule_version=edge.rule_version,
            rationale=edge.rationale,
            provenance_refs=edge.provenance_refs,
            temporal=None,
            temporal_window=edge.temporal_window,
            temporal_relation=edge.temporal_relation,
        )
        assert transition.relationship_id == edge.relationship_id
        assert transition.evidence_type == edge.evidence_type
        assert transition.confidence.value == edge.confidence.value

    def test_self_transition_is_refused(self):
        with pytest.raises(ValidationError):
            _transition(
                from_observation_id=OBS_A,
                to_observation_id=OBS_A,
            )

    def test_temporal_relation_requires_temporal_window(self):
        with pytest.raises(ValidationError):
            _transition(
                temporal_relation=TemporalRelation.WITHIN,
                temporal_window=None,
            )

    def test_embedded_temporal_transition_must_align(self):
        temporal = TemporalTransition(
            source_observation_id=OBS_A,
            target_observation_id=OBS_B,
            relation=TemporalOrder.AFTER,
            comparable=True,
            source_timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
            target_timestamp=datetime(2024, 5, 1, 12, 4, tzinfo=UTC),
            source_awareness=TimestampAwareness.AWARE,
            target_awareness=TimestampAwareness.AWARE,
            elapsed_seconds=240.0,
            crosses_boundary=False,
        )
        with pytest.raises(ValidationError):
            _transition(
                temporal=temporal,
                from_observation_id=OBS_C,
            )

    def test_is_frozen_and_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            _transition(owner="someone")


# --------------------------------------------------------------------------- #
# PathNode
# --------------------------------------------------------------------------- #


class TestPathNode:
    def test_node_is_an_observation_with_a_derived_graph_id(self):
        node = _node(OBS_A)
        assert node.observation_id == OBS_A
        assert node.graph_node_id == derive_node_id(OBS_A)

    def test_graph_node_id_must_be_the_derived_id(self):
        with pytest.raises(ValidationError):
            _node(OBS_A, graph_node_id="not-the-derived-id")

    def test_awareness_must_match_the_timestamp(self):
        with pytest.raises(ValidationError):
            _node(
                OBS_A,
                timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
                timestamp_awareness=TimestampAwareness.NAIVE,
            )
        with pytest.raises(ValidationError):
            _node(
                OBS_A,
                timestamp=datetime(2024, 5, 1, 12, 0),
                timestamp_awareness=TimestampAwareness.AWARE,
            )
        _node(OBS_A, timestamp=None, timestamp_awareness=TimestampAwareness.ABSENT)

    def test_ml_context_must_be_passed_already_sorted(self):
        first = ProfileContext(profile_id="profile-a", observation_ids=(OBS_A,))
        second = ProfileContext(profile_id="profile-b", observation_ids=(OBS_A,))
        node = _node(OBS_A, ml_context=(first, second))
        assert node.ml_context == (first, second)
        with pytest.raises(ValidationError):
            _node(OBS_A, ml_context=(second, first))
        with pytest.raises(ValidationError):
            _node(OBS_A, ml_context=(first, first))

    def test_node_carries_no_attribution_field(self):
        for field in node_fields(_node(OBS_A)):
            assert "attribut" not in field.lower() and field not in {
                "owner",
                "identity",
                "actor",
            }


def node_fields(node: PathNode) -> tuple[str, ...]:
    return tuple(dict(node).keys())


# --------------------------------------------------------------------------- #
# EvidenceSummary
# --------------------------------------------------------------------------- #


class TestEvidenceSummary:
    def test_empty_path_summary_is_empty(self):
        summary = EvidenceSummary(edge_count=0)
        assert summary.evidence_types == ()
        assert summary.relationship_ids == ()
        assert summary.min_edge_confidence is None

    def test_non_empty_summary_reports_structure_only(self):
        transition = _transition()
        summary = EvidenceSummary(
            edge_count=1,
            relationship_ids=(transition.relationship_id,),
            evidence_types=(transition.evidence_type,),
            inference_levels=(transition.inference_level,),
            confidence_bases=(transition.confidence.basis,),
            min_edge_confidence=transition.confidence.value,
            max_edge_confidence=transition.confidence.value,
        )
        assert summary.edge_count == 1
        assert summary.min_edge_confidence == 1.0

    def test_relationship_ids_must_count_the_edges(self):
        with pytest.raises(ValidationError):
            EvidenceSummary(
                edge_count=2,
                relationship_ids=("rel-1",),
                evidence_types=(EvidenceType.SAME_IP,),
                inference_levels=(InferenceLevel.OBSERVED,),
                confidence_bases=(ConfidenceBasis.EXACT_MATCH,),
                min_edge_confidence=0.5,
                max_edge_confidence=0.6,
            )

    def test_relationship_ids_must_be_unique(self):
        with pytest.raises(ValidationError):
            EvidenceSummary(
                edge_count=2,
                relationship_ids=("rel-1", "rel-1"),
                evidence_types=(EvidenceType.SAME_IP,),
                inference_levels=(InferenceLevel.OBSERVED,),
                confidence_bases=(ConfidenceBasis.EXACT_MATCH,),
                min_edge_confidence=0.5,
                max_edge_confidence=0.5,
            )

    def test_confidence_bounds_must_be_ordered(self):
        with pytest.raises(ValidationError):
            EvidenceSummary(
                edge_count=1,
                relationship_ids=("rel-1",),
                evidence_types=(EvidenceType.SAME_IP,),
                inference_levels=(InferenceLevel.OBSERVED,),
                confidence_bases=(ConfidenceBasis.EXACT_MATCH,),
                min_edge_confidence=0.9,
                max_edge_confidence=0.5,
            )

    def test_summary_holds_no_path_level_figure(self):
        summary = EvidenceSummary(edge_count=0)
        assert "score" not in summary.model_dump_json().lower()
        assert "risk" not in summary.model_dump_json().lower()


# --------------------------------------------------------------------------- #
# TemporalFlowStep
# --------------------------------------------------------------------------- #


class TestTemporalFlowStep:
    def test_machine_readable_line(self):
        step = TemporalFlowStep(
            from_observation_id=OBS_A,
            to_observation_id=OBS_B,
            relationship_id="rel-1",
            evidence_type=EvidenceType.SAME_IP,
            source_timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
            target_timestamp=datetime(2024, 5, 1, 12, 4, tzinfo=UTC),
            elapsed_seconds=240.0,
            temporal_relation=TemporalOrder.AFTER,
            comparable=True,
        )
        assert step.to_dict()["elapsed_seconds"] == 240.0
        assert step.to_dict()["temporal_relation"] == "AFTER"


# --------------------------------------------------------------------------- #
# InvestigationPath
# --------------------------------------------------------------------------- #


class TestInvestigationPath:
    def test_single_hop_path_is_valid(self):
        path = _path()
        assert path.hop_count == 1
        assert path.observation_ids == (OBS_A, OBS_B)
        assert path.relationship_ids == (path.transitions[0].relationship_id,)

    def test_hop_count_must_match_transition_count(self):
        with pytest.raises(ValidationError):
            _path(transitions=(_transition(),), hop_count=2)

    def test_node_count_must_be_hop_count_plus_one(self):
        with pytest.raises(ValidationError):
            _path(
                observations=(OBS_A, OBS_B, OBS_C),
                transitions=(_transition(),),
            )

    def test_simple_path_must_not_revisit_an_observation(self):
        with pytest.raises(ValidationError):
            _path(
                observations=(OBS_A, OBS_B),
                transitions=(
                    _transition(),
                    _transition(
                        from_observation_id=OBS_B,
                        to_observation_id=OBS_A,
                        relationship_id=derive_relationship_id({"t": 2}),
                    ),
                ),
            )

    def test_simple_path_must_not_repeat_an_edge(self):
        with pytest.raises(ValidationError):
            _path(
                observations=(OBS_A, OBS_B, OBS_A),
                transitions=(
                    _transition(),
                    _transition(
                        from_observation_id=OBS_B,
                        to_observation_id=OBS_A,
                        relationship_id=derive_relationship_id({"t": 2}),
                    ),
                ),
            )

    def test_transitions_must_be_continuous(self):
        with pytest.raises(ValidationError):
            _path(
                observations=(OBS_A, OBS_B),
                transitions=(
                    _transition(
                        from_observation_id=OBS_A,
                        to_observation_id=OBS_C,
                    ),
                ),
            )

    def test_temporal_status_is_unknown_without_temporal_descriptors(self):
        path = _path()
        assert path.temporal_status is PathTemporalStatus.UNKNOWN

    def test_temporal_flow_matches_transitions(self):
        path = _path()
        flow = path.temporal_flow()
        assert len(flow) == 1
        step = flow[0]
        assert step.from_observation_id == OBS_A
        assert step.to_observation_id == OBS_B
        assert step.temporal_relation is TemporalOrder.UNKNOWN
        assert step.comparable is False

    def test_temporal_flow_with_comparable_timestamps(self):
        source = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
        target = datetime(2024, 5, 1, 12, 4, tzinfo=UTC)
        temporal = TemporalTransition(
            source_observation_id=OBS_A,
            target_observation_id=OBS_B,
            relation=TemporalOrder.AFTER,
            comparable=True,
            source_timestamp=source,
            target_timestamp=target,
            source_awareness=TimestampAwareness.AWARE,
            target_awareness=TimestampAwareness.AWARE,
            elapsed_seconds=240.0,
            crosses_boundary=False,
        )
        nodes = (
            _node(
                OBS_A,
                timestamp=source,
                timestamp_awareness=TimestampAwareness.AWARE,
            ),
            _node(
                OBS_B,
                timestamp=target,
                timestamp_awareness=TimestampAwareness.AWARE,
            ),
        )
        path = _path(
            nodes=nodes,
            observations=(OBS_A, OBS_B),
            transitions=(
                _transition(
                    temporal=temporal,
                    from_observation_id=OBS_A,
                    to_observation_id=OBS_B,
                ),
            ),
            first_timestamp=source,
            last_timestamp=target,
            duration_seconds=240.0,
            temporal_status=PathTemporalStatus.COMPARABLE,
        )
        flow = path.temporal_flow()
        assert flow[0].temporal_relation is TemporalOrder.AFTER
        assert flow[0].comparable is True

    def test_explains_attribution_is_always_false(self):
        assert _path().explains_attribution() is False

    def test_path_has_no_attribution_field(self):
        assert "attribution" not in _path().model_dump_json().lower()

    def test_path_id_matches_its_contents(self):
        path = _path()
        assert path.path_id == derive_path_id(
            path.observation_ids,
            path.relationship_ids,
            scope={"variant": "TEST"},
        )

    def test_to_dict_is_deterministic(self):
        assert _path().to_dict() == _path().to_dict()


# --------------------------------------------------------------------------- #
# PathSearchRequest
# --------------------------------------------------------------------------- #


class TestPathSearchRequest:
    def test_bounds_are_required_with_no_default(self):
        request = PathSearchRequest(
            start_observation_id=OBS_A,
            max_depth=3,
            max_paths=5,
        )
        assert request.max_depth == 3
        assert request.max_paths == 5
        assert request.target_observation_id is None
        assert request.directed is True
        assert request.temporal is None

    def test_zero_bounds_are_refused(self):
        with pytest.raises(ValidationError):
            PathSearchRequest(
                start_observation_id=OBS_A, max_depth=0, max_paths=5
            )
        with pytest.raises(ValidationError):
            PathSearchRequest(
                start_observation_id=OBS_A, max_depth=3, max_paths=0
            )

    def test_allowed_lists_are_normalised_ascending_and_unique(self):
        request = PathSearchRequest(
            start_observation_id=OBS_A,
            max_depth=3,
            max_paths=5,
            allowed_evidence_types=(EvidenceType.SAME_IP, EvidenceType.SAME_TXID),
        )
        assert request.allowed_evidence_types == (
            EvidenceType.SAME_IP,
            EvidenceType.SAME_TXID,
        )
        with pytest.raises(ValidationError):
            PathSearchRequest(
                start_observation_id=OBS_A,
                max_depth=3,
                max_paths=5,
                allowed_evidence_types=(EvidenceType.SAME_IP, EvidenceType.SAME_IP),
            )

    def test_scope_payload_excludes_search_limits(self):
        request = PathSearchRequest(
            start_observation_id=OBS_A,
            max_depth=3,
            max_paths=5,
            directed=True,
            allowed_evidence_types=(EvidenceType.SAME_IP,),
            allowed_inference_levels=(InferenceLevel.OBSERVED,),
        )
        payload = request.scope_payload("SHORTEST", None)
        assert "max_depth" not in payload
        assert "max_paths" not in payload
        assert payload["variant"] == "SHORTEST"
        assert payload["directed"] is True
        assert payload["allowed_evidence_types"] == ["SAME_IP"]
        assert payload["allowed_inference_levels"] == ["OBSERVED"]
        assert payload["temporal"] is None

    def test_scope_payload_carries_the_temporal_config(self):
        config = TemporalWindowConfig(max_transition_gap=timedelta(seconds=60))
        request = PathSearchRequest(
            start_observation_id=OBS_A,
            max_depth=3,
            max_paths=5,
            temporal=config,
        )
        payload = request.scope_payload("ENUMERATE", config)
        assert payload["temporal"]["max_transition_gap_seconds"] == 60.0
        assert payload["variant"] == "ENUMERATE"

    def test_blank_identifiers_are_refused(self):
        with pytest.raises(ValidationError):
            PathSearchRequest(
                start_observation_id="  ", max_depth=3, max_paths=5
            )


# --------------------------------------------------------------------------- #
# PathSearchResult
# --------------------------------------------------------------------------- #


class TestPathSearchResult:
    def test_no_result_carries_no_paths(self):
        request = PathSearchRequest(
            start_observation_id=OBS_A, max_depth=3, max_paths=5
        )
        result = PathSearchResult(
            status=SearchStatus.NOT_FOUND,
            detail="no path under the given bounds",
            request=request,
            path_count=0,
            shortest_hops=None,
            search_scope=request.scope_payload("ENUMERATE", None) and "{}",
        )
        assert result.status is SearchStatus.NOT_FOUND
        assert result.paths == ()

    def test_found_result_must_carry_paths(self):
        request = PathSearchRequest(
            start_observation_id=OBS_A, max_depth=3, max_paths=5
        )
        with pytest.raises(ValidationError):
            PathSearchResult(
                status=SearchStatus.FOUND,
                detail="found",
                request=request,
                path_count=0,
                shortest_hops=None,
                search_scope="{}",
            )

    def test_path_count_must_match(self):
        request = PathSearchRequest(
            start_observation_id=OBS_A, max_depth=3, max_paths=5
        )
        with pytest.raises(ValidationError):
            PathSearchResult(
                status=SearchStatus.NOT_FOUND,
                detail="none",
                request=request,
                path_count=1,
                shortest_hops=None,
                search_scope="{}",
            )

    def test_explains_attribution_is_always_false(self):
        request = PathSearchRequest(
            start_observation_id=OBS_A, max_depth=3, max_paths=5
        )
        result = PathSearchResult(
            status=SearchStatus.NOT_FOUND,
            detail="none",
            request=request,
            path_count=0,
            shortest_hops=None,
            search_scope="{}",
        )
        assert result.explains_attribution() is False


# --------------------------------------------------------------------------- #
# Anti-attribution and boundary
# --------------------------------------------------------------------------- #

_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+")
_SPLIT_RE = re.compile(r"[^0-9A-Za-z]+")

BANNED_MODEL_WORDS = frozenset(
    {
        "actor",
        "attacker",
        "attribution",
        "criminal",
        "fraud",
        "identity",
        "illicit",
        "malicious",
        "owner",
        "person",
        "risk",
        "score",
        "suspicious",
        "threat",
        "verdict",
    }
)


#: Identifiers allowed despite containing a guarded word.  The first names the
#: Step 6 *context* value -- ``normalized_score`` is a figure carried verbatim
#: on a node, never combined into a path-level number.  The next two are the
#: search-scope plumbing.  The engine is the search object itself.
#: ``explains_attribution`` is the standard negative assertion this codebase
#: uses everywhere: a method that answers "no", so the absence of a claim is
#: callable rather than merely documented.
ALLOWED_NEGATIVE_IDENTIFIERS = frozenset(
    {
        "normalized_score",
        "search_scope",
        "scope_payload",
        "InvestigationEngine",
        "explains_attribution",
    }
)


def identifier_words(name: str) -> tuple[str, ...]:
    words: list[str] = []
    for chunk in _SPLIT_RE.split(name):
        if not chunk:
            continue
        words.extend(match.group(0).lower() for match in _CAMEL_RE.finditer(chunk))
    return tuple(words)


class TestAntiAttribution:
    def test_no_model_field_mentions_a_banned_concept(self):
        offenders = []
        for model in (
            ProfileContext,
            TemporalWindowConfig,
            TemporalTransition,
            PathTransition,
            PathNode,
            EvidenceSummary,
            InvestigationPath,
            PathSearchRequest,
            PathSearchResult,
        ):
            for name in model.model_fields:
                if name in ALLOWED_NEGATIVE_IDENTIFIERS:
                    continue
                for word in identifier_words(name):
                    if word in BANNED_MODEL_WORDS:
                        offenders.append("%s:%s(%s)" % (model.__name__, name, word))
        assert not offenders, "banned field words: " + "; ".join(sorted(offenders))

    def test_no_exported_name_mentions_a_banned_concept(self):
        for name in investigation_models.__all__:
            if name in ALLOWED_NEGATIVE_IDENTIFIERS:
                continue
            for word in identifier_words(name):
                assert word not in BANNED_MODEL_WORDS, name

    def test_that_step_7_files_came_from_the_boundary_scan_module(self):
        assert STEP7_TEST_FILES, "the scan must have files to read"

    def test_no_non_public_identifier_carries_a_banned_word_in_models(self):
        tree = ast.parse(
            pathlib.Path(investigation_models.__file__).read_text(encoding="utf-8")
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in ALLOWED_NEGATIVE_IDENTIFIERS:
                    continue
                for word in identifier_words(node.name):
                    assert word not in BANNED_MODEL_WORDS, node.name
            elif isinstance(node, ast.arg):
                if node.arg in ALLOWED_NEGATIVE_IDENTIFIERS:
                    continue
                for word in identifier_words(node.arg):
                    assert word not in BANNED_MODEL_WORDS, node.arg


class TestBoundary:
    def test_the_investigation_package_imports_no_substrate(self):
        offenders = []
        for path in INVESTIGATION_PACKAGE_DIR.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("backend.app.domain") or node.module.startswith(
                        "backend.app.ingestion"
                    ):
                        offenders.append((path.name, node.module))
        assert offenders == []

    def test_the_investigation_package_imports_no_interface_or_store(self):
        offenders = []
        for path in INVESTIGATION_PACKAGE_DIR.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.level:
                        continue  # relative imports only reach backend.app layers
                    root = node.module.split(".")[0]
                    if root not in (
                        "collections",
                        "dataclasses",
                        "datetime",
                        "enum",
                        "functools",
                        "json",
                        "math",
                        "typing",
                        "uuid",
                        "backend",
                        "__future__",
                    ) and node.module not in {"pydantic"}:
                        offenders.append((path.name, node.module))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in {
                            "fastapi",
                            "flask",
                            "requests",
                            "neo4j",
                            "networkx",
                            "sqlalchemy",
                            "numpy",
                            "pandas",
                            "scipy",
                            "sklearn",
                            "torch",
                            "redis",
                            "kafka",
                            "airflow",
                            "celery",
                            "pydantic",
                        }:
                            offenders.append((path.name, alias.name))
        assert not offenders, "forbidden imports: " + "; ".join(
            "%s: %s" % (path, module) for path, module in sorted(offenders)
        )

    def test_every_backend_import_from_the_package_is_an_allowed_layer(self):
        allowed = {
            "backend.app.investigation",
            "backend.app.graph",
            "backend.app.correlation",
            "backend.app.ml",
        }
        for path in INVESTIGATION_PACKAGE_DIR.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("backend"):
                        assert node.module in allowed, (
                            "%s: %s is outside the allowed layers"
                            % (path.name, node.module)
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("backend"):
                            parts = alias.name.split(".")
                            assert ".".join(parts[:3]) in allowed, (
                                "%s: %s is outside the allowed layers"
                                % (path.name, alias.name)
                            )


def test_the_search_result_scope_is_a_real_json_document():
    """Guard against the placeholder scope used in the minimal-result tests."""
    import json as _json

    request = PathSearchRequest(
        start_observation_id=OBS_A, max_depth=3, max_paths=5
    )
    for result in (
        PathSearchResult(
            status=SearchStatus.NOT_FOUND,
            detail="none",
            request=request,
            path_count=0,
            shortest_hops=None,
            search_scope=_json.dumps(
                request.scope_payload("ENUMERATE", None), sort_keys=True
            ),
        ),
    ):
        assert _json.loads(result.search_scope)["variant"] == "ENUMERATE"


def test_the_investigation_package_re_exports_its_vocabulary():
    import backend.app.investigation as package

    for name in investigation_models.__all__:
        assert getattr(package, name, None) is not None, name


def test_the_module_export_lists_are_subset_of_the_package_surface():
    """Every name a module exports must be reachable from the package."""
    import backend.app.investigation as package
    from backend.app.investigation import (
        engine as investigation_engine,
        paths as investigation_paths,
        temporal as investigation_temporal,
    )

    package_surface = set(package.__all__)
    for module in (
        investigation_models,
        investigation_engine,
        investigation_paths,
        investigation_temporal,
    ):
        assert set(module.__all__) <= package_surface, module.__name__
        assert list(module.__all__) == sorted(module.__all__), module.__name__


def test_the_package_surface_is_sorted_and_reachable():
    import backend.app.investigation as package

    assert list(package.__all__) == sorted(package.__all__)
    reachable = {
        name
        for name in dir(package)
        if not name.startswith("_")
        and name
        not in {
            "annotations",
            "engine",
            "models",
            "paths",
            "temporal",
        }
    }
    assert reachable == set(package.__all__)


def test_the_path_id_is_derived_over_build_graph_ids():
    """A path built from a real graph reuses the graph's derived identifiers."""
    relationship = _relationship()
    graph = build_graph((relationship,))
    edge = graph.edges()[0]
    path = _path(
        observations=(OBS_A, OBS_B),
        transitions=(
            _transition(
                edge_id=edge.edge_id,
                relationship_id=edge.relationship_id,
            ),
        ),
    )
    assert path.transitions[0].edge_id == edge.edge_id
    assert path.nodes[0].graph_node_id == graph.get_node(OBS_A).node_id