"""Step 8 -- the 11 signal adapters (SIH26146 - Step 8).

Every adapter reads one earlier-layer object and reports a typed, deterministic
number about the assessed subject.  These tests pin the three properties that
make the numbers publishable: the value is copied verbatim from the producing
layer, missing producers are refused a numeric value, and a measured zero is a
real answer that is honestly reported as PRESENT.
"""

from __future__ import annotations

import pytest

from backend.app.behavior.profiles import BehaviorProfileBuilder
from backend.app.correlation.enums import EvidenceType, InferenceLevel
from backend.app.domain.models import ObservationType
from backend.app.fusion import (
    FusionContext,
    FusionError,
    SignalFamily,
    SignalRole,
    SignalStatus,
    build_signals,
    derive_signal_id,
)
from backend.app.graph import build_graph, derive_node_id
from backend.app.investigation.models import (
    EvidenceSummary,
    InvestigationPath,
    PathNode,
    PathTemporalStatus,
    TimestampAwareness,
)
from backend.app.ml.anomaly import IsolationForestAnomalyDetector

from tests.test_step8_fusion import TZ, _context, _observation, _pipeline, _relationship

SIGNAL_TYPES = {
    "ANOMALY_NORMALIZED_SCORE",
    "BEHAVIORAL_FEATURE_COVERAGE",
    "CLUSTER_CENTROID_DISTANCE",
    "EVIDENCE_INCIDENT_COUNT",
    "EVIDENCE_TYPE_DIVERSITY",
    "PROVENANCE_CITED_EDGE_RATIO",
    "PROVENANCE_SOURCE_FILE_DIVERSITY",
    "TEMPORAL_COMPARABILITY",
    "TEMPORAL_FORWARD_FLOW",
    "PATH_REACHABILITY_FOUND",
    "PATH_STRUCTURE_COUNT",
}

#: Which families a signal type belongs to, for the coverage and ordering tests.
TYPE_FAMILY = {
    "ANOMALY_NORMALIZED_SCORE": SignalFamily.ANOMALY,
    "BEHAVIORAL_FEATURE_COVERAGE": SignalFamily.BEHAVIORAL,
    "CLUSTER_CENTROID_DISTANCE": SignalFamily.CLUSTER_CONTEXT,
    "EVIDENCE_INCIDENT_COUNT": SignalFamily.EVIDENCE_GRAPH,
    "EVIDENCE_TYPE_DIVERSITY": SignalFamily.EVIDENCE_GRAPH,
    "PROVENANCE_CITED_EDGE_RATIO": SignalFamily.PROVENANCE,
    "PROVENANCE_SOURCE_FILE_DIVERSITY": SignalFamily.PROVENANCE,
    "TEMPORAL_COMPARABILITY": SignalFamily.TEMPORAL,
    "TEMPORAL_FORWARD_FLOW": SignalFamily.TEMPORAL,
    "PATH_REACHABILITY_FOUND": SignalFamily.PATH_STRUCTURE,
    "PATH_STRUCTURE_COUNT": SignalFamily.PATH_STRUCTURE,
}

#: Which semantic role each signal type fulfills.
TYPE_ROLE = {
    "ANOMALY_NORMALIZED_SCORE": SignalRole.PRIORITY_SIGNAL,
    "BEHAVIORAL_FEATURE_COVERAGE": SignalRole.EVIDENCE_STRENGTH,
    "CLUSTER_CENTROID_DISTANCE": SignalRole.PRIORITY_SIGNAL,
    "EVIDENCE_INCIDENT_COUNT": SignalRole.PRIORITY_SIGNAL,
    "EVIDENCE_TYPE_DIVERSITY": SignalRole.EVIDENCE_STRENGTH,
    "PROVENANCE_CITED_EDGE_RATIO": SignalRole.EVIDENCE_STRENGTH,
    "PROVENANCE_SOURCE_FILE_DIVERSITY": SignalRole.EVIDENCE_STRENGTH,
    "TEMPORAL_COMPARABILITY": SignalRole.EVIDENCE_STRENGTH,
    "TEMPORAL_FORWARD_FLOW": SignalRole.CONTEXT_ONLY,
    "PATH_REACHABILITY_FOUND": SignalRole.PRIORITY_SIGNAL,
    "PATH_STRUCTURE_COUNT": SignalRole.PRIORITY_SIGNAL,
}

#: Which pipeline step produced a signal type.
TYPE_STEP = {
    "ANOMALY_NORMALIZED_SCORE": 6,
    "BEHAVIORAL_FEATURE_COVERAGE": 5,
    "CLUSTER_CENTROID_DISTANCE": 6,
    "EVIDENCE_INCIDENT_COUNT": 4,
    "EVIDENCE_TYPE_DIVERSITY": 4,
    "PROVENANCE_CITED_EDGE_RATIO": 4,
    "PROVENANCE_SOURCE_FILE_DIVERSITY": 4,
    "TEMPORAL_COMPARABILITY": 7,
    "TEMPORAL_FORWARD_FLOW": 7,
    "PATH_REACHABILITY_FOUND": 7,
    "PATH_STRUCTURE_COUNT": 7,
}


def _build(context):
    return build_signals(context)


def _signal_by_type(signals, signal_type):
    return next(signal for signal in signals if signal.signal_type == signal_type)


# --------------------------------------------------------------------------- #
# The full fixture
# --------------------------------------------------------------------------- #


class TestFullFixture:
    def test_exactly_eleven_typed_signals_are_emitted(self):
        signals = _build(_context())
        assert len(signals) == 11
        assert {signal.signal_type for signal in signals} == SIGNAL_TYPES

    def test_every_family_is_represented(self):
        signals = _build(_context())
        assert {signal.family for signal in signals} == set(SignalFamily)

    def test_signals_are_ordered_by_signal_id(self):
        signals = _build(_context())
        assert [signal.signal_id for signal in signals] == sorted(
            signal.signal_id for signal in signals
        )

    def test_each_signal_comes_from_its_own_step(self):
        signals = _build(_context())
        for signal in signals:
            assert signal.source_step == TYPE_STEP[signal.signal_type]

    def test_each_signal_names_the_threshold_profile_as_its_source(self):
        subject, _population, *_ = _pipeline()
        signals = _build(_context())
        assert all(signal.source_id == subject.profile_id for signal in signals)
        assert all(signal.subject_id == subject.profile_id for signal in signals)
        assert all(signal.signal_id == derive_signal_id(
            subject.profile_id, signal.signal_type, signal.source_step, subject.profile_id
        ) for signal in signals)

    def test_every_signal_traces_back_to_the_profile_in_its_lineage(self):
        subject, _population, *_ = _pipeline()
        signals = _build(_context())
        assert all(subject.profile_id in signal.lineage for signal in signals)

    def test_every_number_in_the_full_fixture_is_present(self):
        signals = _build(_context())
        assert all(signal.status is SignalStatus.PRESENT for signal in signals)
        assert all(signal.raw_value is not None for signal in signals)
        assert all(signal.normalized_value is not None for signal in signals)


# --------------------------------------------------------------------------- #
# Anomaly (Step 6)
# --------------------------------------------------------------------------- #


class TestAnomalySignal:
    def test_the_figure_is_carried_verbatim_without_renormalisation(self):
        subject, _population, _graph, run, _clustering, _paths = _pipeline()
        result = run.result_for(subject.profile_id)
        context = FusionContext(profile=subject, anomaly=run)
        signal = _signal_by_type(_build(context), "ANOMALY_NORMALIZED_SCORE")
        assert signal.status is SignalStatus.PRESENT
        assert signal.raw_value == float(result.raw_score)
        assert signal.normalized_value == float(result.normalized_score)
        assert (2.0 ** -result.raw_score) == pytest.approx(float(result.normalized_score))
        assert signal.family is SignalFamily.ANOMALY
        assert signal.source_step == 6

    def test_without_a_run_the_signal_is_absent(self):
        subject, _population, *_ = _pipeline()
        signal = _signal_by_type(_build(FusionContext(profile=subject)), "ANOMALY_NORMALIZED_SCORE")
        assert signal.status is SignalStatus.ABSENT
        assert signal.normalized_value is None
        assert signal.raw_value is None

    def test_a_never_fitted_run_is_unavailable(self):
        subject, _population, *_ = _pipeline()
        _model, run = IsolationForestAnomalyDetector().fit_score([])
        signal = _signal_by_type(
            _build(FusionContext(profile=subject, anomaly=run)), "ANOMALY_NORMALIZED_SCORE"
        )
        assert signal.status is SignalStatus.UNAVAILABLE
        assert signal.normalized_value is None

    def test_a_fitted_run_that_did_not_score_this_profile_is_unavailable(self):
        subject, _population, _graph, run, _clustering, _paths = _pipeline()
        outsider = BehaviorProfileBuilder().build(
            [
                _observation("obs-outsider-00", 10, "tx-out-0"),
                _observation("obs-outsider-01", 11, "tx-out-1"),
            ],
            group_id="grp-outsider",
        )
        signal = _signal_by_type(
            _build(FusionContext(profile=outsider, anomaly=run)), "ANOMALY_NORMALIZED_SCORE"
        )
        assert signal.status is SignalStatus.UNAVAILABLE


# --------------------------------------------------------------------------- #
# Behavioural (Step 5)
# --------------------------------------------------------------------------- #


class TestBehavioralSignal:
    def test_coverage_is_the_share_of_measured_categories(self):
        subject, _population, *_ = _pipeline()
        assert subject.feature_count > 0
        signal = _signal_by_type(_build(FusionContext(profile=subject)), "BEHAVIORAL_FEATURE_COVERAGE")
        assert signal.status is SignalStatus.PRESENT
        assert signal.family is SignalFamily.BEHAVIORAL
        assert signal.source_step == 5
        measured = {
            feature.category
            for feature in subject.features
            if feature.status.value != "ABSENT"
        }
        assert signal.raw_value == float(len(measured))
        assert signal.normalized_value == float(len(measured)) / 7.0
        assert signal.context == (
            ("feature_category_total", "7"),
            ("feature_count", str(subject.feature_count)),
            ("measured_categories", str(len(measured))),
        )

    def test_the_profile_is_always_measured(self):
        subject, _population, *_ = _pipeline()
        signal = _signal_by_type(_build(FusionContext(profile=subject)), "BEHAVIORAL_FEATURE_COVERAGE")
        assert 0.0 <= signal.normalized_value <= 1.0


# --------------------------------------------------------------------------- #
# Cluster context (Step 6)
# --------------------------------------------------------------------------- #


class TestClusterSignal:
    def test_the_figure_is_geometric_context_not_membership(self):
        subject, _population, _graph, _anomaly, run, _paths = _pipeline()
        assignment = run.cluster_of(subject.profile_id)
        signal = _signal_by_type(
            _build(FusionContext(profile=subject, clustering=run)), "CLUSTER_CENTROID_DISTANCE"
        )
        assert signal.status is SignalStatus.PRESENT
        assert signal.raw_value == float(assignment.distance_to_centroid)
        assert 0.0 <= signal.normalized_value <= 1.0
        context = dict(signal.context)
        assert context["cluster_id"] == assignment.cluster_id
        assert context["normalization"].startswith("distance / max")

    def test_without_a_run_the_signal_is_absent(self):
        subject, _population, *_ = _pipeline()
        signal = _signal_by_type(
            _build(FusionContext(profile=subject)), "CLUSTER_CENTROID_DISTANCE"
        )
        assert signal.status is SignalStatus.ABSENT
        assert signal.raw_value is None

    def test_a_fitted_run_that_did_not_cluster_this_profile_is_unavailable(self):
        subject, _population, _graph, _anomaly, run, _paths = _pipeline()
        outsider = BehaviorProfileBuilder().build(
            [
                _observation("obs-outsider-00", 10, "tx-out-0"),
                _observation("obs-outsider-01", 11, "tx-out-1"),
            ],
            group_id="grp-outsider",
        )
        signal = _signal_by_type(
            _build(FusionContext(profile=outsider, clustering=run)), "CLUSTER_CENTROID_DISTANCE"
        )
        assert signal.status is SignalStatus.UNAVAILABLE


# --------------------------------------------------------------------------- #
# Evidence graph (Step 4C)
# --------------------------------------------------------------------------- #


class TestEvidenceSignals:
    def test_without_a_graph_both_signals_are_absent(self):
        subject, _population, *_ = _pipeline()
        signals = _build(FusionContext(profile=subject))
        assert _signal_by_type(signals, "EVIDENCE_INCIDENT_COUNT").status is SignalStatus.ABSENT
        assert _signal_by_type(signals, "EVIDENCE_TYPE_DIVERSITY").status is SignalStatus.ABSENT
        for signal in signals:
            if signal.family is SignalFamily.EVIDENCE_GRAPH:
                assert signal.normalized_value is None

    def test_incident_edges_are_counted_and_capped(self):
        subject, _population, graph, *_ = _pipeline()
        signal = _signal_by_type(
            _build(FusionContext(profile=subject, graph=graph)), "EVIDENCE_INCIDENT_COUNT"
        )
        assert signal.status is SignalStatus.PRESENT
        expected = 2  # obs-000-00 and obs-000-01 each carry one incident edge
        assert signal.raw_value == float(expected)
        assert signal.normalized_value == float(min(1.0, expected / 20.0))
        assert signal.evidence_refs

    def test_diversity_is_the_share_of_observed_evidence_types(self):
        subject, _population, graph, *_ = _pipeline()
        signal = _signal_by_type(
            _build(FusionContext(profile=subject, graph=graph)), "EVIDENCE_TYPE_DIVERSITY"
        )
        assert signal.status is SignalStatus.PRESENT
        assert signal.normalized_value == pytest.approx(2.0 / 8.0)

    def test_an_empty_measured_set_is_present_zero(self):
        subject, _population, *_ = _pipeline()
        graph = build_graph(
            (_relationship("obs-100-00", "obs-200-00", EvidenceType.SAME_IP, InferenceLevel.OBSERVED),)
        )
        signals = _build(FusionContext(profile=subject, graph=graph))
        count = _signal_by_type(signals, "EVIDENCE_INCIDENT_COUNT")
        diversity = _signal_by_type(signals, "EVIDENCE_TYPE_DIVERSITY")
        assert count.status is SignalStatus.PRESENT
        assert count.normalized_value == 0.0
        assert diversity.status is SignalStatus.PRESENT
        assert diversity.normalized_value == 0.0


# --------------------------------------------------------------------------- #
# Provenance (Steps 2/4A)
# --------------------------------------------------------------------------- #


class TestProvenanceSignals:
    def test_without_a_graph_both_signals_are_absent(self):
        subject, _population, *_ = _pipeline()
        signals = _build(FusionContext(profile=subject))
        assert _signal_by_type(signals, "PROVENANCE_CITED_EDGE_RATIO").status is SignalStatus.ABSENT
        assert _signal_by_type(signals, "PROVENANCE_SOURCE_FILE_DIVERSITY").status is SignalStatus.ABSENT

    def test_the_citation_ratio_counts_cited_incident_edges(self):
        subject, _population, graph, *_ = _pipeline()
        signal = _signal_by_type(
            _build(FusionContext(profile=subject, graph=graph)), "PROVENANCE_CITED_EDGE_RATIO"
        )
        assert signal.status is SignalStatus.PRESENT
        assert signal.normalized_value == 1.0  # every fixture edge carries a citation
        assert all(ref.source_file for ref in signal.provenance_refs)

    def test_source_file_diversity_uses_the_disclosed_cap(self):
        subject, _population, graph, *_ = _pipeline()
        signal = _signal_by_type(
            _build(FusionContext(profile=subject, graph=graph)), "PROVENANCE_SOURCE_FILE_DIVERSITY"
        )
        assert signal.status is SignalStatus.PRESENT
        assert signal.normalized_value == pytest.approx(1.0 / 10.0)


# --------------------------------------------------------------------------- #
# Temporal structure (Step 7)
# --------------------------------------------------------------------------- #


class TestTemporalSignals:
    def test_forward_moving_paths_are_comparable_and_flow_forward(self):
        subject, _population, _graph, _anomaly, _clustering, paths = _pipeline()
        signals = _build(FusionContext(profile=subject, paths=paths))
        comparability = _signal_by_type(signals, "TEMPORAL_COMPARABILITY")
        forward = _signal_by_type(signals, "TEMPORAL_FORWARD_FLOW")
        assert comparability.status is SignalStatus.PRESENT
        assert comparability.normalized_value == 1.0
        assert forward.status is SignalStatus.PRESENT
        assert forward.normalized_value == 1.0
        assert forward.evidence_refs

    def test_without_paths_both_signals_are_absent(self):
        subject, _population, *_ = _pipeline()
        signals = _build(FusionContext(profile=subject))
        assert _signal_by_type(signals, "TEMPORAL_COMPARABILITY").status is SignalStatus.ABSENT
        assert _signal_by_type(signals, "TEMPORAL_FORWARD_FLOW").status is SignalStatus.ABSENT

    def test_paths_that_touch_no_subject_observation_are_unavailable(self):
        subject, _population, *_ = _pipeline()
        signals = _build(FusionContext(profile=subject, paths=()))
        assert _signal_by_type(signals, "TEMPORAL_COMPARABILITY").status is SignalStatus.UNAVAILABLE
        assert _signal_by_type(signals, "TEMPORAL_FORWARD_FLOW").status is SignalStatus.UNAVAILABLE

    def test_a_zero_hop_path_makes_temporal_signals_not_applicable(self):
        subject, _population, *_ = _pipeline()
        observation_id = subject.observation_ids[0]
        node = PathNode(
            observation_id=observation_id,
            graph_node_id=derive_node_id(observation_id),
            timestamp=TZ,
            timestamp_awareness=TimestampAwareness.AWARE,
        )
        path = InvestigationPath(
            path_id="path-single-node",
            nodes=(node,),
            transitions=(),
            hop_count=0,
            first_timestamp=TZ,
            last_timestamp=TZ,
            duration_seconds=0.0,
            temporal_status=PathTemporalStatus.UNKNOWN,
            evidence_summary=EvidenceSummary(edge_count=0),
        )
        signals = _build(FusionContext(profile=subject, paths=(path,)))
        assert _signal_by_type(signals, "TEMPORAL_COMPARABILITY").status is SignalStatus.NOT_APPLICABLE
        assert _signal_by_type(signals, "TEMPORAL_FORWARD_FLOW").status is SignalStatus.NOT_APPLICABLE


# --------------------------------------------------------------------------- #
# Path structure (Step 7)
# --------------------------------------------------------------------------- #


class TestPathSignals:
    def test_a_found_trail_is_a_measured_one(self):
        subject, _population, _graph, _anomaly, _clustering, paths = _pipeline()
        signals = _build(FusionContext(profile=subject, paths=paths))
        reachability = _signal_by_type(signals, "PATH_REACHABILITY_FOUND")
        structure = _signal_by_type(signals, "PATH_STRUCTURE_COUNT")
        assert reachability.status is SignalStatus.PRESENT
        assert reachability.normalized_value == 1.0
        assert structure.status is SignalStatus.PRESENT
        assert structure.normalized_value == pytest.approx(1.0 / 10.0)
        assert structure.evidence_refs

    def test_without_paths_both_signals_are_absent(self):
        subject, _population, *_ = _pipeline()
        signals = _build(FusionContext(profile=subject))
        assert _signal_by_type(signals, "PATH_REACHABILITY_FOUND").status is SignalStatus.ABSENT
        assert _signal_by_type(signals, "PATH_STRUCTURE_COUNT").status is SignalStatus.ABSENT

    def test_a_supplied_search_that_found_nothing_is_a_present_zero(self):
        subject, _population, *_ = _pipeline()
        signals = _build(FusionContext(profile=subject, paths=()))
        assert _signal_by_type(signals, "PATH_REACHABILITY_FOUND").status is SignalStatus.PRESENT
        assert _signal_by_type(signals, "PATH_REACHABILITY_FOUND").normalized_value == 0.0
        assert _signal_by_type(signals, "PATH_STRUCTURE_COUNT").status is SignalStatus.PRESENT
        assert _signal_by_type(signals, "PATH_STRUCTURE_COUNT").normalized_value == 0.0


# --------------------------------------------------------------------------- #
# Adapter surface discipline
# --------------------------------------------------------------------------- #


class TestSurfaceDiscipline:
    def test_non_positive_caps_are_refused(self):
        with pytest.raises(FusionError):
            build_signals(_context(), evidence_count_cap=0)
        with pytest.raises(FusionError):
            build_signals(_context(), path_count_cap=False)
        with pytest.raises(FusionError):
            build_signals(_context(), provenance_file_cap=-1)

    def test_duplicate_ids_are_a_collision_not_a_silent_addition(self):
        subject, _population, *_ = _pipeline()
        first = build_signals(FusionContext(profile=subject))
        second = build_signals(FusionContext(profile=subject))
        assert [signal.signal_id for signal in first] == [
            signal.signal_id for signal in second
        ]

    def test_context_must_be_a_fusion_context(self):
        with pytest.raises(FusionError):
            build_signals(None)
        with pytest.raises(FusionError):
            build_signals(object())  # type: ignore[arg-type]

    def test_a_non_graph_is_refused_by_incident_edges(self):
        subject, _population, *_ = _pipeline()
        with pytest.raises(FusionError):
            build_signals(FusionContext(profile=subject, graph=object()))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Missing is never zero
# --------------------------------------------------------------------------- #


class TestMissingIsNotZero:
    def test_non_present_signals_carry_no_number_in_any_absent_scenario(self):
        subject, _population, *_ = _pipeline()
        for context in (
            FusionContext(profile=subject),
            FusionContext(profile=subject, paths=()),  # type: ignore[arg-type]
        ):
            for signal in build_signals(context):
                if signal.status is not SignalStatus.PRESENT:
                    assert signal.raw_value is None
                    assert signal.normalized_value is None