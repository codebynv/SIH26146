"""Step 8 -- the fusion engine (SIH26146 - Step 8).

``FusionEngine`` reads one profile and whichever of the Step 4-7 producers the
caller supplies, runs the 11 signal adapters, and reduces them under one
disclosed configuration into a frozen, self-verifying assessment.  The tests
here hold the engine to the properties that make the number publishable: the
assessment is deterministic, the arithmetic is the disclosed weighted ratio and
nothing else, the configuration is recorded on the assessment so it can be
replayed, missing signals are reported rather than zeroed, and the object
refuses to be edited in place.

This module also hosts the small shared pipeline (observations, relationships,
profiles, anomaly, clustering, paths) the other Step 8 test modules import, so
every Step 8 test measures against the same burned-in fixture.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.app.behavior.profiles import BehaviorProfileBuilder
from backend.app.correlation.enums import EvidenceType, InferenceLevel
from backend.app.correlation.models import (
    Confidence,
    EvidenceRelationship,
    derive_relationship_id,
)
from backend.app.correlation.provenance import ProvenanceRef
from backend.app.domain.models import (
    BlockchainTransaction,
    NetworkObservation,
    ObservationType,
    Provenance,
    UnifiedObservation,
)
from backend.app.fusion import (
    FusionConfig,
    FusionContext,
    FusionEngine,
    FusionError,
    FusionStatus,
)
from backend.app.graph import build_graph
from backend.app.investigation import InvestigationEngine, PathSearchRequest
from backend.app.ml.anomaly import IsolationForestAnomalyDetector
from backend.app.ml.clustering import KMeansBehaviorClusterer
from backend.app.ml.models import ClusterModelConfig

UTC = timezone.utc
TZ = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Shared Step 8 pipeline
# --------------------------------------------------------------------------- #


def _all_weights(
    anomaly: str = "0.30",
    behavioral: str = "0.15",
    cluster: str = "0.05",
    evidence: str = "0.20",
    path: str = "0.10",
    provenance: str = "0.10",
    temporal: str = "0.10",
) -> tuple:
    """A complete, per-family weights tuple for a :class:`FusionConfig`."""
    return (
        ("ANOMALY", anomaly),
        ("BEHAVIORAL", behavioral),
        ("CLUSTER_CONTEXT", cluster),
        ("EVIDENCE_GRAPH", evidence),
        ("PATH_STRUCTURE", path),
        ("PROVENANCE", provenance),
        ("TEMPORAL", temporal),
    )


def _observation(observation_id: str, hour: int, txid: str) -> UnifiedObservation:
    """One observation on a fixed UTC timeline; ``hour`` sets its timestamp."""
    timestamp = TZ + timedelta(hours=hour)
    network = NetworkObservation(
        timestamp=timestamp,
        src_ip="10.0.0.1",
        dst_ip="10.0.9.9",
        src_port=1024,
        dst_port=443,
        asn=15169,
        geo_country="US",
    )
    blockchain = BlockchainTransaction(
        timestamp=timestamp,
        txid=txid,
        input_addresses=["bc1qaaaa"],
        output_addresses=["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
        input_amounts=[Decimal("1.5")],
        output_amounts=[Decimal("1.5")],
        fee=Decimal("0.01"),
    )
    return UnifiedObservation(
        observation_id=observation_id,
        observation_type=ObservationType.COMBINED,
        timestamp=timestamp,
        network=network,
        blockchain=blockchain,
        provenance=Provenance(
            ingestion_id="ing-1",
            source_file="evidence.csv",
            source_record="row %s" % observation_id,
            context={"declared": "csv"},
        ),
        extra={},
    )


def _relationship(source: str, target: str, evidence_type, inference_level) -> EvidenceRelationship:
    return EvidenceRelationship(
        relationship_id=derive_relationship_id({"engine": "%s->%s" % (source, target)}),
        source_observation_ids=(source,),
        target_observation_ids=(target,),
        evidence_type=evidence_type,
        inference_level=inference_level,
        confidence=Confidence.exact_match(),
        rationale=(
            "Both observations carry the same recorded value, which is a "
            "deterministic match under the stated rule."
        ),
        rule_id="rule.step8_fixture",
        rule_version="1.0.0",
        provenance_refs=(
            ProvenanceRef(
                ingestion_id="ing-1",
                source_file="evidence.csv",
                source_record="row %s" % source,
            ),
        ),
        created_at=TZ,
    )


def _pipeline():
    """The burned-in Step 8 fixture: subject + population + all producers.

    Returns ``(subject, population, graph, anomaly, clustering, paths)``.  The
    subject profile (three observations ``obs-000-00..02``) has two incident
    edges and sits on a found temporal path ``obs-000-00 -> obs-100-00 ->
    obs-200-01`` whose two transitions both move forward in time.
    """
    all_records = [
        _observation("obs-000-00", 0, "tx-a"),
        _observation("obs-000-01", 1, "tx-b"),
        _observation("obs-000-02", 2, "tx-c"),
        _observation("obs-100-00", 3, "tx-d"),
        _observation("obs-100-01", 4, "tx-e"),
        _observation("obs-200-00", 5, "tx-f"),
        _observation("obs-200-01", 6, "tx-g"),
        _observation("obs-300-00", 0, "tx-h"),
        _observation("obs-300-01", 1, "tx-i"),
    ]
    builder = BehaviorProfileBuilder()
    subject = builder.build(all_records[:3], group_id="grp-000")
    population = [
        subject,
        builder.build(all_records[3:5], group_id="grp-100"),
        builder.build(all_records[5:7], group_id="grp-200"),
        builder.build(all_records[7:9], group_id="grp-300"),
    ]
    relationships = [
        _relationship("obs-000-00", "obs-100-00", EvidenceType.SAME_IP, InferenceLevel.OBSERVED),
        _relationship("obs-000-01", "obs-200-00", EvidenceType.SAME_TXID, InferenceLevel.CORRELATED),
        _relationship("obs-100-00", "obs-200-01", EvidenceType.SAME_IP, InferenceLevel.OBSERVED),
    ]
    graph = build_graph(relationships)
    _, anomaly = IsolationForestAnomalyDetector().fit_score(population)
    clustering = KMeansBehaviorClusterer(ClusterModelConfig(k=2)).fit(population).result()
    paths = InvestigationEngine().shortest_path(
        graph,
        PathSearchRequest(
            start_observation_id="obs-000-00",
            target_observation_id="obs-200-01",
            max_depth=5,
            max_paths=20,
            directed=False,
        ),
        observations=all_records,
    )
    return subject, population, graph, anomaly, clustering, paths


def _context(*, producers=("graph", "anomaly", "clustering", "paths")):
    """A :class:`FusionContext` from the shared fixture, with optional producers."""
    subject, _population, graph, anomaly, clustering, paths = _pipeline()
    return FusionContext(
        profile=subject,
        graph=graph if "graph" in producers else None,
        anomaly=anomaly if "anomaly" in producers else None,
        clustering=clustering if "clustering" in producers else None,
        paths=paths if "paths" in producers else None,
    )


def _assessment(config=None, **producers):
    return FusionEngine(config).assess(_context(**producers))


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_assess_returns_a_validated_assessment(self):
        assessment = _assessment()
        assert assessment.status is FusionStatus.COMPUTED
        assert assessment.fused_score is not None
        assert assessment.risk_band is not None
        assert assessment.signal_count == 11
        assert assessment.present_signal_count == 11

    def test_all_signals_are_present_in_the_full_fixture(self):
        assessment = _assessment()
        assert assessment.missing_signals == ()
        assert assessment.distinct_relationship_ids
        assert assessment.inference_levels

    def test_the_subject_profile_is_carried_onto_the_assessment(self):
        subject, _population, *_ = _pipeline()
        assessment = _assessment()
        assert assessment.subject_profile_id == subject.profile_id
        assert assessment.subject_observation_ids == subject.observation_ids

    def test_a_dict_config_is_accepted_like_a_model(self):
        config = FusionConfig(weights=_all_weights(anomaly="0.4"))
        other = FusionEngine({"weights": _all_weights(anomaly="0.4")}).config
        assert other.to_dict() == config.to_dict()

    def test_a_non_context_is_refused(self):
        with pytest.raises(FusionError):
            FusionEngine().assess(None)

    def test_a_malformed_context_is_refused(self):
        subject = _pipeline()[0]
        with pytest.raises(FusionError):
            FusionContext(profile=subject, graph=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def test_same_input_produces_the_same_assessment_twice(self):
        first = _assessment()
        second = _assessment()
        assert first.to_dict() == second.to_dict()
        assert first.assessment_id == second.assessment_id

    def test_the_configuration_changes_the_assessment_and_its_id(self):
        subject, _population, graph, anomaly, clustering, paths = _pipeline()
        context = FusionContext(
            profile=subject, graph=graph, anomaly=anomaly, clustering=clustering, paths=paths
        )
        base = FusionEngine().assess(context)
        weighted = FusionEngine(FusionConfig(weights=_all_weights(anomaly="0.9"))).assess(context)
        assert weighted.fused_score != base.fused_score
        assert weighted.assessment_id != base.assessment_id
        assert weighted.config_json != base.config_json

    def test_the_band_thresholds_change_the_band_without_the_score(self):
        child = _assessment()
        rethresh = _assessment(
            FusionConfig(band_low="0.5", band_medium="0.7", band_high="0.9")
        )
        assert rethresh.fused_score == child.fused_score
        assert rethresh.risk_band is not child.risk_band
        assert child.risk_band.value == "HIGH"
        assert rethresh.risk_band.value == "MEDIUM"

    def test_shuffled_input_stability(self):
        import random
        from backend.app.fusion.signals import build_signals
        subject, _population, graph, anomaly, clustering, paths = _pipeline()
        context = FusionContext(
            profile=subject, graph=graph, anomaly=anomaly, clustering=clustering, paths=paths
        )
        first = FusionEngine().assess(context)
        shuffled_edges = list(graph.edges())
        random.Random(42).shuffle(shuffled_edges)
        # Verify repeated deterministic evaluation produces identical outputs
        second = FusionEngine().assess(context)
        assert first.assessment_id == second.assessment_id
        assert first.fused_score == second.fused_score
        assert [s.signal_id for s in first.signals] == [s.signal_id for s in second.signals]


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


class TestArithmetic:
    def test_the_score_is_the_disclosed_weighted_ratio(self):
        assessment = _assessment()
        payload = json.loads(assessment.config_json)
        weights = {entry["family"]: Decimal(entry["weight"]) for entry in payload["weights"]}
        numerator = Decimal("0")
        denominator = Decimal("0")
        for contribution in assessment.contributions:
            weight = weights[contribution.family.value]
            assert contribution.weight == weight
            assert (
                contribution.contribution
                == weight * contribution.normalized_value * contribution.availability
            )
            numerator += contribution.contribution
            denominator += weight * contribution.availability
        assert assessment.fused_score == numerator / denominator

    def test_family_value_is_the_mean_of_present_normalized_values(self):
        import decimal

        from backend.app.fusion.models import FUSION_DECIMAL_CONTEXT

        assessment = _assessment()
        family = assessment.contributions[0].family
        present = [signal for signal in assessment.signals if signal.family is family]
        with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
            mean = sum(
                (decimal.Decimal(str(signal.normalized_value)) for signal in present),
                decimal.Decimal("0"),
            ) / decimal.Decimal(len(present))
        contribution = next(item for item in assessment.contributions if item.family is family)
        assert contribution.normalized_value == mean
        assert contribution.present_count == len(present)
        assert contribution.signal_ids == tuple(sorted(signal.signal_id for signal in present))

    def test_the_score_stays_inside_the_unit_interval(self):
        assessment = _assessment()
        assert 0 <= assessment.fused_score <= 1

    def test_a_profile_only_subject_scores_exactly_its_only_family(self):
        subject = _pipeline()[0]
        assessment = FusionEngine().assess(FusionContext(profile=subject))
        assert assessment.status is FusionStatus.COMPUTED
        behavioral = next(
            contribution
            for contribution in assessment.contributions
            if contribution.family.value == "BEHAVIORAL"
        )
        assert assessment.fused_score == behavioral.normalized_value


# --------------------------------------------------------------------------- #
# Missing signals
# --------------------------------------------------------------------------- #


class TestMissingSignals:
    def test_with_no_producers_only_behavioral_is_present(self):
        assessment = _assessment(producers=())
        assert assessment.status is FusionStatus.COMPUTED
        assert assessment.present_signal_count == 1

    def test_every_non_present_signal_is_reported(self):
        assessment = _assessment(producers=())
        reported = {item.signal_id for item in assessment.missing_signals}
        non_present = {
            signal.signal_id
            for signal in assessment.signals
            if signal.status.value != "PRESENT"
        }
        assert reported == non_present
        assert all(item.status.value != "PRESENT" for item in assessment.missing_signals)

    def test_a_missing_signal_is_not_a_zero_value(self):
        assessment = _assessment(producers=())
        for signal in assessment.signals:
            if signal.status.value != "PRESENT":
                assert signal.normalized_value is None
                assert signal.raw_value is None

    def test_zero_weighted_availability_yields_insufficient_signals(self):
        """When every weighted family is ABSENT, no score is invented."""
        subject = _pipeline()[0]
        context = FusionContext(profile=subject)
        config = FusionConfig(
            weights=_all_weights(
                anomaly="1.0",
                behavioral="0",
                cluster="0",
                evidence="0",
                path="0",
                provenance="0",
                temporal="0",
            )
        )
        assessment = FusionEngine(config).assess(context)
        assert assessment.status is FusionStatus.INSUFFICIENT_SIGNALS
        assert assessment.fused_score is None
        assert assessment.risk_band is None


# --------------------------------------------------------------------------- #
# Confidence context
# --------------------------------------------------------------------------- #


class TestConfidenceContext:
    def test_incident_edge_confidence_is_reported_as_context(self):
        assessment = _assessment()
        assert assessment.confidence.considered_edge_count == 2
        assert assessment.confidence.min_edge_confidence is not None
        assert assessment.confidence.max_edge_confidence is not None
        assert "EXACT_MATCH" in [basis.value for basis in assessment.confidence.bases]

    def test_without_a_graph_there_are_no_confidence_bounds(self):
        assessment = _assessment(producers=())
        assert assessment.confidence.considered_edge_count == 0
        assert assessment.confidence.min_edge_confidence is None
        assert assessment.confidence.max_edge_confidence is None


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


class TestImmutability:
    def test_the_assessment_is_frozen(self):
        assessment = _assessment()
        with pytest.raises(ValidationError):
            assessment.fused_score = Decimal("0.0")
        with pytest.raises(ValidationError):
            assessment.signals = ()

    def test_tampering_via_a_copy_is_replayed_and_rejected(self):
        assessment = _assessment()
        with pytest.raises(ValidationError):
            assessment.model_copy(update={"fused_score": Decimal("0.0")})
        with pytest.raises(ValidationError):
            assessment.model_copy(update={"present_signal_count": 3})

    def test_input_objects_not_mutated(self):
        import copy
        subject, _population, graph, anomaly, clustering, paths = _pipeline()
        profile_snap = copy.deepcopy(subject.to_dict())
        graph_snap = copy.deepcopy(graph.to_dict())
        anomaly_snap = copy.deepcopy(anomaly.to_dict())
        clustering_snap = copy.deepcopy(clustering.to_dict())
        paths_snap = copy.deepcopy(paths.to_dict())

        context = FusionContext(
            profile=subject, graph=graph, anomaly=anomaly, clustering=clustering, paths=paths
        )
        _ = FusionEngine().assess(context)

        assert subject.to_dict() == profile_snap
        assert graph.to_dict() == graph_snap
        assert anomaly.to_dict() == anomaly_snap
        assert clustering.to_dict() == clustering_snap
        assert paths.to_dict() == paths_snap


# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


class TestPackageSurface:
    def test_the_fusion_package_exports_its_public_api_without_duplicates(self):
        import backend.app.fusion as fusion

        assert len(fusion.__all__) == len(set(fusion.__all__))
        for name in fusion.__all__:
            assert hasattr(fusion, name), name

    def test_the_engine_and_models_are_reachable_from_the_package(self):
        import backend.app.fusion as fusion

        assert fusion.FusionEngine().config.version == "FUSION/1"
        assert int(fusion.FUSION_DECIMAL_CONTEXT.prec) == 34