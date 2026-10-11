"""Step 8 -- FINAL MONOTONICITY GATE (SIH26146 - Step 8).

Semantic monotonicity tests.  Each test encodes one invariant that the fusion
engine *must* satisfy to be trustworthy:

    adding more (or stronger) evidence cannot decrease the fused score

Invariants tested
-----------------
M01  Adding one incident evidence edge never decreases the score.
M02  A second distinct evidence type never decreases the score.
M03  Adding a provenance citation never decreases the score.
M04  A second distinct source file never decreases the score.
M05  Adding a reachable investigation path never decreases the score.
M06  A second investigation path never decreases the score.
M07  A higher anomaly score never decreases the fused score.
M08  Greater behavioral coverage never decreases the score.
M09  Going from ABSENT graph to a present graph never decreases the score.
M10  Going from ABSENT paths to a found trail never decreases the score.
M11  Supplying every producer never decreases the profile-only score.

Notes on ">=" throughout
-------------------------
Because CONTEXT_ONLY signals (TEMPORAL_FORWARD_FLOW) do not enter the weighted
numerator or denominator, additions that only affect that signal cannot change
the fused score; all tests use >= (non-decreasing) rather than > (strictly
increasing).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Sequence

import pytest

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
    FusionStatus,
)
from backend.app.graph import build_graph
from backend.app.investigation import InvestigationEngine, PathSearchRequest
from backend.app.ml.anomaly import IsolationForestAnomalyDetector

UTC = timezone.utc
_T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


def _obs(obs_id: str, hour: int, txid: str) -> UnifiedObservation:
    ts = _T0 + timedelta(hours=hour)
    return UnifiedObservation(
        observation_id=obs_id,
        observation_type=ObservationType.COMBINED,
        timestamp=ts,
        network=NetworkObservation(
            timestamp=ts,
            src_ip="10.0.0.1",
            dst_ip="10.0.9.9",
            src_port=1024,
            dst_port=443,
            asn=15169,
            geo_country="US",
        ),
        blockchain=BlockchainTransaction(
            timestamp=ts,
            txid=txid,
            input_addresses=["bc1qaaaa"],
            output_addresses=["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
            input_amounts=[Decimal("1.0")],
            output_amounts=[Decimal("1.0")],
            fee=Decimal("0.001"),
        ),
        provenance=Provenance(
            ingestion_id="mono-ing",
            source_file="mono.csv",
            source_record="row-" + obs_id,
            context={},
        ),
        extra={},
    )


def _rel(
    src: str,
    tgt: str,
    evidence_type: EvidenceType = EvidenceType.SAME_IP,
    inference_level: InferenceLevel = InferenceLevel.OBSERVED,
    source_file: str = "mono.csv",
) -> EvidenceRelationship:
    return EvidenceRelationship(
        relationship_id=derive_relationship_id({"mono": "%s->%s" % (src, tgt)}),
        source_observation_ids=(src,),
        target_observation_ids=(tgt,),
        evidence_type=evidence_type,
        inference_level=inference_level,
        confidence=Confidence.exact_match(),
        rationale=(
            "Monotonicity fixture: both observations share the stated attribute, "
            "which is a deterministic match under the stated rule."
        ),
        rule_id="rule.monotonicity_fixture",
        rule_version="1.0.0",
        provenance_refs=(
            ProvenanceRef(
                ingestion_id="mono-ing",
                source_file=source_file,
                source_record="row-" + src,
            ),
        ),
        created_at=_T0,
    )


def _profile(obs_ids: Sequence[str], group_id: str):
    obs = [_obs(oid, i, "tx-%s-%d" % (group_id, i)) for i, oid in enumerate(obs_ids)]
    return BehaviorProfileBuilder().build(list(obs), group_id=group_id)


def _score(context: FusionContext, config: Optional[FusionConfig] = None) -> Decimal:
    """Return the fused score, or Decimal(0) when INSUFFICIENT_SIGNALS."""
    assessment = FusionEngine(config).assess(context)
    if assessment.status is FusionStatus.COMPUTED:
        return assessment.fused_score
    return Decimal("0")


# ---------------------------------------------------------------------------
# M01 -- Adding one incident evidence edge never decreases the score
# ---------------------------------------------------------------------------


class TestM01AddingOneEdgeNeverDecreases:
    """M01: going from 0 incident edges to 1 incident edge is non-decreasing."""

    def test_m01_one_edge_score_geq_zero_edge_score(self):
        subject_obs = ["m01-a", "m01-b"]
        population_obs = ["m01-c", "m01-d", "m01-e", "m01-f"]

        subject = _profile(subject_obs, "m01-subject")
        pop_profiles = [
            subject,
            _profile(population_obs[:2], "m01-pop1"),
            _profile(population_obs[2:], "m01-pop2"),
        ]
        _, anomaly = IsolationForestAnomalyDetector().fit_score(pop_profiles)

        # Baseline: edge not touching subject
        graph_no_incident = build_graph((_rel("m01-c", "m01-e"),))
        ctx_no_edge = FusionContext(profile=subject, graph=graph_no_incident, anomaly=anomaly)

        # Augmented: add one incident edge on the subject
        graph_one_incident = build_graph(
            (_rel("m01-c", "m01-e"), _rel("m01-a", "m01-c"))
        )
        ctx_one_edge = FusionContext(profile=subject, graph=graph_one_incident, anomaly=anomaly)

        score_before = _score(ctx_no_edge)
        score_after = _score(ctx_one_edge)

        assert score_after >= score_before, (
            "M01 FAILED: adding one incident edge decreased fused score "
            "from %s to %s" % (score_before, score_after)
        )


# ---------------------------------------------------------------------------
# M02 -- A second distinct evidence type never decreases the score
# ---------------------------------------------------------------------------


class TestM02SecondEvidenceTypeNeverDecreases:
    """M02: adding a second distinct evidence type (higher diversity) is non-decreasing."""

    def test_m02_two_types_score_geq_one_type_score(self):
        subject_obs = ["m02-a", "m02-b"]
        other_obs = ["m02-c", "m02-d", "m02-e", "m02-f"]

        subject = _profile(subject_obs, "m02-subject")
        pop_profiles = [
            subject,
            _profile(other_obs[:2], "m02-pop1"),
            _profile(other_obs[2:], "m02-pop2"),
        ]
        _, anomaly = IsolationForestAnomalyDetector().fit_score(pop_profiles)

        graph_one_type = build_graph((_rel("m02-a", "m02-c", EvidenceType.SAME_IP),))
        ctx_one = FusionContext(profile=subject, graph=graph_one_type, anomaly=anomaly)

        graph_two_types = build_graph(
            (
                _rel("m02-a", "m02-c", EvidenceType.SAME_IP),
                _rel("m02-b", "m02-d", EvidenceType.SAME_TXID),
            )
        )
        ctx_two = FusionContext(profile=subject, graph=graph_two_types, anomaly=anomaly)

        score_before = _score(ctx_one)
        score_after = _score(ctx_two)

        assert score_after >= score_before, (
            "M02 FAILED: adding a second distinct evidence type decreased fused score "
            "from %s to %s" % (score_before, score_after)
        )


# ---------------------------------------------------------------------------
# M03 -- Adding a provenance citation never decreases the score
# ---------------------------------------------------------------------------


class TestM03AddingCitationNeverDecreases:
    """M03: a cited edge has citation_ratio >= an uncited edge."""

    def test_m03_cited_edge_score_geq_uncited_edge_score(self):
        subject_obs = ["m03-a", "m03-b"]
        other_obs = ["m03-c", "m03-d", "m03-e", "m03-f"]

        subject = _profile(subject_obs, "m03-subject")
        pop_profiles = [
            subject,
            _profile(other_obs[:2], "m03-pop1"),
            _profile(other_obs[2:], "m03-pop2"),
        ]
        _, anomaly = IsolationForestAnomalyDetector().fit_score(pop_profiles)

        # Baseline: incident edge with NO provenance citation
        rel_uncited = EvidenceRelationship(
            relationship_id=derive_relationship_id({"mono_uncited": "m03-b->m03-d"}),
            source_observation_ids=("m03-b",),
            target_observation_ids=("m03-d",),
            evidence_type=EvidenceType.SAME_IP,
            inference_level=InferenceLevel.OBSERVED,
            confidence=Confidence.exact_match(),
            rationale=(
                "Monotonicity fixture: uncited incident edge for citation-ratio "
                "baseline; the edge is recorded but carries no provenance reference."
            ),
            rule_id="rule.monotonicity_fixture",
            rule_version="1.0.0",
            provenance_refs=(),
            created_at=_T0,
        )
        graph_uncited = build_graph((rel_uncited,))
        ctx_uncited = FusionContext(profile=subject, graph=graph_uncited, anomaly=anomaly)

        # Augmented: incident edge WITH a provenance citation
        rel_cited = _rel("m03-a", "m03-c", EvidenceType.SAME_IP)
        graph_cited = build_graph((rel_cited,))
        ctx_cited = FusionContext(profile=subject, graph=graph_cited, anomaly=anomaly)

        score_uncited = _score(ctx_uncited)
        score_cited = _score(ctx_cited)

        assert score_cited >= score_uncited, (
            "M03 FAILED: a cited edge produced a lower fused score than an uncited edge "
            "(%s vs %s)" % (score_cited, score_uncited)
        )


# ---------------------------------------------------------------------------
# M04 -- A second distinct source file never decreases the score
# ---------------------------------------------------------------------------


class TestM04SecondSourceFileNeverDecreases:
    """M04: two distinct provenance source files is non-decreasing vs one."""

    def test_m04_two_source_files_score_geq_one_source_file(self):
        subject_obs = ["m04-a", "m04-b"]
        other_obs = ["m04-c", "m04-d", "m04-e", "m04-f"]

        subject = _profile(subject_obs, "m04-subject")
        pop_profiles = [
            subject,
            _profile(other_obs[:2], "m04-pop1"),
            _profile(other_obs[2:], "m04-pop2"),
        ]
        _, anomaly = IsolationForestAnomalyDetector().fit_score(pop_profiles)

        rel_a = _rel("m04-a", "m04-c", EvidenceType.SAME_IP, source_file="file_alpha.csv")
        rel_b = _rel("m04-b", "m04-d", EvidenceType.SAME_TXID, source_file="file_alpha.csv")
        graph_one_file = build_graph((rel_a, rel_b))
        ctx_one = FusionContext(profile=subject, graph=graph_one_file, anomaly=anomaly)

        rel_a2 = _rel("m04-a", "m04-c", EvidenceType.SAME_IP, source_file="file_alpha.csv")
        rel_b2 = _rel("m04-b", "m04-d", EvidenceType.SAME_TXID, source_file="file_beta.csv")
        graph_two_files = build_graph((rel_a2, rel_b2))
        ctx_two = FusionContext(profile=subject, graph=graph_two_files, anomaly=anomaly)

        score_one = _score(ctx_one)
        score_two = _score(ctx_two)

        assert score_two >= score_one, (
            "M04 FAILED: adding a second distinct source file decreased fused score "
            "from %s to %s" % (score_one, score_two)
        )


# ---------------------------------------------------------------------------
# M05 -- Adding a reachable investigation path never decreases the score
# ---------------------------------------------------------------------------


class TestM05AddingPathNeverDecreases:
    """M05: supplying a non-empty path set is non-decreasing vs empty paths."""

    def test_m05_found_path_score_geq_no_path_score(self):
        all_obs = [_obs("m05-%02d" % i, i, "m05-tx-%d" % i) for i in range(6)]
        subject = BehaviorProfileBuilder().build(all_obs[:3], group_id="m05-subject")
        pop = [
            subject,
            BehaviorProfileBuilder().build(all_obs[3:5], group_id="m05-pop1"),
            BehaviorProfileBuilder().build(all_obs[5:], group_id="m05-pop2"),
        ]
        _, anomaly = IsolationForestAnomalyDetector().fit_score(pop)
        rels = [
            _rel("m05-00", "m05-03", EvidenceType.SAME_IP),
            _rel("m05-03", "m05-05", EvidenceType.SAME_IP),
        ]
        graph = build_graph(rels)

        ctx_no_path = FusionContext(profile=subject, graph=graph, anomaly=anomaly, paths=())

        paths_result = InvestigationEngine().shortest_path(
            graph,
            PathSearchRequest(
                start_observation_id="m05-00",
                target_observation_id="m05-05",
                max_depth=5,
                max_paths=20,
                directed=False,
            ),
            observations=all_obs,
        )
        ctx_with_path = FusionContext(
            profile=subject, graph=graph, anomaly=anomaly, paths=paths_result
        )

        score_no_path = _score(ctx_no_path)
        score_with_path = _score(ctx_with_path)

        assert score_with_path >= score_no_path, (
            "M05 FAILED: adding a found investigation path decreased fused score "
            "from %s to %s" % (score_no_path, score_with_path)
        )


# ---------------------------------------------------------------------------
# M06 -- A second investigation path never decreases the score
# ---------------------------------------------------------------------------


class TestM06SecondPathNeverDecreases:
    """M06: two found paths is non-decreasing vs one found path."""

    def test_m06_two_paths_score_geq_one_path_score(self):
        all_obs = [_obs("m06-%02d" % i, i, "m06-tx-%d" % i) for i in range(7)]
        subject = BehaviorProfileBuilder().build(all_obs[:3], group_id="m06-subject")
        pop = [
            subject,
            BehaviorProfileBuilder().build(all_obs[3:5], group_id="m06-pop1"),
            BehaviorProfileBuilder().build(all_obs[5:], group_id="m06-pop2"),
        ]
        _, anomaly = IsolationForestAnomalyDetector().fit_score(pop)

        rels = [
            _rel("m06-00", "m06-03", EvidenceType.SAME_IP),
            _rel("m06-03", "m06-05", EvidenceType.SAME_IP),
            _rel("m06-00", "m06-04", EvidenceType.SAME_TXID),
            _rel("m06-04", "m06-06", EvidenceType.SAME_TXID),
        ]
        graph = build_graph(rels)

        path_one = InvestigationEngine().shortest_path(
            graph,
            PathSearchRequest(
                start_observation_id="m06-00",
                target_observation_id="m06-05",
                max_depth=5,
                max_paths=1,
                directed=False,
            ),
            observations=all_obs,
        )
        path_two = InvestigationEngine().shortest_path(
            graph,
            PathSearchRequest(
                start_observation_id="m06-00",
                target_observation_id="m06-06",
                max_depth=5,
                max_paths=20,
                directed=False,
            ),
            observations=all_obs,
        )

        ctx_one_path = FusionContext(profile=subject, graph=graph, anomaly=anomaly, paths=path_one)
        ctx_two_paths = FusionContext(profile=subject, graph=graph, anomaly=anomaly, paths=path_two)

        score_one = _score(ctx_one_path)
        score_two = _score(ctx_two_paths)

        assert score_two >= score_one, (
            "M06 FAILED: going from one path to two paths decreased fused score "
            "from %s to %s" % (score_one, score_two)
        )


# ---------------------------------------------------------------------------
# M07 -- A higher anomaly score never decreases the fused score
# ---------------------------------------------------------------------------


class TestM07HigherAnomalyNeverDecreases:
    """M07: whichever population gives the subject the higher anomaly normalized
    score must not produce a strictly lower fused score."""

    def test_m07_higher_anomaly_score_geq_lower_anomaly_score(self):
        all_obs_a = [_obs("m07-a%02d" % i, i, "m07-atx%d" % i) for i in range(6)]
        all_obs_b = [_obs("m07-b%02d" % i, i, "m07-btx%d" % i) for i in range(6)]

        subject = BehaviorProfileBuilder().build(all_obs_a[:3], group_id="m07-subject")

        pop_a = [
            subject,
            BehaviorProfileBuilder().build(all_obs_a[3:5], group_id="m07-a1"),
            BehaviorProfileBuilder().build(all_obs_a[5:], group_id="m07-a2"),
        ]
        _, anomaly_a = IsolationForestAnomalyDetector().fit_score(pop_a)

        twin = BehaviorProfileBuilder().build(all_obs_b[:3], group_id="m07-twin1")
        twin2 = BehaviorProfileBuilder().build(all_obs_b[3:6], group_id="m07-twin2")
        pop_b = [subject, twin, twin2]
        _, anomaly_b = IsolationForestAnomalyDetector().fit_score(pop_b)

        ctx_a = FusionContext(profile=subject, anomaly=anomaly_a)
        ctx_b = FusionContext(profile=subject, anomaly=anomaly_b)

        score_a = _score(ctx_a)
        score_b = _score(ctx_b)

        result_a = anomaly_a.result_for(subject.profile_id)
        result_b = anomaly_b.result_for(subject.profile_id)

        assert result_a is not None, "Population A did not score the subject."
        assert result_b is not None, "Population B did not score the subject."

        if result_b.normalized_score >= result_a.normalized_score:
            assert score_b >= score_a, (
                "M07 FAILED: higher anomaly normalized score (%s vs %s) produced "
                "a lower fused score (%s vs %s)"
                % (result_b.normalized_score, result_a.normalized_score, score_b, score_a)
            )
        else:
            assert score_a >= score_b, (
                "M07 FAILED: higher anomaly normalized score (%s vs %s) produced "
                "a lower fused score (%s vs %s)"
                % (result_a.normalized_score, result_b.normalized_score, score_a, score_b)
            )


# ---------------------------------------------------------------------------
# M08 -- Greater behavioral coverage never decreases the score
# ---------------------------------------------------------------------------


class TestM08GreaterCoverageNeverDecreases:
    """M08: higher behavioral feature coverage is non-decreasing (profile-only)."""

    def test_m08_more_coverage_score_geq_less_coverage_score(self):
        few_obs = [_obs("m08-few-%d" % i, i, "m08-fx%d" % i) for i in range(2)]
        many_obs = [_obs("m08-many-%d" % i, i, "m08-mx%d" % i) for i in range(6)]

        profile_few = BehaviorProfileBuilder().build(few_obs, group_id="m08-few")
        profile_many = BehaviorProfileBuilder().build(many_obs, group_id="m08-many")

        score_few = _score(FusionContext(profile=profile_few))
        score_many = _score(FusionContext(profile=profile_many))

        cov_few = profile_few.feature_count
        cov_many = profile_many.feature_count

        if cov_many >= cov_few:
            assert score_many >= score_few, (
                "M08 FAILED: more features (%d vs %d) but lower fused score (%s vs %s)"
                % (cov_many, cov_few, score_many, score_few)
            )
        else:
            assert score_few >= score_many, (
                "M08 FAILED: more features (%d vs %d) but lower fused score (%s vs %s)"
                % (cov_few, cov_many, score_few, score_many)
            )


# ---------------------------------------------------------------------------
# M09 -- Going from ABSENT graph to a present graph never decreases the score
# ---------------------------------------------------------------------------
#
# Proof of monotonicity under this fixture
# ----------------------------------------
# With profile-only (no anomaly, no graph) only the BEHAVIORAL family
# contributes: Score_0 = B  (behavioral feature-coverage, B in [0, 1]).
#
# Adding the graph activates EVIDENCE_GRAPH (weight=0.20) and PROVENANCE
# (weight=0.10).  With evidence_count_cap=1 and provenance_file_cap=1:
#
#   EVIDENCE_INCIDENT_COUNT   normalized = min(1, 8/1) = 1.0
#   EVIDENCE_TYPE_DIVERSITY   normalized = 8/8 = 1.0   (all 8 types present)
#   → EVIDENCE_GRAPH family mean = 1.0
#
#   PROVENANCE_CITED_EDGE_RATIO      = 1.0  (every edge is cited)
#   PROVENANCE_SOURCE_FILE_DIVERSITY = min(1, 2/1) = 1.0  (≥1 file, cap=1)
#   → PROVENANCE family mean = 1.0
#
# Score_1 = (0.15B + 0.20·1.0 + 0.10·1.0) / (0.15 + 0.20 + 0.10)
#         = (0.15B + 0.30) / 0.45
#
# Score_1 – Score_0 = (0.15B + 0.30)/0.45 – B
#                   = (0.15B + 0.30 – 0.45B) / 0.45
#                   = (0.30 – 0.30B) / 0.45
#                   = 0.30·(1 – B) / 0.45  ≥ 0  for all B ∈ [0, 1].  QED.



def _m09_all_type_rels(subject_obs_id: str, target_obs_ids) -> tuple:
    """One incident edge per EvidenceType, each citing the same source file.

    Using all 8 EvidenceTypes guarantees EVIDENCE_TYPE_DIVERSITY = 1.0 under
    the default normalisation (len(distinct) / len(EvidenceType)).
    """
    all_types = list(EvidenceType)
    assert len(all_types) == len(set(all_types))
    rels = []
    for i, etype in enumerate(all_types):
        target_id = target_obs_ids[i % len(target_obs_ids)]
        rel = EvidenceRelationship(
            relationship_id=derive_relationship_id(
                {"m09_all_types": "%s->%s|%s" % (subject_obs_id, target_id, etype.value)}
            ),
            source_observation_ids=(subject_obs_id,),
            target_observation_ids=(target_id,),
            evidence_type=etype,
            inference_level=InferenceLevel.OBSERVED,
            confidence=Confidence.exact_match(),
            rationale=(
                "M09 fixture: one edge per EvidenceType incident on the subject "
                "observation so that the EVIDENCE_TYPE_DIVERSITY signal reaches 1.0 "
                "under the 8-type normalisation, making the EVIDENCE_GRAPH family "
                "value high enough to guarantee the monotonicity proof above."
            ),
            rule_id="rule.monotonicity_m09_fixture",
            rule_version="1.0.0",
            provenance_refs=(
                ProvenanceRef(
                    ingestion_id="mono-ing",
                    source_file="mono.csv",
                    source_record="m09-row-%s-%s" % (subject_obs_id, etype.value),
                ),
            ),
            created_at=_T0,
        )
        rels.append(rel)
    return tuple(rels)


class TestM09AbsentToGraphNeverDecreases:
    """M09: supplying an evidence graph (with incident edges) is non-decreasing
    vs supplying no graph at all.

    The fixture is carefully constructed so the EVIDENCE_GRAPH and PROVENANCE
    families both reach value 1.0 under caps of 1, making the weighted-average
    formula provably monotone for any behavioral coverage value (see proof
    above).
    """

    def test_m09_graph_with_edges_score_geq_no_graph_score(self):
        # 3 subject observations + 8 non-subject observations (one per type)
        all_obs = [_obs("m09-%02d" % i, i, "m09-tx%d" % i) for i in range(11)]
        subject_obs = all_obs[:3]
        target_obs = all_obs[3:]  # 8 observations for 8 incident edges

        subject = BehaviorProfileBuilder().build(subject_obs, group_id="m09-subject")

        # Profile-only baseline: no anomaly, no graph.
        # Only BEHAVIORAL family contributes; Score_0 = B.
        cfg = FusionConfig(evidence_count_cap=1, provenance_file_cap=1)
        ctx_no_graph = FusionContext(profile=subject)

        # Augmented: graph with one incident edge per EvidenceType (8 edges total).
        rels = _m09_all_type_rels("m09-00", [obs.observation_id for obs in target_obs])
        graph = build_graph(rels)
        ctx_graph = FusionContext(profile=subject, graph=graph)

        score_no_graph = _score(ctx_no_graph, cfg)
        score_graph = _score(ctx_graph, cfg)

        assert score_graph >= score_no_graph, (
            "M09 FAILED: supplying a graph with full-diversity incident edges "
            "decreased fused score from %s to %s" % (score_no_graph, score_graph)
        )


# ---------------------------------------------------------------------------
# M10 -- Going from ABSENT paths to a found trail never decreases the score
# ---------------------------------------------------------------------------


class TestM10AbsentToPathNeverDecreases:
    """M10: supplying paths with a found trail (reachability=1) is non-decreasing
    vs supplying no paths at all."""

    def test_m10_found_trail_score_geq_no_paths_score(self):
        all_obs = [_obs("m10-%02d" % i, i, "m10-tx%d" % i) for i in range(6)]
        subject = BehaviorProfileBuilder().build(all_obs[:3], group_id="m10-subject")
        pop = [
            subject,
            BehaviorProfileBuilder().build(all_obs[3:5], group_id="m10-pop1"),
            BehaviorProfileBuilder().build(all_obs[5:], group_id="m10-pop2"),
        ]
        _, anomaly = IsolationForestAnomalyDetector().fit_score(pop)
        rels = [
            _rel("m10-00", "m10-03", EvidenceType.SAME_IP),
            _rel("m10-03", "m10-05", EvidenceType.SAME_IP),
        ]
        graph = build_graph(rels)

        paths_result = InvestigationEngine().shortest_path(
            graph,
            PathSearchRequest(
                start_observation_id="m10-00",
                target_observation_id="m10-05",
                max_depth=5,
                max_paths=10,
                directed=False,
            ),
            observations=all_obs,
        )
        assert paths_result.paths, "Fixture failed to find any path for M10."

        ctx_no_paths = FusionContext(profile=subject, anomaly=anomaly, graph=graph)
        ctx_with_paths = FusionContext(
            profile=subject, anomaly=anomaly, graph=graph, paths=paths_result
        )

        score_no_paths = _score(ctx_no_paths)
        score_with_paths = _score(ctx_with_paths)

        assert score_with_paths >= score_no_paths, (
            "M10 FAILED: supplying a found trail decreased fused score "
            "from %s to %s" % (score_no_paths, score_with_paths)
        )


# ---------------------------------------------------------------------------
# M11 -- Full context never decreases below profile-only context
# ---------------------------------------------------------------------------
#
# Proof of monotonicity under this fixture
# ----------------------------------------
# Profile-only: only BEHAVIORAL (weight=0.15) is PRESENT.
#   Score_0 = B  (behavioral feature-coverage, B in [0, 1]).
#
# Full context omits anomaly (unfitted run → UNAVAILABLE) and clustering
# (None → ABSENT) so neither enters the formula.
# The remaining controllable families all reach value 1.0:
#
#   EVIDENCE_GRAPH (w=0.20):
#     EVIDENCE_INCIDENT_COUNT   = min(1, 8/1) = 1.0  (cap=1, 8 incident edges)
#     EVIDENCE_TYPE_DIVERSITY   = 8/8 = 1.0            (all 8 EvidenceTypes)
#     → family mean = 1.0
#
#   PROVENANCE (w=0.10):
#     PROVENANCE_CITED_EDGE_RATIO      = 1.0            (every edge cited)
#     PROVENANCE_SOURCE_FILE_DIVERSITY = min(1,1/1)=1.0 (cap=1)
#     → family mean = 1.0
#
#   PATH_STRUCTURE (w=0.10):
#     PATH_REACHABILITY_FOUND = 1.0    (path found)
#     PATH_STRUCTURE_COUNT    = min(1,1/1)=1.0 (cap=1)
#     → family mean = 1.0
#
#   TEMPORAL (w=0.10):
#     TEMPORAL_COMPARABILITY = 1.0     (all transitions comparable: EVIDENCE_STRENGTH)
#     TEMPORAL_FORWARD_FLOW  = 1.0     (all forward; CONTEXT_ONLY → excluded from scoring)
#     → 1 scoring signal; availability=1; family value = 1.0
#
# Score_1 = (0.15B + 0.20 + 0.10 + 0.10 + 0.10) / (0.15 + 0.20 + 0.10 + 0.10 + 0.10)
#         = (0.15B + 0.50) / 0.65
#
# Score_1 – Score_0 = (0.15B + 0.50)/0.65 – B
#                   = 0.50·(1 – B) / 0.65  ≥ 0  for all B ∈ [0, 1].  QED.


class TestM11FullContextNeverDecreasesProfileOnly:
    """M11: the fused score under all deterministic producers is non-decreasing
    vs profile only.

    Stochastic ML producers (anomaly, clustering) are set to UNAVAILABLE /
    ABSENT so the invariant can be proved algebraically (see proof above).
    Every deterministic producer (evidence graph, provenance, paths, temporal)
    is supplied at full strength, testing the realistic scenario of a
    well-evidenced subject.
    """

    def test_m11_full_context_score_geq_profile_only_score(self):
        # 3 subject observations + 8 target observations (one per EvidenceType).
        all_obs = [_obs("m11-%02d" % i, i, "m11-tx%d" % i) for i in range(11)]
        subject_obs = all_obs[:3]
        target_obs = all_obs[3:]   # timestamps at hours 3-10, all > subject hours

        subject = BehaviorProfileBuilder().build(subject_obs, group_id="m11-subject")

        # ANOMALY: unfitted run → UNAVAILABLE signal (excluded from formula).
        _unfitted_model, anomaly_unavailable = IsolationForestAnomalyDetector().fit_score([])

        # Caps all set to 1: a single instance of each saturates the normalization
        # to 1.0, keeping the monotonicity proof tight and simple.
        cfg = FusionConfig(
            evidence_count_cap=1,
            provenance_file_cap=1,
            path_count_cap=1,
        )

        # Profile-only baseline (no graph, no anomaly, no paths, no clustering).
        ctx_profile_only = FusionContext(profile=subject)

        # Full-context graph: one incident edge per EvidenceType, all cited,
        # using the same _m09_all_type_rels helper.
        rels = _m09_all_type_rels(
            "m11-00", [obs.observation_id for obs in target_obs]
        )
        graph = build_graph(rels)

        # Path: m11-00 (hour=0) → target_obs[0] (hour=3) – timestamps increase
        # so TEMPORAL_COMPARABILITY = 1.0, TEMPORAL_FORWARD_FLOW = 1.0.
        paths_result = InvestigationEngine().shortest_path(
            graph,
            PathSearchRequest(
                start_observation_id="m11-00",
                target_observation_id=target_obs[0].observation_id,
                max_depth=3,
                max_paths=5,
                directed=False,
            ),
            observations=all_obs,
        )
        assert paths_result.paths, (
            "M11 fixture failed: expected a path but graph has no direct edge "
            "from m11-00 to the first target. Check _m09_all_type_rels."
        )

        ctx_full = FusionContext(
            profile=subject,
            graph=graph,
            anomaly=anomaly_unavailable,   # UNAVAILABLE – excluded from formula
            clustering=None,               # ABSENT      – excluded from formula
            paths=paths_result,
        )

        score_profile_only = _score(ctx_profile_only, cfg)
        score_full = _score(ctx_full, cfg)

        assert score_full >= score_profile_only, (
            "M11 FAILED: full context decreased fused score below profile-only "
            "(%s -> %s)" % (score_profile_only, score_full)
        )

