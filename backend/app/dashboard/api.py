"""Dashboard API router (SIH26146 - Step 9A).

Read-only endpoints exposing Step 1-8 intelligence to the investigator dashboard.
All endpoints are GET-only. No persistence. No external dependencies.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..behavior.profiles import BehaviorProfileBuilder
from ..correlation.enums import EvidenceType, InferenceLevel
from ..correlation.models import (
    Confidence,
    EvidenceRelationship,
    derive_relationship_id,
)
from ..correlation.provenance import ProvenanceRef
from ..domain.models import (
    BlockchainTransaction,
    NetworkObservation,
    ObservationType,
    Provenance,
    UnifiedObservation,
)
from ..fusion import FusionConfig, FusionContext, FusionEngine, FusionStatus
from ..graph import build_graph
from ..investigation import InvestigationEngine, PathSearchRequest
from ..ml.anomaly import IsolationForestAnomalyDetector
from ..ml.clustering import KMeansBehaviorClusterer
from .adapter import build_dashboard_data
from .models import DashboardData

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _create_demo_context() -> FusionContext:
    """Create a deterministic demo FusionContext for the dashboard.

    This provides a fixture pathway so the dashboard can be demonstrated
    without requiring a full ingestion pipeline run.

    Enhanced to exercise:
    - Multiple evidence types (SAME_IP, SAME_TXID, SAME_ADDRESS, TEMPORAL_ADJACENCY, AMOUNT_MATCH, PROVENANCE_LINK, GEO_COINCIDENCE)
    - OBSERVED, CORRELATED, INFERRED, REQUIRES_VERIFICATION inference levels
    - Provenance references on all edges
    - Temporal windows and relations
    - Investigation paths with temporal flow
    - Anomaly context
    - Clustering context
    - Appropriate missing/unavailable state
    """
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal

    from ..correlation.models import TemporalWindow, TemporalRelation

    UTC = timezone.utc
    T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)

    def _obs(obs_id: str, hour: int, txid: str, ip_suffix: int = 1) -> UnifiedObservation:
        ts = T0 + timedelta(hours=hour)
        return UnifiedObservation(
            observation_id=obs_id,
            observation_type=ObservationType.COMBINED,
            timestamp=ts,
            network=NetworkObservation(
                timestamp=ts,
                src_ip="10.0.0.%d" % ip_suffix,
                dst_ip="10.0.9.9",
                src_port=1024 + (ip_suffix % 10),
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
                ingestion_id="demo-ingestion",
                source_file="demo.csv",
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
        confidence=None,
        rationale: str = "Demo fixture: both observations share the stated attribute.",
        rule_id: str = "rule.demo",
        rule_version: str = "1.0.0",
        source_file: str = "demo.csv",
        temporal_window=None,
        temporal_relation=None,
    ) -> EvidenceRelationship:
        if confidence is None:
            confidence = Confidence.exact_match()
        return EvidenceRelationship(
            relationship_id=derive_relationship_id({"demo": "%s->%s->%s" % (src, tgt, evidence_type.value)}),
            source_observation_ids=(src,),
            target_observation_ids=(tgt,),
            evidence_type=evidence_type,
            inference_level=inference_level,
            confidence=confidence,
            rationale=rationale,
            rule_id=rule_id,
            rule_version=rule_version,
            provenance_refs=(
                ProvenanceRef(
                    ingestion_id="demo-ingestion",
                    source_file=source_file,
                    source_record="row-" + src,
                ),
                ProvenanceRef(
                    ingestion_id="demo-ingestion",
                    source_file=source_file,
                    source_record="row-" + tgt,
                ),
            ),
            created_at=T0,
            temporal_window=temporal_window,
            temporal_relation=temporal_relation,
        )

    # Build subject profile - 3 observations
    subject_obs_ids = ["demo-a", "demo-b", "demo-c"]
    subject_obs = [_obs(oid, i, "demo-tx-%d" % i, i + 1) for i, oid in enumerate(subject_obs_ids)]
    profile = BehaviorProfileBuilder().build(subject_obs, group_id="demo-subject")

    # Population for anomaly detection - 4 observations
    pop_obs_ids = ["demo-d", "demo-e", "demo-f", "demo-g"]
    pop_obs = [_obs(oid, i + 3, "demo-pop-tx-%d" % i, i + 4) for i, oid in enumerate(pop_obs_ids)]
    pop_profiles = [
        profile,
        BehaviorProfileBuilder().build(pop_obs[:2], group_id="demo-pop1"),
        BehaviorProfileBuilder().build(pop_obs[2:], group_id="demo-pop2"),
    ]
    _, anomaly = IsolationForestAnomalyDetector().fit_score(pop_profiles)

    # Graph with evidence relationships - multiple types and inference levels
    # Subject incident edges (direct evidence)
    # 1-hop neighbor edges (contextual evidence)
    rels = (
        # OBSERVED - direct network observation matches
        _rel("demo-a", "demo-d", EvidenceType.SAME_IP, InferenceLevel.OBSERVED,
             rationale="Both observations recorded from the same source IP 10.0.0.1 at ingestion."),
        _rel("demo-b", "demo-e", EvidenceType.SAME_TXID, InferenceLevel.OBSERVED,
             rationale="Blockchain transaction demo-tx-1 appears in both observation records."),
        _rel("demo-c", "demo-f", EvidenceType.SAME_ADDRESS, InferenceLevel.OBSERVED,
             rationale="Output address 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2 shared between observations."),
        
        # CORRELATED - inferred from multiple matching attributes
        _rel("demo-a", "demo-b", EvidenceType.SAME_IP, InferenceLevel.CORRELATED,
             rationale="Correlated: demo-a and demo-b share IP subnet and temporal proximity."),
        _rel("demo-b", "demo-c", EvidenceType.SAME_TXID, InferenceLevel.CORRELATED,
             rationale="Correlated: sequential transactions with common input address pattern."),
        _rel("demo-d", "demo-e", EvidenceType.TEMPORAL_ADJACENCY, InferenceLevel.CORRELATED,
             rationale="Correlated: temporal adjacency within the same hour window."),
        
        # INFERRED - behavioral pattern inference
        _rel("demo-a", "demo-c", EvidenceType.SAME_ADDRESS, InferenceLevel.INFERRED,
             rationale="Inferred: repeated address usage pattern across demo-a and demo-c suggests common controller.",
             rule_id="rule.behavioral.repeated_address",
             rule_version="1.0.0"),
        
        # REQUIRES_VERIFICATION - needs additional evidence (lower confidence from rule)
        _rel("demo-a", "demo-g", EvidenceType.GEO_COINCIDENCE, InferenceLevel.REQUIRES_VERIFICATION,
             confidence=Confidence.from_rule(0.4),
             rationale="Same country code observed but requires verification of common origin.",
             rule_id="rule.network.geo_attribution",
             rule_version="1.0.0"),
        
        # Additional correlations
        _rel("demo-e", "demo-f", EvidenceType.AMOUNT_MATCH, InferenceLevel.CORRELATED,
             rationale="Correlated: matching transaction amounts observed."),
        
        # Provenance link
        _rel("demo-f", "demo-g", EvidenceType.PROVENANCE_LINK, InferenceLevel.OBSERVED,
             rationale="Both observations share ingestion provenance."),
        
        # Temporal edges with windows
        _rel("demo-d", "demo-e", EvidenceType.SAME_IP, InferenceLevel.OBSERVED,
             temporal_window=TemporalWindow(
                 start=T0 + timedelta(hours=3),
                 end=T0 + timedelta(hours=4),
             ),
             temporal_relation=TemporalRelation.OVERLAPS,
             rationale="Overlapping temporal window for IP correlation."),
    )
    graph = build_graph(rels)

    # Investigation paths
    all_obs = subject_obs + pop_obs
    paths_result = InvestigationEngine().shortest_path(
        graph,
        PathSearchRequest(
            start_observation_id="demo-a",
            target_observation_id="demo-f",
            max_depth=5,
            max_paths=10,
            directed=False,
        ),
        observations=all_obs,
    )

    # Also find paths from demo-b to demo-g for more path diversity
    paths_result_2 = InvestigationEngine().shortest_path(
        graph,
        PathSearchRequest(
            start_observation_id="demo-b",
            target_observation_id="demo-g",
            max_depth=5,
            max_paths=10,
            directed=False,
        ),
        observations=all_obs,
    )

    # Combine paths (deduplicate by path_id)
    all_paths = list(paths_result.paths)
    for p in paths_result_2.paths:
        if p.path_id not in {x.path_id for x in all_paths}:
            all_paths.append(p)
    
    from ..investigation.models import PathSearchResult, SearchStatus
    combined_paths = PathSearchResult(
        status=SearchStatus.FOUND,
        detail="Combined paths for demo",
        request=PathSearchRequest(
            start_observation_id="demo-a",
            target_observation_id="demo-f",
            max_depth=5,
            max_paths=10,
            directed=False,
        ),
        paths=tuple(all_paths),
        path_count=len(all_paths),
        shortest_hops=min((p.hop_count for p in all_paths), default=None),
        temporal_exclusions=0,
        incomparable_exclusions=0,
        search_scope="demo-combined",
    )

    # Clustering - deterministic using the same population as anomaly
    clusterer = KMeansBehaviorClusterer()
    clustering_model = clusterer.fit(pop_profiles)
    clustering_result = clustering_model.result()

    return FusionContext(
        profile=profile,
        graph=graph,
        anomaly=anomaly,
        clustering=clustering_result,
        paths=combined_paths,
    )


# Cache the demo context for determinism
_demo_context: Optional[FusionContext] = None
_demo_assessment: Optional[DashboardData] = None


def _get_demo_data() -> DashboardData:
    """Get or create the deterministic demo dashboard data."""
    global _demo_context, _demo_assessment
    if _demo_assessment is None:
        _demo_context = _create_demo_context()
        engine = FusionEngine()
        assessment = engine.assess(_demo_context)
        _demo_assessment = build_dashboard_data(assessment, _demo_context)
    return _demo_assessment


@router.get("/overview", response_model=dict)
async def get_overview() -> dict:
    """Get investigation overview.

    Returns the subject profile identifier, investigative priority, status,
    evidence summary, and context from all available producers.
    """
    data = _get_demo_data()
    return data.overview.model_dump(mode="json")


@router.get("/signals", response_model=dict)
async def get_signals() -> dict:
    """Get all Step 8 signals for the subject.

    Returns all 11 signals with their roles (PRIORITY_SIGNAL, EVIDENCE_STRENGTH,
    CONTEXT_ONLY), statuses, raw/normalized values, explanations, evidence refs,
    inference levels, and provenance. CONTEXT_ONLY signals are explicitly marked.
    """
    data = _get_demo_data()
    return {"signals": [s.model_dump(mode="json") for s in data.signals]}


@router.get("/evidence", response_model=dict)
async def get_evidence() -> dict:
    """Get evidence relationships for the subject.

    Returns all incident evidence relationships with their types, inference
    levels (OBSERVED, CORRELATED, INFERRED, REQUIRES_VERIFICATION), confidence,
    provenance references, and rule identifiers.
    """
    data = _get_demo_data()
    return {"evidence": [e.model_dump(mode="json") for e in data.evidence]}


@router.get("/paths", response_model=dict)
async def get_paths() -> dict:
    """Get investigation paths for the subject.

    Returns all trails touching the subject's observations with hop counts,
    temporal status, evidence summaries, and search status.
    """
    data = _get_demo_data()
    return {"paths": [p.model_dump(mode="json") for p in data.paths]}


@router.get("/behavior", response_model=dict)
async def get_behavior() -> dict:
    """Get behavioral and ML context for the subject.

    Returns feature summary, anomaly/unusualness context, cluster membership/
    context, and relevant explanations. Uses language like "behavioral
    unusualness" and "cluster context" - never "fraud probability" or "identity".
    """
    data = _get_demo_data()
    if data.behavior:
        return data.behavior.model_dump(mode="json")
    return {"detail": "No behavioral context available"}


@router.get("/provenance", response_model=dict)
async def get_provenance() -> dict:
    """Get provenance and data quality summary.

    Returns source files, source records, ingestion identifiers, evidence
    provenance, temporal comparability, and missing/unavailable information.
    """
    data = _get_demo_data()
    if data.provenance:
        return data.provenance.model_dump(mode="json")
    return {"detail": "No provenance context available"}


@router.get("/all", response_model=dict)
async def get_all() -> dict:
    """Get complete dashboard data in one response.

    Deterministic ordering enforced across all collections.
    """
    data = _get_demo_data()
    return data.model_dump(mode="json")


@router.get("/health", response_model=dict)
async def dashboard_health() -> dict:
    """Dashboard health check."""
    return {"status": "ok", "service": "BitTrace AI Dashboard", "version": "STEP9A/1"}


