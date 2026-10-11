"""Dashboard adapter: converts Step 1-8 outputs to dashboard data contracts (SIH26146 - Step 9A).

This adapter is the ONLY place where Step 8+ objects are transformed for the
dashboard. It is read-only and does not mutate any source objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from typing import Optional, Tuple

from ..behavior.profiles import BehavioralProfile
from ..correlation.enums import InferenceLevel
from ..correlation.models import EvidenceRelationship
from ..fusion.models import (
    FusionAssessment,
    FusionSignal,
    MissingSignal,
    RiskBand,
    SignalFamily,
    SignalRole,
    SignalStatus,
)
from ..fusion.signals import FusionContext
from ..graph.models import EvidenceGraph
from ..investigation.models import InvestigationPath, PathSearchResult
from ..ml.models import AnomalyRunResult, ClusteringRunResult, ModelStatus
from .models import (
    DashboardBehaviorContext,
    DashboardData,
    DashboardEvidence,
    DashboardOverview,
    DashboardPath,
    DashboardProvenance,
    DashboardSignal,
    InferenceLevelDisplay,
    SignalRoleDisplay,
    SignalStatusDisplay,
    _sort_by_id,
    _sort_strings,
)

__all__ = ["build_dashboard_data"]


def _deterministic_timestamp(seed: str) -> datetime:
    """Derive a deterministic UTC timestamp from a seed string.

    Uses SHA-256 of the seed to produce a stable timestamp. The timestamp
    is derived from the hash value to ensure the same seed always produces
    the same timestamp, without using wall-clock time.
    """
    # Use first 8 bytes of SHA-256 as a microsecond offset from epoch
    hash_bytes = sha256(seed.encode("utf-8")).digest()[:8]
    microseconds = int.from_bytes(hash_bytes, "big")
    # Convert to a reasonable datetime range (year 2000-2100)
    # Use modulo to keep within ~100 years from epoch
    epoch_microseconds = 946684800_000_000  # 2000-01-01 00:00:00 UTC in microseconds
    range_microseconds = 31556952_000_000  # ~100 years in microseconds
    timestamp_microseconds = epoch_microseconds + (microseconds % range_microseconds)
    return datetime.fromtimestamp(timestamp_microseconds / 1_000_000, tz=timezone.utc)


def _map_signal_role(role: SignalRole) -> SignalRoleDisplay:
    return SignalRoleDisplay(role.value)


def _map_signal_status(status: SignalStatus) -> SignalStatusDisplay:
    return SignalStatusDisplay(status.value)


def _map_inference_level(level: InferenceLevel) -> InferenceLevelDisplay:
    return InferenceLevelDisplay(level.value)


def _fusion_signal_to_dashboard(signal: FusionSignal) -> DashboardSignal:
    return DashboardSignal(
        signal_id=signal.signal_id,
        signal_type=signal.signal_type,
        family=signal.family.value,
        role=_map_signal_role(signal.role),
        status=_map_signal_status(signal.status),
        source_step=signal.source_step,
        source_id=signal.source_id,
        subject_id=signal.subject_id,
        raw_value=signal.raw_value,
        normalized_value=signal.normalized_value,
        explanation=signal.explanation,
        evidence_refs=signal.evidence_refs,
        inference_levels=tuple(_map_inference_level(l) for l in signal.inference_levels),
        provenance_refs=tuple(
            {
                "ingestion_id": ref.ingestion_id,
                "source_file": ref.source_file,
                "source_record": ref.source_record,
            }
            for ref in signal.provenance_refs
        ),
        lineage=signal.lineage,
        context=signal.context,
    )


def _missing_signal_to_dashboard(signal: MissingSignal) -> DashboardSignal:
    return DashboardSignal(
        signal_id=signal.signal_id,
        signal_type=signal.signal_type,
        family=signal.family.value,
        role=SignalRoleDisplay.PRIORITY_SIGNAL,
        status=_map_signal_status(signal.status),
        source_step=signal.source_step,
        source_id=signal.signal_type,
        subject_id=signal.signal_type,
        raw_value=None,
        normalized_value=None,
        explanation=signal.explanation,
        evidence_refs=(),
        inference_levels=(),
        provenance_refs=(),
        lineage=(),
        context=(),
    )


def _evidence_edge_to_dashboard(edge) -> DashboardEvidence:
    """Convert an EvidenceEdge to DashboardEvidence."""
    return DashboardEvidence(
        relationship_id=edge.relationship_id,
        source_observation_ids=edge.source_observation_ids,
        target_observation_ids=edge.target_observation_ids,
        evidence_type=edge.evidence_type.value,
        inference_level=_map_inference_level(edge.inference_level),
        confidence_value=float(edge.confidence.value),
        confidence_basis=edge.confidence.basis.value,
        rationale=edge.rationale,
        rule_id=edge.rule_id,
        rule_version=edge.rule_version,
        provenance_refs=tuple(
            {
                "ingestion_id": ref.ingestion_id,
                "source_file": ref.source_file,
                "source_record": ref.source_record,
            }
            for ref in edge.provenance_refs
        ),
        created_at=edge.created_at,
    )


def _investigation_path_to_dashboard(path: InvestigationPath) -> DashboardPath:
    temporal_flow = tuple(f"{step.from_observation_id} -> {step.to_observation_id} ({step.evidence_type.value}, {step.temporal_relation.value}, elapsed={step.elapsed_seconds}s)" for step in path.temporal_flow()) if path.temporal_flow else ()

    return DashboardPath(
        path_id=path.path_id,
        source_observation_id=path.observation_ids[0] if path.observation_ids else "",
        target_observation_id=path.observation_ids[-1] if path.observation_ids else "",
        hop_count=path.hop_count,
        node_count=path.hop_count + 1,
        observation_ids=path.observation_ids,
        relationship_ids=path.relationship_ids,
        temporal_status=path.temporal_status.value,
        temporal_flow=temporal_flow,
        duration_seconds=path.duration_seconds,
        evidence_summary={
            "relationship_count": len(path.relationship_ids),
            "evidence_types": list(path.evidence_summary.evidence_types) if path.evidence_summary else [],
            "confidence_bounds": {
                "min": path.evidence_summary.min_edge_confidence if path.evidence_summary else None,
                "max": path.evidence_summary.max_edge_confidence if path.evidence_summary else None,
            } if path.evidence_summary else {},
        },
        search_status="FOUND",
    )


def _build_overview(assessment: FusionAssessment, context: FusionContext) -> DashboardOverview:
    profile = context.profile

    evidence_summary = {}
    if context.graph:
        edges = tuple(edge for edge in context.graph.edges() if set(edge.observation_ids()) & set(profile.observation_ids))
        evidence_types = set(edge.evidence_type.value for edge in edges)
        inference_levels = set(edge.inference_level.value for edge in edges)
        evidence_summary = {
            "incident_edge_count": len(edges),
            "evidence_types": sorted(evidence_types),
            "inference_levels": sorted(inference_levels),
        }

    anomaly_context = {}
    if context.anomaly and context.anomaly.status == ModelStatus.FITTED:
        result = context.anomaly.result_for(profile.profile_id)
        if result:
            anomaly_context = {
                "band": result.band.value if result.band else None,
                "normalized_score": float(result.normalized_score) if result.normalized_score else None,
                "is_in_sample": result.is_in_sample,
            }

    cluster_context = {}
    if context.clustering and context.clustering.status == ModelStatus.FITTED:
        assignment = context.clustering.cluster_of(profile.profile_id)
        if assignment and not assignment.is_noise:
            summary = context.clustering.cluster_summary(assignment.cluster_id)
            cluster_context = {
                "cluster_id": assignment.cluster_id,
                "cluster_size": summary.member_count if summary else 0,
                "distance_to_centroid": float(assignment.distance_to_centroid) if assignment.distance_to_centroid is not None else None,
            }

    provenance_context = {}
    if context.graph:
        edges = tuple(edge for edge in context.graph.edges() if set(edge.observation_ids()) & set(profile.observation_ids))
        cited = sum(1 for e in edges if e.provenance_refs)
        source_files = set()
        for ref in (r for e in edges for r in e.provenance_refs):
            if ref.source_file:
                source_files.add(ref.source_file)
        for pf in profile.provenance_source_files:
            source_files.add(pf)
        provenance_context = {
            "cited_edge_count": cited,
            "incident_edge_count": len(edges),
            "distinct_source_files": len(source_files),
        }

    temporal_context = {}
    if context.paths:
        paths = tuple(p for p in context.paths.paths if set(p.observation_ids) & set(profile.observation_ids))
        if paths:
            total = 0
            comparable = 0
            after = 0
            for path in paths:
                for trans in path.transitions:
                    total += 1
                    if trans.temporal and trans.temporal.comparable:
                        comparable += 1
                        if trans.temporal.relation.value == "AFTER":
                            after += 1
            temporal_context = {
                "path_count": len(paths),
                "total_transitions": total,
                "comparable_transitions": comparable,
                "forward_transitions": after,
            }

    risk_band = assessment.risk_band.value if assessment.risk_band else None

    return DashboardOverview(
        subject_profile_id=assessment.subject_profile_id,
        subject_observation_ids=assessment.subject_observation_ids,
        investigative_priority=float(assessment.fused_score) if assessment.fused_score is not None else None,
        risk_band=risk_band,
        status=assessment.status.value,
        evidence_summary=evidence_summary,
        observation_count=len(profile.observation_ids),
        evidence_relationship_count=len(assessment.distinct_relationship_ids),
        investigation_path_count=len(paths) if context.paths else 0,
        anomaly_context=anomaly_context,
        cluster_context=cluster_context,
        provenance_context=provenance_context,
        temporal_context=temporal_context,
        assessment_id=assessment.assessment_id,
        config_version=assessment.config_version,
        interpretation=assessment.interpretation,
    )


def _build_behavior_context(context: FusionContext) -> Optional[DashboardBehaviorContext]:
    profile = context.profile
    feature_categories = tuple(sorted({f.category.value for f in profile.features}))

    anomaly_band = None
    anomaly_score = None
    anomaly_in_sample = None
    if context.anomaly and context.anomaly.status == ModelStatus.FITTED:
        result = context.anomaly.result_for(profile.profile_id)
        if result:
            anomaly_band = result.band.value if result.band else None
            anomaly_score = float(result.normalized_score) if result.normalized_score else None
            anomaly_in_sample = result.is_in_sample

    cluster_id = None
    cluster_size = None
    cluster_distance = None
    cluster_normalized = None
    if context.clustering and context.clustering.status == ModelStatus.FITTED:
        assignment = context.clustering.cluster_of(profile.profile_id)
        if assignment and not assignment.is_noise:
            cluster_id = assignment.cluster_id
            cluster_distance = float(assignment.distance_to_centroid) if assignment.distance_to_centroid is not None else None
            summary = context.clustering.cluster_summary(assignment.cluster_id)
            cluster_size = summary.member_count if summary else 0
            non_noise = [item.distance_to_centroid for item in context.clustering.assignments if not item.is_noise and item.distance_to_centroid is not None]
            if non_noise:
                pop_max = max(non_noise)
                if pop_max > 0:
                    cluster_normalized = min(1.0, float(cluster_distance) / float(pop_max))

    explanation_items = tuple(sorted(
        item.text for item in []  # Explanation items would come from assessment
    ))

    return DashboardBehaviorContext(
        profile_id=profile.profile_id,
        grouping=profile.grouping.value,
        feature_count=profile.feature_count,
        feature_categories=feature_categories,
        anomaly_band=anomaly_band,
        anomaly_normalized_score=anomaly_score,
        anomaly_is_in_sample=anomaly_in_sample,
        cluster_id=cluster_id,
        cluster_size=cluster_size,
        cluster_distance=cluster_distance,
        cluster_normalized=cluster_normalized,
        explanation_items=explanation_items,
    )


def _build_provenance_context(assessment: FusionAssessment, context: FusionContext) -> Optional[DashboardProvenance]:
    profile = context.profile

    citation_ratio = None
    cited_count = None
    incident_count = None
    distinct_files = None
    temporal_comp = None
    temporal_flow = None
    missing = tuple(sorted(m.signal_type for m in assessment.missing_signals))

    if context.graph:
        edges = tuple(edge for edge in context.graph.edges() if set(edge.observation_ids()) & set(profile.observation_ids))
        incident_count = len(edges)
        cited_count = sum(1 for e in edges if e.provenance_refs)
        if incident_count > 0:
            citation_ratio = cited_count / incident_count

        source_files = set()
        for ref in (r for e in edges for r in e.provenance_refs):
            if ref.source_file:
                source_files.add(ref.source_file)
        for pf in profile.provenance_source_files:
            source_files.add(pf)
        distinct_files = len(source_files)

    if context.paths:
        paths = tuple(p for p in context.paths.paths if set(p.observation_ids) & set(profile.observation_ids))
        if paths:
            total = 0
            comparable = 0
            after = 0
            for path in paths:
                for trans in path.transitions:
                    total += 1
                    if trans.temporal and trans.temporal.comparable:
                        comparable += 1
                        if trans.temporal.relation.value == "AFTER":
                            after += 1
            if total > 0:
                temporal_comp = comparable / total
            if comparable > 0:
                temporal_flow = after / comparable

    return DashboardProvenance(
        ingestion_id=profile.provenance_ingestion_ids[0] if profile.provenance_ingestion_ids else "unknown",
        source_files=tuple(sorted(profile.provenance_source_files)) if hasattr(profile, 'provenance_source_files') else (),
        source_records=(),
        evidence_citation_ratio=citation_ratio,
        cited_edge_count=cited_count,
        incident_edge_count=incident_count,
        distinct_source_files=distinct_files,
        temporal_comparability=temporal_comp,
        temporal_forward_flow=temporal_flow,
        missing_signals=missing,
    )


def build_dashboard_data(
    assessment: FusionAssessment,
    context: FusionContext,
) -> DashboardData:
    """Build complete, deterministic dashboard data from assessment and context.

    This is a pure function: same inputs always produce same output.
    All source objects are treated as read-only.
    """
    # Build signals (both present and missing)
    present_signals = tuple(_fusion_signal_to_dashboard(s) for s in assessment.signals if s.status == SignalStatus.PRESENT)
    missing_signals = tuple(_missing_signal_to_dashboard(m) for m in assessment.missing_signals)
    all_signals = _sort_by_id(present_signals + missing_signals)

    # Build evidence
    evidence_list = ()
    if context.graph:
        edges = tuple(edge for edge in context.graph.edges() if set(edge.observation_ids()) & set(context.profile.observation_ids))
        evidence_list = _sort_by_id(
            tuple(_evidence_edge_to_dashboard(edge) for edge in edges),
            key_attr="relationship_id"
        )

    # Build paths
    paths_list = ()
    if context.paths:
        subject_paths = tuple(p for p in context.paths.paths if set(p.observation_ids) & set(context.profile.observation_ids))
        paths_list = _sort_by_id(
            tuple(_investigation_path_to_dashboard(p) for p in subject_paths),
            key_attr="path_id"
        )

    # Build behavior context
    behavior = _build_behavior_context(context)

    # Build provenance
    provenance = _build_provenance_context(assessment, context)

    # Build overview
    overview = _build_overview(assessment, context)

    return DashboardData(
        overview=overview,
        signals=all_signals,
        evidence=evidence_list,
        paths=paths_list,
        behavior=behavior,
        provenance=provenance,
        generated_at=_deterministic_timestamp(assessment.assessment_id),
        version="STEP9A/1",
    )




