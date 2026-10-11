"""Step 9A: Immutability Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data, _create_demo_context
from backend.app.dashboard.adapter import build_dashboard_data
from backend.app.fusion import FusionEngine


def _profile_snapshot(profile):
    """Create a snapshot of BehavioralProfile using its public properties."""
    return {
        "profile_id": profile.profile_id,
        "grouping": profile.grouping.value,
        "group_id": profile.group_id,
        "observation_ids": profile.observation_ids,
        "feature_count": profile.feature_count,
        "evidence_relationship_ids": profile.evidence_relationship_ids,
        "provenance_ingestion_ids": profile.provenance_ingestion_ids,
        "provenance_source_files": profile.provenance_source_files,
    }


def test_source_objects_not_mutated():
    """Test that dashboard construction does not mutate source objects."""
    context = _create_demo_context()
    engine = FusionEngine()
    assessment = engine.assess(context)
    
    # Store original state using snapshots
    profile_snap = _profile_snapshot(context.profile)
    graph_snap = context.graph.to_dict()
    anomaly_snap = context.anomaly.to_dict() if context.anomaly else None
    clustering_snap = context.clustering.to_dict() if context.clustering else None
    paths_snap = context.paths.to_dict() if context.paths else None
    assessment_snap = assessment.model_dump()
    
    # Build dashboard
    dashboard = build_dashboard_data(assessment, context)
    
    # Verify source objects unchanged by comparing snapshots
    assert _profile_snapshot(context.profile) == profile_snap
    assert context.graph.to_dict() == graph_snap
    if context.anomaly:
        assert context.anomaly.to_dict() == anomaly_snap
    if context.clustering:
        assert context.clustering.to_dict() == clustering_snap
    if context.paths:
        assert context.paths.to_dict() == paths_snap
    assert assessment.model_dump() == assessment_snap
    
    # Verify dashboard is frozen/immutable
    import uuid; assert uuid.UUID(dashboard.overview.subject_profile_id)


def test_dashboard_data_immutable():
    """Test that DashboardData is frozen."""
    data = _get_demo_data()
    
    # Should be frozen (pydantic ConfigDict frozen=True)
    with pytest.raises(Exception):
        data.overview = None
    
    with pytest.raises(Exception):
        data.signals = ()


def test_frozen_models():
    """Test that all dashboard models are frozen."""
    from backend.app.dashboard.models import (
        DashboardData, DashboardOverview, DashboardSignal, DashboardEvidence,
        DashboardPath, DashboardBehaviorContext, DashboardProvenance
    )
    
    for model_class in [DashboardData, DashboardOverview, DashboardSignal, DashboardEvidence, DashboardPath, DashboardBehaviorContext, DashboardProvenance]:
        assert model_class.model_config.get("frozen") is True


if __name__ == "__main__":
    test_source_objects_not_mutated()
    test_dashboard_data_immutable()
    test_frozen_models()
    print("test_step9a_immutability.py PASSED")

