"""Step 9A: Dashboard Overview Contract Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data
from backend.app.dashboard.models import DashboardOverview


def test_overview_contract():
    """Test that overview contains all required fields with correct types."""
    data = _get_demo_data()
    overview = data.overview
    
    assert isinstance(overview, DashboardOverview)
    import uuid; assert uuid.UUID(overview.subject_profile_id)
    assert isinstance(overview.subject_observation_ids, tuple)
    assert len(overview.subject_observation_ids) == 3
    
    # Investigative Priority (not fraud probability)
    assert overview.investigative_priority is not None
    assert 0.0 <= overview.investigative_priority <= 1.0
    
    # Risk band
    assert overview.risk_band in ("LOW", "MEDIUM", "HIGH", "CRITICAL", None)
    
    # Status
    assert overview.status in ("COMPUTED", "INSUFFICIENT_SIGNALS")
    
    # Evidence summary
    assert isinstance(overview.evidence_summary, dict)
    assert "incident_edge_count" in overview.evidence_summary
    assert "evidence_types" in overview.evidence_summary
    assert "inference_levels" in overview.evidence_summary
    
    # Counts
    assert overview.observation_count == 3
    assert overview.evidence_relationship_count >= 3
    assert overview.investigation_path_count >= 1
    
    # Context objects
    assert isinstance(overview.anomaly_context, dict)
    assert isinstance(overview.cluster_context, dict)
    assert isinstance(overview.provenance_context, dict)
    assert isinstance(overview.temporal_context, dict)
    
    # Metadata
    assert overview.assessment_id is not None
    assert overview.config_version is not None
    assert overview.interpretation is not None
    assert "deterministic priority-ranking" in overview.interpretation.lower()
    assert "not a probability" in overview.interpretation.lower()


def test_overview_priority_terminology():
    """Verify terminology uses 'Investigative Priority' not 'fraud probability'."""
    data = _get_demo_data()
    overview = data.overview
    
    # The field is called investigative_priority
    assert hasattr(overview, "investigative_priority")
    assert not hasattr(overview, "fraud_probability")
    assert not hasattr(overview, "crime_probability")
    assert not hasattr(overview, "maliciousness_probability")
    assert not hasattr(overview, "ownership_probability")
    assert not hasattr(overview, "identity_confidence")


if __name__ == "__main__":
    test_overview_contract()
    test_overview_priority_terminology()
    print("test_step9a_overview_contract.py PASSED")

