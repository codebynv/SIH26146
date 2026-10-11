"""Step 9A: API Overview Endpoint Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_api_overview():
    """Test overview endpoint returns correct structure."""
    client = TestClient(app)
    response = client.get("/api/dashboard/overview")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "subject_profile_id" in data
    assert "subject_observation_ids" in data
    assert "investigative_priority" in data
    assert "risk_band" in data
    assert "status" in data
    assert "evidence_summary" in data
    assert "observation_count" in data
    assert "evidence_relationship_count" in data
    assert "investigation_path_count" in data
    assert "anomaly_context" in data
    assert "cluster_context" in data
    assert "provenance_context" in data
    assert "temporal_context" in data
    assert "assessment_id" in data
    assert "config_version" in data
    assert "interpretation" in data


if __name__ == "__main__":
    test_api_overview()
    print("test_step9a_api_overview.py PASSED")
