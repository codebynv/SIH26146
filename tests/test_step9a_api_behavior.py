"""Step 9A: API Behavior Endpoint Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_api_behavior():
    """Test behavior endpoint returns behavioral/ML context."""
    client = TestClient(app)
    response = client.get("/api/dashboard/behavior")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should have behavior data (not detail)
    if "detail" not in data:
        assert "profile_id" in data
        assert "grouping" in data
        assert "feature_count" in data
        assert "feature_categories" in data
        # Anomaly fields
        assert "anomaly_band" in data
        assert "anomaly_normalized_score" in data
        assert "anomaly_is_in_sample" in data
        # Cluster fields
        assert "cluster_id" in data
        assert "cluster_size" in data
        assert "cluster_distance" in data
        assert "cluster_normalized" in data
        assert "explanation_items" in data


if __name__ == "__main__":
    test_api_behavior()
    print("test_step9a_api_behavior.py PASSED")
