"""Step 9A: API Paths Endpoint Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_api_paths():
    """Test paths endpoint returns investigation paths."""
    client = TestClient(app)
    response = client.get("/api/dashboard/paths")
    
    assert response.status_code == 200
    data = response.json()
    assert "paths" in data
    paths = data["paths"]
    
    assert len(paths) >= 1
    
    for path in paths:
        assert "path_id" in path
        assert "source_observation_id" in path
        assert "target_observation_id" in path
        assert "hop_count" in path
        assert "node_count" in path
        assert "observation_ids" in path
        assert "relationship_ids" in path
        assert "temporal_status" in path
        assert "temporal_flow" in path
        assert "duration_seconds" in path
        assert "evidence_summary" in path
        assert "search_status" in path
        
        assert path["hop_count"] >= 1
        assert path["node_count"] == path["hop_count"] + 1
        assert len(path["observation_ids"]) == path["node_count"]
        assert len(path["relationship_ids"]) == path["hop_count"]


if __name__ == "__main__":
    test_api_paths()
    print("test_step9a_api_paths.py PASSED")
