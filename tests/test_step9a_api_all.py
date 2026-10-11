"""Step 9A: API All Endpoint Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_api_all():
    """Test all endpoint returns complete dashboard data."""
    client = TestClient(app)
    response = client.get("/api/dashboard/all")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "overview" in data
    assert "signals" in data
    assert "evidence" in data
    assert "paths" in data
    assert "behavior" in data
    assert "provenance" in data
    assert "generated_at" in data
    assert "version" in data
    
    # Check counts
    assert len(data["signals"]) == 11
    assert len(data["evidence"]) >= 3
    assert len(data["paths"]) >= 1
    assert data["behavior"] is not None
    assert data["provenance"] is not None
    
    # Deterministic ordering
    signal_ids = [s["signal_id"] for s in data["signals"]]
    assert signal_ids == sorted(signal_ids)
    
    evidence_ids = [e["relationship_id"] for e in data["evidence"]]
    assert evidence_ids == sorted(evidence_ids)
    
    path_ids = [p["path_id"] for p in data["paths"]]
    assert path_ids == sorted(path_ids)


if __name__ == "__main__":
    test_api_all()
    print("test_step9a_api_all.py PASSED")
