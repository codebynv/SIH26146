"""Step 9A: API Evidence Endpoint Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_api_evidence():
    """Test evidence endpoint returns evidence relationships."""
    client = TestClient(app)
    response = client.get("/api/dashboard/evidence")
    
    assert response.status_code == 200
    data = response.json()
    assert "evidence" in data
    evidence = data["evidence"]
    
    assert len(evidence) >= 3
    
    for e in evidence:
        assert "relationship_id" in e
        assert "source_observation_ids" in e
        assert "target_observation_ids" in e
        assert "evidence_type" in e
        assert "inference_level" in e
        assert "confidence_value" in e
        assert "confidence_basis" in e
        assert "rationale" in e
        assert "rule_id" in e
        assert "rule_version" in e
        assert "provenance_refs" in e
        assert "created_at" in e
        
        assert e["inference_level"] in ("OBSERVED", "CORRELATED", "INFERRED", "REQUIRES_VERIFICATION")
        assert 0.0 <= e["confidence_value"] <= 1.0


if __name__ == "__main__":
    test_api_evidence()
    print("test_step9a_api_evidence.py PASSED")
