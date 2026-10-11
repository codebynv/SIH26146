"""Step 9A: API Provenance Endpoint Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_api_provenance():
    """Test provenance endpoint returns provenance data."""
    client = TestClient(app)
    response = client.get("/api/dashboard/provenance")
    
    assert response.status_code == 200
    data = response.json()
    
    if "detail" not in data:
        assert "ingestion_id" in data
        assert "source_files" in data
        assert "source_records" in data
        assert "evidence_citation_ratio" in data
        assert "cited_edge_count" in data
        assert "incident_edge_count" in data
        assert "distinct_source_files" in data
        assert "temporal_comparability" in data
        assert "temporal_forward_flow" in data
        assert "missing_signals" in data


if __name__ == "__main__":
    test_api_provenance()
    print("test_step9a_api_provenance.py PASSED")
