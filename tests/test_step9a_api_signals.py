"""Step 9A: API Signals Endpoint Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_api_signals():
    """Test signals endpoint returns all 11 signals."""
    client = TestClient(app)
    response = client.get("/api/dashboard/signals")
    
    assert response.status_code == 200
    data = response.json()
    assert "signals" in data
    signals = data["signals"]
    
    assert len(signals) == 11
    
    for signal in signals:
        assert "signal_id" in signal
        assert "signal_type" in signal
        assert "family" in signal
        assert "role" in signal
        assert "status" in signal
        assert "source_step" in signal
        assert "source_id" in signal
        assert "subject_id" in signal
        assert "explanation" in signal
        assert "evidence_refs" in signal
        assert "inference_levels" in signal
        assert "provenance_refs" in signal
        assert "lineage" in signal
        assert "context" in signal
        
        assert signal["role"] in ("PRIORITY_SIGNAL", "EVIDENCE_STRENGTH", "CONTEXT_ONLY")
        assert signal["status"] in ("PRESENT", "ABSENT", "UNAVAILABLE", "NOT_APPLICABLE")


if __name__ == "__main__":
    test_api_signals()
    print("test_step9a_api_signals.py PASSED")
