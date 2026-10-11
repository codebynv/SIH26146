"""Step 9A: API Health Endpoint Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_dashboard_health():
    """Test dashboard health endpoint."""
    client = TestClient(app)
    response = client.get("/api/dashboard/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "BitTrace AI Dashboard"
    assert data["version"] == "STEP9A/1"


if __name__ == "__main__":
    test_dashboard_health()
    print("test_step9a_api_health.py PASSED")
