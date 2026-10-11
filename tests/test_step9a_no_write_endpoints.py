"""Step 9A: No Write Endpoints Test"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_no_write_endpoints():
    """Test that no POST/PUT/PATCH/DELETE endpoints exist on /api/dashboard/."""
    client = TestClient(app)
    
    # Get all routes
    routes = [route for route in app.routes if hasattr(route, "path") and route.path.startswith("/api/dashboard")]
    
    write_methods = {"POST", "PUT", "PATCH", "DELETE"}
    
    for route in routes:
        if hasattr(route, "methods"):
            for method in route.methods:
                assert method not in write_methods, f"Found write method {method} on {route.path}"
    
    # Explicitly test each write method on known endpoints
    endpoints = [
        "/api/dashboard/overview",
        "/api/dashboard/signals",
        "/api/dashboard/evidence",
        "/api/dashboard/paths",
        "/api/dashboard/behavior",
        "/api/dashboard/provenance",
        "/api/dashboard/all",
        "/api/dashboard/health",
    ]
    
    for endpoint in endpoints:
        for method in write_methods:
            func = getattr(client, method.lower())
            response = func(endpoint)
            # Should return 405 Method Not Allowed or 404
            assert response.status_code in (405, 404), f"{method} {endpoint} should not be allowed"


if __name__ == "__main__":
    test_no_write_endpoints()
    print("test_step9a_no_write_endpoints.py PASSED")
