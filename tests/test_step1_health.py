"""Step 1 - foundation and health endpoints."""

from __future__ import annotations

import pytest

from backend.app.main import app


def test_root_endpoint_serves_the_investigator_dashboard(client):
    """The browser entry point is the dashboard, not a JSON greeting."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "BitTrace AI" in response.text


def test_welcome_endpoint_returns_welcome(client):
    """The original Step 1 greeting stays available as JSON."""
    response = client.get("/api/welcome")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to BitTrace AI backend"}


def test_health_endpoint_returns_exact_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    # The SIH26146 health contract is exact: no extra or renamed keys.
    assert response.json() == {"status": "ok", "service": "BitTrace AI"}


def test_health_reports_ok_status(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_health_declares_expected_service(client):
    body = client.get("/health").json()
    assert body["service"] == "BitTrace AI"


def test_health_is_available_repeatedly(client):
    """The health check must not be stateful or rate-limited."""
    for _ in range(3):
        assert client.get("/health").status_code == 200


def test_unknown_route_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_health_rejects_post(client):
    assert client.post("/health").status_code == 405


def test_root_rejects_post(client):
    assert client.post("/").status_code == 405


def test_openapi_schema_is_served(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/" in paths
    assert "/health" in paths


def test_application_metadata():
    assert app.title == "BitTrace AI Backend"
    assert app.version == "0.1.0"


def test_health_is_declared_in_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/health"]
    assert "get" in paths["/"]


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/", "text/html"),
        ("/health", "application/json"),
        ("/api/welcome", "application/json"),
    ],
)
def test_endpoints_return_expected_content_type(client, path, content_type):
    assert client.get(path).headers["content-type"].startswith(content_type)
