"""Same-origin serving of the Investigator Dashboard demo.

The demo is opened as a normal browser page at ``/``, so the backend has to
hand out the dashboard document and every asset it references from one origin.
The dashboard itself stays read-only: only the serving contract is under test.
"""

from __future__ import annotations

import re

import pytest

from backend.app.main import DASHBOARD_HTML, FRONTEND_DIR

_DASHBOARD_API = (
    "/api/dashboard/overview",
    "/api/dashboard/signals",
    "/api/dashboard/evidence",
    "/api/dashboard/paths",
    "/api/dashboard/behavior",
    "/api/dashboard/provenance",
    "/api/dashboard/all",
    "/api/dashboard/health",
)


def _referenced_assets() -> list[str]:
    """Every same-origin path index.html pulls in, in document order."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    return re.findall(r'(?:href|src)="(css/[^"]+|js/[^"]+)"', html)


class TestDashboardDocument:
    def test_root_serves_the_dashboard_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "<title>" in response.text
        assert 'id="dashboard-main"' in response.text
        assert 'id="cross-layer-panels"' in response.text

    def test_dashboard_markup_is_the_repository_file(self, client):
        assert client.get("/").content == DASHBOARD_HTML.read_bytes()

    def test_every_referenced_asset_is_served(self, client):
        assets = _referenced_assets()
        assert assets, "index.html should reference its own css and js"
        for asset in assets:
            response = client.get("/" + asset)
            assert response.status_code == 200, asset
            assert response.content, asset

    def test_javascript_is_served_with_a_script_content_type(self, client):
        response = client.get("/js/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]

    def test_dashboard_is_also_reachable_under_a_prefix(self, client):
        response = client.get("/dashboard/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    def test_the_welcome_greeting_is_still_available_as_json(self, client):
        response = client.get("/api/welcome")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to BitTrace AI backend"}


class TestServingDoesNotShadowTheApi:
    @pytest.mark.parametrize("path", _DASHBOARD_API)
    def test_dashboard_api_endpoints_still_answer(self, client, path):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

    def test_health_contract_is_untouched(self, client):
        assert client.get("/health").json() == {
            "status": "ok",
            "service": "BitTrace AI",
        }

    def test_ingest_route_is_untouched(self, client):
        assert client.get("/ingest").status_code == 405

    def test_unknown_routes_still_return_404(self, client):
        assert client.get("/does-not-exist").status_code == 404

    def test_root_still_rejects_post(self, client):
        assert client.post("/").status_code == 405


def test_frontend_directory_is_the_repository_frontend():
    assert FRONTEND_DIR.name == "frontend"
    assert (FRONTEND_DIR / "index.html").is_file()
    assert (FRONTEND_DIR / "js" / "app.js").is_file()


def test_root_falls_back_to_the_greeting_when_assets_are_absent(monkeypatch, client):
    """The Step 10 runtime image ships no frontend, and must stay usable there."""
    from backend.app import main

    monkeypatch.setattr(main, "DASHBOARD_HTML", main.FRONTEND_DIR / "absent.html")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to BitTrace AI backend"}
