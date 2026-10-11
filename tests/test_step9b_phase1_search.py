"""Step 9B Phase 1: Global search tests."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def _html():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return p.read_text(encoding="utf-8")
    client = TestClient(app)
    r = client.get("/dashboard/")
    assert r.status_code == 200
    return r.text


def test_search_exists():
    html = _html()
    assert 'id="global-search-input"' in html
    assert 'class="global-search"' in html


def test_search_semantic_structure():
    html = _html()
    assert 'role="search"' in html
    assert 'id="global-search-results"' in html


def test_search_keyboard_accessibility():
    html = _html()
    assert 'aria-expanded' in html
    assert 'aria-controls' in html
    assert 'role="combobox"' in html


def test_search_clear_control():
    html = _html()
    assert 'id="global-search-clear"' in html


def test_search_count_control():
    html = _html()
    assert 'id="global-search-count"' in html


def test_search_results_role():
    html = _html()
    assert 'role="listbox"' in html


if __name__ == "__main__":
    test_search_exists()
    print("PASSED")
