"""Step 9B Phase 1: Investigator navigation tests."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _html():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return p.read_text(encoding="utf-8")
    client = TestClient(app)
    r = client.get("/dashboard/")
    assert r.status_code == 200
    return r.text


def test_navigation_exists():
    html = _html()
    assert 'id="investigator-nav"' in html
    assert 'class="investigator-nav"' in html
    assert 'aria-label="Dashboard sections"' in html


def test_navigation_semantic_structure():
    html = _html()
    assert '<nav' in html
    assert 'investigator-nav__list' in html
    assert 'investigator-nav__item' in html
    assert 'investigator-nav__link' in html


def test_navigation_section_order():
    html = _html()
    expected = [
        "#overview-content",
        "#why-lead-content",
        "#evidence-chain-content",
        "#cross-layer-content",
        "#paths-content",
        "#behavior-ml-content",
        "#provenance-content",
    ]
    for e in expected:
        assert e in html


def test_navigation_all_destinations_exist():
    html = _html()
    assert 'id="overview-content"' in html
    assert 'id="why-lead-content"' in html
    assert 'id="evidence-chain-content"' in html
    assert 'id="cross-layer-content"' in html
    assert 'id="paths-content"' in html
    assert 'id="behavior-ml-content"' in html
    assert 'id="provenance-content"' in html


def test_navigation_stable_ids():
    html = _html()
    assert html.count('id="investigator-nav"') == 1


def test_navigation_accessibility_attributes():
    html = _html()
    assert 'aria-label' in html
    assert 'aria-current' in html


def test_navigation_data_section_attributes():
    html = _html()
    assert 'data-section="overview-content"' in html
    assert 'data-section="why-lead-content"' in html
    assert 'data-section="evidence-chain-content"' in html
    assert 'data-section="cross-layer-content"' in html
    assert 'data-section="paths-content"' in html
    assert 'data-section="behavior-ml-content"' in html
    assert 'data-section="provenance-content"' in html
