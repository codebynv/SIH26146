"""Step 9B Phase 2: Graph selection + filtering tests."""

import pytest
from pathlib import Path


def _html():
    p = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return p.read_text(encoding="utf-8")


def test_graph_filters_and_details_exist():
    html = _html()
    assert 'id="graph-filters"' in html
    assert 'id="graph-selection-details"' in html
    assert "No graph element selected" in html


def test_graph_container_exists():
    html = _html()
    assert 'class="graph-container"' in html


def test_phase1_search_still_intact():
    html = _html()
    assert 'id="global-search-input"' in html
    assert 'id="investigator-nav"' in html
