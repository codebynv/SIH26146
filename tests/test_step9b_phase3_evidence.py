"""Step 9B Phase 3: Evidence interaction tests.

The evidence chain is a client-side vanilla-JS feature, so the tests of record
are behavioural. This module executes ``tests/js/run_evidence.mjs``, which drives
the real ``frontend/js/evidence-chain.js`` against the DOM it renders into and
reports what it observed. Each scenario below is one pytest test, and the
assertion runs twice: once in JavaScript against the real implementation and once
in Python against that JavaScript's reported result.

Keeping the assertion in both places means a passing suite cannot come from a
stub, a mock, or a runner that silently stopped checking anything.

The scenario names below are the Phase 3 evidence behaviour checklist. They must
match ``CHECK_NAMES`` in ``tests/js/run_evidence.mjs`` exactly; a mismatch fails
loudly rather than quietly skipping a behaviour.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tests" / "js" / "run_evidence.mjs"
EVIDENCE_MODULE = REPO_ROOT / "frontend" / "js" / "evidence-chain.js"
PHASE3_SECTIONS = REPO_ROOT / "frontend" / "index.html"

# One test per required evidence behaviour.
CHECK_NAMES = [
    # Expansion and accessibility
    "expansion toggles collapsed to expanded and back",
    "aria-expanded is derived from state for every item",
    "expand all then collapse all moves every item",
    "click activates expansion and reports the new state",
    "Enter and Space both toggle, and neither double-toggles",
    # Filtering
    "filtering by evidence type returns only that type",
    "filtering by inference level returns only that level",
    "combined filters use AND semantics",
    "clearing filters restores the complete evidence set",
    # Selection
    "selecting a relationship resolves that exact record",
    "selection is deterministic and marked in the DOM",
    "source and target identifiers come from the selected record",
    # Graph synchronisation
    "evidence selection synchronises with the graph on exact id",
    "no graph relationship is fabricated when the id is absent",
    # Data integrity
    "source data is unchanged by filtering and selection",
]

# Scenarios owned by tests/js/run_graph_sync.mjs that this module asserts on.
# The remaining graph scenarios are asserted by tests/test_step9b_phase3_paths.py.
GRAPH_CHECK_NAMES = [
    "an id the graph does not hold is refused, not approximated",
    "only one graph element is selected at a time",
]

GRAPH_RUNNER = REPO_ROOT / "tests" / "js" / "run_graph_sync.mjs"

REQUIRED_EVIDENCE_FEATURES = [
    "filterEvidence",
    "toggleExpand",
    "expandAll",
    "collapseAll",
    "selectRelationship",
    "renderEvidenceChain",
    "initEvidenceChain",
    "syncToGraph",
]


@pytest.fixture(scope="module")
def runner_output():
    """Execute the JS behavioural runner once and parse its JSON report."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise the frontend behaviour")

    assert RUNNER.exists(), f"missing behavioural runner: {RUNNER}"

    completed = subprocess.run(
        [node, str(RUNNER)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PHASE3_JSON": "1"},
        check=False,
    )
    payload = completed.stdout.strip()
    assert payload, (
        "the behavioural runner produced no JSON report\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

    report = json.loads(payload)
    assert report["failures"] == 0, (
        "behavioural failures in frontend/js/evidence-chain.js:\n"
        + "\n".join(
            f"  - {r['name']}: {r.get('error')}"
            for r in report["results"]
            if not r["ok"]
        )
    )

    # The graph handshake runs in its own runner and is asserted on below.
    assert GRAPH_RUNNER.exists(), f"missing behavioural runner: {GRAPH_RUNNER}"
    graph_run = subprocess.run(
        [node, str(GRAPH_RUNNER)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PHASE3_JSON": "1"},
        check=False,
    )
    graph_payload = graph_run.stdout.strip()
    assert graph_payload, (
        "the graph synchronisation runner produced no JSON report\n"
        f"stdout:\n{graph_run.stdout}\nstderr:\n{graph_run.stderr}"
    )
    graph_report = json.loads(graph_payload)
    assert graph_report["failures"] == 0, (
        "behavioural failures in the graph synchronisation handshake:\n"
        + "\n".join(
            f"  - {r['name']}: {r.get('error')}"
            for r in graph_report["results"]
            if not r["ok"]
        )
    )

    report["graph"] = graph_report
    return report


def _check(runner_output, name):
    """Return the observed result of one scenario, failing if it did not pass."""
    matches = [
        result
        for result in runner_output["results"] + runner_output["graph"]["results"]
        if result["name"] == name
    ]
    assert matches, f"behaviour was never exercised: {name!r}"
    assert len(matches) == 1, f"behaviour was exercised more than once: {name!r}"
    result = matches[0]
    assert result["ok"], f"{name}: {result.get('error')}"
    return result


# --- Expansion and accessibility -------------------------------------------


def test_expansion_toggles_both_ways(runner_output):
    result = _check(runner_output, "expansion toggles collapsed to expanded and back")
    assert result["detail"]["initialExpanded"] is False
    assert result["detail"]["afterFirstClick"] is True
    assert result["detail"]["afterSecondClick"] is False


def test_aria_expanded_reflects_state_for_every_item(runner_output):
    result = _check(runner_output, "aria-expanded is derived from state for every item")
    detail = result["detail"]
    assert detail["itemsWithoutAria"] == 0
    assert detail["collapsedAria"] == detail["itemCount"] - detail["expandedCount"]
    assert detail["staleAriaCount"] == 0


def test_expand_all_and_collapse_all_move_every_item(runner_output):
    result = _check(runner_output, "expand all then collapse all moves every item")
    detail = result["detail"]
    assert detail["expandedAfterExpandAll"] == detail["itemCount"] > 0
    assert detail["expandedAfterCollapseAll"] == 0
    assert detail["allAriaAfterCollapse"] == "false"


def test_click_activates_expansion_and_reports_new_state(runner_output):
    result = _check(runner_output, "click activates expansion and reports the new state")
    detail = result["detail"]
    assert detail["clicked"] is True
    # Both activation directions are reported, and the DOM follows the last one.
    assert detail["reportedTransitions"] == ["expanded:true", "expanded:false"]
    assert detail["reported"] == {"id": "rel-observed-ip", "nowExpanded": False}
    assert detail["ariaAfterSecondClick"] == "false"


def test_keyboard_enter_and_space_toggle_without_double_toggling(runner_output):
    result = _check(runner_output, "Enter and Space both toggle, and neither double-toggles")
    detail = result["detail"]
    assert detail["enterExpanded"] == "true"
    assert detail["spaceAfterEnterExpanded"] == "false"
    assert detail["nonActivationKeyIgnored"] is True
    assert detail["toggles"] == 2, "Enter and Space each toggle exactly once"
    assert detail["keysTested"] == ["Enter", "Space", "a"]


# --- Filtering --------------------------------------------------------------


def test_filter_by_evidence_type_returns_only_that_type(runner_output):
    result = _check(runner_output, "filtering by evidence type returns only that type")
    detail = result["detail"]
    assert detail["matched"] > 0
    assert set(detail["matchedEvidenceTypes"]) == {detail["filterValue"]}
    assert detail["matched"] < detail["total"]


def test_filter_by_inference_level_returns_only_that_level(runner_output):
    result = _check(runner_output, "filtering by inference level returns only that level")
    detail = result["detail"]
    assert detail["matched"] > 0
    assert set(detail["matchedInferenceLevels"]) == {detail["filterValue"]}
    assert detail["matched"] < detail["total"]


def test_combined_filters_use_and_semantics(runner_output):
    result = _check(runner_output, "combined filters use AND semantics")
    detail = result["detail"]
    # AND means the combined set is exactly the intersection of either single
    # filter, so it can be no larger than either one.
    assert detail["combinedEqualsIntersection"] is True
    assert detail["sameIpAndCorrelated"] == len(
        set(detail["sameIpOnly"]) & set(detail["correlatedOnly"])
    )
    assert detail["sameIpAndCorrelated"] <= len(detail["sameIpOnly"])
    assert detail["sameIpAndCorrelated"] <= len(detail["correlatedOnly"])


def test_clearing_filters_restores_the_complete_evidence_set(runner_output):
    result = _check(runner_output, "clearing filters restores the complete evidence set")
    detail = result["detail"]
    assert detail["renderedAfterClear"] == detail["total"]
    assert detail["renderedWhenFiltered"] < detail["total"]


# --- Selection --------------------------------------------------------------


def test_selecting_a_relationship_resolves_that_exact_record(runner_output):
    result = _check(runner_output, "selecting a relationship resolves that exact record")
    detail = result["detail"]
    assert detail["selectedRelationshipId"] == detail["requestedId"]
    assert detail["resolvedRecordMatches"] is True
    assert detail["missingIdResolvesToNull"] is True
    assert detail["prefixIdResolvesToNull"] is True
    assert detail["idIsCaseSensitive"] is True


def test_selection_is_deterministic_and_marked_in_the_dom(runner_output):
    result = _check(runner_output, "selection is deterministic and marked in the DOM")
    detail = result["detail"]
    assert detail["selectedInDom"] == 1
    assert detail["markedRelationshipId"] == detail["selectedId"]
    assert detail["selectedInDom"] == 1, "selection must be a single record, not a set"
    assert detail["repeatSelectionSameRecord"] is True
    assert detail["selectedInDomAfterRepeat"] == 1


def test_source_and_target_identifiers_come_from_the_selected_record(runner_output):
    result = _check(runner_output, "source and target identifiers come from the selected record")
    detail = result["detail"]
    assert detail["renderedSourceIds"] == detail["recordSourceIds"]
    assert detail["renderedTargetIds"] == detail["recordTargetIds"]
    assert detail["renderedSourceIds"], "the selected record has sources to show"
    assert detail["renderedTargetIds"], "the selected record has targets to show"


# --- Graph synchronisation --------------------------------------------------


def test_evidence_selection_synchronises_graph_on_exact_id(runner_output):
    result = _check(runner_output, "evidence selection synchronises with the graph on exact id")
    detail = result["detail"]
    assert detail["requestedId"] in detail["graphRelationshipIds"]
    assert detail["sync"]["synchronized"] is True
    assert detail["sync"]["relationshipId"] == detail["requestedId"]
    assert detail["sync"]["graphRecord"] is not None


def test_no_graph_relationship_is_fabricated_for_an_absent_id(runner_output):
    result = _check(runner_output, "no graph relationship is fabricated when the id is absent")
    detail = result["detail"]
    assert detail["sync"]["synchronized"] is False
    assert detail["sync"]["graphRecord"] is None, "no graph record may be invented"
    assert detail["sync"]["relationshipId"] == detail["requestedId"], (
        "the requested id is echoed so the panel can say which id has no match"
    )
    assert detail["sync"]["relationshipId"] not in detail["graphRelationshipIds"]
    assert detail["graphStillExcludesIt"] is True


# --- Data integrity ---------------------------------------------------------


def test_source_data_is_unchanged_by_filtering_and_selection(runner_output):
    result = _check(runner_output, "source data is unchanged by filtering and selection")
    detail = result["detail"]
    assert detail["recordsUnchanged"] is True
    assert detail["recordsLengthUnchanged"] is True
    assert detail["orderUnchanged"] is True


# --- Graph synchronisation, evidence side -----------------------------------
#
# The same runner that drives the evidence scenarios also wires the three
# sections together, so these assertions cover the live handshake rather than a
# helper called in isolation.


def test_selecting_evidence_selects_that_graph_relationship(runner_output):
    detail = _check(
        runner_output, "clicking an evidence record selects that exact graph relationship"
    )["detail"]
    assert detail["selectedEdges"] == ["rel-observed-ip"]
    assert detail["selection"]["relationshipId"] == "rel-observed-ip"
    assert detail["selection"]["observationId"] is None, (
        "selecting an edge must not also select a node"
    )


def test_graph_aria_selected_agrees_with_the_evidence_selection(runner_output):
    detail = _check(
        runner_output, "aria-selected on the graph agrees with the selected evidence"
    )["detail"]
    assert detail["ariaSelectedCount"] == 1
    assert detail["afterClear"] == 0, "clearing must clear aria-selected everywhere"


def test_graph_click_selects_the_matching_evidence_record(runner_output):
    detail = _check(
        runner_output, "a graph relationship click selects the matching evidence record"
    )["detail"]
    assert detail["evidenceSelection"] == "rel-inferred-address"
    assert detail["markedInDom"] == 1


def test_only_one_graph_element_is_selected_at_a_time(runner_output):
    detail = _check(runner_output, "only one graph element is selected at a time")["detail"]
    assert detail["afterNode"] == {"edges": [], "nodes": ["obs-b1"]}, (
        "selecting a node clears the selected edge"
    )


def test_an_absent_id_is_refused_rather_than_approximated(runner_output):
    detail = _check(runner_output, "an id the graph does not hold is refused, not approximated")[
        "detail"
    ]
    assert detail["refused"] >= 8, "prefixes, case variants and indexes must all be refused"
    assert detail["selectionRemainedEmpty"] is True


# --- Implementation present -------------------------------------------------


def test_evidence_module_exports_the_required_behaviour():
    source = EVIDENCE_MODULE.read_text(encoding="utf-8")
    for name in REQUIRED_EVIDENCE_FEATURES:
        assert f"export function {name}" in source, f"missing export: {name}"


def test_every_required_behaviour_has_a_test(runner_output):
    """The checklist must be unique and must match what the runner really ran.

    Comparing the list against itself would always pass, so this asserts the
    names are distinct and that the runner reported exactly these scenarios.
    """
    assert len(set(CHECK_NAMES)) == len(CHECK_NAMES), "behaviour list must be unique"
    assert len(CHECK_NAMES) >= 8, "Phase 3 requires at least 8 behavioural tests"
    assert (
        len(set(GRAPH_CHECK_NAMES)) == len(GRAPH_CHECK_NAMES)
    ), "graph behaviour list must be unique"

    reported = {result["name"] for result in runner_output["results"]}
    assert reported == set(CHECK_NAMES), (
        "CHECK_NAMES is out of step with tests/js/run_evidence.mjs; "
        f"only in checklist: {sorted(set(CHECK_NAMES) - reported)}; "
        f"only in runner: {sorted(reported - set(CHECK_NAMES))}"
    )

    graph_reported = {result["name"] for result in runner_output["graph"]["results"]}
    assert set(GRAPH_CHECK_NAMES) <= graph_reported, (
        "these behaviours are not exercised by tests/js/run_graph_sync.mjs: "
        f"{sorted(set(GRAPH_CHECK_NAMES) - graph_reported)}"
    )


def test_phase3_section_exists_in_the_page():
    html = PHASE3_SECTIONS.read_text(encoding="utf-8")
    assert 'id="evidence-chain-content"' in html
    assert "Evidence Chain" in html


def test_evidence_controls_exist_in_the_page():
    html = PHASE3_SECTIONS.read_text(encoding="utf-8")
    for element_id in [
        "evidence-filter-type",
        "evidence-filter-inference",
        "evidence-clear-filters",
        "evidence-expand-all",
        "evidence-collapse-all",
        "evidence-selection-details",
    ]:
        assert f'id="{element_id}"' in html, f"missing control: {element_id}"


def test_earlier_phases_still_intact():
    html = PHASE3_SECTIONS.read_text(encoding="utf-8")
    assert 'id="global-search-input"' in html
    assert 'id="investigator-nav"' in html
    assert 'id="graph-filters"' in html
    assert 'class="graph-container"' in html