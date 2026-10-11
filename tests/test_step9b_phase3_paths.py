"""Step 9B Phase 3: Investigation path tests.

The paths section is client-side vanilla JS, so the tests of record are
behavioural. This module executes ``tests/js/run_paths.mjs``, which drives the
real ``frontend/js/paths.js`` and ``frontend/js/temporal.js`` against the DOM
they render into and reports what they observed. Each scenario below is one
pytest test, and each assertion runs twice: once in JavaScript against the real
implementation and once in Python against that JavaScript's reported result.

The temporal scenarios assert that Step 7's answers survive into the frontend.
Where Step 7 declined to order two timestamps, the test asserts the frontend
still reports NOT_COMPARABLE or UNKNOWN rather than inventing an ordering. No
path-level score, ranking, or attribution is permitted anywhere in the
comparison output.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tests" / "js" / "run_paths.mjs"
GRAPH_RUNNER = REPO_ROOT / "tests" / "js" / "run_graph_sync.mjs"
PATHS_MODULE = REPO_ROOT / "frontend" / "js" / "paths.js"
TEMPORAL_MODULE = REPO_ROOT / "frontend" / "js" / "temporal.js"
PAGE = REPO_ROOT / "frontend" / "index.html"

# One test per required path behaviour.
CHECK_NAMES = [
    # Selection
    "selecting a path selects exactly that path",
    "the selected path's nodes and transitions are shown in order",
    "selecting a second path replaces the first and swaps the body",
    # Details
    "path details report evidence summary, temporal status and window",
    # Comparison
    "two distinct paths can be selected for comparison",
    "comparison output preserves both paths' sequences and evidence",
    "comparison produces no path-level score or ranking",
    "a comparison needs two distinct paths",
    "a third path is refused rather than silently replacing one",
    "clearing the comparison empties it and restores normal browsing",
    # Accessibility
    "keyboard selection activates a path and updates aria-expanded",
    "comparison controls are labelled and keyboard operable",
    # Temporal
    "comparable aware timestamps keep chronological order",
    "NOT_COMPARABLE stays NOT_COMPARABLE with no elapsed time",
    "missing timestamps stay UNKNOWN",
    "a naive timestamp is not silently read as UTC",
    "no temporal ordering is fabricated for an unparsable entry",
    # Graph synchronisation
    "a path node synchronises with the graph on exact observation_id",
    # Data integrity
    "path data is unchanged by selection, comparison and rendering",
    "comparison rendering omits any score and keeps selection order",
]

# Scenarios owned by tests/js/run_graph_sync.mjs that this module asserts on.
# The remaining graph scenarios are asserted by tests/test_step9b_phase3_evidence.py.
GRAPH_CHECK_NAMES = [
    "selecting a path highlights its exact nodes and transitions",
    "a path highlights no observation the graph does not contain",
    "clicking a graph node selects the path that contains it",
    "a node shared by two paths resolves to the same path every time",
    "a graph node belonging to no path selects no path",
    "clearing the graph clears the evidence selection",
    "comparing paths clears the graph rather than highlighting one",
    "applyGraphSelection ignores ids with no element",
    "a store redraw restores the selection and drops ids the new data lacks",
]

REQUIRED_PATH_FEATURES = [
    "selectPath",
    "clearSelection",
    "toggleComparison",
    "comparisonPaths",
    "buildComparison",
    "resetComparison",
    "orderedNodes",
    "orderedTransitions",
    "evidenceSummary",
    "syncNodeToGraph",
    "syncRelationshipToGraph",
    "renderPaths",
    "renderComparison",
    "renderTimeline",
    "initPaths",
]

REQUIRED_TEMPORAL_FEATURES = [
    "parseTemporalFlowEntry",
    "parseTemporalFlow",
    "isComparableStep",
    "pathTemporalStatus",
    "pathTemporalWindow",
    "isChronologicallyOrdered",
    "buildTimeline",
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
        "behavioural failures in frontend/js/paths.js:\n"
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


# --- Step 7 parity ----------------------------------------------------------

PARITY_RUNNER = REPO_ROOT / "tests" / "js" / "run_temporal_parity.mjs"


def _flow_string(source_id, target_id, evidence_type, transition):
    """Render a transition exactly as backend/app/dashboard/adapter.py does.

    The frontend parser is only required to read the shape the API really emits,
    so the string here is built with the same f-string the adapter uses rather
    than by hand.
    """
    return (
        f"{source_id} -> {target_id} ({evidence_type.value}, "
        f"{transition.relation.value}, elapsed={transition.elapsed_seconds}s)"
    )


def _step7_cases():
    """Build temporal cases by calling the real Step 7 functions."""
    from datetime import datetime, timezone

    from backend.app.investigation.models import EvidenceType
    from backend.app.investigation.temporal import build_temporal_transition

    def aware(text):
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)

    def naive(text):
        return datetime.fromisoformat(text)

    scenarios = [
        # label, source, target
        ("aware after", aware("2026-10-01T10:00:00"), aware("2026-10-01T16:00:00")),
        ("aware before", aware("2026-10-01T16:00:00"), aware("2026-10-01T10:00:00")),
        ("aware same", aware("2026-10-01T10:00:00"), aware("2026-10-01T10:00:00")),
        ("aware across zones", aware("2026-10-01T10:00:00"), aware("2026-10-01T12:30:00")),
        ("aware negative elapsed", aware("2026-10-02T10:00:00"), aware("2026-10-01T10:00:00")),
        ("naive pair", naive("2026-10-01T10:00:00"), naive("2026-10-01T11:00:00")),
        ("mixed aware then naive", aware("2026-10-01T10:00:00"), naive("2026-10-01T11:00:00")),
        ("mixed naive then aware", naive("2026-10-01T10:00:00"), aware("2026-10-01T11:00:00")),
        ("missing target", aware("2026-10-01T10:00:00"), None),
        ("missing source", None, aware("2026-10-01T10:00:00")),
        ("both missing", None, None),
        ("sub-second aware", aware("2026-10-01T10:00:00"), aware("2026-10-01T10:00:00.25")),
    ]

    cases = []
    for index, (label, source, target) in enumerate(scenarios):
        transition = build_temporal_transition(
            f"obs-{index}-a", f"obs-{index}-b", source, target
        )
        text = _flow_string(
            f"obs-{index}-a",
            f"obs-{index}-b",
            EvidenceType.GEO_COINCIDENCE,
            transition,
        )
        cases.append(
            {
                "label": label,
                "flow": [text],
                "observationIds": [f"obs-{index}-a", f"obs-{index}-b"],
                "relationshipIds": [f"rel-{index}"],
                "temporalStatus": "COMPARABLE" if transition.comparable else "NOT_COMPARABLE",
                "durationSeconds": transition.elapsed_seconds,
                "expected": {
                    "relation": transition.relation.value,
                    "comparable": transition.comparable,
                    "elapsedSeconds": transition.elapsed_seconds,
                },
            }
        )
    return cases


@pytest.fixture(scope="module")
def step7_parity():
    """Run the frontend temporal module against real Step 7 output."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise the frontend behaviour")

    assert PARITY_RUNNER.exists(), f"missing parity harness: {PARITY_RUNNER}"

    cases = _step7_cases()
    completed = subprocess.run(
        [node, str(PARITY_RUNNER)],
        cwd=str(REPO_ROOT),
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    payload = completed.stdout.strip()
    assert payload, (
        "the parity harness produced no report\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return list(zip(cases, json.loads(payload)))


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


# --- Selection --------------------------------------------------------------


def test_selecting_a_path_selects_exactly_that_path(runner_output):
    detail = _check(runner_output, "selecting a path selects exactly that path")["detail"]
    assert detail["firstSelection"] == "path-alpha"
    assert detail["secondSelection"] == "path-beta"
    assert detail["clearedSelection"] is None


def test_selected_path_shows_its_nodes_and_transitions_in_order(runner_output):
    detail = _check(
        runner_output, "the selected path's nodes and transitions are shown in order"
    )["detail"]
    assert detail["pathId"] == "path-alpha"
    assert detail["nodes"] == ["obs-a1", "obs-b1", "obs-c2"]
    assert detail["transitions"] == ["rel-observed-ip", "rel-correlated-amount"]


def test_selecting_a_second_path_replaces_the_first(runner_output):
    detail = _check(
        runner_output, "selecting a second path replaces the first and swaps the body"
    )["detail"]
    assert detail["switchedTo"] == "path-beta"
    assert detail["ariaBeta"] == "true", "the newly selected path must be expanded"
    assert detail["ariaAlpha"] == "false", "the previous path must be collapsed"
    assert detail["betaNodes"], "the second path's own nodes must be rendered"


# --- Details ----------------------------------------------------------------


def test_path_details_report_evidence_summary_and_temporal_state(runner_output):
    detail = _check(
        runner_output, "path details report evidence summary, temporal status and window"
    )["detail"]
    assert detail["pathA"]["status"] == "COMPARABLE"
    assert detail["pathA"]["duration"] == 1800.0
    assert detail["pathC"]["status"] == "UNKNOWN"
    assert detail["pathC"]["known"] is False, "an unknown window must not be reported as known"


# --- Comparison -------------------------------------------------------------


def test_two_distinct_paths_can_be_compared(runner_output):
    detail = _check(runner_output, "two distinct paths can be selected for comparison")[
        "detail"
    ]
    assert detail["comparePathIds"] == ["path-alpha", "path-beta"]
    assert detail["distinctCount"] == 2
    assert detail["selectionUnaffectedByCompare"] is True, (
        "comparing must not change which path is selected"
    )


def test_comparison_preserves_both_paths_sequences_and_evidence(runner_output):
    detail = _check(
        runner_output, "comparison output preserves both paths' sequences and evidence"
    )["detail"]
    assert detail["pathIds"] == ["path-alpha", "path-gamma"]
    assert detail["alphaNodes"] == ["obs-a1", "obs-b1", "obs-c2"]
    assert detail["gammaNodes"] == ["obs-b2", "obs-c1"]
    assert detail["alphaRelations"] == ["AFTER", "BEFORE"]
    assert detail["gammaRelation"] == "UNKNOWN", (
        "a path with no comparable timestamps must not gain an ordering"
    )


def test_comparison_carries_no_score_or_ranking(runner_output):
    detail = _check(runner_output, "comparison produces no path-level score or ranking")[
        "detail"
    ]
    assert detail["noScoreFields"] is True
    # Order follows the investigator's selection, not any notion of merit.
    assert detail["pathIds"] == ["path-alpha", "path-beta"]
    assert detail["reversedPathIds"] == ["path-beta", "path-alpha"]


def test_comparison_requires_two_distinct_paths(runner_output):
    _check(runner_output, "a comparison needs two distinct paths")
    # The scenario asserts internally that an empty set, a single path, and a
    # repeated toggle all produce no comparison.


def test_a_third_path_is_refused_rather_than_replacing_one(runner_output):
    detail = _check(
        runner_output, "a third path is refused rather than silently replacing one"
    )["detail"]
    assert detail["thirdRefused"] is True
    assert detail["comparePathIds"] == ["path-alpha", "path-beta"], (
        "the investigator's two choices must survive a third attempt"
    )


def test_clearing_the_comparison_restores_normal_browsing(runner_output):
    detail = _check(
        runner_output, "clearing the comparison empties it and restores normal browsing"
    )["detail"]
    assert detail["comparePathIdsAfterReset"] == []
    assert detail["cardsVisibleAfterReset"] == 3, "all paths must be listed again"
    assert detail["selectionPreserved"] == "path-alpha", (
        "clearing a comparison must not discard the selected path"
    )


# --- Accessibility ----------------------------------------------------------


def test_keyboard_selection_activates_a_path(runner_output):
    detail = _check(runner_output, "keyboard selection activates a path and updates aria-expanded")[
        "detail"
    ]
    assert detail["enterSelects"] == "path-beta"
    assert detail["spaceSelects"] == "path-gamma"
    assert detail["nonActivationKeyIgnored"] is True


def test_comparison_controls_are_labelled_and_keyboard_operable(runner_output):
    detail = _check(runner_output, "comparison controls are labelled and keyboard operable")[
        "detail"
    ]
    assert detail["compareButtonLabel"] == "Compare path path-alpha"
    assert detail["ariaPressedAfterCompare"] is True
    assert detail["resetViaKeyboard"] is True


# --- Temporal ---------------------------------------------------------------


def test_comparable_aware_timestamps_keep_chronological_order(runner_output):
    detail = _check(runner_output, "comparable aware timestamps keep chronological order")[
        "detail"
    ]
    assert detail["status"] == "COMPARABLE"
    assert detail["relations"] == ["AFTER", "BEFORE"]
    assert detail["elapsed"] == [3600.0, -1800.0]


def test_not_comparable_stays_not_comparable(runner_output):
    detail = _check(runner_output, "NOT_COMPARABLE stays NOT_COMPARABLE with no elapsed time")[
        "detail"
    ]
    assert detail["status"] == "NOT_COMPARABLE"
    assert detail["relation"] == "NOT_COMPARABLE"
    assert detail["elapsedSeconds"] is None, "no elapsed time may be computed"
    assert detail["windowKnown"] is False


def test_missing_timestamps_stay_unknown(runner_output):
    detail = _check(runner_output, "missing timestamps stay UNKNOWN")["detail"]
    assert detail["status"] == "UNKNOWN"
    assert detail["relation"] == "UNKNOWN"
    assert detail["elapsedSeconds"] is None
    assert detail["emptyFlowIsNotOrdered"] is True


def test_naive_timestamps_are_not_silently_read_as_utc(runner_output):
    detail = _check(runner_output, "a naive timestamp is not silently read as UTC")["detail"]
    assert detail["naiveRelation"] == "AFTER", "the recorded relation is still shown"
    assert detail["mixedRelation"] == "NOT_COMPARABLE"
    assert detail["partialStatus"] == "PARTIALLY_COMPARABLE"
    assert detail["comparability"] == [False, True], (
        "one incomparable step must not be presented as an ordering"
    )


def test_no_temporal_ordering_is_fabricated_for_unparsable_input(runner_output):
    detail = _check(runner_output, "no temporal ordering is fabricated for an unparsable entry")[
        "detail"
    ]
    assert detail["unparsableIsNull"] is True
    assert detail["inventedRelationRejected"] is True, (
        "a relation outside Step 7's vocabulary must not be accepted"
    )
    assert detail["garbledStepStanding"] in {"UNKNOWN", "NOT_COMPARABLE"}


# --- Graph synchronisation --------------------------------------------------


def test_path_node_synchronises_with_graph_on_exact_id(runner_output):
    detail = _check(
        runner_output, "a path node synchronises with the graph on exact observation_id"
    )["detail"]
    assert detail["presentNode"] == "obs-c1"
    assert detail["absentNode"] is None, "no graph node may be invented"
    assert detail["prefixRejected"] is True, "a prefix must not match a node"
    assert detail["indexRejected"] is True, "an array index must not match a node"
    assert detail["relationshipSync"] is True


# --- Graph synchronisation, paths side --------------------------------------


def test_selecting_a_path_highlights_its_exact_nodes_and_transitions(runner_output):
    detail = _check(runner_output, "selecting a path highlights its exact nodes and transitions")[
        "detail"
    ]
    assert detail["nodes"] == ["obs-a1", "obs-b1", "obs-c2"]
    assert detail["edges"] == ["rel-correlated-amount", "rel-observed-ip"]


def test_a_path_highlights_nothing_the_graph_does_not_contain(runner_output):
    detail = _check(runner_output, "a path highlights no observation the graph does not contain")[
        "detail"
    ]
    assert detail["highlighted"] == ["obs-b2", "obs-c1"]
    assert detail["graphNodeCount"] > 0


def test_a_graph_node_selects_the_path_containing_it(runner_output):
    detail = _check(runner_output, "clicking a graph node selects the path that contains it")[
        "detail"
    ]
    assert detail["selectedPath"] == "path-beta"


def test_a_shared_node_resolves_deterministically(runner_output):
    detail = _check(runner_output, "a node shared by two paths resolves to the same path every time")[
        "detail"
    ]
    attempts = detail["attempts"]
    assert len(attempts) >= 3, "the scenario must click more than once to prove anything"
    assert len(set(attempts)) == 1, (
        "clicking the same node repeatedly must not resolve to a different path"
    )
    assert attempts[0] == detail["resolvedTo"]
    assert attempts[0] is not None, "a shared node must still resolve to some path"


def test_a_node_in_no_path_selects_no_path(runner_output):
    detail = _check(runner_output, "a graph node belonging to no path selects no path")["detail"]
    assert detail["selectedPath"] == "path-alpha", (
        "an observation outside every path must leave the selection alone"
    )


def test_comparing_paths_clears_the_graph(runner_output):
    detail = _check(
        runner_output, "comparing paths clears the graph rather than highlighting one"
    )["detail"]
    assert detail["selectedEdges"] == []
    assert detail["comparison"] == ["path-alpha", "path-beta"], (
        "clearing the graph must not disturb the comparison set"
    )


def test_highlighting_ignores_ids_with_no_graph_element(runner_output):
    detail = _check(runner_output, "applyGraphSelection ignores ids with no element")["detail"]
    assert detail["applied"] == {"relationships": ["rel-observed-ip"], "observations": ["obs-a1"]}
    assert detail["withNoElements"] == {"relationships": [], "observations": []}


def test_selection_survives_a_store_redraw(runner_output):
    detail = _check(
        runner_output, "a store redraw restores the selection and drops ids the new data lacks"
    )["detail"]
    assert detail["survivedRedraw"] is True, (
        "a data refresh must not silently drop the user's graph selection"
    )
    assert detail["vanishedIdDropped"] is True, (
        "an id the new data no longer holds must be dropped, not re-pointed"
    )
    assert detail["selectionAfterDrop"] == {"relationshipId": None, "observationId": None}


# --- Step 7 parity ---------------------------------------------------------


def test_frontend_temporal_matches_step7_decisions(step7_parity):
    """Every relation, comparability flag and elapsed time must match Step 7.

    The cases are built by calling Step 7 directly, so this fails if the
    frontend re-decides anything rather than reporting what Step 7 established.
    """
    assert step7_parity, "the parity harness reported no cases"

    for case, observed in step7_parity:
        label = case["label"]
        expected = case["expected"]
        entry = observed["entries"][0]

        assert entry["unparsed"] is False, f"{label}: the adapter's own shape did not parse"
        assert entry["relation"] == expected["relation"], (
            f"{label}: frontend reported {entry['relation']!r}, "
            f"Step 7 reported {expected['relation']!r}"
        )
        assert entry["comparable"] == expected["comparable"], (
            f"{label}: frontend changed comparability"
        )
        assert entry["elapsedSeconds"] == expected["elapsedSeconds"], (
            f"{label}: frontend reported elapsed={entry['elapsedSeconds']!r}, "
            f"Step 7 reported {expected['elapsedSeconds']!r}"
        )
        assert observed["chronologicallyOrdered"] == expected["comparable"], (
            f"{label}: a non-comparable step must not read as an ordering"
        )
        assert observed["window"]["durationSeconds"] == expected["elapsedSeconds"], (
            f"{label}: the temporal window does not match Step 7"
        )
        assert observed["timelineSteps"][0]["relation"] == expected["relation"]


def test_non_comparable_and_unknown_never_gain_an_elapsed_time(step7_parity):
    """Step 7's refusals must survive as refusals in the frontend."""
    refused = {
        case["label"]
        for case, observed in step7_parity
        if not case["expected"]["comparable"]
    }
    assert refused, "the parity set must include at least one refused comparison"

    for case, observed in step7_parity:
        if case["expected"]["comparable"]:
            continue
        entry = observed["entries"][0]
        assert entry["elapsedSeconds"] is None, (
            f"{case['label']}: an elapsed time was invented for a refused pair"
        )
        assert entry["standing"] == case["expected"]["relation"], (
            f"{case['label']}: the frontend replaced Step 7's wording"
        )
        assert observed["window"]["durationSeconds"] is None, (
            f"{case['label']}: the window was reconstructed from a refused pair"
        )


def test_a_single_entry_reads_the_same_as_the_whole_flow(step7_parity):
    """Reading one transition at a time must not give a different answer."""
    for case, observed in step7_parity:
        assert observed["singleEntryMatches"] is True, (
            f"{case['label']}: parsing the entry alone disagreed with the flow"
        )


def test_the_frontend_uses_only_step7_vocabulary(step7_parity):
    """The frontend must not widen or rename Step 7's vocabularies."""
    from backend.app.investigation.models import PathTemporalStatus, TemporalOrder

    for _, observed in step7_parity:
        assert observed["orderVocabulary"] == [o.value for o in TemporalOrder]
        assert observed["statusVocabulary"] == [s.value for s in PathTemporalStatus]


# --- Data integrity ---------------------------------------------------------


def test_path_data_is_unchanged_by_interaction(runner_output):
    detail = _check(runner_output, "path data is unchanged by selection, comparison and rendering")[
        "detail"
    ]
    assert detail["recordsUnchanged"] is True
    assert detail["pathsLengthUnchanged"] == 3


def test_comparison_rendering_carries_no_score(runner_output):
    detail = _check(runner_output, "comparison rendering omits any score and keeps selection order")[
        "detail"
    ]
    assert detail["renderedSides"] == 2
    assert detail["renderedOrder"] == ["path-gamma", "path-alpha"]
    assert detail["noScoreInHtml"] is True


# --- Implementation present -------------------------------------------------


def test_paths_module_exports_the_required_behaviour():
    source = PATHS_MODULE.read_text(encoding="utf-8")
    for name in REQUIRED_PATH_FEATURES:
        assert f"export function {name}" in source, f"missing export: {name}"


def test_temporal_module_exports_the_step7_semantics():
    source = TEMPORAL_MODULE.read_text(encoding="utf-8")
    for name in REQUIRED_TEMPORAL_FEATURES:
        assert f"export function {name}" in source, f"missing export: {name}"


def test_every_required_behaviour_has_a_test(runner_output):
    """The checklist must be unique and must match what the runner really ran.

    Comparing the list against itself would always pass, so this asserts the
    names are distinct and that run_paths.mjs reported exactly these scenarios.
    """
    assert len(CHECK_NAMES) == len(set(CHECK_NAMES)), "behaviour list must be unique"
    assert len(CHECK_NAMES) >= 8, "Phase 3 requires at least 8 behavioural tests"
    assert (
        len(GRAPH_CHECK_NAMES) == len(set(GRAPH_CHECK_NAMES))
    ), "graph behaviour list must be unique"

    reported = {result["name"] for result in runner_output["results"]}
    assert reported == set(CHECK_NAMES), (
        "CHECK_NAMES is out of step with tests/js/run_paths.mjs; "
        f"only in checklist: {sorted(set(CHECK_NAMES) - reported)}; "
        f"only in runner: {sorted(reported - set(CHECK_NAMES))}"
    )

    graph_reported = {result["name"] for result in runner_output["graph"]["results"]}
    assert set(GRAPH_CHECK_NAMES) <= graph_reported, (
        "these behaviours are not exercised by tests/js/run_graph_sync.mjs: "
        f"{sorted(set(GRAPH_CHECK_NAMES) - graph_reported)}"
    )


def test_phase3_section_exists_in_the_page():
    html = PAGE.read_text(encoding="utf-8")
    assert 'id="paths-content"' in html
    assert "Investigation Paths" in html


def test_earlier_phases_still_intact():
    html = PAGE.read_text(encoding="utf-8")
    assert 'id="global-search-input"' in html
    assert 'id="investigator-nav"' in html
    assert 'id="graph-filters"' in html
    assert 'id="evidence-chain-content"' in html