"""Step 10 evaluation tests.

These tests exercise the evaluator itself, not a restatement of its output.
Where a test reads the evaluation it does so to assert a property of the
evaluator (that it is deterministic, that it refuses to invent metrics, that it
declares what it normalised) rather than to assert that a stored number is the
number already stored.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import canonical, corpus, harness, run_evaluation  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def outcome():
    """One evaluation run, shared by every test in this module.

    A run takes over a second because it really re-runs the whole pipeline,
    which is the point. Repeating it per test would spend the suite's time
    proving the same thing sixty times.
    """
    return harness.run_all()


@pytest.fixture(scope="module")
def report(outcome):
    return run_evaluation.build_report(outcome, include_expected=True)


@pytest.fixture(scope="module")
def expected_payload():
    path = run_evaluation.EXPECTED_DIR / "expected_output.json"
    assert path.exists(), (
        "no stored expectation; create it deliberately with "
        "'python -m evaluation.run_evaluation --write-expected'"
    )
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


# --------------------------------------------------------------------------- #
# The evaluator runs at all
# --------------------------------------------------------------------------- #


class TestEvaluationRuns:
    def test_every_dimension_is_reported(self, outcome):
        assert set(outcome.dimensions) == set(harness.DIMENSIONS)

    def test_each_dimension_has_content(self, outcome):
        """No dimension may be empty.

        A dimension that returned nothing would still pass an expected-output
        comparison against a fixture recording nothing. This is the check that
        stops that from counting as a pass.
        """
        empty = [
            name
            for name, payload in outcome.dimensions.items()
            if not payload
        ]
        assert empty == []

    def test_every_dimension_carries_a_digest(self, outcome):
        assert len(outcome.dimensions_digest) == 64
        int(outcome.dimensions_digest, 16)

    def test_the_command_line_entry_point_exits_zero(self):
        completed = subprocess.run(
            [sys.executable, "-m", "evaluation.run_evaluation", "--quiet"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=900,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]

    def test_the_runner_writes_a_report(self, report, tmp_path):
        path = run_evaluation.write_report(report, tmp_path / "r.json")
        assert path.exists()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["report_version"] == "STEP10_EVALUATION/1"
        assert written["result"]["passed"] is True


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def test_two_runs_in_one_process_agree(self):
        first = harness.run_all()
        second = harness.run_all()
        assert first.dimensions_digest == second.dimensions_digest
        assert canonical.canonical_bytes(first.dimensions) == (
            canonical.canonical_bytes(second.dimensions)
        )

    def test_a_fresh_interpreter_agrees(self):
        """The check an in-process repeat cannot make.

        A module-level cache, a class attribute or a ``functools`` memo would
        survive two in-process runs and hide exactly the drift this is meant to
        find. A subprocess shares nothing but the code on disk.
        """
        result = run_evaluation.cross_process_check()
        assert result["performed"] is True, result.get("reason")
        assert result["identical"] is True, result

    def test_the_canonical_form_is_stable_across_processes(self, tmp_path):
        script = (
            "import sys; sys.path.insert(0, %r); "
            "from evaluation import harness; "
            "from evaluation.canonical import canonical_bytes; "
            "sys.stdout.buffer.write(canonical_bytes(harness.run_all().dimensions))"
            % str(REPO_ROOT)
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=900,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        mine = canonical.canonical_bytes(harness.run_all().dimensions)
        assert completed.stdout == mine

    def test_reversed_input_order_changes_nothing(self, outcome):
        count = outcome.dimensions["ingestion_normalization"]["summary"][
            "total_observations"
        ]
        reversed_run = harness.run_all(list(reversed(range(count))))
        for name in ("evidence_correlation", "evidence_graph", "path_determinism",
                     "fusion_priority"):
            assert (
                canonical.digest(reversed_run.dimensions[name])
                == canonical.digest(outcome.dimensions[name])
            ), "%s depended on input order" % name

    def test_rotated_input_order_changes_nothing(self, outcome):
        count = outcome.dimensions["ingestion_normalization"]["summary"][
            "total_observations"
        ]
        rotated = harness.run_all([(i + 5) % count for i in range(count)])
        for name in ("evidence_correlation", "evidence_graph", "path_determinism",
                     "fusion_priority"):
            assert (
                canonical.digest(rotated.dimensions[name])
                == canonical.digest(outcome.dimensions[name])
            ), "%s depended on input order" % name

    def test_the_runner_reports_order_independence(self, report):
        checks = report["determinism"]["shuffle_checks"]
        assert checks["all_orders_independent"] is True
        assert set(checks["orders"]) == {"reversed", "rotated_by_five"}
        for entry in checks["orders"].values():
            assert entry["mismatched_dimensions"] == []

    def test_run_scoped_values_are_declared_not_hidden(self, outcome):
        """Anything normalised away has to be named.

        Two fields hold an ingestion-run identifier. If they were quietly
        dropped the determinism result would be meaningless, so the evaluator
        substitutes a visible placeholder *and* lists where it did so.
        """
        assert outcome.run_scoped_paths, (
            "no run-scoped substitutions were reported; if the corpus became "
            "fully content-derived that is a real change and this assertion "
            "should be revisited, but an empty list must not be assumed benign"
        )
        for path in outcome.run_scoped_paths:
            assert path, "a substitution was reported without a path"

    def test_run_scoped_substitutions_reach_the_report(self, report):
        declared = report["determinism"]["run_scoped_substitutions"]
        assert declared, "the report omitted the list of normalised fields"
        note = report["determinism"]["run_scoped_substitution_note"]
        assert "rather than deleted" in note

    def test_metadata_is_reported_but_never_compared(self, report):
        """A timestamp must not be able to fail a comparison.

        ``generated_at`` exists only in the report, never inside ``dimensions``,
        so two runs a day apart cannot differ because of the clock.
        """
        assert "generated_at" in report
        assert "generated_at" not in report["dimensions"]
        assert "generated_at" not in report["dimensions_digest"]
        environment = report["metadata"]["environment"]
        assert environment["python_version"]
        assert "note" in report["metadata"]

    def test_canonical_encoding_is_stable(self):
        payload = {
            "b": 1,
            "a": {"z": 2, "y": 3},
            "decimal": __import__("decimal").Decimal("1.500"),
            "when": __import__("datetime").datetime(
                2024, 6, 1, tzinfo=__import__("datetime").timezone.utc
            ),
            "unordered": {"b", "a"},
        }
        first = canonical.canonical_bytes(payload)
        second = canonical.canonical_bytes(
            {
                "unordered": {"a", "b"},
                "when": payload["when"],
                "decimal": payload["decimal"],
                "a": {"y": 3, "z": 2},
                "b": 1,
            }
        )
        assert first == second
        text = first.decode("utf-8")
        assert '"1.500"' in text, "a Decimal must render as exact text"
        assert text.index('"a"') < text.index('"b"'), "keys must be sorted"

    def test_a_difference_is_reported_with_its_path(self):
        matched, differences = harness.diff_canonical(
            {"a": {"b": 1}}, {"a": {"b": 2}}, "$"
        )
        assert matched is False
        assert any("$.a.b" in difference for difference in differences)


# --------------------------------------------------------------------------- #
# Nothing is invented
# --------------------------------------------------------------------------- #


class TestNoFabricatedMetrics:
    FORBIDDEN = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "detection_rate",
        "false_positive_rate",
        "true_positive",
        "false_negative",
    )

    def test_the_evaluator_names_the_metrics_it_refuses(self, report):
        assert set(report["metadata"]["metrics_not_computed"]) == set(
            harness.NOT_COMPUTED
        )
        assert report["metadata"]["metrics_not_computed_reason"]

    def test_no_evaluation_payload_carries_a_performance_metric(self, outcome):
        """No key anywhere in the payload may be a performance figure.

        Searched by key name across the whole canonical payload rather than
        checked at the call sites, so a metric introduced into a new dimension
        would still be caught.
        """
        found = []

        def walk(node, prefix):
            if isinstance(node, dict):
                for key, item in node.items():
                    lowered = str(key).lower()
                    if any(word in lowered for word in self.FORBIDDEN):
                        found.append("%s.%s" % (prefix, key))
                    walk(item, "%s.%s" % (prefix, key))
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    walk(item, "%s[%d]" % (prefix, index))

        walk(outcome.dimensions, "dimensions")
        assert found == [], "performance metrics present: %s" % found

    def test_the_corpus_carries_no_ground_truth_labels(self):
        """No file in the corpus may contain a label column.

        If the corpus were labelled, every metric above would become computable
        and their absence would be a choice rather than a necessity. The check
        keeps the claim honest.
        """
        label_words = re.compile(
            r"\b(label|ground[_-]?truth|is_malicious|malicious|"
            r"ground_truth_class|target_class)\b",
            re.IGNORECASE,
        )
        offenders = []
        for name in corpus.SCENARIO_FILES:
            text = (corpus.SCENARIO_DIR / name).read_text(encoding="utf-8")
            if label_words.search(text):
                offenders.append(name)
        assert offenders == []

    def test_the_fused_score_is_labelled_as_not_a_probability(self, outcome):
        interpretation = outcome.dimensions["fusion_priority"]["interpretation"]
        assert "not a probability" in interpretation
        assert "not an accuracy" in interpretation

    def test_no_confidence_claims_certainty_without_an_exact_match(self, outcome):
        result = outcome.invariants["inference_does_not_claim_certainty"]
        assert result["holds"] is True, result
        assert result["certainty_without_exact_match_basis"] == []
        assert result["inference_claiming_exact_match"] == []

    def test_confidence_is_reserved_for_exact_matches_only(self, outcome):
        """Certainty, where it exists, must be earned by an exact match.

        The rule-derived edges all sit below 1.0 because they are judgements,
        not facts. Asserting only the first half would pass an evaluator that
        reported every edge as certain, so the exact-match basis is checked to
        reach 1.0 as well: certainty is not merely withheld, it is reserved.
        """
        graph_edges = outcome.dimensions["evidence_graph"]["edges"]
        by_basis: dict = {}
        for edge in graph_edges:
            by_basis.setdefault(edge["confidence_basis"], set()).add(
                edge["confidence"]
            )
        assert by_basis
        for basis, values in by_basis.items():
            numeric = [float(value) for value in values]
            if basis != "EXACT_MATCH":
                assert max(numeric) < 1.0, (basis, values)
            else:
                assert min(numeric) >= 1.0, (basis, values)


# --------------------------------------------------------------------------- #
# The corpus is real and is not mutated
# --------------------------------------------------------------------------- #


class TestCorpus:
    def test_every_declared_input_exists(self):
        for name in corpus.SCENARIO_FILES:
            path = corpus.SCENARIO_DIR / name
            assert path.exists(), "missing corpus input %s" % name
            assert path.stat().st_size > 0

    def test_all_three_supported_formats_are_covered(self):
        formats = {
            path.suffix.lstrip(".")
            for path in (corpus.SCENARIO_DIR / name for name in corpus.SCENARIO_FILES)
        }
        assert formats == {"csv", "json", "xml"}

    def test_ingestion_does_not_modify_its_input(self):
        """Ingestion is read-only over the caller's file."""
        before = {
            name: canonical.sha256_file(corpus.SCENARIO_DIR / name)
            for name in corpus.SCENARIO_FILES
        }
        corpus.load_observations()
        after = {
            name: canonical.sha256_file(corpus.SCENARIO_DIR / name)
            for name in corpus.SCENARIO_FILES
        }
        assert before == after
        summary = corpus.load_observations()[1]
        assert all(entry["source_bytes_unchanged"] for entry in summary["files"])

    def test_malformed_records_are_reported_not_dropped(self, outcome):
        summary = outcome.dimensions["ingestion_normalization"]["summary"]
        assert summary["observations_with_warnings"] > 0
        assert summary["observations_without_timestamp"] > 0
        assert summary["total_observations"] == summary["distinct_observation_ids"]
        codes = {
            code
            for entry in outcome.dimensions["ingestion_normalization"]["files"]
            for code in entry["warning_codes"]
        }
        assert "NAIVE_TIMESTAMP" in codes

    def test_observation_ids_are_content_derived_and_stable(self):
        """Ids must be a function of record content, not of run order."""
        first = [o.observation_id for o in corpus.load_observations()[0]]
        second = [o.observation_id for o in corpus.load_observations()[0]]
        assert first == second
        assert len(set(first)) == len(first)

    def test_selection_does_not_depend_on_the_list_order(self):
        observations = corpus.load_observations()[0]
        forward = corpus.stable_order(observations)
        backward = corpus.stable_order(list(reversed(observations)))
        assert [o.observation_id for o in forward] == [
            o.observation_id for o in backward
        ]


# --------------------------------------------------------------------------- #
# Case coverage
# --------------------------------------------------------------------------- #


class TestCaseCoverage:
    def test_all_sixteen_cases_are_declared(self):
        assert len(harness.CASE_COVERAGE) == 16
        names = [case["case"] for case in harness.CASE_COVERAGE]
        assert len(set(names)) == 16

    def test_every_case_names_a_real_dimension(self, outcome):
        for case in harness.CASE_COVERAGE:
            assert case["exercised_by"] in outcome.dimensions, case["case"]
            assert case["evidence"], case["case"]
            assert case["description"], case["case"]

    def test_the_required_cases_are_present(self):
        required = {
            "same_txid",
            "same_address",
            "same_role_ip",
            "temporal_adjacency",
            "amount_match",
            "provenance_relationship",
            "geo_coincidence",
            "mixed_evidence",
            "multiple_inference_levels",
            "comparable_temporal_data",
            "not_comparable_temporal_data",
            "missing_temporal_data",
            "anomaly_context",
            "clustering_context",
            "multi_hop_path",
            "evidence_supported_investigative_priority",
        }
        assert {case["case"] for case in harness.CASE_COVERAGE} == required

    def test_the_report_lists_case_coverage(self, report):
        assert len(report["case_coverage"]) == 16

    def test_declared_evidence_files_exist(self):
        for case in harness.CASE_COVERAGE:
            match = re.match(r"^evaluation/[A-Za-z0-9_/\.\-]+", case["evidence"])
            if match:
                assert (REPO_ROOT / match.group(0)).exists(), case["evidence"]


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


class TestInvariants:
    def test_every_invariant_holds(self, outcome):
        failing = {
            name: payload
            for name, payload in outcome.invariants.items()
            if not payload["holds"]
        }
        assert failing == {}

    def test_every_invariant_explains_itself(self, outcome):
        for name, payload in outcome.invariants.items():
            assert payload.get("why"), "invariant %s has no reason" % name

    def test_the_checklist_is_not_a_restatement_of_the_output(self, outcome):
        """Each invariant must be falsifiable by the data, not by the payload.

        An invariant whose condition merely repeats something the dimension
        already reports would hold no matter what the pipeline did. Each one
        below has to carry at least one condition the pipeline did not itself
        assert.
        """
        tautological = []
        for name, payload in outcome.invariants.items():
            evidence = set(payload) - {"holds", "why"}
            if not evidence:
                tautological.append(name)
        assert tautological == []

    def test_the_report_counts_the_invariants(self, report):
        assert report["invariants"]["count"] == len(report["invariants"]["results"])
        assert report["invariants"]["failed"] == 0
        assert report["invariants"]["all_hold"] is True

    def test_fusion_carries_all_four_inference_levels(self, outcome):
        levels = outcome.dimensions["fusion_priority"]["inference_levels"]
        assert sorted(levels) == [
            "CORRELATED",
            "INFERRED",
            "OBSERVED",
            "REQUIRES_VERIFICATION",
        ]

    def test_fusion_reports_missing_signals_as_well_as_present(self, outcome):
        fusion = outcome.dimensions["fusion_priority"]
        assert fusion["missing_signals"], "no unavailable signal was reported"
        assert fusion["signal_count"] == (
            fusion["present_signal_count"] + len(fusion["missing_signals"])
        )
        assert all(item["explanation"] for item in fusion["missing_signals"])

    def test_temporal_comparison_refuses_to_invent_elapsed_time(self, outcome):
        pairs = {
            case["case"]: case
            for case in outcome.dimensions["path_determinism"][
                "temporal_comparison"
            ]["pairwise"]
        }
        assert pairs["comparable_temporal_data"]["comparable"] is True
        assert pairs["comparable_temporal_data"]["elapsed_seconds"] is not None
        for case in ("not_comparable_temporal_data", "missing_temporal_data"):
            assert pairs[case]["comparable"] is False, case
            assert pairs[case]["elapsed_seconds"] is None, case

    def test_the_graph_keeps_every_relationship(self, outcome):
        graph = outcome.dimensions["evidence_graph"]
        assert graph["edge_count"] == graph["relationship_count"]
        assert graph["every_relationship_became_an_edge"] is True
        assert graph["isolated_node_count"] == 0

    def test_a_multi_hop_path_was_actually_found(self, outcome):
        paths = outcome.dimensions["path_determinism"]
        assert paths["max_hop_count"] > 1
        assert paths["searches"]["shortest"]["search"]["status"] == "FOUND"

    def test_the_ml_stage_recovered_the_declared_grouping(self, outcome):
        """Clustering must separate the four high-burst members on its own.

        The corpus is written so those four differ from the rest in amount
        alone. Nothing in the pipeline is told that; recovering the split is
        the clustering stage working, not merely running.
        """
        ml = outcome.dimensions["ml_determinism"]
        assert ml["clustering"]["cluster_count"] == 2
        memberships = [
            set(cluster["member_population_indices"])
            for cluster in ml["clustering"]["clusters"]
        ]
        assert {8, 9, 10, 11} in memberships, memberships
        assert {0, 1, 2, 3, 4, 5, 6, 7} in memberships, memberships

    def test_the_cluster_that_separated_does_so_on_amount_features(self, outcome):
        """The split above must be explained, not just observed."""
        ml = outcome.dimensions["ml_determinism"]
        separating = [
            cluster
            for cluster in ml["clustering"]["clusters"]
            if set(cluster["member_population_indices"]) == {8, 9, 10, 11}
        ]
        assert len(separating) == 1
        above = separating[0]["features_above_reference"]
        assert above, "the outlier cluster names no distinguishing feature"
        assert any("amount" in name for name in above), above

    def test_behavioural_profiles_claim_no_attribution(self, outcome):
        behaviour = outcome.dimensions["behavioral_features"]
        assert behaviour["any_profile_claims_attribution"] is False
        for profile in behaviour["profiles"]:
            assert profile["describes_attribution"] is False


# --------------------------------------------------------------------------- #
# The stored expectation
# --------------------------------------------------------------------------- #


class TestStoredExpectation:
    def test_the_run_matches_the_stored_expectation(self, outcome):
        matched, differences = harness.compare_to_expected(
            outcome, run_evaluation.EXPECTED_DIR / "expected_output.json"
        )
        assert matched, "\n".join(differences[:20])

    def test_the_expectation_is_a_compact_projection(self, expected_payload):
        """A fixture nobody can read is a fixture nobody maintains.

        The full payload carries every feature value of every profile; the
        stored expectation must not. A size ceiling is the cheapest honest way
        to keep that true, and it fails loudly if someone reverts to a dump.
        """
        size = (run_evaluation.EXPECTED_DIR / "expected_output.json").stat().st_size
        assert size < 100_000, "stored expectation is %d bytes" % size
        assert set(expected_payload) == set(harness.DIMENSIONS)

    def test_the_expectation_omits_per_item_measurements(self, expected_payload):
        """No feature values, no raw signal numbers, no edge confidences."""
        assert "features" not in expected_payload["behavioral_features"]
        assert "signals" not in expected_payload["ml_determinism"]["anomaly"]
        for edge in expected_payload["evidence_graph"]["edge_labels"]:
            assert isinstance(edge, str)

    def test_the_expectation_keeps_the_contract(self, expected_payload):
        """Counts, levels, types, order and statuses must all be there."""
        correlation = expected_payload["evidence_correlation"]
        assert correlation["evidence_types_present"]
        assert correlation["evidence_type_counts"]
        assert correlation["rules_that_fired"]
        dashboard = expected_payload["dashboard_payload"]
        assert dashboard["signal_ids_in_order"] == sorted(
            dashboard["signal_ids_in_order"]
        )
        assert dashboard["path_ids_in_order"] == sorted(
            dashboard["path_ids_in_order"]
        )
        assert expected_payload["fusion_priority"]["explanation"]

    def test_the_expectation_is_plain_readable_json(self):
        path = run_evaluation.EXPECTED_DIR / "expected_output.json"
        text = path.read_text(encoding="utf-8")
        assert "\r\n" not in text, "the expectation must use LF line endings"
        assert text.endswith("\n")
        assert json.loads(text)

    def test_a_changed_expectation_is_reported_not_silently_accepted(self, tmp_path):
        stored = tmp_path / "expected_output.json"
        stored.write_text(json.dumps({"dimensions": {}}), encoding="utf-8")
        matched, differences = harness.diff_canonical(
            {"dimensions": {}}, {"dimensions": {"a": 1}}, "$"
        )
        assert matched is False
        assert differences


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #


class TestOffline:
    def test_the_evaluation_opens_no_socket(self, monkeypatch):
        """Run the whole evaluation with sockets disabled.

        A functional evaluator that reached the network could return different
        numbers on a connected machine than on an air-gapped one, and the
        determinism result would be worthless. Making any socket creation fail
        turns that from a claim into a test.
        """

        def refuse(*args, **kwargs):
            raise AssertionError(
                "the Step 10 evaluation attempted a network connection; it "
                "must run offline"
            )

        monkeypatch.setattr(socket, "socket", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)
        result = harness.run_all()
        assert result.dimensions_digest

    def test_no_evaluation_module_imports_a_network_client(self):
        """No requests/httpx/urllib3 in the evaluation package."""
        banned = re.compile(
            r"^\s*(?:import|from)\s+(requests|httpx|urllib3|aiohttp|httplib2)\b",
            re.MULTILINE,
        )
        for path in (REPO_ROOT / "evaluation").rglob("*.py"):
            match = banned.search(path.read_text(encoding="utf-8"))
            assert match is None, "%s imports a network client" % path.name

    def test_the_evaluation_reads_only_the_corpus(self, outcome):
        """The only data files the evaluator touches are its own inputs."""
        summary = outcome.dimensions["ingestion_normalization"]["summary"]
        assert summary["files_ingested"] == len(corpus.SCENARIO_FILES)
        for entry in outcome.dimensions["ingestion_normalization"]["files"]:
            assert (corpus.SCENARIO_DIR / entry["file"]).exists()


# --------------------------------------------------------------------------- #
# Command line behaviour
# --------------------------------------------------------------------------- #


class TestCommandLine:
    def test_the_summary_names_what_was_not_computed(self, report, capsys):
        text = run_evaluation.summarise(report)
        assert "metrics deliberately not computed" in text
        for metric in harness.NOT_COMPUTED:
            assert metric in text
        assert "RESULT: PASS" in text

    def test_a_missing_expectation_is_a_failure_not_a_skip(self, report, tmp_path):
        """No stored expectation must not quietly pass."""
        outcome = harness.run_all()
        empty_dir = tmp_path / "expected"
        empty_dir.mkdir()
        original = run_evaluation.EXPECTED_DIR
        try:
            run_evaluation.EXPECTED_DIR = empty_dir
            comparison = run_evaluation.compare_expected(outcome)
        finally:
            run_evaluation.EXPECTED_DIR = original
        assert comparison["exists"] is False
        assert comparison["matched"] is False

    def test_write_expected_is_explicit_and_reproducible(self, tmp_path):
        outcome = harness.run_all()
        original = run_evaluation.EXPECTED_DIR
        try:
            run_evaluation.EXPECTED_DIR = tmp_path
            first = run_evaluation.write_expected(outcome)
            payload = first.read_bytes()
            run_evaluation.write_expected(harness.run_all())
        finally:
            run_evaluation.EXPECTED_DIR = original
        assert first.read_bytes() == payload

    def test_the_report_is_written_with_lf_endings(self, report, tmp_path):
        path = run_evaluation.write_report(report, tmp_path / "nested" / "r.json")
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n")