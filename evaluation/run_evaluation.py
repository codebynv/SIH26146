"""Run the Step 10 evaluation and write a reproducibility report.

    python -m evaluation.run_evaluation

The run repeats the whole evaluation twice in one process and once more in a
fresh interpreter, then compares all three canonical encodings byte for byte.
Cross-process comparison is the part that matters: an in-process repeat would
pass even if a module-level cache carried state between runs.

Exit status is ``0`` only when every determinism check, every invariant and the
comparison against ``evaluation/expected/`` all hold.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import corpus, harness
from .canonical import canonical_bytes, digest, sha256_file

__all__ = [
    "EXPECTED_DIR",
    "REPORT_DIR",
    "build_report",
    "cross_process_check",
    "main",
]

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
EXPECTED_DIR = EVAL_DIR / "expected"
REPORT_DIR = EVAL_DIR / "reports"


# --------------------------------------------------------------------------- #
# Environment metadata
# --------------------------------------------------------------------------- #


def _installed_version(name: str) -> Optional[str]:
    """The installed version of *name*, or None when it is not installed.

    Read from the working interpreter rather than hard-coded, because the report
    has to describe the environment it was produced in.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8 is not supported
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_metadata() -> Dict[str, Any]:
    """Everything about the environment that a reader needs to reproduce.

    Deliberately excluded from the comparison. A timestamp or a platform string
    differs between two runs on purpose, so folding it into the digest would
    mean either comparing noise or hiding real drift. It is reported here,
    once, and never compared.
    """
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "dependencies": {
            name: _installed_version(name)
            for name in (
                "fastapi",
                "uvicorn",
                "pydantic",
                "starlette",
                "python-multipart",
                "httpx",
                "pytest",
            )
        },
    }


def corpus_metadata() -> Dict[str, Any]:
    """A hash per corpus input file, so a changed corpus is a visible change."""
    return {
        "scenario_dir": "evaluation/scenarios",
        "files": [
            {
                "file": name,
                "sha256": sha256_file(corpus.SCENARIO_DIR / name),
                "bytes": (corpus.SCENARIO_DIR / name).stat().st_size,
            }
            for name in corpus.SCENARIO_FILES
        ],
        "correlation_window_seconds": corpus.CORRELATION_WINDOW.total_seconds(),
        "subject_position": corpus.SUBJECT_POSITION,
        "population_size": 12,
        "epoch": corpus.UTC_EPOCH.isoformat(),
    }


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def repeat_digest_check(repeats: int = 2) -> Dict[str, Any]:
    """Run the evaluation *repeats* times in this process and compare."""
    digests = [harness.run_all().dimensions_digest for _ in range(repeats)]
    return {
        "repeats": repeats,
        "digests": digests,
        "identical": len(set(digests)) == 1,
        "digest": digests[0] if digests else None,
    }


def cross_process_check() -> Dict[str, Any]:
    """Run the evaluation in a fresh interpreter and compare its digest.

    This is the check that a module-level cache cannot pass: a fresh process
    shares nothing with this one except the code on disk.
    """
    script = (
        "import sys; sys.path.insert(0, %r); "
        "from evaluation import harness; "
        "print(harness.run_all().dimensions_digest)" % str(REPO_ROOT)
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "performed": False,
            "reason": "could not start a subprocess: %s" % error,
        }

    if completed.returncode != 0:
        return {
            "performed": False,
            "reason": "subprocess exited %d: %s"
            % (completed.returncode, completed.stderr.strip()[:400]),
        }

    foreign = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    own = harness.run_all().dimensions_digest
    return {
        "performed": True,
        "digest_this_process": own,
        "digest_fresh_process": foreign,
        "identical": bool(foreign) and foreign == own,
    }


# --------------------------------------------------------------------------- #
# Expected output
# --------------------------------------------------------------------------- #


def compare_expected(outcome: harness.EvaluationOutcome) -> Dict[str, Any]:
    """Compare against the stored expectation, reporting every difference."""
    path = EXPECTED_DIR / "expected_output.json"
    matched, differences = harness.compare_to_expected(outcome, path)
    return {
        "expected_file": "evaluation/expected/expected_output.json",
        "exists": path.exists(),
        "matched": matched,
        "difference_count": len(differences),
        "differences": differences[:100],
        "differences_truncated": len(differences) > 100,
    }


def write_expected(outcome: harness.EvaluationOutcome) -> Path:
    """Store the current run's compact view as the expectation.

    The compact projection is stored, not the full payload: an expectation has
    to be readable end to end by a person checking that it says the right
    thing. An explicit action rather than a side effect of running, so the
    expectation can only change by someone deciding to change it.
    """
    path = EXPECTED_DIR / "expected_output.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = harness.compact_view(outcome.dimensions)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
    return path


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def build_report(
    outcome: harness.EvaluationOutcome,
    *,
    include_expected: bool = True,
) -> Dict[str, Any]:
    """Assemble the full report, separating compared output from metadata."""
    in_process = repeat_digest_check()
    cross = cross_process_check()

    invariant_rows = [
        {"invariant": name, **payload}
        for name, payload in sorted(outcome.invariants.items())
    ]
    failing = [row["invariant"] for row in invariant_rows if not row["holds"]]

    report: Dict[str, Any] = {
        "report_version": "STEP10_EVALUATION/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "note": (
                "metadata is reported once and never compared; a timestamp or "
                "platform string differs between runs by design"
            ),
            "environment": environment_metadata(),
            "corpus": corpus_metadata(),
            "dimensions": list(harness.DIMENSIONS),
            "cases_covered": len(harness.CASE_COVERAGE),
            "metrics_not_computed": list(harness.NOT_COMPUTED),
            "metrics_not_computed_reason": (
                "the corpus is synthetic and carries no ground-truth labels, "
                "so every one of these would be invented; what is measured "
                "instead is repeatability, order independence, and whether each "
                "stage reported what it actually had"
            ),
        },
        "determinism": {
            "in_process": in_process,
            "cross_process": cross,
            "all_identical": in_process["identical"]
            and cross.get("identical", False),
            "shuffle_checks": outcome.shuffle_checks,
            "run_scoped_substitutions": list(outcome.run_scoped_paths),
            "run_scoped_substitution_note": (
                "these fields hold an identifier of the ingestion run, not of "
                "the evidence; each was substituted with an explicit "
                "placeholder rather than deleted, and is listed here"
            ),
        },
        "invariants": {
            "count": len(invariant_rows),
            "passed": len(invariant_rows) - len(failing),
            "failed": len(failing),
            "failing": failing,
            "all_hold": not failing,
            "results": invariant_rows,
        },
        "case_coverage": [dict(case) for case in outcome.case_coverage],
        "dimensions_digest": outcome.dimensions_digest,
        "dimensions": outcome.dimensions,
    }

    if include_expected:
        report["expected_output_comparison"] = compare_expected(outcome)

    report["result"] = _verdict(report)
    return report


def _verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """A single pass/fail, with the reasons spelled out.

    Reported as data rather than printed as prose so the same file can be read
    by a person and by a test without either having to parse the other.
    """
    determinism_ok = report["determinism"]["all_identical"]
    shuffle_ok = report["determinism"]["shuffle_checks"]["all_orders_independent"]
    invariants_ok = report["invariants"]["all_hold"]
    expected = report.get("expected_output_comparison")

    reasons: List[str] = []
    if not determinism_ok:
        reasons.append(
            "repeated runs did not produce byte-identical canonical output"
        )
    if not shuffle_ok:
        mismatched = sorted(
            {
                dimension
                for entry in report["determinism"]["shuffle_checks"]["orders"].values()
                for dimension in entry["mismatched_dimensions"]
            }
        )
        reasons.append(
            "output depended on the order records arrived in: %s"
            % ", ".join(mismatched)
        )
    if not invariants_ok:
        reasons.append(
            "invariant(s) failed: %s" % ", ".join(report["invariants"]["failing"])
        )
    if expected is not None and not expected["matched"]:
        if not expected["exists"]:
            reasons.append("no stored expectation to compare against")
        else:
            reasons.append(
                "output differs from the stored expectation in %d place(s)"
                % expected["difference_count"]
            )

    return {
        "passed": not reasons,
        "determinism_identical": determinism_ok,
        "order_independent": shuffle_ok,
        "invariants_hold": invariants_ok,
        "matches_stored_expectation": (
            None if expected is None else expected["matched"]
        ),
        "failure_reasons": reasons,
    }


def write_report(report: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Write the report as readable JSON."""
    target = path or (REPORT_DIR / "latest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
    return target


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #


def summarise(report: Dict[str, Any]) -> str:
    """A short human-readable summary of the report."""
    lines: List[str] = []
    ingest = report["dimensions"]["ingestion_normalization"]["summary"]
    correlation = report["dimensions"]["evidence_correlation"]
    graph = report["dimensions"]["evidence_graph"]
    paths = report["dimensions"]["path_determinism"]
    fusion = report["dimensions"]["fusion_priority"]
    ml = report["dimensions"]["ml_determinism"]

    lines.append("BitTrace Step 10 evaluation")
    lines.append("=" * 60)
    lines.append(
        "corpus        %d observations from %d files (%s)"
        % (
            ingest["total_observations"],
            ingest["files_ingested"],
            ", ".join(ingest["formats"]),
        )
    )
    lines.append(
        "correlation   %d relationships, %d evidence types"
        % (correlation["relationship_count"], len(correlation["evidence_types_present"]))
    )
    lines.append(
        "graph         %d nodes, %d edges, %d orphan nodes"
        % (graph["node_count"], graph["edge_count"], graph["isolated_node_count"])
    )
    lines.append(
        "paths         %d found, deepest %d hops"
        % (paths["total_paths_found"], paths["max_hop_count"])
    )
    lines.append(
        "ml            %d clusters, anomaly bands %s"
        % (
            ml["clustering"]["cluster_count"],
            ", ".join(
                "%s=%d" % (band, count)
                for band, count in sorted(ml["anomaly"]["band_counts"].items())
            ),
        )
    )
    lines.append(
        "priority      score %s, band %s, %d/%d signals present"
        % (
            fusion["fused_score"],
            fusion["risk_band"],
            fusion["present_signal_count"],
            fusion["signal_count"],
        )
    )
    lines.append(
        "levels        %s" % ", ".join(fusion["inference_levels"])
    )
    lines.append("")
    determinism = report["determinism"]
    lines.append(
        "determinism   in-process %s, cross-process %s, order-independent %s"
        % (
            "identical" if determinism["in_process"]["identical"] else "DIFFERS",
            "identical"
            if determinism["cross_process"].get("identical")
            else "DIFFERS",
            "yes" if determinism["shuffle_checks"]["all_orders_independent"] else "NO",
        )
    )
    lines.append(
        "invariants    %d/%d hold"
        % (report["invariants"]["passed"], report["invariants"]["count"])
    )
    expected = report.get("expected_output_comparison")
    if expected is not None:
        lines.append(
            "expected      %s"
            % (
                "matches %s" % expected["expected_file"]
                if expected["matched"]
                else "DIFFERS in %d place(s)" % expected["difference_count"]
            )
        )
    lines.append(
        "digest        %s" % report["dimensions_digest"]
    )
    lines.append("")
    lines.append("metrics deliberately not computed:")
    lines.append(
        "  %s" % ", ".join(report["metadata"]["metrics_not_computed"])
    )
    lines.append("  %s" % report["metadata"]["metrics_not_computed_reason"])
    lines.append("")
    result = report["result"]
    lines.append(
        "RESULT: %s" % ("PASS" if result["passed"] else "FAIL")
    )
    for reason in result["failure_reasons"]:
        lines.append("  - %s" % reason)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_evaluation",
        description=(
            "Run the Step 10 evaluation twice in-process and once in a fresh "
            "interpreter, compare the canonical output, and write a report."
        ),
    )
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help=(
            "store the current run as evaluation/expected/expected_output.json; "
            "an explicit action, so the expectation cannot drift unnoticed"
        ),
    )
    parser.add_argument(
        "--no-expected",
        action="store_true",
        help="skip the comparison against the stored expectation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="where to write the report (default evaluation/reports/latest.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write the report without printing the summary",
    )
    arguments = parser.parse_args(argv)

    outcome = harness.run_all()

    if arguments.write_expected:
        path = write_expected(outcome)
        if not arguments.quiet:
            print("stored expectation at %s" % path.relative_to(REPO_ROOT).as_posix())

    report = build_report(
        outcome, include_expected=not arguments.no_expected
    )
    path = write_report(report, arguments.report)
    if not arguments.quiet:
        print(summarise(report))
        print()
        print("report written to %s" % path.relative_to(REPO_ROOT).as_posix())

    return 0 if report["result"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())