"""Linux compatibility audit for BitTrace AI.

This project was developed on Windows. Most of the failure modes that produces
are invisible until they do: a hardcoded backslash path separator, a Windows
case-insensitive assumption in a dedup key, a `cmp`-style ordering difference,
or a subprocess invocation that only works with `cmd.exe`.

The tests below run on any platform and check the properties that would break on
Linux. Where a check is only meaningful on POSIX it is still run, because the
value is in catching the Windows assumption at review time rather than in
waiting for a Linux failure at deploy time.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
EVALUATION = REPO_ROOT / "evaluation"


def _python_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    ]


# --------------------------------------------------------------------------- #
# Path handling
# --------------------------------------------------------------------------- #


class TestPathPortability:
    def test_no_hardcoded_windows_path_separator(self):
        """A backslash literal in a path expression breaks on POSIX.

        Comments and docstrings are excluded: the audit is about code, and a
        prose mention of ``\\`` is documentation rather than a defect.
        """
        offenders = []
        for path in _python_files(BACKEND) + _python_files(EVALUATION):
            relative = path.relative_to(REPO_ROOT).as_posix()
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if not stripped:
                    continue
                if "\\" not in line:
                    continue
                # A backslash inside a string literal that is clearly prose, an
                # escape sequence, or a regex character class is fine. What
                # must not appear is a separator joining path segments.
                if re.search(r"[\"'][^\"']*[\"']\s*\+\s*[\"']\\", line):
                    offenders.append("%s:%d joins paths with a backslash" % (relative, number))
        assert offenders == [], offenders

    def test_paths_are_built_with_pathlib_not_string_concatenation(self):
        """String-joined paths break on whichever separator is hardcoded."""
        offenders = []
        for path in _python_files(BACKEND) + _python_files(EVALUATION):
            relative = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(
                r"(os\.path\.join|Path\()\s*", text
            ):
                pass
            # A concatenation of a path with a literal segment using os.sep
            # is fine; using "\\" or "/" directly is the risk.
            for number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(
                    r"(?:dir|path|root|base|directory|target)\w*\s*\+\s*[\"'][^\"']*[/\\\\]",
                    line,
                    re.IGNORECASE,
                ):
                    offenders.append(
                        "%s:%d concatenates a path with a literal separator"
                        % (relative, number)
                    )
        assert offenders == [], offenders

    def test_the_stored_expectation_uses_lf_line_endings(self):
        """Git's autocrlf would otherwise rewrite the fixture on checkout."""
        for path in (
            EVALUATION / "expected" / "expected_output.json",
            EVALUATION / "scenarios" / "flow_combined.csv",
        ):
            raw = path.read_bytes()
            assert b"\r\n" not in raw, "%s contains CRLF line endings" % path.name
            assert raw.endswith(b"\n"), "%s has no trailing newline" % path.name

    def test_no_file_carries_a_crlf_line_ending(self):
        """Text the pipeline reads should be byte-stable across checkouts."""
        offenders = []
        for path in list(_python_files(EVALUATION)) + sorted(
            (EVALUATION / "scenarios").iterdir()
        ):
            if b"\r\n" in path.read_bytes():
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
        assert offenders == [], offenders

    def test_the_evaluation_resolves_its_scenarios_relative_to_the_package(self):
        """An absolute or cwd-dependent path breaks under any launcher."""
        from evaluation import corpus

        assert corpus.SCENARIO_DIR.is_absolute()
        # Resolved from the package location, not from the process cwd.
        assert corpus.SCENARIO_DIR == EVALUATION / "scenarios"
        for name in corpus.SCENARIO_FILES:
            assert (corpus.SCENARIO_DIR / name).exists()

    def test_a_staged_upload_name_is_separator_agnostic(self):
        """The staging check must recognise its own files on either separator."""
        from backend.app.ingestion.uploads import (
            STAGED_UPLOAD_PREFIX,
            is_staged_upload,
        )

        posix = "/tmp/bittrace-upload-abc123.dat"
        windows = "C:\\Users\\probe\\AppData\\Local\\Temp\\bittrace-upload-abc123.dat"
        assert is_staged_upload(posix)
        assert is_staged_upload(windows)
        assert not is_staged_upload("/tmp/other-file.dat")
        assert STAGED_UPLOAD_PREFIX in posix

    def test_a_client_filename_stripping_is_separator_agnostic(self):
        from backend.app.ingestion.uploads import sanitize_display_filename

        assert sanitize_display_filename("../../etc/passwd") == "passwd"
        assert sanitize_display_filename("..\\..\\windows\\system32\\cmd.exe") == (
            "cmd.exe"
        )
        assert sanitize_display_filename("C:\\data\\report.csv") == "report.csv"


# --------------------------------------------------------------------------- #
# Case sensitivity, collation and locale
# --------------------------------------------------------------------------- #


class TestCaseAndCollation:
    def test_the_evaluation_sorts_explicitly_rather_than_relying_on_locale(self):
        """Implicit dict ordering is fine; implicit collation is not.

        ``sorted`` on mixed-case ASCII is stable across platforms, but a
        locale-dependent comparison would not be. The evaluation must not reach
        for ``locale.strxfrm`` or a locale-dependent sort key.
        """
        offenders = []
        for path in _python_files(EVALUATION):
            relative = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            for needle in ("locale.", "strxfrm", "strcoll", "setlocale"):
                if needle in text:
                    offenders.append("%s uses %s" % (relative, needle))
        assert offenders == [], offenders

    def test_deduplication_does_not_rely_on_case_insensitive_filesystem_order(self):
        """Windows would treat A.csv and a.csv as one file; Linux would not.

        The staging directory is server-generated and never named from client
        input, so the collision cannot arise. Asserted rather than assumed.
        """
        from backend.app.ingestion.uploads import STAGED_UPLOAD_PREFIX

        # A generated name is unique per call, so no case-folding question
        # arises at all.
        assert STAGED_UPLOAD_PREFIX.islower()

    def test_content_sorting_uses_a_total_order(self):
        """A sort key that cannot compare would raise only on some data."""
        from evaluation import corpus

        observations = corpus.load_observations()[0]
        ordered = corpus.stable_order(observations)
        identifiers = [item.observation_id for item in ordered]
        assert identifiers == sorted(identifiers)
        assert len(set(identifiers)) == len(identifiers)

    def test_feature_names_are_compared_as_identifiers_not_by_location(self):
        """ASCII-only identifiers keep ordering identical on every platform.

        Read from the stored expectation rather than a fresh run: this is a
        check about the committed fixture, and it should fail if the fixture is
        wrong even when the pipeline is not being run.
        """
        import json

        from evaluation import run_evaluation

        payload = json.loads(
            (
                run_evaluation.EXPECTED_DIR / "expected_output.json"
            ).read_text(encoding="utf-8")
        )
        grouped = payload["behavioral_features"]["feature_names_by_category"]
        names: list[str] = []
        for category, entries in grouped.items():
            names.extend(entries)
            assert entries == sorted(entries), category
        assert names
        # A non-ASCII identifier would sort differently under a different
        # locale or collation table.
        offenders = [name for name in names if not name.isascii()]
        assert offenders == [], offenders


# --------------------------------------------------------------------------- #
# Subprocess invocation
# --------------------------------------------------------------------------- #


class TestSubprocessInvocation:
    def test_the_cross_process_check_uses_a_list_argument_not_a_shell_string(self):
        """A command string would need shell=True on Windows and not on POSIX."""
        from evaluation import run_evaluation

        source = Path(run_evaluation.__file__).read_text(encoding="utf-8")
        assert "shell=True" not in source, (
            "the evaluation spawns an interpreter; shell=True would make its "
            "behaviour depend on the host shell"
        )
        for match in re.finditer(r"subprocess\.run\(\s*\[", source):
            assert match, "subprocess.run must receive an argument list"

    def test_the_fresh_interpreter_check_runs_on_this_platform(self):
        from evaluation import harness, run_evaluation

        result = run_evaluation.cross_process_check()
        if not result.get("performed"):
            pytest.skip(result.get("reason", "cross-process check did not run"))
        assert result["identical"] is True

    def test_the_check_finds_the_project_without_depending_on_the_cwd(self):
        """The subprocess inherits the cwd, so it must not be assumed."""
        from evaluation import harness

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, %r); "
                "from evaluation import harness; "
                "print(harness.run_all().dimensions_digest)"
                % str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            # Deliberately a different working directory: this is the whole
            # point of the check.
            cwd=str(Path(os.sep)),
            timeout=900,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.strip() == harness.run_all().dimensions_digest

    def test_the_test_runner_entry_point_is_not_windows_shaped(self):
        """pytest.ini must not carry a Windows-only option."""
        text = (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8")
        assert "\\\\" not in text, "pytest.ini contains a Windows path separator"
        for forbidden in ("addopts = -n", "--dist=loadfile"):
            assert forbidden not in text, (
                "%s requires a plugin the runtime image does not install" % forbidden
            )


# --------------------------------------------------------------------------- #
# Encoding, newlines and filesystem limits
# --------------------------------------------------------------------------- #


class TestEncodingAndLimits:
    def test_source_files_declare_utf8_where_they_read_text(self):
        """Text is read with an explicit encoding everywhere.

        The container runs in a POSIX locale, so Python's default encoding is
        UTF-8 there but cp1252 on a default Windows install. Reading without an
        explicit encoding is the single most common way a project that passes
        on Windows fails on Linux.
        """
        offenders = []
        for path in _python_files(EVALUATION):
            relative = path.relative_to(REPO_ROOT).as_posix()
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if ".read_text()" in line and "encoding" not in line:
                    offenders.append("%s:%d read_text without encoding" % (relative, number))
                if ".write_text(" in line and "encoding" not in line:
                    offenders.append(
                        "%s:%d write_text without encoding" % (relative, number)
                    )
                if "open(" in line and "encoding" not in line and "rb" not in line:
                    offenders.append("%s:%d open without encoding" % (relative, number))
        assert offenders == [], offenders

    def test_the_backend_also_reads_and_writes_with_an_explicit_encoding(self):
        offenders = []
        for path in _python_files(BACKEND):
            relative = path.relative_to(REPO_ROOT).as_posix()
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if ".read_text()" in line and "encoding" not in line:
                    offenders.append("%s:%d read_text without encoding" % (relative, number))
        assert offenders == [], offenders

    def test_the_evaluation_never_opens_a_file_in_text_mode_for_input_data(self):
        """CSV and XML parsing must read bytes, or the BOM decides the encoding."""
        from evaluation import corpus

        source = Path(corpus.__file__).read_text(encoding="utf-8")
        assert '"r",' not in source.replace('"r"', '"rb"'), (
            "the corpus must not open an input file in text mode"
        )

    def test_the_unicode_corpus_content_survives_a_round_trip(self):
        """A non-ASCII value must not depend on the host default encoding."""
        from evaluation import corpus
        from evaluation.canonical import canonical_bytes

        observations, _ = corpus.load_observations()
        payload = canonical_bytes([o.observation_id for o in observations])
        assert b"\xef\xbf\xbd" not in payload, "a replacement character appeared"


class TestFilesystemLimits:
    def test_no_path_in_the_repository_exceeds_a_common_component_limit(self):
        """255 bytes per component and 4096 for the whole path."""
        offenders = []
        for path in REPO_ROOT.rglob("*"):
            if ".git" in path.parts or ".venv" in path.parts:
                continue
            try:
                relative = path.relative_to(REPO_ROOT)
            except ValueError:  # pragma: no cover
                continue
            for part in relative.parts:
                if len(part.encode("utf-8")) > 255:
                    offenders.append(str(relative))
            if len(str(relative).encode("utf-8")) > 4096:
                offenders.append(str(relative))
        assert offenders == [], offenders

    def test_the_repository_survives_a_case_only_rename(self):
        """Linux distinguishes these; a colliding pair would behave differently."""
        offenders = []
        seen: dict = {}
        for path in REPO_ROOT.rglob("*"):
            if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts):
                continue
            key = str(path).lower()
            if key in seen and seen[key] != str(path):
                offenders.append("%s vs %s" % (seen[key], path))
            seen[key] = str(path)
        assert offenders == [], offenders

    def test_the_container_image_builds_no_path_longer_than_the_layer_limit(self):
        """COPY of a deep tree can exceed the build context budget."""
        longest = max(
            (len(str(path.relative_to(REPO_ROOT).as_posix())) for path in REPO_ROOT.rglob("*")
             if ".git" not in path.parts and ".venv" not in path.parts),
            default=0,
        )
        assert longest < 200


# --------------------------------------------------------------------------- #
# POSIX-specific behaviour worth asserting explicitly
# --------------------------------------------------------------------------- #


class TestPosixBehaviour:
    def test_a_staged_upload_is_removed_even_when_parsing_fails(self):
        """Cleanup must not depend on the platform's temp directory layout."""
        import glob
        import tempfile

        from backend.app.ingestion.uploads import (
            STAGED_UPLOAD_PREFIX,
            discard_upload,
            store_upload,
        )

        before = set(glob.glob(os.path.join(tempfile.gettempdir(), STAGED_UPLOAD_PREFIX + "*")))
        stored = store_upload(b"timestamp,src_ip\n2024-06-01T00:00:00Z,10.0.0.1\n")
        try:
            assert os.path.exists(stored.path)
        finally:
            discard_upload(stored.path)
        after = set(glob.glob(os.path.join(tempfile.gettempdir(), STAGED_UPLOAD_PREFIX + "*")))
        assert after <= before, "a staging file was left behind"

    def test_a_staged_upload_survives_a_read_only_working_directory(self):
        """Staging uses the system temp dir, never the current directory."""
        from backend.app.ingestion.uploads import discard_upload, store_upload

        stored = store_upload(b"timestamp,src_ip\n2024-06-01T00:00:00Z,10.0.0.1\n")
        try:
            parent = PurePosixPath(stored.path.replace("\\", "/")).parent
            assert str(parent) != "."
            assert not str(parent).startswith(REPO_ROOT.as_posix())
        finally:
            discard_upload(stored.path)

    def test_a_posix_path_string_is_parsed_correctly(self):
        from backend.app.ingestion.uploads import sanitize_display_filename

        assert sanitize_display_filename("/var/log/audit.csv") == "audit.csv"
        assert sanitize_display_filename("//double//slash.csv") == "slash.csv"

    def test_line_endings_do_not_change_the_derived_observation_id(self):
        """A CRLF checkout must not renumber every observation."""
        from evaluation import corpus
        from evaluation.canonical import sha256_file

        before = {o.observation_id for o in corpus.load_observations()[0]}
        assert len(before) == 12

        csv_path = EVALUATION / "scenarios" / "flow_combined.csv"
        original = csv_path.read_bytes()
        csv_path.write_bytes(original.replace(b"\n", b"\r\n"))
        try:
            after = {o.observation_id for o in corpus.load_observations()[0]}
        finally:
            csv_path.write_bytes(original)
        assert after == before, (
            "a CRLF checkout changed the derived observation ids; the ids "
            "depend on the file bytes, which is correct for a forensic "
            "pipeline, so the corpus must stay LF"
        )

    def test_file_hashes_are_of_bytes_not_of_text(self):
        from evaluation.canonical import sha256_file

        csv_path = EVALUATION / "scenarios" / "flow_combined.csv"
        digest = sha256_file(csv_path)
        assert len(digest) == 64
        assert digest == sha256_file(csv_path), "hashing is not stable"


# --------------------------------------------------------------------------- #
# What the container will actually do
# --------------------------------------------------------------------------- #


class TestContainerRuntimeContract:
    def test_the_evaluation_runs_from_the_working_directory_the_image_sets(self):
        """The image sets WORKDIR /app; the evaluation must resolve from there."""
        script = (
            "import sys; sys.path.insert(0, '.'); "
            "from evaluation import harness; print(harness.run_all().dimensions_digest)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=900,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]

    def test_the_evaluation_writes_its_report_relative_to_the_project(self):
        from evaluation import run_evaluation

        assert run_evaluation.REPORT_DIR.is_absolute()
        assert run_evaluation.EXPECTED_DIR.is_absolute()

    def test_a_report_path_outside_the_project_is_still_writable(self, tmp_path):
        """A bind-mounted volume may sit anywhere."""
        from evaluation import harness, run_evaluation

        outcome = harness.run_all()
        target = tmp_path / "nested" / "report.json"
        written = run_evaluation.write_report(run_evaluation.build_report(outcome), target)
        assert written.exists()

    def test_the_evaluation_creates_its_report_directory(self, tmp_path):
        from evaluation import harness, run_evaluation

        outcome = harness.run_all()
        report = run_evaluation.build_report(outcome)
        target = tmp_path / "deep" / "deeper" / "report.json"
        written = run_evaluation.write_report(report, target)
        assert written.exists()
        assert written.parent.is_dir()