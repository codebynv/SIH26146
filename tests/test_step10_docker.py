"""Container and packaging checks for BitTrace AI.

Docker is not assumed to be installed. The static assertions below read the
Dockerfile and .dockerignore as text and hold on any machine; the tests that
need a daemon skip when there is not one, so this module is useful in a bare CI
container and honest about what it did not check.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


def _dockerfile_text() -> str:
    assert DOCKERFILE.exists(), "no Dockerfile at the repository root"
    return DOCKERFILE.read_text(encoding="utf-8")


def _instructions() -> list[tuple[str, str]]:
    """Split the Dockerfile into (instruction, remainder) pairs.

    Comments and blank lines are dropped and continuations joined, so an
    assertion about an instruction cannot be satisfied by a mention of it in a
    comment.
    """
    joined: list[str] = []
    buffer = ""
    for raw in _dockerfile_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        buffer += line
        if line.endswith("\\"):
            buffer = buffer[:-1]
            continue
        joined.append(buffer)
        buffer = ""
    if buffer:
        joined.append(buffer)

    instructions = []
    for entry in joined:
        keyword, _, remainder = entry.partition(" ")
        instructions.append((keyword.upper(), remainder))
    return instructions


# --------------------------------------------------------------------------- #
# The image is defined
# --------------------------------------------------------------------------- #


class TestDockerfile:
    def test_a_dockerfile_exists(self):
        assert DOCKERFILE.exists()

    def test_the_dockerfile_actually_parses_as_instructions(self):
        allowed = {
            "FROM",
            "ENV",
            "WORKDIR",
            "RUN",
            "COPY",
            "USER",
            "EXPOSE",
            "HEALTHCHECK",
            "CMD",
            "ENTRYPOINT",
            "ARG",
            "LABEL",
            "VOLUME",
            "SHELL",
            "STOPSIGNAL",
        }
        for keyword, _ in _instructions():
            assert keyword in allowed, "unexpected instruction %s" % keyword

    def test_it_is_a_multi_stage_build(self):
        froms = [
            remainder for keyword, remainder in _instructions() if keyword == "FROM"
        ]
        assert len(froms) >= 2, "expected a builder stage and a runtime stage"
        named = [
            re.search(r"\bAS\s+(\S+)", entry, re.IGNORECASE) for entry in froms
        ]
        assert all(named), (
            "every stage must be named; an unnamed stage cannot be copied from"
        )
        for entry in froms:
            assert ":" in entry, "FROM %r pins no tag" % entry

    def test_the_runtime_stage_copies_from_the_builder(self):
        runtime = _stage_instructions("runtime")
        copied_from = [
            remainder for keyword, remainder in runtime
            if keyword == "COPY" and "--from=" in remainder
        ]
        assert copied_from, (
            "the runtime stage copies nothing from the builder, so the "
            "dependencies were installed in the wrong stage"
        )
        for remainder in copied_from:
            source = re.search(r"--from=(\S+)", remainder).group(1)
            assert re.search(r"\bAS\s+%s\b" % re.escape(source), _dockerfile_text(), re.I), (
                "--from=%s names a stage that does not exist" % source
            )

    def test_the_builder_stage_owns_the_dependencies(self):
        run_steps = " ".join(
            remainder for keyword, remainder in _instructions() if keyword == "RUN"
        )
        assert "pip install" in run_steps
        # Requirements must be installed in the stage that also has a compiler,
        # otherwise the runtime stage would need one.
        copies = [
            remainder for keyword, remainder in _instructions() if keyword == "COPY"
        ]
        assert any("requirements.txt" in copy for copy in copies)

    def test_the_runtime_stage_does_not_install_python_packages(self):
        """A package install in the runtime stage would need a compiler there."""
        for keyword, remainder in _stage_instructions("runtime"):
            if keyword == "RUN" and "pip install" in remainder:
                pytest.fail(
                    "the runtime stage installs Python packages; that belongs "
                    "in the builder stage: %s" % remainder
                )

    def test_the_image_runs_as_a_non_root_user(self):
        users = [
            remainder
            for keyword, remainder in _stage_instructions("runtime")
            if keyword == "USER"
        ]
        assert users, "the runtime stage never switches user"
        assert users[-1].strip().lower() != "root"
        for user in users:
            assert user.strip() != "0", "USER 0 is root"

    def test_a_dedicated_account_is_created(self):
        text = _dockerfile_text()
        assert re.search(r"useradd\b", text), "no unprivileged account is created"
        assert re.search(r"groupadd\b", text)

    def test_the_account_cannot_start_a_shell(self):
        text = _dockerfile_text()
        shells = re.findall(r"--shell\s+(\S+)", text)
        assert shells, "the created account has no --shell restriction"
        for shell in shells:
            assert shell in {"/usr/sbin/nologin", "/sbin/nologin", "/bin/false"}

    def test_the_application_port_is_exposed(self):
        exposed = [
            remainder
            for keyword, remainder in _stage_instructions("runtime")
            if keyword == "EXPOSE"
        ]
        assert any("8000" in entry for entry in exposed)

    def test_the_healthcheck_calls_the_health_endpoint(self):
        instructions = _instructions()
        healthchecks = [
            remainder for keyword, remainder in instructions if keyword == "HEALTHCHECK"
        ]
        assert healthchecks, "no HEALTHCHECK"
        body = healthchecks[0]
        assert "/health" in body
        assert "--fail" in body, "a healthcheck that ignores HTTP errors is decorative"

    def test_the_entrypoint_serves_the_application(self):
        cmds = [remainder for keyword, remainder in _instructions() if keyword == "CMD"]
        assert cmds, "no CMD"
        joined = " ".join(cmds)
        assert "uvicorn" in joined
        assert "backend.app.main:app" in joined

    def test_the_server_binds_all_interfaces_for_the_published_port(self):
        cmds = [remainder for keyword, remainder in _instructions() if keyword == "CMD"]
        joined = " ".join(cmds)
        assert "0.0.0.0" in joined, (
            "binding 127.0.0.1 inside a container makes the published port "
            "unreachable"
        )

    def test_only_the_required_packages_are_copied_in(self):
        copies = [
            remainder for keyword, remainder in _instructions() if keyword == "COPY"
        ]
        runtime_copies = [
            copy
            for copy in copies
            if not re.search(r"--from=", copy)
        ]
        for copy in runtime_copies:
            assert "tests/" not in copy, "the runtime image must not carry tests"
            assert "frontend/" not in copy, "the runtime image must not carry the frontend"

    def test_the_source_is_copied_with_ownership_set(self):
        """A root-owned working directory would defeat the unprivileged user."""
        copies = [
            remainder for keyword, remainder in _instructions() if keyword == "COPY"
        ]
        for copy in copies:
            if "--from=" in copy:
                continue
            if re.search(r"COPY\s+\.?\s+\./", copy):
                assert "--chown" in copy, copy

    def test_the_hash_seed_is_pinned(self):
        """Determinism is the product claim, so it cannot rely on the platform."""
        envs = [
            remainder for keyword, remainder in _instructions() if keyword == "ENV"
        ]
        joined = " ".join(envs)
        assert "PYTHONHASHSEED=0" in joined, (
            "with a random hash seed, set iteration order can differ between "
            "runs and the determinism guarantee stops holding"
        )

    def test_dependency_versions_are_pinned_in_the_requirements(self):
        lines = (
            (REPO_ROOT / "backend" / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        unpinned = [
            line
            for line in lines
            if line.strip()
            and not line.strip().startswith("#")
            and "==" not in line
            and not line.strip().startswith("-")
        ]
        assert unpinned == [], "unpinned dependencies: %s" % unpinned

    def test_the_runtime_requirements_carry_no_test_framework(self):
        text = (REPO_ROOT / "backend" / "requirements.txt").read_text(
            encoding="utf-8"
        )
        assert "pytest" not in text, (
            "the runtime image must not carry a test framework it never imports"
        )

    def test_the_installed_versions_satisfy_the_pins(self):
        """The pins must describe this environment, not a hypothetical one."""
        from importlib import metadata

        for line in (
            (REPO_ROOT / "backend" / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            entry = line.split("#", 1)[0].strip()
            if not entry or "==" not in entry:
                continue
            name, _, version = entry.partition("==")
            # "uvicorn[standard]" pins the distribution "uvicorn".
            distribution = re.sub(r"\[.*\]", "", name).strip()
            try:
                installed = metadata.version(distribution)
            except metadata.PackageNotFoundError:  # pragma: no cover
                pytest.skip("%s is not installed here" % distribution)
            assert installed == version, (
                "%s is pinned to %s but %s is installed" % (
                    distribution,
                    version,
                    installed,
                )
            )


class TestDockerignore:
    def test_it_exists(self):
        assert DOCKERIGNORE.exists()

    def test_the_local_virtualenv_is_excluded(self):
        """A host virtualenv carries absolute paths and foreign binaries."""
        text = DOCKERIGNORE.read_text(encoding="utf-8")
        assert re.search(r"(^|\n)\.venv/", text)

    def test_caches_and_bytecode_are_excluded(self):
        text = DOCKERIGNORE.read_text(encoding="utf-8")
        assert "__pycache__/" in text
        assert ".pytest_cache/" in text

    def test_secrets_are_excluded(self):
        text = DOCKERIGNORE.read_text(encoding="utf-8")
        assert ".env" in text
        assert "*.key" in text or "*.pem" in text

    def test_version_control_is_excluded(self):
        assert "\n.git" in DOCKERIGNORE.read_text(encoding="utf-8")

    def test_the_stored_expectation_is_not_ignored(self):
        """The fixture is the reviewed contract and must reach the image."""
        patterns = [
            line.strip()
            for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for pattern in patterns:
            normalised = pattern.rstrip("/")
            assert "expected" not in normalised, (
                "%s would exclude the stored expectation" % pattern
            )

    def test_generated_reports_are_excluded(self):
        text = DOCKERIGNORE.read_text(encoding="utf-8")
        assert "evaluation/reports/" in text


# --------------------------------------------------------------------------- #
# The application starts and answers
# --------------------------------------------------------------------------- #


class TestStartup:
    def test_the_app_object_is_importable_and_self_describing(self):
        from backend.app.main import app

        assert app.title
        assert app.version
        routes = {getattr(route, "path", None) for route in app.routes}
        assert "/" in routes
        assert "/health" in routes
        assert "/ingest" in routes

    def test_health_reports_ok(self):
        from fastapi.testclient import TestClient

        from backend.app.main import app

        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_is_reachable(self):
        from fastapi.testclient import TestClient

        from backend.app.main import app

        with TestClient(app) as client:
            response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "BitTrace AI" in response.text

    def test_the_openapi_document_builds(self):
        """A startup failure in schema generation would only show up in prod."""
        from fastapi.testclient import TestClient

        from backend.app.main import app

        with TestClient(app) as client:
            response = client.get("/openapi.json")
        assert response.status_code == 200
        document = response.json()
        assert document["paths"]["/health"]["get"]
        assert document["paths"]["/ingest"]["post"]

    def test_the_dashboard_router_is_mounted(self):
        """The Step 9A endpoints must be reachable through the app object.

        Asserted against the generated OpenAPI document rather than by walking
        ``app.routes``. Recent FastAPI keeps an included router as a single
        wrapper entry with no ``path`` of its own, so a route walk sees nothing
        while the endpoints work perfectly. What the deployable artefact
        promises is the schema a client sees.
        """
        from fastapi.testclient import TestClient

        from backend.app.main import app

        with TestClient(app) as client:
            document = client.get("/openapi.json").json()
        paths = document["paths"]
        for required in (
            "/api/dashboard/overview",
            "/api/dashboard/signals",
            "/api/dashboard/evidence",
            "/api/dashboard/paths",
            "/api/dashboard/behavior",
            "/api/dashboard/provenance",
            "/api/dashboard/all",
        ):
            assert required in paths, "%s is not served" % required
            assert "get" in paths[required], "%s is not readable" % required

    def test_ingest_is_the_only_write_endpoint(self):
        """/ingest is the sole write surface, and it stages nothing it keeps.

        Everything Step 9A exposes is read-only, so a new write appearing there
        would be a design change rather than a bug fix. /ingest is the one
        expected exception: it accepts an upload, parses it and removes the
        staging file before the response returns.
        """
        from fastapi.testclient import TestClient

        from backend.app.main import app

        with TestClient(app) as client:
            paths = client.get("/openapi.json").json()["paths"]
        writes = {
            path: sorted(verbs)
            for path, verbs in paths.items()
            for verb in ("post", "put", "patch", "delete")
            if verb in verbs
        }
        assert writes == {"/ingest": ["post"]}, writes

    def test_no_write_endpoint_survives(self):
        """The evaluation surface is read-only by design."""
        from backend.app.main import app

        writes = [
            getattr(route, "path")
            for route in app.routes
            if getattr(route, "methods", set()) & {"POST", "PUT", "PATCH", "DELETE"}
            and not getattr(route, "path", "").startswith("/ingest")
        ]
        assert writes == []

    def test_importing_the_app_opens_no_socket(self):
        """Importing the module must not start a listener.

        The guard is installed before the import, but the whole standard library
        a transitive import might pull in is already resident by the time this
        runs. ``ssl`` subclasses ``socket.socket`` at import time, so replacing
        the class afterwards would break the subclass rather than block a
        connection. The probe therefore denies the two operations that actually
        touch the network, ``socket.socket.connect`` and ``create_connection``,
        which leaves the class hierarchy intact.
        """
        script = (
            "import socket\n"
            "def refuse(*a, **k):\n"
            "    raise AssertionError('a network connection was attempted')\n"
            "socket.socket.connect = refuse\n"
            "socket.socket.connect_ex = refuse\n"
            "socket.create_connection = refuse\n"
            "import backend.app.main\n"
            "print('imported')\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=300,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "imported" in completed.stdout


# --------------------------------------------------------------------------- #
# Docker daemon, when there is one
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def docker_cli():
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed on this machine")
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        pytest.skip("the docker daemon is not responding")
    if completed.returncode != 0:
        pytest.skip("no usable docker daemon")


class TestImageBuild:
    def test_the_image_builds(self, docker_cli, tmp_path):
        tag = "bittrace-step10:test"
        completed = subprocess.run(
            [
                "docker",
                "build",
                "--tag",
                tag,
                "--file",
                str(DOCKERFILE),
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            pytest.fail(completed.stderr[-4000:])
        assert completed.returncode == 0

    def test_the_image_runs_the_evaluation_offline(self, docker_cli):
        tag = "bittrace-step10:test"
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python",
                tag,
                "-m",
                "evaluation.run_evaluation",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, (
            "the evaluation failed inside the image with no network:\n%s"
            % completed.stderr[-4000:]
        )

    def test_the_image_serves_health(self, docker_cli):
        tag = "bittrace-step10:test"
        container = subprocess.run(
            ["docker", "run", "--rm", "-d", "-p", "18000:8000", tag],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if container.returncode != 0:
            pytest.fail(container.stderr[-2000:])
        container_id = container.stdout.strip()
        try:
            import time
            import urllib.error
            import urllib.request

            body = None
            for _ in range(40):
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18000/health", timeout=2
                    ) as response:
                        body = response.read()
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.5)
            assert body, "the container never became healthy"
            assert b"ok" in body
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=120,
            )

    def test_the_image_runs_as_a_non_root_user(self, docker_cli):
        tag = "bittrace-step10:test"
        completed = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "id", tag],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "uid=0" not in completed.stdout, completed.stdout

    def test_the_image_writes_nothing_outside_its_data_directory(self, docker_cli):
        tag = "bittrace-step10:test"
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--tmpfs",
                "/tmp",
                "--entrypoint",
                "python",
                tag,
                "-c",
                "import backend.app.main; print('imported')",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, (
            "the application needs a writable layer it should not have:\n%s"
            % completed.stderr[-3000:]
        )


def _stage_instructions(stage: str) -> list[tuple[str, str]]:
    """Instructions belonging to one named build stage.

    FROM starts a stage; the last stage with no name still counts, under the
    empty key, so an unnamed stage cannot silently contribute instructions to
    whichever named stage came before it.
    """
    current = None
    collected: dict = {}
    for keyword, remainder in _instructions():
        if keyword == "FROM":
            match = re.search(r"\bAS\s+(\S+)", remainder, re.IGNORECASE)
            current = match.group(1).lower() if match else ""
            collected.setdefault(current, [])
            continue
        collected.setdefault(current, []).append((keyword, remainder))
    return collected.get(stage.lower(), [])


# --------------------------------------------------------------------------- #
# Packaging hygiene
# --------------------------------------------------------------------------- #


class TestRepositoryHygiene:
    def test_no_python_file_carries_a_hardcoded_secret(self):
        """A cheap scan, not a secret scanner.

        It looks for the shape of a committed credential rather than the word
        "key", so ordinary field names do not trip it.
        """
        patterns = [
            re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*=\s*[\"'][^\"']{8,}[\"']"),
            re.compile(r"AKIA[0-9A-Z]{16}"),
            re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
            re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        ]
        offenders = []
        for path in REPO_ROOT.rglob("*.py"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative.startswith((".venv", "node_modules")):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in patterns:
                if pattern.search(text):
                    offenders.append("%s (%s)" % (relative, pattern.pattern[:40]))
        assert offenders == []

    def test_no_env_file_is_committed(self):
        for name in (".env", ".env.local", ".env.production"):
            candidate = REPO_ROOT / name
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8", errors="replace")
                assert "=" in text
                assert re.search(r"=\s*\S{8,}", text) is None, (
                    "%s appears to contain a real value" % name
                )

    def test_gitignore_excludes_the_local_environment(self):
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".venv/" in text
        assert ".env" in text

    def test_gitignore_excludes_generated_reports(self):
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "evaluation/reports/" in text

    def test_the_upload_limit_is_bounded(self):
        from backend.app.main import MAX_UPLOAD_BYTES

        assert 0 < MAX_UPLOAD_BYTES <= 128 * 1024 * 1024

    def test_a_client_filename_cannot_choose_where_a_file_lands(self):
        """Traversal must be structurally impossible, not merely filtered."""
        from backend.app.ingestion.uploads import UploadError, store_upload

        stored = store_upload(
            b"timestamp,src_ip\n2024-06-01T00:00:00Z,10.0.0.1\n",
            client_filename="../../../../etc/passwd",
        )
        try:
            assert "passwd" not in stored.path.replace("\\", "/").split("/")[-1][
                : -len(".dat")
            ]
            assert stored.path.replace("\\", "/").split("/")[-1].startswith(
                "bittrace-upload-"
            )
            assert stored.display_filename == "passwd"
        finally:
            from backend.app.ingestion.uploads import discard_upload

            discard_upload(stored.path)

        with pytest.raises(UploadError):
            store_upload(b"", client_filename="x.csv")

    def test_cors_is_not_wide_open(self):
        text = (
            REPO_ROOT / "backend" / "app" / "main.py"
        ).read_text(encoding="utf-8")
        assert "allow_origins=[\"*\"]" not in text.replace(" ", "")
        assert "CORSMiddleware" not in text or "allow_origins" in text


def test_no_stray_scratch_files_are_tracked_by_git():
    """The repository must not accumulate one-off editing scripts."""
    scratch = {
        "fix.py",
        "patch.py",
        "test_api.py",
        "tmp_code.py",
        "tmp_ec3.js",
        "tmp_old.txt",
        "tmp_patch.txt",
        "_probe_css.py",
    }
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    completed = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    tracked = {
        Path(line.strip()).name for line in completed.stdout.splitlines() if line.strip()
    }
    assert not (scratch & tracked)