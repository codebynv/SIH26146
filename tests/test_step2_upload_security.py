"""Step 2 - secure upload handling and the /ingest endpoint."""

from __future__ import annotations

import os
import stat

import pytest

from backend.app.domain.enums import SourceFormat
from backend.app.ingestion.parsers import IngestionError
from backend.app.ingestion.uploads import (
    STAGED_UPLOAD_PREFIX,
    UploadError,
    discard_upload,
    sanitize_display_filename,
    store_upload,
)
from tests.conftest import TXID_A, csv_bytes, csv_row, json_bytes

CSV_CONTENT = csv_bytes("2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,", "1.2.3.5,443")
JSON_CONTENT = json_bytes([{"txid": TXID_A}])
XML_CONTENT = ("<records><record><txid>%s</txid></record></records>" % TXID_A).encode()


# ---------------------------------------------------------------- filenames
@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\cmd.exe", "cmd.exe"),
        ("/etc/shadow", "shadow"),
        ("/absolute/path/report.csv", "report.csv"),
        ("normal.csv", "normal.csv"),
        ("", None),
        (None, None),
        (".", None),
        ("..", None),
        ("   ", None),
    ],
)
def test_client_filename_is_reduced_to_a_bare_display_name(supplied, expected):
    assert sanitize_display_filename(supplied) == expected


def test_control_characters_are_stripped_from_display_name():
    assert sanitize_display_filename("re\x00port\x1b.csv") == "report.csv"


def test_overlong_display_name_is_truncated():
    name = sanitize_display_filename("a" * 500 + ".csv")
    assert len(name) == 255


# ---------------------------------------------------------------- storage
def test_stored_path_never_contains_the_client_filename(tmp_path):
    stored = store_upload(
        CSV_CONTENT, client_filename="../../evil.csv", destination=str(tmp_path)
    )
    try:
        assert stored.display_filename == "evil.csv"
        # The basename of the real path is server-generated, not client-derived.
        assert os.path.basename(stored.path).startswith("bittrace-upload-")
        assert "evil" not in stored.path
    finally:
        discard_upload(stored.path)


def test_stored_file_is_inside_the_requested_directory(tmp_path):
    stored = store_upload(CSV_CONTENT, destination=str(tmp_path))
    try:
        assert os.path.dirname(os.path.abspath(stored.path)) == str(tmp_path)
    finally:
        discard_upload(stored.path)


def test_stored_content_matches_the_upload(tmp_path):
    stored = store_upload(CSV_CONTENT, destination=str(tmp_path))
    try:
        with open(stored.path, "rb") as handle:
            assert handle.read() == CSV_CONTENT
    finally:
        discard_upload(stored.path)


def test_temporary_file_is_removed_by_discard(tmp_path):
    stored = store_upload(CSV_CONTENT, destination=str(tmp_path))
    path = stored.path
    discard_upload(path)
    assert not os.path.exists(path)


def test_discard_is_idempotent(tmp_path):
    stored = store_upload(CSV_CONTENT, destination=str(tmp_path))
    discard_upload(stored.path)
    discard_upload(stored.path)
    discard_upload(None)


def test_ingest_path_cleans_up_a_file_it_owns(tmp_path):
    """A staged upload the service is told it owns is removed afterwards."""
    from backend.app.ingestion.service import IngestionService

    stored = store_upload(CSV_CONTENT, destination=str(tmp_path))
    path = stored.path
    IngestionService().ingest_path(path, owned=True)
    assert not os.path.exists(path)


def test_ingest_path_cleans_up_even_when_parsing_fails(tmp_path):
    from backend.app.ingestion.service import IngestionService

    stored = store_upload(b"{not json", destination=str(tmp_path))
    path = stored.path
    with pytest.raises(IngestionError):
        IngestionService().ingest_path(path, owned=True)
    assert not os.path.exists(path)


def test_ingest_path_preserves_a_caller_owned_file(tmp_path):
    """The default must never delete the caller's evidence file.

    A forensic tool that destroys its own source data on read is unusable, so
    ingestion leaves caller-owned paths exactly where it found them.
    """
    from backend.app.ingestion.service import IngestionService

    path = tmp_path / "investigator_source.csv"
    path.write_bytes(CSV_CONTENT)
    result = IngestionService().ingest_path(str(path))
    assert result.record_count > 0
    assert path.exists()
    assert path.read_bytes() == CSV_CONTENT


def test_ingest_path_preserves_a_caller_owned_file_on_failure(tmp_path):
    """A parse failure must not escalate into deleting the source file."""
    from backend.app.ingestion.service import IngestionService

    path = tmp_path / "broken_evidence.json"
    path.write_bytes(b'{"txid": "abc",}')
    with pytest.raises(IngestionError):
        IngestionService().ingest_path(str(path))
    assert path.exists()
    assert path.read_bytes() == b'{"txid": "abc",}'


def test_ingest_path_preserves_a_caller_owned_file_that_looks_staged(tmp_path):
    """Ownership is never inferred from a filename.

    A caller's own file that happens to start with the staging prefix must
    still survive; otherwise a naming coincidence becomes unrecoverable data
    loss.
    """
    from backend.app.ingestion.service import IngestionService

    path = tmp_path / ("%sevil.csv" % STAGED_UPLOAD_PREFIX)
    path.write_bytes(CSV_CONTENT)
    IngestionService().ingest_path(str(path))
    assert path.exists()


def test_ingest_path_owned_true_does_not_rescue_an_unowned_default(tmp_path):
    """``owned=True`` is what authorises deletion; it is never automatic."""
    from backend.app.ingestion.service import IngestionService

    path = tmp_path / "kept.csv"
    path.write_bytes(CSV_CONTENT)
    IngestionService().ingest_path(str(path), owned=False)
    assert path.exists()


def test_ingest_path_preserves_the_source_file_across_many_reads(tmp_path):
    """Repeated ingestion of the same evidence file is non-destructive."""
    from backend.app.ingestion.service import IngestionService

    path = tmp_path / "corpus.csv"
    path.write_bytes(CSV_CONTENT)
    service = IngestionService()
    for _ in range(3):
        service.ingest_path(str(path))
        assert path.exists()
    assert path.read_bytes() == CSV_CONTENT


def test_is_staged_upload_recognises_only_staging_names():
    from backend.app.ingestion.uploads import is_staged_upload

    assert is_staged_upload(os.path.join("tmp", "bittrace-upload-abc123.dat"))
    assert not is_staged_upload(os.path.join("tmp", "evidence.csv"))
    assert not is_staged_upload(None)
    assert not is_staged_upload("")


def test_ingest_path_reports_unreadable_file(tmp_path):
    from backend.app.ingestion.service import IngestionService

    with pytest.raises(IngestionError) as excinfo:
        IngestionService().ingest_path(str(tmp_path / "nope.csv"))
    assert excinfo.value.code == "UNREADABLE_FILE"


def test_upload_directory_does_not_leak_after_success(tmp_path):
    stored = store_upload(CSV_CONTENT, destination=str(tmp_path))
    discard_upload(stored.path)
    assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------- validation
def test_empty_upload_is_rejected():
    with pytest.raises(UploadError) as excinfo:
        store_upload(b"")
    assert excinfo.value.code == "EMPTY_UPLOAD"


def test_oversized_upload_is_rejected():
    with pytest.raises(UploadError) as excinfo:
        store_upload(b"a" * 100, max_bytes=10)
    assert excinfo.value.code == "UPLOAD_TOO_LARGE"


def test_non_bytes_content_is_rejected():
    with pytest.raises(UploadError) as excinfo:
        store_upload("string is not bytes")
    assert excinfo.value.code == "INVALID_CONTENT_TYPE"


def test_rejected_upload_writes_nothing(tmp_path):
    with pytest.raises(UploadError):
        store_upload(b"", destination=str(tmp_path))
    assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------- format trust
def _staged_uploads() -> set:
    return {
        name
        for name in os.listdir(_temp_dir())
        if name.startswith("bittrace-upload-")
    }


def _temp_dir() -> str:
    import tempfile

    return tempfile.gettempdir()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (CSV_CONTENT, SourceFormat.CSV),
        (JSON_CONTENT, SourceFormat.JSON),
        (XML_CONTENT, SourceFormat.XML),
    ],
)
def test_format_is_detected_from_content(content, expected):
    stored = store_upload(content)
    try:
        assert stored.detected_format is expected
    finally:
        discard_upload(stored.path)


def test_csv_named_xml_file_is_parsed_as_xml_not_csv(tmp_path):
    """The extension must never choose the parser; the content does."""
    stored = store_upload(XML_CONTENT, client_filename="actually_xml.csv")
    try:
        assert stored.detected_format is SourceFormat.XML
    finally:
        discard_upload(stored.path)

def test_json_named_csv_file_is_parsed_as_json(tmp_path):
    stored = store_upload(JSON_CONTENT, client_filename="actually_json.csv")
    try:
        assert stored.detected_format is SourceFormat.JSON
    finally:
        discard_upload(stored.path)


def test_mismatched_declared_format_is_rejected():
    with pytest.raises(UploadError) as excinfo:
        store_upload(CSV_CONTENT, declared_format="json")
    assert excinfo.value.code == "FORMAT_MISMATCH"


def test_matching_declared_format_is_accepted(tmp_path):
    stored = store_upload(CSV_CONTENT, declared_format="csv", destination=str(tmp_path))
    try:
        assert stored.declared_format is SourceFormat.CSV
    finally:
        discard_upload(stored.path)


def test_declared_extension_is_accepted_as_a_hint():
    stored = store_upload(CSV_CONTENT, declared_format="flows.CSV")
    try:
        assert stored.detected_format is SourceFormat.CSV
    finally:
        discard_upload(stored.path)


def test_unrecognised_declaration_is_ignored_rather_than_fatal(tmp_path):
    stored = store_upload(CSV_CONTENT, declared_format="parquet")
    try:
        assert stored.declared_format is None
        assert stored.detected_format is SourceFormat.CSV
    finally:
        discard_upload(stored.path)


# ---------------------------------------------------------------- endpoint
def test_ingest_endpoint_accepts_csv(client):
    response = client.post(
        "/ingest", files={"file": ("flows.csv", CSV_CONTENT, "text/csv")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source_format"] == "csv"
    assert body["record_count"] == 2


def test_ingest_endpoint_accepts_json(client):
    response = client.post(
        "/ingest", files={"file": ("data.json", JSON_CONTENT, "application/json")}
    )
    assert response.status_code == 200
    assert response.json()["source_format"] == "json"


def test_ingest_endpoint_accepts_xml(client):
    response = client.post(
        "/ingest", files={"file": ("data.xml", XML_CONTENT, "application/xml")}
    )
    assert response.status_code == 200
    assert response.json()["source_format"] == "xml"


def test_ingest_endpoint_ignores_a_lying_extension(client):
    response = client.post(
        "/ingest", files={"file": ("data.json", XML_CONTENT, "application/json")}
    )
    assert response.status_code == 200
    assert response.json()["source_format"] == "xml"


def test_ingest_endpoint_rejects_empty_upload(client):
    response = client.post("/ingest", files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 400


def test_ingest_endpoint_reports_malformed_payload(client):
    response = client.post(
        "/ingest", files={"file": ("bad.json", b"{not json", "application/json")}
    )
    assert response.status_code == 422


def test_ingest_endpoint_rejects_mismatched_declared_format(client):
    response = client.post(
        "/ingest",
        files={"file": ("flows.csv", CSV_CONTENT, "text/csv")},
        data={"declared_format": "json"},
    )
    assert response.status_code == 400


def test_ingest_endpoint_requires_a_file(client):
    assert client.post("/ingest").status_code == 422


def test_ingest_endpoint_survives_traversal_filename(client, tmp_path):
    before = set(os.listdir(tmp_path))
    response = client.post(
        "/ingest", files={"file": ("../../evil.csv", CSV_CONTENT, "text/csv")}
    )
    assert response.status_code == 200
    assert set(os.listdir(tmp_path)) == before
    assert not os.path.exists("evil.csv")
    assert not os.path.exists("../../evil.csv")


def test_ingest_endpoint_reports_record_issues(client):
    content = csv_bytes("2024-05-01T00:00:00Z,not-an-ip,,,,,,,,,,,,")
    body = client.post(
        "/ingest", files={"file": ("flows.csv", content, "text/csv")}
    ).json()
    assert body["error_count"] >= 1
    codes = [issue["code"] for issue in body["issues"]]
    assert "INVALID_FIELD" in codes


def test_ingest_endpoint_preserves_canonical_fields(client):
    content = csv_bytes(
        csv_row(
            timestamp="2024-05-01T12:00:00Z",
            src_ip="203.0.113.5",
            dst_ip="198.51.100.9",
            src_port=51000,
            dst_port=443,
            txid=TXID_A,
            fee="0.0001",
            script_type="P2TR",
            geo_country="IN",
            asn="AS15169",
        )
    )
    body = client.post(
        "/ingest", files={"file": ("flows.csv", content, "text/csv")}
    ).json()
    record = body["records"][0]["record"]
    assert record["txid"] == TXID_A
    assert record["script_type"] == "P2TR"
    assert record["asn"] == 15169
    assert record["fee"] == pytest.approx(0.0001)


def test_ingest_endpoint_includes_provenance(client):
    body = client.post(
        "/ingest", files={"file": ("flows.csv", CSV_CONTENT, "text/csv")}
    ).json()
    provenance = body["records"][0]["record"]["provenance"]
    assert provenance["source_file"] == "flows.csv"
    assert provenance["source_format"] == "csv"
    assert provenance["source_record"] == "row 2"
    assert provenance["ingestion_id"] == body["ingestion_id"]


def test_ingest_endpoint_keeps_unmapped_fields(client):
    content = b"timestamp,block_height\n2024-05-01T00:00:00Z,840000\n"
    body = client.post(
        "/ingest", files={"file": ("blocks.csv", content, "text/csv")}
    ).json()
    assert body["records"][0]["record"]["extra"] == {"block_height": "840000"}


def test_ingest_endpoint_leaves_no_temporary_files(client):
    """The real property: an ingest creates no new staging file."""
    before = _staged_uploads()
    client.post("/ingest", files={"file": ("flows.csv", CSV_CONTENT, "text/csv")})
    assert _staged_uploads() == before


def test_ingest_endpoint_leaves_no_temporary_files_on_error(client):
    before = _staged_uploads()
    client.post("/ingest", files={"file": ("bad.json", b"{not json", "application/json")})
    client.post("/ingest", files={"file": ("empty.csv", b"", "text/csv")})
    assert _staged_uploads() == before
