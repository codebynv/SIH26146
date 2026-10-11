"""Step 2 - provenance is retained on every ingested record."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.ingestion.models import NormalizedRecord, new_provenance
from tests.conftest import TXID_A, csv_bytes, json_bytes, make_record


def test_ingestion_id_is_assigned(service):
    result = service.ingest_bytes(csv_bytes("2024-05-01T00:00:00Z,,,,,,,,,,,,,"))
    assert result.records[0].provenance.ingestion_id == result.ingestion_id


def test_all_records_from_one_run_share_the_ingestion_id(service):
    result = service.ingest_bytes(
        csv_bytes(
            "2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,",
            "2024-05-02T00:00:00Z,1.2.3.5,,,,,,,,,,,,",
        )
    )
    ids = {record.provenance.ingestion_id for record in result.records}
    assert ids == {result.ingestion_id}


def test_separate_runs_get_separate_ingestion_ids(service):
    first = service.ingest_bytes(csv_bytes("2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,"))
    second = service.ingest_bytes(csv_bytes("2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,"))
    assert first.ingestion_id != second.ingestion_id


def test_source_file_is_recorded(service):
    result = service.ingest_bytes(
        csv_bytes("2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,"), source_file="flows.csv"
    )
    assert result.records[0].provenance.source_file == "flows.csv"


def test_source_format_is_recorded(service):
    result = service.ingest_bytes(json_bytes([{"txid": TXID_A}]))
    assert result.records[0].provenance.source_format.value == "json"


def test_source_record_locator_is_recorded_per_row(service):
    result = service.ingest_bytes(
        csv_bytes(
            "2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,",
            "2024-05-02T00:00:00Z,1.2.3.5,,,,,,,,,,,,",
        )
    )
    locators = [record.provenance.source_record for record in result.records]
    assert locators == ["row 2", "row 3"]


def test_ingested_at_is_timezone_aware_utc(service):
    result = service.ingest_bytes(csv_bytes("2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,"))
    stamp = result.records[0].provenance.ingested_at
    assert stamp.tzinfo is not None
    assert stamp.utcoffset().total_seconds() == 0


def test_ingested_at_is_recent(service):
    result = service.ingest_bytes(csv_bytes("2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,"))
    stamp = result.records[0].provenance.ingested_at
    delta = abs((datetime.now(timezone.utc) - stamp).total_seconds())
    assert delta < 300


def test_ingestion_context_is_carried_through(service):
    result = service.ingest_bytes(
        csv_bytes("2024-05-01T00:00:00Z,1.2.3.4,,,,,,,,,,,,"),
        context={"analyst": "unit-test"},
    )
    assert result.records[0].provenance.context["analyst"] == "unit-test"


def test_issue_carries_the_source_locator(provenance):
    record = make_record(provenance, timestamp="nope")
    assert record.issues[0].source_record == "row 2"


def test_new_provenance_helper_defaults_are_safe():
    provenance = new_provenance(ingestion_id="abc")
    assert provenance.source_file is None
    assert provenance.source_format is None
    assert provenance.context == {}


def test_provenance_is_preserved_into_the_domain_model(service):
    from backend.app.domain.adapters import normalized_to_unified

    result = service.ingest_bytes(
        csv_bytes("2024-05-01T00:00:00Z,1.2.3.4,1.2.3.5,1,2,,,,,,,,,,"),
        source_file="flows.csv",
    )
    observation = normalized_to_unified(result.records[0])
    assert observation.provenance == result.records[0].provenance
    assert observation.provenance.source_file == "flows.csv"
    assert observation.provenance.source_record == "row 2"
    assert observation.provenance.ingestion_id == result.ingestion_id


def test_provenance_is_immutable(provenance):
    try:
        provenance.source_file = "tampered"
    except Exception:
        return
    raise AssertionError("Provenance should be frozen")


def test_record_is_immutable(provenance):
    record = make_record(provenance, src_ip="1.2.3.4")
    try:
        record.src_ip = "9.9.9.9"
    except Exception:
        return
    raise AssertionError("NormalizedRecord should be frozen")
