"""Shared fixtures for the BitTrace AI test suite (SIH26146, Steps 1-3)."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Optional

import pytest

from backend.app.domain.models import Provenance
from backend.app.ingestion.models import NormalizedRecord

#: A syntactically plausible Bitcoin txid.
TXID_A = "a" * 64
TXID_B = "b" * 64

ADDR_1 = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
ADDR_2 = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
ADDR_3 = "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy"


@pytest.fixture
def service():
    """A default IngestionService with no custom upload directory."""
    from backend.app.ingestion.service import IngestionService

    return IngestionService()


@pytest.fixture
def client():
    """A FastAPI TestClient for the application under test."""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def provenance() -> Provenance:
    """Provenance for hand-built records in unit tests."""
    from datetime import datetime, timezone

    return Provenance(
        ingestion_id="test-ingestion",
        source_file="unit-test.csv",
        source_format="csv",
        source_record="row 2",
        ingested_at=datetime(2024, 5, 1, tzinfo=timezone.utc),
        context={"test": True},
    )


def make_record(
    provenance: Provenance,
    /,
    timestamp: Any = "2024-05-01T12:00:00Z",
    **fields: Any,
) -> NormalizedRecord:
    """Build a :class:`NormalizedRecord` from canonical keyword fields.

    Extra keyword arguments are merged over the defaults, so a test only has
    to state the fields it cares about.
    """
    raw: dict[str, Any] = {"timestamp": timestamp}
    raw.update(fields)
    for key in ("input_addresses", "output_addresses", "input_amounts", "output_amounts"):
        raw.setdefault(key, [])
    return NormalizedRecord.from_raw(raw, provenance)


CSV_HEADER = (
    "timestamp,src_ip,dst_ip,src_port,dst_port,txid,input_addresses,"
    "output_addresses,input_amounts,output_amounts,fee,script_type,"
    "geo_country,asn"
)


def csv_row(**fields: Any) -> str:
    """Build one correctly-aligned CSV data row from canonical field names.

    Emitting exactly the header's columns in order removes the single most
    common source of error in these tests: hand-counting commas.
    """
    values = []
    for column in CSV_HEADER.split(","):
        value = fields.pop(column, "")
        if value is None:
            value = ""
        if isinstance(value, (list, tuple)):
            value = ";".join(str(item) for item in value)
        elif isinstance(value, bool):
            value = "true" if value else "false"
        text = str(value)
        if any(ch in text for ch in ',"\n'):
            text = '"%s"' % text.replace('"', '""')
        values.append(text)
    if fields:
        raise TypeError("unknown CSV field(s): %s" % sorted(fields))
    return ",".join(values)


def csv_bytes(*rows: str, header: Optional[str] = None) -> bytes:
    """Build a CSV document from a header and rows."""
    if header is None:
        header = CSV_HEADER
    lines = [header, *rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


def xml_bytes(*records: str, root: str = "records", item: str = "record") -> bytes:
    body = "".join(records)
    return ("<%s>%s</%s>" % (root, body, root)).encode("utf-8")


def dec(value: Any) -> Decimal:
    return Decimal(str(value))
