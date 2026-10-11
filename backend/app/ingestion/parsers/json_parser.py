"""JSON parser (SIH26146 - Step 2)."""

from __future__ import annotations

import json
from typing import Any, Iterator, Mapping

from backend.app.ingestion.parsers.base import (
    IngestionError,
    RawRecord,
    decode_text,
)

__all__ = ["parse_json", "RECORD_CONTAINER_KEYS"]

#: Keys commonly used to wrap a list of records inside a JSON object.
RECORD_CONTAINER_KEYS = (
    "data",
    "records",
    "rows",
    "items",
    "results",
    "events",
    "transactions",
    "observations",
    "entries",
)


def _extract_records(payload: Any) -> list[tuple[Mapping[str, Any], str]]:
    if isinstance(payload, list):
        return _from_list(payload, "$")
    if isinstance(payload, Mapping):
        for key in RECORD_CONTAINER_KEYS:
            if key in payload and isinstance(payload[key], list):
                return _from_list(payload[key], "$['%s']" % key)
        # A bare object is a single record.
        return [(payload, "$")]
    raise IngestionError(
        "JSON input must be an object or an array, got %s"
        % type(payload).__name__,
        code="UNEXPECTED_JSON_SHAPE",
    )


def _from_list(items: list[Any], path: str) -> list[tuple[Mapping[str, Any], str]]:
    records: list[tuple[Mapping[str, Any], str]] = []
    for index, item in enumerate(items):
        locator = "%s[%d]" % (path, index)
        if isinstance(item, Mapping):
            records.append((item, locator))
        else:
            # A non-object element cannot be a record; report it rather than
            # inventing a schema for it.
            raise IngestionError(
                "expected a JSON object at %s but found %s"
                % (locator, type(item).__name__),
                code="UNEXPECTED_JSON_ELEMENT",
            )
    return records


def parse_json(content: bytes | str) -> Iterator[RawRecord]:
    """Parse JSON into raw records.

    Handles a top-level array, a single object, and an object wrapping a list
    under a conventional key such as ``data`` or ``records``.  An empty array
    is a valid, empty dataset.  Malformed JSON raises
    :class:`~backend.app.ingestion.parsers.base.IngestionError`.
    """
    text = decode_text(content)
    if not text.strip():
        return
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise IngestionError(
            "malformed JSON: %s" % exc, code="MALFORMED_JSON"
        ) from exc
    for data, locator in _extract_records(payload):
        yield RawRecord(data=data, locator=locator)
