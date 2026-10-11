"""CSV parser (SIH26146 - Step 2)."""

from __future__ import annotations

import csv
import io
import sys
from typing import Iterator

from backend.app.ingestion.parsers.base import (
    IngestionError,
    RawRecord,
    decode_text,
)

__all__ = ["parse_csv", "sniff_delimiter"]

_SNIFF_DELIMITERS = ",;\t|"


def _raise_field_limit() -> None:
    """Allow long cells, since lists are often JSON-encoded inside a CSV field."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sniff_delimiter(sample: str) -> str:
    """Detect the delimiter, defaulting to comma.

    ``csv.Sniffer`` is unreliable on short or single-column samples, so any
    failure falls back to a comma rather than raising.
    """
    if not sample.strip():
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=_SNIFF_DELIMITERS).delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        for candidate in _SNIFF_DELIMITERS:
            if candidate in first_line:
                return candidate
        return ","


def parse_csv(content: bytes | str) -> Iterator[RawRecord]:
    """Parse CSV text into raw records, one per data row.

    Row numbering is 1-based and counts the header, so ``"row 2"`` is the
    first data row -- matching what an analyst sees in a spreadsheet.
    Completely empty input yields no records rather than an error.
    """
    _raise_field_limit()
    text = decode_text(content)
    if not text.strip():
        return

    delimiter = sniff_delimiter(text[:8192])
    # StringIO, not text.splitlines(): the csv reader needs the line
    # terminators to recognise a quoted field that spans several lines.
    # splitlines() discards them, which silently joins "line1\nline2" into
    # "line1line2" and corrupts the value.
    reader = csv.reader(
        io.StringIO(text, newline=""), delimiter=delimiter, skipinitialspace=True
    )
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise IngestionError(
            "malformed CSV: %s" % exc, code="MALFORMED_CSV"
        ) from exc

    rows = [row for row in rows if row]
    if not rows:
        return

    header = [column.strip().lstrip("\ufeff") for column in rows[0]]
    if not header:
        return

    for offset, row in enumerate(rows[1:], start=2):
        issues: list[str] = []
        data: dict[str, str] = {}
        overflow = row[len(header) :]
        for index, column in enumerate(header):
            if index < len(row):
                data[column] = row[index]
            else:
                data[column] = ""
                issues.append(
                    "row has %d column(s) but the header declares %d; missing "
                    "values were treated as empty" % (len(row), len(header))
                )
        if overflow:
            issues.append(
                "row has %d value(s) beyond the %d declared column(s); the extra "
                "values were discarded" % (len(overflow), len(header))
            )
        yield RawRecord(
            data=data, locator="row %d" % offset, extra_issues=tuple(issues)
        )
