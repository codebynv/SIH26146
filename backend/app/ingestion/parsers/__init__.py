"""Format parsers (SIH26146 - Step 2)."""

from backend.app.ingestion.parsers.base import (
    IngestionError,
    RawRecord,
    UnsupportedFormatError,
    coerce_declared_format,
    decode_text,
    detect_format,
    raise_for_mismatched_format,
)
from backend.app.ingestion.parsers.csv_parser import parse_csv
from backend.app.ingestion.parsers.json_parser import parse_json
from backend.app.ingestion.parsers.xml_parser import parse_xml

__all__ = [
    "IngestionError",
    "RawRecord",
    "UnsupportedFormatError",
    "coerce_declared_format",
    "decode_text",
    "detect_format",
    "raise_for_mismatched_format",
    "parse_csv",
    "parse_json",
    "parse_xml",
    "PARSERS",
]

#: Wire format -> parser.  Selected from the *detected* format, never from a
#: client-supplied extension.
PARSERS = {
    "csv": parse_csv,
    "json": parse_json,
    "xml": parse_xml,
}
