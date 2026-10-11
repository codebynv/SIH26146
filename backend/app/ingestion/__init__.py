"""Data ingestion engine (SIH26146 - Step 2).

Reads CSV, JSON and XML investigative data, validates and normalises it onto
the canonical SIH26146 field set, preserves provenance, and removes only
provably identical duplicates.

This package deliberately performs no attribution, correlation, graph
building, risk scoring or persistence.
"""

from backend.app.ingestion.aliases import FIELD_ALIASES, normalize_keys, resolve_key
from backend.app.ingestion.dedup import (
    DeduplicationGroup,
    DeduplicationResult,
    deduplicate,
    record_fingerprint,
)
from backend.app.ingestion.models import NormalizedRecord, new_provenance
from backend.app.ingestion.parsers import (
    IngestionError,
    UnsupportedFormatError,
    detect_format,
    parse_csv,
    parse_json,
    parse_xml,
)
from backend.app.ingestion.service import IngestionResult, IngestionService
from backend.app.ingestion.uploads import (
    MAX_UPLOAD_BYTES,
    UploadError,
    sanitize_display_filename,
    store_upload,
)

__all__ = [
    "DeduplicationGroup",
    "DeduplicationResult",
    "FIELD_ALIASES",
    "IngestionError",
    "IngestionResult",
    "IngestionService",
    "MAX_UPLOAD_BYTES",
    "NormalizedRecord",
    "UnsupportedFormatError",
    "UploadError",
    "deduplicate",
    "detect_format",
    "new_provenance",
    "normalize_keys",
    "parse_csv",
    "parse_json",
    "parse_xml",
    "record_fingerprint",
    "resolve_key",
    "sanitize_display_filename",
    "store_upload",
]
