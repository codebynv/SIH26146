"""Format detection and shared parser plumbing (SIH26146 - Step 2).

The detected format is derived from the *content* of an upload.  A client
supplied filename or extension is only ever used as a cross-check and can
never override what the bytes actually say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from backend.app.domain.enums import SourceFormat

__all__ = [
    "IngestionError",
    "UnsupportedFormatError",
    "RawRecord",
    "decode_text",
    "detect_format",
    "coerce_declared_format",
    "raise_for_mismatched_format",
]


class IngestionError(Exception):
    """A controlled, reportable ingestion failure.

    Raised for malformed input.  Carries a stable ``code`` so callers and
    tests can assert on the failure without matching free text.
    """

    def __init__(self, message: str, *, code: str = "MALFORMED_INPUT") -> None:
        super().__init__(message)
        self.code = code


class UnsupportedFormatError(IngestionError):
    """The content is not a format BitTrace AI can ingest."""

    def __init__(self, message: str, *, code: str = "UNSUPPORTED_FORMAT") -> None:
        super().__init__(message, code=code)


@dataclass(frozen=True)
class RawRecord:
    """One unparsed source record plus a locator for where it came from."""

    data: Mapping[str, Any]
    locator: str
    extra_issues: tuple[str, ...] = field(default=())


def decode_text(content: bytes | str, *, encoding: str = "utf-8") -> str:
    """Decode bytes to text, tolerating a BOM and falling back to latin-1.

    ``utf-8-sig`` transparently strips a UTF-8 BOM.  latin-1 never fails,
    which keeps a mis-encoded file a *reported* parse failure downstream
    rather than an unhandled exception here.
    """
    if isinstance(content, str):
        return content.lstrip("\ufeff")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _looks_like_json(text: str) -> bool:
    """True when the document is *structurally* JSON.

    Detection is deliberately prefix-based rather than parse-based. If it
    required a successful parse, malformed JSON would fall through to the CSV
    parser and be ingested as nonsense rows instead of raising a controlled
    error. A document that opens with '{' or '[' is JSON-shaped and belongs
    to the JSON parser, valid or not.
    """
    return text.lstrip()[:1] in {"{", "["}


def _looks_like_xml(text: str) -> bool:
    return text.lstrip().startswith("<")


def detect_format(content: bytes | str) -> SourceFormat:
    """Detect the wire format of *content* from the bytes themselves.

    Detection is content-first and structural: XML for markup, JSON for a
    document that opens with ``{`` or ``[``, CSV otherwise.  An empty document
    is reported as CSV because there is nothing that contradicts it and an
    empty CSV is a valid, if boring, dataset.
    """
    text = decode_text(content)
    if _looks_like_xml(text):
        return SourceFormat.XML
    if _looks_like_json(text):
        return SourceFormat.JSON
    return SourceFormat.CSV


def coerce_declared_format(declared: Any) -> SourceFormat | None:
    """Normalise a client-declared format name, or return ``None``.

    Accepts a :class:`SourceFormat`, a string, or a filename extension.  An
    unrecognised declaration is ``None`` (ignored) rather than an error, since
    the detected format is authoritative either way.
    """
    if declared is None:
        return None
    if isinstance(declared, SourceFormat):
        return declared
    text = str(declared).strip().lower().lstrip(".")
    if not text:
        return None
    try:
        return SourceFormat(text)
    except ValueError:
        pass
    if text.endswith(".json"):
        return SourceFormat.JSON
    if text.endswith(".xml"):
        return SourceFormat.XML
    if text.endswith(".csv") or text.endswith(".tsv"):
        return SourceFormat.CSV
    return None


def raise_for_mismatched_format(
    detected: SourceFormat, declared: SourceFormat | None
) -> None:
    """Reject an upload whose declared format contradicts its content.

    This is what stops a ``.csv``-named file full of XML (or the reverse) from
    being parsed with the wrong parser and producing nonsense.
    """
    if declared is not None and declared is not detected:
        raise IngestionError(
            "declared format %r does not match the detected format %r"
            % (declared.value, detected.value),
            code="FORMAT_MISMATCH",
        )
