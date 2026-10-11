"""Secure upload handling (SIH26146 - Step 2).

Rules enforced here
-------------------
* The client-supplied filename is **never** used to build a path.  It is
  reduced to a bare display name and kept for the audit trail only.
* Temporary files are created with :func:`tempfile.mkstemp` inside a
  server-controlled directory, using a server-generated random name.  No part
  of the client's input reaches the filesystem path, so traversal via
  ``../../etc/passwd`` or an absolute path is structurally impossible.
* Temporary files are always cleaned up, including on the error path.
* The format is detected from the content, never trusted from the extension.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import PurePath
from typing import Optional

from backend.app.domain.enums import SourceFormat
from backend.app.ingestion.parsers import (
    IngestionError,
    coerce_declared_format,
    detect_format,
    raise_for_mismatched_format,
)

__all__ = [
    "UploadError",
    "StoredUpload",
    "sanitize_display_filename",
    "store_upload",
    "discard_upload",
    "is_staged_upload",
    "STAGED_UPLOAD_PREFIX",
    "MAX_UPLOAD_BYTES",
]

#: 32 MiB. Generous for investigative extracts, bounded so a single request
#: cannot exhaust server memory or disk.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

#: Prefix given to every server-generated staging file.  It is the *only*
#: signal that lets a caller recognise a temporary file as disposable.
STAGED_UPLOAD_PREFIX = "bittrace-upload-"

_MAX_DISPLAY_NAME = 255


def is_staged_upload(path: Optional[str]) -> bool:
    """True when *path* names a server-generated staging file.

    This is a *naming convention*, not proof of ownership.  It is used only
    to recover a human-readable display name; it deliberately does **not**
    authorise deletion.  Whether a file may be removed is always an explicit
    decision by the code that created it, never an inference from its name --
    inferring that would let any caller-owned file whose name happened to
    start with the prefix be deleted.
    """
    if not path:
        return False
    return PurePath(str(path).replace("\\", "/")).name.startswith(STAGED_UPLOAD_PREFIX)


class UploadError(Exception):
    """A rejected upload."""

    def __init__(self, message: str, *, code: str = "INVALID_UPLOAD") -> None:
        super().__init__(message)
        self.code = code


def sanitize_display_filename(client_filename: Optional[str]) -> Optional[str]:
    """Reduce a client filename to a safe, human-readable display name.

    Strips every directory component, so ``"../../etc/passwd"`` becomes
    ``"passwd"``.  The result is only ever stored as metadata: it never
    influences where data is written.
    """
    if not client_filename:
        return None
    name = PurePath(str(client_filename).replace("\\", "/")).name.strip()
    # Defence in depth: even after taking the basename, drop control characters
    # that could corrupt logs or terminal output downstream.
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if not name or name in {".", ".."}:
        return None
    if len(name) > _MAX_DISPLAY_NAME:
        name = name[:_MAX_DISPLAY_NAME]
    return name


@dataclass(frozen=True)
class StoredUpload:
    """A validated upload written to a server-generated temporary path."""

    path: str
    size: int
    display_filename: Optional[str]
    declared_format: Optional[SourceFormat]
    detected_format: SourceFormat


def store_upload(
    content: bytes,
    *,
    client_filename: Optional[str] = None,
    declared_format: Optional[str] = None,
    destination: Optional[str] = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> StoredUpload:
    """Validate and persist an upload under a server-generated filename.

    :raises UploadError: if the upload is empty, oversized, or its declared
        format contradicts its content.
    """
    if not isinstance(content, (bytes, bytearray)):
        raise UploadError("upload content must be bytes", code="INVALID_CONTENT_TYPE")
    content = bytes(content)
    if not content:
        raise UploadError("uploaded file is empty", code="EMPTY_UPLOAD")
    if len(content) > max_bytes:
        raise UploadError(
            "uploaded file is %d bytes, which exceeds the %d byte limit"
            % (len(content), max_bytes),
            code="UPLOAD_TOO_LARGE",
        )

    detected = detect_format(content)
    declared = coerce_declared_format(declared_format)
    # A contradicting declaration is a client bug or an attempt to confuse the
    # parser; either way the content decides, and the mismatch is reported.
    try:
        raise_for_mismatched_format(detected, declared)
    except IngestionError as exc:
        raise UploadError(str(exc), code=exc.code) from exc

    handle, path = tempfile.mkstemp(
        prefix=STAGED_UPLOAD_PREFIX, suffix=".dat", dir=destination
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
    except OSError:
        discard_upload(path)
        raise

    return StoredUpload(
        path=path,
        size=len(content),
        display_filename=sanitize_display_filename(client_filename),
        declared_format=declared,
        detected_format=detected,
    )


def discard_upload(path: Optional[str]) -> None:
    """Remove a temporary upload, ignoring an already-absent file."""
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError:  # pragma: no cover - permissions/race on cleanup
        return
