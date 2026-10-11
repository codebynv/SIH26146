"""Ingestion service (SIH26146 - Step 2).

Pipeline: detect format -> parse -> normalise/validate -> deduplicate.

Scope boundary
--------------
This service only *reads, validates and normalises*.  It performs no
attribution, no correlation, no graph building, no risk scoring and no
persistence.  Those are Step 4 and later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from backend.app.domain.enums import IssueSeverity, SourceFormat
from backend.app.domain.models import ValidationIssue
from backend.app.ingestion.dedup import DeduplicationResult, deduplicate
from backend.app.ingestion.models import NormalizedRecord, new_ingestion_id, new_provenance
from backend.app.ingestion.parsers import (
    PARSERS,
    IngestionError,
    RawRecord,
    coerce_declared_format,
    detect_format,
    raise_for_mismatched_format,
)
from backend.app.ingestion.uploads import discard_upload, is_staged_upload

__all__ = ["IngestionResult", "IngestionService"]


@dataclass(frozen=True)
class IngestionResult:
    """Everything one ingestion run produced."""

    ingestion_id: str
    source_format: SourceFormat
    records: list[NormalizedRecord]
    issues: tuple[ValidationIssue, ...] = ()
    raw_record_count: int = 0
    dedup: DeduplicationResult | None = None
    warnings: tuple[str, ...] = field(default=())

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is IssueSeverity.WARNING)

    @property
    def duplicate_count(self) -> int:
        return self.dedup.duplicate_count if self.dedup else 0


class IngestionService:
    """Parses and normalises raw investigative data.

    The service is stateless apart from an optional temporary directory used
    for upload handling, so a single instance can serve every request.
    """

    def __init__(self, *, upload_dir: Optional[str] = None) -> None:
        self._upload_dir = upload_dir

    # ------------------------------------------------------------------
    def ingest_bytes(
        self,
        content: bytes | str,
        *,
        source_file: Optional[str] = None,
        declared_format: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        deduplicate_records: bool = True,
        drop_invalid: bool = False,
    ) -> IngestionResult:
        """Ingest in-memory content.

        The wire format is detected from *content*; *declared_format* is only a
        cross-check and a mismatch is reported as
        :class:`~backend.app.ingestion.parsers.base.IngestionError`.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        detected = detect_format(content)
        declared = coerce_declared_format(declared_format)
        raise_for_mismatched_format(detected, declared)
        return self._ingest_detected(
            content,
            detected,
            source_file=source_file,
            context=context,
            deduplicate_records=deduplicate_records,
            drop_invalid=drop_invalid,
        )

    def ingest_path(
        self,
        path: str,
        *,
        declared_format: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        deduplicate_records: bool = True,
        drop_invalid: bool = False,
        owned: bool = False,
    ) -> IngestionResult:
        """Ingest a file from disk.

        File ownership
        --------------
        This method **never deletes a caller's source file** unless the caller
        explicitly says it owns *path*:

        * ``owned=False`` (the default) reads and ingests the file and leaves
          it exactly where it was.  This is the correct setting for a real
          evidence file on disk -- investigative source data must outlive the
          read.
        * ``owned=True`` declares that *path* is a server-generated temporary
          file, and removes it once ingestion completes, including when
          parsing fails.

        Ownership is never inferred from the filename.  A caller-owned file
        whose name happens to resemble a staging file is still preserved,
        because deleting evidence on a naming coincidence is unrecoverable.

        :param owned: whether this service may delete *path* when finished.
        """
        try:
            with open(path, "rb") as stream:
                content = stream.read()
        except OSError as exc:
            if owned:
                discard_upload(path)
            raise IngestionError(
                "could not read %s: %s" % (os.path.basename(path), exc),
                code="UNREADABLE_FILE",
            ) from exc

        display_name = os.path.basename(path)
        if is_staged_upload(path):
            # Server-generated staging name: expose the client name if we have
            # one instead of leaking the internal path.
            display_name = (context or {}).get("client_filename") or display_name

        try:
            return self.ingest_bytes(
                content,
                source_file=display_name,
                declared_format=declared_format,
                context=context,
                deduplicate_records=deduplicate_records,
                drop_invalid=drop_invalid,
            )
        finally:
            if owned:
                discard_upload(path)

    # ------------------------------------------------------------------
    def _ingest_detected(
        self,
        content: bytes,
        source_format: SourceFormat,
        *,
        source_file: Optional[str],
        context: Optional[Mapping[str, Any]],
        deduplicate_records: bool,
        drop_invalid: bool,
    ) -> IngestionResult:
        ingestion_id = new_ingestion_id()
        parser = PARSERS[source_format.value]

        records: list[NormalizedRecord] = []
        issues: list[ValidationIssue] = []
        warnings: list[str] = []
        raw_count = 0

        for raw in parser(content):
            raw_count += 1
            provenance = new_provenance(
                ingestion_id=ingestion_id,
                source_file=source_file,
                source_format=source_format,
                source_record=raw.locator,
                context=context,
            )
            for message in raw.extra_issues:
                issues.append(
                    ValidationIssue(
                        field=None,
                        code="RAGGED_RECORD",
                        message=message,
                        severity=IssueSeverity.WARNING,
                        source_record=raw.locator,
                    )
                )
            try:
                record = NormalizedRecord.from_raw(raw.data, provenance)
            except ValueError as exc:  # pragma: no cover - only strict_empty raises
                issues.append(
                    ValidationIssue(
                        field=None,
                        code="UNUSABLE_RECORD",
                        message=str(exc),
                        severity=IssueSeverity.ERROR,
                        source_record=raw.locator,
                    )
                )
                continue
            if not record.is_valid and drop_invalid:
                issues.extend(record.errors)
                continue
            records.append(record)
            issues.extend(record.issues)

        dedup_result: Optional[DeduplicationResult] = None
        if deduplicate_records:
            dedup_result = deduplicate(records)
            records = dedup_result.records

        if raw_count == 0:
            warnings.append("no records were found in the input")

        return IngestionResult(
            ingestion_id=ingestion_id,
            source_format=source_format,
            records=records,
            issues=tuple(issues),
            raw_record_count=raw_count,
            dedup=dedup_result,
            warnings=tuple(warnings),
        )
