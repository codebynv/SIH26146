"""A lightweight reference back to source evidence (SIH26146 - Step 4A).

Why a reference and not a copy
------------------------------
A derived relationship needs to point at the evidence that produced it, so a
reader can walk from a link back to the exact file, row and ingestion run.  It
does **not** need the whole :class:`~backend.app.domain.models.UnifiedObservation`
anywhere near it.

Duplicating full observations inside every relationship would multiply the
evidence substrate by the number of links, invite the two copies to drift
apart, and re-store data Steps 1-3 already own.  So this is a reference: a
small, immutable, serialisable pointer that names the source, and leaves the
authoritative payload where it already lives.

Nothing here is required to be unique or self-verifying.  It is a citation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["ProvenanceRef"]


def _require_non_blank(value: str, field: str) -> str:
    """Reject empty or whitespace-only identifiers.

    A blank locator is worse than none: it looks like a real reference while
    pointing nowhere, which is exactly the kind of untraceable claim this
    model exists to prevent.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty, non-blank string" % field)
    return value


class ProvenanceRef(BaseModel):
    """A citation identifying where a piece of source evidence came from.

    Sufficient to answer: which ingestion run, which source file, and which
    record within it.  Mirrors the corresponding fields of
    :class:`~backend.app.domain.models.Provenance` without inheriting its
    transport and context details, which a relationship has no use for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ingestion_id: str = Field(
        description="Identifier of the ingestion run that produced the evidence."
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Display name of the source, as recorded at ingestion. Never "
        "a filesystem path derived at correlation time.",
    )
    source_record: Optional[str] = Field(
        default=None,
        description="Locator within the source, such as 'row 12' or '$[3]'.",
    )

    @field_validator("ingestion_id")
    @classmethod
    def _check_ingestion_id(cls, value: str) -> str:
        return _require_non_blank(value, "ingestion_id")

    @field_validator("source_file", "source_record")
    @classmethod
    def _check_optional_locator(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be null or a non-blank string, not blank")
        return value

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "%s::%s::%s" % (
            self.ingestion_id,
            self.source_file or "<unknown>",
            self.source_record or "<unknown>",
        )
