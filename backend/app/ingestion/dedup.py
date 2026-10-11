"""Conservative deduplication (SIH26146 - Step 2).

Threat model
------------
Over-eager deduplication in a forensic pipeline is a *data-loss* bug, not a
tidiness win.  Two blockchain transactions can legitimately share an amount,
a fee, a script type, a timestamp second, or even a full set of addresses
without being the same observation.  Two network flows can share every IP and
port and still be distinct events seconds apart.

The rule implemented here is therefore maximally conservative: **two records
are duplicates only when every canonical SIH field, every unmapped source
field, and every reported issue is identical.**  Anything weaker -- same txid,
same IP pair, same fee, or even the same fourteen SIH columns with a different
``block_height`` alongside -- is deliberately *not* treated as a duplicate,
because any such heuristic can merge genuinely distinct observations.

Provenance is excluded from the comparison, so the same observation arriving
in two files still collapses into one record.  That is the only thing dropped
by a merge, and it is the reason a merge is lossless rather than destructive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from backend.app.ingestion.models import NormalizedRecord

__all__ = [
    "record_fingerprint",
    "DeduplicationGroup",
    "DeduplicationResult",
    "deduplicate",
]


def record_fingerprint(record: NormalizedRecord) -> str:
    """Return a stable fingerprint over a record's deduplication identity.

    The fingerprint covers the canonical SIH fields, every unmapped source
    field, and the reported issue codes (see
    :meth:`NormalizedRecord.identity_payload`).  Two records share a
    fingerprint only when nothing observable about them differs, which is
    what makes a merge lossless rather than a silent discard.

    The encoding is sorted-key compact JSON over a deterministic rendering, so
    the same input always produces the same digest across runs and machines.
    """
    payload = record.identity_payload()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DeduplicationGroup:
    """A set of records that collapsed into a single retained record.

    A group only ever forms between records whose *identity payloads* are
    identical: same canonical SIH fields, same unmapped fields, same reported
    issues.  Everything the group discards is therefore redundant, and the
    record itself is not lost -- only a repeat of it is.
    """

    fingerprint: str
    retained_locator: str | None
    merged: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    @property
    def merged_count(self) -> int:
        return len(self.merged)


@dataclass(frozen=True)
class DeduplicationResult:
    """Outcome of a deduplication pass."""

    records: list[NormalizedRecord]
    unique_count: int = 0
    duplicate_count: int = 0
    groups: tuple[DeduplicationGroup, ...] = field(default=())

    @property
    def input_count(self) -> int:
        return self.unique_count + self.duplicate_count


def deduplicate(records: Sequence[NormalizedRecord]) -> DeduplicationResult:
    """Collapse byte-identical canonical records, preserving first-seen order.

    The first occurrence of a fingerprint is retained, keeping its own
    provenance untouched.  Rather than mutating it, the locators and source
    files of every merged record are reported in :attr:`DeduplicationResult.groups`
    so nothing about the original data is lost.
    """
    retained: list[NormalizedRecord] = []
    index: dict[str, int] = {}
    merged: dict[str, list[tuple[str, str | None]]] = {}

    for record in records:
        if not record.is_valid:
            # A record with rejected fields is not a trustworthy candidate for
            # identity: two different malformed records must not be merged.
            retained.append(record)
            continue
        fingerprint = record_fingerprint(record)
        position = index.get(fingerprint)
        if position is None:
            index[fingerprint] = len(retained)
            retained.append(record)
            merged[fingerprint] = []
            continue
        provenance = record.provenance
        merged[fingerprint].append(
            (provenance.source_record or "<unknown>", provenance.source_file)
        )

    groups: list[DeduplicationGroup] = []
    duplicate_count = 0
    for fingerprint, entries in merged.items():
        if not entries:
            continue
        duplicate_count += len(entries)
        kept = retained[index[fingerprint]]
        groups.append(
            DeduplicationGroup(
                fingerprint=fingerprint,
                retained_locator=kept.provenance.source_record,
                merged=tuple(locator for locator, _ in entries),
                sources=tuple(
                    sorted({source for _, source in entries if source is not None})
                ),
            )
        )

    groups.sort(key=lambda group: group.retained_locator or "")
    return DeduplicationResult(
        records=retained,
        unique_count=len(retained),
        duplicate_count=duplicate_count,
        groups=tuple(groups),
    )
