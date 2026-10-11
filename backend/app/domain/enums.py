"""Enumerations for the BitTrace AI unified domain model (SIH26146 - Step 3)."""

from __future__ import annotations

from enum import Enum


class ObservationType(str, Enum):
    """Classification of a unified observation.

    The classification describes *what data a record carries*, not what it
    proves.  A :attr:`COMBINED` observation means a single source record
    contained both network and blockchain fields.  It is explicitly **not** a
    claim that any particular IP address is controlled by any particular
    wallet: co-occurrence of an IP and a TXID in one record is not proof of
    identity or attribution.  Cross-layer attribution is out of scope for
    Steps 1-3.
    """

    NETWORK = "NETWORK"
    BLOCKCHAIN = "BLOCKCHAIN"
    COMBINED = "COMBINED"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class SourceFormat(str, Enum):
    """Wire format a record was parsed from."""

    CSV = "csv"
    JSON = "json"
    XML = "xml"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class IssueSeverity(str, Enum):
    """Severity of a validation issue raised while normalising a record."""

    ERROR = "error"
    WARNING = "warning"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


__all__ = ["ObservationType", "SourceFormat", "IssueSeverity"]
