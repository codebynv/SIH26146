"""BitTrace AI unified domain model (SIH26146 - Step 3).

Layering note: this package depends on nothing in :mod:`backend.app.ingestion`.
The adapter accepts a structural protocol rather than importing the ingestion
record type, so the domain model stays independent of file formats and
transport.  ``backend.app.ingestion`` may import *from* this package.
"""

from backend.app.domain.adapters import (
    classify_observation_type,
    normalized_to_unified,
)
from backend.app.domain.enums import IssueSeverity, ObservationType, SourceFormat
from backend.app.domain.models import (
    BlockchainTransaction,
    NetworkObservation,
    Provenance,
    UnifiedObservation,
    ValidationIssue,
    derive_observation_id,
)

__all__ = [
    "BlockchainTransaction",
    "IssueSeverity",
    "NetworkObservation",
    "ObservationType",
    "Provenance",
    "SourceFormat",
    "UnifiedObservation",
    "ValidationIssue",
    "classify_observation_type",
    "derive_observation_id",
    "normalized_to_unified",
]
