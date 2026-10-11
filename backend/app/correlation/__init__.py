"""BitTrace AI evidence relationships and deterministic rules (SIH26146).

Step 4A supplied the semantics
-----------------------------
This package began as the *semantic foundation* for evidence-aware
cross-layer correlation.  It defines what a *relationship between
observations* means and what must accompany one to make it explainable and
reproducible:

* an :class:`InferenceLevel` -- how the relationship may be characterised;
* an :class:`EvidenceType` -- what kind of match it is;
* a typed :class:`Confidence` -- how strong the evidence is, and on what basis;
* a :class:`ProvenanceRef` -- which source records it rests on;
* an :class:`EvidenceRelationship` tying those together, carrying a rationale
  and the identity of the rule that produced it.

Step 4B added the rules
-----------------------
Seven deterministic rules in :mod:`backend.app.correlation.rules` decide when
observations match, and :class:`~backend.app.correlation.engine.CorrelationEngine`
applies them and assembles the results.  Both produce relationships at
``CORRELATED``: a named, versioned rule found a reproducible match.  Neither
changes anything Step 4A defined, and no Step 4A model was modified.

What is still not here
----------------------
The engine is **not** a graph and stores nothing.  There is no node or edge
type, no traversal, no path finding, no behavioural profile, no clustering, no
anomaly detection, no temporal flow engine, no investigation paths, no
evidence fusion, no risk scoring, no dashboard, no database and no API
endpoint.  Nothing in Steps 1-3 was changed to make room for any of it.

The boundary that matters
-------------------------
Steps 1-3 own the evidence: :class:`~backend.app.domain.models.UnifiedObservation`
is the immutable substrate, and Step 4 reads it without mutating it.

This package never asserts that an IP address belongs to a wallet, that two
records describe the same person, or that a co-occurrence proves ownership.
A relationship records that a match exists, which rule found it, and how
strong the evidence is.  :meth:`EvidenceRelationship.explains_attribution`
returns ``False`` unconditionally, and is there so the guarantee can be
asserted rather than assumed.

Layering note: this package depends on nothing from
:mod:`backend.app.ingestion` and does not import
:mod:`backend.app.domain` at module level, keeping the derived layer
independent of the substrate it is derived from.  Step 4B reads
``UnifiedObservation`` structurally, through the
:class:`~backend.app.correlation.rules.SupportsObservation` protocol, rather
than by importing it.
"""

from backend.app.correlation.engine import CorrelationEngine
from backend.app.correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    InferenceLevel,
    TemporalRelation,
)
from backend.app.correlation.models import (
    RELATIONSHIP_NAMESPACE,
    Confidence,
    EvidenceRelationship,
    TemporalWindow,
    derive_relationship_id,
)
from backend.app.correlation.provenance import ProvenanceRef
from backend.app.correlation.rules import (
    CorrelationConfig,
    CorrelationRule,
    MatchGroup,
    SameAddressRule,
    SameIpRule,
    SameTxidRule,
    TemporalAdjacencyRule,
    AmountMatchRule,
    ProvenanceLinkRule,
    GeoCoinidenceRule,
    build_default_rules,
)

__all__ = [
    "RELATIONSHIP_NAMESPACE",
    "AmountMatchRule",
    "Confidence",
    "ConfidenceBasis",
    "CorrelationConfig",
    "CorrelationEngine",
    "CorrelationRule",
    "EvidenceRelationship",
    "EvidenceType",
    "GeoCoinidenceRule",
    "InferenceLevel",
    "MatchGroup",
    "ProvenanceLinkRule",
    "ProvenanceRef",
    "SameAddressRule",
    "SameIpRule",
    "SameTxidRule",
    "TemporalAdjacencyRule",
    "TemporalRelation",
    "TemporalWindow",
    "build_default_rules",
    "derive_relationship_id",
]
