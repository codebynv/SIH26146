"""Enumerations for the BitTrace AI evidence model (SIH26146 - Step 4A).

Scope
-----
This module is the *vocabulary* of the evidence semantic foundation and
nothing more.  There is no engine here: no rules fire, nothing is matched,
nothing is indexed, and no graph is built.  Those belong to Step 4B and later.

Two axes, deliberately kept apart
---------------------------------
:class:`EvidenceType` and :class:`InferenceLevel` are separate enumerations
because they answer different questions and must never be collapsed into one:

``EvidenceType``
    *What kind of relationship is this?*
    "These two observations share an IP address."

``InferenceLevel``
    *How should this relationship be characterised?*
    "That link came from a deterministic rule, so it is worth recording -- but
    it is still not proof of real-world identity."

Conflating them would force every future rule into one flat list and make it
impossible to say, for example, that a ``GEO_COINCIDENCE`` found by a
deterministic rule is merely ``CORRELATED`` while a guess about the same pair
is ``REQUIRES_VERIFICATION``.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "InferenceLevel",
    "EvidenceType",
    "ConfidenceBasis",
    "TemporalRelation",
]


class InferenceLevel(str, Enum):
    """How strongly a relationship may be characterised.

    This is an ordered spectrum of *evidential strength*, not a boolean.  A
    boolean would collapse "observed", "rule-derived" and "needs a human" into a
    single yes/no and lose exactly the distinction an investigator needs.
    """

    OBSERVED = "OBSERVED"
    """Directly present in the source or the unified observation.

    Nothing was inferred: the value appears as given.  An ``OBSERVED``
    relationship asserts no link between two records beyond the presence of
    the value itself.
    """

    CORRELATED = "CORRELATED"
    """Established by an explicit, deterministic correlation rule.

    Still **not proof of real-world identity**.  "CORRELATED" means a named,
    versioned rule found a match and recorded why.  It does not mean the two
    observations belong to the same person, device, wallet or entity.
    """

    INFERRED = "INFERRED"
    """An analytical inference drawn from one or more evidence signals.

    Must carry a rationale explaining the reasoning.  An inference is a
    hypothesis produced by the system, never a finding.
    """

    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"
    """A potentially useful lead that is not yet strong enough to classify.

    This state is a permanent part of the model, not a temporary or
    transitional one.  It tells the investigator plainly that the link is
    suggestive and that independent verification is required before the
    relationship is relied upon.
    """

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class EvidenceType(str, Enum):
    """What kind of relationship or evidence a link represents.

    Every value names an *observed match between supplied values*.  None of
    them asserts ownership, control or identity, and none may be read as
    doing so:

    * ``SAME_IP`` means two observations share an IP address as recorded.  It
      does **not** mean one person or entity used both, and it does not mean
      the address is controlled by anyone in particular.
    * ``SAME_TXID`` means two observations reference the same transaction.  It
      does **not** mean any IP address belongs to that transaction's wallet.
    * ``GEO_COINCIDENCE`` means two observations carry the same country value.
      It does **not** mean the two subjects were in the same physical place,
      and country-level data says nothing about identity.
    """

    SAME_SOURCE_RECORD = "SAME_SOURCE_RECORD"
    """Both sides came from one and the same source record."""

    SAME_TXID = "SAME_TXID"
    """Both sides reference the same transaction identifier."""

    SAME_ADDRESS = "SAME_ADDRESS"
    """Both sides reference the same blockchain address."""

    SAME_IP = "SAME_IP"
    """Both sides carry the same IP address, exactly as recorded."""

    TEMPORAL_ADJACENCY = "TEMPORAL_ADJACENCY"
    """Both sides fall within a stated temporal window."""

    AMOUNT_MATCH = "AMOUNT_MATCH"
    """Both sides carry a matching amount value."""

    PROVENANCE_LINK = "PROVENANCE_LINK"
    """Both sides trace back to a shared ingestion run or source file."""

    GEO_COINCIDENCE = "GEO_COINCIDENCE"
    """Both sides carry the same country value.  Not a location or identity claim."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class ConfidenceBasis(str, Enum):
    """Documented origin of a :class:`~backend.app.correlation.models.Confidence`.

    Declaring the vocabulary now means a confidence figure always carries an
    explanation of where it came from, instead of appearing as a bare float
    that nobody can justify later.

    There is deliberately **no** machine-learned basis in this list.  No
    statistical or ML model is built in Step 4A, and inventing the category
    before a policy exists would invite an unexplained number into the
    evidence model.  A deterministic rule may later publish a documented
    confidence policy under one of the bases already here.
    """

    EXACT_MATCH = "EXACT_MATCH"
    """The values are equal by direct comparison; the match is certain as a
    *comparison*, and no probability is being claimed about the real world."""

    DETERMINISTIC_RULE = "DETERMINISTIC_RULE"
    """A named, versioned rule assigned this value under a documented policy."""

    HUMAN_ASSESSMENT = "HUMAN_ASSESSMENT"
    """Supplied by an analyst.  Always attributable to a person, never to the
    system."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class TemporalRelation(str, Enum):
    """How one temporal window sits relative to another.

    These are the four relations this model recognises.  Any further ordering
    (or a tolerance-based notion of "adjacency") is deliberately absent: a
    temporal *engine*, index or matching pass belongs to a later step and must
    not be smuggled in here.
    """

    BEFORE = "BEFORE"
    """The window ends strictly before the other window begins."""

    AFTER = "AFTER"
    """The window begins strictly after the other window ends."""

    WITHIN = "WITHIN"
    """The window falls entirely inside the other window."""

    OVERLAPS = "OVERLAPS"
    """The windows share at least one instant without either containing the
    other."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value
