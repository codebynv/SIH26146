"""Deterministic correlation rules (SIH26146 - Step 4B).

What this module is
-------------------
The seven named, versioned rules that decide when two or more observations
share a stated property, plus the normalisation each rule relies on and the
:class:`CorrelationConfig` that parameterises the two rules which need one.

A rule here answers exactly one question -- *do these observations match on
this field, under this criterion?* -- and reports what it matched so the
engine can record it.  Nothing here decides what a match *means* for the
people or systems involved.

What this module deliberately cannot do
---------------------------------------
No rule in this file asserts that two observations belong to the same person,
device, wallet, operator or organisation.  ``same_ip`` records that two
records carry the same source IP; it does not record that one person used both.
``geo_coincidence`` records that two records carry the same country value; it
does not place anyone anywhere.  Every :class:`MatchGroup` a rule produces is
a statement about supplied values and about nothing else, and the wording used
to describe one is fixed by the rule so it cannot drift into a claim.

There is no fuzzy matching, no similarity score, no learned model, no
behavioural profile and no ranking in this module.  A rule either matches
under its stated criterion or it does not.

The substrate boundary
----------------------
Steps 1-3 own the observations, and this package deliberately does not import
them.  Step 4A locked that boundary with a test that rejects any import of
:mod:`backend.app.domain` or :mod:`backend.app.ingestion` from anywhere in this
package, so the engine reads observations *structurally*: through the
:class:`SupportsObservation` protocol below, which names the attributes a rule
touches without importing the class that provides them.  A real
:class:`~backend.app.domain.models.UnifiedObservation` satisfies that protocol,
and so does any test double, so rules stay unit-testable without a Step 3
fixture and without a dependency cycle between the layers.

The rules only ever *read* an observation.  Nothing here assigns to a field, so
the mutable ``extra`` mapping carried by a frozen Step 3 model cannot be
disturbed by a correlation pass.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import (
    Any,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Sequence,
)

from backend.app.correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    TemporalRelation,
)
from backend.app.correlation.models import Confidence, TemporalWindow

__all__ = [
    "BECH32_PREFIXES",
    "RULE_VERSION",
    "TEMPORAL_EVIDENCE_CONFIDENCE",
    "TOLERANT_AMOUNT_EVIDENCE_CONFIDENCE",
    "MAX_EMBEDDED_VALUE_LENGTH",
    "SupportsObservation",
    "CorrelationConfig",
    "MatchGroup",
    "CorrelationRule",
    "SameTxidRule",
    "SameAddressRule",
    "SameIpRule",
    "TemporalAdjacencyRule",
    "AmountMatchRule",
    "ProvenanceLinkRule",
    "GeoCoinidenceRule",
    "build_default_rules",
    "describe_window",
    "join_facts",
    "normalize_address",
    "normalize_country",
    "normalize_ip",
    "normalize_txid",
]

#: Version stamped on every relationship these rules produce.  It is a constant
#: rather than a per-rule value so that a relationship recorded before a rule
#: changes can always be told apart from one recorded after.
RULE_VERSION = "1.0.0"

#: Human-readable prefixes of the bech32 address encodings (BIP-173).  Bech32 is
#: case-insensitive by specification, so an address written ``BC1Q...`` and one
#: written ``bc1q...`` are the same address and must compare equal.
BECH32_PREFIXES = ("bc1", "tb1", "bcrt1")

#: Evidence strength assigned by the temporal rule.  This is a *documented
#: policy constant for the evidence*, not a probability that two observations
#: are related: proximity in time is a real, reproducible comparison, and the
#: value records that this rule is certain its own criterion held.  It says
#: nothing about why the two events were close together.
TEMPORAL_EVIDENCE_CONFIDENCE = 0.9

#: Evidence strength assigned when an amount matched only because it fell
#: inside the configured tolerance.  Lower than an exact equality, because the
#: criterion is a bound rather than an identity, and higher than nothing at all,
#: because the bound is explicit and reproducible.
TOLERANT_AMOUNT_EVIDENCE_CONFIDENCE = 0.9

#: Longest value echoed into a rationale.  A rationale is an explanation, not a
#: data dump; an unbounded field would let a source record dictate the text an
#: analyst reads, and could even smuggle a newline and a second sentence into
#: it.
MAX_EMBEDDED_VALUE_LENGTH = 120

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


# --------------------------------------------------------------------------- #
# The structural substrate
# --------------------------------------------------------------------------- #


class SupportsObservation(Protocol):
    """The attributes a rule is allowed to read from an observation.

    Declared structurally on purpose.  Step 4A forbids this package from
    importing :mod:`backend.app.domain`, so the concrete
    :class:`~backend.app.domain.models.UnifiedObservation` is named in the
    documentation but never imported, and the sub-objects stay typed as
    :class:`Any` -- the rules use ``getattr`` against these names and treat a
    missing one as "this observation has no such layer".

    ``extra`` is deliberately absent.  No rule reads it, which is what makes
    "a correlation pass never mutates an observation" true even though that
    mapping is mutable inside an otherwise frozen model.
    """

    observation_id: str
    timestamp: Optional[datetime]
    network: Optional[Any]
    blockchain: Optional[Any]
    provenance: Any


def _layer(observation: SupportsObservation, name: str) -> Optional[Any]:
    """Return observation layer *name*, or ``None`` when it is absent."""
    value = getattr(observation, name, None)
    return value if value is not None else None


def _is_aware(moment: Any) -> bool:
    """True when *moment* is a datetime that genuinely carries a timezone.

    Mirrors the Step 3 test for a usable offset, so a datetime whose ``tzinfo``
    is set but resolves to ``None`` is treated as naive rather than trusted.
    """
    if not isinstance(moment, datetime):
        return False
    if moment.tzinfo is None:
        return False
    return moment.tzinfo.utcoffset(moment) is not None


def join_facts(facts: Sequence[str]) -> str:
    """Join *facts* into a readable clause list.

    Sorted first, then joined, so a rationale reads the same way regardless of
    the order the matching keys happened to be discovered in.  Without the
    sort, shuffling the input observations would reorder the sentences of an
    otherwise identical relationship.
    """
    ordered = sorted(dict.fromkeys(fact for fact in facts if fact))
    if not ordered:
        return "matched values"
    if len(ordered) == 1:
        return ordered[0]
    return ", ".join(ordered[:-1]) + " and " + ordered[-1]


def _safe_value(value: Any) -> str:
    """Render *value* for inclusion in a rationale.

    Strips control characters -- a newline in a source-supplied field would
    otherwise let that field append its own sentence to the explanation -- and
    truncates.  The value itself is never rewritten beyond that: mangling an
    address or a transaction id to tidy it would make the rationale describe
    something other than what was actually matched.
    """
    text = _CONTROL_RE.sub(" ", str(value)).strip()
    if len(text) > MAX_EMBEDDED_VALUE_LENGTH:
        text = text[: MAX_EMBEDDED_VALUE_LENGTH - 3] + "..."
    return text


def describe_window(window: timedelta) -> str:
    """Render a :class:`~datetime.timedelta` in whole seconds when possible.

    Used only in rationale text, where ``0:00:05`` reads worse than ``5
    seconds``.  The rounding is presentation only: acceptance is always tested
    against the exact timedelta, never against this string.
    """
    total = window.total_seconds()
    if float(total).is_integer():
        return "%d seconds" % int(total)
    return "%s seconds" % (total,)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalize_address(value: Any) -> Optional[str]:
    """Normalise a Bitcoin address for equality comparison.

    Bech32 addresses are case-insensitive, so a recognised bech32 prefix
    lowercases the whole address.  Base58 is a case-*sensitive* encoding: in
    Base58 the case of each character is part of the value, so ``1BvBM...``
    and ``1bvbM...`` are different strings and folding them would manufacture
    a match between two unrelated addresses.

    Blank or non-textual input yields ``None`` so a caller can treat "no
    address" as absent rather than as a value.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith(BECH32_PREFIXES):
        return lowered
    return text


def normalize_txid(value: Any) -> Optional[str]:
    """Normalise a transaction id for equality comparison.

    A transaction id is hexadecimal, where letter case carries no meaning, so
    the value is stripped and lowercased.  This is a *representation*
    normalisation only; it asserts nothing about the transaction itself.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def normalize_country(value: Any) -> Optional[str]:
    """Normalise a country value for equality comparison.

    Step 3 preserves ``geo_country`` exactly as the source supplied it, because
    discarding or rewriting source evidence is not this system's job.  Equality
    comparison is a different question from preservation, so the *key* is
    stripped and uppercased while the stored observation is left untouched.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text or None


def normalize_ip(value: Any) -> Optional[str]:
    """Render an IP address in its canonical comparable form.

    Step 3 already validates these as ``IPvAnyAddress``, so ``str()`` is the
    canonical rendering and comparing strings is safe.  A plain string is
    accepted too, so a rule can be unit-tested without a Step 3 model.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CorrelationConfig:
    """Everything the rules need that a caller must decide.

    Two of the seven rules cannot have a sensible default, and a default would
    be the wrong kind of decision to bury:

    * :attr:`temporal_window` has **no** default.  "Close together" is not a
      property of the data, it is a question the analyst has to answer, so an
      unset window means the temporal rule is switched off rather than silently
      matching everything.
    * :attr:`amount_tolerance` defaults to exact equality, because a tolerance
      of zero cannot invent a match that the data does not contain.

    :attr:`now` is injectable so a test can pin it.  It is recorded on each
    relationship as ``created_at`` and is deliberately **excluded** from the
    derived relationship identity, so re-running a correlation tomorrow
    produces the same relationship ids.
    """

    temporal_window: Optional[timedelta] = None
    amount_tolerance: Decimal = Decimal("0")
    now: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def __post_init__(self) -> None:
        window = self.temporal_window
        if window is not None:
            if not isinstance(window, timedelta):
                raise ValueError(
                    "temporal_window must be a timedelta or None, not %r"
                    % type(window).__name__
                )
            if window <= timedelta(0):
                raise ValueError(
                    "temporal_window must be strictly positive, got %r; a "
                    "non-positive window would either match only simultaneous "
                    "events or match everything" % (window,)
                )

        tolerance = self.amount_tolerance
        try:
            coerced = tolerance if isinstance(tolerance, Decimal) else Decimal(str(tolerance))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(
                "amount_tolerance must be a decimal number, got %r" % (tolerance,)
            ) from exc
        if coerced < 0:
            raise ValueError(
                "amount_tolerance must not be negative, got %s" % (coerced,)
            )
        object.__setattr__(self, "amount_tolerance", coerced)

        if not _is_aware(self.now):
            raise ValueError(
                "CorrelationConfig.now must be timezone-aware; a naive value "
                "would be interpreted against the host's local zone"
            )


# --------------------------------------------------------------------------- #
# What a rule reports
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MatchGroup:
    """One match a rule found, ready for the engine to group.

    :attr:`observation_ids` are the observations sharing the matched key, and
    :attr:`facts` describe *what* matched in neutral terms.  The facts are data,
    not sentences: the rule's :meth:`~CorrelationRule.describe` turns them into
    the final rationale, so the wording stays under the rule's control and a
    rule can never narrate its own match into a claim.
    """

    observation_ids: tuple[str, ...]
    facts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_ids:
            raise ValueError("a MatchGroup must name at least one observation")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError(
                "MatchGroup must not repeat an observation identifier: %r"
                % (self.observation_ids,)
            )


class CorrelationRule:
    """Base class for the seven deterministic rules.

    A subclass declares its :attr:`rule_id`, :attr:`evidence_type` and
    :meth:`evaluate`, and inherits the shared vocabulary.  Instances are cheap
    and stateless apart from their configuration, so one rule object may be
    reused across many passes.
    """

    #: Stable identifier recorded on every relationship this rule produces.
    rule_id: str = ""

    #: Version stamped on every relationship this rule produces.  Recorded so
    #: a result can be reproduced exactly, or deliberately not reproduced.
    rule_version: str = RULE_VERSION

    #: Which kind of match this rule looks for.  Answers "what matched?".
    evidence_type: EvidenceType = EvidenceType.SAME_SOURCE_RECORD

    #: Whether this rule reports a shared-key group or one pair at a time.
    grouped: bool = True

    @property
    def confidence(self) -> Confidence:
        """Evidence strength this rule reports.

        Certainty is always about the *comparison*, never about the world.  A
        rule that finds two identical transaction ids is certain the strings
        are equal; it has no opinion whatsoever on whether the transactions
        were related.
        """
        return Confidence.exact_match()

    def evaluate(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        """Yield every match this rule finds.  Must not mutate *observations*."""
        raise NotImplementedError

    def describe(self, facts: Sequence[str]) -> str:
        """Turn *facts* into the rationale stored on the relationship.

        Every override is a fixed sentence describing a comparison.  None of
        them names an owner, a person, an operator or a motive, because a
        deterministic field match is not evidence of any of those.
        """
        return "Observations share the same %s." % (join_facts(facts),)

    def _group_by_key(
        self,
        observations: Sequence[SupportsObservation],
        keys_of: "Any",
    ) -> Iterator[MatchGroup]:
        """Shared-key grouping helper.

        Builds ``key -> observation ids`` for every non-``None`` key and yields
        one :class:`MatchGroup` per key shared by two or more observations.

        Grouping rather than pairing is a deliberate, approved choice.  For a
        shared-key rule, a group of *n* observations produces **one**
        relationship: the lowest ``observation_id`` becomes the source anchor
        and the remaining *n-1* observations become targets.  That keeps the
        output linear in the number of distinct keys instead of quadratic in the
        size of a group, and it means a group of fifty records yields one
        relationship rather than 1 225 pairs.  The source and target sets are
        therefore disjoint by construction, and a relationship can never
        reference an observation against itself.
        """
        index: dict[Any, list[str]] = {}
        for observation in observations:
            for key in keys_of(observation):
                if key is None:
                    continue
                index.setdefault(key, []).append(observation.observation_id)
        for key in sorted(index, key=repr):
            members = sorted(dict.fromkeys(index[key]))
            if len(members) < 2:
                continue
            yield MatchGroup(observation_ids=tuple(members), facts=self._facts(key))


# --------------------------------------------------------------------------- #
# 1. same_txid
# --------------------------------------------------------------------------- #


class SameTxidRule(CorrelationRule):
    """Match observations that carry the same transaction id.

    A transaction id is a *reference*, not an observation identity: one
    transaction can legitimately appear in many records, and that is exactly
    the repetition this rule surfaces.  Sharing a transaction id says the two
    records point at the same transaction.  It does not say who sent it, who
    received it, or that either record's other fields belong together.
    """

    rule_id = "same_txid"
    evidence_type = EvidenceType.SAME_TXID

    def evaluate(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        def keys(observation: SupportsObservation) -> Iterator[Optional[str]]:
            blockchain = _layer(observation, "blockchain")
            if blockchain is None:
                return
            yield normalize_txid(getattr(blockchain, "txid", None))

        return self._group_by_key(observations, keys)

    def _facts(self, key: Any) -> tuple[str, ...]:
        return ("transaction id %s" % _safe_value(key),)

    def describe(self, facts: Sequence[str]) -> str:
        return "Observations reference the same %s." % (join_facts(facts),)


# --------------------------------------------------------------------------- #
# 2. same_address
# --------------------------------------------------------------------------- #


class SameAddressRule(CorrelationRule):
    """Match observations that reference the same blockchain address.

    Input and output addresses are pooled into one index, so an address that
    appears as an output in one record and an input in another is a match.
    That is a genuinely useful correlation -- a value moving from one record's
    output side to another's input side is a repetition in the evidence -- and
    it is still only a repetition.

    It is emphatically not a statement that one party controls the address.  A
    shared address is the *expected* condition for ordinary transaction
    traffic, and reading it as ownership would be both wrong and, on-chain,
    trivially falsified.
    """

    rule_id = "same_address"
    evidence_type = EvidenceType.SAME_ADDRESS

    def evaluate(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        def keys(observation: SupportsObservation) -> Iterator[Optional[str]]:
            blockchain = _layer(observation, "blockchain")
            if blockchain is None:
                return
            for address in getattr(blockchain, "input_addresses", ()) or ():
                yield normalize_address(address)
            for address in getattr(blockchain, "output_addresses", ()) or ():
                yield normalize_address(address)

        return self._group_by_key(observations, keys)

    def _facts(self, key: Any) -> tuple[str, ...]:
        return ("normalized blockchain address %s" % _safe_value(key),)

    def describe(self, facts: Sequence[str]) -> str:
        return (
            "Observations reference the same %s. A shared address is a "
            "repetition in the evidence and is not a statement of control "
            "over it." % (join_facts(facts),)
        )


# --------------------------------------------------------------------------- #
# 3. same_ip
# --------------------------------------------------------------------------- #


class SameIpRule(CorrelationRule):
    """Match observations that carry the same IP address *in the same position*.

    Only same-position matches are considered: a source address is compared
    with a source address, a destination address with a destination address.

    A source-to-destination comparison is deliberately excluded.  A record
    where ``192.0.2.1`` is the source and another where ``192.0.2.1`` is the
    destination describe *opposite* ends of a connection, and treating that as
    a shared endpoint would manufacture a relationship out of the single most
    common address in any traffic sample.  Role is therefore part of the key
    rather than something checked afterwards.

    Two records carrying the same IP is a fact about the records.  It is not a
    fact about a person: an address is shared by a NAT gateway, a VPN exit, a
    botnet host and an office router, and the data cannot tell those apart.
    """

    rule_id = "same_ip"
    evidence_type = EvidenceType.SAME_IP

    #: ``(key component, field name, human label)`` for each address position.
    #: The field name is kept explicit so the key component can stay short and
    #: stable while still naming the exact field that was read.
    _ROLES = (
        ("src", "src_ip", "source"),
        ("dst", "dst_ip", "destination"),
    )

    def evaluate(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        def keys(observation: SupportsObservation) -> Iterator[Optional[Any]]:
            network = _layer(observation, "network")
            if network is None:
                return
            for role, attribute, _ in self._ROLES:
                address = normalize_ip(getattr(network, attribute, None))
                if address is not None:
                    yield (role, address)

        return self._group_by_key(observations, keys)

    def _facts(self, key: Any) -> tuple[str, ...]:
        role, address = key
        label = {entry[0]: entry[2] for entry in self._ROLES}[role]
        return ("%s IP address %s" % (label, _safe_value(address)),)

    def describe(self, facts: Sequence[str]) -> str:
        return (
            "Observations carry the same %s, in the same field position. "
            "A shared IP address is a property of the records, not evidence "
            "that one party was behind both." % (join_facts(facts),)
        )


# --------------------------------------------------------------------------- #
# 4. temporal_adjacency
# --------------------------------------------------------------------------- #


class TemporalAdjacencyRule(CorrelationRule):
    """Match observations that fall within the configured window of each other.

    This rule is pairwise rather than grouped: there is no shared key to group
    on, only a distance.  Each qualifying pair is reported once.

    Naive timestamps are skipped, not localised
    ------------------------------------------
    An observation whose timestamp carries no timezone cannot be compared to
    one that does, because doing so requires assuming an offset that the source
    never supplied.  Step 3 already refuses to invent one and flags the value;
    this rule inherits that decision and simply declines to participate.  A
    mixed batch therefore correlates the timezone-aware records and leaves the
    rest alone, rather than either crashing on the subtraction or shifting
    every result by the host's offset.

    The rule uses bucketing to keep the search near-linear, then applies the
    real distance test to every candidate.  Bucketing is a candidate generator
    only: no pair is ever accepted because it landed in a neighbouring bucket.
    """

    rule_id = "temporal_adjacency"
    evidence_type = EvidenceType.TEMPORAL_ADJACENCY
    grouped = False

    def __init__(self, temporal_window: Optional[timedelta] = None) -> None:
        if temporal_window is None:
            raise ValueError(
                "TemporalAdjacencyRule requires an explicit temporal_window; "
                "there is no defensible default for how close is close"
            )
        if not isinstance(temporal_window, timedelta):
            raise ValueError(
                "temporal_window must be a timedelta, not %r"
                % type(temporal_window).__name__
            )
        if temporal_window <= timedelta(0):
            raise ValueError(
                "temporal_window must be strictly positive, got %r"
                % (temporal_window,)
            )
        self._window = temporal_window

    @property
    def temporal_window(self) -> timedelta:
        return self._window

    @property
    def confidence(self) -> Confidence:
        return Confidence(
            value=TEMPORAL_EVIDENCE_CONFIDENCE,
            basis=ConfidenceBasis.DETERMINISTIC_RULE,
        )

    def evaluate(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        aware: list[tuple[datetime, str]] = []
        for observation in observations:
            moment = getattr(observation, "timestamp", None)
            if _is_aware(moment):
                aware.append((moment, observation.observation_id))
        if len(aware) < 2:
            return iter(())

        aware.sort(key=lambda item: (item[0], item[1]))
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        window = self._window
        buckets: dict[int, list[tuple[datetime, str]]] = {}
        for moment, observation_id in aware:
            index = int((moment - epoch) // window)
            buckets.setdefault(index, []).append((moment, observation_id))

        found: set[tuple[str, str]] = set()
        for moment, observation_id in aware:
            index = int((moment - epoch) // window)
            for neighbour in (index - 1, index, index + 1):
                for other_moment, other_id in buckets.get(neighbour, ()):
                    if other_id <= observation_id:
                        continue
                    if abs(other_moment - moment) <= window:
                        found.add((observation_id, other_id))

        for source_id, target_id in sorted(found):
            yield MatchGroup(
                observation_ids=(source_id, target_id),
                facts=(describe_window(window),),
            )

    def describe(self, facts: Sequence[str]) -> str:
        return (
            "Observations occur within the configured temporal window of %s. "
            "Proximity in time is an observable coincidence and is not on its "
            "own an explanation for it." % (join_facts(facts),)
        )


# --------------------------------------------------------------------------- #
# 5. amount_match
# --------------------------------------------------------------------------- #


class AmountMatchRule(CorrelationRule):
    """Match observations carrying a comparable input or output amount.

    ``fee`` is excluded, and not as an oversight.  Step 3 established that a
    ``fee`` column is frequently incidental -- plenty of non-blockchain exports
    carry one -- so treating two records' fees as a shared value would link
    records on a field that need not mean the same thing in both.  Only the
    input and output amount lists participate.

    Amounts are :class:`~decimal.Decimal` throughout.  Binary floating point
    cannot represent most Bitcoin amounts exactly, so a float comparison would
    both invent and miss matches at the eighth decimal place, and the resulting
    relationship would not be reproducible.

    With a zero tolerance the rule groups on exact value equality.  With a
    positive tolerance there is no longer a shared value to group on, so the
    rule reports pairs within the bound -- and it uses an ordered index so the
    search only visits amounts that are actually in range.
    """

    rule_id = "amount_match"
    evidence_type = EvidenceType.AMOUNT_MATCH

    def __init__(self, amount_tolerance: Decimal = Decimal("0")) -> None:
        try:
            tolerance = (
                amount_tolerance
                if isinstance(amount_tolerance, Decimal)
                else Decimal(str(amount_tolerance))
            )
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(
                "amount_tolerance must be a decimal number, got %r"
                % (amount_tolerance,)
            ) from exc
        if tolerance < 0:
            raise ValueError("amount_tolerance must not be negative, got %s" % (tolerance,))
        self._tolerance = tolerance

    @property
    def amount_tolerance(self) -> Decimal:
        return self._tolerance

    @property
    def confidence(self) -> Confidence:
        if self._tolerance == 0:
            return Confidence.exact_match()
        return Confidence(
            value=TOLERANT_AMOUNT_EVIDENCE_CONFIDENCE,
            basis=ConfidenceBasis.DETERMINISTIC_RULE,
        )

    def evaluate(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        if self._tolerance == 0:
            return self._evaluate_exact(observations)
        return self._evaluate_tolerant(observations)

    def _evaluate_exact(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        index: dict[Decimal, dict[str, set[str]]] = {}
        for observation in observations:
            for kind, value in self._amounts(observation):
                # ``is None`` rather than truthiness: Decimal("0") is a real,
                # matchable amount and must not be discarded.
                if value is None:
                    continue
                index.setdefault(value, {}).setdefault(
                    observation.observation_id, set()
                ).add(kind)
        for value in sorted(index):
            members = index[value]
            if len(members) < 2:
                continue
            kinds: set[str] = set()
            for observation_kinds in members.values():
                kinds |= observation_kinds
            label = ", ".join(sorted(kinds))
            yield MatchGroup(
                observation_ids=tuple(sorted(members)),
                facts=("amount %s (%s)" % (_safe_value(value), label),),
            )

    def _evaluate_tolerant(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        entries: list[tuple[Decimal, str]] = []
        for observation in observations:
            for _kind, value in self._amounts(observation):
                if value is not None:
                    entries.append((value, observation.observation_id))
        if len(entries) < 2:
            return

        entries.sort(key=lambda item: (item[0], item[1]))
        values = [value for value, _ in entries]
        tolerance = self._tolerance
        found: set[tuple[str, str]] = set()
        for value, observation_id in entries:
            low = bisect.bisect_left(values, value - tolerance)
            high = bisect.bisect_right(values, value + tolerance)
            for other_value, other_id in entries[low:high]:
                if other_id <= observation_id:
                    continue
                if abs(other_value - value) <= tolerance:
                    found.add((observation_id, other_id))

        for source_id, target_id in sorted(found):
            yield MatchGroup(
                observation_ids=(source_id, target_id),
                facts=(
                    "input/output amounts within the configured tolerance of %s"
                    % _safe_value(tolerance),
                ),
            )

    @staticmethod
    def _amounts(
        observation: SupportsObservation,
    ) -> Iterator[tuple[str, Decimal]]:
        """Yield ``(kind, amount)`` for input and output amounts only."""
        blockchain = _layer(observation, "blockchain")
        if blockchain is None:
            return
        for kind, attribute in (("input", "input_amounts"), ("output", "output_amounts")):
            for amount in getattr(blockchain, attribute, ()) or ():
                yield kind, amount

    def describe(self, facts: Sequence[str]) -> str:
        return (
            "Observations carry a matching %s. A repeated amount is a "
            "coincidence in the data and carries no financial conclusion."
            % (join_facts(facts),)
        )


# --------------------------------------------------------------------------- #
# 6. provenance_link
# --------------------------------------------------------------------------- #


class ProvenanceLinkRule(CorrelationRule):
    """Match observations that came from the very same source record.

    All three locator fields -- ``ingestion_id``, ``source_file`` and
    ``source_record`` -- must be present and identical.  Requiring the record
    locator matters: two rows from one file are two records, and linking them
    because they share a filename would assert that a file-level coincidence
    is a record-level fact.  A partial triple does not match at all.
    """

    rule_id = "provenance_link"
    evidence_type = EvidenceType.PROVENANCE_LINK

    def evaluate(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        def keys(observation: SupportsObservation) -> Iterator[Optional[Any]]:
            provenance = _layer(observation, "provenance")
            if provenance is None:
                return
            ingestion_id = getattr(provenance, "ingestion_id", None)
            source_file = getattr(provenance, "source_file", None)
            source_record = getattr(provenance, "source_record", None)
            if not _present(ingestion_id) or not _present(source_file):
                return
            if not _present(source_record):
                return
            yield (
                str(ingestion_id).strip(),
                str(source_file).strip(),
                str(source_record).strip(),
            )

        return self._group_by_key(observations, keys)

    def _facts(self, key: Any) -> tuple[str, ...]:
        ingestion_id, source_file, source_record = key
        return (
            "source record %s / %s / %s"
            % (
                _safe_value(ingestion_id),
                _safe_value(source_file),
                _safe_value(source_record),
            ),
        )

    def describe(self, facts: Sequence[str]) -> str:
        return (
            "Observations resolve to the same %s, so the two links describe "
            "one piece of source evidence." % (join_facts(facts),)
        )


def _present(value: Any) -> bool:
    """True when *value* is a non-blank string usable as a locator."""
    return isinstance(value, str) and bool(value.strip())


# --------------------------------------------------------------------------- #
# 7. geo_coincidence
# --------------------------------------------------------------------------- #


class GeoCoinidenceRule(CorrelationRule):
    """Match observations that carry the same country value.

    Country-level and nothing more.  A shared country code is one of the
    weakest signals available: it is shared by millions of unrelated parties,
    and it says nothing about where anyone was.  The rationale states that
    explicitly so the relationship cannot be read as a location claim.
    """

    rule_id = "geo_coincidence"
    evidence_type = EvidenceType.GEO_COINCIDENCE

    def evaluate(
        self, observations: Sequence[SupportsObservation]
    ) -> Iterator[MatchGroup]:
        def keys(observation: SupportsObservation) -> Iterator[Optional[str]]:
            network = _layer(observation, "network")
            if network is None:
                return
            yield normalize_country(getattr(network, "geo_country", None))

        return self._group_by_key(observations, keys)

    def _facts(self, key: Any) -> tuple[str, ...]:
        return ("country value %s" % _safe_value(key),)

    def describe(self, facts: Sequence[str]) -> str:
        return (
            "Observations carry the same %s, which is country-level "
            "coincidence only and establishes no shared location."
            % (join_facts(facts),)
        )


# --------------------------------------------------------------------------- #
# Assembling the rule set
# --------------------------------------------------------------------------- #


def build_default_rules(config: CorrelationConfig) -> tuple[CorrelationRule, ...]:
    """Return the standard rule set for *config*, in a fixed order.

    The temporal rule is included only when a window is configured, and the
    order is fixed rather than derived from a set or a dict, so two engines
    built from equal configuration always evaluate in the same order.
    """
    rules: list[CorrelationRule] = [
        SameTxidRule(),
        SameAddressRule(),
        SameIpRule(),
        AmountMatchRule(config.amount_tolerance),
        ProvenanceLinkRule(),
        GeoCoinidenceRule(),
    ]
    if config.temporal_window is not None:
        rules.append(TemporalAdjacencyRule(config.temporal_window))
    return tuple(rules)
