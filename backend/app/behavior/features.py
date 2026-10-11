"""Deterministic behavioral feature extraction (SIH26146 - Step 5).

This module turns observations, and optionally a recorded-evidence graph, into
typed :class:`~backend.app.behavior.models.BehavioralFeature` values.  It
describes what is present in the data and stops there.  It never says what the
presence means, and it contains nothing that could: no score, no threshold, no
verdict, no grouping beyond the one the caller already supplied, and no field
naming a person, organisation or system behind a record.

Four rules shape every decision here
-----------------------------------

**One event time per observation.**  An observation carries its timestamp both
at the top level and inside the layer sub-objects, because that is how the
locked domain model preserves it.  They are the same instant, so counting both
would double-count every record.  Only the top-level ``timestamp`` is read.

**Normalisation is reused, never re-derived.**  Address, country and
transaction-id equality use the locked Step 4B helpers, so this layer cannot
disagree with the layer that produced the evidence about when two addresses are
the same address.  A second implementation would be a chance to drift; an
import cannot.

**Every name is always emitted.**  A measurement that could not be taken is
emitted with ``ABSENT`` and no value, never omitted.  A consumer can therefore
distinguish "the extractor did not look" from "the extractor looked and found
nothing" without inspecting a missing key.

**Seconds are exact.**  Spans and rates are computed from integer microseconds
and exact decimals.  No elapsed time ever passes through a float, which cannot
represent most of the values in range and would make a span depend on rounding
rather than on the data.

Timezone policy
---------------
An aware timestamp may be ordered against other aware timestamps.  A naive one
has no known offset, and calling it UTC would be a guess, so naive timestamps
are never placed on a UTC timeline.  They are counted, reported separately, and
tagged ``NAIVE`` so a consumer can see what it is looking at.  When a set
mixes the two, the aware-derived features are marked ``PARTIAL`` and record
how many records they left out.  Nothing is silently normalised, and nothing is
silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterable, Optional, Sequence

from ..correlation.rules import (
    normalize_address,
    normalize_country,
    normalize_txid,
)
from ..domain.enums import ObservationType
from ..domain.models import UnifiedObservation
from ..graph.models import EvidenceEdge, EvidenceGraph
from .models import (
    AmountKind,
    BehavioralFeature,
    CountValue,
    DecimalValue,
    FeatureCategory,
    FeatureOrigin,
    FeatureScope,
    LabelSetValue,
    LabelValue,
    MeasurementStatus,
    TimestampKind,
    TimestampState,
    TimestampValue,
    derive_feature_id,
    divide_exactly,
)

__all__ = [
    "BEHAVIORAL_CATEGORIES",
    "BehaviorFeatureError",
    "BehaviorFeatureExtractor",
]

#: The categories a feature can belong to.  Every extraction covers all of
#: them, so a profile always carries the same names.
BEHAVIORAL_CATEGORIES: tuple[FeatureCategory, ...] = (
    FeatureCategory.ACTIVITY_VOLUME,
    FeatureCategory.TEMPORAL_ACTIVITY,
    FeatureCategory.TRANSACTION_AMOUNTS,
    FeatureCategory.NETWORK_BEHAVIOR,
    FeatureCategory.ADDRESS_PARTICIPATION,
    FeatureCategory.EVIDENCE_GRAPH,
    FeatureCategory.PROVENANCE_DIVERSITY,
)

#: A measurement awaiting emission: name, category, typed value, contributing
#: observations, and the recorded relationships consulted.
_Measurement = tuple[str, FeatureCategory, Any, set[str], tuple[str, ...]]

_SECONDS_PER_DAY = Decimal(86400)
_MICROSECONDS_PER_SECOND = 1_000_000


class BehaviorFeatureError(ValueError):
    """Raised when a feature extraction request cannot be honoured.

    A :class:`ValueError`, because the only way to arrive here is to hand the
    extractor something other than what it asked for: the wrong type, or a
    record that is not an observation.  That is a caller mistake, and quietly
    extracting from half-understood input would produce numbers that look
    measured but are not.
    """


@dataclass
class _Dimension:
    """A distinct-value accumulator that remembers who contributed and how often.

    Three things are tracked because three questions get asked: how many
    distinct values appeared, which records carried one, and how many times it
    appeared in total.  The contributor set is the audit trail -- a count of
    "4 distinct source IP addresses" is only checkable if you can see which
    four observations carried one.

    Internal bookkeeping rather than a public model, so it is deliberately
    mutable and unfrozen; the frozen models downstream are what leave this
    class.
    """

    values: set[str] = dataclass_field(default_factory=set)
    contributors: set[str] = dataclass_field(default_factory=set)
    occurrence_count: int = 0

    def add(self, value: Optional[str], observation_id: str) -> None:
        if value is None:
            return
        self.values.add(value)
        self.contributors.add(observation_id)
        self.occurrence_count += 1

    def sorted_values(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))

    def contributors_tuple(self) -> tuple[str, ...]:
        return tuple(sorted(self.contributors))


@dataclass
class _Facts:
    """Everything one pass over a set of observations found.

    Collection is separated from emission on purpose: the walk is a single
    linear pass, and turning its results into typed features is a separate pure
    step.  That keeps the arithmetic auditable and keeps the emitted order
    independent of the order records arrived in.
    """

    observation_ids: tuple[str, ...] = ()
    network_ids: set[str] = dataclass_field(default_factory=set)
    blockchain_ids: set[str] = dataclass_field(default_factory=set)
    transaction_ids: set[str] = dataclass_field(default_factory=set)
    type_ids: dict[ObservationType, set[str]] = dataclass_field(default_factory=dict)
    txid: _Dimension = dataclass_field(default_factory=_Dimension)
    aware_timestamps: list[datetime] = dataclass_field(default_factory=list)
    naive_timestamps: list[datetime] = dataclass_field(default_factory=list)
    timestamp_ids: set[str] = dataclass_field(default_factory=set)
    input_amounts: list[Decimal] = dataclass_field(default_factory=list)
    output_amounts: list[Decimal] = dataclass_field(default_factory=list)
    fees: list[Decimal] = dataclass_field(default_factory=list)
    amount_ids: set[str] = dataclass_field(default_factory=set)
    fee_ids: set[str] = dataclass_field(default_factory=set)
    input_addresses: _Dimension = dataclass_field(default_factory=_Dimension)
    output_addresses: _Dimension = dataclass_field(default_factory=_Dimension)
    src_ip: _Dimension = dataclass_field(default_factory=_Dimension)
    dst_ip: _Dimension = dataclass_field(default_factory=_Dimension)
    src_port: _Dimension = dataclass_field(default_factory=_Dimension)
    dst_port: _Dimension = dataclass_field(default_factory=_Dimension)
    asn: _Dimension = dataclass_field(default_factory=_Dimension)
    country: _Dimension = dataclass_field(default_factory=_Dimension)
    ingestion: _Dimension = dataclass_field(default_factory=_Dimension)
    source_file: _Dimension = dataclass_field(default_factory=_Dimension)
    source_record: _Dimension = dataclass_field(default_factory=_Dimension)
    provenance: dict[str, tuple[str, Optional[str], Optional[str]]] = (
        dataclass_field(default_factory=dict)
    )


def _span_seconds(ordered: Sequence[datetime]) -> Optional[Decimal]:
    """Exact difference between the earliest and latest instant, in seconds.

    Built from the integer microsecond difference and shifted, never
    converted to a float: ``timedelta.total_seconds()`` returns a binary float,
    and a span that is 1.1 seconds would come back as 1.1000000000000000888.

    Returns ``None`` only when there is no instant at all.  A *single* instant
    spans exactly zero seconds, and that is a real measurement rather than a
    missing one: the earliest recorded event and the latest recorded event are
    demonstrably the same moment, and reporting that as ``ABSENT`` would claim
    the data was missing when it is sitting right there.  A span of zero
    across two distinct instants is the same answer for a different reason.

    This is also why a *rate* stays ``ABSENT`` for a single event: a rate
    divides by an interval, and a zero-length interval is not one.
    """
    if not ordered:
        return None
    if len(ordered) == 1:
        return Decimal(0)
    delta = ordered[-1] - ordered[0]
    microseconds = (
        (delta.days * 86400 + delta.seconds) * _MICROSECONDS_PER_SECOND
        + delta.microseconds
    )
    return Decimal(microseconds).scaleb(-6)


def _mean(values: Sequence[Decimal]) -> Optional[Decimal]:
    """Exact mean under the pinned decimal context, or ``None`` if empty."""
    if not values:
        return None
    return divide_exactly(sum(values, Decimal(0)), len(values))


def _median(values: Sequence[Decimal]) -> Optional[Decimal]:
    """Exact median, averaging the middle pair when the count is even."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return divide_exactly(ordered[middle - 1] + ordered[middle], 2)


def _events_per_day(count: int, span: Optional[Decimal]) -> Optional[Decimal]:
    """Events per day, or ``None`` when there is no positive interval.

    A rate needs a duration to divide by.  With zero or one event there is no
    interval, so there is no rate, and the feature is ``ABSENT`` rather than
    zero or infinite.
    """
    if span is None or span <= 0:
        return None
    return divide_exactly(Decimal(count) * _SECONDS_PER_DAY, span)


class BehaviorFeatureExtractor:
    """Extracts descriptive features from observations and recorded evidence.

    Stateless apart from a read-only index of the graph it was given, and it
    never writes to its inputs.  Two extractions of the same input produce the
    same features in the same order, and an extraction of shuffled input
    produces the same features again.

    The evidence graph is optional.  When supplied, its relationships
    contribute *structural* descriptions only: how many recorded relationships
    name a set of observations, how many are internal to it, and which evidence
    types and rules they cite.  It is never walked for anything stronger than
    adjacency, and no code here would compute importance, standing, reach or
    any ranking of observations.
    """

    def __init__(self, graph: Optional[EvidenceGraph] = None) -> None:
        if graph is not None and not isinstance(graph, EvidenceGraph):
            raise BehaviorFeatureError(
                "graph must be an EvidenceGraph or None, not %r"
                % type(graph).__name__
            )
        self._graph = graph
        self._incidence: Optional[dict[str, tuple[EvidenceEdge, ...]]] = None

    @property
    def graph(self) -> Optional[EvidenceGraph]:
        """The evidence graph being measured against, if one was supplied."""
        return self._graph

    def extract_observation(
        self, observation: UnifiedObservation
    ) -> tuple[BehavioralFeature, ...]:
        """Describe a single observation.

        The full set of feature names is emitted whether or not the record
        carries the data behind them, so a caller can rely on a measurement
        being present rather than on its absence implying anything.
        """
        checked = self._materialise([observation])
        facts = self._collect(checked)
        subject = facts.observation_ids[0] if facts.observation_ids else None
        return self._render(
            facts,
            scope=FeatureScope.OBSERVATION,
            observation_id=subject,
            profile_id=None,
        )

    def extract_observations(
        self,
        observations: Iterable[UnifiedObservation],
        profile_id: Optional[str] = None,
    ) -> tuple[BehavioralFeature, ...]:
        """Describe a caller-supplied collection of observations.

        The collection is a grouping the caller has already decided; this
        method only measures it.  It does not decide who belongs together, does
        not invent an entity-resolution scheme, and looks at nothing outside the
        records handed to it.  ``profile_id``, when given, is recorded on the
        profile-scoped features so a feature can always be traced back to the
        profile that carries it.
        """
        checked = self._materialise(observations)
        facts = self._collect(checked)
        return self._render(
            facts, scope=FeatureScope.PROFILE, observation_id=None, profile_id=profile_id
        )

    # -- input handling --------------------------------------------------- #

    def _materialise(
        self, observations: Iterable[UnifiedObservation]
    ) -> tuple[UnifiedObservation, ...]:
        """Copy the caller's sequence into a tuple, checking each record.

        Reading the input into a tuple is not defensiveness for its own sake:
        it means the emitted features cannot depend on a list the caller later
        mutates, and it makes "no input mutation" a structural property rather
        than a promise.
        """
        if isinstance(observations, (str, bytes)):
            raise BehaviorFeatureError(
                "observations must be a sequence of UnifiedObservation, not text"
            )
        try:
            candidate = tuple(observations)
        except TypeError as exc:
            raise BehaviorFeatureError(
                "observations must be an iterable of UnifiedObservation: %s" % exc
            ) from exc
        for position, item in enumerate(candidate):
            if not isinstance(item, UnifiedObservation):
                raise BehaviorFeatureError(
                    "observation at position %d is %r, not a UnifiedObservation"
                    % (position, type(item).__name__)
                )
        return candidate

    # -- collection ------------------------------------------------------- #

    def _collect(self, observations: tuple[UnifiedObservation, ...]) -> _Facts:
        """Walk the observations once and gather every descriptive fact.

        Order-independent by construction: every accumulator is a set, a dict
        keyed by observation id, or a list whose contents are sorted before any
        arithmetic touches them.  Shuffled input therefore reaches the
        statistics in the same order every time.
        """
        facts = _Facts(
            observation_ids=tuple(
                sorted(observation.observation_id for observation in observations)
            )
        )
        for observation in observations:
            self._collect_observation(observation, facts)
        return facts

    def _collect_observation(
        self, observation: UnifiedObservation, facts: _Facts
    ) -> None:
        """Fold one observation into the accumulating facts."""
        observation_id = observation.observation_id

        network = observation.network
        if network is not None:
            if network.has_network_data:
                facts.network_ids.add(observation_id)
            self._add_field(facts.src_ip, network.src_ip, observation_id)
            self._add_field(facts.dst_ip, network.dst_ip, observation_id)
            if network.src_port is not None:
                facts.src_port.add(str(network.src_port), observation_id)
            if network.dst_port is not None:
                facts.dst_port.add(str(network.dst_port), observation_id)
            if network.asn is not None:
                facts.asn.add(str(network.asn), observation_id)
            facts.country.add(normalize_country(network.geo_country), observation_id)

        blockchain = observation.blockchain
        if blockchain is not None:
            facts.blockchain_ids.add(observation_id)
            if blockchain.has_structured_transaction_data:
                facts.transaction_ids.add(observation_id)
            facts.txid.add(normalize_txid(blockchain.txid), observation_id)
            for amount in blockchain.input_amounts:
                facts.input_amounts.append(amount)
                facts.amount_ids.add(observation_id)
            for amount in blockchain.output_amounts:
                facts.output_amounts.append(amount)
                facts.amount_ids.add(observation_id)
            if blockchain.fee is not None:
                facts.fees.append(blockchain.fee)
                facts.fee_ids.add(observation_id)
            for address in blockchain.input_addresses:
                facts.input_addresses.add(
                    normalize_address(address), observation_id
                )
            for address in blockchain.output_addresses:
                facts.output_addresses.add(
                    normalize_address(address), observation_id
                )

        # The event time is the observation's own top-level timestamp.  The
        # layer sub-objects repeat the same instant, so they are not read
        # again here; counting them would double-count every record.
        timestamp = observation.timestamp
        if timestamp is not None:
            facts.timestamp_ids.add(observation_id)
            if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
                facts.naive_timestamps.append(timestamp)
            else:
                facts.aware_timestamps.append(timestamp)

        provenance = observation.provenance
        facts.ingestion.add(provenance.ingestion_id, observation_id)
        facts.source_file.add(provenance.source_file, observation_id)
        facts.source_record.add(provenance.source_record, observation_id)
        facts.provenance[observation_id] = (
            provenance.ingestion_id,
            provenance.source_file,
            provenance.source_record,
        )

        facts.type_ids.setdefault(observation.observation_type, set()).add(
            observation_id
        )

    @staticmethod
    def _add_field(dimension: _Dimension, value: Any, observation_id: str) -> None:
        """Add an address or port field to a distinct-value accumulator.

        Ports and IP addresses are already canonical: the locked domain model
        validated them into ``int`` and ``IPvAnyAddress``, and ``str()`` of
        each is the comparable rendering.
        """
        if value is None:
            return
        dimension.add(str(value), observation_id)

    # -- evidence graph --------------------------------------------------- #

    def _index(self) -> dict[str, tuple[EvidenceEdge, ...]]:
        """Map each observation to the recorded relationships that name it.

        Built once per extractor and reused, so measuring many observations
        against one graph costs a single pass over the edges instead of a pass
        per observation.  This is plain adjacency: which relationships mention
        an observation, and nothing else.
        """
        if self._incidence is not None:
            return self._incidence
        collected: dict[str, list[EvidenceEdge]] = {}
        if self._graph is not None:
            for edge in self._graph.edges():
                for observation_id in edge.observation_ids():
                    collected.setdefault(observation_id, []).append(edge)
        self._incidence = {
            observation_id: tuple(
                sorted(edges, key=lambda edge: edge.relationship_id)
            )
            for observation_id, edges in collected.items()
        }
        return self._incidence

    def _incident_edges(self, members: set[str]) -> tuple[EvidenceEdge, ...]:
        """Recorded relationships naming at least one member, deduplicated.

        Deduplicated because a relationship between two members is found twice,
        once from each side, and counting it twice would inflate the count.
        """
        index = self._index()
        collected: dict[str, EvidenceEdge] = {}
        for member in members:
            for edge in index.get(member, ()):
                collected[edge.relationship_id] = edge
        return tuple(sorted(collected.values(), key=lambda edge: edge.relationship_id))

    # -- emission --------------------------------------------------------- #

    def _render(
        self,
        facts: _Facts,
        scope: FeatureScope,
        observation_id: Optional[str],
        profile_id: Optional[str],
    ) -> tuple[BehavioralFeature, ...]:
        """Turn collected facts into the complete, ordered feature set."""
        members = set(facts.observation_ids)
        measurements: list[_Measurement] = []
        measurements.extend(self._activity(facts, members))
        measurements.extend(self._temporal(facts))
        measurements.extend(self._amounts(facts))
        measurements.extend(self._network(facts))
        measurements.extend(self._addresses(facts))
        measurements.extend(self._graph_features(facts, members))
        measurements.extend(self._provenance(facts, members))

        features = [
            self._build(
                name=name,
                category=category,
                value=value,
                scope=scope,
                observation_id=observation_id,
                profile_id=profile_id,
                contributors=contributors,
                relationship_ids=relationship_ids,
                provenance=facts.provenance,
            )
            for name, category, value, contributors, relationship_ids in measurements
        ]
        features.sort(key=lambda feature: (feature.category.value, feature.name))
        return tuple(features)

    def _build(
        self,
        name: str,
        category: FeatureCategory,
        value: Any,
        scope: FeatureScope,
        observation_id: Optional[str],
        profile_id: Optional[str],
        contributors: set[str],
        relationship_ids: tuple[str, ...],
        provenance: dict[str, tuple[str, Optional[str], Optional[str]]],
    ) -> BehavioralFeature:
        """Attach a typed value, its origin, and its derived identifier.

        The origin's ingestion, file and record locators are read from the
        contributing observations rather than carried alongside each
        measurement, so the audit trail cannot drift out of step with the
        records a number was actually taken from.
        """
        return BehavioralFeature(
            feature_id=derive_feature_id(name, scope, observation_id, profile_id),
            name=name,
            category=category,
            scope=scope,
            value=value,
            observation_id=observation_id,
            profile_id=profile_id,
            origin=self._origin(contributors, relationship_ids, provenance),
        )

    def _origin(
        self,
        contributors: set[str],
        relationship_ids: tuple[str, ...],
        provenance: dict[str, tuple[str, Optional[str], Optional[str]]],
    ) -> FeatureOrigin:
        """Ingestion, file and record locators behind a set of contributors.

        Absence of a field on an observation is not recorded here.  An
        observation with no source IP simply is not among the contributors to
        the source-IP features, which is exactly what a contributor list should
        say.
        """
        ingestion_ids: set[str] = set()
        files: set[str] = set()
        records: set[str] = set()
        for contributor in contributors:
            record = provenance.get(contributor)
            if record is None:
                continue
            ingestion_ids.add(record[0])
            if record[1] is not None:
                files.add(record[1])
            if record[2] is not None:
                records.add(record[2])
        return FeatureOrigin(
            observation_ids=tuple(sorted(contributors)),
            ingestion_ids=tuple(sorted(ingestion_ids)),
            source_files=tuple(sorted(files)),
            source_records=tuple(sorted(records)),
            relationship_ids=tuple(relationship_ids),
        )

    # -- category emitters ------------------------------------------------ #

    def _activity(
        self, facts: _Facts, members: set[str]
    ) -> list[_Measurement]:
        """Counts of records, of which layers carry data, and of relationships."""
        out: list[_Measurement] = []
        out.append(
            (
                "observation_count",
                FeatureCategory.ACTIVITY_VOLUME,
                CountValue(value=len(facts.observation_ids), unit="observations"),
                set(facts.observation_ids),
                (),
            )
        )
        for name, ids, description in (
            ("network_observation_count", facts.network_ids, "network data"),
            ("blockchain_observation_count", facts.blockchain_ids, "blockchain data"),
            ("transaction_count", facts.transaction_ids, "transaction data"),
        ):
            out.append(
                (
                    name,
                    FeatureCategory.ACTIVITY_VOLUME,
                    CountValue(
                        value=len(ids),
                        unit="observations carrying %s" % description,
                    ),
                    set(ids),
                    (),
                )
            )
        for observation_type in ObservationType:
            ids = facts.type_ids.get(observation_type, set())
            out.append(
                (
                    "%s_type_observation_count"
                    % observation_type.value.lower(),
                    FeatureCategory.ACTIVITY_VOLUME,
                    CountValue(
                        value=len(ids),
                        unit="observations typed %s" % observation_type.value,
                    ),
                    set(ids),
                    (),
                )
            )
        out.append(
            (
                "txid_occurrence_count",
                FeatureCategory.ACTIVITY_VOLUME,
                CountValue(
                    value=facts.txid.occurrence_count,
                    unit="transaction id appearances",
                ),
                set(facts.txid.contributors),
                (),
            )
        )
        out.extend(
            self._count_and_values(
                "unique_txid_count",
                "txid_values",
                facts.txid,
                FeatureCategory.ACTIVITY_VOLUME,
                "distinct transaction ids",
            )
        )

        if self._graph is None:
            for name in (
                "evidence_relationship_count",
                "internal_evidence_relationship_count",
            ):
                out.append(
                    (
                        name,
                        FeatureCategory.ACTIVITY_VOLUME,
                        CountValue(status=MeasurementStatus.ABSENT),
                        set(),
                        (),
                    )
                )
            return out
        incident = self._incident_edges(members)
        internal = tuple(
            edge for edge in incident if set(edge.observation_ids()).issubset(members)
        )
        out.append(
            (
                "evidence_relationship_count",
                FeatureCategory.ACTIVITY_VOLUME,
                CountValue(
                    value=len(incident),
                    unit="recorded relationships naming this set",
                ),
                set(members),
                tuple(edge.relationship_id for edge in incident),
            )
        )
        out.append(
            (
                "internal_evidence_relationship_count",
                FeatureCategory.ACTIVITY_VOLUME,
                CountValue(
                    value=len(internal),
                    unit="recorded relationships with both sides inside this set",
                ),
                set(members),
                tuple(edge.relationship_id for edge in internal),
            )
        )
        return out

    def _temporal(self, facts: _Facts) -> list[_Measurement]:
        """When activity happened, without ever guessing a timezone.

        The four cases are all represented and none quietly becomes another:

        * no timestamps -- the counts are zero, the instants are ``ABSENT``;
        * one aware timestamp -- first and last are that instant, the span is
          exactly zero seconds because it is a single instant, and the rate is
          ``ABSENT`` because a rate needs an interval;
        * several aware timestamps -- first, last, span and rate are measured;
        * a mix of aware and naive -- the aware-derived features are
          ``PARTIAL`` and state how many naive records they left out, and the
          naive extents are reported separately and tagged ``NAIVE``.
        """
        out: list[_Measurement] = []
        aware = sorted(facts.aware_timestamps)
        naive = sorted(facts.naive_timestamps)
        timestamp_ids = set(facts.timestamp_ids)
        excluded = len(naive) if (aware and naive) else 0
        status = (
            MeasurementStatus.PARTIAL if excluded else MeasurementStatus.OBSERVED
        )

        for name, value, unit in (
            ("timestamp_count", len(aware) + len(naive), "event times"),
            ("aware_timestamp_count", len(aware), "event times with a timezone"),
            ("naive_timestamp_count", len(naive), "event times without a timezone"),
        ):
            out.append(
                (
                    name,
                    FeatureCategory.TEMPORAL_ACTIVITY,
                    CountValue(value=value, unit=unit),
                    timestamp_ids,
                    (),
                )
            )
        if aware and naive:
            state = TimestampState.MIXED
        elif aware:
            state = TimestampState.AWARE_ONLY
        elif naive:
            state = TimestampState.NAIVE_ONLY
        else:
            state = TimestampState.NONE
        out.append(
            (
                "timestamp_state",
                FeatureCategory.TEMPORAL_ACTIVITY,
                LabelValue(value=state.value),
                timestamp_ids,
                (),
            )
        )

        for name, value in (
            ("first_seen", aware[0] if aware else None),
            ("last_seen", aware[-1] if aware else None),
        ):
            out.append(
                (
                    name,
                    FeatureCategory.TEMPORAL_ACTIVITY,
                    (
                        TimestampValue(
                            status=status,
                            value=value,
                            excluded_count=excluded,
                            timestamp_kind=TimestampKind.AWARE,
                        )
                        if value is not None
                        else TimestampValue(status=MeasurementStatus.ABSENT)
                    ),
                    timestamp_ids,
                    (),
                )
            )

        span = _span_seconds(aware)
        for name in ("event_time_span_seconds", "active_duration_seconds"):
            out.append(
                (
                    name,
                    FeatureCategory.TEMPORAL_ACTIVITY,
                    (
                        DecimalValue(
                            status=status, value=span, excluded_count=excluded,
                            unit="seconds",
                        )
                        if span is not None
                        else DecimalValue(
                            status=MeasurementStatus.ABSENT, unit="seconds"
                        )
                    ),
                    timestamp_ids,
                    (),
                )
            )
        rate = _events_per_day(len(aware), span)
        out.append(
            (
                "events_per_day",
                FeatureCategory.TEMPORAL_ACTIVITY,
                (
                    DecimalValue(
                        status=status, value=rate, excluded_count=excluded,
                        unit="events/day",
                    )
                    if rate is not None
                    else DecimalValue(
                        status=MeasurementStatus.ABSENT, unit="events/day"
                    )
                ),
                timestamp_ids,
                (),
            )
        )

        naive_span = _span_seconds(naive)
        out.append(
            (
                "naive_first_seen",
                FeatureCategory.TEMPORAL_ACTIVITY,
                (
                    TimestampValue(value=naive[0], timestamp_kind=TimestampKind.NAIVE)
                    if naive
                    else TimestampValue(status=MeasurementStatus.ABSENT)
                ),
                timestamp_ids,
                (),
            )
        )
        out.append(
            (
                "naive_last_seen",
                FeatureCategory.TEMPORAL_ACTIVITY,
                (
                    TimestampValue(value=naive[-1], timestamp_kind=TimestampKind.NAIVE)
                    if naive
                    else TimestampValue(status=MeasurementStatus.ABSENT)
                ),
                timestamp_ids,
                (),
            )
        )
        out.append(
            (
                "naive_time_span_seconds",
                FeatureCategory.TEMPORAL_ACTIVITY,
                (
                    DecimalValue(value=naive_span, unit="seconds")
                    if naive_span is not None
                    else DecimalValue(status=MeasurementStatus.ABSENT, unit="seconds")
                ),
                timestamp_ids,
                (),
            )
        )
        return out

    def _amounts(self, facts: _Facts) -> list[_Measurement]:
        """Exact :class:`~decimal.Decimal` statistics over recorded amounts.

        Amounts are never converted to a float, and no price is applied: a
        Bitcoin amount is only ever compared with other Bitcoin amounts, since
        a fiat figure would be an external claim this layer holds no evidence
        for.

        Empty-set behaviour: with no amount recorded at all, the sums and the
        summary statistics are ``ABSENT`` with no value.  The *counts* remain
        ``OBSERVED`` at zero, because zero amounts were genuinely found.  An
        amount that is present and equal to zero is a third, separate case: an
        ``OBSERVED`` ``Decimal("0")``.
        """
        out: list[_Measurement] = []
        inputs = facts.input_amounts
        outputs = facts.output_amounts
        combined = inputs + outputs
        amount_ids = set(facts.amount_ids)

        for name, values in (
            ("input_amount_count", inputs),
            ("output_amount_count", outputs),
        ):
            out.append(
                (
                    name,
                    FeatureCategory.TRANSACTION_AMOUNTS,
                    CountValue(value=len(values), unit="amounts"),
                    amount_ids,
                    (),
                )
            )
        out.append(
            (
                "amount_count",
                FeatureCategory.TRANSACTION_AMOUNTS,
                CountValue(value=len(combined), unit="amounts"),
                amount_ids,
                (),
            )
        )
        out.append(
            (
                "fee_amount_count",
                FeatureCategory.TRANSACTION_AMOUNTS,
                CountValue(value=len(facts.fees), unit="fees"),
                set(facts.fee_ids),
                (),
            )
        )
        for name, values, ids, kind in (
            ("total_input_amount", inputs, amount_ids, AmountKind.TOTAL_INPUT),
            ("total_output_amount", outputs, amount_ids, AmountKind.TOTAL_OUTPUT),
            ("total_fee_amount", facts.fees, set(facts.fee_ids), AmountKind.TOTAL_FEE),
        ):
            out.append(
                (
                    name,
                    FeatureCategory.TRANSACTION_AMOUNTS,
                    (
                        DecimalValue(
                            value=sum(values, Decimal(0)), unit="BTC",
                            amount_kind=kind,
                        )
                        if values
                        else DecimalValue(
                            status=MeasurementStatus.ABSENT, unit="BTC",
                            amount_kind=kind,
                        )
                    ),
                    ids,
                    (),
                )
            )
        for name, value, kind in (
            ("minimum_amount", min(combined) if combined else None, AmountKind.MINIMUM),
            ("maximum_amount", max(combined) if combined else None, AmountKind.MAXIMUM),
            ("mean_amount", _mean(combined), AmountKind.MEAN),
            ("median_amount", _median(combined), AmountKind.MEDIAN),
        ):
            out.append(
                (
                    name,
                    FeatureCategory.TRANSACTION_AMOUNTS,
                    (
                        DecimalValue(value=value, unit="BTC", amount_kind=kind)
                        if value is not None
                        else DecimalValue(
                            status=MeasurementStatus.ABSENT, unit="BTC",
                            amount_kind=kind,
                        )
                    ),
                    amount_ids,
                    (),
                )
            )
        return out

    def _network(self, facts: _Facts) -> list[_Measurement]:
        """Distinct values seen in the network layer.

        Source and destination are counted separately and never merged: an IP
        address appearing as a source is a different record from the same
        address appearing as a destination, and folding them together would
        lose that.  They are addresses.  Nothing here says whose they are, and
        an address shared by two records says exactly that much.
        """
        return [
            measurement
            for count_name, values_name, dimension, unit in (
                (
                    "unique_src_ip_count", "src_ip_values", facts.src_ip,
                    "distinct source IP addresses",
                ),
                (
                    "unique_dst_ip_count", "dst_ip_values", facts.dst_ip,
                    "distinct destination IP addresses",
                ),
                (
                    "unique_src_port_count", "src_port_values", facts.src_port,
                    "distinct source ports",
                ),
                (
                    "unique_dst_port_count", "dst_port_values", facts.dst_port,
                    "distinct destination ports",
                ),
                (
                    "unique_asn_count", "asn_values", facts.asn,
                    "distinct autonomous system numbers",
                ),
                (
                    "unique_country_count", "country_values", facts.country,
                    "distinct country codes",
                ),
            )
            for measurement in self._count_and_values(
                count_name, values_name, dimension,
                FeatureCategory.NETWORK_BEHAVIOR, unit,
            )
        ]

    def _addresses(self, facts: _Facts) -> list[_Measurement]:
        """Distinct and total address participation, inputs and outputs apart.

        Equality follows the locked Step 4B rules: a bech32 address is
        case-insensitive by specification, a base58 address is case-sensitive
        because in base58 each character's case is data.  Occurrence counts sit
        beside distinct counts so a set that names one address twice stays
        visibly different from one that names it once.
        """
        out: list[_Measurement] = []
        out.extend(
            self._count_and_values(
                "unique_input_address_count",
                "input_address_values",
                facts.input_addresses,
                FeatureCategory.ADDRESS_PARTICIPATION,
                "distinct input addresses",
            )
        )
        out.extend(
            self._count_and_values(
                "unique_output_address_count",
                "output_address_values",
                facts.output_addresses,
                FeatureCategory.ADDRESS_PARTICIPATION,
                "distinct output addresses",
            )
        )
        every_address = _Dimension(
            values=set(facts.input_addresses.values)
            | set(facts.output_addresses.values),
            contributors=set(facts.input_addresses.contributors)
            | set(facts.output_addresses.contributors),
        )
        out.extend(
            self._count_and_values(
                "unique_address_count",
                "address_values",
                every_address,
                FeatureCategory.ADDRESS_PARTICIPATION,
                "distinct addresses",
            )
        )
        for name, dimension, role in (
            ("input_address_occurrences", facts.input_addresses, "input"),
            ("output_address_occurrences", facts.output_addresses, "output"),
        ):
            out.append(
                (
                    name,
                    FeatureCategory.ADDRESS_PARTICIPATION,
                    CountValue(
                        value=dimension.occurrence_count,
                        unit="%s address appearances" % role,
                    ),
                    set(dimension.contributors),
                    (),
                )
            )
        return out

    def _graph_features(
        self, facts: _Facts, members: set[str]
    ) -> list[_Measurement]:
        """Structural descriptions of the recorded-evidence graph.

        Everything here is adjacency: how many relationships exist, how many
        name a given set, and which evidence types and rules those
        relationships cite.  There is no ranking of observations, no notion of
        importance, no path between two records, and nothing that propagates
        from one record to another.  Why a relationship was recorded is the
        correlation layer's business; this layer counts what that layer
        produced.

        With no graph supplied, the structural features are ``ABSENT`` rather
        than zero: zero would claim "no recorded relationships exist", which is
        not what is known.  Only the distinct-value features stay ``OBSERVED``
        and empty, because an empty list of evidence types is a fact about
        *this* extraction, not about a graph that was never handed over.
        """
        out: list[_Measurement] = []

        if self._graph is None:
            for name in (
                "graph_node_count",
                "graph_edge_count",
                "incident_observation_degree",
                "incoming_evidence_edge_count",
                "outgoing_evidence_edge_count",
                "unique_evidence_type_count",
                "unique_rule_count",
            ):
                out.append(
                    (
                        name,
                        FeatureCategory.EVIDENCE_GRAPH,
                        CountValue(status=MeasurementStatus.ABSENT),
                        set(),
                        (),
                    )
                )
            for name in (
                "evidence_type_values",
                "rule_id_values",
                "incident_observation_values",
            ):
                out.append(
                    (
                        name,
                        FeatureCategory.EVIDENCE_GRAPH,
                        LabelSetValue(value=()),
                        set(),
                        (),
                    )
                )
            return out

        incident = self._incident_edges(members)
        incident_ids = tuple(edge.relationship_id for edge in incident)
        neighbours: set[str] = set()
        incoming = 0
        outgoing = 0
        for edge in incident:
            endpoints = set(edge.observation_ids())
            neighbours.update(endpoints - members)
            if endpoints.intersection(edge.source_observation_ids):
                outgoing += 1
            if endpoints.intersection(edge.target_observation_ids):
                incoming += 1

        out.append(
            (
                "graph_node_count",
                FeatureCategory.EVIDENCE_GRAPH,
                CountValue(
                    value=self._graph.node_count(),
                    unit="observations in the graph",
                ),
                set(members),
                incident_ids,
            )
        )
        out.append(
            (
                "graph_edge_count",
                FeatureCategory.EVIDENCE_GRAPH,
                CountValue(
                    value=self._graph.edge_count(),
                    unit="recorded relationships in the graph",
                ),
                set(members),
                incident_ids,
            )
        )
        out.append(
            (
                "incident_observation_degree",
                FeatureCategory.EVIDENCE_GRAPH,
                CountValue(
                    value=len(neighbours),
                    unit="observations named to this set by a recorded "
                         "relationship",
                ),
                set(members),
                incident_ids,
            )
        )
        out.append(
            (
                "incident_observation_values",
                FeatureCategory.EVIDENCE_GRAPH,
                LabelSetValue(value=tuple(sorted(neighbours))),
                set(members),
                incident_ids,
            )
        )
        out.append(
            (
                "incoming_evidence_edge_count",
                FeatureCategory.EVIDENCE_GRAPH,
                CountValue(
                    value=incoming,
                    unit="recorded relationships naming this set as target",
                ),
                set(members),
                incident_ids,
            )
        )
        out.append(
            (
                "outgoing_evidence_edge_count",
                FeatureCategory.EVIDENCE_GRAPH,
                CountValue(
                    value=outgoing,
                    unit="recorded relationships naming this set as source",
                ),
                set(members),
                incident_ids,
            )
        )

        evidence_types = _Dimension()
        rule_ids = _Dimension()
        for edge in incident:
            for endpoint in edge.observation_ids():
                evidence_types.add(edge.evidence_type.value, endpoint)
                rule_ids.add(edge.rule_id, endpoint)
        out.extend(
            self._count_and_values(
                "unique_evidence_type_count",
                "evidence_type_values",
                evidence_types,
                FeatureCategory.EVIDENCE_GRAPH,
                "distinct evidence types",
                incident_ids,
            )
        )
        out.extend(
            self._count_and_values(
                "unique_rule_count",
                "rule_id_values",
                rule_ids,
                FeatureCategory.EVIDENCE_GRAPH,
                "distinct correlation rules",
                incident_ids,
            )
        )
        return out

    def _provenance(
        self, facts: _Facts, members: set[str]
    ) -> list[_Measurement]:
        """How many distinct sources contributed, and which.

        Diversity of provenance is a fact about the data, not about anyone's
        conduct: several ingestion runs simply means several places were
        ingested from.  A citation count spans the in-scope observations' own
        provenance and the citations on the recorded relationships that name
        them, because both point back at source records and both are needed to
        answer "how much of this came from where".
        """
        out = []
        for count_name, values_name, dimension, unit in (
            (
                "unique_ingestion_count", "ingestion_id_values", facts.ingestion,
                "distinct ingestion runs",
            ),
            (
                "unique_source_file_count", "source_file_values", facts.source_file,
                "distinct source files",
            ),
            (
                "unique_source_record_count", "source_record_values",
                facts.source_record, "distinct record locators",
            ),
        ):
            out.extend(
                self._count_and_values(
                    count_name, values_name, dimension,
                    FeatureCategory.PROVENANCE_DIVERSITY, unit,
                )
            )
        citations = set(facts.provenance.values())
        if self._graph is not None:
            for edge in self._incident_edges(members):
                for reference in edge.provenance_refs:
                    citations.add(
                        (
                            reference.ingestion_id,
                            reference.source_file,
                            reference.source_record,
                        )
                    )
        out.append(
            (
                "provenance_reference_count",
                FeatureCategory.PROVENANCE_DIVERSITY,
                CountValue(
                    value=len(citations),
                    unit="distinct provenance citations",
                ),
                set(facts.provenance),
                (),
            )
        )
        return out

    # -- shared emitters -------------------------------------------------- #

    def _count_and_values(
        self,
        count_name: str,
        values_name: str,
        dimension: _Dimension,
        category: FeatureCategory,
        unit: str,
        relationship_ids: tuple[str, ...] = (),
    ) -> list[_Measurement]:
        """Emit a distinct-value count and the distinct values behind it.

        Always as a pair.  A count with no accompanying values is a number a
        reader has to take on trust; naming the values is what makes the count
        checkable, and it costs one extra feature.

        *relationship_ids* is passed through to both origins so that a
        measurement read off the recorded-evidence graph can name the
        relationships it was read from.  Without that, a count of "2 distinct
        correlation rules" would be a claim no reader could check.
        """
        return [
            (
                count_name,
                category,
                CountValue(value=len(dimension.values), unit=unit),
                set(dimension.contributors),
                relationship_ids,
            ),
            (
                values_name,
                category,
                LabelSetValue(value=dimension.sorted_values()),
                set(dimension.contributors),
                relationship_ids,
            ),
        ]
