"""Temporal semantics for investigation paths (SIH26146 - Step 7).

This module is about *how to compare*, never *what the comparison means*.

Rules
-----
1. Timezone-aware timestamps are normalised to UTC and compared normally.
2. Naive timestamps are never silently interpreted as UTC.  Two naive
   timestamps can only ever be compared with each other, and directly.
3. A mixed aware/naive pair is ``NOT_COMPARABLE``; no elapsed time is invented
   for it.
4. Ordering is deterministic: equal instants collapse to ``SAME_TIME`` and any
   caller that needs a stable order ties on identifiers.
5. Original timestamps are never mutated.  This module reads them and builds
   new, frozen descriptors.

The observation timestamps a caller supplies are the Step 1-3 ``timestamp``
values carried verbatim: aware values arrive already normalised to UTC and
naive values arrive still naive, and both are preserved here as they were
supplied.  ``build_temporal_index`` only *copies references* into a lookup
structure; it does not rewrite the observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from .models import (
    InvestigationError,
    TemporalOrder,
    TemporalTransition,
    TimestampAwareness,
)

__all__ = [
    "TemporalComparison",
    "build_temporal_index",
    "build_temporal_transition",
    "compare_timestamps",
    "is_aware",
    "normalize_aware",
    "timestamp_awareness",
]

_UTC = timezone.utc


def is_aware(value: datetime) -> bool:
    """True when *value* carries usable timezone information.

    A ``tzinfo`` that is not ``None`` but whose ``utcoffset`` is ``None`` is
    not *usable*, mirroring the Step 3 definition of a naive timestamp.
    """
    return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None


def normalize_aware(value: datetime) -> datetime:
    """Normalise a timezone-aware datetime to UTC.

    Both sides of a comparison are normalised to the same zone before they are
    compared, so the result does not depend on the zones the caller happened to
    supply.  Naive values are refused: this function's contract is "make an
    aware instant comparable", and there is no zone to normalise a naive value
    to without inventing one.
    """
    if not is_aware(value):
        raise InvestigationError(
            "cannot normalise a timezone-naive datetime to UTC; naive "
            "timestamps are never interpreted as UTC"
        )
    return value.astimezone(_UTC)


def timestamp_awareness(value: Any) -> TimestampAwareness:
    """Classify a carried timestamp as aware, naive or absent."""
    if value is None:
        return TimestampAwareness.ABSENT
    if not isinstance(value, datetime):
        raise InvestigationError(
            "a timestamp must be a datetime or None, not %s"
            % type(value).__name__
        )
    if is_aware(value):
        return TimestampAwareness.AWARE
    return TimestampAwareness.NAIVE


@dataclass(frozen=True)
class TemporalComparison:
    """The result of comparing two timestamps.

    ``relation`` describes how *target* sits relative to *source*
    (``AFTER`` = target later).  ``elapsed`` is ``target - source`` and is
    ``None`` when the two cannot be compared.  ``comparable`` says whether any
    ordering was established at all.
    """

    relation: TemporalOrder
    comparable: bool
    elapsed: Optional[timedelta] = None


def compare_timestamps(
    source: Optional[datetime], target: Optional[datetime]
) -> TemporalComparison:
    """Compare *target* against *source* under the module's five rules.

    One side missing: ``UNKNOWN``, not comparable.  One aware and one naive:
    ``NOT_COMPARABLE``, not comparable.  Two aware: normalised to UTC and
    compared.  Two naive: compared directly -- they are both naive, so no
    timezone is being assumed for either.
    """
    if source is None or target is None:
        return TemporalComparison(
            relation=TemporalOrder.UNKNOWN, comparable=False
        )
    source_aware = is_aware(source)
    target_aware = is_aware(target)
    if source_aware != target_aware:
        return TemporalComparison(
            relation=TemporalOrder.NOT_COMPARABLE, comparable=False
        )
    if source_aware:
        left = normalize_aware(source)
        right = normalize_aware(target)
    else:
        left = source
        right = target
    elapsed = right - left
    if elapsed > timedelta(0):
        relation = TemporalOrder.AFTER
    elif elapsed < timedelta(0):
        relation = TemporalOrder.BEFORE
    else:
        relation = TemporalOrder.SAME_TIME
    return TemporalComparison(
        relation=relation, comparable=True, elapsed=elapsed
    )


def build_temporal_transition(
    source_observation_id: str,
    target_observation_id: str,
    source_timestamp: Optional[datetime],
    target_timestamp: Optional[datetime],
    boundary: Optional[timedelta] = None,
) -> TemporalTransition:
    """Build the temporal descriptor of one adjacent pair.

    ``boundary``, when supplied, drives the purely descriptive
    ``crosses_boundary`` flag: the pair reports that its gap exceeds the
    boundary's magnitude.  Descriptive only -- a large gap is still just a gap
    and is never labelled anything here.
    """
    comparison = compare_timestamps(source_timestamp, target_timestamp)
    crosses_boundary: Optional[bool] = None
    if comparison.comparable and boundary is not None and comparison.elapsed is not None:
        crosses_boundary = abs(comparison.elapsed).total_seconds() > boundary.total_seconds()
    return TemporalTransition(
        source_observation_id=source_observation_id,
        target_observation_id=target_observation_id,
        relation=comparison.relation,
        comparable=comparison.comparable,
        source_timestamp=source_timestamp,
        target_timestamp=target_timestamp,
        source_awareness=timestamp_awareness(source_timestamp),
        target_awareness=timestamp_awareness(target_timestamp),
        elapsed_seconds=(
            None if comparison.elapsed is None else comparison.elapsed.total_seconds()
        ),
        crosses_boundary=crosses_boundary,
    )


def build_temporal_index(observations: Optional[Any]) -> dict[str, Optional[datetime]]:
    """Build the observation-id -> timestamp lookup for a search.

    Accepts either a mapping of ``observation_id`` to a timestamp (or ``None``)
    or a sequence of objects carrying ``observation_id`` and ``timestamp``
    attributes -- the Step 1-3 :class:`UnifiedObservation` shape among others.

    The returned dict is the engine's private working copy; the observations
    themselves are only read.  A duplicated observation id is an inconsistent
    input and is refused rather than guessed.
    """
    index: dict[str, Optional[datetime]] = {}
    if observations is None:
        return index
    if isinstance(observations, Mapping):
        for observation_id, timestamp in observations.items():
            if not isinstance(observation_id, str) or not observation_id.strip():
                raise InvestigationError(
                    "a temporal index maps observation ids to timestamps; %r "
                    "is not a usable key" % (observation_id,)
                )
            _put_index_entry(index, observation_id, timestamp)
        return index
    for observation in observations:
        observation_id = getattr(observation, "observation_id", None)
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise InvestigationError(
                "an observation without a usable observation_id cannot be "
                "indexed for time"
            )
        _put_index_entry(index, observation_id, getattr(observation, "timestamp", None))
    return index


def _put_index_entry(
    index: dict[str, Optional[datetime]],
    observation_id: str,
    timestamp: Any,
) -> None:
    if timestamp is not None and not isinstance(timestamp, datetime):
        raise InvestigationError(
            "the timestamp for observation %r must be a datetime or None, not "
            "%s" % (observation_id, type(timestamp).__name__)
        )
    if observation_id in index:
        raise InvestigationError(
            "duplicate observation_id %r in the temporal index; an "
            "observation can carry exactly one timestamp per search"
            % observation_id
        )
    index[observation_id] = timestamp