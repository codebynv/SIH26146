"""Tests for the Step 7 temporal semantics (SIH26146 - Step 7).

The guarantee under test is the five comparison rules in
``backend.app.investigation.temporal``: aware values compare in UTC, naive
values compare directly, a mixed pair is never comparable, a missing side is
unknown, and ordering is deterministic.  Everything here checks *how* the layer
compares, never what a comparison *means*.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo

import pytest
from pydantic import ValidationError

from backend.app.investigation import temporal as temporal_module
from backend.app.investigation.models import (
    InvestigationError,
    TemporalOrder,
    TemporalTransition,
    TimestampAwareness,
)
from backend.app.investigation.temporal import (
    TemporalComparison,
    build_temporal_index,
    build_temporal_transition,
    compare_timestamps,
    is_aware,
    normalize_aware,
    timestamp_awareness,
)

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2024, 5, 1, 12, 4, tzinfo=UTC)
T2 = datetime(2024, 5, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
NAIVE_0 = datetime(2024, 5, 1, 12, 0)
NAIVE_1 = datetime(2024, 5, 1, 12, 4)

OBS_A = "obs-a"
OBS_B = "obs-b"


class _NullOffsetTz(tzinfo):
    """A tzinfo whose utcoffset is ``None``: present but not usable."""

    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None

    def tzname(self, dt):
        return "null-offset"


# --------------------------------------------------------------------------- #
# The public surface
# --------------------------------------------------------------------------- #


class TestPublicSurface:
    def test_the_module_exports_a_sorted_complete_list(self):
        assert list(temporal_module.__all__) == sorted(temporal_module.__all__)
        assert set(temporal_module.__all__) <= set(dir(temporal_module))
        for name in temporal_module.__all__:
            assert getattr(temporal_module, name) is not None, name


# --------------------------------------------------------------------------- #
# Awareness
# --------------------------------------------------------------------------- #


class TestAwareness:
    def test_aware_detection(self):
        assert is_aware(T0) is True
        assert is_aware(NAIVE_0) is False

    def test_a_null_offset_tzinfo_is_not_usable_awareness(self):
        value = datetime(2024, 5, 1, 12, 0, tzinfo=_NullOffsetTz())
        assert is_aware(value) is False
        assert timestamp_awareness(value) is TimestampAwareness.NAIVE

    def test_awareness_classification(self):
        assert timestamp_awareness(None) is TimestampAwareness.ABSENT
        assert timestamp_awareness(T0) is TimestampAwareness.AWARE
        assert timestamp_awareness(NAIVE_0) is TimestampAwareness.NAIVE

    def test_non_datetime_is_refused(self):
        with pytest.raises(InvestigationError):
            timestamp_awareness(123)

    def test_normalize_aware_preserves_the_instant(self):
        assert normalize_aware(T2) == T0
        assert normalize_aware(T2).utcoffset() == timedelta(0)

    def test_normalize_aware_refuses_naive_values(self):
        with pytest.raises(InvestigationError):
            normalize_aware(NAIVE_0)


# --------------------------------------------------------------------------- #
# compare_timestamps
# --------------------------------------------------------------------------- #


class TestCompareTimestamps:
    def test_missing_side_is_unknown(self):
        assert compare_timestamps(None, T0) == TemporalComparison(
            relation=TemporalOrder.UNKNOWN, comparable=False
        )
        assert compare_timestamps(T0, None) == TemporalComparison(
            relation=TemporalOrder.UNKNOWN, comparable=False
        )
        assert compare_timestamps(None, None) == TemporalComparison(
            relation=TemporalOrder.UNKNOWN, comparable=False
        )

    def test_mixed_awareness_is_not_comparable(self):
        result = compare_timestamps(T0, NAIVE_1)
        assert result.comparable is False
        assert result.relation is TemporalOrder.NOT_COMPARABLE
        assert result.elapsed is None

    def test_aware_pair_compares_in_utc(self):
        result = compare_timestamps(T0, T1)
        assert result.comparable is True
        assert result.relation is TemporalOrder.AFTER
        assert result.elapsed == timedelta(minutes=4)

    def test_zones_do_not_change_the_answer(self):
        result = compare_timestamps(T0, T2)
        assert result.comparable is True
        assert result.relation is TemporalOrder.SAME_TIME
        assert result.elapsed == timedelta(0)

    def test_naive_pair_compares_directly(self):
        assert compare_timestamps(NAIVE_0, NAIVE_1).relation is TemporalOrder.AFTER
        assert compare_timestamps(NAIVE_1, NAIVE_0).relation is TemporalOrder.BEFORE

    def test_equal_instants_collapse_to_same_time(self):
        assert compare_timestamps(T0, T0).relation is TemporalOrder.SAME_TIME

    def test_timestamps_are_never_mutated(self):
        before = T2.replace()
        compare_timestamps(T0, T2)
        assert T2 == before
        assert T2.utcoffset() == timedelta(hours=2)


# --------------------------------------------------------------------------- #
# TemporalComparison
# --------------------------------------------------------------------------- #


class TestTemporalComparison:
    def test_is_frozen(self):
        comparison = TemporalComparison(
            relation=TemporalOrder.AFTER, comparable=True, elapsed=timedelta(1)
        )
        with pytest.raises(Exception):
            comparison.relation = TemporalOrder.BEFORE


# --------------------------------------------------------------------------- #
# build_temporal_transition
# --------------------------------------------------------------------------- #


class TestBuildTemporalTransition:
    def test_comparable_pair(self):
        transition = build_temporal_transition(OBS_A, OBS_B, T0, T1)
        assert transition.comparable is True
        assert transition.relation is TemporalOrder.AFTER
        assert transition.elapsed_seconds == 240.0
        assert transition.source_awareness is TimestampAwareness.AWARE
        assert transition.target_awareness is TimestampAwareness.AWARE

    def test_mixed_pair_reports_not_comparable_without_elapsed(self):
        transition = build_temporal_transition(OBS_A, OBS_B, T0, NAIVE_1)
        assert transition.comparable is False
        assert transition.relation is TemporalOrder.NOT_COMPARABLE
        assert transition.elapsed_seconds is None
        assert transition.crosses_boundary is None

    def test_missing_sides_report_unknown(self):
        transition = build_temporal_transition(OBS_A, OBS_B, None, None)
        assert transition.comparable is False
        assert transition.relation is TemporalOrder.UNKNOWN
        assert transition.source_awareness is TimestampAwareness.ABSENT

    def test_boundary_is_descriptive_only(self):
        transition = build_temporal_transition(
            OBS_A, OBS_B, T0, T1, boundary=timedelta(minutes=3)
        )
        assert transition.crosses_boundary is True
        transition = build_temporal_transition(
            OBS_A, OBS_B, T0, T1, boundary=timedelta(minutes=5)
        )
        assert transition.crosses_boundary is False

    def test_no_boundary_means_no_boundary_flag(self):
        transition = build_temporal_transition(OBS_A, OBS_B, T0, T1)
        assert transition.crosses_boundary is None

    def test_boundary_is_never_reported_for_non_comparable_pairs(self):
        transition = build_temporal_transition(
            OBS_A, OBS_B, T0, NAIVE_1, boundary=timedelta(minutes=3)
        )
        assert transition.crosses_boundary is None

    def test_the_result_is_a_valid_frozen_transition(self):
        transition = build_temporal_transition(OBS_A, OBS_B, T0, T1)
        assert isinstance(transition, TemporalTransition)
        with pytest.raises(ValidationError):
            transition.relation = TemporalOrder.BEFORE


# --------------------------------------------------------------------------- #
# build_temporal_index
# --------------------------------------------------------------------------- #


class _Observation:
    def __init__(self, observation_id, timestamp=None):
        self.observation_id = observation_id
        self.timestamp = timestamp


class TestBuildTemporalIndex:
    def test_none_builds_an_empty_index(self):
        assert build_temporal_index(None) == {}

    def test_mapping_form_copies_references_only(self):
        source = {OBS_A: T0, OBS_B: None}
        index = build_temporal_index(source)
        assert index == source
        assert index[OBS_A] is T0
        source[OBS_A] = T1
        assert index[OBS_A] is T0

    def test_sequence_form_uses_observation_attributes(self):
        observations = [_Observation(OBS_A, T0), _Observation(OBS_B, None)]
        assert build_temporal_index(observations) == {OBS_A: T0, OBS_B: None}

    def test_sequence_items_may_be_plain_mappings_of_observations(self):
        observations = (
            _Observation(OBS_A, T0),
            _Observation(OBS_B, None),
        )
        index = build_temporal_index(observations)
        assert index[OBS_A] is T0

    def test_duplicate_observation_id_is_refused(self):
        with pytest.raises(InvestigationError):
            build_temporal_index(
                [_Observation(OBS_A, T0), _Observation(OBS_A, T1)]
            )

    def test_non_datetime_timestamp_is_refused(self):
        with pytest.raises(InvestigationError):
            build_temporal_index({OBS_A: "not-a-datetime"})
        with pytest.raises(InvestigationError):
            build_temporal_index([_Observation(OBS_A, "not-a-datetime")])

    def test_a_blank_mapping_key_is_refused(self):
        with pytest.raises(InvestigationError):
            build_temporal_index({"  ": T0})

    def test_a_sequence_item_without_an_id_is_refused(self):
        class _Nameless:
            timestamp = T0

        with pytest.raises(InvestigationError):
            build_temporal_index([_Nameless()])