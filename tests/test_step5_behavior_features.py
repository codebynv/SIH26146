"""Step 5 tests: behavioral feature extraction.

Every required category is covered against observations built with the locked
domain model, and the evidence-graph features are covered against a graph
produced by the locked correlation engine -- not a hand-built stub, so the
descriptors are checked against the relationships that really get recorded.

Three things are tested hardest, because they are the ones that would quietly
produce a wrong number: timezone handling, exact decimal arithmetic, and input
immutability.
"""

from __future__ import annotations

import ast
import copy
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

import pytest

from backend.app import behavior
from backend.app.behavior.features import (
    BEHAVIORAL_CATEGORIES,
    BehaviorFeatureError,
    BehaviorFeatureExtractor,
)
from backend.app.behavior.models import (
    AmountKind,
    BehavioralFeature,
    CountValue,
    DecimalValue,
    FeatureCategory,
    FeatureScope,
    LabelSetValue,
    LabelValue,
    MeasurementStatus,
    TimestampKind,
    TimestampState,
    TimestampValue,
)
from backend.app.correlation.engine import CorrelationEngine
from backend.app.domain.enums import ObservationType
from backend.app.domain.models import (
    BlockchainTransaction,
    NetworkObservation,
    Provenance,
    UnifiedObservation,
)
from backend.app.graph.builder import build_graph

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2024, 5, 1, 12, 0, 30, tzinfo=UTC)
T2 = datetime(2024, 5, 2, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)

#: The complete set of feature names, by category.  Pinned as a literal so that
#: adding, removing or renaming a measurement is a deliberate, visible change
#: rather than something a consumer discovers when a number goes missing.
EXPECTED_FEATURES = {
    FeatureCategory.ACTIVITY_VOLUME: (
        "blockchain_observation_count",
        "blockchain_type_observation_count",
        "combined_type_observation_count",
        "evidence_relationship_count",
        "internal_evidence_relationship_count",
        "network_observation_count",
        "network_type_observation_count",
        "observation_count",
        "transaction_count",
        "txid_occurrence_count",
        "txid_values",
        "unique_txid_count",
        "unknown_type_observation_count",
    ),
    FeatureCategory.ADDRESS_PARTICIPATION: (
        "address_values",
        "input_address_occurrences",
        "input_address_values",
        "output_address_occurrences",
        "output_address_values",
        "unique_address_count",
        "unique_input_address_count",
        "unique_output_address_count",
    ),
    FeatureCategory.EVIDENCE_GRAPH: (
        "evidence_type_values",
        "graph_edge_count",
        "graph_node_count",
        "incident_observation_degree",
        "incident_observation_values",
        "incoming_evidence_edge_count",
        "outgoing_evidence_edge_count",
        "rule_id_values",
        "unique_evidence_type_count",
        "unique_rule_count",
    ),
    FeatureCategory.NETWORK_BEHAVIOR: (
        "asn_values",
        "country_values",
        "dst_ip_values",
        "dst_port_values",
        "src_ip_values",
        "src_port_values",
        "unique_asn_count",
        "unique_country_count",
        "unique_dst_ip_count",
        "unique_dst_port_count",
        "unique_src_ip_count",
        "unique_src_port_count",
    ),
    FeatureCategory.PROVENANCE_DIVERSITY: (
        "ingestion_id_values",
        "provenance_reference_count",
        "source_file_values",
        "source_record_values",
        "unique_ingestion_count",
        "unique_source_file_count",
        "unique_source_record_count",
    ),
    FeatureCategory.TEMPORAL_ACTIVITY: (
        "active_duration_seconds",
        "aware_timestamp_count",
        "event_time_span_seconds",
        "events_per_day",
        "first_seen",
        "last_seen",
        "naive_first_seen",
        "naive_last_seen",
        "naive_time_span_seconds",
        "naive_timestamp_count",
        "timestamp_count",
        "timestamp_state",
    ),
    FeatureCategory.TRANSACTION_AMOUNTS: (
        "amount_count",
        "fee_amount_count",
        "input_amount_count",
        "maximum_amount",
        "mean_amount",
        "median_amount",
        "minimum_amount",
        "output_amount_count",
        "total_fee_amount",
        "total_input_amount",
        "total_output_amount",
    ),
}

EXPECTED_FEATURE_COUNT = sum(len(names) for names in EXPECTED_FEATURES.values())


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def make_observation(
    observation_id: str,
    *,
    timestamp: Optional[datetime] = None,
    observation_type: ObservationType = ObservationType.COMBINED,
    src_ip: Any = None,
    dst_ip: Any = None,
    src_port: Any = None,
    dst_port: Any = None,
    asn: Any = None,
    geo_country: Optional[str] = None,
    txid: Optional[str] = None,
    input_addresses: Iterable[Any] = (),
    output_addresses: Iterable[Any] = (),
    input_amounts: Iterable[Any] = (),
    output_amounts: Iterable[Any] = (),
    fee: Any = None,
    ingestion_id: str = "ing-1",
    source_file: Optional[str] = "data.csv",
    source_record: Optional[str] = None,
    extra: Optional[dict] = None,
) -> UnifiedObservation:
    """Build one observation with the locked Step 3 model."""
    network = None
    if any(
        value is not None
        for value in (src_ip, dst_ip, src_port, dst_port, asn, geo_country)
    ):
        network = NetworkObservation(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            asn=asn,
            geo_country=geo_country,
        )
    blockchain = None
    inputs_in = [Decimal(str(value)) for value in input_amounts]
    outputs_out = [Decimal(str(value)) for value in output_amounts]
    inputs_addr = list(input_addresses)
    outputs_addr = list(output_addresses)
    if (
        txid is not None
        or fee is not None
        or inputs_addr
        or outputs_addr
        or inputs_in
        or outputs_out
    ):
        blockchain = BlockchainTransaction(
            timestamp=timestamp,
            txid=txid,
            input_addresses=inputs_addr,
            output_addresses=outputs_addr,
            input_amounts=inputs_in,
            output_amounts=outputs_out,
            fee=None if fee is None else Decimal(str(fee)),
        )
    return UnifiedObservation(
        observation_id=observation_id,
        observation_type=observation_type,
        timestamp=timestamp,
        network=network,
        blockchain=blockchain,
        provenance=Provenance(
            ingestion_id=ingestion_id,
            source_file=source_file,
            source_record=source_record if source_record is not None else "row %s" % observation_id,
            context={"declared": "csv"},
        ),
        extra=dict(extra or {}),
    )


def as_map(features: tuple[BehavioralFeature, ...]) -> dict[str, BehavioralFeature]:
    """Index a feature tuple by name."""
    return {feature.name: feature for feature in features}


def value_of(features, name):
    return as_map(features)[name].value.value


def status_of(features, name):
    return as_map(features)[name].status


def build_graph_for(observations: list[UnifiedObservation]):
    """Build a recorded-evidence graph with the locked correlation layer."""
    return build_graph(list(CorrelationEngine().run(observations)))


def synthetic(count: int) -> list[UnifiedObservation]:
    """A deterministic synthetic set: no clock, no randomness, no shared state."""
    records = []
    for index in range(count):
        records.append(
            make_observation(
                "obs-%05d" % index,
                timestamp=T0 + timedelta(hours=index % 48),
                src_ip="10.0.%d.%d" % (index % 5, index % 17),
                dst_ip="10.1.%d.%d" % (index % 3, index % 11),
                src_port=1024 + (index % 900),
                dst_port=443,
                asn=15169 + (index % 4),
                geo_country="IN" if index % 2 else "US",
                txid="%064x" % (index % 700),
                input_addresses=["bc1q%08d" % (index % 900)],
                output_addresses=["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
                input_amounts=[Decimal("0.%03d" % (index % 999))],
                output_amounts=[Decimal("0.%03d" % ((index + 1) % 999))],
                fee=Decimal("0.0001"),
                ingestion_id="ing-%d" % (index % 3),
                source_file="feed-%d.csv" % (index % 3),
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Shape of the feature set
# --------------------------------------------------------------------------- #


class TestFeatureSetShape:
    def test_every_extraction_returns_the_complete_named_set(self):
        features = BehaviorFeatureExtractor().extract_observations([make_observation("o1")])
        assert len(features) == EXPECTED_FEATURE_COUNT
        assert as_map(features).keys() == {
            name for names in EXPECTED_FEATURES.values() for name in names
        }

    def test_the_feature_set_does_not_depend_on_the_data(self):
        """A record with no data at all still yields every measurement name.

        That is the point of emitting ``ABSENT`` rather than omitting: a
        consumer never has to guess whether a missing key means "not looked
        at" or "looked at, found nothing".
        """
        bare = BehaviorFeatureExtractor().extract_observations(
            [make_observation("bare", observation_type=ObservationType.UNKNOWN)]
        )
        full = BehaviorFeatureExtractor().extract_observations(synthetic(3))
        assert as_map(bare).keys() == as_map(full).keys()
        assert len(bare) == len(full) == EXPECTED_FEATURE_COUNT

    def test_every_category_is_represented(self):
        features = BehaviorFeatureExtractor().extract_observations(synthetic(2))
        present = {feature.category for feature in features}
        assert present == set(BEHAVIORAL_CATEGORIES)
        assert tuple(BEHAVIORAL_CATEGORIES) == tuple(FeatureCategory)

    def test_features_are_ordered_by_category_then_name(self):
        features = BehaviorFeatureExtractor().extract_observations(synthetic(4))
        keys = [(feature.category.value, feature.name) for feature in features]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)

    def test_observation_scope_binds_to_one_record(self):
        features = BehaviorFeatureExtractor().extract_observation(
            make_observation("only-one")
        )
        assert all(feature.scope is FeatureScope.OBSERVATION for feature in features)
        assert {feature.observation_id for feature in features} == {"only-one"}
        assert all(feature.profile_id is None for feature in features)

    def test_group_scope_is_free_of_observation_ids(self):
        features = BehaviorFeatureExtractor().extract_observations(synthetic(2))
        assert all(feature.scope is FeatureScope.PROFILE for feature in features)
        assert all(feature.observation_id is None for feature in features)

    def test_a_supplied_profile_id_is_recorded_on_every_feature(self):
        profile_id = "c" * 8 + "-0000-5000-8000-000000000000"
        features = BehaviorFeatureExtractor().extract_observations(
            synthetic(2), profile_id=profile_id
        )
        assert {feature.profile_id for feature in features} == {profile_id}

    def test_the_documented_names_match_the_extractor(self):
        """Pin the two lists together so neither can drift unnoticed."""
        features = BehaviorFeatureExtractor().extract_observations(synthetic(1))
        emitted: dict[FeatureCategory, set[str]] = {}
        for feature in features:
            emitted.setdefault(feature.category, set()).add(feature.name)
        for category, names in EXPECTED_FEATURES.items():
            assert emitted[category] == set(names), category


# --------------------------------------------------------------------------- #
# Input handling and immutability
# --------------------------------------------------------------------------- #


class TestInputHandling:
    def test_a_non_observation_is_refused(self):
        with pytest.raises(BehaviorFeatureError) as excinfo:
            BehaviorFeatureExtractor().extract_observations(
                [make_observation("a"), {"observation_id": "b"}]
            )
        assert "position 1" in str(excinfo.value)

    def test_text_is_not_a_sequence_of_observations(self):
        with pytest.raises(BehaviorFeatureError):
            BehaviorFeatureExtractor().extract_observations("obs-a")

    def test_something_uniterable_is_refused(self):
        with pytest.raises(BehaviorFeatureError):
            BehaviorFeatureExtractor().extract_observations(42)

    def test_a_graph_of_the_wrong_type_is_refused(self):
        with pytest.raises(BehaviorFeatureError):
            BehaviorFeatureExtractor(graph="not a graph")

    def test_extraction_does_not_mutate_the_observations(self):
        """Snapshot before, compare after -- including nested mutable data."""
        observations = [
            make_observation(
                "mut-%d" % index,
                timestamp=T0 + timedelta(hours=index),
                src_ip="10.0.0.%d" % (index + 1),
                src_port=5000 + index,
                input_addresses=["bc1q%04d" % index],
                input_amounts=[Decimal("0.%02d" % (index + 1))],
                extra={"nested": {"values": [index, index + 1]}},
            )
            for index in range(4)
        ]
        snapshots = copy.deepcopy(observations)
        nested_ids = [
            id(observation.extra["nested"]) for observation in observations
        ]
        context_ids = [
            id(observation.provenance.context) for observation in observations
        ]
        amount_ids = [
            tuple(id(amount) for amount in observation.blockchain.input_amounts)
            for observation in observations
        ]

        BehaviorFeatureExtractor().extract_observations(observations)

        assert observations == snapshots
        assert [id(o.extra["nested"]) for o in observations] == nested_ids
        assert [id(o.provenance.context) for o in observations] == context_ids
        assert [
            tuple(id(amount) for amount in o.blockchain.input_amounts) for o in observations
        ] == amount_ids

    def test_extraction_does_not_mutate_the_graph(self):
        observations = [
            make_observation("g-a", src_ip="10.0.0.1", timestamp=T0),
            make_observation("g-b", src_ip="10.0.0.1", timestamp=T0 + DAY),
        ]
        graph = build_graph_for(observations)
        snapshot = copy.deepcopy(graph.to_dict())
        BehaviorFeatureExtractor(graph).extract_observations(observations)
        assert graph.to_dict() == snapshot

    def test_a_generator_input_is_accepted(self):
        features = BehaviorFeatureExtractor().extract_observations(
            make_observation("gen-%d" % index) for index in range(3)
        )
        assert value_of(features, "observation_count") == 3

    def test_an_empty_input_is_answered_not_rejected(self):
        features = BehaviorFeatureExtractor().extract_observations([])
        assert value_of(features, "observation_count") == 0
        assert status_of(features, "total_input_amount") is MeasurementStatus.ABSENT


# --------------------------------------------------------------------------- #
# Activity volume
# --------------------------------------------------------------------------- #


class TestActivityFeatures:
    def test_observation_count(self):
        features = BehaviorFeatureExtractor().extract_observations(synthetic(9))
        assert value_of(features, "observation_count") == 9

    def test_layer_counts_look_at_the_data_that_is_present(self):
        records = [
            make_observation("net-1", src_ip="10.0.0.1", observation_type=ObservationType.NETWORK),
            make_observation("chain-1", txid="a" * 64, observation_type=ObservationType.BLOCKCHAIN),
            make_observation(
                "both-1",
                src_ip="10.0.0.2",
                txid="b" * 64,
                observation_type=ObservationType.COMBINED,
            ),
            make_observation("bare-1", observation_type=ObservationType.UNKNOWN),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "observation_count") == 4
        assert value_of(features, "network_observation_count") == 2
        assert value_of(features, "blockchain_observation_count") == 2
        assert value_of(features, "transaction_count") == 2

    def test_a_lone_fee_is_not_a_transaction(self):
        """A generic ``fee`` column is not blockchain evidence on its own."""
        records = [
            make_observation("fee-only", fee=Decimal("0.001"), observation_type=ObservationType.NETWORK)
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "blockchain_observation_count") == 1
        assert value_of(features, "transaction_count") == 0

    def test_observation_types_are_counted_separately(self):
        records = [
            make_observation("n-1", src_ip="10.0.0.1", observation_type=ObservationType.NETWORK),
            make_observation("n-2", dst_ip="10.0.0.2", observation_type=ObservationType.NETWORK),
            make_observation("b-1", txid="c" * 64, observation_type=ObservationType.BLOCKCHAIN),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "network_type_observation_count") == 2
        assert value_of(features, "blockchain_type_observation_count") == 1
        assert value_of(features, "combined_type_observation_count") == 0
        assert value_of(features, "unknown_type_observation_count") == 0

    def test_transaction_id_occurrences_and_distinct_ids_differ(self):
        records = [
            make_observation("t-1", txid="AB" + "0" * 62),
            make_observation("t-2", txid="ab" + "0" * 62),
            make_observation("t-3", txid="cd" + "0" * 62),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "txid_occurrence_count") == 3
        assert value_of(features, "unique_txid_count") == 2
        assert value_of(features, "txid_values") == (
            "ab" + "0" * 62,
            "cd" + "0" * 62,
        )

    def test_every_measured_number_declares_its_unit(self):
        """A count or a total without a unit is a number a reader must guess at.

        The exemption runs both ways: an instant is not measured *in* seconds
        and a label has no unit at all, so attaching one would be as much of an
        invention as leaving one off a count.
        """
        features = BehaviorFeatureExtractor().extract_observations(synthetic(3))
        for feature in features:
            if (
                isinstance(feature.value, (CountValue, DecimalValue))
                and feature.status is MeasurementStatus.OBSERVED
            ):
                assert feature.unit, feature.name
            elif isinstance(feature.value, (LabelValue, LabelSetValue, TimestampValue)):
                assert feature.unit is None, feature.name


# --------------------------------------------------------------------------- #
# Temporal behaviour
# --------------------------------------------------------------------------- #


class TestTemporalFeatures:
    def test_no_timestamps_at_all(self):
        records = [make_observation("t-1"), make_observation("t-2")]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "timestamp_count") == 0
        assert value_of(features, "timestamp_state") == TimestampState.NONE.value
        for name in (
            "first_seen",
            "last_seen",
            "event_time_span_seconds",
            "active_duration_seconds",
            "events_per_day",
            "naive_first_seen",
            "naive_last_seen",
            "naive_time_span_seconds",
        ):
            assert status_of(features, name) is MeasurementStatus.ABSENT, name

    def test_one_aware_timestamp(self):
        features = BehaviorFeatureExtractor().extract_observations(
            [make_observation("t-1", timestamp=T0)]
        )
        assert value_of(features, "timestamp_state") == TimestampState.AWARE_ONLY.value
        assert value_of(features, "first_seen") == T0
        assert value_of(features, "last_seen") == T0
        assert value_of(features, "aware_timestamp_count") == 1
        # A single instant really does span zero seconds; that is a
        # measurement, not a missing one.
        assert value_of(features, "event_time_span_seconds") == Decimal("0")
        assert status_of(features, "event_time_span_seconds") is MeasurementStatus.OBSERVED
        # A rate needs an interval, so there is none to report.
        assert status_of(features, "events_per_day") is MeasurementStatus.ABSENT

    def test_two_aware_timestamps_at_the_same_instant_do_not_invent_a_rate(self):
        features = BehaviorFeatureExtractor().extract_observations(
            [make_observation("t-1", timestamp=T0), make_observation("t-2", timestamp=T0)]
        )
        assert value_of(features, "event_time_span_seconds") == Decimal("0")
        assert status_of(features, "events_per_day") is MeasurementStatus.ABSENT

    def test_several_aware_timestamps(self):
        records = [
            make_observation("t-1", timestamp=T2),
            make_observation("t-2", timestamp=T0),
            make_observation("t-3", timestamp=T1),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "first_seen") == T0
        assert value_of(features, "last_seen") == T2
        assert value_of(features, "event_time_span_seconds") == DAY.total_seconds()
        assert value_of(features, "active_duration_seconds") == DAY.total_seconds()
        # 3 events across exactly 1 day.
        assert value_of(features, "events_per_day") == Decimal("3")
        assert status_of(features, "first_seen") is MeasurementStatus.OBSERVED

    def test_spans_are_exact_decimal_seconds(self):
        """1.1 seconds must not come back as a binary float approximation."""
        records = [
            make_observation("t-1", timestamp=T0),
            make_observation("t-2", timestamp=T0 + timedelta(seconds=1, microseconds=100000)),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        span = as_map(features)["event_time_span_seconds"].value.value
        assert isinstance(span, Decimal)
        assert span == Decimal("1.1")
        assert span.as_tuple().exponent <= -1

    def test_naive_timestamps_alone(self):
        records = [
            make_observation("t-1", timestamp=datetime(2024, 5, 1, 12, 0)),
            make_observation("t-2", timestamp=datetime(2024, 5, 1, 12, 0, 30)),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "timestamp_state") == TimestampState.NAIVE_ONLY.value
        assert value_of(features, "naive_timestamp_count") == 2
        # No UTC instant may be claimed from a naive timestamp.
        for name in ("first_seen", "last_seen", "event_time_span_seconds"):
            assert status_of(features, name) is MeasurementStatus.ABSENT, name
        naive_first = as_map(features)["naive_first_seen"].value
        assert naive_first.timestamp_kind is TimestampKind.NAIVE
        assert naive_first.value == datetime(2024, 5, 1, 12, 0)
        assert value_of(features, "naive_time_span_seconds") == Decimal("30")

    def test_mixed_timestamps_keep_the_two_apart(self):
        records = [
            make_observation("t-1", timestamp=T0),
            make_observation("t-2", timestamp=datetime(2024, 5, 1, 9, 0)),
            make_observation("t-3", timestamp=T2),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "timestamp_state") == TimestampState.MIXED.value
        assert value_of(features, "aware_timestamp_count") == 2
        assert value_of(features, "naive_timestamp_count") == 1
        # The aware-derived measurements are marked partial, and say how many
        # records they left out rather than quietly ignoring them.
        for name in ("first_seen", "last_seen", "event_time_span_seconds", "events_per_day"):
            feature = as_map(features)[name]
            assert feature.status is MeasurementStatus.PARTIAL, name
            assert feature.value.excluded_count == 1, name
        assert value_of(features, "first_seen") == T0
        assert value_of(features, "naive_first_seen") == datetime(2024, 5, 1, 9, 0)

    def test_the_offset_is_settled_by_the_locked_layer_and_not_here(self):
        """A +05:30 instant is normalised once, by the layer that owns the record.

        ``UnifiedObservation`` converts an aware timestamp to UTC and leaves a
        naive one alone, so by the time a record arrives here an aware
        timestamp is already UTC and a naive one is still ambiguous.  What this
        layer must not do is make a *second* decision about the offset, so the
        feature carries the record's own datetime, the same object, unchanged.
        """
        offset = timezone(timedelta(hours=5, minutes=30))
        moment = datetime(2024, 5, 1, 17, 30, tzinfo=offset)
        record = make_observation("t-1", timestamp=moment)
        assert record.timestamp == moment.astimezone(UTC)
        features = BehaviorFeatureExtractor().extract_observations([record])
        first_seen = as_map(features)["first_seen"]
        assert first_seen.value.value is record.timestamp
        assert first_seen.to_dict()["value"]["value"] == record.timestamp.isoformat()

    def test_no_offset_is_converted_anywhere_in_this_package(self):
        """A silent second normalisation would move evidence by a hidden amount."""
        package = Path(behavior.__file__).parent
        offenders = []
        for source in sorted(package.glob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in {
                    "astimezone",
                    "localize",
                }:
                    offenders.append("%s:%d" % (source.name, node.lineno))
        assert offenders == []

    def test_the_sub_object_timestamp_is_not_counted_twice(self):
        """The layer copies repeat the same instant; counting them inflates."""
        record = make_observation(
            "t-1", timestamp=T0, src_ip="10.0.0.1", txid="a" * 64
        )
        assert record.network.timestamp == T0
        assert record.blockchain.timestamp == T0
        features = BehaviorFeatureExtractor().extract_observations([record])
        assert value_of(features, "timestamp_count") == 1
        assert value_of(features, "events_per_day") is None or status_of(
            features, "events_per_day"
        ) is MeasurementStatus.ABSENT

    def test_contributors_cover_only_the_records_carrying_a_time(self):
        records = [
            make_observation("t-1", timestamp=T0),
            make_observation("t-2"),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert as_map(features)["first_seen"].origin.observation_ids == ("t-1",)


# --------------------------------------------------------------------------- #
# Amounts
# --------------------------------------------------------------------------- #


class TestAmountFeatures:
    def test_amount_counts_and_totals(self):
        records = [
            make_observation(
                "a-1", input_amounts=["0.5", "0.25"], output_amounts=["0.7"], fee="0.0001"
            ),
            make_observation("a-2", input_amounts=["0.5"], output_amounts=["0.1"]),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "input_amount_count") == 3
        assert value_of(features, "output_amount_count") == 2
        assert value_of(features, "amount_count") == 5
        assert value_of(features, "fee_amount_count") == 1
        assert value_of(features, "total_input_amount") == Decimal("1.25")
        assert value_of(features, "total_output_amount") == Decimal("0.8")
        assert value_of(features, "total_fee_amount") == Decimal("0.0001")

    def test_summary_statistics_are_exact_decimals(self):
        records = [
            make_observation("a-1", input_amounts=["0.1", "0.2", "0.3"]),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        for name in ("minimum_amount", "maximum_amount", "mean_amount", "median_amount"):
            value = as_map(features)[name].value.value
            assert isinstance(value, Decimal), name
        assert value_of(features, "minimum_amount") == Decimal("0.1")
        assert value_of(features, "maximum_amount") == Decimal("0.3")
        assert value_of(features, "median_amount") == Decimal("0.2")
        assert value_of(features, "mean_amount") == Decimal("0.2")

    def test_a_third_keeps_its_precision_rather_than_rounding_to_a_float(self):
        """One bitcoin split three ways has no exact decimal rendering.

        At the pinned precision of 34 significant digits the mean is 0.333...
        carried out to the last digit the context allows.  Two things are being
        shown at once: the division stays in :class:`~decimal.Decimal` (a float
        would give 0.3333333333333333, sixteen digits and already wrong), and
        the answer does not depend on whatever decimal context the caller left
        installed -- the last line compares against a division performed at the
        ambient default precision of 28, which is a different number.
        """
        records = [make_observation("a-1", input_amounts=["1", "0", "0"])]
        features = BehaviorFeatureExtractor().extract_observations(records)
        mean = as_map(features)["mean_amount"].value.value
        assert str(mean).startswith("0.333333333333333333333333333333333")
        assert mean != Decimal(1) / Decimal(3)

    def test_an_even_count_median_averages_the_middle_pair(self):
        records = [make_observation("a-1", input_amounts=["1", "2"])]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "median_amount") == Decimal("1.5")

    def test_inputs_and_outputs_are_told_apart_in_the_totals(self):
        records = [make_observation("a-1", input_amounts=["3"], output_amounts=["1"])]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "total_input_amount") == Decimal("3")
        assert value_of(features, "total_output_amount") == Decimal("1")

    def test_no_amounts_is_absent_rather_than_zero(self):
        """A fabricated zero would be a claim that money did not move."""
        features = BehaviorFeatureExtractor().extract_observations(
            [make_observation("a-1", src_ip="10.0.0.1")]
        )
        for name in (
            "total_input_amount",
            "total_output_amount",
            "total_fee_amount",
            "minimum_amount",
            "maximum_amount",
            "mean_amount",
            "median_amount",
        ):
            feature = as_map(features)[name]
            assert feature.status is MeasurementStatus.ABSENT, name
            assert feature.value.value is None, name
        # The counts, by contrast, are real measurements of zero.
        assert value_of(features, "amount_count") == 0
        assert status_of(features, "amount_count") is MeasurementStatus.OBSERVED

    def test_a_zero_amount_is_observed_and_is_not_absence(self):
        features = BehaviorFeatureExtractor().extract_observations(
            [make_observation("a-1", input_amounts=["0"], output_amounts=["0"])]
        )
        for name in ("total_input_amount", "minimum_amount", "maximum_amount", "median_amount"):
            feature = as_map(features)[name]
            assert feature.status is MeasurementStatus.OBSERVED, name
            assert feature.value.value == Decimal("0"), name
        assert value_of(features, "mean_amount") == Decimal("0")
        assert value_of(features, "amount_count") == 2

    def test_amounts_cover_inputs_and_outputs_together(self):
        records = [
            make_observation("a-1", input_amounts=["1"], output_amounts=["9"])
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "minimum_amount") == Decimal("1")
        assert value_of(features, "maximum_amount") == Decimal("9")
        assert value_of(features, "amount_count") == 2

    def test_every_amount_declares_which_statistic_it_is(self):
        features = BehaviorFeatureExtractor().extract_observations(synthetic(2))
        money = [f for f in features if f.unit == "BTC"]
        assert {f.value.amount_kind for f in money} == {
            AmountKind.TOTAL_INPUT,
            AmountKind.TOTAL_OUTPUT,
            AmountKind.TOTAL_FEE,
            AmountKind.MINIMUM,
            AmountKind.MAXIMUM,
            AmountKind.MEAN,
            AmountKind.MEDIAN,
        }

    def test_no_price_or_fiat_figure_is_introduced(self):
        """A Bitcoin amount is only ever compared with Bitcoin amounts.

        There is no exchange rate anywhere in this layer, so no amount here can
        be in a currency whose value this system has no evidence for.  The unit
        list is pinned so that adding one would fail loudly.
        """
        records = synthetic(3)
        graph = build_graph_for(records)
        features = BehaviorFeatureExtractor(graph).extract_observations(records)
        units = {feature.unit for feature in features}
        assert units == {
            None,
            "BTC",
            "amounts",
            "distinct addresses",
            "distinct autonomous system numbers",
            "distinct correlation rules",
            "distinct country codes",
            "distinct destination IP addresses",
            "distinct destination ports",
            "distinct evidence types",
            "distinct ingestion runs",
            "distinct input addresses",
            "distinct output addresses",
            "distinct provenance citations",
            "distinct record locators",
            "distinct source IP addresses",
            "distinct source files",
            "distinct source ports",
            "distinct transaction ids",
            "event times",
            "event times with a timezone",
            "event times without a timezone",
            "events/day",
            "fees",
            "input address appearances",
            "observations",
            "observations carrying blockchain data",
            "observations carrying network data",
            "observations carrying transaction data",
            "observations in the graph",
            "observations named to this set by a recorded relationship",
            "observations typed BLOCKCHAIN",
            "observations typed COMBINED",
            "observations typed NETWORK",
            "observations typed UNKNOWN",
            "output address appearances",
            "recorded relationships in the graph",
            "recorded relationships naming this set",
            "recorded relationships naming this set as source",
            "recorded relationships naming this set as target",
            "recorded relationships with both sides inside this set",
            "seconds",
            "transaction id appearances",
        }
        for feature in features:
            rendered = feature.to_dict()["value"]["value"]
            assert not isinstance(rendered, float)
            assert "usd" not in str(feature.unit or "").lower()
            assert "fiat" not in str(feature.unit or "").lower()

    def test_amount_contributors_are_the_records_that_carried_amounts(self):
        records = [
            make_observation("a-1", input_amounts=["1"]),
            make_observation("a-2", src_ip="10.0.0.1"),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert as_map(features)["total_input_amount"].origin.observation_ids == ("a-1",)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class TestNetworkFeatures:
    def test_source_and_destination_addresses_stay_apart(self):
        records = [
            make_observation("n-1", src_ip="10.0.0.1", dst_ip="10.0.0.2"),
            make_observation("n-2", src_ip="10.0.0.1", dst_ip="10.0.0.3"),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_src_ip_count") == 1
        assert value_of(features, "unique_dst_ip_count") == 2
        assert value_of(features, "src_ip_values") == ("10.0.0.1",)
        assert value_of(features, "dst_ip_values") == ("10.0.0.2", "10.0.0.3")

    def test_the_same_address_on_both_sides_counts_in_each(self):
        records = [make_observation("n-1", src_ip="10.0.0.1", dst_ip="10.0.0.1")]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_src_ip_count") == 1
        assert value_of(features, "unique_dst_ip_count") == 1

    def test_repeated_addresses_are_distinctly_counted(self):
        records = [
            make_observation("n-%d" % index, src_ip="10.0.0.1") for index in range(4)
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_src_ip_count") == 1
        assert value_of(features, "network_observation_count") == 4

    def test_ports_are_counted_per_side(self):
        records = [
            make_observation("n-1", src_ip="10.0.0.1", src_port=443, dst_port=80),
            make_observation("n-2", src_ip="10.0.0.2", src_port=443, dst_port=8080),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_src_port_count") == 1
        assert value_of(features, "unique_dst_port_count") == 2
        assert value_of(features, "src_port_values") == ("443",)
        assert value_of(features, "dst_port_values") == ("80", "8080")

    def test_asns_are_counted_distinctly(self):
        records = [
            make_observation("n-1", src_ip="10.0.0.1", asn="AS15169"),
            make_observation("n-2", src_ip="10.0.0.2", asn=15169),
            make_observation("n-3", src_ip="10.0.0.3", asn=64500),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_asn_count") == 2
        assert value_of(features, "asn_values") == ("15169", "64500")

    def test_countries_are_compared_by_the_locked_normalisation(self):
        records = [
            make_observation("n-1", src_ip="10.0.0.1", geo_country="in"),
            make_observation("n-2", src_ip="10.0.0.2", geo_country="IN"),
            make_observation("n-3", src_ip="10.0.0.3", geo_country=" United States "),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_country_count") == 2
        assert value_of(features, "country_values") == ("IN", "UNITED STATES")

    def test_a_preserved_country_without_an_address_is_still_counted(self):
        records = [make_observation("n-1", geo_country="IN", observation_type=ObservationType.NETWORK)]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_country_count") == 1
        assert value_of(features, "unique_src_ip_count") == 0

    def test_no_network_data_at_all(self):
        features = BehaviorFeatureExtractor().extract_observations([make_observation("n-1")])
        assert value_of(features, "unique_src_ip_count") == 0
        assert value_of(features, "src_ip_values") == ()
        assert status_of(features, "unique_src_ip_count") is MeasurementStatus.OBSERVED

    def test_an_address_is_described_and_never_interpreted(self):
        """The distinct values are published; nothing is concluded from them."""
        records = [
            make_observation("n-1", src_ip="10.0.0.1"),
            make_observation("n-2", src_ip="10.0.0.1"),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        src = as_map(features)["unique_src_ip_count"]
        assert src.value.value == 1
        assert src.value.unit == "distinct source IP addresses"
        assert not hasattr(src.value, "confidence")
        assert not hasattr(src.value, "interpretation")


# --------------------------------------------------------------------------- #
# Addresses
# --------------------------------------------------------------------------- #


class TestAddressFeatures:
    def test_inputs_and_outputs_are_told_apart(self):
        records = [
            make_observation(
                "a-1", input_addresses=["bc1qaaa"], output_addresses=["bc1qbbb"]
            ),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_input_address_count") == 1
        assert value_of(features, "unique_output_address_count") == 1
        assert value_of(features, "input_address_values") == ("bc1qaaa",)
        assert value_of(features, "output_address_values") == ("bc1qbbb",)

    def test_the_union_covers_both_sides(self):
        records = [
            make_observation(
                "a-1", input_addresses=["bc1qaaa"], output_addresses=["bc1qbbb"]
            )
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_address_count") == 2
        assert value_of(features, "address_values") == ("bc1qaaa", "bc1qbbb")

    def test_an_address_on_both_sides_is_one_distinct_address(self):
        records = [
            make_observation(
                "a-1", input_addresses=["bc1qaaa"], output_addresses=["bc1qaaa"]
            )
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_address_count") == 1
        assert value_of(features, "input_address_occurrences") == 1
        assert value_of(features, "output_address_occurrences") == 1

    def test_occurrences_count_repetitions_and_distinct_does_not(self):
        records = [
            make_observation(
                "a-%d" % index, input_addresses=["bc1qaaa", "bc1qaaa", "bc1qbbb"]
            )
            for index in range(2)
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "input_address_occurrences") == 6
        assert value_of(features, "unique_input_address_count") == 2

    @pytest.mark.parametrize("prefix", ["bc1", "tb1", "bcrt1"])
    def test_bech32_addresses_fold_case(self, prefix):
        upper = prefix + "QWERTY"
        records = [
            make_observation("a-1", input_addresses=[upper]),
            make_observation("a-2", input_addresses=[upper.lower()]),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_input_address_count") == 1
        assert value_of(features, "input_address_values") == (prefix + "qwerty",)

    def test_base58_case_is_preserved_and_makes_two_addresses(self):
        lower = "1bvbmsEyStWetqTFn5Au4m4GFg7xJaNVn2"
        upper = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
        assert lower != upper
        records = [
            make_observation("a-1", input_addresses=[upper]),
            make_observation("a-2", input_addresses=[lower]),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        # Folding base58 case would manufacture a match between two addresses
        # that are, as far as the evidence shows, different.
        assert value_of(features, "unique_input_address_count") == 2

    def test_surrounding_whitespace_is_trimmed(self):
        records = [make_observation("a-1", input_addresses=["  bc1qaaa  "])]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "input_address_values") == ("bc1qaaa",)

    def test_an_address_is_published_and_nothing_is_claimed_about_it(self):
        features = BehaviorFeatureExtractor().extract_observations(
            [make_observation("a-1", input_addresses=["bc1qaaa"])]
        )
        feature = as_map(features)["input_address_values"]
        assert feature.value.value == ("bc1qaaa",)
        assert feature.unit is None
        assert not any("owner" in name for name in feature.to_dict())


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


class TestProvenanceFeatures:
    def test_distinct_ingestions_files_and_records(self):
        records = [
            make_observation(
                "p-1", ingestion_id="ing-a", source_file="a.csv", source_record="row 1"
            ),
            make_observation(
                "p-2", ingestion_id="ing-a", source_file="a.csv", source_record="row 1"
            ),
            make_observation(
                "p-3", ingestion_id="ing-b", source_file="b.csv", source_record="row 9"
            ),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_ingestion_count") == 2
        assert value_of(features, "ingestion_id_values") == ("ing-a", "ing-b")
        assert value_of(features, "unique_source_file_count") == 2
        assert value_of(features, "source_file_values") == ("a.csv", "b.csv")
        assert value_of(features, "unique_source_record_count") == 2
        # Three records, but only two distinct (ingestion, file, record)
        # triples: p-1 and p-2 name the same source record.  Counting the
        # records instead of the references would report provenance diversity
        # that the sources do not have.
        assert value_of(features, "provenance_reference_count") == 2

    def test_a_single_source_is_diversity_of_one(self):
        records = [make_observation("p-1", ingestion_id="ing-a", source_file="a.csv")]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_ingestion_count") == 1
        assert value_of(features, "provenance_reference_count") == 1

    def test_contributors_carry_the_locators_they_were_read_from(self):
        records = [
            make_observation(
                "p-1", ingestion_id="ing-a", source_file="a.csv", source_record="row 1"
            ),
            make_observation("p-2", ingestion_id="ing-b", source_file="b.csv"),
        ]
        features = BehaviorFeatureExtractor().extract_observations(records)
        origin = as_map(features)["unique_ingestion_count"].origin
        assert origin.observation_ids == ("p-1", "p-2")
        assert origin.ingestion_ids == ("ing-a", "ing-b")
        assert origin.source_files == ("a.csv", "b.csv")

    def test_a_record_with_no_source_file_contributes_nothing_to_it(self):
        records = [make_observation("p-1", source_file=None)]
        features = BehaviorFeatureExtractor().extract_observations(records)
        assert value_of(features, "unique_source_file_count") == 0
        assert value_of(features, "source_file_values") == ()


# --------------------------------------------------------------------------- #
# Evidence graph
# --------------------------------------------------------------------------- #


class TestEvidenceGraphFeatures:
    def test_without_a_graph_the_structural_features_are_absent(self):
        features = BehaviorFeatureExtractor().extract_observations(synthetic(3))
        for name in (
            "graph_node_count",
            "graph_edge_count",
            "incident_observation_degree",
            "incoming_evidence_edge_count",
            "outgoing_evidence_edge_count",
            "unique_evidence_type_count",
            "unique_rule_count",
        ):
            assert status_of(features, name) is MeasurementStatus.ABSENT, name

    def test_an_empty_graph_is_observed_as_zero(self):
        """Zero recorded relationships is not the same as no graph supplied."""
        graph = build_graph_for([])
        features = BehaviorFeatureExtractor(graph).extract_observations(
            [make_observation("g-1")]
        )
        assert value_of(features, "graph_node_count") == 0
        assert value_of(features, "graph_edge_count") == 0
        assert value_of(features, "incident_observation_degree") == 0
        assert status_of(features, "graph_node_count") is MeasurementStatus.OBSERVED

    def test_descriptors_come_from_the_locked_correlation_layer(self):
        records = [
            make_observation("g-a", src_ip="10.0.0.1", timestamp=T0),
            make_observation("g-b", src_ip="10.0.0.1", timestamp=T0 + DAY),
        ]
        graph = build_graph_for(records)
        assert graph.edge_count() > 0
        features = BehaviorFeatureExtractor(graph).extract_observations(records)
        assert value_of(features, "graph_node_count") == graph.node_count()
        assert value_of(features, "graph_edge_count") == graph.edge_count()
        assert value_of(features, "evidence_relationship_count") == graph.edge_count()
        assert value_of(features, "internal_evidence_relationship_count") == graph.edge_count()

    def test_incident_degree_names_the_observations_on_the_far_side(self):
        records = [
            make_observation("g-a", src_ip="10.0.0.1", timestamp=T0),
            make_observation("g-b", src_ip="10.0.0.1", timestamp=T0 + DAY),
        ]
        graph = build_graph_for(records)
        features = BehaviorFeatureExtractor(graph).extract_observations([records[0]])
        assert value_of(features, "incident_observation_values") == ("g-b",)
        assert value_of(features, "incident_observation_degree") == 1
        assert value_of(features, "internal_evidence_relationship_count") == 0

    def test_a_whole_component_has_no_external_neighbours(self):
        records = [
            make_observation("g-a", src_ip="10.0.0.1", timestamp=T0),
            make_observation("g-b", src_ip="10.0.0.1", timestamp=T0 + DAY),
        ]
        graph = build_graph_for(records)
        features = BehaviorFeatureExtractor(graph).extract_observations(records)
        assert value_of(features, "incident_observation_degree") == 0
        assert value_of(features, "incident_observation_values") == ()

    def test_evidence_types_and_rules_are_diversity_descriptors(self):
        records = [
            make_observation("g-a", src_ip="10.0.0.1", input_addresses=["bc1qaaa"]),
            make_observation("g-b", src_ip="10.0.0.1", input_addresses=["bc1qaaa"]),
        ]
        graph = build_graph_for(records)
        features = BehaviorFeatureExtractor(graph).extract_observations(records)
        types = value_of(features, "evidence_type_values")
        rules = value_of(features, "rule_id_values")
        assert "SAME_IP" in types
        assert "SAME_ADDRESS" in types
        assert "same_ip" in rules
        assert value_of(features, "unique_evidence_type_count") == len(types)
        assert value_of(features, "unique_rule_count") == len(rules)

    def test_incoming_and_outgoing_counts_describe_the_sides(self):
        records = [
            make_observation("g-a", src_ip="10.0.0.1", timestamp=T0),
            make_observation("g-b", src_ip="10.0.0.1", timestamp=T0 + DAY),
        ]
        graph = build_graph_for(records)
        features = BehaviorFeatureExtractor(graph).extract_observations(records)
        assert value_of(features, "outgoing_evidence_edge_count") == graph.edge_count()
        assert value_of(features, "incoming_evidence_edge_count") == graph.edge_count()

    def test_relationship_citations_reach_the_profile(self):
        records = [
            make_observation("g-a", src_ip="10.0.0.1"),
            make_observation("g-b", src_ip="10.0.0.1"),
        ]
        graph = build_graph_for(records)
        features = BehaviorFeatureExtractor(graph).extract_observations(records)
        cited = as_map(features)["unique_rule_count"].origin.relationship_ids
        assert cited == tuple(sorted(edge.relationship_id for edge in graph.edges()))

    def test_citations_span_observations_and_relationships(self):
        records = [
            make_observation("g-a", src_ip="10.0.0.1", ingestion_id="ing-a"),
            make_observation("g-b", src_ip="10.0.0.1", ingestion_id="ing-a"),
        ]
        graph = build_graph_for(records)
        without_graph = BehaviorFeatureExtractor().extract_observations(records)
        with_graph = BehaviorFeatureExtractor(graph).extract_observations(records)
        # Both records come from one ingestion and name two different source
        # records, so there are two distinct provenance references.  The
        # relationships cite those same two references, so handing the graph
        # over adds no new ones.
        assert value_of(without_graph, "provenance_reference_count") == 2
        assert value_of(with_graph, "provenance_reference_count") == 2
        # What the graph does add is the relationship citations themselves.
        assert as_map(without_graph)["unique_rule_count"].origin.relationship_ids == ()
        assert as_map(with_graph)["unique_rule_count"].origin.relationship_ids == tuple(
            sorted(edge.relationship_id for edge in graph.edges())
        )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def test_shuffling_the_input_changes_nothing(self):
        records = synthetic(30)
        extractor = BehaviorFeatureExtractor()
        reference = [f.to_dict() for f in extractor.extract_observations(records)]
        for seed in range(5):
            shuffled = list(records)
            # A fixed rotation: no randomness, so a failure is reproducible.
            shuffled = shuffled[seed:] + shuffled[:seed]
            assert [
                f.to_dict() for f in extractor.extract_observations(shuffled)
            ] == reference

    def test_repeating_the_extraction_gives_an_equal_result(self):
        records = synthetic(12)
        extractor = BehaviorFeatureExtractor()
        first = extractor.extract_observations(records)
        second = extractor.extract_observations(records)
        assert first == second
        assert [f.feature_id for f in first] == [f.feature_id for f in second]

    def test_a_fresh_extractor_agrees_with_a_reused_one(self):
        records = synthetic(12)
        reused = BehaviorFeatureExtractor()
        reused.extract_observations(records)
        assert reused.extract_observations(records) == (
            BehaviorFeatureExtractor().extract_observations(records)
        )

    def test_measurement_order_does_not_reach_the_values(self):
        records = [
            make_observation("d-1", input_amounts=["0.1", "0.9"]),
            make_observation("d-2", input_amounts=["0.5"]),
        ]
        forward = BehaviorFeatureExtractor().extract_observations(records)
        backward = BehaviorFeatureExtractor().extract_observations(list(reversed(records)))
        for name in ("minimum_amount", "maximum_amount", "mean_amount", "median_amount"):
            assert value_of(forward, name) == value_of(backward, name), name

    def test_the_serialised_form_is_byte_stable(self):
        import json

        records = synthetic(8)
        extractor = BehaviorFeatureExtractor()
        first = json.dumps([f.to_dict() for f in extractor.extract_observations(records)])
        second = json.dumps([f.to_dict() for f in extractor.extract_observations(records)])
        assert first == second


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #


class TestModerateSyntheticDataset:
    def test_a_thousand_records_stay_deterministic(self):
        records = synthetic(1000)
        extractor = BehaviorFeatureExtractor()
        started = time.perf_counter()
        features = extractor.extract_observations(records)
        elapsed = time.perf_counter() - started
        assert len(features) == EXPECTED_FEATURE_COUNT
        assert value_of(features, "observation_count") == 1000
        assert value_of(features, "unique_src_ip_count") == 5 * 17
        # Generous, because this asserts "does not fall over", not "is fast".
        assert elapsed < 20.0, "extraction of 1000 records took %.1fs" % elapsed

    def test_the_cost_grows_with_the_data_rather_than_its_square(self):
        """Feature extraction is linear in records plus relevant edges.

        The bound is loose on purpose: a squaring implementation would be far
        beyond it, while a linear one has ample room for a noisy measurement.
        """
        extractor = BehaviorFeatureExtractor()

        def timed(count: int) -> float:
            records = synthetic(count)
            started = time.perf_counter()
            extractor.extract_observations(records)
            return time.perf_counter() - started

        timed(200)  # warm up so first-call import cost is not measured
        small = timed(200)
        large = timed(800)
        assert large < max(small * 12.0, 1.0), (
            "800 records took %.3fs against 0.6s for 200" % large
        )

    def test_a_large_set_shuffles_to_the_same_answer(self):
        records = synthetic(400)
        extractor = BehaviorFeatureExtractor()
        reference = [f.to_dict() for f in extractor.extract_observations(records)]
        reordered = records[200:] + records[:200]
        assert [
            f.to_dict() for f in extractor.extract_observations(reordered)
        ] == reference


# --------------------------------------------------------------------------- #
# Vocabulary of the extractor itself
# --------------------------------------------------------------------------- #


class TestExtractorSurface:
    def test_the_public_surface_is_small_and_documented(self):
        """Two entry points and a read-only view of the graph, and nothing else.

        A small surface is not austerity for its own sake: every extra public
        name is a thing a caller can reach for later, and a behavioural layer
        should offer as few of those as it can get away with.
        """
        public = {
            name
            for name in dir(BehaviorFeatureExtractor)
            if not name.startswith("_")
        }
        assert public == {"graph", "extract_observation", "extract_observations"}
        assert BehaviorFeatureExtractor.extract_observation.__doc__
        assert BehaviorFeatureExtractor.extract_observations.__doc__

    def test_no_extractor_method_reads_an_evaluative_field(self):
        """Nothing here could produce a judgement even by accident."""
        features = BehaviorFeatureExtractor().extract_observations(synthetic(4))
        for feature in features:
            rendered = str(feature.to_dict()).lower()
            for word in ("suspicious", "malicious", "fraud", "risky", "attacker"):
                assert word not in rendered

    def test_the_error_type_is_a_value_error(self):
        assert issubclass(BehaviorFeatureError, ValueError)

    def test_counts_are_count_values_and_amounts_are_decimal_values(self):
        features = BehaviorFeatureExtractor().extract_observations(synthetic(2))
        kinds = {f.name: f.value.kind for f in features}
        assert kinds["observation_count"] == "COUNT"
        assert kinds["first_seen"] == "TIMESTAMP"
        assert kinds["timestamp_state"] == "LABEL"
        assert kinds["src_ip_values"] == "LABEL_SET"
        assert kinds["median_amount"] == "DECIMAL"
        assert isinstance(
            as_map(features)["observation_count"].value, CountValue
        )
        assert isinstance(
            as_map(features)["median_amount"].value, DecimalValue
        )
        assert isinstance(
            as_map(features)["src_ip_values"].value, LabelSetValue
        )
