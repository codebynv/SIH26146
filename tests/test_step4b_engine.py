"""Tests for the Step 4B deterministic correlation engine.

The engine owns the guarantees that hold *across* rules rather than inside one:
grouping shape, merging, ordering, provenance retention, input immutability and
determinism.  Those are what this file checks, plus two whole-pipeline
properties that unit tests cannot establish -- exact relationship counts on a
large synthetic corpus, and semantic equivalence against a brute-force
reference implementation.

Almost all fixtures are :class:`types.SimpleNamespace` objects so a failure
points at the engine rather than at Step 3.  :class:`TestRealSubstrate` builds
genuine Step 3 models to confirm the structural contract holds for the actual
:class:`~backend.app.domain.models.UnifiedObservation` the system will use.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import pytest

from backend.app.correlation.engine import CorrelationEngine
from backend.app.correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    InferenceLevel,
    TemporalRelation,
)
from backend.app.correlation.rules import (
    CorrelationConfig,
    CorrelationRule,
    SameIpRule,
    normalize_address,
    normalize_country,
    normalize_ip,
    normalize_txid,
)

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)
FROZEN_NOW = datetime(2024, 6, 1, 9, 0, 0, tzinfo=UTC)

#: Words that would turn an observation-level match into a claim about a person,
#: a system or a motive.  No generated rationale may contain one.
FORBIDDEN_CLAIM_WORDS = (
    "ownership",
    "owner",
    "person",
    "actor",
    "entity",
    "fraud",
    "guilt",
    "suspicious",
    "malicious",
    "attacker",
    "attribution",
    "identity",
)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def make_observation(
    observation_id: str,
    *,
    txid: Optional[str] = None,
    src_ip: Any = None,
    dst_ip: Any = None,
    geo_country: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    input_addresses: Iterable[str] = (),
    output_addresses: Iterable[str] = (),
    input_amounts: Iterable[Any] = (),
    output_amounts: Iterable[Any] = (),
    fee: Optional[Any] = None,
    ingestion_id: Optional[str] = None,
    source_file: Optional[str] = None,
    source_record: Optional[str] = None,
) -> SimpleNamespace:
    """Build a duck-typed observation exposing the attributes rules may read.

    Provenance defaults to a triple unique to *observation_id*.  A shared
    default would make every pair in a test match ``provenance_link`` as well,
    quietly inflating counts that a test means to be about something else, so
    sharing a triple has to be asked for explicitly.
    """
    inputs = list(input_addresses)
    outputs = list(output_addresses)
    in_amounts = [Decimal(str(value)) for value in input_amounts]
    out_amounts = [Decimal(str(value)) for value in output_amounts]

    blockchain = None
    if (
        txid is not None
        or fee is not None
        or inputs
        or outputs
        or in_amounts
        or out_amounts
    ):
        blockchain = SimpleNamespace(
            txid=txid,
            input_addresses=inputs,
            output_addresses=outputs,
            input_amounts=in_amounts,
            output_amounts=out_amounts,
            fee=None if fee is None else Decimal(str(fee)),
        )

    network = None
    if any(value is not None for value in (src_ip, dst_ip, geo_country)):
        network = SimpleNamespace(
            src_ip=src_ip, dst_ip=dst_ip, geo_country=geo_country
        )

    return SimpleNamespace(
        observation_id=observation_id,
        timestamp=timestamp,
        network=network,
        blockchain=blockchain,
        provenance=SimpleNamespace(
            ingestion_id=ingestion_id if ingestion_id is not None else "ing-%s" % observation_id,
            source_file=source_file if source_file is not None else "records.csv",
            source_record=(
                source_record if source_record is not None else "row-%s" % observation_id
            ),
        ),
        # Deliberately mutable nested state: a correlation pass must leave it
        # byte-for-byte identical even though the surrounding model is frozen.
        extra={"mutable": {"nested": [1, 2, {"deep": "value"}]}},
    )


def make_engine(
    temporal_window: Optional[timedelta] = None,
    amount_tolerance: str = "0",
    now: datetime = FROZEN_NOW,
) -> CorrelationEngine:
    return CorrelationEngine(
        CorrelationConfig(
            temporal_window=temporal_window,
            amount_tolerance=Decimal(amount_tolerance),
            now=now,
        )
    )


def freeze(value: Any) -> Any:
    """Recursively render *value* into a comparable, order-stable structure."""
    if isinstance(value, dict):
        return ("map", tuple(sorted((repr(k), freeze(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(freeze(item) for item in value))
    if isinstance(value, set):
        return ("set", tuple(sorted(repr(freeze(item)) for item in value)))
    return ("leaf", repr(value))


def relationship_keys(relationships) -> set[tuple]:
    """The ``(rule, source, targets)`` identity of each relationship."""
    return {
        (item.rule_id, item.source_observation_ids, item.target_observation_ids)
        for item in relationships
    }


def all_pairs(relationships):
    """Every ``(rule_id, source, target)`` pairing a relationship asserts."""
    for item in relationships:
        for source in item.source_observation_ids:
            for target in item.target_observation_ids:
                yield item.rule_id, source, target


def all_rationales(relationships) -> list[str]:
    return [item.rationale for item in relationships]


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


class TestGrouping:
    def test_two_observations_produce_one_relationship(self):
        observations = [
            make_observation("obs-a", txid="a" * 64),
            make_observation("obs-b", txid="a" * 64),
        ]
        relationships = make_engine().run(observations)
        assert len(relationships) == 1
        assert relationships[0].source_observation_ids == ("obs-a",)
        assert relationships[0].target_observation_ids == ("obs-b",)

    def test_three_observations_produce_one_anchor_and_two_targets(self):
        """Grouping is deliberate: a 3-group is one relationship, not three pairs."""
        observations = [
            make_observation("obs-%s" % name, txid="a" * 64)
            for name in ("c", "a", "b")
        ]
        relationships = make_engine().run(observations)
        assert len(relationships) == 1
        relationship = relationships[0]
        assert relationship.source_observation_ids == ("obs-a",)
        assert relationship.target_observation_ids == ("obs-b", "obs-c")

    def test_the_anchor_is_the_lowest_observation_id(self):
        observations = [
            make_observation("obs-%02d" % index, txid="a" * 64)
            for index in (7, 3, 9, 1)
        ]
        relationship = make_engine().run(observations)[0]
        assert relationship.source_observation_ids == ("obs-01",)
        assert relationship.target_observation_ids == ("obs-03", "obs-07", "obs-09")

    def test_source_and_targets_are_always_disjoint(self):
        observations = [
            make_observation("obs-%02d" % index, txid="a" * 64, src_ip="192.0.2.1")
            for index in range(8)
        ]
        for relationship in make_engine(timedelta(seconds=5)).run(observations):
            assert not set(relationship.source_observation_ids) & set(
                relationship.target_observation_ids
            )
            assert not relationship.is_self_referential

    def test_targets_are_sorted(self):
        observations = [
            make_observation("obs-%s" % name, txid="a" * 64)
            for name in ("d", "b", "c", "a")
        ]
        relationship = make_engine().run(observations)[0]
        assert list(relationship.target_observation_ids) == sorted(
            relationship.target_observation_ids
        )

    def test_a_lone_observation_produces_nothing(self):
        assert make_engine().run([make_observation("obs-a", txid="a" * 64)]) == []

    def test_an_empty_batch_produces_nothing(self):
        assert make_engine().run([]) == []

    def test_temporal_matching_is_pairwise_not_grouped(self):
        """A distance is not a shared key, so there is nothing to group on."""
        observations = [
            make_observation("obs-%d" % index, timestamp=T0 + timedelta(seconds=index))
            for index in range(3)
        ]
        relationships = make_engine(timedelta(seconds=5)).run(observations)
        assert relationship_keys(relationships) == {
            ("temporal_adjacency", ("obs-0",), ("obs-1",)),
            ("temporal_adjacency", ("obs-0",), ("obs-2",)),
            ("temporal_adjacency", ("obs-1",), ("obs-2",)),
        }


# --------------------------------------------------------------------------- #
# Merging and duplicates
# --------------------------------------------------------------------------- #


class TestMerging:
    def test_one_rule_matching_two_keys_yields_one_relationship(self):
        """Same source IP *and* same destination IP is one link, with both facts."""
        observations = [
            make_observation("obs-a", src_ip="192.0.2.1", dst_ip="198.51.100.7"),
            make_observation("obs-b", src_ip="192.0.2.1", dst_ip="198.51.100.7"),
        ]
        relationships = make_engine().run(observations)
        assert len(relationships) == 1
        relationship = relationships[0]
        assert relationship.rule_id == "same_ip"
        assert "source IP address 192.0.2.1" in relationship.rationale
        assert "destination IP address 198.51.100.7" in relationship.rationale

    def test_merged_facts_are_ordered_regardless_of_discovery(self):
        observations = [
            make_observation("obs-a", src_ip="192.0.2.1", dst_ip="198.51.100.7"),
            make_observation("obs-b", src_ip="192.0.2.1", dst_ip="198.51.100.7"),
        ]
        rationales = {
            make_engine().run(observations)[0].rationale,
            make_engine().run(list(reversed(observations)))[0].rationale,
        }
        assert len(rationales) == 1

    def test_no_duplicate_relationship_keys(self):
        observations = [
            make_observation(
                "obs-%d" % index,
                txid="a" * 64,
                src_ip="192.0.2.1",
                geo_country="IN",
                input_addresses=["bc1qshared"],
                input_amounts=["1"],
            )
            for index in range(6)
        ]
        relationships = make_engine().run(observations)
        assert len(relationship_keys(relationships)) == len(relationships)

    def test_no_duplicate_relationship_ids(self):
        observations = [
            make_observation("obs-%d" % index, txid="a" * 64, src_ip="192.0.2.1")
            for index in range(6)
        ]
        relationships = make_engine().run(observations)
        identifiers = [item.relationship_id for item in relationships]
        assert len(set(identifiers)) == len(identifiers)

    def test_different_rules_produce_separate_relationships(self):
        observations = [
            make_observation("obs-a", txid="a" * 64, src_ip="192.0.2.1"),
            make_observation("obs-b", txid="a" * 64, src_ip="192.0.2.1"),
        ]
        assert {item.rule_id for item in make_engine().run(observations)} == {
            "same_txid",
            "same_ip",
        }

    def test_a_repeated_observation_does_not_correlate_with_itself(self):
        observation = make_observation("obs-a", txid="a" * 64, src_ip="192.0.2.1")
        assert make_engine().run([observation, observation]) == []

    def test_duplicate_observations_do_not_break_a_valid_match(self):
        """A repeated copy of one member must not drop the real relationship."""
        observations = [
            make_observation("obs-a", txid="a" * 64),
            make_observation("obs-b", txid="a" * 64),
        ]
        relationships = make_engine().run([observations[0], observations[0], observations[1]])
        assert relationship_keys(relationships) == {
            ("same_txid", ("obs-a",), ("obs-b",))
        }

    def test_rejects_an_observation_without_an_id(self):
        observation = make_observation("obs-a", txid="a" * 64)
        observation.observation_id = "  "
        with pytest.raises(ValueError, match="observation_id"):
            make_engine().run([observation])

    def test_rejects_a_none_observation(self):
        with pytest.raises(ValueError, match="None"):
            make_engine().run([None])

    def test_rejects_an_engine_with_no_rules(self):
        with pytest.raises(ValueError, match="at least one rule"):
            CorrelationEngine(CorrelationConfig(), rules=[])

    def test_rejects_a_rule_without_an_id(self):
        class Anonymous(CorrelationRule):
            """A rule that never named itself, as a mistake would."""

            def evaluate(self, observations):
                return iter(())

        with pytest.raises(ValueError, match="rule_id"):
            CorrelationEngine(CorrelationConfig(), rules=[Anonymous()])


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def _determinism_corpus() -> list[SimpleNamespace]:
    """A corpus that trips every rule, used by the determinism checks."""
    return [
        make_observation(
            "obs-a",
            txid="a" * 64,
            src_ip="192.0.2.1",
            dst_ip="198.51.100.7",
            geo_country="IN",
            input_addresses=["BC1QShared"],
            input_amounts=["1.5"],
            output_addresses=["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
            timestamp=T0,
        ),
        make_observation(
            "obs-b",
            txid="A" * 64,
            src_ip="192.0.2.1",
            dst_ip="198.51.100.7",
            geo_country="in",
            input_addresses=["bc1qshared"],
            input_amounts=["1.5"],
            output_addresses=["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
            timestamp=T0 + timedelta(seconds=3),
        ),
        make_observation(
            "obs-c",
            txid="b" * 64,
            src_ip="203.0.113.9",
            geo_country="US",
            input_addresses=["bc1qother"],
            input_amounts=["9"],
            timestamp=T0 + timedelta(hours=3),
        ),
    ]


class TestDeterminism:
    def test_repeated_execution_is_identical(self):
        corpus = _determinism_corpus()
        first = make_engine(timedelta(seconds=5)).run(corpus)
        second = make_engine(timedelta(seconds=5)).run(corpus)
        assert first == second

    def test_a_reused_engine_gives_the_same_answer(self):
        engine = make_engine(timedelta(seconds=5))
        corpus = _determinism_corpus()
        assert engine.run(corpus) == engine.run(corpus)

    @pytest.mark.parametrize("seed", [0, 1, 7, 99])
    def test_shuffling_the_input_changes_nothing(self, seed):
        corpus = _determinism_corpus()
        shuffled = list(corpus)
        random.Random(seed).shuffle(shuffled)
        assert make_engine(timedelta(seconds=5)).run(shuffled) == make_engine(
            timedelta(seconds=5)
        ).run(corpus)

    def test_output_is_sorted_by_rule_then_observations(self):
        relationships = make_engine(timedelta(seconds=5)).run(_determinism_corpus())
        keys = [
            (item.rule_id, item.source_observation_ids, item.target_observation_ids)
            for item in relationships
        ]
        assert keys == sorted(keys)

    def test_relationship_ids_do_not_depend_on_the_injected_clock(self):
        """A relationship recorded tomorrow is the same relationship."""
        early = make_engine(timedelta(seconds=5), now=FROZEN_NOW).run(_determinism_corpus())
        late = make_engine(
            timedelta(seconds=5), now=datetime(2030, 1, 1, tzinfo=UTC)
        ).run(_determinism_corpus())
        assert [item.relationship_id for item in early] == [
            item.relationship_id for item in late
        ]
        assert [item.created_at for item in early] != [item.created_at for item in late]

    def test_relationship_ids_are_uuid_shaped_and_stable(self):
        import uuid

        relationships = make_engine().run(_determinism_corpus())
        for relationship in relationships:
            assert uuid.UUID(relationship.relationship_id)

    def test_two_engines_with_equal_configuration_agree(self):
        corpus = _determinism_corpus()
        assert make_engine(timedelta(seconds=5)).run(corpus) == make_engine(
            timedelta(seconds=5)
        ).run(corpus)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


class TestProvenance:
    def test_provenance_is_carried_onto_the_relationship(self):
        observations = [
            make_observation(
                "obs-a", txid="a" * 64, ingestion_id="ing-1", source_file="a.csv", source_record="row 1"
            ),
            make_observation(
                "obs-b", txid="a" * 64, ingestion_id="ing-2", source_file="b.csv", source_record="row 7"
            ),
        ]
        relationship = make_engine().run(observations)[0]
        assert [ref.ingestion_id for ref in relationship.provenance_refs] == ["ing-1", "ing-2"]
        assert relationship.provenance_refs[1].source_file == "b.csv"
        assert relationship.provenance_refs[1].source_record == "row 7"

    def test_provenance_is_deduplicated(self):
        observations = [
            make_observation(
                "obs-%d" % index,
                txid="a" * 64,
                ingestion_id="ing-1",
                source_file="a.csv",
                source_record="row 1",
            )
            for index in range(4)
        ]
        relationship = make_engine().run(observations)[0]
        assert len(relationship.provenance_refs) == 1

    def test_provenance_is_sorted(self):
        observations = [
            make_observation(
                "obs-0", txid="a" * 64, ingestion_id="ing-c", source_file="a.csv", source_record="r1"
            ),
            make_observation(
                "obs-1", txid="a" * 64, ingestion_id="ing-a", source_file="a.csv", source_record="r1"
            ),
            make_observation(
                "obs-2", txid="a" * 64, ingestion_id="ing-b", source_file="a.csv", source_record="r1"
            ),
        ]
        relationship = make_engine().run(observations)[0]
        assert [ref.ingestion_id for ref in relationship.provenance_refs] == [
            "ing-a",
            "ing-b",
            "ing-c",
        ]

    def test_a_blank_locator_is_treated_as_absent(self):
        """A citation pointing nowhere is worse than no citation."""
        observations = [
            make_observation(
                "obs-a",
                txid="a" * 64,
                ingestion_id="ing-1",
                source_file="   ",
                source_record="row 1",
            ),
            make_observation(
                "obs-b",
                txid="a" * 64,
                ingestion_id="ing-1",
                source_file="   ",
                source_record="row 1",
            ),
        ]
        relationship = make_engine().run(observations)[0]
        assert len(relationship.provenance_refs) == 1
        assert relationship.provenance_refs[0].source_file is None

    def test_an_observation_without_usable_provenance_is_skipped(self):
        observations = [
            make_observation(
                "obs-a",
                txid="a" * 64,
                ingestion_id="  ",
                source_file="a.csv",
                source_record="row 1",
            ),
            make_observation(
                "obs-b",
                txid="a" * 64,
                ingestion_id="ing-2",
                source_file="a.csv",
                source_record="row 1",
            ),
        ]
        relationship = make_engine().run(observations)[0]
        assert [ref.ingestion_id for ref in relationship.provenance_refs] == ["ing-2"]

    def test_observations_without_a_provenance_object(self):
        observations = [
            make_observation("obs-a", txid="a" * 64),
            make_observation("obs-b", txid="a" * 64),
        ]
        for observation in observations:
            observation.provenance = None
        relationship = make_engine().run(observations)[0]
        assert relationship.provenance_refs == ()


# --------------------------------------------------------------------------- #
# Relationship fields
# --------------------------------------------------------------------------- #


class TestRelationshipFields:
    def test_every_relationship_is_correlated(self):
        for relationship in make_engine(timedelta(seconds=5)).run(_determinism_corpus()):
            assert relationship.inference_level is InferenceLevel.CORRELATED

    def test_rule_id_and_version_are_recorded(self):
        for relationship in make_engine(timedelta(seconds=5)).run(_determinism_corpus()):
            assert relationship.rule_id
            assert relationship.rule_version

    def test_evidence_type_agrees_with_the_rule(self):
        expected = {
            "same_txid": EvidenceType.SAME_TXID,
            "same_address": EvidenceType.SAME_ADDRESS,
            "same_ip": EvidenceType.SAME_IP,
            "temporal_adjacency": EvidenceType.TEMPORAL_ADJACENCY,
            "amount_match": EvidenceType.AMOUNT_MATCH,
            "provenance_link": EvidenceType.PROVENANCE_LINK,
            "geo_coincidence": EvidenceType.GEO_COINCIDENCE,
        }
        for relationship in make_engine(timedelta(seconds=5)).run(_determinism_corpus()):
            assert relationship.evidence_type is expected[relationship.rule_id]

    def test_confidence_is_typed_and_deterministic(self):
        for relationship in make_engine(timedelta(seconds=5)).run(_determinism_corpus()):
            assert 0.0 <= relationship.confidence.value <= 1.0
            assert isinstance(relationship.confidence.basis, ConfidenceBasis)

    def test_exact_rules_report_exact_match_confidence(self):
        relationships = make_engine().run(_determinism_corpus())
        for relationship in relationships:
            if relationship.rule_id != "temporal_adjacency":
                assert relationship.confidence.basis is ConfidenceBasis.EXACT_MATCH

    def test_temporal_reports_a_deterministic_rule_confidence(self):
        relationships = make_engine(timedelta(seconds=5)).run(_determinism_corpus())
        temporal = [item for item in relationships if item.rule_id == "temporal_adjacency"]
        assert temporal
        for relationship in temporal:
            assert relationship.confidence.basis is ConfidenceBasis.DETERMINISTIC_RULE

    def test_a_bounded_amount_reports_a_deterministic_rule_confidence(self):
        observations = [
            make_observation("obs-a", input_amounts=["1.000"]),
            make_observation("obs-b", input_amounts=["1.001"]),
        ]
        relationship = make_engine(amount_tolerance="0.01").run(observations)[0]
        assert relationship.confidence.basis is ConfidenceBasis.DETERMINISTIC_RULE

    def test_rationales_are_present_and_substantive(self):
        for relationship in make_engine(timedelta(seconds=5)).run(_determinism_corpus()):
            assert len(relationship.rationale.strip()) >= 10
            assert relationship.rationale == relationship.rationale.strip()

    def test_rationale_names_what_matched(self):
        observations = [
            make_observation("obs-a", txid="a" * 64),
            make_observation("obs-b", txid="a" * 64),
        ]
        assert "a" * 64 in make_engine().run(observations)[0].rationale

    def test_created_at_comes_from_the_injected_clock(self):
        relationships = make_engine(now=FROZEN_NOW).run(_determinism_corpus())
        assert all(item.created_at == FROZEN_NOW for item in relationships)

    def test_nothing_explains_attribution(self):
        for relationship in make_engine(timedelta(seconds=5)).run(_determinism_corpus()):
            assert relationship.explains_attribution() is False

    def test_a_temporal_relationship_carries_a_window_and_a_relation(self):
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=T0 + timedelta(seconds=3)),
        ]
        relationship = make_engine(timedelta(seconds=5)).run(observations)[0]
        assert relationship.temporal_window is not None
        assert relationship.temporal_window.start == T0
        assert relationship.temporal_window.end == T0 + timedelta(seconds=3)
        assert relationship.temporal_relation is TemporalRelation.BEFORE

    def test_temporal_relation_follows_the_source_side(self):
        observations = [
            make_observation("obs-b", timestamp=T0 + timedelta(seconds=3)),
            make_observation("obs-a", timestamp=T0),
        ]
        relationship = make_engine(timedelta(seconds=5)).run(observations)[0]
        # obs-a is the anchor and is the earlier of the two.
        assert relationship.source_observation_ids == ("obs-a",)
        assert relationship.temporal_relation is TemporalRelation.BEFORE

    def test_simultaneous_timestamps_are_within(self):
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=T0),
        ]
        relationship = make_engine(timedelta(seconds=5)).run(observations)[0]
        assert relationship.temporal_relation is TemporalRelation.WITHIN

    def test_a_non_temporal_relationship_carries_no_window(self):
        observations = [
            make_observation("obs-a", timestamp=T0, txid="a" * 64),
            make_observation("obs-b", timestamp=T0, txid="a" * 64),
        ]
        for relationship in make_engine(timedelta(seconds=5)).run(observations):
            if relationship.rule_id != "temporal_adjacency":
                assert relationship.temporal_window is None
                assert relationship.temporal_relation is None

    def test_the_engine_reports_its_configuration_and_rules(self):
        engine = make_engine(timedelta(seconds=5))
        assert engine.config.temporal_window == timedelta(seconds=5)
        assert "temporal_adjacency" in [rule.rule_id for rule in engine.rules]


# --------------------------------------------------------------------------- #
# Input immutability
# --------------------------------------------------------------------------- #


class TestNoInputMutation:
    def test_observations_are_unchanged_by_a_full_pass(self):
        corpus = _determinism_corpus()
        before = [freeze(observation) for observation in corpus]
        make_engine(timedelta(seconds=5)).run(corpus)
        assert [freeze(observation) for observation in corpus] == before

    def test_mutable_extra_survives_untouched(self):
        """``extra`` is a plain dict inside a frozen model; the pass must not touch it."""
        observations = [
            make_observation("obs-a", txid="a" * 64, src_ip="192.0.2.1"),
            make_observation("obs-b", txid="a" * 64, src_ip="192.0.2.1"),
        ]
        expected = copy.deepcopy([observation.extra for observation in observations])
        make_engine(timedelta(seconds=5)).run(observations)
        assert [observation.extra for observation in observations] == expected

    def test_every_layer_is_unchanged(self):
        observations = _determinism_corpus()
        before = [
            (
                observation.observation_id,
                freeze(observation.network),
                freeze(observation.blockchain),
                freeze(observation.provenance),
                observation.timestamp,
            )
            for observation in observations
        ]
        make_engine(timedelta(seconds=5)).run(observations)
        after = [
            (
                observation.observation_id,
                freeze(observation.network),
                freeze(observation.blockchain),
                freeze(observation.provenance),
                observation.timestamp,
            )
            for observation in observations
        ]
        assert before == after

    def test_address_lists_are_not_reordered(self):
        observations = [
            make_observation("obs-a", input_addresses=["bc1qz", "bc1qa", "bc1qm"]),
            make_observation("obs-b", input_addresses=["bc1qz", "bc1qa", "bc1qm"]),
        ]
        make_engine().run(observations)
        for observation in observations:
            assert observation.blockchain.input_addresses == ["bc1qz", "bc1qa", "bc1qm"]


# --------------------------------------------------------------------------- #
# Anti-attribution
# --------------------------------------------------------------------------- #


class TestAntiAttribution:
    @pytest.fixture
    def every_rule_fired(self):
        """A batch that trips all seven rules, so every rationale is scanned."""
        observations = [
            make_observation(
                "obs-a",
                txid="a" * 64,
                src_ip="192.0.2.1",
                dst_ip="198.51.100.7",
                geo_country="IN",
                input_addresses=["BC1QShared"],
                input_amounts=["1.5"],
                fee="0.0001",
                timestamp=T0,
                ingestion_id="ing-1",
                source_file="a.csv",
                source_record="row 1",
            ),
            make_observation(
                "obs-b",
                txid="A" * 64,
                src_ip="192.0.2.1",
                dst_ip="198.51.100.7",
                geo_country="in",
                input_addresses=["bc1qshared"],
                input_amounts=["1.5"],
                fee="0.0001",
                timestamp=T0 + timedelta(seconds=3),
                ingestion_id="ing-1",
                source_file="a.csv",
                source_record="row 1",
            ),
        ]
        relationships = make_engine(timedelta(seconds=5)).run(observations)
        assert {item.rule_id for item in relationships} == {
            "same_txid",
            "same_address",
            "same_ip",
            "temporal_adjacency",
            "amount_match",
            "provenance_link",
            "geo_coincidence",
        }
        return relationships

    def test_all_seven_rules_were_exercised(self, every_rule_fired):
        assert len(every_rule_fired) == 7

    @pytest.mark.parametrize("word", FORBIDDEN_CLAIM_WORDS)
    def test_no_rationale_contains_a_claim_word(self, every_rule_fired, word):
        for relationship in every_rule_fired:
            assert word not in relationship.rationale.lower(), relationship.rule_id

    def test_no_rationale_mentions_a_party(self, every_rule_fired):
        for relationship in every_rule_fired:
            lowered = relationship.rationale.lower()
            for word in ("belong", "controlled", "operator", "victim", "criminal"):
                assert word not in lowered, relationship.rule_id

    def test_every_relationship_is_only_correlated(self, every_rule_fired):
        for relationship in every_rule_fired:
            assert relationship.inference_level is InferenceLevel.CORRELATED

    def test_nothing_explains_attribution(self, every_rule_fired):
        assert all(
            relationship.explains_attribution() is False
            for relationship in every_rule_fired
        )

    def test_the_model_cannot_carry_an_attribution_field(self, every_rule_fired):
        """``extra="forbid"`` means a claim cannot be attached to a relationship."""
        for relationship in every_rule_fired:
            assert "attributed_to" not in type(relationship).model_fields
            assert "owner" not in type(relationship).model_fields
            with pytest.raises(Exception):
                type(relationship).model_validate(
                    {**relationship.model_dump(), "owner": "someone"}
                )


# --------------------------------------------------------------------------- #
# The real Step 3 substrate
# --------------------------------------------------------------------------- #


class TestRealSubstrate:
    """The structural contract must hold for genuine Step 3 models."""

    @staticmethod
    def _observation(observation_id, **overrides):
        from backend.app.domain.enums import ObservationType
        from backend.app.domain.models import (
            BlockchainTransaction,
            NetworkObservation,
            Provenance,
            UnifiedObservation,
        )

        network_overrides = overrides.pop("network", {})
        blockchain_overrides = overrides.pop("blockchain", {})
        return UnifiedObservation(
            observation_id=observation_id,
            observation_type=ObservationType.COMBINED,
            provenance=Provenance(
                ingestion_id="ing-1",
                source_file="a.csv",
                source_record="row %s" % observation_id[-1],
            ),
            network=NetworkObservation(**network_overrides),
            blockchain=BlockchainTransaction(**blockchain_overrides),
            **overrides,
        )

    def test_genuine_observations_correlate(self):
        first = self._observation(
            "obs-a",
            network={"src_ip": "192.0.2.1", "geo_country": "IN"},
            blockchain={"txid": "a" * 64, "input_amounts": [Decimal("1.5")]},
        )
        second = self._observation(
            "obs-b",
            network={"src_ip": "192.0.2.1", "geo_country": "IN"},
            blockchain={"txid": "A" * 64, "input_amounts": [Decimal("1.5")]},
        )
        relationships = make_engine().run([first, second])
        assert {item.rule_id for item in relationships} == {
            "same_txid",
            "same_ip",
            "geo_coincidence",
            "amount_match",
        }

    def test_genuine_observations_are_not_mutated(self):
        observations = [
            self._observation(
                "obs-a",
                network={"src_ip": "192.0.2.1"},
                blockchain={"txid": "a" * 64},
            ),
            self._observation(
                "obs-b",
                network={"src_ip": "192.0.2.1"},
                blockchain={"txid": "a" * 64},
            ),
        ]
        before = [observation.model_dump() for observation in observations]
        make_engine().run(observations)
        assert [observation.model_dump() for observation in observations] == before

    def test_an_ip_typed_field_renders_canonically(self):
        from backend.app.domain.models import NetworkObservation

        network = NetworkObservation(src_ip="::FFFF:192.0.2.1")
        assert normalize_ip(network.src_ip)


# --------------------------------------------------------------------------- #
# Large corpus: exact counts
# --------------------------------------------------------------------------- #


def _large_corpus() -> tuple[list[SimpleNamespace], dict[str, int]]:
    """Build 2 000 observations with collisions in known, disjoint places.

    Every observation carries a constant ``fee``; since ``fee`` is excluded from
    amount matching, any fee-derived relationship would show up as a surplus and
    fail the exact counts below.  Only the designated buckets share anything,
    and each bucket is disjoint from the others, so the expected per-rule counts
    can be read straight off the construction.
    """
    observations: list[SimpleNamespace] = []

    def add(index: int, **kwargs) -> None:
        observations.append(make_observation("obs-%04d" % index, **kwargs))

    for index in range(0, 300):  # transaction id groups of three
        add(
            index,
            txid="tx%04d" % (index // 3),
            input_addresses=["bc1quniq%05d" % index],
            input_amounts=[Decimal("%d.00000001" % index)],
            geo_country="C%04d" % index,
            ingestion_id="ing-%04d" % index,
            source_file="f%04d.csv" % index,
            source_record="row-%04d" % index,
        )
    for index in range(300, 600):  # source IP groups of ten
        add(
            index,
            txid="tx%04d" % index,
            src_ip="10.0.0.%d" % ((index - 300) // 10),
            input_addresses=["bc1quniq%05d" % index],
            input_amounts=[Decimal("%d.00000001" % index)],
            geo_country="C%04d" % index,
            ingestion_id="ing-%04d" % index,
            source_file="f%04d.csv" % index,
            source_record="row-%04d" % index,
        )
    for index in range(600, 800):  # destination IP groups of ten
        add(
            index,
            txid="tx%04d" % index,
            dst_ip="10.0.1.%d" % ((index - 600) // 10),
            input_addresses=["bc1quniq%05d" % index],
            input_amounts=[Decimal("%d.00000001" % index)],
            geo_country="C%04d" % index,
            ingestion_id="ing-%04d" % index,
            source_file="f%04d.csv" % index,
            source_record="row-%04d" % index,
        )
    for index in range(800, 900):  # bech32 address groups of five
        add(
            index,
            txid="tx%04d" % index,
            input_addresses=["BC1QGroup%04d" % ((index - 800) // 5)],
            input_amounts=[Decimal("%d.00000001" % index)],
            geo_country="C%04d" % index,
            ingestion_id="ing-%04d" % index,
            source_file="f%04d.csv" % index,
            source_record="row-%04d" % index,
        )
    for index in range(900, 1000):  # country groups of five
        add(
            index,
            txid="tx%04d" % index,
            input_addresses=["bc1quniq%05d" % index],
            input_amounts=[Decimal("%d.00000001" % index)],
            geo_country="ZZ%02d" % ((index - 900) // 5),
            ingestion_id="ing-%04d" % index,
            source_file="f%04d.csv" % index,
            source_record="row-%04d" % index,
        )
    for index in range(1000, 1200):  # amount groups of four, each its own value
        add(
            index,
            txid="tx%04d" % index,
            input_addresses=["bc1quniq%05d" % index],
            input_amounts=[Decimal((index - 1000) // 4)],
            geo_country="C%04d" % index,
            ingestion_id="ing-%04d" % index,
            source_file="f%04d.csv" % index,
            source_record="row-%04d" % index,
        )
    for index in range(1200, 1300):  # provenance groups of five
        group = (index - 1200) // 5
        add(
            index,
            txid="tx%04d" % index,
            input_addresses=["bc1quniq%05d" % index],
            input_amounts=[Decimal("%d.00000001" % index)],
            geo_country="C%04d" % index,
            ingestion_id="ing-shared-%02d" % group,
            source_file="shared-%02d.csv" % group,
            source_record="row-1",
        )
    for index in range(1300, 1500):  # temporal pairs one second apart
        offset = (index - 1300) // 2
        add(
            index,
            txid="tx%04d" % index,
            input_addresses=["bc1quniq%05d" % index],
            input_amounts=[Decimal("%d.00000001" % index)],
            geo_country="C%04d" % index,
            ingestion_id="ing-%04d" % index,
            source_file="f%04d.csv" % index,
            source_record="row-%04d" % index,
            timestamp=T0 + timedelta(seconds=offset * 100 + (index - 1300) % 2),
        )
    for index in range(1500, 2000):  # filler, sharing nothing
        add(
            index,
            txid="tx%04d" % index,
            input_addresses=["bc1quniq%05d" % index],
            input_amounts=[Decimal("%d.00000001" % index)],
            geo_country="C%04d" % index,
            ingestion_id="ing-%04d" % index,
            source_file="f%04d.csv" % index,
            source_record="row-%04d" % index,
            fee="0.0001",
        )

    # Give every observation the same incidental fee, including the bucket
    # members above, so a fee-derived match would be visible in the counts.
    for observation in observations:
        if observation.blockchain is not None and observation.blockchain.fee is None:
            observation.blockchain.fee = Decimal("0.0001")

    expected = {
        "same_txid": 100,          # 300 observations in groups of three
        "same_ip": 50,             # 30 source groups of ten + 20 destination groups
        "same_address": 20,        # 100 observations in groups of five
        "geo_coincidence": 20,     # 100 observations in groups of five
        "amount_match": 50,        # 200 observations in groups of four
        "provenance_link": 20,     # 100 observations in groups of five
        "temporal_adjacency": 100,  # 100 pairs one second apart
    }
    return observations, expected


class TestLargeCorpus:
    def test_the_corpus_really_holds_two_thousand_observations(self):
        observations, _ = _large_corpus()
        assert len(observations) == 2000
        assert len({observation.observation_id for observation in observations}) == 2000

    def test_exact_relationship_counts_per_rule(self):
        observations, expected = _large_corpus()
        relationships = make_engine(timedelta(seconds=5)).run(observations)
        assert dict(Counter(item.rule_id for item in relationships)) == expected

    def test_exact_total_relationship_count(self):
        observations, expected = _large_corpus()
        relationships = make_engine(timedelta(seconds=5)).run(observations)
        assert len(relationships) == sum(expected.values()) == 360

    def test_the_shared_fee_produces_no_relationships(self):
        """All 2 000 observations carry fee 0.0001; fee must contribute nothing."""
        observations, _ = _large_corpus()
        fees = {
            observation.blockchain.fee
            for observation in observations
            if observation.blockchain is not None
        }
        assert fees == {Decimal("0.0001")}
        relationships = make_engine(timedelta(seconds=5)).run(observations)
        assert all(item.rule_id != "amount_match" or "fee" not in item.rationale for item in relationships)

    def test_large_corpus_is_deterministic_and_shuffle_stable(self):
        observations, _ = _large_corpus()
        baseline = make_engine(timedelta(seconds=5)).run(observations)
        shuffled = list(observations)
        random.Random(4242).shuffle(shuffled)
        assert make_engine(timedelta(seconds=5)).run(shuffled) == baseline

    def test_large_corpus_is_not_mutated(self):
        observations, _ = _large_corpus()
        before = [freeze(observation) for observation in observations]
        make_engine(timedelta(seconds=5)).run(observations)
        assert [freeze(observation) for observation in observations] == before

    def test_large_corpus_has_no_self_references_or_duplicates(self):
        observations, _ = _large_corpus()
        relationships = make_engine(timedelta(seconds=5)).run(observations)
        for relationship in relationships:
            assert not relationship.is_self_referential
        assert len(relationship_keys(relationships)) == len(relationships)


# --------------------------------------------------------------------------- #
# Brute-force equivalence
# --------------------------------------------------------------------------- #


def _equivalence_corpus(seed: int = 20240501, size: int = 300) -> list[SimpleNamespace]:
    """Observations carrying exactly one value per rule-relevant field.

    One value per field matters: it makes "shares this value" an equivalence
    relation, so the groups the engine builds and the groups a pairwise
    reference reconstructs are the same partition.  With two addresses on one
    record, a class could strictly contain another and the two implementations
    would legitimately disagree.
    """
    rng = random.Random(seed)
    observations = []
    for index in range(size):
        observations.append(
            make_observation(
                "obs-%04d" % index,
                txid="tx%04d" % rng.randrange(0, 90),
                src_ip="10.1.%d.%d" % (rng.randrange(0, 4), rng.randrange(0, 20)),
                dst_ip="10.2.%d.%d" % (rng.randrange(0, 4), rng.randrange(0, 20)),
                geo_country="C%02d" % rng.randrange(0, 12),
                input_addresses=["bc1q%05d" % rng.randrange(0, 70)],
                input_amounts=[Decimal(rng.randrange(0, 40))],
                ingestion_id="ing-%02d" % rng.randrange(0, 25),
                source_file="f%02d.csv" % rng.randrange(0, 15),
                source_record="row-%02d" % rng.randrange(0, 30),
                timestamp=T0 + timedelta(minutes=rng.randrange(0, 600)),
            )
        )
    return observations


def _shared_keys(observation: SimpleNamespace) -> list[tuple[str, Any]]:
    """The ``(rule_id, key)`` pairs a single observation offers each rule."""
    keys: list[tuple[str, Any]] = []
    blockchain = observation.blockchain
    if blockchain is not None:
        txid = normalize_txid(blockchain.txid)
        if txid is not None:
            keys.append(("same_txid", txid))
        for address in list(blockchain.input_addresses) + list(blockchain.output_addresses):
            normalized = normalize_address(address)
            if normalized is not None:
                keys.append(("same_address", normalized))
        for amount in list(blockchain.input_amounts) + list(blockchain.output_amounts):
            if amount is not None:
                keys.append(("amount_match", amount))
    network = observation.network
    if network is not None:
        for role, attribute in (("src", "src_ip"), ("dst", "dst_ip")):
            address = normalize_ip(getattr(network, attribute, None))
            if address is not None:
                keys.append(("same_ip", (role, address)))
        country = normalize_country(network.geo_country)
        if country is not None:
            keys.append(("geo_coincidence", country))
    provenance = observation.provenance
    if provenance is not None:
        parts = (
            getattr(provenance, "ingestion_id", None),
            getattr(provenance, "source_file", None),
            getattr(provenance, "source_record", None),
        )
        if all(isinstance(part, str) and part.strip() for part in parts):
            keys.append(("provenance_link", tuple(part.strip() for part in parts)))
    return keys


def _brute_force(
    observations: list[SimpleNamespace], config: CorrelationConfig
) -> set[tuple]:
    """An obviously-correct O(n^2) reference for the same specification.

    Deliberately naive: it compares every pair with every other, rebuilding the
    groups from the matches it discovers rather than indexing each record once.
    """
    by_id: dict[str, SimpleNamespace] = {}
    for observation in observations:
        by_id.setdefault(observation.observation_id, observation)
    ids = sorted(by_id)
    result: set[tuple] = set()

    # Shared-key rules: reconstruct each key's members from matching pairs.
    members: dict[tuple[str, Any], set[str]] = {}
    for position, left in enumerate(ids):
        left_keys = set(_shared_keys(by_id[left]))
        for right in ids[position + 1:]:
            for rule_id, key in left_keys & set(_shared_keys(by_id[right])):
                if config.amount_tolerance > 0 and rule_id == "amount_match":
                    continue  # a bound is not a shared value; handled pairwise below
                members.setdefault((rule_id, key), set()).update((left, right))
    for (rule_id, _key), group in members.items():
        ordered = sorted(group)
        if len(ordered) >= 2:
            result.add((rule_id, (ordered[0],), tuple(ordered[1:])))

    # Pairwise rules.
    for position, left in enumerate(ids):
        for right in ids[position + 1:]:
            if config.temporal_window is not None:
                left_time = by_id[left].timestamp
                right_time = by_id[right].timestamp
                if _aware(left_time) and _aware(right_time):
                    if abs(right_time - left_time) <= config.temporal_window:
                        result.add(("temporal_adjacency", (left,), (right,)))
            if config.amount_tolerance > 0:
                left_amounts = _amounts(by_id[left])
                right_amounts = _amounts(by_id[right])
                if any(
                    abs(a - b) <= config.amount_tolerance
                    for a in left_amounts
                    for b in right_amounts
                ):
                    result.add(("amount_match", (left,), (right,)))
    return result


def _aware(moment) -> bool:
    return (
        isinstance(moment, datetime)
        and moment.tzinfo is not None
        and moment.tzinfo.utcoffset(moment) is not None
    )


def _amounts(observation: SimpleNamespace) -> list[Decimal]:
    blockchain = observation.blockchain
    if blockchain is None:
        return []
    return [
        value
        for value in list(blockchain.input_amounts) + list(blockchain.output_amounts)
        if value is not None
    ]


class TestBruteForceEquivalence:
    def test_indexed_engine_matches_the_reference_with_exact_amounts(self):
        config = CorrelationConfig(
            temporal_window=timedelta(minutes=3), amount_tolerance=Decimal("0"), now=FROZEN_NOW
        )
        corpus = _equivalence_corpus()
        engine_result = relationship_keys(CorrelationEngine(config).run(corpus))
        reference_result = _brute_force(corpus, config)
        assert engine_result == reference_result

    def test_indexed_engine_matches_the_reference_with_a_tolerance(self):
        config = CorrelationConfig(
            temporal_window=timedelta(minutes=3),
            amount_tolerance=Decimal("2"),
            now=FROZEN_NOW,
        )
        corpus = _equivalence_corpus(seed=99, size=200)
        engine_result = relationship_keys(CorrelationEngine(config).run(corpus))
        reference_result = _brute_force(corpus, config)
        assert engine_result == reference_result

    def test_indexed_engine_matches_the_reference_without_a_temporal_window(self):
        config = CorrelationConfig(amount_tolerance=Decimal("0"), now=FROZEN_NOW)
        corpus = _equivalence_corpus(seed=7, size=200)
        assert relationship_keys(CorrelationEngine(config).run(corpus)) == _brute_force(
            corpus, config
        )

    def test_every_emitted_pair_satisfies_its_criterion(self):
        """Soundness: no relationship may assert a match that does not hold."""
        config = CorrelationConfig(
            temporal_window=timedelta(minutes=3), amount_tolerance=Decimal("0"), now=FROZEN_NOW
        )
        corpus = _equivalence_corpus(seed=31337, size=250)
        by_id = {observation.observation_id: observation for observation in corpus}
        for relationship in CorrelationEngine(config).run(corpus):
            for rule_id, left, right in all_pairs([relationship]):
                assert rule_id == relationship.rule_id
                shared = set(_shared_keys(by_id[left])) & set(_shared_keys(by_id[right]))
                if rule_id == "temporal_adjacency":
                    assert abs(by_id[right].timestamp - by_id[left].timestamp) <= (
                        config.temporal_window
                    )
                else:
                    assert (rule_id, _key_of(shared, rule_id)) in shared

    def test_the_reference_finds_several_rules_so_the_test_is_meaningful(self):
        config = CorrelationConfig(
            temporal_window=timedelta(minutes=3), amount_tolerance=Decimal("0"), now=FROZEN_NOW
        )
        reference = _brute_force(_equivalence_corpus(), config)
        assert {item[0] for item in reference} == {
            "same_txid",
            "same_address",
            "same_ip",
            "geo_coincidence",
            "amount_match",
            "provenance_link",
            "temporal_adjacency",
        }


def _key_of(shared: set, rule_id: str) -> Any:
    for candidate_rule, key in shared:
        if candidate_rule == rule_id:
            return key
    raise AssertionError("no %s key in %r" % (rule_id, shared))
