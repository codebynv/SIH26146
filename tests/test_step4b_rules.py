"""Unit tests for the Step 4B deterministic correlation rules.

These tests exercise the rules directly, without the engine, so a failure names
the rule at fault rather than "correlation produced the wrong thing".  The
engine's own guarantees -- grouping, merging, ordering, determinism -- are
covered in ``test_step4b_engine.py``.

Observations here are plain :class:`types.SimpleNamespace` objects rather than
:class:`~backend.app.domain.models.UnifiedObservation` instances.  That is
deliberate: Step 4A forbids the correlation package from importing the domain
layer, so the rules read observations structurally, and these fixtures prove
the structural contract holds without dragging Step 3 into a rule unit test.
One end-to-end test in the engine suite uses a real Step 3 model to confirm
the two agree.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import pytest

from backend.app.correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    TemporalRelation,
)
from backend.app.correlation.rules import (
    RULE_VERSION,
    AmountMatchRule,
    CorrelationConfig,
    GeoCoinidenceRule,
    MatchGroup,
    ProvenanceLinkRule,
    SameAddressRule,
    SameIpRule,
    SameTxidRule,
    TemporalAdjacencyRule,
    build_default_rules,
    describe_window,
    join_facts,
    normalize_address,
    normalize_country,
    normalize_ip,
    normalize_txid,
)

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fixtures
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
    ingestion_id: str = "ing-1",
    source_file: Optional[str] = "records.csv",
    source_record: Optional[str] = "row 1",
) -> SimpleNamespace:
    """Build a duck-typed observation exposing the attributes rules may read."""
    inputs = list(input_addresses)
    outputs = list(output_addresses)
    in_amounts = [Decimal(str(value)) for value in input_amounts]
    out_amounts = [Decimal(str(value)) for value in output_amounts]

    blockchain = None
    if any(
        value is not None
        for value in (txid, fee)
    ) or inputs or outputs or in_amounts or out_amounts:
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
            src_ip=src_ip,
            dst_ip=dst_ip,
            geo_country=geo_country,
        )

    return SimpleNamespace(
        observation_id=observation_id,
        timestamp=timestamp,
        network=network,
        blockchain=blockchain,
        provenance=SimpleNamespace(
            ingestion_id=ingestion_id,
            source_file=source_file,
            source_record=source_record,
        ),
        # Present to prove no rule reads it: a mutable mapping inside an
        # otherwise read-only surface.
        extra={"mutable": {"nested": [1, 2, 3]}},
    )


def groups_of(rule, observations) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Run *rule* and return its matches as ``(ids, facts)`` pairs."""
    return [(group.observation_ids, group.facts) for group in rule.evaluate(observations)]


def matched_ids(rule, observations) -> set[frozenset[str]]:
    """The sets of observation identifiers *rule* considers matched."""
    return {frozenset(group.observation_ids) for group in rule.evaluate(observations)}


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


class TestNormalisation:
    def test_bech32_addresses_fold_case(self):
        """Bech32 is case-insensitive, so case-only differences are one address."""
        assert normalize_address("BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4") == normalize_address(
            "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
        )

    def test_bech32_prefixes_are_recognised(self):
        for prefix in ("bc1", "tb1", "bcrt1"):
            assert normalize_address(prefix + "QwErTy") == prefix + "qwerty"

    def test_base58_case_is_preserved(self):
        """Base58 encodes case as data: folding it would invent a match."""
        lower = "1bvbmsestwetqtfN5au4m4gfG7xjanvN2"
        upper = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
        assert normalize_address(lower) == lower
        assert normalize_address(upper) == upper
        assert normalize_address(lower) != normalize_address(upper)

    def test_address_whitespace_is_stripped(self):
        assert normalize_address("  1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2 ") == (
            "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
        )

    @pytest.mark.parametrize("value", ["", "   ", None, 42, [], {}])
    def test_blank_or_non_text_addresses_are_absent(self, value):
        assert normalize_address(value) is None

    def test_txid_is_stripped_and_lowercased(self):
        assert normalize_txid("  AABBCC ") == "aabbcc"

    @pytest.mark.parametrize("value", ["", "  ", None, 7])
    def test_blank_txid_is_absent(self, value):
        assert normalize_txid(value) is None

    def test_country_is_stripped_and_uppercased(self):
        assert normalize_country("  in ") == "IN"

    @pytest.mark.parametrize("value", ["", "   ", None, 3])
    def test_blank_country_is_absent(self, value):
        assert normalize_country(value) is None

    def test_ip_renders_canonically(self):
        assert normalize_ip("192.0.2.1") == "192.0.2.1"
        assert normalize_ip(" 192.0.2.1 ") == "192.0.2.1"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_ip_is_absent(self, value):
        assert normalize_ip(value) is None

    def test_join_facts_is_order_independent(self):
        assert join_facts(["b", "a"]) == join_facts(["a", "b"]) == "a and b"

    def test_join_facts_of_one_reads_naturally(self):
        assert join_facts(["only"]) == "only"

    def test_join_facts_of_none(self):
        assert join_facts([]) == "matched values"

    def test_describe_window_uses_whole_seconds(self):
        assert describe_window(timedelta(seconds=5)) == "5 seconds"

    def test_describe_window_keeps_fractions(self):
        assert "0.5" in describe_window(timedelta(milliseconds=500))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestCorrelationConfig:
    def test_defaults_to_exact_amount_matching(self):
        assert CorrelationConfig().amount_tolerance == Decimal("0")

    def test_temporal_rule_is_off_without_a_window(self):
        """No window means no temporal correlation, not a default window."""
        config = CorrelationConfig()
        assert config.temporal_window is None
        rules = build_default_rules(config)
        assert not any(rule.rule_id == "temporal_adjacency" for rule in rules)

    def test_configured_window_enables_the_temporal_rule(self):
        rules = build_default_rules(CorrelationConfig(temporal_window=timedelta(seconds=5)))
        assert any(rule.rule_id == "temporal_adjacency" for rule in rules)

    def test_zero_window_is_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            CorrelationConfig(temporal_window=timedelta(0))

    def test_negative_window_is_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            CorrelationConfig(temporal_window=timedelta(seconds=-1))

    def test_non_timedelta_window_is_rejected(self):
        with pytest.raises(ValueError, match="timedelta"):
            CorrelationConfig(temporal_window=5)

    def test_negative_amount_tolerance_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            CorrelationConfig(amount_tolerance=Decimal("-0.1"))

    def test_naive_now_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            CorrelationConfig(now=datetime(2024, 5, 1, 12, 0, 0))

    def test_tolerance_accepts_a_numeric_string(self):
        assert CorrelationConfig(amount_tolerance="0.25").amount_tolerance == Decimal("0.25")

    def test_default_now_is_timezone_aware(self):
        assert CorrelationConfig().now.tzinfo is not None

    def test_default_rule_set_is_fixed_and_complete(self):
        rules = build_default_rules(CorrelationConfig())
        assert [rule.rule_id for rule in rules] == [
            "same_txid",
            "same_address",
            "same_ip",
            "amount_match",
            "provenance_link",
            "geo_coincidence",
        ]


# --------------------------------------------------------------------------- #
# MatchGroup
# --------------------------------------------------------------------------- #


class TestMatchGroup:
    def test_rejects_an_empty_group(self):
        with pytest.raises(ValueError, match="at least one observation"):
            MatchGroup(observation_ids=())

    def test_rejects_a_repeated_observation(self):
        with pytest.raises(ValueError, match="must not repeat"):
            MatchGroup(observation_ids=("a", "a"))

    def test_carries_the_matched_facts(self):
        group = MatchGroup(observation_ids=("a", "b"), facts=("x",))
        assert group.facts == ("x",)


# --------------------------------------------------------------------------- #
# 1. same_txid
# --------------------------------------------------------------------------- #


class TestSameTxidRule:
    def test_matches_a_shared_transaction_id(self):
        observations = [
            make_observation("obs-a", txid="a" * 64),
            make_observation("obs-b", txid="a" * 64),
        ]
        assert groups_of(SameTxidRule(), observations) == [
            (("obs-a", "obs-b"), ("transaction id %s" % ("a" * 64),))
        ]

    def test_ignores_case_differences(self):
        observations = [
            make_observation("obs-a", txid="ABCDEF"),
            make_observation("obs-b", txid="abcdef"),
        ]
        assert matched_ids(SameTxidRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_distinct_ids_do_not_match(self):
        observations = [
            make_observation("obs-a", txid="a" * 64),
            make_observation("obs-b", txid="b" * 64),
        ]
        assert groups_of(SameTxidRule(), observations) == []

    @pytest.mark.parametrize("txid", [None, "", "   "])
    def test_missing_or_blank_transaction_ids_are_skipped(self, txid):
        observations = [
            make_observation("obs-a", txid=txid),
            make_observation("obs-b", txid=txid),
        ]
        assert groups_of(SameTxidRule(), observations) == []

    def test_observation_without_a_blockchain_layer(self):
        observations = [make_observation("obs-a"), make_observation("obs-b")]
        assert groups_of(SameTxidRule(), observations) == []

    def test_groups_three_or_more_into_one_match(self):
        observations = [
            make_observation("obs-%d" % index, txid="deadbeef")
            for index in range(3)
        ]
        matches = groups_of(SameTxidRule(), observations)
        assert len(matches) == 1
        assert matches[0][0] == ("obs-0", "obs-1", "obs-2")

    def test_reports_the_matched_value_in_its_rationale(self):
        rule = SameTxidRule()
        rationale = rule.describe(("transaction id abc123",))
        assert "abc123" in rationale
        assert rationale.endswith(".")
        assert len(rationale) >= 10

    def test_evidence_type_and_version(self):
        rule = SameTxidRule()
        assert rule.evidence_type is EvidenceType.SAME_TXID
        assert rule.rule_version == RULE_VERSION

    def test_confidence_is_exact_equality(self):
        assert SameTxidRule().confidence.basis is ConfidenceBasis.EXACT_MATCH
        assert SameTxidRule().confidence.value == 1.0


# --------------------------------------------------------------------------- #
# 2. same_address
# --------------------------------------------------------------------------- #


class TestSameAddressRule:
    def test_bech32_matches_across_case(self):
        observations = [
            make_observation("obs-a", input_addresses=["BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4"]),
            make_observation("obs-b", input_addresses=["bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"]),
        ]
        assert matched_ids(SameAddressRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_base58_case_difference_does_not_match(self):
        observations = [
            make_observation("obs-a", input_addresses=["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"]),
            make_observation("obs-b", input_addresses=["1bvbmseystwetqtfN5Au4m4GFg7xJaNVN2"]),
        ]
        assert groups_of(SameAddressRule(), observations) == []

    def test_base58_exact_match_still_matches(self):
        address = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
        observations = [
            make_observation("obs-a", input_addresses=[address]),
            make_observation("obs-b", output_addresses=[address]),
        ]
        assert matched_ids(SameAddressRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_input_and_output_are_pooled(self):
        """An address leaving one record and entering another is a repetition."""
        address = "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"
        observations = [
            make_observation("obs-a", output_addresses=[address]),
            make_observation("obs-b", input_addresses=[address]),
        ]
        assert matched_ids(SameAddressRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_distinct_addresses_do_not_match(self):
        observations = [
            make_observation("obs-a", input_addresses=["bc1qaaaa"]),
            make_observation("obs-b", input_addresses=["bc1qbbbb"]),
        ]
        assert groups_of(SameAddressRule(), observations) == []

    @pytest.mark.parametrize("address", ["", "   "])
    def test_blank_addresses_are_skipped(self, address):
        observations = [
            make_observation("obs-a", input_addresses=[address]),
            make_observation("obs-b", input_addresses=[address]),
        ]
        assert groups_of(SameAddressRule(), observations) == []

    def test_rationale_does_not_claim_control(self):
        rule = SameAddressRule()
        rationale = rule.describe(("normalized blockchain address bc1qxyz",))
        assert "bc1qxyz" in rationale
        assert "not a statement of control" in rationale

    def test_evidence_type(self):
        assert SameAddressRule().evidence_type is EvidenceType.SAME_ADDRESS


# --------------------------------------------------------------------------- #
# 3. same_ip
# --------------------------------------------------------------------------- #


class TestSameIpRule:
    def test_source_to_source_matches(self):
        observations = [
            make_observation("obs-a", src_ip="192.0.2.1"),
            make_observation("obs-b", src_ip="192.0.2.1"),
        ]
        assert matched_ids(SameIpRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_destination_to_destination_matches(self):
        observations = [
            make_observation("obs-a", dst_ip="192.0.2.2"),
            make_observation("obs-b", dst_ip="192.0.2.2"),
        ]
        assert matched_ids(SameIpRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_source_to_destination_does_not_match(self):
        """Opposite ends of a connection are not a shared endpoint."""
        observations = [
            make_observation("obs-a", src_ip="192.0.2.1"),
            make_observation("obs-b", dst_ip="192.0.2.1"),
        ]
        assert groups_of(SameIpRule(), observations) == []

    def test_destination_to_source_does_not_match(self):
        observations = [
            make_observation("obs-a", dst_ip="192.0.2.1"),
            make_observation("obs-b", src_ip="192.0.2.1"),
        ]
        assert groups_of(SameIpRule(), observations) == []

    def test_both_roles_matching_produces_two_merged_facts(self):
        observations = [
            make_observation("obs-a", src_ip="192.0.2.1", dst_ip="198.51.100.1"),
            make_observation("obs-b", src_ip="192.0.2.1", dst_ip="198.51.100.1"),
        ]
        matches = groups_of(SameIpRule(), observations)
        assert len(matches) == 2
        assert {match[0] for match in matches} == {("obs-a", "obs-b")}

    def test_a_single_observation_never_matches_itself(self):
        observations = [make_observation("obs-a", src_ip="192.0.2.1")]
        assert groups_of(SameIpRule(), observations) == []

    def test_missing_ips_are_skipped(self):
        observations = [
            make_observation("obs-a"),
            make_observation("obs-b"),
        ]
        assert groups_of(SameIpRule(), observations) == []

    def test_facts_name_the_role(self):
        rule = SameIpRule()
        assert rule._facts(("src", "192.0.2.1")) == ("source IP address 192.0.2.1",)
        assert rule._facts(("dst", "192.0.2.1")) == ("destination IP address 192.0.2.1",)

    def test_rationale_disclaims_a_shared_party(self):
        rationale = SameIpRule().describe(("source IP address 192.0.2.1",))
        assert "not evidence that one party was behind both" in rationale

    def test_evidence_type(self):
        assert SameIpRule().evidence_type is EvidenceType.SAME_IP


# --------------------------------------------------------------------------- #
# 4. temporal_adjacency
# --------------------------------------------------------------------------- #


class TestTemporalAdjacencyRule:
    def test_requires_an_explicit_window(self):
        with pytest.raises(ValueError, match="explicit temporal_window"):
            TemporalAdjacencyRule()

    def test_rejects_a_non_positive_window(self):
        with pytest.raises(ValueError, match="strictly positive"):
            TemporalAdjacencyRule(timedelta(0))
        with pytest.raises(ValueError, match="strictly positive"):
            TemporalAdjacencyRule(timedelta(seconds=-5))

    def test_rejects_a_non_timedelta_window(self):
        with pytest.raises(ValueError, match="timedelta"):
            TemporalAdjacencyRule(30)

    def test_matches_inside_the_window(self):
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=T0 + timedelta(seconds=4)),
        ]
        assert matched_ids(
            TemporalAdjacencyRule(timedelta(seconds=5)), observations
        ) == {frozenset({"obs-a", "obs-b"})}

    def test_does_not_match_outside_the_window(self):
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=T0 + timedelta(seconds=6)),
        ]
        assert groups_of(TemporalAdjacencyRule(timedelta(seconds=5)), observations) == []

    def test_matches_exactly_at_the_boundary(self):
        """The criterion is ``<=``, so the boundary itself is inside."""
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=T0 + timedelta(seconds=5)),
        ]
        assert matched_ids(
            TemporalAdjacencyRule(timedelta(seconds=5)), observations
        ) == {frozenset({"obs-a", "obs-b"})}

    def test_does_not_match_one_microsecond_past_the_boundary(self):
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=T0 + timedelta(seconds=5, microseconds=1)),
        ]
        assert groups_of(TemporalAdjacencyRule(timedelta(seconds=5)), observations) == []

    def test_naive_timestamps_are_skipped_not_localised(self):
        """A naive value has no offset; assuming one would shift every result."""
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=datetime(2024, 5, 1, 12, 0, 1)),
        ]
        assert groups_of(TemporalAdjacencyRule(timedelta(hours=1)), observations) == []

    def test_two_naive_timestamps_do_not_match(self):
        observations = [
            make_observation("obs-a", timestamp=datetime(2024, 5, 1, 12, 0, 0)),
            make_observation("obs-b", timestamp=datetime(2024, 5, 1, 12, 0, 1)),
        ]
        assert groups_of(TemporalAdjacencyRule(timedelta(hours=1)), observations) == []

    def test_mixed_naive_and_aware_is_handled_safely(self):
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=datetime(2024, 5, 1, 12, 0, 1)),
            make_observation("obs-c", timestamp=T0 + timedelta(seconds=1)),
        ]
        assert matched_ids(
            TemporalAdjacencyRule(timedelta(seconds=5)), observations
        ) == {frozenset({"obs-a", "obs-c"})}

    def test_missing_timestamps_are_skipped(self):
        observations = [
            make_observation("obs-a"),
            make_observation("obs-b"),
        ]
        assert groups_of(TemporalAdjacencyRule(timedelta(seconds=5)), observations) == []

    def test_offsets_are_normalised_before_comparison(self):
        """Equal instants expressed in different zones are the same instant."""
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation(
                "obs-b", timestamp=datetime(2024, 5, 1, 17, 0, 1, tzinfo=timezone(timedelta(hours=5)))
            ),
        ]
        assert matched_ids(
            TemporalAdjacencyRule(timedelta(seconds=5)), observations
        ) == {frozenset({"obs-a", "obs-b"})}

    def test_reports_each_qualifying_pair_once(self):
        observations = [
            make_observation("obs-%d" % index, timestamp=T0 + timedelta(seconds=index))
            for index in range(4)
        ]
        assert len(groups_of(TemporalAdjacencyRule(timedelta(seconds=5)), observations)) == 6

    def test_far_apart_groups_are_not_bridged_by_bucketing(self):
        """Candidates come from neighbouring buckets; distance decides."""
        observations = [
            make_observation("obs-a", timestamp=T0),
            make_observation("obs-b", timestamp=T0 + timedelta(days=400)),
        ]
        assert groups_of(TemporalAdjacencyRule(timedelta(seconds=5)), observations) == []

    def test_requires_at_least_two_observations(self):
        rule = TemporalAdjacencyRule(timedelta(seconds=5))
        assert groups_of(rule, [make_observation("obs-a", timestamp=T0)]) == []

    def test_confidence_is_a_documented_rule_value(self):
        confidence = TemporalAdjacencyRule(timedelta(seconds=5)).confidence
        assert confidence.basis is ConfidenceBasis.DETERMINISTIC_RULE
        assert 0.0 < confidence.value < 1.0

    def test_evidence_type(self):
        assert (
            TemporalAdjacencyRule(timedelta(seconds=5)).evidence_type
            is EvidenceType.TEMPORAL_ADJACENCY
        )


# --------------------------------------------------------------------------- #
# 5. amount_match
# --------------------------------------------------------------------------- #


class TestAmountMatchRule:
    def test_matches_an_exact_amount(self):
        observations = [
            make_observation("obs-a", input_amounts=["0.5"]),
            make_observation("obs-b", input_amounts=["0.5"]),
        ]
        assert matched_ids(AmountMatchRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_decimal_equality_is_value_equality(self):
        """``0.50`` and ``0.5`` are the same amount; a float test could differ."""
        observations = [
            make_observation("obs-a", input_amounts=["0.50"]),
            make_observation("obs-b", input_amounts=["0.5"]),
        ]
        assert matched_ids(AmountMatchRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_distinct_amounts_do_not_match(self):
        observations = [
            make_observation("obs-a", input_amounts=["0.5"]),
            make_observation("obs-b", input_amounts=["0.6"]),
        ]
        assert groups_of(AmountMatchRule(), observations) == []

    def test_zero_amounts_are_real_values(self):
        """``Decimal("0")`` is falsy but matchable; discarding it would lose evidence."""
        observations = [
            make_observation("obs-a", input_amounts=["0"]),
            make_observation("obs-b", output_amounts=["0"]),
        ]
        assert matched_ids(AmountMatchRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_input_and_output_amounts_are_pooled(self):
        observations = [
            make_observation("obs-a", input_amounts=["1.25"]),
            make_observation("obs-b", output_amounts=["1.25"]),
        ]
        assert matched_ids(AmountMatchRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_fee_never_matches(self):
        """A shared incidental fee column is not a shared blockchain amount."""
        observations = [
            make_observation("obs-a", fee="0.0001"),
            make_observation("obs-b", fee="0.0001"),
        ]
        assert groups_of(AmountMatchRule(), observations) == []

    def test_fee_does_not_match_across_a_tolerance(self):
        observations = [
            make_observation("obs-a", fee="0.0001"),
            make_observation("obs-b", fee="0.0002"),
        ]
        assert groups_of(
            AmountMatchRule(Decimal("0.01")), observations
        ) == []

    def test_tolerance_zero_requires_exact_equality(self):
        observations = [
            make_observation("obs-a", input_amounts=["1.000"]),
            make_observation("obs-b", input_amounts=["1.001"]),
        ]
        assert groups_of(AmountMatchRule(Decimal("0")), observations) == []

    def test_positive_tolerance_matches_inside_the_bound(self):
        observations = [
            make_observation("obs-a", input_amounts=["1.000"]),
            make_observation("obs-b", input_amounts=["1.001"]),
        ]
        assert matched_ids(
            AmountMatchRule(Decimal("0.01")), observations
        ) == {frozenset({"obs-a", "obs-b"})}

    def test_positive_tolerance_matches_exactly_at_the_boundary(self):
        observations = [
            make_observation("obs-a", input_amounts=["1.000"]),
            make_observation("obs-b", input_amounts=["1.001"]),
        ]
        assert matched_ids(
            AmountMatchRule(Decimal("0.001")), observations
        ) == {frozenset({"obs-a", "obs-b"})}

    def test_positive_tolerance_excludes_one_unit_past_the_boundary(self):
        observations = [
            make_observation("obs-a", input_amounts=["1.000"]),
            make_observation("obs-b", input_amounts=["1.0011"]),
        ]
        assert groups_of(AmountMatchRule(Decimal("0.001")), observations) == []

    def test_tolerance_matches_in_both_directions(self):
        observations = [
            make_observation("obs-a", input_amounts=["1.100"]),
            make_observation("obs-b", input_amounts=["1.000"]),
        ]
        assert matched_ids(
            AmountMatchRule(Decimal("0.2")), observations
        ) == {frozenset({"obs-a", "obs-b"})}

    def test_far_apart_amounts_are_not_bridged_by_the_index(self):
        observations = [
            make_observation("obs-a", input_amounts=["1"]),
            make_observation("obs-b", input_amounts=["1000"]),
        ]
        assert groups_of(AmountMatchRule(Decimal("0.5")), observations) == []

    def test_exact_path_groups_and_tolerant_path_pairs(self):
        """A shared value is groupable; a bound is not."""
        exact = [
            make_observation("obs-%d" % index, input_amounts=["2"])
            for index in range(3)
        ]
        assert len(groups_of(AmountMatchRule(Decimal("0")), exact)) == 1

        tolerant = [
            make_observation("obs-%d" % index, input_amounts=["2.0%d" % index])
            for index in range(3)
        ]
        assert len(groups_of(AmountMatchRule(Decimal("0.1")), tolerant)) == 3

    def test_amount_facts_name_the_side(self):
        """Amount facts are built inline, recording which side(s) matched."""
        matches = groups_of(
            AmountMatchRule(),
            [
                make_observation("obs-a", input_amounts=["3"]),
                make_observation("obs-b", output_amounts=["3"]),
            ],
        )
        assert matches[0][1] == ("amount 3 (input, output)",)

    def test_confidence_reflects_whether_a_bound_was_used(self):
        assert AmountMatchRule(Decimal("0")).confidence.basis is ConfidenceBasis.EXACT_MATCH
        assert (
            AmountMatchRule(Decimal("0.1")).confidence.basis
            is ConfidenceBasis.DETERMINISTIC_RULE
        )

    def test_rejects_a_negative_tolerance(self):
        with pytest.raises(ValueError, match="negative"):
            AmountMatchRule(Decimal("-1"))

    def test_rationale_carries_no_financial_conclusion(self):
        rationale = AmountMatchRule().describe(("amount 3 (input)",))
        assert "carries no financial conclusion" in rationale

    def test_evidence_type(self):
        assert AmountMatchRule().evidence_type is EvidenceType.AMOUNT_MATCH

    def test_observation_without_a_blockchain_layer(self):
        observations = [make_observation("obs-a"), make_observation("obs-b")]
        assert groups_of(AmountMatchRule(), observations) == []


# --------------------------------------------------------------------------- #
# 6. provenance_link
# --------------------------------------------------------------------------- #


class TestProvenanceLinkRule:
    def test_matches_a_complete_identical_triple(self):
        observations = [
            make_observation("obs-a", ingestion_id="ing-9", source_file="a.csv", source_record="row 1"),
            make_observation("obs-b", ingestion_id="ing-9", source_file="a.csv", source_record="row 1"),
        ]
        assert matched_ids(ProvenanceLinkRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_a_shared_file_alone_does_not_match(self):
        observations = [
            make_observation("obs-a", source_file="a.csv", source_record="row 1"),
            make_observation("obs-b", source_file="a.csv", source_record="row 2"),
        ]
        assert groups_of(ProvenanceLinkRule(), observations) == []

    def test_a_shared_ingestion_run_alone_does_not_match(self):
        observations = [
            make_observation("obs-a", ingestion_id="ing-9", source_file="a.csv"),
            make_observation("obs-b", ingestion_id="ing-9", source_file="b.csv"),
        ]
        assert groups_of(ProvenanceLinkRule(), observations) == []

    @pytest.mark.parametrize(
        "ingestion_id,source_file,source_record",
        [
            ("", "a.csv", "row 1"),
            ("ing-9", "", "row 1"),
            ("ing-9", "a.csv", ""),
            ("   ", "a.csv", "row 1"),
            ("ing-9", "   ", "row 1"),
            ("ing-9", "a.csv", "   "),
        ],
    )
    def test_incomplete_provenance_does_not_match(
        self, ingestion_id, source_file, source_record
    ):
        observations = [
            make_observation(
                "obs-a",
                ingestion_id=ingestion_id,
                source_file=source_file,
                source_record=source_record,
            ),
            make_observation(
                "obs-b",
                ingestion_id=ingestion_id,
                source_file=source_file,
                source_record=source_record,
            ),
        ]
        assert groups_of(ProvenanceLinkRule(), observations) == []

    @pytest.mark.parametrize("missing", ["source_file", "source_record"])
    def test_absent_optional_locator_does_not_match(self, missing):
        kwargs_a = {"ingestion_id": "ing-9", "source_file": "a.csv", "source_record": "row 1"}
        kwargs_b = dict(kwargs_a)
        kwargs_b[missing] = None
        observations = [
            make_observation("obs-a", **kwargs_a),
            make_observation("obs-b", **kwargs_b),
        ]
        assert groups_of(ProvenanceLinkRule(), observations) == []

    def test_observation_without_provenance(self):
        observation = make_observation("obs-a")
        observation.provenance = None
        other = make_observation("obs-b")
        other.provenance = None
        assert groups_of(ProvenanceLinkRule(), [observation, other]) == []

    def test_facts_name_the_three_locators(self):
        rule = ProvenanceLinkRule()
        assert rule._facts(("ing-9", "a.csv", "row 1")) == (
            "source record ing-9 / a.csv / row 1",
        )

    def test_evidence_type(self):
        assert ProvenanceLinkRule().evidence_type is EvidenceType.PROVENANCE_LINK


# --------------------------------------------------------------------------- #
# 7. geo_coincidence
# --------------------------------------------------------------------------- #


class TestGeoCoinidenceRule:
    def test_matches_a_shared_country(self):
        observations = [
            make_observation("obs-a", geo_country="IN"),
            make_observation("obs-b", geo_country="IN"),
        ]
        assert matched_ids(GeoCoinidenceRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_country_comparison_normalises_case_and_spacing(self):
        observations = [
            make_observation("obs-a", geo_country=" in "),
            make_observation("obs-b", geo_country="IN"),
        ]
        assert matched_ids(GeoCoinidenceRule(), observations) == {frozenset({"obs-a", "obs-b"})}

    def test_distinct_countries_do_not_match(self):
        observations = [
            make_observation("obs-a", geo_country="IN"),
            make_observation("obs-b", geo_country="US"),
        ]
        assert groups_of(GeoCoinidenceRule(), observations) == []

    @pytest.mark.parametrize("country", [None, "", "   "])
    def test_missing_country_is_ignored(self, country):
        observations = [
            make_observation("obs-a", geo_country=country),
            make_observation("obs-b", geo_country=country),
        ]
        assert groups_of(GeoCoinidenceRule(), observations) == []

    def test_rationale_states_country_level_only(self):
        rationale = GeoCoinidenceRule().describe(("country value IN",))
        assert "establishes no shared location" in rationale

    def test_evidence_type(self):
        assert GeoCoinidenceRule().evidence_type is EvidenceType.GEO_COINCIDENCE

    def test_observation_without_a_network_layer(self):
        observations = [make_observation("obs-a"), make_observation("obs-b")]
        assert groups_of(GeoCoinidenceRule(), observations) == []


# --------------------------------------------------------------------------- #
# Cross-cutting rule behaviour
# --------------------------------------------------------------------------- #


class TestRuleContract:
    @pytest.mark.parametrize(
        "rule_id",
        [
            "same_txid",
            "same_address",
            "same_ip",
            "amount_match",
            "provenance_link",
            "geo_coincidence",
        ],
    )
    def test_every_default_rule_is_named_versioned_and_typed(self, rule_id):
        rules = {rule.rule_id: rule for rule in build_default_rules(CorrelationConfig())}
        rule = rules[rule_id]
        assert rule.rule_id == rule_id
        assert rule.rule_version == RULE_VERSION
        assert isinstance(rule.evidence_type, EvidenceType)
        assert rule.grouped is True

    def test_every_default_rule_produces_a_substantive_rationale(self):
        """The Step 4A model refuses a rationale under ten characters."""
        for rule in build_default_rules(CorrelationConfig()):
            rationale = rule.describe(("some matched value",))
            assert len(rationale.strip()) >= 10, rule.rule_id
            assert rationale.strip() == rationale, rule.rule_id

    def test_every_default_rule_ignores_an_empty_batch(self):
        for rule in build_default_rules(CorrelationConfig()):
            assert groups_of(rule, []) == [], rule.rule_id

    def test_rules_never_correlate_an_observation_with_itself(self):
        observation = make_observation(
            "obs-a",
            txid="a" * 64,
            src_ip="192.0.2.1",
            dst_ip="192.0.2.1",
            geo_country="IN",
            input_addresses=["bc1qaaaa"],
            input_amounts=["1"],
            timestamp=T0,
        )
        for rule in build_default_rules(CorrelationConfig()):
            assert groups_of(rule, [observation]) == [], rule.rule_id

    def test_rules_do_not_mutate_the_observations_they_read(self):
        observations = [
            make_observation(
                "obs-a",
                txid="a" * 64,
                src_ip="192.0.2.1",
                geo_country="IN",
                input_addresses=["bc1qaaaa"],
                input_amounts=["1"],
                timestamp=T0,
            ),
            make_observation(
                "obs-b",
                txid="a" * 64,
                src_ip="192.0.2.1",
                geo_country="IN",
                input_addresses=["bc1qaaaa"],
                input_amounts=["1"],
                timestamp=T0,
            ),
        ]
        before = [(o.observation_id, dict(o.extra)) for o in observations]
        rules = build_default_rules(CorrelationConfig(temporal_window=timedelta(seconds=5)))
        for rule in rules:
            list(rule.evaluate(observations))
        after = [(o.observation_id, dict(o.extra)) for o in observations]
        assert before == after
