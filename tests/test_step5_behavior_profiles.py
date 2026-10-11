"""Step 5 tests: deterministic behavioral profiles.

A profile is a descriptive aggregate over a stated set of observations.  The
tests here are written around the three things that make such an object
trustworthy -- that it measures exactly the set it says it measures, that it
means the same thing every time it is built, and that it has nowhere to record
a judgement even if someone wanted to record one.

The graph-backed cases use a graph produced by the locked correlation engine
rather than a hand-built stub, so structural grouping is exercised against the
relationships that really get recorded.
"""

from __future__ import annotations

import ast
import copy
import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

import pytest

from backend.app.behavior import profiles as profiles_module
from backend.app.behavior.features import BehaviorFeatureExtractor
from backend.app.behavior.models import FeatureCategory, FeatureScope
from backend.app.behavior.profiles import (
    BehaviorProfileBuilder,
    BehaviorProfileError,
    BehavioralProfile,
    ProfileGrouping,
    derive_profile_id,
    graph_structural_groups,
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
from backend.app.graph.models import EvidenceGraph, ObservationNode

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)

#: The number of features one profile carries.  The authoritative pinned list
#: of names lives in the feature tests; this number is a tripwire, so that
#: changing the size of the feature set fails here as well as there.
FEATURE_COUNT = 73


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
    asn: Any = None,
    geo_country: Optional[str] = None,
    txid: Optional[str] = None,
    input_addresses: Iterable[Any] = (),
    output_addresses: Iterable[Any] = (),
    input_amounts: Iterable[Any] = (),
    fee: Any = None,
    ingestion_id: str = "ing-1",
    source_file: Optional[str] = "data.csv",
    source_record: Optional[str] = None,
) -> UnifiedObservation:
    """Build one observation with the locked Step 3 model."""
    network = None
    if any(
        value is not None
        for value in (src_ip, dst_ip, src_port, asn, geo_country)
    ):
        network = NetworkObservation(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            asn=asn,
            geo_country=geo_country,
        )
    blockchain = None
    inputs_addr = list(input_addresses)
    outputs_addr = list(output_addresses)
    inputs_in = [Decimal(str(value)) for value in input_amounts]
    if (
        txid is not None
        or fee is not None
        or inputs_addr
        or outputs_addr
        or inputs_in
    ):
        blockchain = BlockchainTransaction(
            timestamp=timestamp,
            txid=txid,
            input_addresses=inputs_addr,
            output_addresses=outputs_addr,
            input_amounts=inputs_in,
            output_amounts=[],
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
            source_record=(
                source_record
                if source_record is not None
                else "row %s" % observation_id
            ),
            context={"declared": "csv"},
        ),
    )


def build_graph_for(observations: list[UnifiedObservation]) -> EvidenceGraph:
    """Build a recorded-evidence graph with the locked correlation layer."""
    return build_graph(list(CorrelationEngine().run(observations)))


def linked_pair() -> list[UnifiedObservation]:
    """Two records the correlation layer will join on their shared source IP."""
    return [
        make_observation("a-1", timestamp=T0, src_ip="10.0.0.1"),
        make_observation("a-2", timestamp=T0 + DAY, src_ip="10.0.0.1"),
    ]


def chained_triple() -> list[UnifiedObservation]:
    """Three records joined only through the middle one.

    ``a`` and ``b`` share a source IP, ``b`` and ``c`` share an address, and
    ``a`` and ``c`` share nothing -- so the only way they can end up in one
    component is by a traversal that follows relationships transitively.
    """
    return [
        make_observation("a", timestamp=T0, src_ip="10.0.0.1", input_addresses=["bc1qaaa"]),
        make_observation(
            "b", timestamp=T0 + DAY, src_ip="10.0.0.1", input_addresses=["bc1qzzz"]
        ),
        make_observation(
            "c", timestamp=T0 + 2 * DAY, src_ip="10.0.0.2", input_addresses=["bc1qaaa"]
        ),
    ]


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
                asn=15169 + (index % 4),
                geo_country="IN" if index % 2 else "US",
                txid="%064x" % (index % 700),
                input_addresses=["bc1q%08d" % (index % 900)],
                output_addresses=["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
                input_amounts=[Decimal("0.%03d" % (index % 999))],
                fee=Decimal("0.0001"),
                ingestion_id="ing-%d" % (index % 3),
                source_file="feed-%d.csv" % (index % 3),
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Identity derivation
# --------------------------------------------------------------------------- #


class TestProfileIdentifiers:
    def test_the_same_membership_and_basis_give_the_same_identifier(self):
        first = derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["a", "b"])
        second = derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["a", "b"])
        assert first == second
        assert first != derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["a", "c"])

    def test_input_order_cannot_reach_the_identifier(self):
        ordered = derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["a", "b", "c"])
        shuffled = derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["c", "a", "b"])
        assert ordered == shuffled

    def test_the_basis_is_part_of_the_identifier(self):
        """The same records grouped for two stated reasons are two profiles.

        Without the grouping in the payload, a caller could not tell two
        descriptions apart once they were stored, and the reason for the
        grouping -- the part a reader most needs -- would be lost.
        """
        single = derive_profile_id(ProfileGrouping.SINGLE_OBSERVATION, ["a"])
        collection = derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["a"])
        structural = derive_profile_id(ProfileGrouping.GRAPH_STRUCTURAL_GROUP, ["a"])
        assert len({single, collection, structural}) == 3

    def test_repeats_in_the_membership_collapse(self):
        with_repeat = derive_profile_id(
            ProfileGrouping.SUPPLIED_COLLECTION, ["a", "a", "b"]
        )
        without = derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["a", "b"])
        assert with_repeat == without

    def test_the_three_groupings_are_exactly_three_named_reasons(self):
        assert tuple(grouping.value for grouping in ProfileGrouping) == (
            "SINGLE_OBSERVATION",
            "SUPPLIED_COLLECTION",
            "GRAPH_STRUCTURAL_GROUP",
        )

    def test_no_grouping_claims_who_the_records_belong_to(self):
        rendered = " ".join(
            grouping.value.lower() for grouping in ProfileGrouping
        )
        for word in ("entity", "wallet", "account", "cluster", "actor"):
            assert word not in rendered


# --------------------------------------------------------------------------- #
# Structural grouping
# --------------------------------------------------------------------------- #


class TestStructuralGroups:
    def test_records_joined_by_a_relationship_form_one_group(self):
        records = linked_pair()
        groups = graph_structural_groups(build_graph_for(records))
        assert groups == (("a-1", "a-2"),)

    def test_joining_is_transitive(self):
        groups = graph_structural_groups(build_graph_for(chained_triple()))
        assert groups == (("a", "b", "c"),)

    def test_records_sharing_nothing_are_not_joined(self):
        records = [
            make_observation("x-1", timestamp=T0, src_ip="10.0.0.1", ingestion_id="ing-x"),
            make_observation("y-1", timestamp=T0, src_ip="10.9.9.9", ingestion_id="ing-y"),
        ]
        graph = build_graph_for(records)
        # Nothing was recorded between them, so there is nothing to join them
        # on.  A grouping that put them together would be this layer inventing
        # a connection the evidence layer never asserted.
        assert graph.edge_count() == 0
        assert graph_structural_groups(graph) == ()

    def test_every_graph_node_lands_in_exactly_one_group(self):
        records = synthetic(12)
        graph = build_graph_for(records)
        groups = graph_structural_groups(graph)
        members = [member for group in groups for member in group]
        assert sorted(members) == sorted(
            node.observation_id for node in graph.nodes()
        )
        assert len(members) == len(set(members))

    def test_direction_does_not_change_membership(self):
        """An edge says which side named which; it does not say what joined.

        Traversing only in the direction an edge was recorded would split a
        group in two for reasons that have nothing to do with the evidence, so
        the traversal is direction-agnostic by construction.
        """
        records = chained_triple()
        graph = build_graph_for(records)
        for edge in graph.edges():
            assert graph_structural_groups(graph) == (("a", "b", "c"),), edge

    def test_an_isolated_node_stands_alone(self):
        """A record nothing was recorded about still gets a group: itself."""
        graph = EvidenceGraph(
            observation_nodes=(ObservationNode.for_observation("iso-1"),),
            evidence_edges=(),
        )
        assert graph_structural_groups(graph) == (("iso-1",),)

    def test_the_result_is_ordered_and_repeatable(self):
        records = synthetic(9)
        graph = build_graph_for(records)
        first = graph_structural_groups(graph)
        assert first == graph_structural_groups(graph)
        assert tuple(sorted(first)) == first

    def test_a_non_graph_is_refused(self):
        with pytest.raises(BehaviorProfileError):
            graph_structural_groups("a graph, allegedly")

    def test_the_error_type_is_a_value_error(self):
        assert issubclass(BehaviorProfileError, ValueError)


# --------------------------------------------------------------------------- #
# Profile contents
# --------------------------------------------------------------------------- #


class TestProfileContents:
    def test_a_profile_carries_the_whole_feature_set(self):
        profile = BehaviorProfileBuilder().build(synthetic(3))
        assert profile.feature_count == FEATURE_COUNT
        assert profile.observation_count == 3
        assert profile.categories() == tuple(FeatureCategory)

    def test_every_feature_on_a_profile_is_scoped_to_that_profile(self):
        profile = BehaviorProfileBuilder().build(synthetic(2))
        assert all(
            feature.scope is FeatureScope.PROFILE for feature in profile.features
        )
        assert {feature.profile_id for feature in profile.features} == {
            profile.profile_id
        }
        assert all(feature.observation_id is None for feature in profile.features)

    def test_features_are_ordered_and_uniquely_named(self):
        profile = BehaviorProfileBuilder().build(synthetic(2))
        keys = [(feature.category.value, feature.name) for feature in profile.features]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)

    def test_a_named_feature_is_reachable_by_name(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        assert profile.feature("observation_count").value.value == 2
        assert profile.value_of("unique_src_ip_count") == 1
        assert profile.value_of("src_ip_values") == ("10.0.0.1",)

    def test_an_unknown_name_is_a_caller_error(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        with pytest.raises(KeyError) as excinfo:
            profile.feature("no_such_measurement")
        assert "no_such_measurement" in str(excinfo.value)

    def test_provenance_of_the_described_records_is_carried_up(self):
        records = [
            make_observation("p-1", ingestion_id="ing-a", source_file="a.csv"),
            make_observation("p-2", ingestion_id="ing-b", source_file="b.csv"),
        ]
        profile = BehaviorProfileBuilder().build(records)
        assert profile.provenance_ingestion_ids == ("ing-a", "ing-b")
        assert profile.provenance_source_files == ("a.csv", "b.csv")

    def test_recorded_relationships_are_listed_and_are_the_cited_ones(self):
        records = linked_pair()
        graph = build_graph_for(records)
        profile = BehaviorProfileBuilder(graph).build(records)
        assert profile.evidence_relationship_ids == tuple(
            sorted(edge.relationship_id for edge in graph.edges())
        )
        cited = {
            relationship_id
            for feature in profile.features
            for relationship_id in feature.origin.relationship_ids
        }
        assert cited == set(profile.evidence_relationship_ids)

    def test_the_serialised_profile_is_json_safe(self):
        profile = BehaviorProfileBuilder().build(synthetic(4))
        rendered = json.dumps(profile.to_dict())
        assert json.loads(rendered)["profile_id"] == profile.profile_id

    def test_the_serialised_keys_are_fixed(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        assert list(profile.to_dict()) == [
            "profile_id",
            "grouping",
            "group_id",
            "observation_count",
            "observation_ids",
            "feature_count",
            "features",
            "evidence_relationship_ids",
            "provenance_ingestion_ids",
            "provenance_source_files",
        ]

    def test_an_empty_profile_is_still_a_complete_answer(self):
        """A caller who asked about an empty set gets a profile, not ``None``.

        Every feature is present and every measurement that needs data says it
        is ``ABSENT``, so "nothing was found" is expressed as a description
        rather than as the absence of one.
        """
        profile = BehaviorProfileBuilder().build_empty()
        assert profile.observation_ids == ()
        assert profile.observation_count == 0
        assert profile.feature_count == FEATURE_COUNT
        assert profile.value_of("observation_count") == 0
        assert profile.value_of("total_input_amount") is None
        assert profile.grouping is ProfileGrouping.SUPPLIED_COLLECTION


# --------------------------------------------------------------------------- #
# Immutability and identity of the object
# --------------------------------------------------------------------------- #


class TestProfileImmutability:
    def test_nothing_can_be_assigned_after_construction(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        for name, value in (
            ("_features", ()),
            ("profile_id", derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["z"])),
            ("something_new", 1),
        ):
            with pytest.raises(AttributeError):
                setattr(profile, name, value)

    def test_there_is_no_attribute_dict_to_slip_things_into(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        assert not hasattr(profile, "__dict__")
        with pytest.raises(AttributeError):
            profile.extra_field = 1

    def test_two_builds_of_the_same_input_are_equal_and_hash_alike(self):
        records = synthetic(5)
        first = BehaviorProfileBuilder().build(records)
        second = BehaviorProfileBuilder().build(records)
        assert first == second
        assert hash(first) == hash(second)
        assert first.profile_id == second.profile_id

    def test_different_record_sets_are_different_profiles(self):
        one = BehaviorProfileBuilder().build([make_observation("a", src_ip="10.0.0.1")])
        two = BehaviorProfileBuilder().build([make_observation("b", src_ip="10.0.0.1")])
        assert one != two

    def test_a_profile_never_equals_something_that_is_not_one(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        assert profile != profile.to_dict()
        assert (profile == "profile") is False

    def test_the_public_surface_is_a_short_list_of_descriptions(self):
        public = {
            name
            for name in dir(BehavioralProfile)
            if not name.startswith("_")
        }
        assert public == {
            "categories",
            "describes_attribution",
            "evidence_relationship_ids",
            "feature",
            "feature_count",
            "features",
            "group_id",
            "grouping",
            "observation_count",
            "observation_ids",
            "profile_id",
            "provenance_ingestion_ids",
            "provenance_source_files",
            "to_dict",
            "value_of",
        }

    def test_the_repr_states_only_what_was_measured(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        rendered = repr(profile)
        assert profile.profile_id in rendered
        assert "observations=2" in rendered
        assert "features=%d" % FEATURE_COUNT in rendered


# --------------------------------------------------------------------------- #
# The constructor refuses a profile that disagrees with itself
# --------------------------------------------------------------------------- #


def _rebuild(profile: BehavioralProfile, **overrides: Any) -> BehavioralProfile:
    """Re-construct *profile* with selected fields replaced."""
    fields: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "grouping": profile.grouping,
        "group_id": profile.group_id,
        "observation_ids": profile.observation_ids,
        "features": profile.features,
        "evidence_relationship_ids": profile.evidence_relationship_ids,
        "provenance_ingestion_ids": profile.provenance_ingestion_ids,
        "provenance_source_files": profile.provenance_source_files,
    }
    fields.update(overrides)
    return BehavioralProfile(**fields)


class TestProfileSelfConsistency:
    def test_an_identifier_that_does_not_match_the_membership_is_refused(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        with pytest.raises(BehaviorProfileError) as excinfo:
            _rebuild(
                profile,
                profile_id=derive_profile_id(ProfileGrouping.SUPPLIED_COLLECTION, ["z"]),
            )
        assert "does not match" in str(excinfo.value)

    def test_an_identifier_that_is_not_a_uuid_is_refused(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        with pytest.raises(BehaviorProfileError):
            _rebuild(profile, profile_id="not-a-uuid")

    def test_unsorted_or_repeated_members_are_refused(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        with pytest.raises(BehaviorProfileError):
            _rebuild(profile, observation_ids=("a-2", "a-1"))
        with pytest.raises(BehaviorProfileError):
            _rebuild(profile, observation_ids=("a-1", "a-1"))

    def test_features_from_another_profile_are_refused(self):
        records = linked_pair()
        builder = BehaviorProfileBuilder()
        profile = builder.build(records)
        other_id = derive_profile_id(ProfileGrouping.SINGLE_OBSERVATION, ("a-1",))
        foreign = BehaviorFeatureExtractor().extract_observations(
            records, profile_id=other_id
        )
        with pytest.raises(BehaviorProfileError) as excinfo:
            _rebuild(profile, features=foreign)
        assert "references profile" in str(excinfo.value)

    def test_observation_scoped_features_are_refused(self):
        records = linked_pair()
        builder = BehaviorProfileBuilder()
        profile = builder.build(records)
        scoped = BehaviorFeatureExtractor().extract_observation(records[0])
        with pytest.raises(BehaviorProfileError) as excinfo:
            _rebuild(profile, features=scoped)
        assert "profile-scoped" in str(excinfo.value)

    def test_unsorted_features_are_refused(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        with pytest.raises(BehaviorProfileError) as excinfo:
            _rebuild(profile, features=tuple(reversed(profile.features)))
        assert "sorted" in str(excinfo.value)

    def test_a_cited_relationship_the_profile_omits_is_refused(self):
        records = linked_pair()
        graph = build_graph_for(records)
        profile = BehaviorProfileBuilder(graph).build(records)
        assert profile.evidence_relationship_ids
        with pytest.raises(BehaviorProfileError) as excinfo:
            _rebuild(profile, evidence_relationship_ids=())
        assert "does not list" in str(excinfo.value)

    def test_a_listed_relationship_no_feature_cites_is_refused(self):
        records = linked_pair()
        graph = build_graph_for(records)
        profile = BehaviorProfileBuilder(graph).build(records)
        with pytest.raises(BehaviorProfileError) as excinfo:
            _rebuild(
                profile,
                evidence_relationship_ids=(
                    "00000000-0000-0000-0000-000000000000",
                )
                + profile.evidence_relationship_ids,
            )
        assert "no feature cites" in str(excinfo.value)

    def test_a_feature_that_is_not_a_feature_is_refused(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        with pytest.raises(BehaviorProfileError):
            _rebuild(profile, features=("observation_count",))


# --------------------------------------------------------------------------- #
# Grouping: measured, never decided
# --------------------------------------------------------------------------- #


class TestGroupingSemantics:
    def test_one_record_is_a_single_observation_profile(self):
        profile = BehaviorProfileBuilder().build_for_observation(
            make_observation("solo", src_ip="10.0.0.1")
        )
        assert profile.grouping is ProfileGrouping.SINGLE_OBSERVATION
        assert profile.observation_ids == ("solo",)
        assert profile.observation_count == 1

    def test_a_supplied_collection_is_measured_as_given(self):
        records = [
            make_observation("m-1", src_ip="10.0.0.1"),
            make_observation("m-2", src_ip="10.9.9.9"),
        ]
        profile = BehaviorProfileBuilder().build(records)
        assert profile.grouping is ProfileGrouping.SUPPLIED_COLLECTION
        # The caller asked for these two together and the layer measured them
        # together; nothing here second-guessed the request.
        assert profile.observation_ids == ("m-1", "m-2")
        assert profile.value_of("unique_src_ip_count") == 2

    def test_a_caller_label_is_carried_unchanged(self):
        profile = BehaviorProfileBuilder().build(
            linked_pair(), group_id="case 2024-05 batch 3"
        )
        assert profile.group_id == "case 2024-05 batch 3"

    def test_a_structural_group_is_built_from_a_member_list(self):
        records = chained_triple()
        graph = build_graph_for(records)
        profile = BehaviorProfileBuilder(graph).build_for_graph_structural_group(
            ("a", "b", "c"), records
        )
        assert profile.grouping is ProfileGrouping.GRAPH_STRUCTURAL_GROUP
        assert profile.observation_ids == ("a", "b", "c")

    def test_a_structural_group_whose_records_are_missing_is_refused(self):
        records = chained_triple()
        graph = build_graph_for(records)
        with pytest.raises(BehaviorProfileError) as excinfo:
            BehaviorProfileBuilder(graph).build_for_graph_structural_group(
                ("a", "b", "c"), records[:2]
            )
        assert "c" in str(excinfo.value)

    def test_one_profile_per_component(self):
        records = [
            make_observation("a-1", timestamp=T0, src_ip="10.0.0.1", ingestion_id="ing-a"),
            make_observation(
                "a-2", timestamp=T0 + DAY, src_ip="10.0.0.1", ingestion_id="ing-a"
            ),
            make_observation("b-1", timestamp=T0, src_ip="10.9.9.9", ingestion_id="ing-b"),
            make_observation(
                "b-2", timestamp=T0 + DAY, src_ip="10.9.9.9", ingestion_id="ing-b"
            ),
        ]
        graph = build_graph_for(records)
        profiles = BehaviorProfileBuilder(graph).build_for_graph_structural_groups(records)
        assert [profile.observation_ids for profile in profiles] == [
            ("a-1", "a-2"),
            ("b-1", "b-2"),
        ]
        assert [profile.group_id for profile in profiles] == [
            "structural group 1",
            "structural group 2",
        ]
        assert all(
            profile.grouping is ProfileGrouping.GRAPH_STRUCTURAL_GROUP
            for profile in profiles
        )

    def test_a_record_the_graph_does_not_name_gets_no_structural_profile(self):
        """An unconnected record is not a group, and inventing one is refused.

        Nothing was recorded between this record and any other, so it is in no
        component.  Giving it a one-member "component" would present a grouping
        the evidence layer never asserted.
        """
        records = linked_pair() + [
            make_observation("lone", timestamp=T0, src_ip="10.7.7.7", ingestion_id="ing-z")
        ]
        graph = build_graph_for(records)
        assert not graph.has_node("lone")
        profiles = BehaviorProfileBuilder(graph).build_for_graph_structural_groups(records)
        assert [profile.observation_ids for profile in profiles] == [
            ("a-1", "a-2")
        ]
        # It can still be described, on its own terms, by the caller.
        alone = BehaviorProfileBuilder(graph).build_for_observation(records[2])
        assert alone.observation_ids == ("lone",)

    def test_a_component_is_never_described_by_fewer_records_than_it_holds(self):
        records = linked_pair()
        graph = build_graph_for(records)
        with pytest.raises(BehaviorProfileError) as excinfo:
            BehaviorProfileBuilder(graph).build_for_graph_structural_groups(records[:1])
        assert "a-2" in str(excinfo.value)

    def test_structural_groups_need_a_graph(self):
        with pytest.raises(BehaviorProfileError) as excinfo:
            BehaviorProfileBuilder().build_for_graph_structural_groups(linked_pair())
        assert "graph" in str(excinfo.value)

    def test_the_same_records_regrouped_keep_their_features(self):
        """The description follows the set, not the label put on it."""
        records = linked_pair()
        builder = BehaviorProfileBuilder()
        as_collection = builder.build(records)
        as_one = builder.build_for_observation(records[0])
        assert as_collection.value_of("observation_count") == 2
        assert as_one.value_of("observation_count") == 1
        assert as_collection.value_of("unique_src_ip_count") == 1
        assert as_one.value_of("unique_src_ip_count") == 1


# --------------------------------------------------------------------------- #
# Labels: free text, with a line drawn at a verdict
# --------------------------------------------------------------------------- #


class TestLabelPolicy:
    @pytest.mark.parametrize(
        "label",
        [
            "case 2024-05",
            "batch_3",
            "structural group 1",
            "run/2024-05-01:second pass",
            "investigation a",
        ],
    )
    def test_a_descriptive_label_is_accepted(self, label):
        profile = BehaviorProfileBuilder().build(linked_pair(), group_id=label)
        assert profile.group_id == label

    @pytest.mark.parametrize(
        "label",
        [
            "suspicious group",
            "high risk set",
            "the fraud ring",
            "anomaly cluster",
            "benign traffic",
            "known owner",
            "suspect account",
            "threat actors",
        ],
    )
    def test_a_judgement_in_a_label_is_refused(self, label):
        with pytest.raises(BehaviorProfileError) as excinfo:
            BehaviorProfileBuilder().build(linked_pair(), group_id=label)
        assert "descriptive" in str(excinfo.value)

    def test_whole_words_only(self):
        """``brisk`` is not ``risky``, but ``risk-scores`` is still a verdict."""
        assert BehaviorProfileBuilder().build(linked_pair(), group_id="brisk pace")
        with pytest.raises(BehaviorProfileError):
            BehaviorProfileBuilder().build(linked_pair(), group_id="risk-scores")

    def test_a_blank_label_is_refused(self):
        with pytest.raises(BehaviorProfileError):
            BehaviorProfileBuilder().build(linked_pair(), group_id="   ")

    def test_a_label_with_unexpected_characters_is_refused(self):
        with pytest.raises(BehaviorProfileError):
            BehaviorProfileBuilder().build(linked_pair(), group_id="<script>")

    def test_source_data_is_never_rewritten_by_the_label_check(self):
        """The check guards this layer's own text, not the evidence.

        A record whose source file happens to be named after a verdict is
        evidence, and preserving it as the source stated it is not this layer
        to revise.  Refusing the record would destroy evidence to protect a
        naming convention.
        """
        records = [
            make_observation(
                "p-1",
                src_ip="10.0.0.1",
                source_file="suspicious-traffic-2024.csv",
                source_record="row 7",
            )
        ]
        profile = BehaviorProfileBuilder().build(records, group_id="batch 1")
        assert profile.provenance_source_files == ("suspicious-traffic-2024.csv",)

    def test_no_label_at_all_is_fine(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        assert profile.group_id is None


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


class TestBuilderRefusals:
    def test_a_graph_of_the_wrong_type_is_refused(self):
        with pytest.raises(BehaviorProfileError):
            BehaviorProfileBuilder(graph="not a graph")

    def test_text_is_not_a_sequence_of_observations(self):
        with pytest.raises(BehaviorProfileError):
            BehaviorProfileBuilder().build("a-1")

    def test_something_uniterable_is_refused(self):
        with pytest.raises(BehaviorProfileError):
            BehaviorProfileBuilder().build(42)

    def test_a_non_observation_is_refused(self):
        with pytest.raises(BehaviorProfileError) as excinfo:
            BehaviorProfileBuilder().build(
                [make_observation("a-1"), {"observation_id": "a-2"}]
            )
        assert "position 1" in str(excinfo.value)

    def test_two_records_with_one_identifier_are_refused(self):
        """Choosing between them silently would discard a record."""
        records = [
            make_observation("same", src_ip="10.0.0.1"),
            make_observation("same", src_ip="10.0.0.2"),
        ]
        with pytest.raises(BehaviorProfileError) as excinfo:
            BehaviorProfileBuilder().build(records)
        assert "more than once" in str(excinfo.value)

    def test_building_does_not_mutate_the_records(self):
        records = synthetic(4)
        before = copy.deepcopy(records)
        BehaviorProfileBuilder().build(records)
        assert records == before

    def test_building_does_not_mutate_the_graph(self):
        records = linked_pair()
        graph = build_graph_for(records)
        before = graph.to_dict()
        BehaviorProfileBuilder(graph).build(records)
        BehaviorProfileBuilder(graph).build_for_graph_structural_groups(records)
        assert graph.to_dict() == before

    def test_the_builder_exposes_the_graph_it_was_given(self):
        graph = build_graph_for(linked_pair())
        assert BehaviorProfileBuilder(graph).graph is graph
        assert BehaviorProfileBuilder().graph is None

    def test_the_builder_public_surface_is_small(self):
        public = {
            name
            for name in dir(BehaviorProfileBuilder)
            if not name.startswith("_")
        }
        assert public == {
            "build",
            "build_empty",
            "build_for_graph_structural_group",
            "build_for_graph_structural_groups",
            "build_for_observation",
            "graph",
        }


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def test_shuffling_the_input_changes_nothing(self):
        records = synthetic(24)
        builder = BehaviorProfileBuilder()
        reference = builder.build(records).to_dict()
        for cut in range(6):
            reordered = records[cut:] + records[:cut]
            assert builder.build(reordered).to_dict() == reference

    def test_building_twice_gives_the_same_profile(self):
        records = synthetic(9)
        builder = BehaviorProfileBuilder()
        assert builder.build(records) == builder.build(records)

    def test_a_fresh_builder_agrees_with_a_reused_one(self):
        records = synthetic(9)
        reused = BehaviorProfileBuilder()
        reused.build(records)
        assert reused.build(records) == BehaviorProfileBuilder().build(records)

    def test_a_rebuilt_graph_gives_the_same_profiles(self):
        records = linked_pair()
        first = BehaviorProfileBuilder(build_graph_for(records)).build_for_graph_structural_groups(records)
        second = BehaviorProfileBuilder(build_graph_for(records)).build_for_graph_structural_groups(records)
        assert [profile.profile_id for profile in first] == [
            profile.profile_id for profile in second
        ]
        assert first == second

    def test_neither_module_asks_the_clock_or_a_random_number(self):
        """Determinism is structural here, not a claim about the inputs.

        A clock or a random source anywhere in the package would make a profile
        identifier depend on when it was built, which no test of the *values*
        could catch.
        """
        package = Path(profiles_module.__file__).parent
        volatile = {"random", "secrets", "time", "os", "uuid4"}
        offenders = []
        for source in sorted(package.glob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots = [node.module.split(".")[0]]
                else:
                    roots = []
                offenders.extend(
                    "%s:%d %s" % (source.name, node.lineno, root)
                    for root in roots
                    if root in volatile
                )
                if isinstance(node, ast.Attribute) and node.attr in {
                    "now",
                    "utcnow",
                    "today",
                    "perf_counter",
                    "monotonic",
                    "uuid4",
                }:
                    offenders.append("%s:%d .%s" % (source.name, node.lineno, node.attr))
        assert offenders == []

    def test_identifiers_are_derived_not_assigned(self):
        """The identifier is a function of membership, checkable by hand."""
        records = linked_pair()
        profile = BehaviorProfileBuilder().build(records)
        assert profile.profile_id == derive_profile_id(
            ProfileGrouping.SUPPLIED_COLLECTION, ("a-1", "a-2")
        )


# --------------------------------------------------------------------------- #
# What a profile cannot say
# --------------------------------------------------------------------------- #


class TestNoPartyOrJudgement:
    def test_the_party_question_has_a_negative_answer(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        assert profile.describes_attribution() is False

    def test_no_field_of_a_profile_names_a_party(self):
        profile = BehaviorProfileBuilder().build(synthetic(3))
        keys = list(profile.to_dict())
        for word in (
            "entity",
            "identity",
            "owner",
            "ownership",
            "actor",
            "wallet",
            "account",
            "name",
            "label",
            "score",
            "severity",
            "verdict",
            "rank",
        ):
            assert word not in keys, word

    def test_the_rendered_profile_contains_no_judgement(self):
        profile = BehaviorProfileBuilder().build(
            synthetic(4), group_id="batch 1"
        )
        rendered = json.dumps(profile.to_dict()).lower()
        for word in (
            "suspicious",
            "malicious",
            "fraud",
            "risky",
            "threat",
            "anomaly",
            "benign",
            "suspect",
        ):
            assert word not in rendered, word

    def test_a_structural_group_is_labelled_structurally(self):
        """The only label this layer writes is one about its own grouping."""
        records = chained_triple()
        graph = build_graph_for(records)
        profiles = BehaviorProfileBuilder(graph).build_for_graph_structural_groups(records)
        assert [profile.group_id for profile in profiles] == ["structural group 1"]
        assert "struct" in profiles[0].group_id

    def test_a_profile_answers_only_measurements(self):
        profile = BehaviorProfileBuilder().build(linked_pair())
        assert not hasattr(profile, "severity")
        assert not hasattr(profile, "confidence")
        assert not hasattr(profile, "interpretation")
        assert not hasattr(profile, "recommendation")


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #


class TestModerateScale:
    def test_a_thousand_records_produce_one_profile(self):
        records = synthetic(1000)
        builder = BehaviorProfileBuilder()
        started = time.perf_counter()
        profile = builder.build(records)
        elapsed = time.perf_counter() - started
        assert profile.observation_count == 1000
        assert profile.feature_count == FEATURE_COUNT
        assert profile.value_of("unique_src_ip_count") == 5 * 17
        # Generous, because this asserts "does not fall over", not "is fast".
        assert elapsed < 20.0, "profiling 1000 records took %.1fs" % elapsed

    def test_structural_profiles_scale_with_the_graph(self):
        records = synthetic(120)
        graph = build_graph_for(records)
        builder = BehaviorProfileBuilder(graph)
        started = time.perf_counter()
        profiles = builder.build_for_graph_structural_groups(records)
        elapsed = time.perf_counter() - started
        assert len(profiles) == len(graph_structural_groups(graph))
        assert all(profile.feature_count == FEATURE_COUNT for profile in profiles)
        assert elapsed < 20.0, "structural profiling took %.1fs" % elapsed

    def test_grouping_a_large_set_does_not_repeat_its_work(self):
        records = synthetic(200)
        graph = build_graph_for(records)
        builder = BehaviorProfileBuilder(graph)
        started = time.perf_counter()
        first = builder.build_for_graph_structural_groups(records)
        second = builder.build_for_graph_structural_groups(records)
        elapsed = time.perf_counter() - started
        assert first == second
        assert elapsed < 20.0, "rebuilding structural profiles took %.1fs" % elapsed
