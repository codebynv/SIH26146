"""Tests for the Step 4C evidence graph builder.

The builder is the only place where relationships become structure, so this file
checks the guarantees that make the structure trustworthy: the graph is a
faithful re-expression of the evidence and nothing more, refusals are loud rather
than silent, ordering is a function of content, and the input is left exactly as
it was found.

The last class builds from genuine Step 4B engine output, which is what makes the
unit tests above it an argument about the graph layer rather than about a
hand-built fiction.
"""

from __future__ import annotations

import copy
import json
import random
from datetime import datetime, timedelta, timezone
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
from backend.app.correlation.models import (
    Confidence,
    EvidenceRelationship,
    derive_relationship_id,
)
from backend.app.correlation.provenance import ProvenanceRef
from backend.app.correlation.rules import CorrelationConfig
from backend.app.graph import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceGraphBuilder,
    EvidenceGraphError,
    ObservationNode,
    build_graph,
    derive_node_id,
)

UTC = timezone.utc
NOW = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)

OBS_A = "obs-a"
OBS_B = "obs-b"
OBS_C = "obs-c"
OBS_D = "obs-d"


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def _ref(ingestion_id: str = "ing-1", record: str = "row 2") -> ProvenanceRef:
    return ProvenanceRef(
        ingestion_id=ingestion_id, source_file="evidence.csv", source_record=record
    )


def _relationship(**overrides) -> EvidenceRelationship:
    payload = {
        "relationship_id": derive_relationship_id({"graph-test": 1}),
        "source_observation_ids": (OBS_A,),
        "target_observation_ids": (OBS_B,),
        "evidence_type": EvidenceType.SAME_IP,
        "inference_level": InferenceLevel.CORRELATED,
        "confidence": Confidence.exact_match(),
        "rationale": "Both observations carry src_ip 203.0.113.5 as recorded.",
        "rule_id": "rule.same_ip",
        "rule_version": "1.0.0",
        "provenance_refs": (_ref(),),
        "created_at": NOW,
    }
    payload.update(overrides)
    return EvidenceRelationship(**payload)


def _chain(count: int) -> list[EvidenceRelationship]:
    """A chain of relationships, each linking one more observation."""
    ids = ["obs-%02d" % index for index in range(count)]
    return [
        _relationship(
            relationship_id=derive_relationship_id({"chain": index}),
            source_observation_ids=(ids[index],),
            target_observation_ids=(ids[index + 1],),
            evidence_type=list(EvidenceType)[index % len(EvidenceType)],
            rule_id="rule.chain.%d" % index,
            rule_version="1.%d.0" % index,
            rationale="Records %d and %d share a compared value as recorded."
            % (index, index + 1),
            provenance_refs=(_ref(ingestion_id="ing-%02d" % index),),
        )
        for index in range(count - 1)
    ]


def _star(count: int) -> list[EvidenceRelationship]:
    """One source linked to each of *count* distinct targets."""
    return [
        _relationship(
            relationship_id=derive_relationship_id({"star": index}),
            source_observation_ids=("obs-00",),
            target_observation_ids=("obs-%02d" % (index + 1),),
            rule_id="rule.star.%d" % index,
            rationale="Observation 00 and %02d share a compared value as recorded."
            % (index + 1),
        )
        for index in range(count)
    ]


def _corpus(count: int = 40, seed: int = 20240501) -> list[EvidenceRelationship]:
    """A larger, messier batch: shared nodes, repeats, varied evidence."""
    rng = random.Random(seed)
    ids = ["obs-%03d" % index for index in range(count)]
    relationships = []
    for index in range(count):
        source = ids[rng.randrange(count)]
        targets = tuple(
            ids[rng.randrange(count)] for _ in range(1 + rng.randrange(3))
        )
        targets = tuple(sorted({item for item in targets if item != source}))
        if not targets:
            continue
        relationships.append(
            _relationship(
                relationship_id=derive_relationship_id({"corpus": index}),
                source_observation_ids=(source,),
                target_observation_ids=targets,
                evidence_type=list(EvidenceType)[rng.randrange(len(EvidenceType))],
                inference_level=list(InferenceLevel)[rng.randrange(len(InferenceLevel))],
                confidence=Confidence.from_rule(round(rng.random(), 3)),
                rationale="Records %d and %s share a compared value as recorded."
                % (index, ", ".join(targets)),
                rule_id="rule.corpus.%d" % (index % 5),
                rule_version="1.%d.0" % (index % 3),
                provenance_refs=tuple(
                    _ref(ingestion_id="ing-%d" % position)
                    for position in range(1 + index % 3)
                ),
                created_at=NOW + timedelta(seconds=index),
            )
        )
    return relationships


def make_observation(
    observation_id: str,
    *,
    txid: Optional[str] = None,
    src_ip: Any = None,
    dst_ip: Any = None,
    geo_country: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    input_addresses: Iterable[str] = (),
    ingestion_id: Optional[str] = None,
    source_file: Optional[str] = None,
    source_record: Optional[str] = None,
) -> SimpleNamespace:
    """A duck-typed observation, structurally the shape Step 3 produces."""
    inputs = list(input_addresses)
    blockchain = None
    if txid is not None or inputs:
        blockchain = SimpleNamespace(
            txid=txid,
            input_addresses=inputs,
            output_addresses=[],
            input_amounts=[],
            output_amounts=[],
            fee=None,
        )
    network = None
    if any(value is not None for value in (src_ip, dst_ip, geo_country)):
        network = SimpleNamespace(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=None,
            dst_port=None,
            geo_country=geo_country,
            asn=None,
        )
    provenance = None
    if ingestion_id is not None:
        provenance = SimpleNamespace(
            ingestion_id=ingestion_id,
            source_file=source_file,
            source_record=source_record,
        )
    return SimpleNamespace(
        observation_id=observation_id,
        observation_type="BLOCKCHAIN_TX",
        timestamp=timestamp,
        network=network,
        blockchain=blockchain,
        provenance=provenance,
    )


@pytest.fixture
def builder() -> EvidenceGraphBuilder:
    return EvidenceGraphBuilder()


# --------------------------------------------------------------------------- #
# 1-3. Empty, single and multiple relationships
# --------------------------------------------------------------------------- #


class TestEmptyGraph:
    def test_no_relationships_gives_an_empty_graph(self, builder):
        graph = builder.build([])
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert graph.nodes() == ()
        assert graph.edges() == ()

    def test_an_empty_graph_serialises_to_an_empty_document(self, builder):
        assert builder.build([]).to_dict() == {
            "node_count": 0,
            "edge_count": 0,
            "nodes": [],
            "edges": [],
        }

    def test_an_empty_graph_answers_every_query_without_inventing_data(self, builder):
        graph = builder.build([])
        with pytest.raises(KeyError):
            graph.get_node(OBS_A)
        with pytest.raises(KeyError):
            graph.get_edge("anything")
        assert graph.nodes() == ()
        assert graph.edges() == ()


class TestSingleRelationship:
    def test_one_relationship_gives_one_edge(self, builder):
        graph = builder.build([_relationship()])
        assert graph.edge_count() == 1
        assert len(graph.edges()) == 1

    def test_one_relationship_gives_exactly_two_nodes(self, builder):
        graph = builder.build([_relationship()])
        assert graph.node_count() == 2
        assert [node.observation_id for node in graph.nodes()] == [OBS_A, OBS_B]

    def test_the_edge_preserves_the_relationship_exactly(self, builder):
        relationship = _relationship()
        edge = builder.build([relationship]).get_edge(relationship.relationship_id)
        assert edge.relationship_id == relationship.relationship_id
        assert edge.source_observation_ids == relationship.source_observation_ids
        assert edge.target_observation_ids == relationship.target_observation_ids
        assert edge.evidence_type is relationship.evidence_type
        assert edge.inference_level is relationship.inference_level
        assert edge.confidence == relationship.confidence
        assert edge.rationale == relationship.rationale
        assert edge.rule_id == relationship.rule_id
        assert edge.rule_version == relationship.rule_version
        assert edge.provenance_refs == relationship.provenance_refs
        assert edge.created_at == relationship.created_at

    def test_both_endpoints_are_addressable(self, builder):
        graph = builder.build([_relationship()])
        assert graph.get_node(OBS_A).observation_id == OBS_A
        assert graph.get_node(OBS_B).observation_id == OBS_B

    def test_node_ids_are_derived_from_the_observation_ids(self, builder):
        graph = builder.build([_relationship()])
        assert graph.get_node(OBS_A).node_id == derive_node_id(OBS_A)


class TestMultipleRelationships:
    def test_every_relationship_becomes_exactly_one_edge(self, builder):
        relationships = _chain(6)
        graph = builder.build(relationships)
        assert graph.edge_count() == len(relationships)

    def test_edges_and_relationships_are_in_bijection(self, builder):
        """No relationship is dropped, merged or duplicated by the build."""
        relationships = _corpus(30)
        graph = builder.build(relationships)
        assert {edge.relationship_id for edge in graph.edges()} == {
            relationship.relationship_id for relationship in relationships
        }
        assert len(graph.edges()) == len(relationships)

    def test_no_edge_is_built_that_no_relationship_stated(self, builder):
        relationships = _chain(5)
        graph = builder.build(relationships)
        expected = set()
        for relationship in relationships:
            expected.add(
                (
                    relationship.source_observation_ids,
                    relationship.target_observation_ids,
                )
            )
        actual = {
            (edge.source_observation_ids, edge.target_observation_ids)
            for edge in graph.edges()
        }
        assert actual == expected

    def test_a_node_exists_only_for_an_observation_some_edge_names(self, builder):
        relationships = _chain(4)
        graph = builder.build(relationships)
        named = {
            observation_id
            for relationship in relationships
            for observation_id in relationship.all_observation_ids
        }
        assert {node.observation_id for node in graph.nodes()} == named

    def test_a_mixed_batch_keeps_every_evidence_type(self, builder):
        relationships = [
            _relationship(
                relationship_id=derive_relationship_id({"mixed": index}),
                source_observation_ids=("obs-%02d" % index,),
                target_observation_ids=("obs-%02d" % (index + 1),),
                evidence_type=evidence_type,
            )
            for index, evidence_type in enumerate(EvidenceType)
        ]
        graph = builder.build(relationships)
        assert {edge.evidence_type for edge in graph.edges()} == set(EvidenceType)


# --------------------------------------------------------------------------- #
# 4-5. Multi-target relationships and shared nodes
# --------------------------------------------------------------------------- #


class TestMultiTargetRelationships:
    def test_several_targets_stay_on_one_edge(self, builder):
        relationship = _relationship(target_observation_ids=(OBS_B, OBS_C, OBS_D))
        graph = builder.build([relationship])
        assert graph.edge_count() == 1
        assert graph.get_edge(relationship.relationship_id).target_observation_ids == (
            OBS_B,
            OBS_C,
            OBS_D,
        )

    def test_every_target_becomes_a_node(self, builder):
        graph = builder.build(
            [_relationship(target_observation_ids=(OBS_B, OBS_C, OBS_D))]
        )
        assert graph.node_count() == 4
        for observation_id in (OBS_A, OBS_B, OBS_C, OBS_D):
            assert graph.get_node(observation_id).observation_id == observation_id

    def test_a_star_of_three_is_three_addressable_edges_not_one(self, builder):
        """The distinction the multi-target rule exists to protect."""
        relationships = _star(3)
        graph = builder.build(relationships)
        assert graph.edge_count() == 3
        for relationship in relationships:
            edge = graph.get_edge(relationship.relationship_id)
            assert edge.target_observation_ids == relationship.target_observation_ids
        assert len({edge.relationship_id for edge in graph.edges()}) == 3

    def test_multi_target_edges_are_not_collapsed_into_one(self, builder):
        relationships = _star(3)
        graph = builder.build(relationships)
        combined = tuple(
            sorted(
                observation_id
                for edge in graph.edges()
                for observation_id in edge.target_observation_ids
            )
        )
        assert combined == ("obs-01", "obs-02", "obs-03")
        assert all(len(edge.target_observation_ids) == 1 for edge in graph.edges())

    def test_a_wide_relationship_does_not_inflate_the_edge_count(self, builder):
        graph = builder.build([_relationship(target_observation_ids=(OBS_B, OBS_C, OBS_D))])
        assert graph.edge_count() == 1
        assert graph.node_count() == 4

    def test_a_multi_source_relationship_is_preserved(self, builder):
        relationship = _relationship(
            source_observation_ids=(OBS_A, OBS_C), target_observation_ids=(OBS_D,)
        )
        edge = builder.build([relationship]).get_edge(relationship.relationship_id)
        assert edge.source_observation_ids == (OBS_A, OBS_C)
        assert edge.observation_ids() == (OBS_A, OBS_C, OBS_D)


class TestSharedNodes:
    def test_one_observation_in_many_relationships_is_one_node(self, builder):
        relationships = _star(4)
        graph = builder.build(relationships)
        assert graph.node_count() == 5
        assert graph.edge_count() == 4

    def test_the_shared_node_is_the_same_object_every_time(self, builder):
        graph = builder.build(_star(3))
        assert graph.get_node("obs-00") is graph.get_node("obs-00")
        assert graph.get_node("obs-00").node_id == derive_node_id("obs-00")

    def test_a_dense_batch_does_not_duplicate_nodes(self, builder):
        relationships = _corpus(40)
        graph = builder.build(relationships)
        assert len({node.node_id for node in graph.nodes()}) == graph.node_count()
        assert len({node.observation_id for node in graph.nodes()}) == graph.node_count()

    def test_every_incident_edge_is_reachable_from_the_node(self, builder):
        relationships = _star(3)
        graph = builder.build(relationships)
        assert len(graph.outgoing_edges("obs-00")) == 3
        assert graph.incoming_edges("obs-00") == ()
        for index in range(1, 4):
            assert graph.incoming_edges("obs-%02d" % index) != ()

    def test_nodes_are_not_duplicated_across_the_two_sides(self, builder):
        graph = builder.build([_relationship(target_observation_ids=(OBS_B, OBS_C))])
        assert graph.node_count() == 3


# --------------------------------------------------------------------------- #
# 6-8. Refusals
# --------------------------------------------------------------------------- #


class TestDuplicateRelationshipIds:
    def test_a_repeated_relationship_id_is_refused(self, builder):
        relationship = _relationship()
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([relationship, relationship])
        assert relationship.relationship_id in str(excinfo.value)

    def test_two_different_relationships_may_not_share_an_id(self, builder):
        first = _relationship(target_observation_ids=(OBS_B,))
        second = _relationship(
            target_observation_ids=(OBS_C,), rule_id="rule.other"
        )
        second = second.model_copy(update={"relationship_id": first.relationship_id})
        with pytest.raises(EvidenceGraphError):
            builder.build([first, second])

    def test_the_nothing_is_silently_replaced_variant(self, builder):
        """Keeping either copy would make the graph depend on input order."""
        kept = _relationship(
            relationship_id="same-id", target_observation_ids=(OBS_B,)
        )
        replaced = _relationship(
            relationship_id="same-id",
            target_observation_ids=(OBS_C,),
            rule_id="rule.replacement",
        )
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([kept, replaced])
        assert "duplicate" in str(excinfo.value).lower()
        with pytest.raises(EvidenceGraphError):
            builder.build([replaced, kept])

    def test_the_error_names_the_first_position_of_the_duplicate(self, builder):
        relationship = _relationship()
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_relationship(relationship_id="other"), relationship, relationship])
        assert "position 1" in str(excinfo.value)

    def test_duplicate_ids_are_refused_however_far_apart_they_are(self, builder):
        relationships = _corpus(20)
        assert relationships
        with pytest.raises(EvidenceGraphError):
            builder.build(relationships + [relationships[0]])

    def test_a_shared_observation_is_not_a_duplicate(self, builder):
        """Many edges may share nodes; only shared *ids* are refused."""
        graph = builder.build(_star(3))
        assert graph.edge_count() == 3


class TestSelfEdgeRejection:
    def test_an_observation_on_both_sides_is_refused(self, builder):
        relationship = _relationship(
            source_observation_ids=(OBS_A,), target_observation_ids=(OBS_A,)
        )
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([relationship])
        assert OBS_A in str(excinfo.value)

    def test_a_self_reference_among_several_ids_is_refused(self, builder):
        relationship = _relationship(
            source_observation_ids=(OBS_A, OBS_C),
            target_observation_ids=(OBS_B, OBS_C),
        )
        with pytest.raises(EvidenceGraphError):
            builder.build([relationship])

    def test_a_self_edge_is_refused_even_alone_in_the_batch(self, builder):
        relationship = _relationship(
            source_observation_ids=(OBS_D,), target_observation_ids=(OBS_D,)
        )
        with pytest.raises(EvidenceGraphError):
            builder.build([relationship])

    def test_a_self_edge_is_refused_even_among_valid_relationships(self, builder):
        with pytest.raises(EvidenceGraphError):
            builder.build(
                [
                    _relationship(relationship_id=derive_relationship_id({"ok": 1})),
                    _relationship(
                        relationship_id=derive_relationship_id({"bad": 1}),
                        source_observation_ids=(OBS_C,),
                        target_observation_ids=(OBS_C,),
                    ),
                ]
            )

    def test_a_self_edge_that_bypassed_validation_is_still_refused(self, builder):
        """The check is the graph's, not a courtesy to Step 4A."""
        malformed = _unvalidated(
            source_observation_ids=(OBS_A,), target_observation_ids=(OBS_A,)
        )
        with pytest.raises(EvidenceGraphError):
            builder.build([malformed])

    def test_no_partial_graph_is_returned_when_a_self_edge_is_found(self, builder):
        with pytest.raises(EvidenceGraphError):
            builder.build(
                [
                    _relationship(relationship_id=derive_relationship_id({"a": 1})),
                    _relationship(
                        relationship_id=derive_relationship_id({"b": 1}),
                        source_observation_ids=(OBS_B,),
                        target_observation_ids=(OBS_B,),
                    ),
                ]
            )

    def test_two_distinct_observations_are_not_a_self_edge(self, builder):
        graph = builder.build(
            [_relationship(source_observation_ids=(OBS_A,), target_observation_ids=(OBS_B,))]
        )
        assert graph.edge_count() == 1


class TestMalformedInput:
    def test_a_non_relationship_is_refused(self, builder):
        for impostor in (None, "a relationship", 7, {"relationship_id": "x"}):
            with pytest.raises(EvidenceGraphError):
                builder.build([impostor])

    def test_a_none_batch_is_refused(self, builder):
        with pytest.raises(TypeError):
            builder.build(None)

    def test_a_text_batch_is_refused(self, builder):
        with pytest.raises(TypeError):
            builder.build("relationships")

    def test_a_non_iterable_batch_is_refused(self, builder):
        with pytest.raises(TypeError):
            builder.build(7)

    def test_a_blank_relationship_id_is_refused(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_unvalidated(relationship_id="  ")])
        assert "relationship_id" in str(excinfo.value)

    def test_a_missing_rule_id_is_refused(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_unvalidated(rule_id="")])
        assert "rule" in str(excinfo.value)

    def test_a_missing_rule_version_is_refused(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_unvalidated(rule_version="")])
        assert "version" in str(excinfo.value)

    def test_a_missing_rationale_is_refused(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_unvalidated(rationale="")])
        assert "rationale" in str(excinfo.value)

    def test_a_missing_evidence_type_is_refused(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_unvalidated(evidence_type=None)])
        assert "EvidenceType" in str(excinfo.value)

    def test_a_missing_inference_level_is_refused(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_unvalidated(inference_level="CORRELATED")])
        assert "InferenceLevel" in str(excinfo.value)

    def test_a_confidence_without_a_basis_is_refused(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_unvalidated(confidence=SimpleNamespace(value=0.9, basis=None))])
        assert "confidence" in str(excinfo.value)

    def test_a_confidence_outside_the_unit_interval_is_refused(self, builder):
        """A real ``Confidence`` object, carrying a value Step 4A would not allow."""
        out_of_range = Confidence.model_construct(
            value=1.7, basis=ConfidenceBasis.EXACT_MATCH
        )
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build([_unvalidated(confidence=out_of_range)])
        assert "unit interval" in str(excinfo.value)

    def test_an_ordering_with_no_window_is_refused(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build(
                [
                    _unvalidated(
                        temporal_window=None, temporal_relation=TemporalRelation.BEFORE
                    )
                ]
            )
        assert "window" in str(excinfo.value)

    def test_the_position_of_a_bad_item_is_reported(self, builder):
        with pytest.raises(EvidenceGraphError) as excinfo:
            builder.build(
                [
                    _relationship(relationship_id=derive_relationship_id({"ok": 1})),
                    _unvalidated(rule_id=""),
                ]
            )
        assert "position 1" in str(excinfo.value)

    def test_a_generator_of_relationships_is_accepted(self, builder):
        """Any iterable is fine; the builder materialises it once."""
        graph = builder.build(iter(_chain(4)))
        assert graph.edge_count() == 3
        assert graph.node_count() == 4

    def test_a_tuple_of_relationships_is_accepted(self, builder):
        graph = builder.build(tuple(_chain(4)))
        assert graph.edge_count() == 3


def _unvalidated(**overrides) -> EvidenceRelationship:
    """A relationship that never passed Step 4A's validators.

    ``model_construct`` skips validation entirely, which is the only realistic
    way a malformed relationship reaches the graph layer.  The field values are
    copied by hand rather than through ``model_dump()`` so that nested models
    stay models -- a dumped ``confidence`` would be a plain ``dict`` and would
    make every later check fail for the wrong reason.
    """
    relationship = _relationship()
    payload = {
        name: getattr(relationship, name)
        for name in EvidenceRelationship.model_fields
    }
    payload.update(overrides)
    return EvidenceRelationship.model_construct(**payload)


# --------------------------------------------------------------------------- #
# 19-22. Determinism
# --------------------------------------------------------------------------- #


class TestOrdering:
    def test_nodes_come_out_sorted_by_observation_id(self, builder):
        graph = builder.build(_corpus(25))
        ids = [node.observation_id for node in graph.nodes()]
        assert ids == sorted(ids)

    def test_nodes_are_sorted_regardless_of_input_order(self, builder):
        relationships = _chain(6)
        forwards = builder.build(relationships)
        backwards = builder.build(list(reversed(relationships)))
        assert forwards.nodes() == backwards.nodes()

    def test_edges_come_out_in_a_deterministic_order(self, builder):
        graph = builder.build(_corpus(30))
        keys = [
            (edge.source_observation_ids, edge.target_observation_ids, edge.relationship_id)
            for edge in graph.edges()
        ]
        assert keys == sorted(keys)

    def test_edge_order_does_not_depend_on_input_order(self, builder):
        relationships = _corpus(30)
        forwards = builder.build(relationships)
        backwards = builder.build(list(reversed(relationships)))
        assert [edge.edge_id for edge in forwards.edges()] == [
            edge.edge_id for edge in backwards.edges()
        ]

    def test_adding_relationships_inserts_nodes_without_reordering_the_rest(
        self, builder
    ):
        """Node order is a total order over ids, so a new node lands in place."""
        relationships = _corpus(20)
        small = builder.build(relationships[:5])
        large = builder.build(relationships)
        small_ids = [node.observation_id for node in small.nodes()]
        large_ids = [node.observation_id for node in large.nodes()]
        assert set(small_ids) <= set(large_ids)
        # Every id the larger graph holds is still in sorted order, and the
        # smaller graph's order is the smaller graph's subsequence of it.
        assert large_ids == sorted(large_ids)
        assert [item for item in large_ids if item in set(small_ids)] == small_ids

    def test_lookup_tables_agree_with_the_public_sequences(self, builder):
        graph = builder.build(_corpus(20))
        assert {node.observation_id for node in graph.nodes()} == set(
            graph._by_observation
        )
        assert {edge.relationship_id for edge in graph.edges()} == set(
            graph._by_relationship
        )


class TestShuffledInputEquivalence:
    def test_every_permutation_yields_the_same_graph(self, builder):
        relationships = _corpus(12, seed=7)
        reference = builder.build(relationships)
        rng = random.Random(99)
        for _ in range(12):
            shuffled = list(relationships)
            rng.shuffle(shuffled)
            assert builder.build(shuffled) == reference

    def test_shuffling_changes_nothing_that_is_visible(self, builder):
        relationships = _corpus(15, seed=11)
        reference = builder.build(relationships)
        shuffled = list(relationships)
        random.Random(3).shuffle(shuffled)
        other = builder.build(shuffled)

        assert other.node_count() == reference.node_count()
        assert other.edge_count() == reference.edge_count()
        assert [node.node_id for node in other.nodes()] == [
            node.node_id for node in reference.nodes()
        ]
        assert [edge.edge_id for edge in other.edges()] == [
            edge.edge_id for edge in reference.edges()
        ]
        assert json.dumps(other.to_dict()) == json.dumps(reference.to_dict())

    def test_node_ids_are_unaffected_by_shuffling(self, builder):
        relationships = _chain(6)
        reference = builder.build(relationships)
        shuffled = list(relationships)
        random.Random(5).shuffle(shuffled)
        assert {
            node.node_id for node in builder.build(shuffled).nodes()
        } == {node.node_id for node in reference.nodes()}

    def test_the_source_list_is_not_perturbed_by_shuffling(self, builder):
        relationships = _corpus(10, seed=13)
        reference = builder.build(relationships)
        shuffled = list(relationships)
        random.Random(17).shuffle(shuffled)
        assert builder.build(shuffled).to_dict() == reference.to_dict()


class TestRepeatedBuildEquivalence:
    def test_two_builds_of_the_same_input_are_equal(self, builder):
        relationships = _corpus(20, seed=21)
        assert builder.build(relationships) == builder.build(relationships)

    def test_two_builders_produce_the_same_graph(self):
        relationships = _corpus(20, seed=23)
        assert EvidenceGraphBuilder().build(relationships) == EvidenceGraphBuilder().build(
            relationships
        )

    def test_one_builder_used_twice_is_stable(self, builder):
        relationships = _chain(6)
        first = builder.build(relationships)
        second = builder.build(relationships)
        assert first == second
        assert json.dumps(first.to_dict()) == json.dumps(second.to_dict())

    def test_the_module_function_agrees_with_the_class(self, builder):
        relationships = _corpus(15, seed=27)
        assert build_graph(relationships) == builder.build(relationships)

    def test_a_fresh_builder_matches_a_used_one(self, builder):
        relationships = _corpus(15, seed=29)
        used = builder.build(relationships)
        assert EvidenceGraphBuilder().build(relationships) == used

    def test_repeated_builds_never_restamp_a_timestamp(self, builder):
        relationships = _corpus(10, seed=31)
        first = sorted(
            (edge.relationship_id, edge.created_at)
            for edge in builder.build(relationships).edges()
        )
        second = sorted(
            (edge.relationship_id, edge.created_at)
            for edge in builder.build(relationships).edges()
        )
        assert first == second
        assert first == sorted(
            (item.relationship_id, item.created_at) for item in relationships
        )

    def test_repeated_builds_never_mint_new_identifiers(self, builder):
        relationships = _corpus(10, seed=33)
        first = builder.build(relationships)
        second = builder.build(relationships)
        assert [node.node_id for node in first.nodes()] == [
            node.node_id for node in second.nodes()
        ]
        assert [edge.edge_id for edge in first.edges()] == [
            edge.edge_id for edge in second.edges()
        ]

    def test_determinism_holds_across_many_seeds(self, builder):
        for seed in range(1, 6):
            relationships = _corpus(12, seed=seed)
            shuffled = list(relationships)
            random.Random(seed).shuffle(shuffled)
            assert builder.build(shuffled) == builder.build(relationships)


# --------------------------------------------------------------------------- #
# 23. Input immutability
# --------------------------------------------------------------------------- #


class TestInputImmutability:
    def test_the_relationship_objects_are_untouched(self, builder):
        relationships = _corpus(20, seed=41)
        before = copy.deepcopy(relationships)
        builder.build(relationships)
        assert relationships == before

    def test_no_relationship_attribute_is_reassigned(self, builder):
        relationships = _chain(5)
        identities = [(id(item), id(item.confidence)) for item in relationships]
        builder.build(relationships)
        assert identities == [(id(item), id(item.confidence)) for item in relationships]

    def test_the_callers_list_is_untouched(self, builder):
        relationships = _chain(5)
        snapshot = list(relationships)
        builder.build(relationships)
        assert relationships == snapshot
        assert len(relationships) == len(snapshot)

    def test_the_provenance_citations_are_the_same_objects_afterwards(self, builder):
        relationship = _relationship(
            provenance_refs=(_ref(ingestion_id="ing-x"), _ref(ingestion_id="ing-y"))
        )
        citations = list(relationship.provenance_refs)
        builder.build([relationship])
        assert list(relationship.provenance_refs) == citations
        assert all(
            new is old for new, old in zip(relationship.provenance_refs, citations)
        )

    def test_the_same_batch_can_be_built_repeatedly(self, builder):
        relationships = _corpus(20, seed=43)
        first = builder.build(relationships)
        second = builder.build(relationships)
        assert first == second

    def test_building_from_a_copy_gives_the_same_graph(self, builder):
        relationships = _corpus(15, seed=47)
        assert builder.build(relationships) == builder.build(copy.deepcopy(relationships))

    def test_the_edge_holds_immutable_id_sequences(self, builder):
        """Nothing a caller still holds can change what the edge says.

        Step 4A hands the builder a tuple, and the builder copies it into another
        tuple, so there is no list anywhere in the chain for either of them to
        reach through.
        """
        relationship = _relationship(
            source_observation_ids=[OBS_A], target_observation_ids=[OBS_B]
        )
        edge = builder.build([relationship]).get_edge(relationship.relationship_id)
        assert isinstance(edge.source_observation_ids, tuple)
        assert isinstance(edge.target_observation_ids, tuple)
        with pytest.raises(TypeError):
            edge.source_observation_ids[0] = OBS_C  # type: ignore[index]

    def test_the_build_cannot_reach_back_through_the_input(self, builder):
        """Edges reference observations by id, never by object identity."""
        relationships = _chain(5)
        graph = builder.build(relationships)
        for edge in graph.edges():
            for observation_id in edge.observation_ids():
                assert isinstance(observation_id, str)
                assert graph.get_node(observation_id).observation_id == observation_id


# --------------------------------------------------------------------------- #
# 25. Query API
# --------------------------------------------------------------------------- #


class TestQueryApi:
    def test_counts_match_the_sequences(self, builder):
        graph = builder.build(_corpus(20, seed=53))
        assert graph.node_count() == len(graph.nodes())
        assert graph.edge_count() == len(graph.edges())

    def test_get_node_returns_the_node_for_an_observation(self, builder):
        graph = builder.build(_star(3))
        node = graph.get_node("obs-02")
        assert isinstance(node, ObservationNode)
        assert node.observation_id == "obs-02"

    def test_get_node_raises_for_an_unknown_observation(self, builder):
        graph = builder.build(_chain(4))
        with pytest.raises(KeyError):
            graph.get_node("obs-nothing")

    def test_get_edge_returns_the_edge_for_a_relationship(self, builder):
        relationships = _star(3)
        graph = builder.build(relationships)
        for relationship in relationships:
            edge = graph.get_edge(relationship.relationship_id)
            assert isinstance(edge, EvidenceEdge)
            assert edge.relationship_id == relationship.relationship_id

    def test_get_edge_raises_for_an_unknown_relationship(self, builder):
        graph = builder.build(_chain(4))
        with pytest.raises(KeyError):
            graph.get_edge("no-such-relationship")

    def test_has_node_and_has_edge_answer_without_raising(self, builder):
        relationships = _chain(4)
        graph = builder.build(relationships)
        assert graph.has_node("obs-00") is True
        assert graph.has_node("obs-99") is False
        assert graph.has_edge(relationships[0].relationship_id) is True
        assert graph.has_edge("nope") is False

    def test_a_node_is_reachable_by_its_graph_id(self, builder):
        graph = builder.build(_star(2))
        node = graph.get_node("obs-00")
        assert graph.get_node_by_id(node.node_id) is node

    def test_an_edge_is_reachable_by_its_graph_id(self, builder):
        relationship = _relationship()
        graph = builder.build([relationship])
        edge = graph.get_edge(relationship.relationship_id)
        assert graph.get_edge_by_id(edge.edge_id) is edge

    def test_outgoing_edges_follow_the_stated_direction(self, builder):
        graph = builder.build(_chain(5))
        assert len(graph.outgoing_edges("obs-00")) == 1
        assert graph.outgoing_edges("obs-00")[0].target_observation_ids == ("obs-01",)
        assert graph.outgoing_edges("obs-04") == ()

    def test_incoming_edges_follow_the_stated_direction(self, builder):
        graph = builder.build(_chain(5))
        assert len(graph.incoming_edges("obs-01")) == 1
        assert graph.incoming_edges("obs-01")[0].source_observation_ids == ("obs-00",)
        assert graph.incoming_edges("obs-00") == ()

    def test_outgoing_and_incoming_never_overlap(self, builder):
        graph = builder.build(_corpus(20, seed=59))
        for node in graph.nodes():
            outgoing = {edge.edge_id for edge in graph.outgoing_edges(node.observation_id)}
            incoming = {edge.edge_id for edge in graph.incoming_edges(node.observation_id)}
            assert outgoing & incoming == set()

    def test_neighbours_include_both_directions(self, builder):
        graph = builder.build(_chain(5))
        assert graph.neighbors("obs-02") == ("obs-01", "obs-03")

    def test_neighbours_exclude_the_node_itself(self, builder):
        """Only meaningful because self-edges are refused at construction."""
        graph = builder.build(_chain(4))
        for node in graph.nodes():
            assert node.observation_id not in graph.neighbors(node.observation_id)

    def test_neighbours_of_an_endpoint(self, builder):
        graph = builder.build(_chain(4))
        assert graph.neighbors("obs-00") == ("obs-01",)
        assert graph.neighbors("obs-03") == ("obs-02",)

    def test_neighbours_are_sorted_and_deduplicated(self, builder):
        graph = builder.build(_star(4))
        neighbours = graph.neighbors("obs-00")
        assert neighbours == tuple(sorted(set(neighbours)))
        assert len(neighbours) == 4

    def test_a_multi_target_edge_contributes_one_outgoing_entry(self, builder):
        graph = builder.build([_relationship(target_observation_ids=(OBS_B, OBS_C, OBS_D))])
        assert len(graph.outgoing_edges(OBS_A)) == 1
        assert graph.neighbors(OBS_A) == (OBS_B, OBS_C, OBS_D)

    def test_queries_about_an_unknown_observation_raise(self, builder):
        graph = builder.build(_chain(4))
        for query in (graph.outgoing_edges, graph.incoming_edges, graph.neighbors):
            with pytest.raises(KeyError):
                query("obs-nothing")

    def test_queries_do_not_mutate_the_graph(self, builder):
        graph = builder.build(_corpus(20, seed=61))
        before = json.dumps(graph.to_dict())
        for node in graph.nodes():
            graph.neighbors(node.observation_id)
            graph.outgoing_edges(node.observation_id)
            graph.incoming_edges(node.observation_id)
        assert json.dumps(graph.to_dict()) == before

    def test_queries_return_the_graph_s_own_objects(self, builder):
        graph = builder.build(_chain(4))
        assert graph.get_edge_by_id(graph.edges()[0].edge_id) is graph.edges()[0]
        assert graph.get_node_by_id(graph.nodes()[0].node_id) is graph.nodes()[0]

    def test_the_graph_offers_no_analysis_method(self, builder):
        graph = builder.build(_chain(4))
        for banned in (
            "shortest_path",
            "path_between",
            "betweenness",
            "pagerank",
            "centrality",
            "cluster",
            "components",
            "detect_anomalies",
            "risk_score",
            "propagate",
            "investigation_path",
        ):
            assert not hasattr(graph, banned), banned


# --------------------------------------------------------------------------- #
# The boundary: evidence in, structure out
# --------------------------------------------------------------------------- #


class TestNoRelationshipIsInvented:
    def test_every_edge_states_the_evidence_of_exactly_one_relationship(self, builder):
        relationships = _corpus(30, seed=67)
        graph = builder.build(relationships)
        by_id = {item.relationship_id: item for item in relationships}
        for edge in graph.edges():
            source = by_id[edge.relationship_id]
            for name in (
                "evidence_type",
                "inference_level",
                "confidence",
                "rationale",
                "rule_id",
                "rule_version",
                "provenance_refs",
                "temporal_window",
                "temporal_relation",
            ):
                assert getattr(edge, name) == getattr(source, name), name

    def test_no_inference_level_is_ever_raised(self, builder):
        order = {
            InferenceLevel.OBSERVED: 0,
            InferenceLevel.CORRELATED: 1,
            InferenceLevel.INFERRED: 2,
            InferenceLevel.REQUIRES_VERIFICATION: 3,
        }
        relationships = [
            _relationship(
                relationship_id=derive_relationship_id({"level": index}),
                source_observation_ids=("obs-%02d" % index,),
                target_observation_ids=("obs-%02d" % (index + 1),),
                inference_level=level,
            )
            for index, level in enumerate(InferenceLevel)
        ]
        graph = builder.build(relationships)
        stated = {
            item.relationship_id: item.inference_level for item in relationships
        }
        for edge in graph.edges():
            assert edge.inference_level is stated[edge.relationship_id]
            assert order[edge.inference_level] <= order[
                stated[edge.relationship_id]
            ]

    def test_every_inference_level_survives_the_build_unchanged(self, builder):
        relationships = [
            _relationship(
                relationship_id=derive_relationship_id({"level": index}),
                source_observation_ids=("obs-%02d" % index,),
                target_observation_ids=("obs-%02d" % (index + 1),),
                inference_level=level,
            )
            for index, level in enumerate(InferenceLevel)
        ]
        graph = builder.build(relationships)
        assert {edge.inference_level for edge in graph.edges()} == set(InferenceLevel)

    def test_correlated_stays_correlated_through_the_whole_graph(self, builder):
        graph = builder.build(_chain(6))
        assert {edge.inference_level for edge in graph.edges()} == {
            InferenceLevel.CORRELATED
        }

    def test_adjacency_implies_no_edge(self, builder):
        """Two nodes being adjacent in the structure is not itself evidence.

        Every neighbour must be adjacent *because* an edge names the pair, in
        one direction or the other.  If the graph ever added an edge for mere
        adjacency, this is where it would show.
        """
        relationships = _chain(5)
        graph = builder.build(relationships)
        linked = {
            frozenset(edge.source_observation_ids + edge.target_observation_ids)
            for edge in graph.edges()
        }
        for node in graph.nodes():
            for neighbour in graph.neighbors(node.observation_id):
                assert frozenset({node.observation_id, neighbour}) in linked, (
                    node.observation_id,
                    neighbour,
                )

    def test_nodes_carry_no_derived_property(self, builder):
        """Degree, centrality and risk would all be node properties."""
        graph = builder.build(_star(4))
        hub = graph.get_node("obs-00")
        assert hub.model_dump() == {
            "node_id": hub.node_id,
            "observation_id": "obs-00",
        }

    def test_the_graph_makes_no_attribution_claim(self, builder):
        graph = builder.build(_corpus(20, seed=71))
        assert graph.explains_attribution() is False
        for edge in graph.edges():
            assert edge.explains_attribution() is False


# --------------------------------------------------------------------------- #
# Against real Step 4B output
# --------------------------------------------------------------------------- #


def _engine_corpus(count: int = 40, seed: int = 20240501) -> list[SimpleNamespace]:
    """Observations arranged so several distinct rules fire."""
    rng = random.Random(seed)
    return [
        make_observation(
            "obs-%03d" % index,
            txid="tx%03d" % rng.randrange(0, 25),
            src_ip="10.1.%d.%d" % (rng.randrange(0, 4), rng.randrange(0, 8)),
            dst_ip="10.2.%d.%d" % (rng.randrange(0, 4), rng.randrange(0, 8)),
            geo_country="C%02d" % rng.randrange(0, 8),
            input_addresses=["bc1q%05d" % rng.randrange(0, 40)],
            ingestion_id="ing-%02d" % rng.randrange(0, 12),
            source_file="f%02d.csv" % rng.randrange(0, 8),
            source_record="row-%02d" % rng.randrange(0, 20),
            timestamp=NOW + timedelta(minutes=rng.randrange(0, 120)),
        )
        for index in range(count)
    ]


class TestAgainstRealEngineOutput:
    def _relationships(self, count: int = 40, seed: int = 20240501):
        config = CorrelationConfig(now=NOW)
        return CorrelationEngine(config).run(_engine_corpus(count, seed))

    def test_the_test_corpus_really_produces_relationships(self):
        relationships = self._relationships()
        assert len(relationships) > 10
        assert len({item.evidence_type for item in relationships}) >= 4

    def test_real_output_builds_a_complete_graph(self, builder):
        relationships = self._relationships()
        graph = builder.build(relationships)
        assert graph.edge_count() == len(relationships)
        named = {
            observation_id
            for relationship in relationships
            for observation_id in relationship.all_observation_ids
        }
        assert {node.observation_id for node in graph.nodes()} == named

    def test_real_output_preserves_every_edge(self, builder):
        relationships = self._relationships()
        graph = builder.build(relationships)
        by_id = {item.relationship_id: item for item in relationships}
        for edge in graph.edges():
            assert edge.to_dict()["rule_id"] == by_id[edge.relationship_id].rule_id
            assert edge.evidence_type is by_id[edge.relationship_id].evidence_type
            assert edge.confidence == by_id[edge.relationship_id].confidence

    def test_real_output_contains_multi_target_relationships(self, builder):
        relationships = self._relationships()
        assert any(len(item.target_observation_ids) > 1 for item in relationships)
        graph = builder.build(relationships)
        assert any(len(edge.target_observation_ids) > 1 for edge in graph.edges())

    def test_real_output_contains_temporal_detail(self, builder):
        config = CorrelationConfig(temporal_window=timedelta(minutes=5), now=NOW)
        relationships = CorrelationEngine(config).run(_engine_corpus())
        assert any(item.temporal_window is not None for item in relationships)
        graph = builder.build(relationships)
        assert any(edge.temporal_window is not None for edge in graph.edges())

    def test_real_output_never_needs_a_self_edge(self, builder):
        """Step 4B keeps the two sides disjoint, so the graph's refusal is
        a backstop rather than a routine outcome."""
        relationships = self._relationships()
        assert not any(item.is_self_referential for item in relationships)
        assert builder.build(relationships).edge_count() == len(relationships)

    def test_real_output_is_deterministic_under_shuffling(self, builder):
        relationships = self._relationships()
        shuffled = list(relationships)
        random.Random(2024).shuffle(shuffled)
        assert builder.build(shuffled) == builder.build(relationships)
        assert json.dumps(builder.build(shuffled).to_dict()) == json.dumps(
            builder.build(relationships).to_dict()
        )

    def test_real_output_is_repeatable(self, builder):
        assert self._relationships() == self._relationships()
        assert builder.build(self._relationships()) == builder.build(self._relationships())

    def test_the_observations_are_not_mutated_by_the_build(self, builder):
        """The graph layer is given relationships, not observations, so it has
        no handle on the substrate at all."""
        corpus = _engine_corpus()
        before = [
            (
                item.observation_id,
                copy.deepcopy(item.__dict__),
            )
            for item in corpus
        ]
        relationships = CorrelationEngine(CorrelationConfig(now=NOW)).run(corpus)
        builder.build(relationships)
        assert before == [
            (item.observation_id, copy.deepcopy(item.__dict__)) for item in corpus
        ]

    def test_no_engine_relationship_is_rewritten_as_a_stronger_one(self, builder):
        relationships = self._relationships()
        graph = builder.build(relationships)
        assert {edge.inference_level for edge in graph.edges()} == {
            InferenceLevel.CORRELATED
        }

    def test_the_graph_reports_only_structure_about_real_output(self, builder):
        graph = builder.build(self._relationships())
        total_incident = sum(
            len(graph.outgoing_edges(node.observation_id))
            + len(graph.incoming_edges(node.observation_id))
            for node in graph.nodes()
        )
        # Every edge touches at least two nodes, so incidences exceed the count.
        assert total_incident >= 2 * graph.edge_count()
        assert graph.explains_attribution() is False


class TestGraphType:
    def test_build_returns_an_evidence_graph(self, builder):
        assert isinstance(builder.build([_relationship()]), EvidenceGraph)
        assert isinstance(builder.build([]), EvidenceGraph)

    def test_an_evidence_graph_can_be_built_directly_from_edges(self):
        """The models stand alone: the builder is a convenience, not a gate."""
        relationship = _relationship()
        edge = EvidenceEdge.from_relationship(relationship)
        graph = EvidenceGraph(
            observation_nodes=(
                ObservationNode.for_observation(OBS_A),
                ObservationNode.for_observation(OBS_B),
            ),
            evidence_edges=(edge,),
        )
        assert graph.edge_count() == 1
        assert graph.get_edge(relationship.relationship_id) is edge
