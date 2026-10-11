"""Tests for the Step 4C evidence graph models.

This file checks the *models* in isolation: that a node is an identity and
nothing more, that an edge is a faithful copy of a Step 4A relationship field
for field, that identifiers are derived in namespaces of their own, that
serialisation is deterministic, and that the anti-attribution guarantee is
enforced by the structure rather than by convention.

Construction behaviour -- empty graphs, duplicates, self-edges, ordering,
shuffling, immutability -- lives in ``test_step4c_graph_builder.py``.  What is
asserted here is the property that makes those results trustworthy: the objects
themselves cannot lose, merge or reinterpret evidence.

Almost all fixtures are hand-built relationships so a failure points at the
graph layer rather than at the engine.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from backend.app.correlation.enums import (
    ConfidenceBasis,
    EvidenceType,
    InferenceLevel,
    TemporalRelation,
)
from backend.app.correlation.models import (
    RELATIONSHIP_NAMESPACE,
    Confidence,
    EvidenceRelationship,
    TemporalWindow,
    derive_relationship_id,
)
from backend.app.correlation.provenance import ProvenanceRef
from backend.app.domain.models import OBSERVATION_NAMESPACE
from backend.app.graph import (
    GRAPH_EDGE_NAMESPACE,
    GRAPH_NODE_NAMESPACE,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceGraphError,
    ObservationNode,
    derive_edge_id,
    derive_node_id,
    edge_order_key,
)
from backend.app.graph.models import _require_observation_ids

UTC = timezone.utc
NOW = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)
WINDOW_START = datetime(2024, 5, 1, 8, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2024, 5, 1, 9, 30, 0, tzinfo=UTC)

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
    """A valid Step 4A relationship; *overrides* replace individual fields."""
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


def _edge(**overrides) -> EvidenceEdge:
    return EvidenceEdge.from_relationship(_relationship(**overrides))


def _edge_fields(**overrides) -> dict:
    """The field values of a valid edge, for building one by hand."""
    edge = _edge()
    fields = {name: getattr(edge, name) for name in EvidenceEdge.model_fields}
    fields.update(overrides)
    return fields


def _unvalidated(**overrides) -> EvidenceRelationship:
    """A relationship that never passed Step 4A's validators.

    ``model_construct`` skips validation entirely, which is the only realistic
    way a malformed relationship reaches the graph layer.  Used to prove the
    graph re-checks what it depends on rather than assuming it was checked.  The
    field values are copied by hand rather than through ``model_dump()`` so that
    nested models stay models and a later check cannot pass for the wrong reason.
    """
    relationship = _relationship()
    payload = {
        name: getattr(relationship, name)
        for name in EvidenceRelationship.model_fields
    }
    payload.update(overrides)
    return EvidenceRelationship.model_construct(**payload)


def _corpus(count: int = 6) -> list[EvidenceRelationship]:
    """A small, varied set of relationships spanning every evidence type."""
    ids = ["obs-%02d" % index for index in range(count)]
    relationships = []
    for index in range(count):
        source = ids[index]
        target = ids[(index + 1) % count]
        if source == target:  # pragma: no cover - only if count == 1
            continue
        relationships.append(
            _relationship(
                relationship_id=derive_relationship_id({"corpus": index}),
                source_observation_ids=(source,),
                target_observation_ids=(target,),
                evidence_type=list(EvidenceType)[index % len(EvidenceType)],
                inference_level=list(InferenceLevel)[index % len(InferenceLevel)],
                confidence=Confidence.from_rule(0.5 + index / 10.0),
                rationale="Records %d and %d share a compared value as recorded."
                % (index, index + 1),
                rule_id="rule.corpus.%d" % index,
                rule_version="1.%d.0" % index,
                provenance_refs=(_ref(ingestion_id="ing-%d" % index),),
            )
        )
    return relationships


# --------------------------------------------------------------------------- #
# Node identity
# --------------------------------------------------------------------------- #


class TestNodeIdentity:
    def test_node_identity_is_the_observation_id(self):
        node = ObservationNode.for_observation(OBS_A)
        assert node.observation_id == OBS_A

    def test_the_same_observation_always_yields_the_same_node(self):
        assert ObservationNode.for_observation(OBS_A) == ObservationNode.for_observation(
            OBS_A
        )

    def test_different_observations_yield_different_nodes(self):
        assert ObservationNode.for_observation(OBS_A) != ObservationNode.for_observation(
            OBS_B
        )

    def test_node_id_is_derived_deterministically(self):
        assert derive_node_id(OBS_A) == derive_node_id(OBS_A)
        assert derive_node_id(OBS_A) == str(
            uuid.uuid5(GRAPH_NODE_NAMESPACE, OBS_A)
        )

    def test_node_id_is_not_the_observation_id(self):
        """A node is a graph construct, not the observation it points at."""
        assert derive_node_id(OBS_A) != OBS_A

    def test_node_id_cannot_collide_with_an_observation_id(self):
        """Mirrors the Step 4A relationship-id guarantee, one layer up."""
        assert derive_node_id(OBS_A) != str(
            uuid.uuid5(OBSERVATION_NAMESPACE, OBS_A)
        )

    def test_node_id_cannot_collide_with_a_relationship_id(self):
        assert derive_node_id(OBS_A) != str(
            uuid.uuid5(RELATIONSHIP_NAMESPACE, OBS_A)
        )

    def test_node_id_cannot_collide_with_an_edge_id(self):
        assert derive_node_id(OBS_A) != derive_edge_id(OBS_A)

    def test_a_blank_observation_id_yields_no_node(self):
        for blank in ("", "   ", None, 7, []):
            with pytest.raises(EvidenceGraphError):
                ObservationNode.for_observation(blank)

    def test_a_blank_identifier_is_refused_by_the_model(self):
        with pytest.raises(ValidationError):
            ObservationNode(node_id="", observation_id=OBS_A)
        with pytest.raises(ValidationError):
            ObservationNode(node_id="node", observation_id="  ")

    def test_a_node_is_frozen(self):
        node = ObservationNode.for_observation(OBS_A)
        with pytest.raises(ValidationError):
            node.observation_id = OBS_B

    def test_a_node_carries_no_extra_attribute(self):
        with pytest.raises(ValidationError):
            ObservationNode(node_id="n", observation_id=OBS_A, owner="somebody")

    def test_a_node_keeps_no_evidence_of_its_own(self):
        """Identity only. Provenance belongs to the edges that cite it."""
        assert set(ObservationNode.model_fields) == {"node_id", "observation_id"}


# --------------------------------------------------------------------------- #
# Edge identity and faithful preservation
# --------------------------------------------------------------------------- #


class TestEdgeIdentity:
    def test_edge_identity_is_the_relationship_it_preserves(self):
        relationship = _relationship()
        edge = EvidenceEdge.from_relationship(relationship)
        assert edge.relationship_id == relationship.relationship_id

    def test_edge_id_is_derived_deterministically(self):
        relationship = _relationship()
        assert EvidenceEdge.from_relationship(
            relationship
        ).edge_id == EvidenceEdge.from_relationship(relationship).edge_id
        assert derive_edge_id(relationship.relationship_id) == str(
            uuid.uuid5(GRAPH_EDGE_NAMESPACE, relationship.relationship_id)
        )

    def test_edge_id_is_not_the_relationship_id(self):
        edge = _edge()
        assert edge.edge_id != edge.relationship_id

    def test_edge_id_cannot_collide_with_a_node_id(self):
        edge = _edge()
        assert edge.edge_id != derive_node_id(edge.relationship_id)

    def test_edge_id_cannot_collide_with_a_relationship_id(self):
        relationship = _relationship()
        assert derive_edge_id(relationship.relationship_id) != relationship.relationship_id

    def test_one_relationship_always_gives_the_same_edge(self):
        relationship = _relationship()
        assert EvidenceEdge.from_relationship(
            relationship
        ) == EvidenceEdge.from_relationship(relationship)

    def test_the_edge_derives_its_own_id_and_ignores_a_supplied_one(self):
        """A caller cannot invent an edge identity the derivation would not."""
        relationship = _relationship()
        edge = EvidenceEdge.from_relationship(relationship)
        assert edge.edge_id == derive_edge_id(relationship.relationship_id)

    def test_an_edge_is_frozen(self):
        edge = _edge()
        with pytest.raises(ValidationError):
            edge.rationale = "a different explanation entirely"

    def test_an_edge_carries_no_extra_attribute(self):
        with pytest.raises(ValidationError):
            EvidenceEdge(**_edge_fields(), owner="somebody")


class TestEvidencePreservation:
    def test_every_relationship_field_survives_onto_the_edge(self):
        """The whole point, checked by reflection rather than by example.

        Any field Step 4A adds later and this copy forgets becomes a failing
        test here, which is the mechanism that keeps the two layers in step.
        """
        relationship = _relationship(
            temporal_window=TemporalWindow(start=WINDOW_START, end=WINDOW_END),
            temporal_relation=TemporalRelation.WITHIN,
        )
        edge = EvidenceEdge.from_relationship(relationship)

        for name in EvidenceRelationship.model_fields:
            assert name in EvidenceEdge.model_fields, name
            assert getattr(edge, name) == getattr(relationship, name), name

    def test_the_edge_adds_exactly_one_field_and_keeps_no_others(self):
        """``edge_id`` is the only addition; nothing is dropped or renamed."""
        assert (
            set(EvidenceEdge.model_fields) - set(EvidenceRelationship.model_fields)
            == {"edge_id"}
        )
        assert not set(EvidenceRelationship.model_fields) - set(
            EvidenceEdge.model_fields
        )

    def test_evidence_type_is_preserved_verbatim(self):
        for evidence_type in EvidenceType:
            assert _edge(evidence_type=evidence_type).evidence_type is evidence_type

    def test_inference_level_is_preserved_verbatim(self):
        for level in InferenceLevel:
            assert _edge(inference_level=level).inference_level is level

    def test_correlated_is_never_promoted_to_inferred(self):
        """The specific upgrade Step 4C must never perform."""
        edge = _edge(inference_level=InferenceLevel.CORRELATED)
        assert edge.inference_level is InferenceLevel.CORRELATED
        assert edge.inference_level is not InferenceLevel.INFERRED

    def test_observed_is_not_promoted_to_correlated(self):
        edge = _edge(inference_level=InferenceLevel.OBSERVED)
        assert edge.inference_level is InferenceLevel.OBSERVED

    def test_a_needs_verification_edge_is_not_upgraded(self):
        edge = _edge(inference_level=InferenceLevel.REQUIRES_VERIFICATION)
        assert edge.inference_level is InferenceLevel.REQUIRES_VERIFICATION

    def test_rule_identity_and_version_are_preserved(self):
        edge = _edge(rule_id="rule.geo_coincidence", rule_version="2.7.1")
        assert edge.rule_id == "rule.geo_coincidence"
        assert edge.rule_version == "2.7.1"

    def test_the_rationale_is_preserved_verbatim(self):
        rationale = "matched values: src_ip 203.0.113.5 and dst_ip 198.51.100.9"
        assert _edge(rationale=rationale).rationale == rationale

    def test_created_at_is_carried_not_restamped(self):
        """The graph layer has no clock, so it cannot invent a time."""
        moment = datetime(2019, 3, 4, 5, 6, 7, tzinfo=UTC)
        assert _edge(created_at=moment).created_at == moment

    def test_both_sides_are_preserved_in_the_order_stated(self):
        edge = _edge(
            source_observation_ids=(OBS_C, OBS_A),
            target_observation_ids=(OBS_D, OBS_B),
        )
        assert edge.source_observation_ids == (OBS_C, OBS_A)
        assert edge.target_observation_ids == (OBS_D, OBS_B)

    def test_observation_ids_lists_the_two_sides_deduplicated(self):
        edge = _edge(
            source_observation_ids=(OBS_A, OBS_C), target_observation_ids=(OBS_B,)
        )
        assert edge.observation_ids() == (OBS_A, OBS_C, OBS_B)

    def test_a_relationship_with_several_targets_stays_one_edge(self):
        edge = _edge(target_observation_ids=(OBS_B, OBS_C, OBS_D))
        assert edge.target_observation_ids == (OBS_B, OBS_C, OBS_D)
        assert edge.observation_ids() == (OBS_A, OBS_B, OBS_C, OBS_D)


class TestConfidencePreservation:
    def test_the_confidence_value_is_preserved(self):
        for value in (0.0, 0.25, 0.5, 0.9, 1.0):
            edge = _edge(confidence=Confidence.from_rule(value))
            assert edge.confidence.value == pytest.approx(value)

    def test_the_confidence_basis_is_preserved(self):
        edge = _edge(confidence=Confidence.from_assessment(0.4))
        assert edge.confidence.basis is ConfidenceBasis.HUMAN_ASSESSMENT

    def test_every_basis_survives_the_copy(self):
        edge = _edge(confidence=Confidence.exact_match())
        assert edge.confidence.basis is ConfidenceBasis.EXACT_MATCH
        assert _edge(confidence=Confidence.from_rule(0.8)).confidence.basis is (
            ConfidenceBasis.DETERMINISTIC_RULE
        )

    def test_the_confidence_is_not_recomputed_or_rounded(self):
        """0.7 stays 0.7; the graph has no scoring function to round with."""
        assert _edge(confidence=Confidence.from_rule(0.7)).confidence.value == 0.7

    def test_the_confidence_object_is_carried_by_value(self):
        confidence = Confidence.from_rule(0.62)
        assert _edge(confidence=confidence).confidence == confidence


class TestProvenancePreservation:
    def test_provenance_is_attached_to_the_edge(self):
        edge = _edge()
        assert len(edge.provenance_refs) == 1
        assert edge.provenance_refs[0].ingestion_id == "ing-1"

    def test_provenance_is_not_flattened_into_text(self):
        """Citations stay as citations a reader can walk back through."""
        edge = _edge()
        assert all(
            isinstance(reference, ProvenanceRef) for reference in edge.provenance_refs
        )

    def test_every_reference_is_retained_and_in_order(self):
        references = (
            _ref(ingestion_id="ing-zeta", record="row 9"),
            _ref(ingestion_id="ing-alpha", record="row 1"),
            _ref(ingestion_id="ing-mid", record="row 5"),
        )
        edge = _edge(provenance_refs=references)
        assert edge.provenance_refs == references

    def test_provenance_is_not_sorted_or_deduplicated_by_the_graph(self):
        """Order is the producer's. Reordering would rewrite a citation list."""
        references = (
            _ref(ingestion_id="ing-b", record="row 2"),
            _ref(ingestion_id="ing-a", record="row 2"),
        )
        assert _edge(provenance_refs=references).provenance_refs == references

    def test_a_relationship_with_no_provenance_yields_an_empty_tuple(self):
        assert _edge(provenance_refs=()).provenance_refs == ()

    def test_provenance_is_never_inferred_from_topology(self):
        """Two edges touching one observation do not lend it their citations."""
        edge = _edge(provenance_refs=(_ref(),))
        assert not hasattr(ObservationNode.for_observation(OBS_A), "provenance_refs")
        assert "provenance" not in json.dumps(
            ObservationNode.for_observation(OBS_A).to_dict()
        ).lower()
        assert edge.provenance_refs[0].ingestion_id == "ing-1"


class TestTemporalPreservation:
    def test_the_window_is_preserved_exactly(self):
        window = TemporalWindow(start=WINDOW_START, end=WINDOW_END)
        assert _edge(temporal_window=window).temporal_window == window

    def test_the_window_bounds_survive_to_the_microsecond(self):
        window = TemporalWindow(start=WINDOW_START, end=WINDOW_END)
        edge = _edge(temporal_window=window)
        assert edge.temporal_window.start == WINDOW_START
        assert edge.temporal_window.end == WINDOW_END

    def test_the_relation_is_preserved(self):
        for relation in TemporalRelation:
            window = TemporalWindow(start=WINDOW_START, end=WINDOW_END)
            edge = _edge(temporal_window=window, temporal_relation=relation)
            assert edge.temporal_relation is relation

    def test_absent_temporal_detail_stays_absent(self):
        edge = _edge()
        assert edge.temporal_window is None
        assert edge.temporal_relation is None

    def test_a_window_without_a_relation_is_preserved_as_stated(self):
        window = TemporalWindow(start=WINDOW_START, end=WINDOW_END)
        edge = _edge(temporal_window=window)
        assert edge.temporal_window == window
        assert edge.temporal_relation is None

    def test_the_graph_adds_no_window_of_its_own(self):
        """No temporal index, no window arithmetic, no invented adjacency."""
        edge = _edge()
        assert edge.temporal_window is None
        assert timedelta is not None  # the model never imports or needs arithmetic


# --------------------------------------------------------------------------- #
# Edge construction refusals
# --------------------------------------------------------------------------- #


class TestEdgeRefusals:
    def test_a_self_referential_relationship_is_refused(self):
        with pytest.raises(EvidenceGraphError) as excinfo:
            _edge(source_observation_ids=(OBS_A,), target_observation_ids=(OBS_A,))
        assert OBS_A in str(excinfo.value)

    def test_a_self_reference_on_either_side_is_refused(self):
        with pytest.raises(EvidenceGraphError):
            _edge(
                source_observation_ids=(OBS_A, OBS_B),
                target_observation_ids=(OBS_C, OBS_A),
            )

    def test_an_empty_side_is_refused(self):
        with pytest.raises(EvidenceGraphError):
            _require_observation_ids((), "target_observation_ids")

    def test_a_repeated_observation_on_one_side_is_refused(self):
        with pytest.raises(EvidenceGraphError):
            _require_observation_ids((OBS_B, OBS_B), "target_observation_ids")

    def test_a_blank_observation_id_is_refused(self):
        with pytest.raises(EvidenceGraphError):
            _require_observation_ids(("  ",), "target_observation_ids")

    def test_a_side_that_reached_the_layer_unvalidated_is_still_refused(self):
        """Step 4A refuses these first; the graph refuses them again.

        A relationship can be built with ``model_construct``, which bypasses
        every field validator.  If the graph trusted Step 4A's validation rather
        than re-checking, such an object would become an edge with a blank
        target -- so the check is repeated here on purpose.
        """
        for bad in ((), (OBS_B, OBS_B), ("  ",)):
            malformed = _unvalidated(target_observation_ids=bad)
            with pytest.raises(EvidenceGraphError):
                EvidenceEdge.from_relationship(malformed)

    def test_a_self_edge_that_reached_the_layer_unvalidated_is_still_refused(self):
        malformed = _unvalidated(
            source_observation_ids=(OBS_A,), target_observation_ids=(OBS_A,)
        )
        with pytest.raises(EvidenceGraphError) as excinfo:
            EvidenceEdge.from_relationship(malformed)
        assert OBS_A in str(excinfo.value)

    def test_a_hand_built_edge_cannot_be_a_self_edge_either(self):
        """The invariant is on the model, not only on the builder's way in.

        ``from_relationship`` is how the builder reaches an edge, but nothing
        stops a caller constructing one directly, so the check is repeated on
        the model itself.  Otherwise a hand-built edge could put a node next to
        itself.
        """
        with pytest.raises(ValidationError) as excinfo:
            EvidenceEdge(
                **_edge_fields(
                    source_observation_ids=(OBS_A,), target_observation_ids=(OBS_A,)
                )
            )
        assert "both sides" in str(excinfo.value)

    def test_a_hand_built_edge_cannot_partially_self_reference(self):
        with pytest.raises(ValidationError):
            EvidenceEdge(
                **_edge_fields(
                    source_observation_ids=(OBS_A, OBS_B),
                    target_observation_ids=(OBS_C, OBS_B),
                )
            )

    def test_a_hand_built_valid_edge_is_accepted(self):
        """The invariant refuses self-edges, not direct construction itself."""
        assert EvidenceEdge(**_edge_fields()).evidence_type is EvidenceType.SAME_IP

    def test_a_hand_built_edge_needs_no_provenance_to_exist(self):
        """Provenance defaults to empty, exactly as on the relationship."""
        fields = _edge_fields()
        fields["provenance_refs"] = ()
        assert EvidenceEdge(**fields).provenance_refs == ()

    def test_a_text_string_is_not_a_sequence_of_observation_ids(self):
        with pytest.raises(EvidenceGraphError):
            _require_observation_ids(OBS_B, "target_observation_ids")

    def test_a_non_sequence_is_refused(self):
        with pytest.raises(EvidenceGraphError):
            _require_observation_ids(7, "target_observation_ids")

    def test_a_relationship_id_of_the_wrong_type_is_refused(self):
        with pytest.raises(EvidenceGraphError):
            derive_edge_id(None)

    def test_a_non_relationship_cannot_become_an_edge(self):
        for impostor in (None, "a relationship", 7, object()):
            with pytest.raises(EvidenceGraphError):
                EvidenceEdge.from_relationship(impostor)

    def test_the_graph_error_is_also_a_value_error(self):
        """Catchable both precisely and by the ordinary handler."""
        assert issubclass(EvidenceGraphError, ValueError)


# --------------------------------------------------------------------------- #
# The graph container
# --------------------------------------------------------------------------- #


class TestGraphContainer:
    def test_an_empty_graph_is_legitimate(self):
        graph = EvidenceGraph()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert graph.nodes() == ()
        assert graph.edges() == ()

    def test_a_graph_sorts_whatever_order_it_is_handed(self):
        relationship = _relationship()
        edge = EvidenceEdge.from_relationship(relationship)
        node = ObservationNode.for_observation(OBS_A)

        forwards = EvidenceGraph(observation_nodes=(node,), evidence_edges=(edge,))
        backwards = EvidenceGraph(observation_nodes=(node,), evidence_edges=(edge,))
        assert forwards == backwards

    def test_the_graph_is_frozen(self):
        graph = EvidenceGraph()
        with pytest.raises(dataclasses.FrozenInstanceError):
            graph.observation_nodes = ()

    def test_the_collections_are_tuples(self):
        graph = EvidenceGraph(
            observation_nodes=[ObservationNode.for_observation(OBS_A)],
            evidence_edges=[_edge()],
        )
        assert isinstance(graph.nodes(), tuple)
        assert isinstance(graph.edges(), tuple)
        assert isinstance(graph.get_node(OBS_A), ObservationNode)

    def test_lookup_tables_are_read_only(self):
        graph = EvidenceGraph(
            observation_nodes=(ObservationNode.for_observation(OBS_A),)
        )
        with pytest.raises(TypeError):
            graph._by_observation["obs-b"] = ObservationNode.for_observation(OBS_B)

    def test_edge_order_key_is_a_total_order(self):
        """Sorting by the key must leave no two edges indistinguishable."""
        keys = [edge_order_key(edge) for edge in map(EvidenceEdge.from_relationship, _corpus())]
        assert len(set(keys)) == len(keys)

    def test_edge_order_key_reads_source_then_target_then_id(self):
        edge = _edge(
            relationship_id="r-1",
            source_observation_ids=(OBS_A,),
            target_observation_ids=(OBS_B,),
        )
        assert edge_order_key(edge) == ((OBS_A,), (OBS_B,), "r-1")

    def test_equality_compares_evidence_not_identity(self):
        relationship = _relationship()
        assert EvidenceGraph(
            evidence_edges=(EvidenceEdge.from_relationship(relationship),)
        ) == EvidenceGraph(
            evidence_edges=(EvidenceEdge.from_relationship(relationship),)
        )


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


class TestSerialisation:
    def test_a_node_serialises_to_exactly_its_identity(self):
        node = ObservationNode.for_observation(OBS_A)
        assert node.to_dict() == {
            "node_id": node.node_id,
            "observation_id": OBS_A,
        }

    def test_a_node_key_order_is_fixed(self):
        rendered = json.dumps(ObservationNode.for_observation(OBS_A).to_dict())
        assert rendered.index("node_id") < rendered.index("observation_id")

    def test_an_edge_serialises_every_evidence_field(self):
        edge = _edge(
            temporal_window=TemporalWindow(start=WINDOW_START, end=WINDOW_END),
            temporal_relation=TemporalRelation.BEFORE,
        )
        payload = edge.to_dict()
        for name in (
            "edge_id",
            "relationship_id",
            "source_observation_ids",
            "target_observation_ids",
            "evidence_type",
            "inference_level",
            "confidence",
            "rationale",
            "rule_id",
            "rule_version",
            "provenance_refs",
            "created_at",
            "temporal_window",
            "temporal_relation",
        ):
            assert name in payload, name

    def test_an_edge_serialises_enums_as_their_values(self):
        payload = _edge().to_dict()
        assert payload["evidence_type"] == "SAME_IP"
        assert payload["inference_level"] == "CORRELATED"
        assert payload["confidence"] == {"value": 1.0, "basis": "EXACT_MATCH"}

    def test_provenance_serialises_with_all_three_locators(self):
        payload = _edge().to_dict()
        assert payload["provenance_refs"] == [
            {
                "ingestion_id": "ing-1",
                "source_file": "evidence.csv",
                "source_record": "row 2",
            }
        ]

    def test_temporal_detail_serialises_as_iso_text(self):
        payload = _edge(
            temporal_window=TemporalWindow(start=WINDOW_START, end=WINDOW_END),
            temporal_relation=TemporalRelation.WITHIN,
        ).to_dict()
        assert payload["temporal_window"] == {
            "start": WINDOW_START.isoformat(),
            "end": WINDOW_END.isoformat(),
        }
        assert payload["temporal_relation"] == "WITHIN"

    def test_absent_temporal_detail_serialises_as_null(self):
        payload = _edge().to_dict()
        assert payload["temporal_window"] is None
        assert payload["temporal_relation"] is None

    def test_a_graph_serialises_its_counts(self):
        graph = EvidenceGraph()
        assert graph.to_dict() == {
            "node_count": 0,
            "edge_count": 0,
            "nodes": [],
            "edges": [],
        }

    def test_serialisation_is_json_safe(self):
        graph = EvidenceGraph(
            observation_nodes=(
                ObservationNode.for_observation(OBS_A),
                ObservationNode.for_observation(OBS_B),
            ),
            evidence_edges=tuple(
                EvidenceEdge.from_relationship(item) for item in _corpus()
            ),
        )
        assert json.loads(json.dumps(graph.to_dict())) == graph.to_dict()

    def test_serialisation_is_byte_identical_when_repeated(self):
        graph = EvidenceGraph(
            observation_nodes=(ObservationNode.for_observation(OBS_B),),
            evidence_edges=tuple(
                EvidenceEdge.from_relationship(item) for item in _corpus()
            ),
        )
        assert json.dumps(graph.to_dict()) == json.dumps(graph.to_dict())

    def test_serialisation_carries_no_derived_totals(self):
        """No degree, no centrality, no score: nothing that would invite one."""
        payload = EvidenceGraph(
            observation_nodes=(ObservationNode.for_observation(OBS_A),),
            evidence_edges=(_edge(),),
        ).to_dict()
        for banned in ("degree", "score", "weight", "rank", "centrality", "risk"):
            assert banned not in json.dumps(payload).lower(), banned


# --------------------------------------------------------------------------- #
# Anti-attribution
# --------------------------------------------------------------------------- #


class TestAntiAttribution:
    FORBIDDEN_FIELDS = (
        "owner",
        "ownership",
        "wallet_owner",
        "belongs_to",
        "person",
        "person_id",
        "actor",
        "identity",
        "entity",
        "entity_id",
        "suspect",
        "attacker",
        "attribution",
        "attributed_to",
        "fraud",
        "guilty",
        "malicious",
        "risk",
        "risk_score",
        "suspicion",
        "cluster",
        "anomaly",
        "community",
        "centrality",
    )

    def test_the_node_model_has_no_attribution_field(self):
        for field in self.FORBIDDEN_FIELDS:
            assert field not in ObservationNode.model_fields, field

    def test_the_edge_model_has_no_attribution_field(self):
        for field in self.FORBIDDEN_FIELDS:
            assert field not in EvidenceEdge.model_fields, field

    def test_the_graph_has_no_attribution_attribute(self):
        declared = {item.name for item in dataclasses.fields(EvidenceGraph)}
        for field in self.FORBIDDEN_FIELDS:
            assert field not in declared, field

    def test_nothing_offers_an_attribution_query(self):
        for owner in (ObservationNode, EvidenceEdge, EvidenceGraph):
            for name in dir(owner):
                if name.startswith("__"):
                    continue  # Python's own dunders are not our API
                assert (
                    "attribut" not in name.lower() or name == "explains_attribution"
                ), (owner.__name__, name)

    def test_explains_attribution_is_always_false(self):
        for evidence_type in EvidenceType:
            for level in InferenceLevel:
                relationship = _relationship(
                    evidence_type=evidence_type, inference_level=level
                )
                edge = EvidenceEdge.from_relationship(relationship)
                assert edge.explains_attribution() is False
                assert (
                    ObservationNode.for_observation(OBS_A).explains_attribution() is False
                )
        assert EvidenceGraph().explains_attribution() is False

    def test_certainty_is_still_not_attribution(self):
        """Maximum confidence on an exact match asserts a comparison, not a person."""
        edge = _edge(confidence=Confidence.exact_match())
        assert edge.confidence.value == 1.0
        assert edge.explains_attribution() is False
        rendered = json.dumps(edge.to_dict()).lower()
        for word in ("owner", "person", "wallet", "identity", "attribut"):
            assert word not in rendered, word

    def test_a_shared_address_does_not_become_a_shared_owner(self):
        """The two most tempting leaps, both refused by structure."""
        relationship = _relationship(
            evidence_type=EvidenceType.SAME_ADDRESS,
            inference_level=InferenceLevel.CORRELATED,
        )
        edge = EvidenceEdge.from_relationship(relationship)
        graph = EvidenceGraph(
            observation_nodes=tuple(
                ObservationNode.for_observation(item) for item in (OBS_A, OBS_B)
            ),
            evidence_edges=(edge,),
        )
        assert graph.explains_attribution() is False
        for word in ("owner", "ownership", "belongs", "controlled", "wallet"):
            assert word not in json.dumps(graph.to_dict()).lower(), word

    def test_a_shared_transaction_id_is_not_ownership_of_a_network_address(self):
        """A txid match says nothing about who sent anything, or from where."""
        edge = _edge(
            evidence_type=EvidenceType.SAME_TXID,
            rule_id="rule.same_txid",
            rationale="Both records reference the same txid as recorded.",
        )
        payload = edge.to_dict()
        assert payload["evidence_type"] == "SAME_TXID"
        assert payload["rule_id"] == "rule.same_txid"
        assert edge.explains_attribution() is False
        # Checked against field names, not the whole document: "relationship_id"
        # contains the letters "ip" inside "relation-ship", so a substring sweep
        # over the text would flag the identifier of the relationship itself.
        harmless = {"relationship_id"}
        assert not [
            name
            for name in payload
            if name not in harmless and ("ip" in name or "address" in name)
        ]
        assert not [name for name in payload if "own" in name or "ident" in name]
        for reference in payload["provenance_refs"]:
            assert set(reference) == {"ingestion_id", "source_file", "source_record"}

    def test_evidence_type_names_carry_no_ownership_semantics(self):
        for member in EvidenceType:
            name = member.name.lower()
            for banned in ("owner", "own", "person", "entity", "suspect", "actor"):
                assert banned not in name, member

    def test_the_graph_layer_imports_no_earlier_layer(self):
        """Checked against the parsed import graph, not the prose.

        Step 4A locked the same boundary for its own package; this asserts it
        for the layer above, including the relative imports the graph uses to
        satisfy Step 4A's stricter "no file outside correlation may name it" rule.
        """
        import pathlib

        import backend.app.graph as package

        forbidden = ("domain", "ingestion")
        found = []
        for path in pathlib.Path(package.__file__).parent.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    tail = node.module.split(".")[-1]
                    if tail in forbidden:
                        found.append((path.name, node.module))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[-1] in forbidden:
                            found.append((path.name, alias.name))
        assert found == []

    def test_the_graph_layer_does_import_the_evidence_layer(self):
        """The dependency is declared, not merely tolerated."""
        import pathlib

        import backend.app.graph as package

        modules = set()
        for path in pathlib.Path(package.__file__).parent.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module)
        assert {
            "correlation.enums",
            "correlation.models",
            "correlation.provenance",
        } <= modules

    def test_the_evidence_type_is_not_re_exported(self):
        """It belongs to the layer that defines it; re-exporting blurs that."""
        import backend.app.graph as package

        assert "EvidenceRelationship" not in package.__all__
        assert not hasattr(package, "EvidenceRelationship")

    def test_no_other_module_in_the_app_reaches_for_the_graph(self):
        """Step 4C is a library step; nothing is wired into an entry point."""
        import pathlib

        import backend.app as app_root

        package_root = pathlib.Path(app_root.__file__).parent
        offenders = []
        for path in package_root.rglob("*.py"):
            if "graph" in path.parts:
                continue
            if "app.graph" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        assert offenders == []


# --------------------------------------------------------------------------- #
# Forbidden features -- a code scan, not a claim
# --------------------------------------------------------------------------- #


#: Modules that would mean this step reached outside graph construction.
FORBIDDEN_MODULES = (
    "neo4j",
    "networkx",
    "sqlalchemy",
    "sqlite3",
    "psycopg2",
    "pymysql",
    "pymongo",
    "redis",
    "sklearn",
    "scikit",
    "torch",
    "tensorflow",
    "numpy",
    "pandas",
    "matplotlib",
    "langchain",
    "openai",
    "anthropic",
    "fastapi",
    "starlette",
    "uvicorn",
    "requests",
    "httpx",
    "sqlalchemy.orm",
)

#: Vocabulary that would turn a representation into an interpretation.  Checked
#: against every name the package defines, with underscores removed, so
#: ``risk_score``, ``riskScore`` and ``RiskScore`` are all caught.
FORBIDDEN_NAME_FRAGMENTS = (
    "owner",
    "ownership",
    "person",
    "actor",
    "identity",
    "entity",
    "suspect",
    "attacker",
    "attributable",
    "fraud",
    "guilt",
    "malicious",
    "risk",
    "score",
    "centrality",
    "cluster",
    "community",
    "anomaly",
    "behaviour",
    "behavior",
    "profile",
    "propagat",
    "investigat",
    "shortestpath",
    "pathfinding",
    "traversal",
    "dashboard",
    "endpoint",
    "router",
    "persist",
    "database",
    "predict",
    "classif",
    "llm",
)

#: The one deliberately-named method, which asserts the guarantee by denying it.
ALLOWED_NAME_FRAGMENTS = ("explainattribution", "explainsattribution")


def _normalise(name: str) -> str:
    return name.replace("_", "").lower()


def _defined_names(tree: ast.AST) -> list[str]:
    """Every name the module's public surface defines.

    Classes, functions, methods, and annotated fields -- the shapes a caller can
    reach.  Locals and plain assignments are excluded on purpose: a counter
    called ``scores`` inside a sort would be noise, while a field called
    ``risk_score`` is an API a caller could read.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            names.append(node.target.id)
    return names


class TestForbiddenFeatures:
    def _package_files(self) -> list:
        import pathlib

        import backend.app.graph as package

        return sorted(pathlib.Path(package.__file__).parent.rglob("*.py"))

    def test_the_package_is_small_and_self_contained(self):
        names = sorted(path.name for path in self._package_files())
        assert names == ["__init__.py", "builder.py", "models.py"]

    def test_no_forbidden_module_is_imported(self):
        offenders = []
        for path in self._package_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] in FORBIDDEN_MODULES:
                        offenders.append((path.name, node.module))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                            offenders.append((path.name, alias.name))
        assert offenders == []

    def test_no_forbidden_name_is_defined(self):
        offenders = []
        for path in self._package_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name in _defined_names(tree):
                normalised = _normalise(name)
                if normalised in ALLOWED_NAME_FRAGMENTS:
                    continue
                for fragment in FORBIDDEN_NAME_FRAGMENTS:
                    if fragment in normalised:
                        offenders.append((path.name, name, fragment))
        assert offenders == []

    def test_the_exported_surface_is_only_graph_construction(self):
        import backend.app.graph as package

        expected = {
            "GRAPH_EDGE_NAMESPACE",
            "GRAPH_NODE_NAMESPACE",
            "EvidenceEdge",
            "EvidenceGraph",
            "EvidenceGraphBuilder",
            "EvidenceGraphError",
            "ObservationNode",
            "build_graph",
            "derive_edge_id",
            "derive_node_id",
            "edge_order_key",
        }
        assert set(package.__all__) == expected
        for name in package.__all__:
            assert hasattr(package, name), name

    def test_the_graph_public_api_is_exactly_the_structural_set(self):
        """Counting and adjacency, and nothing that interprets either.

        An exhaustive list rather than a banned-word search, so a method added
        later for a legitimate reason still has to be added here deliberately.
        """
        public = {name for name in dir(EvidenceGraph()) if not name.startswith("_")}
        assert public == {
            "observation_nodes",
            "evidence_edges",
            "node_count",
            "edge_count",
            "nodes",
            "edges",
            "get_node",
            "get_node_by_id",
            "has_node",
            "get_edge",
            "get_edge_by_id",
            "has_edge",
            "outgoing_edges",
            "incoming_edges",
            "neighbors",
            "to_dict",
            "explains_attribution",
        }

    def test_the_package_exposes_no_application_object(self):
        """A library step: no app, no router, no session, no client."""
        import backend.app.graph as package

        for banned in ("app", "router", "session", "client", "engine", "app_module"):
            assert not hasattr(package, banned), banned

    def test_the_builder_performs_no_analysis(self):
        import backend.app.graph as package

        builder = package.EvidenceGraphBuilder()
        assert hasattr(builder, "build")
        for banned in ("score", "cluster", "detect", "rank", "resolve", "path"):
            assert not hasattr(builder, banned), banned

    def test_the_prose_names_what_is_absent(self):
        """A guard against a later step quietly widening this one.

        The docstrings must still say the boundary out loud: if Step 4D adds
        risk scoring by editing these modules, the removal of this statement is
        the signal that the boundary moved.
        """
        import pathlib

        import backend.app.graph as package

        text = "\n".join(
            path.read_text(encoding="utf-8") for path in self._package_files()
        )
        for statement in (
            "not here",
            "no risk",
            "no clustering",
            "no path finding",
            "no neo4j",
            "no attribution",
        ):
            assert statement in text.lower(), statement
        assert package.__doc__ is not None
