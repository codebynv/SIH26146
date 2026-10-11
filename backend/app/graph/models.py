"""In-memory evidence graph models (SIH26146 - Step 4C).

What this module is
-------------------
The *representation* of the relationships Step 4B already derived.  It holds
three things and nothing else:

* :class:`ObservationNode` -- one per observation named by a relationship;
* :class:`EvidenceEdge`   -- one per relationship, carrying its evidence;
* :class:`EvidenceGraph`  -- the two, plus structural indexes and a
  deterministic serialisation.

The single rule everything here obeys
------------------------------------
**The graph invents nothing.**  Every edge is a verbatim copy of one
``EvidenceRelationship`` from the correlation package, and every node is one of
the observations that relationship names.  No edge is inferred from neighbouring
edges, no node is created for an observation nobody linked, no confidence is
recomputed, and no inference level is raised.  A ``CORRELATED`` relationship
arrives as a ``CORRELATED`` edge or the build fails.

Graph-layer invariants
---------------------
Two refusals live here rather than in the engine, because they are properties of
a *graph* and not of any one rule:

* **No self-edges.**  Step 4A deliberately *reports* ``is_self_referential``
  without rejecting it, on the grounds that only a rule knows whether it is
  meaningful.  A graph is the wrong place to leave that ambiguity: an edge from a
  node to itself carries no information, corrupts every neighbour count, and
  would make "these two observations match" mean something it does not.  The
  builder rejects it outright.
* **No duplicate relationship ids.**  Relationship ids are derived from
  semantic content, so two relationships sharing one are the same relationship
  stated twice.  Silently keeping the last one would make a graph depend on input
  order, and keeping the first would discard evidence without saying so.

Why nodes carry almost nothing
------------------------------
A node here is *the fact that an observation was named*, not a copy of the
observation.  The builder's only input is a list of relationships, and a
relationship knows about an observation exactly one way: through the evidence
that connects it to something else.  Aggregating the provenance or timestamps of
incident edges onto a node would be inferring observation properties from graph
topology, which is precisely the reasoning this system is not allowed to do
quietly.  Steps 1-3 own the observation payload; the graph points at it.

Anti-attribution
----------------
A node is an observation, not a person.  Two observations sharing an IP address
are two records that happen to carry the same value -- nothing in this module
records who used them, who owns them, or whether anyone is at fault, and
``extra="forbid"`` means such a field cannot be attached later by accident.
:meth:`ObservationNode.explains_attribution`, :meth:`EvidenceEdge.explains_attribution`
and :meth:`EvidenceGraph.explains_attribution` all return ``False``
unconditionally, so a caller can assert the guarantee instead of inferring it
from an absence.

What is deliberately not here
-----------------------------
No traversal beyond direct adjacency, no path finding, no scoring, no
centrality, no clustering, no anomaly detection, no behavioural profile, no risk
figure, no attribution, no database, no serialisation to a store and no API.
Those are later steps, and none of them is smuggled in under another name.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Relative imports throughout this package, deliberately.  Step 4A locked a
# guarantee that no file in the application outside the correlation package may
# *name* that package -- not only the ones that predate it -- and its test
# enforces it as a plain text scan of the whole tree.  This layer depends on the
# evidence types, so a dotted absolute path here would break a locked
# guarantee; `..correlation` satisfies the dependency and leaves the guarantee
# standing.  Nothing is hidden by this: the imports are the only ones in the
# package, and they are checked in the Step 4C tests.
from ..correlation.enums import (
    EvidenceType,
    InferenceLevel,
    TemporalRelation,
)
from ..correlation.models import (
    Confidence,
    EvidenceRelationship,
    TemporalWindow,
)
from ..correlation.provenance import ProvenanceRef

__all__ = [
    "GRAPH_NODE_NAMESPACE",
    "GRAPH_EDGE_NAMESPACE",
    "EvidenceGraphError",
    "ObservationNode",
    "EvidenceEdge",
    "EvidenceGraph",
    "derive_node_id",
    "derive_edge_id",
    "edge_order_key",
]


#: Stable UUIDv5 namespaces for the two identifiers this layer mints.
#:
#: Both are deliberately different from Step 3's ``OBSERVATION_NAMESPACE`` and
#: Step 4A's ``RELATIONSHIP_NAMESPACE``.  A node id is a *graph* identifier and
#: an edge id is a *graph* identifier; neither is the thing it points at, and
#: keeping them in separate namespaces makes it impossible for a node to be
#: mistaken for its observation, for an edge for its relationship, or for a node
#: for an edge -- even if two payloads were ever byte-identical.
GRAPH_NODE_NAMESPACE = uuid.UUID("c7e4b9d2-1f5a-5a63-9d84-2e6c0b8a3f17")
GRAPH_EDGE_NAMESPACE = uuid.UUID("5a90f3c8-6b21-5d47-8e19-7c2d4a6b0e83")


class EvidenceGraphError(ValueError):
    """Raised when a set of relationships cannot be represented faithfully.

    A :class:`ValueError`, so ordinary ``except ValueError`` still works, but
    with its own type so a caller can tell "this evidence could not be drawn as
    the graph asked" apart from every other bad value in the system.

    Nothing is caught and repaired.  A relationship that cannot be represented
    without losing or strengthening something is a defect in its producer, and
    quietly dropping it would produce a graph that looks complete and is not.
    """


def derive_node_id(observation_id: str) -> str:
    """Derive the graph node identifier for *observation_id*.

    A UUIDv5 in :data:`GRAPH_NODE_NAMESPACE`, so the same observation always
    yields the same node id, and so the node id can never equal an observation
    id, a relationship id or an edge id.

    The mapping is pure.  It is not derived from the graph's contents, so adding
    a relationship never renames an existing node.
    """
    if not isinstance(observation_id, str) or not observation_id.strip():
        raise EvidenceGraphError(
            "an observation id must be a non-empty, non-blank string; a node "
            "without one points at nothing"
        )
    return str(uuid.uuid5(GRAPH_NODE_NAMESPACE, observation_id))


def derive_edge_id(relationship_id: str) -> str:
    """Derive the graph edge identifier for *relationship_id*.

    A UUIDv5 in :data:`GRAPH_EDGE_NAMESPACE`.  One edge per relationship, always
    with the same identifier, which is what makes ``source -> target1`` and
    ``source -> target2`` two independently addressable edges rather than one
    edge with a merged payload.
    """
    if not isinstance(relationship_id, str) or not relationship_id.strip():
        raise EvidenceGraphError(
            "a relationship id must be a non-empty, non-blank string; an edge "
            "without one cannot be looked up"
        )
    return str(uuid.uuid5(GRAPH_EDGE_NAMESPACE, relationship_id))


def _require_observation_ids(value: Any, label: str) -> tuple[str, ...]:
    """Normalise and validate one side of an edge.

    Accepts any sequence and returns an immutable tuple, so a caller cannot
    later mutate the list an edge was built from.  Order *within* a side is
    preserved exactly as the relationship stated it: reordering the ids of a
    single relationship would change which observation is the stated source, and
    that is a statement, not a detail.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise EvidenceGraphError(
            "%s must be a sequence of observation ids, not %s"
            % (label, type(value).__name__)
        )
    ids = tuple(value)
    if not ids:
        raise EvidenceGraphError(
            "%s must name at least one observation; an edge with no %s asserts "
            "nothing" % (label, label.split("_")[0])
        )
    for observation_id in ids:
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise EvidenceGraphError(
                "%s contains an entry that is not a non-blank observation id"
                % label
            )
    if len(set(ids)) != len(ids):
        raise EvidenceGraphError(
            "%s repeats an observation; a duplicated entry adds no evidence"
            % label
        )
    return ids


def _reject_self_edge(
    source: tuple[str, ...], target: tuple[str, ...], where: str
) -> None:
    """Refuse an edge that names the same observation on both sides.

    One rule, one message, called from both places an edge can be born:
    :meth:`EvidenceEdge.from_relationship`, which is the path the builder takes,
    and the model validator below, which is the only thing standing between a
    hand-built edge and a graph that counts a node as its own neighbour.
    """
    shared = set(source) & set(target)
    if shared:
        raise EvidenceGraphError(
            "%s places observation(s) %s on both sides of itself. A self-edge "
            "carries no information, so it is refused rather than drawn."
            % (where, ", ".join(sorted(shared)))
        )


class ObservationNode(BaseModel):
    """One observation, as it appears in the graph.

    The node records identity and nothing else.  It is **not** a person, an
    account holder, an organisation or an attacker: it is the record Steps 1-3
    produced, referred to by id because a relationship referenced it.

    The absence of an ``owner``, ``identity`` or ``risk`` field is enforced by
    ``extra="forbid"``, not merely by convention.  And because the graph layer
    receives relationships rather than observations, a node cannot quietly
    accumulate attributes: the only thing known about an observation here is
    that some evidence connected it to something.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(
        description="Deterministic graph identifier, derived from "
        "observation_id. Never equal to the observation id itself."
    )
    observation_id: str = Field(
        description="Identifier of the observation this node refers to. This "
        "is the node's identity: two nodes with the same observation_id are the "
        "same node."
    )

    @field_validator("node_id", "observation_id")
    @classmethod
    def _require_non_blank(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("%s must be a non-empty, non-blank string" % info.field_name)
        return value

    @classmethod
    def for_observation(cls, observation_id: str) -> "ObservationNode":
        """Build the node for *observation_id*.

        The node id is derived, never supplied, so two calls with the same
        observation always produce equal nodes and no caller can invent a node
        identity that the derivation would not produce.
        """
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise EvidenceGraphError(
                "cannot create a node for %r; an observation id must be a "
                "non-empty, non-blank string" % (observation_id,)
            )
        return cls(node_id=derive_node_id(observation_id), observation_id=observation_id)

    def explains_attribution(self) -> bool:
        """Always ``False``.

        Present for the same reason as the identical method on the correlation
        package's relationship: so the guarantee can be asserted rather than
        inferred from an absence.
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe, deterministically ordered mapping for this node."""
        return {"node_id": self.node_id, "observation_id": self.observation_id}

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "node(%s)" % self.observation_id


class EvidenceEdge(BaseModel):
    """One ``EvidenceRelationship``, preserved exactly, as a graph edge.

    Every field below is a copy.  The evidence type, the inference level, the
    confidence and its basis, the rule identity and version, the rationale, the
    provenance citations and the temporal detail are carried through unchanged
    and in their original order, so an edge can always answer "what evidence is
    this, how strong, found by which version of which rule, and resting on which
    source records?".

    Two refusals are structural.  An edge may not name the same observation on
    both sides, and it may not be a source without a target or a target without
    a source.  Both would make the edge assert nothing while appearing to assert
    something.

    An edge is not an inference about the world.  ``SAME_IP`` on this edge means
    two records carry the same IP address; it does not mean one person, wallet or
    device was involved, and the graph layer never strengthens it into one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_id: str = Field(
        description="Deterministic graph identifier, derived from "
        "relationship_id. Never equal to the relationship id itself."
    )
    relationship_id: str = Field(
        description="Identifier of the relationship this edge preserves. Kept "
        "so the edge can always be traced back to the record that produced it."
    )
    source_observation_ids: tuple[str, ...] = Field(
        description="Observations on the source side, in the order the "
        "relationship stated them."
    )
    target_observation_ids: tuple[str, ...] = Field(
        description="Observations on the target side, in the order the "
        "relationship stated them. A relationship with several targets is one "
        "edge, never several collapsed together."
    )
    evidence_type: EvidenceType = Field(
        description="What kind of match this is, copied unchanged."
    )
    inference_level: InferenceLevel = Field(
        description="How the relationship may be characterised, copied "
        "unchanged. The graph never raises a level."
    )
    confidence: Confidence = Field(
        description="Strength of evidence, copied unchanged with its basis."
    )
    rationale: str = Field(description="The relationship's explanation, verbatim.")
    rule_id: str = Field(description="Identity of the rule that produced it.")
    rule_version: str = Field(description="Version of that rule.")
    provenance_refs: tuple[ProvenanceRef, ...] = Field(
        default=(),
        description="Citations back to the source records, in their original "
        "order. Retained, never merged, sorted or dropped. Defaults to empty "
        "only so the model can be built by hand; the builder always copies the "
        "relationship's citations across, empty or not.",
    )
    created_at: datetime = Field(
        description="When the relationship was recorded. Carried, not "
        "regenerated: the graph layer has no clock of its own, so building the "
        "same relationships twice cannot introduce a new timestamp."
    )
    temporal_window: Optional[TemporalWindow] = Field(
        default=None, description="The stated span, when the relationship had one."
    )
    temporal_relation: Optional[TemporalRelation] = Field(
        default=None, description="The stated ordering, when there was one."
    )

    @field_validator("edge_id", "relationship_id", "rule_id", "rule_version")
    @classmethod
    def _require_non_blank(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("%s must be a non-empty, non-blank string" % info.field_name)
        return value

    @field_validator("rationale")
    @classmethod
    def _require_rationale(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("rationale must be a non-empty, non-blank string")
        return value

    @field_validator("source_observation_ids", "target_observation_ids")
    @classmethod
    def _check_sides(cls, value: Any, info: Any) -> tuple[str, ...]:
        return _require_observation_ids(value, str(info.field_name))

    @model_validator(mode="after")
    def _check_no_self_edge(self) -> "EvidenceEdge":
        # Enforced on the model as well as in from_relationship, so the
        # invariant holds for an edge built by hand.  A self-edge would make
        # neighbours() report a node as adjacent to itself and would corrupt
        # every incidence count downstream.
        _reject_self_edge(
            self.source_observation_ids, self.target_observation_ids, "this edge"
        )
        return self

    @classmethod
    def from_relationship(cls, relationship: EvidenceRelationship) -> "EvidenceEdge":
        """Copy *relationship* into an edge without changing anything about it.

        The one transformation is the edge id, which is derived from the
        relationship id.  Every other field, including the order of the
        observation ids on each side and the order of the provenance citations,
        is taken as it stands: this method reads, and does not interpret.
        """
        if not isinstance(relationship, EvidenceRelationship):
            raise EvidenceGraphError(
                "cannot build an edge from %s; an edge preserves an "
                "EvidenceRelationship and nothing else"
                % type(relationship).__name__
            )

        source = _require_observation_ids(
            relationship.source_observation_ids, "source_observation_ids"
        )
        target = _require_observation_ids(
            relationship.target_observation_ids, "target_observation_ids"
        )
        _reject_self_edge(
            source, target, "relationship %s" % relationship.relationship_id
        )

        return cls(
            edge_id=derive_edge_id(relationship.relationship_id),
            relationship_id=relationship.relationship_id,
            source_observation_ids=source,
            target_observation_ids=target,
            evidence_type=relationship.evidence_type,
            inference_level=relationship.inference_level,
            confidence=relationship.confidence,
            rationale=relationship.rationale,
            rule_id=relationship.rule_id,
            rule_version=relationship.rule_version,
            provenance_refs=tuple(relationship.provenance_refs),
            created_at=relationship.created_at,
            temporal_window=relationship.temporal_window,
            temporal_relation=relationship.temporal_relation,
        )

    def explains_attribution(self) -> bool:
        """Always ``False``.  An edge states that evidence exists between two observations, never who or what that makes them."""
        return False

    def observation_ids(self) -> tuple[str, ...]:
        """Every observation this edge names, de-duplicated, in side order."""
        seen: dict[str, None] = {}
        for observation_id in self.source_observation_ids + self.target_observation_ids:
            seen.setdefault(observation_id, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe, deterministically ordered mapping for this edge.

        Field order is fixed by the class definition and is not a function of
        insertion order, so two runs serialise identically.
        """
        return {
            "edge_id": self.edge_id,
            "relationship_id": self.relationship_id,
            "source_observation_ids": list(self.source_observation_ids),
            "target_observation_ids": list(self.target_observation_ids),
            "evidence_type": self.evidence_type.value,
            "inference_level": self.inference_level.value,
            "confidence": {
                "value": self.confidence.value,
                "basis": self.confidence.basis.value,
            },
            "rationale": self.rationale,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "provenance_refs": [
                {
                    "ingestion_id": reference.ingestion_id,
                    "source_file": reference.source_file,
                    "source_record": reference.source_record,
                }
                for reference in self.provenance_refs
            ],
            "created_at": self.created_at.isoformat(),
            "temporal_window": (
                None
                if self.temporal_window is None
                else {
                    "start": self.temporal_window.start.isoformat(),
                    "end": self.temporal_window.end.isoformat(),
                }
            ),
            "temporal_relation": (
                None if self.temporal_relation is None else self.temporal_relation.value
            ),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "edge(%s: %s %s)" % (
            self.relationship_id,
            self.evidence_type.value,
            self.inference_level.value,
        )


def edge_order_key(edge: EvidenceEdge) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """The deterministic sort key for edges.

    Source side, then target side, then relationship id.  Relationship ids are
    unique, so this is a *total* order over the edges, and the resulting
    sequence is a function of the set of relationships rather than of the order
    they arrived in.

    The sides come before the relationship id on purpose: it makes the graph read
    in evidence order -- what points at what, then under which rule -- instead
    of in an order derived from an opaque identifier.  Both are stable; only one
    is legible.
    """
    return (
        edge.source_observation_ids,
        edge.target_observation_ids,
        edge.relationship_id,
    )


@dataclass(frozen=True)
class EvidenceGraph:
    """An immutable, deterministic graph of observation nodes and evidence edges.

    The graph is a *view* over the relationships it was built from: it holds the
    same evidence, addressed by node and edge, and it holds nothing else.  It is
    frozen at every level -- nodes and edges are frozen models, the collections
    are tuples, and the lookup tables are read-only mappings -- so a consumer
    cannot reorder, re-point or extend it in place.

    Ordering
    --------
    Nodes are sorted by ``observation_id`` and edges by
    ``(source_observation_ids, target_observation_ids, relationship_id)``.  Both
    are total orders over the *contents*, so shuffling the input relationships
    cannot reorder the graph, and building twice cannot either.

    Queries
    -------
    Only structure: how many nodes and edges there are, what is at a node, what
    leaves and enters it, and what sits next to it.  There is no path finding,
    no scoring, no centrality, no clustering and no interpretation.  Asking who
    a node "belongs to" is not a question this object can answer, because it has
    no field that could hold the answer.
    """

    observation_nodes: tuple[ObservationNode, ...] = ()
    evidence_edges: tuple[EvidenceEdge, ...] = ()
    _by_observation: Mapping[str, ObservationNode] = field(
        init=False, repr=False, compare=False
    )
    _by_node_id: Mapping[str, ObservationNode] = field(
        init=False, repr=False, compare=False
    )
    _by_relationship: Mapping[str, EvidenceEdge] = field(
        init=False, repr=False, compare=False
    )
    _by_edge_id: Mapping[str, EvidenceEdge] = field(
        init=False, repr=False, compare=False
    )
    _outgoing: Mapping[str, tuple[EvidenceEdge, ...]] = field(
        init=False, repr=False, compare=False
    )
    _incoming: Mapping[str, tuple[EvidenceEdge, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Build the read-only lookup tables, once, at construction.

        Constructed here rather than on each call so that the graph's cost is
        paid once and every query is a plain dictionary hit.  Nothing here
        inspects the *meaning* of an edge; it only files each one under the
        observations it names.
        """
        nodes = tuple(
            sorted(self.observation_nodes, key=lambda node: node.observation_id)
        )
        edges = tuple(sorted(self.evidence_edges, key=edge_order_key))

        by_observation: dict[str, ObservationNode] = {}
        by_node_id: dict[str, ObservationNode] = {}
        for node in nodes:
            by_observation[node.observation_id] = node
            by_node_id[node.node_id] = node

        by_relationship: dict[str, EvidenceEdge] = {}
        by_edge_id: dict[str, EvidenceEdge] = {}
        outgoing: dict[str, list[EvidenceEdge]] = {}
        incoming: dict[str, list[EvidenceEdge]] = {}
        for edge in edges:
            by_relationship[edge.relationship_id] = edge
            by_edge_id[edge.edge_id] = edge
            for observation_id in edge.source_observation_ids:
                outgoing.setdefault(observation_id, []).append(edge)
            for observation_id in edge.target_observation_ids:
                incoming.setdefault(observation_id, []).append(edge)

        object.__setattr__(self, "observation_nodes", nodes)
        object.__setattr__(self, "evidence_edges", edges)
        object.__setattr__(self, "_by_observation", MappingProxyType(by_observation))
        object.__setattr__(self, "_by_node_id", MappingProxyType(by_node_id))
        object.__setattr__(self, "_by_relationship", MappingProxyType(by_relationship))
        object.__setattr__(self, "_by_edge_id", MappingProxyType(by_edge_id))
        object.__setattr__(
            self,
            "_outgoing",
            MappingProxyType(
                {key: tuple(value) for key, value in outgoing.items()}
            ),
        )
        object.__setattr__(
            self,
            "_incoming",
            MappingProxyType(
                {key: tuple(value) for key, value in incoming.items()}
            ),
        )

    # -- counts ------------------------------------------------------------ #

    def node_count(self) -> int:
        """How many observation nodes the graph holds."""
        return len(self.observation_nodes)

    def edge_count(self) -> int:
        """How many evidence edges the graph holds. One per relationship."""
        return len(self.evidence_edges)

    # -- access ------------------------------------------------------------ #

    def nodes(self) -> tuple[ObservationNode, ...]:
        """All nodes, ordered by ``observation_id``."""
        return self.observation_nodes

    def edges(self) -> tuple[EvidenceEdge, ...]:
        """All edges, in :func:`edge_order_key` order."""
        return self.evidence_edges

    def get_node(self, observation_id: str) -> ObservationNode:
        """The node for *observation_id*.

        Raises :class:`KeyError` when no node exists.  Returning ``None``
        instead would let a missing node be silently rendered as "this
        observation has no links", which is a claim; a ``KeyError`` is a
        mistake being reported.
        """
        try:
            return self._by_observation[observation_id]
        except (KeyError, TypeError):
            raise KeyError(
                "no node for observation %r; the graph holds %d node(s)"
                % (observation_id, self.node_count())
            ) from None

    def get_node_by_id(self, node_id: str) -> ObservationNode:
        """The node whose graph identifier is *node_id*."""
        try:
            return self._by_node_id[node_id]
        except (KeyError, TypeError):
            raise KeyError("no node with node_id %r" % (node_id,)) from None

    def has_node(self, observation_id: str) -> bool:
        """True when a node exists for *observation_id*."""
        return observation_id in self._by_observation

    def get_edge(self, relationship_id: str) -> EvidenceEdge:
        """The edge preserving *relationship_id*.

        Raises :class:`KeyError` when the relationship was never built into the
        graph.  The lookup is by relationship id and not by edge id, because the
        relationship id is the identifier the evidence layer already uses and the
        one a reader will have.
        """
        try:
            return self._by_relationship[relationship_id]
        except (KeyError, TypeError):
            raise KeyError(
                "no edge for relationship %r; the graph holds %d edge(s)"
                % (relationship_id, self.edge_count())
            ) from None

    def get_edge_by_id(self, edge_id: str) -> EvidenceEdge:
        """The edge whose graph identifier is *edge_id*."""
        try:
            return self._by_edge_id[edge_id]
        except (KeyError, TypeError):
            raise KeyError("no edge with edge_id %r" % (edge_id,)) from None

    def has_edge(self, relationship_id: str) -> bool:
        """True when an edge exists for *relationship_id*."""
        return relationship_id in self._by_relationship

    # -- adjacency --------------------------------------------------------- #

    def outgoing_edges(self, observation_id: str) -> tuple[EvidenceEdge, ...]:
        """Edges whose *source* side names *observation_id*, in graph order.

        The relationship states which side an observation is on, and the graph
        keeps that.  A multi-target relationship contributes one entry here
        whatever the number of targets, because it is one piece of evidence.
        """
        self.get_node(observation_id)
        return self._outgoing.get(observation_id, ())

    def incoming_edges(self, observation_id: str) -> tuple[EvidenceEdge, ...]:
        """Edges whose *target* side names *observation_id*, in graph order."""
        self.get_node(observation_id)
        return self._incoming.get(observation_id, ())

    def neighbors(self, observation_id: str) -> tuple[str, ...]:
        """Observation ids directly adjacent to *observation_id*, sorted.

        Both directions, de-duplicated, sorted.  This is plain adjacency: an
        observation is a neighbour because an edge names it on one side or the
        other, and for no other reason.  It is emphatically *not* a path, a
        ranking or a relationship of any real-world kind.
        """
        self.get_node(observation_id)
        adjacent: set[str] = set()
        for edge in self._outgoing.get(observation_id, ()):
            adjacent.update(edge.target_observation_ids)
        for edge in self._incoming.get(observation_id, ()):
            adjacent.update(edge.source_observation_ids)
        adjacent.discard(observation_id)
        return tuple(sorted(adjacent))

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """A deterministic, JSON-safe representation of the whole graph.

        Nodes come out ordered by ``observation_id`` and edges in the graph's
        edge order, with every evidence field intact.  This is a value for
        tests and for a later presentation layer to render; it is not a
        persistence format and nothing writes it anywhere.
        """
        return {
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "nodes": [node.to_dict() for node in self.nodes()],
            "edges": [edge.to_dict() for edge in self.edges()],
        }

    def explains_attribution(self) -> bool:
        """Always ``False``.

        A graph of observations and the evidence between them cannot and does not
        say who owns what, who did what, or who is at fault.  Making that a
        method rather than a convention lets a caller assert it.
        """
        return False

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return "EvidenceGraph(%d nodes, %d edges)" % (
            self.node_count(),
            self.edge_count(),
        )
