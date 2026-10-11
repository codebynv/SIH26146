"""Deterministic construction of the evidence graph (SIH26146 - Step 4C).

What this module does
---------------------
Exactly one thing: turn a sequence of ``EvidenceRelationship`` -- the typed
evidence the correlation package produces -- into an :class:`EvidenceGraph`.
For each relationship it creates the nodes for the observations that relationship
names, and one edge that preserves the relationship exactly.  That is the whole
job.

What it will not do
-------------------
It never *discovers* a relationship.  There is no rule here, no threshold, no
comparison of two observations and no scoring function -- the builder is not
given observations at all, only relationships, so it has no data from which a
new link could be derived.  Two records sharing an IP address are linked because
Step 4B said so, and the builder's contribution is to place that statement in
the structure without diluting it.

Because the only input is evidence, three failure modes are impossible *by
construction* rather than by policy, and are worth stating plainly: an edge
cannot claim a stronger inference level than the relationship carried, because
the level is copied; a node cannot acquire a property from its neighbours,
because a node has no property other than its identity; and the graph cannot
develop a meaning the evidence did not have, because nothing in it reads meaning.

What it refuses, and why
------------------------
* **A relationship that is not an ``EvidenceRelationship``.**  The builder
  reads specific, validated fields.  A duck-typed stand-in might omit one, and a
  graph quietly missing a confidence or a rule version would look complete.
* **A self-referential relationship.**  A self-edge is refused by the edge model
  itself, and by the builder before it gets there.  This is the graph layer's
  invariant: an edge from a node to itself carries no information, corrupts
  every neighbour count, and would make "these two observations match" mean
  something it does not.
* **A duplicate ``relationship_id``.**  Ids are derived from semantic content, so
  a repeat means the same evidence arrived twice.  Keeping either copy silently
  would make the graph depend on input order, and dropping the duplicate would
  lose evidence without recording that it was dropped.
* **A blank or missing identifier**, anywhere.

Every refusal raises :class:`EvidenceGraphError` with the offending identifier
in the message.  Nothing is skipped, nothing is defaulted, and no partial graph
is returned: a batch that cannot be drawn faithfully is a defect in its
producer, and a graph that looks complete while having quietly discarded
evidence is worse than no graph at all.

Determinism
-----------
The builder reads the relationships it is given and emits nodes sorted by
``observation_id`` and edges sorted by :func:`~.models.edge_order_key`.  Both
orders are functions of the *contents* of the input, so shuffling the sequence
cannot change the graph, and calling ``build`` twice on the same sequence
produces equal graphs.

The builder also has no clock.  ``created_at`` is carried from the relationship
it came from, never stamped afresh, so a rebuild at a different moment is still
the same graph rather than a near-identical one.

Immutability
------------
Relationships are read.  They are frozen Step 4A models and the builder assigns
to nothing: it builds new node and edge objects from their values, and copies
observation ids and provenance references into fresh immutable tuples.  A caller
may therefore build from a list it still holds, and re-build from it afterwards,
and get the same graph both times.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..correlation.enums import ConfidenceBasis, EvidenceType, InferenceLevel
from ..correlation.models import Confidence, EvidenceRelationship
from .models import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceGraphError,
    ObservationNode,
)

__all__ = [
    "EvidenceGraphBuilder",
    "build_graph",
]


def _is_usable_id(value: Any) -> bool:
    """True when *value* is a non-blank string usable as an identifier."""
    return isinstance(value, str) and bool(value.strip())


class EvidenceGraphBuilder:
    """Builds an :class:`EvidenceGraph` from evidence relationships.

    Stateless, and therefore safe to reuse and to share: a build depends only on
    its argument, never on anything an earlier call did.  Two instances with the
    same input produce the same graph, as does one instance used twice.
    """

    def build(
        self, relationships: Sequence[EvidenceRelationship]
    ) -> EvidenceGraph:
        """Return the graph for *relationships*.

        Raises :class:`EvidenceGraphError` if the relationships cannot be
        represented without altering them, and :class:`TypeError` if the
        argument is not a sequence of ``EvidenceRelationship`` at all.
        """
        materialised = self._materialise(relationships)

        nodes: dict[str, ObservationNode] = {}
        edges: list[EvidenceEdge] = []
        first_use: dict[str, int] = {}

        for position, relationship in enumerate(materialised):
            self._validate(relationship, position)

            relationship_id = relationship.relationship_id
            if relationship_id in first_use:
                raise EvidenceGraphError(
                    "duplicate relationship_id %s: already used by the "
                    "relationship at position %d. Relationship ids are derived "
                    "from semantic content, so this is the same evidence stated "
                    "twice; keeping either copy silently would make the graph "
                    "depend on the order it was supplied in."
                    % (relationship_id, first_use[relationship_id])
                )
            first_use[relationship_id] = position

            # from_relationship copies the evidence and refuses a self-edge.
            edge = EvidenceEdge.from_relationship(relationship)
            edges.append(edge)

            for observation_id in edge.observation_ids():
                # setdefault, so an observation named by several relationships
                # yields exactly one node.  Reuse is expected, not a conflict:
                # a node may carry any number of edges, and collapsing it to
                # one would discard evidence.
                nodes.setdefault(
                    observation_id, ObservationNode.for_observation(observation_id)
                )

        return EvidenceGraph(
            observation_nodes=tuple(
                nodes[observation_id] for observation_id in sorted(nodes)
            ),
            evidence_edges=tuple(edges),
        )

    # -- validation -------------------------------------------------------- #

    @staticmethod
    def _materialise(relationships: Any) -> tuple[EvidenceRelationship, ...]:
        """Take one immutable copy of the input, in the order given.

        Copying the *container* is enough.  The relationships themselves are
        frozen models that the builder never writes to, and this snapshot means
        even a generator supplied in place of a list is consumed exactly once
        and cannot be re-read while the build is in progress.
        """
        if relationships is None:
            raise TypeError("relationships must be a sequence, not None")
        if isinstance(relationships, (str, bytes)):
            raise TypeError(
                "relationships must be a sequence of EvidenceRelationship, not %s"
                % type(relationships).__name__
            )
        if not isinstance(relationships, Iterable):
            raise TypeError(
                "relationships must be an iterable of EvidenceRelationship, not %s"
                % type(relationships).__name__
            )
        return tuple(relationships)

    @staticmethod
    def _validate(relationship: Any, position: int) -> None:
        """Check one relationship is complete enough to be drawn faithfully.

        A Step 4A relationship is already fully validated, so this is defence in
        depth -- against a caller that reached a relationship by other means,
        such as ``model_construct``, which bypasses field validators.  The check
        is structural: it asks whether the fields an edge copies are present and
        of the right kind, and never whether they are *plausible*.  Plausibility
        is Step 4A's judgement, not this layer's to second-guess.
        """
        if not isinstance(relationship, EvidenceRelationship):
            raise EvidenceGraphError(
                "item at position %d is %s, not an EvidenceRelationship; the "
                "graph can only preserve evidence it can read"
                % (position, type(relationship).__name__)
            )

        where = "relationship at position %d" % position
        relationship_id = relationship.relationship_id
        if _is_usable_id(relationship_id):
            where += " (%s)" % relationship_id

        if not _is_usable_id(relationship.relationship_id):
            raise EvidenceGraphError(
                "%s has no usable relationship_id; an edge could never be looked "
                "up again" % where
            )
        if not _is_usable_id(relationship.rule_id):
            raise EvidenceGraphError(
                "%s does not name the rule that produced it, so it cannot be "
                "reproduced" % where
            )
        if not _is_usable_id(relationship.rule_version):
            raise EvidenceGraphError(
                "%s does not name the version of its rule, so a later run could "
                "not be told apart from this one" % where
            )
        if not _is_usable_id(relationship.rationale):
            raise EvidenceGraphError(
                "%s has no rationale; an unexplained edge in a graph is a claim "
                "nobody can defend" % where
            )
        if not isinstance(relationship.evidence_type, EvidenceType):
            raise EvidenceGraphError(
                "%s carries no EvidenceType; what kind of match this is would be "
                "lost" % where
            )
        if not isinstance(relationship.inference_level, InferenceLevel):
            raise EvidenceGraphError(
                "%s carries no InferenceLevel; how the relationship may be read "
                "would be lost" % where
            )
        confidence = relationship.confidence
        if not isinstance(confidence, Confidence) or not isinstance(
            confidence.basis, ConfidenceBasis
        ):
            raise EvidenceGraphError(
                "%s carries no confidence with a documented basis; an "
                "unattributable number in a graph cannot be reviewed" % where
            )
        if not (0.0 <= confidence.value <= 1.0):
            raise EvidenceGraphError(
                "%s carries a confidence of %r, outside the unit interval"
                % (where, confidence.value)
            )
        if relationship.temporal_relation is not None and (
            relationship.temporal_window is None
        ):
            raise EvidenceGraphError(
                "%s states an ordering with no window, which orders against "
                "nothing" % where
            )


def build_graph(relationships: Sequence[EvidenceRelationship]) -> EvidenceGraph:
    """Build a graph from *relationships* with a default builder.

    A convenience for the common case, defined in terms of
    :meth:`EvidenceGraphBuilder.build` so the two can never diverge.
    """
    return EvidenceGraphBuilder().build(relationships)
