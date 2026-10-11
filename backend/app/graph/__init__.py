"""The evidence graph (SIH26146 - Step 4C).

Step 4B produced the evidence
-----------------------------
The correlation package decides when observations match and returns a list of
``EvidenceRelationship``.  Each one states that a match exists, names the
versioned rule that found it, carries a confidence and the basis for that
confidence, cites the source records it rests on, and explains itself in a
rationale.

Step 4C draws it
---------------
That list is a sequence of statements, and a sequence is the wrong shape for
asking "what is this record connected to?".  This package supplies the missing
shape and nothing else:

    EvidenceRelationship[]
            |
            v
    EvidenceGraphBuilder.build(...)
            |
            v
    EvidenceGraph  --->  ObservationNode   (one per observation named)
                 --->  EvidenceEdge       (one per relationship, evidence intact)

The graph is a *representation*.  Every edge is a verbatim copy of one
relationship, every node is one of the observations that relationship names, and
nothing in this package derives, merges, strengthens or discards any of it.  A
``CORRELATED`` relationship becomes a ``CORRELATED`` edge or the build fails; a
``source -> target1``, ``source -> target2``, ``source -> target3`` group becomes
three independently addressable edges rather than one combined edge.

Determinism
-----------
Nodes are ordered by ``observation_id`` and edges by source side, target side,
then relationship id.  Both are total orders over content, so shuffling the input
cannot change the graph and building twice cannot either.  Node and edge
identifiers are derived by UUIDv5 in namespaces of their own, so a node can never
be mistaken for its observation, an edge for its relationship, or a node for an
edge.  The builder has no clock: ``created_at`` is carried from the relationship,
never restamped.

Two refusals
------------
A relationship that places an observation on both sides of itself is refused
rather than drawn, because a self-edge carries no information; and a repeated
``relationship_id`` is refused rather than deduplicated, because relationship ids
are derived from content and a repeat is the same evidence stated twice.  Both
raise :class:`EvidenceGraphError` and neither is repaired silently.

The boundary that matters
-------------------------
A node is an **observation**, not a person.  This package holds no field, and
offers no method, that could answer "who owns this wallet" or "who is behind this
IP" -- the question is not merely refused, it is unaskable, because there is
nowhere for an answer to be stored.  Two observations sharing an address are two
records that carry the same value, and that is the whole of it.
``explains_attribution()`` returns ``False`` on nodes, edges and the graph, so
the guarantee can be asserted rather than assumed.

What is not here
----------------
No path finding, no investigation paths, no risk scoring or propagation, no
centrality, no clustering or community detection, no anomaly detection, no
behavioural profiling, no temporal flow engine, no attribution, no identity
resolution, no learned model, no database, no Neo4j, no SQL, no REST endpoint and
no dashboard.  Those belong to later steps, and none of them is present here
under another name.

Layering
--------
This package depends on the correlation package and on nothing else.  It does
not import the Steps 1-3 substrate -- not the domain models, not the ingestion
service -- and it does not need to: the builder is given relationships, never
observations, so it never touches the substrate at all.  ``EvidenceRelationship``
is deliberately *not* re-exported here: it belongs to the layer that defines it,
and a caller who wants it imports it from there.
"""

from .builder import EvidenceGraphBuilder, build_graph
from .models import (
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

__all__ = [
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
]
