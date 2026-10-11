"""Temporal flow and investigation-path search (SIH26146 - Step 7).

Step 6 measured profiles and grouped them; this layer turns a caller-selected
starting observation into ordered, evidence-supported trails through the
recorded-evidence graph.  It is the *search* layer: everything it produces is a
structure over evidence that earlier steps already recorded, and it decides
nothing beyond that structure.

What it contains
----------------
* ``investigation.models`` -- the typed vocabulary of a path: transitions,
  path nodes, the path itself, search requests and results, and the enums that
  describe temporal relation and status.
* ``investigation.temporal`` -- timestamp comparison that never invents a
  timezone, plus the transition/flow builders that attach timestamps to steps.
* ``investigation.paths`` -- assembling steps into a path, ordering paths, and
  deduplicating paths whose content (observations and relationships) is the same
  even though they were discovered repeatedly.
* ``investigation.engine`` -- bounded BFS and bounded depth-first search over
  the recorded-evidence graph, both deterministic, both refusing to walk more
  than the caller (and the engine) allows.

Boundaries that matter
----------------------
A path is a **structure**, not a conclusion.  Nothing here scores a walk,
ranks an observation as suspicious, classifies it, or attributes it to anyone;
``explains_attribution()`` returns ``False`` everywhere in this package, so the
absence of attribution is assertable rather than assumed.  Step 6 anomaly and
cluster values may be *carried* on a path node as context supplied by the
caller, but they are never combined, compared or used to weigh an edge.  No
decision like a threat level, a verdict, or a priority belongs to this layer,
and none is representable here.

Determinism
-----------
Path ids are UUIDv5 derivations over content and scope, never clocks or input
order.  Neighbour order and result order are total orders over content, so
shuffling a request or the graph input cannot change the answer.

What this package depends on
----------------------------
The graph package (nodes and edges), the correlation package (relationship
fields carried verbatim on paths), and the ml package (profile context shapes).
It does not import the Step 1-3 substrate.
"""

from __future__ import annotations

from .engine import InvestigationEngine
from .models import (
    EdgeOrientation,
    EvidenceSummary,
    IncomparablePolicy,
    InvestigationError,
    InvestigationPath,
    PathSearchRequest,
    PathSearchResult,
    PATH_NAMESPACE,
    PathNode,
    PathTemporalStatus,
    PathTransition,
    ProfileContext,
    SearchStatus,
    TemporalFlowStep,
    TemporalOrder,
    TemporalTransition,
    TemporalWindowConfig,
    TimestampAwareness,
    derive_path_id,
)
from .paths import (
    WalkStep,
    assemble_path,
    compare_paths,
    deduplicate_paths,
)
from .temporal import (
    TemporalComparison,
    build_temporal_index,
    build_temporal_transition,
    compare_timestamps,
    is_aware,
    normalize_aware,
    timestamp_awareness,
)

__all__ = [
    "EdgeOrientation",
    "EvidenceSummary",
    "IncomparablePolicy",
    "InvestigationEngine",
    "InvestigationError",
    "InvestigationPath",
    "PATH_NAMESPACE",
    "PathNode",
    "PathSearchRequest",
    "PathSearchResult",
    "PathTemporalStatus",
    "PathTransition",
    "ProfileContext",
    "SearchStatus",
    "TemporalComparison",
    "TemporalFlowStep",
    "TemporalOrder",
    "TemporalTransition",
    "TemporalWindowConfig",
    "TimestampAwareness",
    "WalkStep",
    "assemble_path",
    "build_temporal_index",
    "build_temporal_transition",
    "compare_paths",
    "compare_timestamps",
    "deduplicate_paths",
    "derive_path_id",
    "is_aware",
    "normalize_aware",
    "timestamp_awareness",
]