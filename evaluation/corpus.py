"""The deterministic Step 10 evaluation corpus.

The corpus is *synthetic, local, deterministic functional data*. It is not a
sample of real Bitcoin traffic and carries no claim about how often anything
occurs in the world. Its only purpose is to make each pipeline stage produce a
reproducible answer that can be stored and compared.

Nothing here is invented intelligence. Every scenario is either a real
ingestion of a file under ``evaluation/scenarios/`` or a relationship built
with the Step 4A public constructors, so the stages under test are the ones
that ship.

Case coverage (the sixteen functional cases Step 10 asks for)
-------------------------------------------------------------
===  ==============================================  ========================
 1   same TXID                                       ``flow_combined.csv``
 2   same address                                    ``flow_combined.csv``
 3   same-role IP                                    ``flow_combined.csv``
 4   temporal adjacency                              ``flow_combined.csv``
 5   amount match                                    ``flow_combined.csv``
 6   provenance relationship                         ``flow_combined.csv``
 7   geo coincidence                                 ``flow_combined.csv``
 8   mixed evidence (IP *and* txid in one record)     ``flow_combined.csv``
 9   multiple inference levels                      :func:`levelled_relationships`
10   comparable temporal data                        :func:`temporal_probes`
11   NOT_COMPARABLE temporal data                     :func:`temporal_probes`
12   missing temporal data                           ``flow_records.xml``
13   anomaly context                                 :func:`population_profiles`
14   clustering context                              :func:`population_profiles`
15   multi-hop investigation path                     :func:`levelled_relationships`
16   evidence-supported investigative priority        :func:`build_fusion_context`
===  ==============================================  ========================
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.app.behavior.profiles import BehaviorProfileBuilder
from backend.app.correlation import (
    CorrelationConfig,
    CorrelationEngine,
    EvidenceRelationship,
    EvidenceType,
    InferenceLevel,
)
from backend.app.correlation.models import (
    Confidence,
    ProvenanceRef,
    TemporalWindow,
    derive_relationship_id,
)
from backend.app.domain.adapters import normalized_to_unified
from backend.app.domain.models import (
    BlockchainTransaction,
    NetworkObservation,
    ObservationType,
    UnifiedObservation,
)
from backend.app.fusion import FusionContext
from backend.app.graph import build_graph
from backend.app.ingestion.service import IngestionService
from backend.app.investigation import (
    InvestigationEngine,
    PathSearchRequest,
    PathSearchResult,
    SearchStatus,
)
from backend.app.ml.anomaly import IsolationForestAnomalyDetector
from backend.app.ml.clustering import KMeansBehaviorClusterer

__all__ = [
    "SCENARIO_DIR",
    "SCENARIO_FILES",
    "UTC_EPOCH",
    "CORRELATION_WINDOW",
    "SUBJECT_POSITION",
    "load_observations",
    "stable_order",
    "relationship_observation_index",
    "run_correlation",
    "build_graph_from_relationships",
    "population_profiles",
    "levelled_relationships",
    "temporal_probes",
    "build_fusion_context",
    "choose_path_endpoints",
    "search_paths",
]

#: Repository-relative location of the raw scenario inputs.
SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"

#: The three formats the corpus is written in, ingested for real.
SCENARIO_FILES: Tuple[str, ...] = (
    "flow_combined.csv",
    "flow_records.json",
    "flow_records.xml",
)

#: A fixed instant every corpus timestamp is relative to. Nothing reads the
#: wall clock, so the corpus is identical on any machine at any time.
UTC_EPOCH = datetime(2024, 6, 1, tzinfo=timezone.utc)

#: Adjacency window the temporal rule is asked to use, pinned so the rule set
#: is reproducible. "Close together" is an analyst's choice, not a default, so
#: the evaluation states it rather than inheriting it silently.
CORRELATION_WINDOW = timedelta(minutes=30)

#: Which comparable observation is the subject fusion assesses.
#:
#: It is the hub of :func:`levelled_relationships`, so the fused priority sees
#: evidence at all four inference levels. Expressed as an index into the
#: content-ordered comparable observations so it does not depend on the order
#: the corpus happened to be read in.
SUBJECT_POSITION = 3


# --------------------------------------------------------------------------- #
# A. Ingestion
# --------------------------------------------------------------------------- #


def load_observations(
    scenario_dir: Optional[Path] = None,
) -> Tuple[List[UnifiedObservation], Dict[str, Any]]:
    """Ingest the whole corpus and adapt it to unified observations.

    Returns the observations plus a summary of what ingestion reported. The
    summary deliberately keeps counts and warnings but not the run-scoped
    ingestion identifier, which is a property of the run rather than of the
    evidence.
    """
    directory = Path(scenario_dir) if scenario_dir is not None else SCENARIO_DIR
    service = IngestionService()

    observations: List[UnifiedObservation] = []
    per_file: List[Dict[str, Any]] = []

    for name in SCENARIO_FILES:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError("missing evaluation scenario input: %s" % path)
        result = service.ingest_path(str(path), context={"client_filename": name})
        adapted = [normalized_to_unified(record) for record in result.records]
        observations.extend(adapted)
        per_file.append(
            {
                "file": name,
                "detected_format": result.source_format.value,
                "raw_record_count": result.raw_record_count,
                "record_count": result.record_count,
                "duplicate_count": result.duplicate_count,
                "error_count": result.error_count,
                "warning_count": result.warning_count,
                "observation_types": sorted(
                    {o.observation_type.value for o in adapted}
                ),
                "observation_ids": [o.observation_id for o in adapted],
                "error_codes": sorted(
                    {issue.code for o in adapted for issue in o.errors}
                ),
                "warning_codes": sorted(
                    {issue.code for o in adapted for issue in o.warnings}
                ),
                # The staged source file is a caller-owned input; ingestion must
                # leave it byte-identical.
                "source_bytes_unchanged": path.read_bytes()
                == (directory / name).read_bytes(),
            }
        )

    summary = {
        "files": per_file,
        "total_observations": len(observations),
        "distinct_observation_ids": len({o.observation_id for o in observations}),
        "classification_counts": _counts(
            o.observation_type for o in observations
        ),
    }
    return observations, summary


def _counts(values) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = value.value if hasattr(value, "value") else str(value)
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _epoch(**offset) -> datetime:
    return UTC_EPOCH + timedelta(**offset)


# --------------------------------------------------------------------------- #
# B/C. Correlation and the evidence graph
# --------------------------------------------------------------------------- #


def stable_order(
    observations: Sequence[UnifiedObservation],
) -> List[UnifiedObservation]:
    """Observations sorted by their content-derived id.

    ``observation_id`` is a UUIDv5 over record content, so this order is the
    same on every machine and every run. Anything that needs "the same record"
    must select through this rather than through a list index: a list index
    changes when the corpus is reordered, and a check that survives a shuffle
    only because it re-picked a different subject would be worthless.
    """
    return sorted(observations, key=lambda o: o.observation_id)


def _is_comparable(observation: UnifiedObservation) -> bool:
    """True when a carried timestamp can take part in an ordering."""
    from backend.app.investigation.temporal import is_aware

    return observation.timestamp is not None and is_aware(observation.timestamp)


def relationship_observation_index(
    observations: Sequence[UnifiedObservation],
) -> Dict[str, int]:
    """Map each observation_id to its position in *observations*."""
    return {o.observation_id: index for index, o in enumerate(observations)}


def run_correlation(
    observations: Sequence[UnifiedObservation],
) -> List[EvidenceRelationship]:
    """Run the shipped rule set over *observations* with pinned configuration.

    ``CorrelationConfig.now`` is pinned to :data:`UTC_EPOCH` because it is
    recorded on every relationship as ``created_at``. The shipped default
    reads the wall clock, which would make ``created_at`` the one field that
    changes on every run and would defeat the byte comparison the harness
    performs.
    """
    config = CorrelationConfig(
        temporal_window=CORRELATION_WINDOW,
        now=UTC_EPOCH,
    )
    return CorrelationEngine(config).run(list(observations))


def build_graph_from_relationships(
    relationships: Sequence[EvidenceRelationship],
):
    """Build an evidence graph with the shipped builder."""
    return build_graph(list(relationships))


# --------------------------------------------------------------------------- #
# D/E. Behavioural features and ML
# --------------------------------------------------------------------------- #


def _synthetic_observation(index: int, *, burst: int, amount: str) -> UnifiedObservation:
    """One deterministic population member, varied only by explicit arguments."""
    ts = _epoch(minutes=index)
    return UnifiedObservation(
        observation_id="eval-pop-%03d" % index,
        observation_type=ObservationType.COMBINED,
        timestamp=ts,
        network=NetworkObservation(
            timestamp=ts,
            src_ip="198.18.%d.%d" % (index // 250, index % 250),
            dst_ip="198.51.100.%d" % (index % 250),
            src_port=40000 + (index % 1000),
            dst_port=443,
            asn=64500 + (index % 3),
            geo_country="IN" if index % 2 == 0 else "DE",
        ),
        blockchain=BlockchainTransaction(
            timestamp=ts,
            txid="evalpop%058d" % index,
            input_addresses=["bc1qevalpop%045d" % index],
            output_addresses=["bc1qevalout%045d" % (index % 7)],
            input_amounts=[Decimal(amount)],
            output_amounts=[Decimal(amount)],
            fee=Decimal("0.00001"),
            script_type="P2WPKH",
        ),
        provenance=_synthetic_provenance(index),
    )


def _synthetic_provenance(index: int):
    from backend.app.domain.models import Provenance

    return Provenance(
        ingestion_id="eval-ingestion",
        source_file="evaluation_population_%03d.csv" % index,
        source_format="csv",
        source_record="row %d" % (index + 2),
        ingested_at=_epoch(minutes=index),
        context={"purpose": "step10-evaluation"},
    )


def population_profiles(count: int = 12):
    """A deterministic behavioural population for the ML stages.

    Members 0-7 form a low-burst group and members 8-11 a high-burst group, so
    clustering has a real structure to find and the anomaly detector has real
    outliers to rank. The grouping is stated here rather than discovered: the
    evaluation verifies that the shipped algorithms reproduce a known answer,
    not that they can find an unknown one.
    """
    builder = BehaviorProfileBuilder()
    profiles = []
    for index in range(count):
        burst = 40 if index < 8 else 400
        amount = "0.01000000" if index < 8 else "9.50000000"
        profiles.append(
            builder.build([_synthetic_observation(index, burst=burst, amount=amount)])
        )
    return tuple(profiles)


# --------------------------------------------------------------------------- #
# F. Explicitly levelled relationships and multi-hop paths
# --------------------------------------------------------------------------- #


def _provenance_ref(
    observation: UnifiedObservation,
) -> ProvenanceRef:
    """A provenance reference pointing at a real ingested record."""
    provenance = observation.provenance
    return ProvenanceRef(
        ingestion_id=provenance.ingestion_id,
        source_file=provenance.source_file,
        source_record=provenance.source_record,
    )


def _relationship(
    source: UnifiedObservation,
    target: UnifiedObservation,
    evidence_type: EvidenceType,
    inference_level: InferenceLevel,
    rationale: str,
    *,
    rule_id: str = "evaluation.rule.v1",
    rule_version: str = "1.0.0",
    confidence: Optional[Confidence] = None,
    temporal_window: Optional[TemporalWindow] = None,
) -> EvidenceRelationship:
    """Build one relationship with the Step 4A public constructor.

    Correlation rules always produce CORRELATED. The other inference levels are
    an analyst's characterisation layered on top of recorded evidence, so they
    can only be exercised by constructing relationships directly. Doing so
    tests the graph, path and fusion stages' handling of each level; it asserts
    nothing about which level a rule would have chosen.

    The identity is derived from the semantic payload only. ``created_at`` is
    excluded, exactly as the shipped builder excludes it, so a relationship
    re-derived tomorrow keeps its id.
    """
    if confidence is None:
        confidence = Confidence.from_rule(0.9)

    identity = derive_relationship_id(
        {
            "source_observation_ids": [source.observation_id],
            "target_observation_ids": [target.observation_id],
            "evidence_type": evidence_type.value,
            "inference_level": inference_level.value,
            "confidence": str(confidence.value),
            "rationale": rationale,
            "rule_id": rule_id,
            "rule_version": rule_version,
        }
    )
    return EvidenceRelationship(
        relationship_id=identity,
        source_observation_ids=(source.observation_id,),
        target_observation_ids=(target.observation_id,),
        evidence_type=evidence_type,
        inference_level=inference_level,
        confidence=confidence,
        provenance_refs=(
            _provenance_ref(source),
            _provenance_ref(target),
        ),
        rationale=rationale,
        rule_id=rule_id,
        rule_version=rule_version,
        created_at=UTC_EPOCH,
        temporal_window=temporal_window,
    )


def levelled_relationships(
    observations: Sequence[UnifiedObservation],
) -> List[EvidenceRelationship]:
    """Relationships carrying all four inference levels, incident to one subject.

    Correlation rules only ever produce CORRELATED, so the other three levels
    can only be exercised by constructing relationships directly with the Step
    4A constructor. Each one is attached to a single hub observation, which is
    also the subject fusion assesses. That way the fused priority is computed
    over evidence of every level rather than over a corpus whose levels happen
    to be spread across unrelated subjects.

    A separate chain supplies the multi-hop case. Members are selected by
    content-derived id and by timestamp comparability, never by position in the
    caller's list, so the same relationships come out whichever order the corpus
    was read in.
    """
    ordered = stable_order(observations)
    comparable = [o for o in ordered if _is_comparable(o)]
    if len(comparable) < 9:
        raise ValueError(
            "the corpus must supply at least 9 timestamp-comparable "
            "observations, got %d" % len(comparable)
        )

    hub = comparable[SUBJECT_POSITION]
    satellites = [
        observation
        for observation in comparable
        if observation.observation_id != hub.observation_id
    ]
    if len(satellites) < 8:
        raise ValueError(
            "the corpus must supply at least 8 satellites, got %d"
            % len(satellites)
        )

    # Four spokes on the hub, one per inference level, in level order. The
    # confidence figures fall as the level weakens, and none reaches certainty:
    # a rule-derived relationship is a judgement even when it is certain that
    # two records match.
    spokes = [
        (EvidenceType.SAME_TXID, InferenceLevel.OBSERVED,
         "Evaluation: the same transaction id appears in both records.", 0.9),
        (EvidenceType.SAME_IP, InferenceLevel.CORRELATED,
         "Evaluation: the same source address recorded on both flows.", 0.9),
        (EvidenceType.SAME_ADDRESS, InferenceLevel.INFERRED,
         "Evaluation: repeated address reuse was inferred, not observed.", 0.75),
        (EvidenceType.GEO_COINCIDENCE, InferenceLevel.REQUIRES_VERIFICATION,
         "Evaluation: a shared country code alone is not sufficient.", 0.35),
    ]

    relationships = []
    for evidence_type, level, rationale, confidence in spokes:
        relationships.append(
            _relationship(
                hub,
                satellites[len(relationships)],
                evidence_type,
                level,
                rationale,
                rule_id="evaluation.rule.%s" % evidence_type.value.lower(),
                confidence=Confidence.from_rule(confidence),
            )
        )

    # A three-hop chain among observations the hub does not already join
    # directly, so the multi-hop case has somewhere to be demonstrated.
    chain = satellites[4:7]
    earlier, later = sorted(
        (chain[0].timestamp, chain[1].timestamp), key=lambda value: value
    )
    relationships.extend(
        [
            _relationship(
                chain[0], chain[1], EvidenceType.TEMPORAL_ADJACENCY,
                InferenceLevel.CORRELATED,
                "Evaluation: the two records fall inside one adjacency window.",
                rule_id="evaluation.rule.temporal_adjacency",
                temporal_window=TemporalWindow(start=earlier, end=later),
            ),
            _relationship(
                chain[1], chain[2], EvidenceType.AMOUNT_MATCH,
                InferenceLevel.CORRELATED,
                "Evaluation: the transferred amounts match exactly.",
                rule_id="evaluation.rule.amount_match",
            ),
            _relationship(
                chain[2], satellites[7], EvidenceType.PROVENANCE_LINK,
                InferenceLevel.OBSERVED,
                "Evaluation: both records were ingested in the same run.",
                rule_id="evaluation.rule.provenance_link",
            ),
        ]
    )
    return relationships


def temporal_probes(
    observations: Sequence[UnifiedObservation],
) -> List[Dict[str, Any]]:
    """Classify every corpus timestamp and prove pairwise comparability.

    Two things are reported. First, how the pipeline classified each carried
    timestamp, using Step 7's own classifier rather than a local reimplementation.
    Second, what happens when two timestamps are put side by side: a pair of
    aware timestamps is comparable and yields an elapsed time, while a mixed
    pair and a pair with a missing side are not, and no elapsed time is
    invented for them.
    """
    from backend.app.investigation.temporal import compare_timestamps, timestamp_awareness

    aware = [o for o in observations if o.timestamp is not None and o.timestamp.tzinfo]
    naive = [o for o in observations if o.timestamp is not None and not o.timestamp.tzinfo]
    absent = [o for o in observations if o.timestamp is None]

    probes: List[Dict[str, Any]] = [
        {
            "corpus_index": index,
            "observation_id": observation.observation_id,
            "source_file": observation.provenance.source_file,
            "awareness": timestamp_awareness(observation.timestamp).value,
            "warning_codes": sorted(issue.code for issue in observation.warnings),
        }
        for index, observation in enumerate(observations)
    ]

    def _pair(label: str, source, target) -> Dict[str, Any]:
        comparison = compare_timestamps(source, target)
        return {
            "case": label,
            "comparable": comparison.comparable,
            "relation": comparison.relation.value,
            "elapsed_seconds": (
                comparison.elapsed.total_seconds()
                if comparison.elapsed is not None
                else None
            ),
        }

    pairs = [
        _pair(
            "comparable_temporal_data",
            aware[0].timestamp if aware else None,
            aware[1].timestamp if len(aware) > 1 else None,
        ),
        _pair(
            "not_comparable_temporal_data",
            aware[0].timestamp if aware else None,
            naive[0].timestamp if naive else None,
        ),
        _pair(
            "missing_temporal_data",
            aware[0].timestamp if aware else None,
            absent[0].timestamp if absent else None,
        ),
    ]

    return {
        "per_observation": probes,
        "awareness_counts": _counts(
            timestamp_awareness(o.timestamp).value for o in observations
        ),
        "pairwise": pairs,
    }


def search_paths(
    graph,
    observations: Sequence[UnifiedObservation],
    *,
    start: str,
    target: str,
    max_depth: int = 5,
    max_paths: int = 10,
    exhaustive: bool = False,
) -> PathSearchResult:
    """Search the recorded graph, supplying timestamps for temporal work.

    *exhaustive* switches from shortest-hop search to a bounded enumeration of
    every simple trail. Both are deterministic; they answer different
    questions, and the multi-hop case needs the second when the first would
    correctly return a single hop.
    """
    timestamps = {o.observation_id: o.timestamp for o in observations}
    request = PathSearchRequest(
        start_observation_id=start,
        target_observation_id=target,
        max_depth=max_depth,
        max_paths=max_paths,
        directed=False,
    )
    engine = InvestigationEngine()
    if exhaustive:
        return engine.enumerate_paths(
            graph, request, observations=timestamps
        )
    return engine.shortest_path(graph, request, observations=timestamps)


def build_search_result_from_paths(
    paths: Sequence[Any],
    *,
    status: SearchStatus,
    detail: str,
    start: str,
    target: str,
    request: PathSearchRequest,
) -> PathSearchResult:
    """Wrap assembled paths into the result the fusion layer expects."""
    ordered = list(paths)
    return PathSearchResult(
        status=status,
        detail=detail,
        request=request,
        paths=tuple(ordered),
        path_count=len(ordered),
        shortest_hops=min((p.hop_count for p in ordered), default=None),
        temporal_exclusions=0,
        incomparable_exclusions=0,
        search_scope="evaluation",
    )


# --------------------------------------------------------------------------- #
# G. Fusion: Evidence-Supported Investigative Priority
# --------------------------------------------------------------------------- #


def hop_distances(graph, start: str) -> Dict[str, int]:
    """Undirected hop distance from *start* to every reachable node.

    Breadth first, with neighbours visited in sorted order, so the traversal
    is deterministic. An unreachable node is simply absent from the result
    rather than given an invented distance.
    """
    distances = {start: 0}
    frontier = [start]
    while frontier:
        current = frontier.pop(0)
        for neighbour in sorted(graph.neighbors(current)):
            if neighbour not in distances:
                distances[neighbour] = distances[current] + 1
                frontier.append(neighbour)
    return distances


def choose_path_endpoints(
    graph,
    observations: Sequence[UnifiedObservation],
    hub: UnifiedObservation,
    *,
    require_multi_hop: bool = True,
) -> Tuple[str, str]:
    """Pick a target that makes the search interesting.

    A multi-hop case cannot be demonstrated on a pair the graph joins
    directly: ``shortest_path`` would correctly return the one-hop answer and
    the case would never be exercised. The furthest reachable observation by
    hop count is chosen, with the content-ordered id as tie-break, so the
    choice is reproducible and the resulting hop count is a property of the
    evidence rather than of the fixture.

    Raises :class:`ValueError` when every reachable observation is one hop away
    and *require_multi_hop* is set, because silently returning a one-hop pair
    would let a multi-hop check pass on a graph that has no multi-hop path.
    """
    distances = hop_distances(graph, hub.observation_id)
    by_id = {o.observation_id: o for o in observations}

    reachable = [
        observation
        for observation_id, observation in sorted(by_id.items())
        if observation_id in distances and observation_id != hub.observation_id
    ]
    if not reachable:
        raise ValueError(
            "no observation is reachable from the hub, so there is no path to "
            "search"
        )
    if require_multi_hop:
        multi = [
            observation
            for observation in reachable
            if distances[observation.observation_id] > 1
        ]
        if not multi:
            raise ValueError(
                "every reachable observation is one hop from the hub, so no "
                "multi-hop path exists to evaluate"
            )
        reachable = multi

    furthest = max(distances[o.observation_id] for o in reachable)
    candidates = [
        o for o in reachable if distances[o.observation_id] == furthest
    ]
    return hub.observation_id, candidates[0].observation_id


def build_fusion_context(
    observations: Sequence[UnifiedObservation],
    relationships: Sequence[EvidenceRelationship],
    *,
    population_size: int = 12,
    search_override: Optional[Any] = None,
    subject_override: Optional[UnifiedObservation] = None,
) -> FusionContext:
    """Assemble a complete FusionContext from real pipeline outputs.

    Nothing here fabricates a signal. The graph and relationships come from the
    correlation stage, the behavioural profile from the Step 5 builder, and the
    anomaly and clustering results from the Step 6 algorithms run over the
    deterministic population.

    The subject is chosen by content-derived id rather than by position, so a
    shuffled corpus produces the same assessment. It defaults to
    :data:`SUBJECT_POSITION` within the comparable observations, which is the
    observation :func:`levelled_relationships` connects at all four inference
    levels, so the fused priority is computed over evidence of every level.
    *search_override* lets a caller reuse a search it already performed against
    the same graph instead of running the engine twice on identical input.
    """
    ordered = stable_order(observations)
    comparable = [o for o in ordered if _is_comparable(o)]
    if len(comparable) <= SUBJECT_POSITION:
        raise ValueError(
            "the corpus must supply more than %d timestamp-comparable "
            "observations, got %d" % (SUBJECT_POSITION, len(comparable))
        )
    subject = subject_override if subject_override is not None else comparable[
        SUBJECT_POSITION
    ]

    builder = BehaviorProfileBuilder()
    profile = builder.build([subject])

    population = population_profiles(population_size)
    detector = IsolationForestAnomalyDetector()
    _, anomaly = detector.fit_score(population)

    clusterer = KMeansBehaviorClusterer()
    clustering = clusterer.fit(population).result()

    if search_override is not None:
        search = search_override
    else:
        graph = build_graph_from_relationships(relationships)
        start, target = choose_path_endpoints(graph, ordered, subject)
        search = search_paths(graph, observations, start=start, target=target)

    return FusionContext(
        profile=profile,
        graph=build_graph_from_relationships(relationships),
        anomaly=anomaly,
        clustering=clustering,
        paths=search,
    )