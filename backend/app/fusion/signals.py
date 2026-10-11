"""Signal adapters: turn Steps 4-7 outputs into the 11 typed fusion signals.

This module is the *measuring* half of Step 8.  Each adapter reads one
already-established layer - Step 4C recorded-evidence graph, Step 5
behavioural profile, Step 6 anomaly and cluster results, Step 7 investigation
paths with their temporal structure - and reports one or two
:class:`~backend.app.fusion.models.FusionSignal` numbers about the assessed
subject.  Everything here is read-only: the adapters copy numbers and cite
evidence, and none of them changes a number, a level or a citation.

Missing is not zero
-------------------
A signal is ``PRESENT`` only when the number was actually measured.  ``ABSENT``
means the producer object was not supplied, ``UNAVAILABLE`` means it was
supplied but produced nothing for this subject (run never fitted, subject not
in the population), and ``NOT_APPLICABLE`` is an explicit "this question has no
answer for this subject".  None of the three is silently replaced by a zero;
a measured zero, on the other hand, is a real answer (an empty graph, an empty
path set) and is reported as ``PRESENT`` with the number zero.

Every family always emits at least one signal, so the per-family availability
used by the weighted average is always defined.

Two aggregate signals deliberately carry no relationship citations.  The
anomaly figure is a population-relative measurement of the whole profile, and
the behavioural coverage is a profile-level aggregate; neither rests on any
single recorded relationship, and each explanation says so instead of implying
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

from ..behavior.models import FeatureCategory, MeasurementStatus
from ..behavior.profiles import BehavioralProfile
from ..correlation.enums import EvidenceType, InferenceLevel
from ..correlation.provenance import ProvenanceRef
from ..fusion.models import (
    EVIDENCE_COUNT_CAP_DEFAULT,
    PATH_COUNT_CAP_DEFAULT,
    PROVENANCE_FILE_CAP_DEFAULT,
    FusionError,
    FusionSignal,
    SignalFamily,
    SignalRole,
    SignalStatus,
    derive_signal_id,
)
from ..graph.models import EvidenceEdge, EvidenceGraph
from ..investigation.models import InvestigationPath, PathSearchResult, TemporalOrder
from ..ml.models import AnomalyRunResult, ClusteringRunResult, ModelStatus

__all__ = [
    "FusionContext",
    "anomaly_signal",
    "behavioral_signal",
    "build_signals",
    "cluster_signal",
    "evidence_signals",
    "incident_edges",
    "path_signals",
    "provenance_signals",
    "temporal_signals",
]


@dataclass(frozen=True)
class FusionContext:
    """Everything the adapters read, framed as one immutable value object.

    The ``profile`` is the subject being assessed and is always required.
    Every other producer is optional; ``None`` means that producer was not
    supplied, which the adapters report as ``ABSENT``.  ``paths`` accepts
    either a Step 7 :class:`PathSearchResult` or a plain tuple of
    :class:`InvestigationPath`.
    """

    profile: BehavioralProfile
    graph: Optional[EvidenceGraph] = None
    anomaly: Optional[AnomalyRunResult] = None
    clustering: Optional[ClusteringRunResult] = None
    paths: Optional[Union[PathSearchResult, Tuple[InvestigationPath, ...]]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile, BehavioralProfile):
            raise FusionError(
                "FusionContext.profile must be a BehavioralProfile, got %s"
                % type(self.profile).__name__
            )
        for name, value, kind in (
            ("graph", self.graph, EvidenceGraph),
            ("anomaly", self.anomaly, AnomalyRunResult),
            ("clustering", self.clustering, ClusteringRunResult),
        ):
            if value is not None and not isinstance(value, kind):
                raise FusionError(
                    "FusionContext.%s must be a %s or None, got %s"
                    % (name, kind.__name__, type(value).__name__)
                )
        paths = self.paths
        if paths is not None:
            valid = isinstance(paths, PathSearchResult) or (
                isinstance(paths, tuple)
                and all(isinstance(item, InvestigationPath) for item in paths)
            )
            if not valid:
                raise FusionError(
                    "FusionContext.paths must be a PathSearchResult, a tuple of "
                    "InvestigationPath, or None"
                )

    @property
    def subject_paths(self) -> Tuple[InvestigationPath, ...]:
        """The supplied paths as a plain tuple, whatever form they arrived in."""
        if self.paths is None:
            return ()
        if isinstance(self.paths, PathSearchResult):
            return self.paths.paths
        return self.paths


def incident_edges(
    graph: EvidenceGraph, observation_ids: Sequence[str]
) -> Tuple[EvidenceEdge, ...]:
    """Edges naming at least one of *observation_ids*, in graph order.

    Edges are already in the graph's deterministic total order, so this is a
    pure filter: the result does not depend on anything but the set of edges
    and the set of observations.
    """
    if not isinstance(graph, EvidenceGraph):
        raise FusionError("incident_edges needs an EvidenceGraph, got %s" % type(graph).__name__)
    subject = set(observation_ids)
    return tuple(edge for edge in graph.edges() if subject & set(edge.observation_ids()))


def _ref_key(reference: ProvenanceRef) -> Tuple[str, str, str]:
    return (
        reference.ingestion_id,
        reference.source_file or "",
        reference.source_record or "",
    )


def _sorted_unique_refs(refs: Iterable[ProvenanceRef]) -> Tuple[ProvenanceRef, ...]:
    """De-duplicate and order provenance citations deterministically.

    ``None`` locators sort as the empty string; the comparison is over plain
    strings, so two citations that are identical in all three fields compare
    equal and collapse into one.
    """
    unique = {_ref_key(reference): reference for reference in refs}
    return tuple(unique[key] for key in sorted(unique))


def _evidence_of(
    edges: Sequence[EvidenceEdge],
) -> Tuple[Tuple[str, ...], Tuple[InferenceLevel, ...], Tuple[ProvenanceRef, ...]]:
    """The relationship ids, inference levels and citations an edge set rests on."""
    relationship_ids = tuple(sorted({edge.relationship_id for edge in edges}))
    levels = tuple(
        sorted({edge.inference_level for edge in edges}, key=lambda level: level.value)
    )
    refs = _sorted_unique_refs(
        reference for edge in edges for reference in edge.provenance_refs
    )
    return relationship_ids, levels, refs


def _signal(
    *,
    subject_id: str,
    signal_type: str,
    family: SignalFamily,
    source_step: int,
    status: SignalStatus,
    explanation: str,
    role: SignalRole = SignalRole.PRIORITY_SIGNAL,
    raw_value: Optional[float] = None,
    normalized_value: Optional[float] = None,
    evidence_refs: Tuple[str, ...] = (),
    inference_levels: Tuple[InferenceLevel, ...] = (),
    provenance_refs: Tuple[ProvenanceRef, ...] = (),
    lineage: Tuple[str, ...] = (),
    context: Tuple[Tuple[str, str], ...] = (),
) -> FusionSignal:
    """Build a signal with its content-derived identifier and profile source_id.

    Every signal about a subject uses the subject profile id as its
    ``source_id``: a signal is one number about the subject from one layer,
    and the layer's *output object that described the subject* is the profile.
    A duplicate type from the same step and subject therefore collides and is
    caught, rather than being silently added twice.
    """
    return FusionSignal(
        signal_id=derive_signal_id(subject_id, signal_type, source_step, subject_id),
        signal_type=signal_type,
        family=family,
        role=role,
        status=status,
        source_step=source_step,
        source_id=subject_id,
        subject_id=subject_id,
        raw_value=raw_value,
        normalized_value=normalized_value,
        explanation=explanation,
        evidence_refs=evidence_refs,
        inference_levels=inference_levels,
        provenance_refs=provenance_refs,
        lineage=lineage,
        context=context,
    )


def _subject_touching_paths(context: FusionContext) -> Tuple[InvestigationPath, ...]:
    """Paths from the context that touch at least one subject observation."""
    subject = set(context.profile.observation_ids)
    keep = [
        path
        for path in context.subject_paths
        if subject & set(path.observation_ids)
    ]
    return tuple(keep)


def anomaly_signal(context: FusionContext) -> FusionSignal:
    """The Step 6 anomaly normalized score, carried verbatim.

    The number is copied exactly as Step 6 produced it (``2 ** -raw``) and is
    never re-normalized, because Step 8 re-normalizing a fixed operating point
    would silently change its meaning.
    """
    subject = context.profile
    profile_id = subject.profile_id
    if context.anomaly is None:
        return _signal(
            subject_id=profile_id,
            signal_type="ANOMALY_NORMALIZED_SCORE",
            family=SignalFamily.ANOMALY,
            source_step=6,
            status=SignalStatus.ABSENT,
            explanation=(
                "The anomaly run result was not supplied, so the Step 6 "
                "normalized score for this profile could not be measured."
            ),
            lineage=(profile_id,),
        )
    run = context.anomaly
    if run.status is not ModelStatus.FITTED:
        return _signal(
            subject_id=profile_id,
            signal_type="ANOMALY_NORMALIZED_SCORE",
            family=SignalFamily.ANOMALY,
            source_step=6,
            status=SignalStatus.UNAVAILABLE,
            explanation=(
                "The anomaly run was supplied but never fitted (%s), so it "
                "produced no score for any profile."
                % run.status.value
            ),
            lineage=(profile_id,),
        )
    result = run.result_for(profile_id)
    if result is None or result.normalized_score is None:
        return _signal(
            subject_id=profile_id,
            signal_type="ANOMALY_NORMALIZED_SCORE",
            family=SignalFamily.ANOMALY,
            source_step=6,
            status=SignalStatus.UNAVAILABLE,
            explanation=(
                "The anomaly run was fitted but this profile is not part of "
                "the scored population, so no score was produced for it."
            ),
            lineage=(profile_id,),
        )
    return _signal(
        subject_id=profile_id,
        signal_type="ANOMALY_NORMALIZED_SCORE",
        family=SignalFamily.ANOMALY,
        role=SignalRole.PRIORITY_SIGNAL,
        source_step=6,
        status=SignalStatus.PRESENT,
        raw_value=float(result.raw_score),
        normalized_value=float(result.normalized_score),
        explanation=(
            "Anomaly normalized score carried verbatim from Step 6 (the "
            "documented 2 ** -raw remap); it is a population-relative "
            "measurement of the whole profile and cites no single relationship."
        ),
        lineage=(profile_id,),
        context=(
            ("band", result.band.value if result.band is not None else "none"),
            ("is_in_sample", "true" if result.is_in_sample else "false"),
            ("schema_id", result.schema_id),
        ),
    )


def behavioral_signal(context: FusionContext) -> FusionSignal:
    """The share of the 7 Step 5 feature categories actually measured.

    A category counts as measured when at least one of its features is
    ``OBSERVED`` or ``PARTIAL`` - a genuinely measured zero still is OBSERVED,
    and only ``ABSENT`` features are no measurement at all.
    """
    profile = context.profile
    measured = {
        feature.category
        for feature in profile.features
        if feature.status is not MeasurementStatus.ABSENT
    }
    denominator = len(FeatureCategory)
    coverage = float(len(measured)) / float(denominator)
    return _signal(
        subject_id=profile.profile_id,
        signal_type="BEHAVIORAL_FEATURE_COVERAGE",
        family=SignalFamily.BEHAVIORAL,
        role=SignalRole.EVIDENCE_STRENGTH,
        source_step=5,
        status=SignalStatus.PRESENT,
        raw_value=float(len(measured)),
        normalized_value=coverage,
        explanation=(
            "Behavioural coverage is the share of the %d Step 5 feature "
            "categories carrying at least one measured feature over the given "
            "profile; it measures profile characterization and evidence "
            "actionability, not behavioral suspiciousness."
            % denominator
        ),
        lineage=(profile.profile_id,),
        context=(
            ("feature_category_total", str(denominator)),
            ("feature_count", str(profile.feature_count)),
            ("measured_categories", str(len(measured))),
        ),
    )


def cluster_signal(context: FusionContext) -> FusionSignal:
    """Centroid distance remapped to a unit-interval centrality figure.

    ``1 - distance / pop_max``, where ``pop_max`` is the largest non-noise
    centroid distance in the fitted population, so the closest member of its
    group scores 1.  Membership is descriptive context only: it never boosts
    identity, and no relationship is created from it.
    """
    profile = context.profile
    profile_id = profile.profile_id
    if context.clustering is None:
        return _signal(
            subject_id=profile_id,
            signal_type="CLUSTER_CENTROID_DISTANCE",
            family=SignalFamily.CLUSTER_CONTEXT,
            source_step=6,
            status=SignalStatus.ABSENT,
            explanation=(
                "The clustering run result was not supplied, so no centroid "
                "distance could be measured for this profile."
            ),
            lineage=(profile_id,),
        )
    run = context.clustering
    if run.status is not ModelStatus.FITTED:
        return _signal(
            subject_id=profile_id,
            signal_type="CLUSTER_CENTROID_DISTANCE",
            family=SignalFamily.CLUSTER_CONTEXT,
            source_step=6,
            status=SignalStatus.UNAVAILABLE,
            explanation=(
                "The clustering run was supplied but never fitted (%s), so no "
                "partition exists to measure against."
                % run.status.value
            ),
            lineage=(profile_id,),
        )
    assignment = run.cluster_of(profile_id)
    if assignment is None:
        return _signal(
            subject_id=profile_id,
            signal_type="CLUSTER_CENTROID_DISTANCE",
            family=SignalFamily.CLUSTER_CONTEXT,
            source_step=6,
            status=SignalStatus.UNAVAILABLE,
            explanation=(
                "The clustering run was fitted but this profile is not part of "
                "the clustered population, so no assignment exists for it."
            ),
            lineage=(profile_id,),
        )
    if assignment.is_noise:
        return _signal(
            subject_id=profile_id,
            signal_type="CLUSTER_CENTROID_DISTANCE",
            family=SignalFamily.CLUSTER_CONTEXT,
            source_step=6,
            status=SignalStatus.NOT_APPLICABLE,
            explanation=(
                "This profile is noise: it was assigned to no group, so there "
                "is no centroid to measure distance from."
            ),
            lineage=(profile_id,),
        )
    distance = assignment.distance_to_centroid
    non_noise = [
        item.distance_to_centroid
        for item in run.assignments
        if not item.is_noise and item.distance_to_centroid is not None
    ]
    pop_max = max(non_noise) if non_noise else None
    if pop_max is None or pop_max <= 0:
        normalized = 0.0
    else:
        normalized = min(1.0, distance / pop_max)
    summary = run.cluster_summary(assignment.cluster_id)
    cluster_size = summary.member_count if summary is not None else 0
    return _signal(
        subject_id=profile_id,
        signal_type="CLUSTER_CENTROID_DISTANCE",
        family=SignalFamily.CLUSTER_CONTEXT,
        role=SignalRole.PRIORITY_SIGNAL,
        source_step=6,
        status=SignalStatus.PRESENT,
        raw_value=float(distance),
        normalized_value=float(normalized),
        explanation=(
            "Cluster centroid distance is the profile's distance to its "
            "assigned cluster centroid divided by the largest non-noise distance "
            "in the fitted population (distance / max non-noise distance); it "
            "measures geometric atypicality from the cluster center as descriptive "
            "context, never a statement of identity or ownership."
        ),
        lineage=tuple(sorted({profile_id, assignment.cluster_id})),
        context=(
            ("cluster_id", assignment.cluster_id),
            ("cluster_size", str(cluster_size)),
            ("normalization", "distance / max non-noise distance"),
        ),
    )


def evidence_signals(
    context: FusionContext, evidence_count_cap: int
) -> Tuple[FusionSignal, FusionSignal]:
    """Two counts over the recorded-evidence graph (Step 4C).

    Both are ``PRESENT`` with measured zeros when the graph is supplied but no
    incident edges exist, because an empty measured set is a real answer.
    """
    profile = context.profile
    profile_id = profile.profile_id
    if context.graph is None:
        absent = (
            "The recorded-evidence graph was not supplied, so no incident-"
            "edge count could be measured for this profile."
        )
        return (
            _signal(
                subject_id=profile_id,
                signal_type="EVIDENCE_INCIDENT_COUNT",
                family=SignalFamily.EVIDENCE_GRAPH,
                source_step=4,
                status=SignalStatus.ABSENT,
                explanation=absent,
                lineage=(profile_id,),
            ),
            _signal(
                subject_id=profile_id,
                signal_type="EVIDENCE_TYPE_DIVERSITY",
                family=SignalFamily.EVIDENCE_GRAPH,
                source_step=4,
                status=SignalStatus.ABSENT,
                explanation=(
                    "The recorded-evidence graph was not supplied, so no "
                    "evidence-type diversity could be measured."
                ),
                lineage=(profile_id,),
            ),
        )
    edges = incident_edges(context.graph, profile.observation_ids)
    relationship_ids, levels, refs = _evidence_of(edges)
    incident_count = len(edges)
    count = _signal(
        subject_id=profile_id,
        signal_type="EVIDENCE_INCIDENT_COUNT",
        family=SignalFamily.EVIDENCE_GRAPH,
        role=SignalRole.PRIORITY_SIGNAL,
        source_step=4,
        status=SignalStatus.PRESENT,
        raw_value=float(incident_count),
        normalized_value=float(min(1.0, incident_count / evidence_count_cap)),
        explanation=(
            "Incident-edge count normalised as min(1, incident / cap) with cap "
            "disclosed in the configuration; incident edges are the recorded "
            "relationships naming this profile's observations.  Zero is a "
            "measured zero."
        ),
        evidence_refs=relationship_ids,
        inference_levels=levels,
        provenance_refs=refs,
        lineage=(profile_id,),
        context=(
            ("evidence_count_cap", str(evidence_count_cap)),
            ("incident_edge_count", str(incident_count)),
        ),
    )
    distinct_types = {edge.evidence_type for edge in edges}
    diversity = _signal(
        subject_id=profile_id,
        signal_type="EVIDENCE_TYPE_DIVERSITY",
        family=SignalFamily.EVIDENCE_GRAPH,
        role=SignalRole.EVIDENCE_STRENGTH,
        source_step=4,
        status=SignalStatus.PRESENT,
        raw_value=float(len(distinct_types)),
        normalized_value=float(len(distinct_types)) / float(len(EvidenceType)),
        explanation=(
            "Evidence-type diversity is the share of the %d recorded evidence "
            "types present among the profile's incident edges; it measures how "
            "many kinds of recorded match the evidence rests on."
            % len(EvidenceType)
        ),
        evidence_refs=relationship_ids,
        inference_levels=levels,
        provenance_refs=refs,
        lineage=(profile_id,),
        context=(
            ("distinct_evidence_types", str(len(distinct_types))),
            ("evidence_type_total", str(len(EvidenceType))),
        ),
    )
    return count, diversity


def provenance_signals(
    context: FusionContext, provenance_file_cap: int
) -> Tuple[FusionSignal, FusionSignal]:
    """Two provenance figures over the subject's incident edges (Steps 2/4A)."""
    profile = context.profile
    profile_id = profile.profile_id
    if context.graph is None:
        return (
            _signal(
                subject_id=profile_id,
                signal_type="PROVENANCE_CITED_EDGE_RATIO",
                family=SignalFamily.PROVENANCE,
                source_step=4,
                status=SignalStatus.ABSENT,
                explanation=(
                    "The recorded-evidence graph was not supplied, so the "
                    "citation ratio over incident edges could not be measured."
                ),
                lineage=(profile_id,),
            ),
            _signal(
                subject_id=profile_id,
                signal_type="PROVENANCE_SOURCE_FILE_DIVERSITY",
                family=SignalFamily.PROVENANCE,
                source_step=4,
                status=SignalStatus.ABSENT,
                explanation=(
                    "The recorded-evidence graph was not supplied, so source-"
                    "file diversity could not be measured."
                ),
                lineage=(profile_id,),
            ),
        )
    edges = incident_edges(context.graph, profile.observation_ids)
    relationship_ids, levels, refs = _evidence_of(edges)
    incident_count = len(edges)
    cited_count = sum(1 for edge in edges if edge.provenance_refs)
    ratio = 0.0 if incident_count == 0 else cited_count / incident_count
    ratio_signal = _signal(
        subject_id=profile_id,
        signal_type="PROVENANCE_CITED_EDGE_RATIO",
        family=SignalFamily.PROVENANCE,
        role=SignalRole.EVIDENCE_STRENGTH,
        source_step=4,
        status=SignalStatus.PRESENT,
        raw_value=float(cited_count),
        normalized_value=float(ratio),
        explanation=(
            "Citation ratio is the share of the profile's incident edges that "
            "carry at least one provenance citation back to source evidence; "
            "no incident edges is a measured zero, not a missing value."
        ),
        evidence_refs=relationship_ids,
        inference_levels=levels,
        provenance_refs=refs,
        lineage=(profile_id,),
        context=(
            ("cited_edge_count", str(cited_count)),
            ("incident_edge_count", str(incident_count)),
        ),
    )
    source_files = set(profile.provenance_source_files)
    for reference in refs:
        if reference.source_file:
            source_files.add(reference.source_file)
    file_count = len(source_files)
    diversity = _signal(
        subject_id=profile_id,
        signal_type="PROVENANCE_SOURCE_FILE_DIVERSITY",
        family=SignalFamily.PROVENANCE,
        role=SignalRole.EVIDENCE_STRENGTH,
        source_step=4,
        status=SignalStatus.PRESENT,
        raw_value=float(file_count),
        normalized_value=float(min(1.0, file_count / provenance_file_cap)),
        explanation=(
            "Source-file diversity is min(1, distinct source files / cap) over "
            "the union of the files cited by the profile's incident edges and "
            "the files the profile itself came from, with the cap disclosed in "
            "the configuration."
        ),
        evidence_refs=relationship_ids,
        inference_levels=levels,
        provenance_refs=refs,
        lineage=(profile_id,),
        context=(
            ("distinct_source_files", str(file_count)),
            ("provenance_file_cap", str(provenance_file_cap)),
        ),
    )
    return ratio_signal, diversity


def temporal_signals(context: FusionContext) -> Tuple[FusionSignal, FusionSignal]:
    """Two temporal figures over paths that touch the subject (Step 7).

    ``TEMPORAL_COMPARABILITY`` is the share of those paths' transitions whose
    two timestamps are comparable; ``TEMPORAL_FORWARD_FLOW`` is the share of
    the comparable transitions whose timestamps move forward along the path.
    Both are undefined for a subject whose supplied paths carry no such
    transitions, which is reported as ``NOT_APPLICABLE``, never as a zero.
    """
    profile = context.profile
    profile_id = profile.profile_id
    if context.paths is None:
        return (
            _signal(
                subject_id=profile_id,
                signal_type="TEMPORAL_COMPARABILITY",
                family=SignalFamily.TEMPORAL,
                source_step=7,
                status=SignalStatus.ABSENT,
                explanation=(
                    "No investigation paths were supplied, so no temporal "
                    "structure could be measured for this profile."
                ),
                lineage=(profile_id,),
            ),
            _signal(
                subject_id=profile_id,
                signal_type="TEMPORAL_FORWARD_FLOW",
                family=SignalFamily.TEMPORAL,
                source_step=7,
                status=SignalStatus.ABSENT,
                explanation=(
                    "No investigation paths were supplied, so no forward "
                    "temporal flow could be measured for this profile."
                ),
                lineage=(profile_id,),
            ),
        )
    paths = _subject_touching_paths(context)
    if not paths:
        return (
            _signal(
                subject_id=profile_id,
                signal_type="TEMPORAL_COMPARABILITY",
                family=SignalFamily.TEMPORAL,
                source_step=7,
                status=SignalStatus.UNAVAILABLE,
                explanation=(
                    "Investigation paths were supplied, but none of them "
                    "touches this profile's observations, so no temporal "
                    "structure exists for it."
                ),
                lineage=(profile_id,),
            ),
            _signal(
                subject_id=profile_id,
                signal_type="TEMPORAL_FORWARD_FLOW",
                family=SignalFamily.TEMPORAL,
                source_step=7,
                status=SignalStatus.UNAVAILABLE,
                explanation=(
                    "Investigation paths were supplied, but none of them "
                    "touches this profile's observations, so no forward flow "
                    "could be measured for it."
                ),
                lineage=(profile_id,),
            ),
        )
    total = 0
    comparable = 0
    after = 0
    relationships: set[str] = set()
    levels: set[InferenceLevel] = set()
    for path in paths:
        for transition in path.transitions:
            total += 1
            relationships.add(transition.relationship_id)
            levels.add(transition.inference_level)
            temporal = transition.temporal
            if temporal is None or not temporal.comparable:
                continue
            comparable += 1
            if temporal.relation is TemporalOrder.AFTER:
                after += 1
    relationship_ids = tuple(sorted(relationships))
    inference_levels = tuple(sorted(levels, key=lambda level: level.value))
    lineage = tuple(sorted({profile_id, *(path.path_id for path in paths)}))

    if total == 0:
        comparability = _signal(
            subject_id=profile_id,
            signal_type="TEMPORAL_COMPARABILITY",
            family=SignalFamily.TEMPORAL,
            source_step=7,
            status=SignalStatus.NOT_APPLICABLE,
            explanation=(
                "The paths touching this profile carry no transitions, so "
                "there is no temporal structure to measure comparability "
                "against."
            ),
            evidence_refs=relationship_ids,
            inference_levels=inference_levels,
            lineage=lineage,
        )
    else:
        comparability = _signal(
            subject_id=profile_id,
            signal_type="TEMPORAL_COMPARABILITY",
            family=SignalFamily.TEMPORAL,
            role=SignalRole.EVIDENCE_STRENGTH,
            source_step=7,
            status=SignalStatus.PRESENT,
            raw_value=float(comparable),
            normalized_value=float(comparable / total),
            explanation=(
                "Temporal comparability is the share of transitions on the "
                "paths touching this profile whose two timestamps are "
                "comparable; a zero is a measured zero across real paths."
            ),
            evidence_refs=relationship_ids,
            inference_levels=inference_levels,
            lineage=lineage,
            context=(
                ("comparable_transitions", str(comparable)),
                ("total_transitions", str(total)),
            ),
        )

    if comparable == 0:
        forward = _signal(
            subject_id=profile_id,
            signal_type="TEMPORAL_FORWARD_FLOW",
            family=SignalFamily.TEMPORAL,
            role=SignalRole.CONTEXT_ONLY,
            source_step=7,
            status=SignalStatus.NOT_APPLICABLE,
            explanation=(
                "None of the transitions on the paths touching this profile is "
                "comparable, so a forward-flow ratio over comparables is "
                "undefined for it."
            ),
            evidence_refs=relationship_ids,
            inference_levels=inference_levels,
            lineage=lineage,
        )
    else:
        forward = _signal(
            subject_id=profile_id,
            signal_type="TEMPORAL_FORWARD_FLOW",
            family=SignalFamily.TEMPORAL,
            role=SignalRole.CONTEXT_ONLY,
            source_step=7,
            status=SignalStatus.PRESENT,
            raw_value=float(after),
            normalized_value=float(after / comparable),
            explanation=(
                "Forward flow is the share of the comparable transitions on "
                "the paths touching this profile whose timestamps move forward "
                "along the path (relation AFTER); it is descriptive context confirming "
                "chronological coherence and does not inflate investigative priority."
            ),
            evidence_refs=relationship_ids,
            inference_levels=inference_levels,
            lineage=lineage,
            context=(
                ("after_transitions", str(after)),
                ("comparable_transitions", str(comparable)),
            ),
        )
    return comparability, forward


def path_signals(context: FusionContext, path_count_cap: int) -> Tuple[FusionSignal, FusionSignal]:
    """Two structural figures over paths that touch the subject (Step 7).

    Supplying a path set makes these measurable, including as measured zeros
    when a search found nothing for the subject; not supplying one makes them
    ``ABSENT``.
    """
    profile = context.profile
    profile_id = profile.profile_id
    if context.paths is None:
        return (
            _signal(
                subject_id=profile_id,
                signal_type="PATH_REACHABILITY_FOUND",
                family=SignalFamily.PATH_STRUCTURE,
                source_step=7,
                status=SignalStatus.ABSENT,
                explanation=(
                    "No investigation paths were supplied, so whether any "
                    "trail touches this profile could not be measured."
                ),
                lineage=(profile_id,),
            ),
            _signal(
                subject_id=profile_id,
                signal_type="PATH_STRUCTURE_COUNT",
                family=SignalFamily.PATH_STRUCTURE,
                source_step=7,
                status=SignalStatus.ABSENT,
                explanation=(
                    "No investigation paths were supplied, so the number of "
                    "trails touching this profile could not be measured."
                ),
                lineage=(profile_id,),
            ),
        )
    paths = _subject_touching_paths(context)
    count = len(paths)
    relationships: set[str] = set()
    levels: set[InferenceLevel] = set()
    for path in paths:
        relationships.update(path.relationship_ids)
        levels.update(
            transition.inference_level for transition in path.transitions
        )
    relationship_ids = tuple(sorted(relationships))
    inference_levels = tuple(sorted(levels, key=lambda level: level.value))
    lineage = tuple(sorted({profile_id, *(path.path_id for path in paths)}))
    reachability = _signal(
        subject_id=profile_id,
        signal_type="PATH_REACHABILITY_FOUND",
        family=SignalFamily.PATH_STRUCTURE,
        role=SignalRole.PRIORITY_SIGNAL,
        source_step=7,
        status=SignalStatus.PRESENT,
        raw_value=float(1.0 if count else 0.0),
        normalized_value=float(1.0 if count else 0.0),
        explanation=(
            "Reachability is 1 when the supplied path set contains at least "
            "one trail touching this profile and 0 otherwise; it reports "
            "whether a trail was found, never how strong one is."
        ),
        evidence_refs=relationship_ids,
        inference_levels=inference_levels,
        lineage=lineage,
        context=(
            ("path_count_cap", str(path_count_cap)),
            ("subject_path_count", str(count)),
        ),
    )
    structure = _signal(
        subject_id=profile_id,
        signal_type="PATH_STRUCTURE_COUNT",
        family=SignalFamily.PATH_STRUCTURE,
        role=SignalRole.PRIORITY_SIGNAL,
        source_step=7,
        status=SignalStatus.PRESENT,
        raw_value=float(count),
        normalized_value=float(min(1.0, count / path_count_cap)),
        explanation=(
            "Path-structure count is min(1, subject-trail count / cap) with "
            "the cap disclosed in the configuration; a search that found "
            "nothing for this profile is a measured zero."
        ),
        evidence_refs=relationship_ids,
        inference_levels=inference_levels,
        lineage=lineage,
        context=(
            ("path_count_cap", str(path_count_cap)),
            ("subject_path_count", str(count)),
        ),
    )
    return reachability, structure


def build_signals(
    context: FusionContext,
    *,
    evidence_count_cap: int = EVIDENCE_COUNT_CAP_DEFAULT,
    path_count_cap: int = PATH_COUNT_CAP_DEFAULT,
    provenance_file_cap: int = PROVENANCE_FILE_CAP_DEFAULT,
) -> Tuple[FusionSignal, ...]:
    """Run every adapter and return the 11 signals, ordered by signal id.

    The three caps are the same disclosed numbers the configuration records;
    passing different ones here would produce different (but still
    self-consistent) signals, which is why they are also stored on the
    assessment's configuration.
    """
    if not isinstance(context, FusionContext):
        raise FusionError(
            "build_signals needs a FusionContext, got %s" % type(context).__name__
        )
    for name, value in (
        ("evidence_count_cap", evidence_count_cap),
        ("path_count_cap", path_count_cap),
        ("provenance_file_cap", provenance_file_cap),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise FusionError("%s must be a positive integer" % name)
    signals = (
        anomaly_signal(context),
        behavioral_signal(context),
        cluster_signal(context),
        *evidence_signals(context, evidence_count_cap),
        *provenance_signals(context, provenance_file_cap),
        *temporal_signals(context),
        *path_signals(context, path_count_cap),
    )
    ordered = tuple(sorted(signals, key=lambda signal: signal.signal_id))
    ids = [signal.signal_id for signal in ordered]
    if len(ids) != len(set(ids)):
        raise FusionError("signal adapters collided: a duplicate signal id was produced")
    families = {signal.family for signal in ordered}
    if families != set(SignalFamily):
        missing = sorted(set(SignalFamily) - families, key=lambda item: item.value)
        raise FusionError(
            "every family must emit at least one signal; missing %s"
            % [item.value for item in missing]
        )
    return ordered