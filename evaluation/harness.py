"""The Step 10 evaluation harness.

Eight dimensions, each run against the real pipeline:

============================  ============================================
``ingestion_normalization``   CSV/JSON/XML ingest, classification, issues
``evidence_correlation``      Step 4B rules over the ingested corpus
``evidence_graph``            Step 4C builder over those relationships
``behavioral_features``       Step 5 profile builder over real observations
``ml_determinism``            Step 6 isolation forest and k-means
``path_determinism``          Step 7 traversal with temporal comparison
``fusion_priority``           Step 8 Evidence-Supported Investigative Priority
``dashboard_payload``         Step 9A dashboard adapter output
============================  ============================================

Three things are deliberately absent
------------------------------------
No accuracy, precision, recall, F1, AUC or detection rate is computed
anywhere in this module. The corpus is synthetic and carries no labels, so any
such number would be invented. See :data:`NOT_COMPUTED`.

No dimension reaches for stored answers. Each one calls the shipped code and
records what came back. Expected-output files are compared *after* the fact by
:func:`compare_to_expected`; they are not an input to producing the numbers.

Order is never assumed. Every dimension sorts its own output by a stable key,
and the shuffle checks re-run whole stages on a reversed and rotated input to
confirm the shipped code does not depend on the order records arrive in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.app.behavior.profiles import BehaviorProfileBuilder
from backend.app.dashboard.adapter import build_dashboard_data
from backend.app.fusion import FusionEngine
from backend.app.ml.anomaly import IsolationForestAnomalyDetector
from backend.app.ml.clustering import KMeansBehaviorClusterer

from . import corpus
from .canonical import canonical, digest, find_run_scoped

__all__ = [
    "DIMENSIONS",
    "CASE_COVERAGE",
    "NOT_COMPUTED",
    "EvaluationOutcome",
    "run_dimension",
    "run_all",
    "compact_view",
    "compare_to_expected",
    "diff_canonical",
]

#: The metrics this project refuses to invent, named so a reader can see the
#: gap is deliberate rather than an oversight. The corpus has no ground truth,
#: so none of these is computable, and reporting any of them would be fiction.
NOT_COMPUTED: Tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "detection_rate",
    "false_positive_rate",
)

#: Dimension identifiers, in report order.
DIMENSIONS: Tuple[str, ...] = (
    "ingestion_normalization",
    "evidence_correlation",
    "evidence_graph",
    "behavioral_features",
    "ml_determinism",
    "path_determinism",
    "fusion_priority",
    "dashboard_payload",
)

#: The sixteen functional cases, mapped to the dimension that exercises each
#: one and to the artifact that makes it reproducible.
CASE_COVERAGE: Tuple[Dict[str, str], ...] = (
    {
        "case": "same_txid",
        "description": "two records carrying the same transaction id",
        "exercised_by": "evidence_correlation",
        "evidence": "evaluation/scenarios/flow_combined.csv",
    },
    {
        "case": "same_address",
        "description": "records sharing a bitcoin address",
        "exercised_by": "evidence_correlation",
        "evidence": "evaluation/scenarios/flow_combined.csv",
    },
    {
        "case": "same_role_ip",
        "description": "records sharing the same source IP",
        "exercised_by": "evidence_correlation",
        "evidence": "evaluation/scenarios/flow_combined.csv",
    },
    {
        "case": "temporal_adjacency",
        "description": "records falling inside one adjacency window",
        "exercised_by": "evidence_correlation",
        "evidence": "evaluation/scenarios/flow_combined.csv",
    },
    {
        "case": "amount_match",
        "description": "records transferring identical amounts",
        "exercised_by": "evidence_correlation",
        "evidence": "evaluation/scenarios/flow_combined.csv",
    },
    {
        "case": "provenance_relationship",
        "description": "records ingested together, sharing provenance",
        "exercised_by": "evidence_graph",
        "evidence": "evaluation/corpus.py:levelled_relationships",
    },
    {
        "case": "geo_coincidence",
        "description": "records sharing only a country code",
        "exercised_by": "evidence_correlation",
        "evidence": "evaluation/scenarios/flow_combined.csv",
    },
    {
        "case": "mixed_evidence",
        "description": "one record matching on IP and on txid at once",
        "exercised_by": "evidence_correlation",
        "evidence": "evaluation/scenarios/flow_combined.csv",
    },
    {
        "case": "multiple_inference_levels",
        "description": "all four inference levels carried through graph and fusion",
        "exercised_by": "fusion_priority",
        "evidence": "evaluation/corpus.py:levelled_relationships",
    },
    {
        "case": "comparable_temporal_data",
        "description": "two aware timestamps yielding a real elapsed time",
        "exercised_by": "path_determinism",
        "evidence": "evaluation/corpus.py:temporal_probes",
    },
    {
        "case": "not_comparable_temporal_data",
        "description": "a mixed aware/naive pair refusing to be ordered",
        "exercised_by": "path_determinism",
        "evidence": "evaluation/corpus.py:temporal_probes",
    },
    {
        "case": "missing_temporal_data",
        "description": "a record with no timestamp at all",
        "exercised_by": "ingestion_normalization",
        "evidence": "evaluation/scenarios/flow_records.xml",
    },
    {
        "case": "anomaly_context",
        "description": "outlier scores over a deterministic population",
        "exercised_by": "ml_determinism",
        "evidence": "evaluation/corpus.py:population_profiles",
    },
    {
        "case": "clustering_context",
        "description": "two behavioural clusters over the same population",
        "exercised_by": "ml_determinism",
        "evidence": "evaluation/corpus.py:population_profiles",
    },
    {
        "case": "multi_hop_path",
        "description": "a path of more than one hop across recorded evidence",
        "exercised_by": "path_determinism",
        "evidence": "evaluation/corpus.py:levelled_relationships",
    },
    {
        "case": "evidence_supported_investigative_priority",
        "description": "the Step 8 fused priority and its recorded basis",
        "exercised_by": "fusion_priority",
        "evidence": "backend/app/fusion",
    },
)


@dataclass(frozen=True)
class EvaluationOutcome:
    """One complete evaluation run."""

    dimensions: Dict[str, Any]
    case_coverage: Tuple[Dict[str, str], ...]
    shuffle_checks: Dict[str, Any]
    invariants: Dict[str, Any]
    dimensions_digest: str
    run_scoped_paths: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "dimensions_digest": self.dimensions_digest,
            "case_coverage": list(self.case_coverage),
            "shuffle_checks": self.shuffle_checks,
            "invariants": self.invariants,
            "run_scoped_substitutions": list(self.run_scoped_paths),
            "not_computed": list(NOT_COMPUTED),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sorted_by_id(items: Sequence[Any], attr: str = "observation_id") -> List[str]:
    return sorted(str(getattr(item, attr)) for item in items)


def _relationship_label(relationship, index: Dict[str, int]) -> str:
    """A stable, readable name for a relationship.

    Corpus positions are used instead of UUIDs so that a stored expected output
    is reviewable by a human, and so a shuffle cannot change it.
    """
    sources = ",".join(str(index.get(oid, oid)) for oid in relationship.source_observation_ids)
    targets = ",".join(str(index.get(oid, oid)) for oid in relationship.target_observation_ids)
    return "%s->%s/%s/%s" % (
        sources,
        targets,
        relationship.evidence_type.value,
        relationship.inference_level.value,
    )


# --------------------------------------------------------------------------- #
# Dimension 1: ingestion and normalization
# --------------------------------------------------------------------------- #


def dimension_ingestion(observations, summary) -> Dict[str, Any]:
    """What ingestion and normalization produced, and what it objected to."""
    index = corpus.relationship_observation_index(observations)
    rows = []
    for position, observation in enumerate(observations):
        network = observation.network
        chain = observation.blockchain
        rows.append(
            {
                "corpus_index": position,
                "observation_id": observation.observation_id,
                "observation_type": observation.observation_type.value,
                "timestamp": observation.timestamp,
                "source_format": observation.provenance.source_format,
                "source_file": observation.provenance.source_file,
                "source_record": observation.provenance.source_record,
                "src_ip": str(network.src_ip) if network else None,
                "src_port": network.src_port if network else None,
                "dst_ip": str(network.dst_ip) if network else None,
                "dst_port": network.dst_port if network else None,
                "asn": network.asn if network else None,
                "geo_country": network.geo_country if network else None,
                "txid": chain.txid if chain else None,
                "input_address_count": len(chain.input_addresses) if chain else 0,
                "output_address_count": len(chain.output_addresses) if chain else 0,
                "error_codes": sorted(issue.code for issue in observation.errors),
                "warning_codes": sorted(issue.code for issue in observation.warnings),
            }
        )

    by_file = summary["files"]
    return {
        "summary": {
            "files_ingested": len(by_file),
            "formats": sorted({entry["detected_format"] for entry in by_file}),
            "total_observations": summary["total_observations"],
            "classification_counts": summary["classification_counts"],
            "distinct_observation_ids": summary["distinct_observation_ids"],
            "observations_with_errors": sum(
                1 for row in rows if row["error_codes"]
            ),
            "observations_with_warnings": sum(
                1 for row in rows if row["warning_codes"]
            ),
            "observations_without_timestamp": sum(
                1 for row in rows if row["timestamp"] is None
            ),
            "observations_with_naive_timestamp": sum(
                1
                for row in rows
                if row["timestamp"] is not None
                and isinstance(row["timestamp"], str)
                and not row["timestamp"].endswith("+00:00")
            ),
        },
        "files": by_file,
        "observations": rows,
        "temporal_classification": corpus.temporal_probes(observations),
        "index_positions": index,
    }


# --------------------------------------------------------------------------- #
# Dimension 2: correlation
# --------------------------------------------------------------------------- #


def dimension_correlation(relationships, index) -> Dict[str, Any]:
    """What the rule set derived, recorded by position rather than by UUID."""
    rows = [
        {
            "relationship_id": relationship.relationship_id,
            "label": _relationship_label(relationship, index),
            "source_positions": [
                index.get(oid) for oid in relationship.source_observation_ids
            ],
            "target_positions": [
                index.get(oid) for oid in relationship.target_observation_ids
            ],
            "evidence_type": relationship.evidence_type.value,
            "inference_level": relationship.inference_level.value,
            "confidence": relationship.confidence.value,
            "confidence_basis": relationship.confidence.basis,
            "rule_id": relationship.rule_id,
            "rule_version": relationship.rule_version,
            "has_temporal_window": relationship.temporal_window is not None,
            "provenance_ref_count": len(relationship.provenance_refs),
        }
        for relationship in sorted(relationships, key=lambda r: r.relationship_id)
    ]
    types = sorted({relationship.evidence_type.value for relationship in relationships})
    levels = sorted({relationship.inference_level.value for relationship in relationships})
    return {
        "relationship_count": len(relationships),
        "evidence_types_present": types,
        "evidence_type_counts": _tally(row["evidence_type"] for row in rows),
        "inference_levels_present": levels,
        "inference_level_counts": _tally(row["inference_level"] for row in rows),
        "rules_that_fired": sorted({row["rule_id"] for row in rows}),
        "relationships": rows,
    }


def _tally(values) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return {key: counts[key] for key in sorted(counts)}


# --------------------------------------------------------------------------- #
# Dimension 3: the evidence graph
# --------------------------------------------------------------------------- #


def dimension_graph(graph, relationships, observations, index) -> Dict[str, Any]:
    """Graph structure, including that every recorded relationship survived."""
    edges = list(graph.edges())
    nodes = list(graph.nodes())
    types = {o.observation_id: o.observation_type.value for o in observations}

    rows = []
    for node in sorted(nodes, key=lambda n: n.observation_id):
        position = index.get(node.observation_id)
        degree = len(graph.outgoing_edges(node.observation_id)) + len(
            graph.incoming_edges(node.observation_id)
        )
        rows.append(
            {
                "position": position,
                "observation_id": node.observation_id,
                "node_id": node.node_id,
                "observation_type": types.get(node.observation_id),
                "degree": degree,
            }
        )

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "relationship_count": len(relationships),
        "every_relationship_became_an_edge": len(edges) == len(relationships),
        "isolated_node_count": sum(1 for row in rows if row["degree"] == 0),
        "degree_counts": _tally(str(row["degree"]) for row in rows),
        "nodes": rows,
        "edges": [
            {
                "relationship_id": edge.relationship_id,
                "label": _relationship_label(edge, index),
                "evidence_type": edge.evidence_type.value,
                "inference_level": edge.inference_level.value,
                "confidence": edge.confidence.value,
                "confidence_basis": edge.confidence.basis,
                "source_positions": [
                    index.get(oid) for oid in edge.source_observation_ids
                ],
                "target_positions": [
                    index.get(oid) for oid in edge.target_observation_ids
                ],
            }
            for edge in sorted(edges, key=lambda e: e.relationship_id)
        ],
    }


# --------------------------------------------------------------------------- #
# Dimension 4: behavioural features
# --------------------------------------------------------------------------- #


def dimension_behavior(profiles) -> Dict[str, Any]:
    """Per-profile behavioural features, reduced to values worth comparing."""
    rows = []
    for profile in profiles:
        rows.append(
            {
                "profile_id": profile.profile_id,
                "grouping": profile.grouping.value,
                "observation_ids": list(profile.observation_ids),
                "observation_count": profile.observation_count,
                "feature_count": profile.feature_count,
                "categories": sorted(
                    category.value for category in profile.categories()
                ),
                "describes_attribution": profile.describes_attribution(),
                "evidence_relationship_count": len(
                    profile.evidence_relationship_ids
                ),
                "provenance_source_files": list(profile.provenance_source_files),
                "features": [
                    {
                        "name": feature.name,
                        "category": feature.category.value,
                        "kind": feature.value.kind,
                        "status": feature.value.status.value,
                        "value": feature.value.value,
                        "unit": feature.value.unit,
                        "excluded_count": feature.value.excluded_count,
                    }
                    for feature in sorted(
                        profile.features, key=lambda f: f.name
                    )
                ],
            }
        )
    return {
        "profile_count": len(profiles),
        "total_features": sum(row["feature_count"] for row in rows),
        "category_counts": _tally(
            category for row in rows for category in row["categories"]
        ),
        "measurement_status_counts": _tally(
            feature["status"] for row in rows for feature in row["features"]
        ),
        "excluded_measurement_count": sum(
            feature["excluded_count"]
            for row in rows
            for feature in row["features"]
        ),
        "any_profile_claims_attribution": any(
            row["describes_attribution"] for row in rows
        ),
        "profiles": rows,
    }


# --------------------------------------------------------------------------- #
# Dimension 5: ML determinism
# --------------------------------------------------------------------------- #


def dimension_ml(profiles) -> Dict[str, Any]:
    """Anomaly scores and cluster assignment over a known grouping.

    The grouping is stated in the corpus rather than discovered, so a change in
    cluster count means the algorithm changed, not that the data did.

    Profile ids are UUIDv5 over profile content and so are stable, but they are
    replaced by population positions in the output: a stored expected file full
    of UUIDs would be unreadable, and a position is easier to check against the
    corpus by eye.
    """
    positions = {profile.profile_id: index for index, profile in enumerate(profiles)}

    detector = IsolationForestAnomalyDetector()
    model, anomaly = detector.fit_score(profiles)
    clusterer = KMeansBehaviorClusterer()
    clustering_model = clusterer.fit(profiles)
    clustering = clustering_model.result()

    anomaly_rows = [
        {
            "population_index": positions.get(item.profile_id),
            "observation_id": item.observation_ids[0] if item.observation_ids else None,
            "raw_score": item.raw_score,
            "normalized_score": item.normalized_score,
            "band": item.band,
            "is_in_sample": item.is_in_sample,
            "top_contributing_feature": (
                item.explanation.contributions[0].feature_name
                if item.explanation is not None
                and item.explanation.contributions
                else None
            ),
        }
        for item in sorted(
            anomaly.results, key=lambda r: positions.get(r.profile_id, 0)
        )
    ]

    cluster_rows = [
        {
            "population_index": positions.get(item.profile_id),
            "observation_id": (
                profiles[positions[item.profile_id]].observation_ids[0]
                if item.profile_id in positions
                and profiles[positions[item.profile_id]].observation_ids
                else None
            ),
            "cluster_id": item.cluster_id,
            "distance_to_centroid": item.distance_to_centroid,
            "is_noise": item.is_noise,
            "is_in_sample": item.is_in_sample,
        }
        for item in sorted(
            clustering.assignments, key=lambda r: positions.get(r.profile_id, 0)
        )
    ]

    cluster_summaries = [
        {
            "cluster_id": cluster.cluster_id,
            "signature": cluster.signature,
            "member_population_indices": sorted(
                positions[pid] for pid in cluster.member_profile_ids if pid in positions
            ),
            "member_count": len(cluster.member_profile_ids),
            "dispersion": cluster.dispersion,
            "features_above_reference": sorted(
                name for name, value in cluster.centroid if value
            ),
        }
        for cluster in sorted(
            clustering.clusters, key=lambda c: str(c.cluster_id)
        )
    ]

    scores = [row["normalized_score"] for row in anomaly_rows]
    return {
        "population_size": len(profiles),
        "feature_name_count": len(anomaly.feature_names),
        "anomaly": {
            "status": anomaly.status.value,
            "model_id": anomaly.metadata.model_id,
            "algorithm": anomaly.metadata.algorithm,
            "algorithm_version": anomaly.metadata.algorithm_version,
            "library": anomaly.metadata.library,
            "feature_schema_id": anomaly.metadata.feature_schema_id,
            "scaling": anomaly.metadata.scaling.value,
            "imputation": anomaly.metadata.imputation.value,
            "random_state": anomaly.metadata.random_state,
            "fitting_population_size": anomaly.metadata.fitting_population_size,
            "band_boundaries": list(anomaly.band_boundaries),
            "band_counts": _tally(row["band"] for row in anomaly_rows),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "results": anomaly_rows,
        },
        "clustering": {
            "status": clustering.status.value,
            "model_id": clustering.metadata.model_id,
            "algorithm": clustering.metadata.algorithm,
            "algorithm_version": clustering.metadata.algorithm_version,
            "library": clustering.metadata.library,
            "feature_schema_id": clustering.metadata.feature_schema_id,
            "random_state": clustering.metadata.random_state,
            "fitting_population_size": clustering.metadata.fitting_population_size,
            "k": clustering.k,
            "cluster_count": len(clustering.clusters),
            "cluster_ids": sorted(str(c.cluster_id) for c in clustering.clusters),
            "noise_profile_count": len(clustering.noise_profile_ids),
            "selection_scores": [list(score) for score in clustering.selection_scores],
            "clusters": cluster_summaries,
            "assignments": cluster_rows,
        },
        "declared_grouping": {
            "note": (
                "the corpus states two groups; this records whether the "
                "algorithm agreed, without claiming the groups are correct"
            ),
            "expected_cluster_count": 2,
            "expected_group_by_population_index": "0-7 low burst, 8-11 high burst",
        },
    }


# --------------------------------------------------------------------------- #
# Dimension 6: path determinism and temporal comparison
# --------------------------------------------------------------------------- #


def dimension_paths(searches, positions, temporal) -> Dict[str, Any]:
    """Named path searches, in the order each search decided to return them.

    Two searches are reported because they answer different questions. The
    shortest-hop search gives the minimal recorded chain to a target the graph
    does not join directly; the bounded enumeration shows the depth actually
    present in the evidence. Reporting only the first would hide a multi-hop
    graph behind a one-hop answer.
    """
    reported = {}
    for name, search in searches.items():
        reported[name] = {
            "search": {
                "status": search.status.value,
                "detail": search.detail,
                "path_count": search.path_count,
                "shortest_hops": search.shortest_hops,
                "temporal_exclusions": search.temporal_exclusions,
                "incomparable_exclusions": search.incomparable_exclusions,
                "search_scope": search.search_scope,
                "request": search.request.model_dump(mode="json"),
            },
            "paths": [_path_row(path, positions) for path in search.paths],
        }

    all_hops = [
        row["hop_count"]
        for entry in reported.values()
        for row in entry["paths"]
    ]
    return {
        "searches": reported,
        "max_hop_count": max(all_hops, default=0),
        "total_paths_found": sum(
            entry["search"]["path_count"] for entry in reported.values()
        ),
        "temporal_comparison": temporal,
    }


def _path_row(path, positions) -> Dict[str, Any]:
    return {
        "path_id": path.path_id,
        "hop_count": path.hop_count,
        # ``path.nodes`` holds PathNode records, not bare ids; the id has to be
        # unwrapped before it can be looked up in the position index.
        "node_positions": [
            positions.get(node.observation_id, node.observation_id)
            for node in path.nodes
        ],
        "node_timestamp_awareness": [
            node.timestamp_awareness.value for node in path.nodes
        ],
        "temporal_status": path.temporal_status,
        "first_timestamp": path.first_timestamp,
        "last_timestamp": path.last_timestamp,
        "duration_seconds": path.duration_seconds,
        "transitions": [
            {
                "from_position": positions.get(
                    transition.from_observation_id,
                    transition.from_observation_id,
                ),
                "to_position": positions.get(
                    transition.to_observation_id,
                    transition.to_observation_id,
                ),
                "relationship_id": transition.relationship_id,
                "evidence_type": transition.evidence_type.value,
                "inference_level": transition.inference_level.value,
                "confidence": transition.confidence.value,
                "orientation": transition.orientation.value,
                "rule_id": transition.rule_id,
                "rule_version": transition.rule_version,
                "temporal": (
                    transition.temporal.model_dump(mode="json")
                    if transition.temporal is not None
                    else None
                ),
            }
            for transition in path.transitions
        ],
        "evidence_summary": path.evidence_summary.model_dump(mode="json"),
    }


# --------------------------------------------------------------------------- #
# Dimension 7: Evidence-Supported Investigative Priority
# --------------------------------------------------------------------------- #


def dimension_fusion(assessment) -> Dict[str, Any]:
    """The fused priority, every signal that fed it, and every one that did not.

    Missing signals are recorded as carefully as present ones. A priority that
    looks high because a stage returned nothing is the failure this dimension
    exists to make visible, so ``availability`` is reported per family: a
    family whose signals were all unavailable contributes zero and says so.
    """
    signals = [
        {
            "signal_id": signal.signal_id,
            "signal_type": signal.signal_type,
            "family": signal.family,
            "role": signal.role.value,
            "status": signal.status.value,
            "source_step": signal.source_step,
            "raw_value": signal.raw_value,
            "normalized_value": signal.normalized_value,
            "evidence_ref_count": len(signal.evidence_refs),
            "inference_levels": sorted(
                level.value for level in signal.inference_levels
            ),
            "provenance_ref_count": len(signal.provenance_refs),
            "explanation": signal.explanation,
        }
        for signal in sorted(
            assessment.signals, key=lambda s: (s.signal_type, str(s.signal_id))
        )
    ]
    missing = [
        {
            "signal_id": item.signal_id,
            "signal_type": item.signal_type,
            "family": item.family,
            "status": item.status.value,
            "source_step": item.source_step,
            "explanation": item.explanation,
        }
        for item in sorted(
            assessment.missing_signals, key=lambda m: str(m.signal_type)
        )
    ]

    return {
        "assessment_id": assessment.assessment_id,
        "status": assessment.status.value,
        "config_version": assessment.config_version,
        "fused_score": assessment.fused_score,
        "risk_band": assessment.risk_band.value,
        "signal_count": assessment.signal_count,
        "present_signal_count": assessment.present_signal_count,
        "interpretation": assessment.interpretation,
        "signals": signals,
        "signal_status_counts": _tally(signal["status"] for signal in signals),
        "signal_source_steps": _tally(
            str(signal["source_step"]) for signal in signals
        ),
        "contributions": [
            {
                "family": contribution.family,
                "weight": contribution.weight,
                "normalized_value": contribution.normalized_value,
                "availability": contribution.availability,
                "contribution": contribution.contribution,
                "present_count": contribution.present_count,
                "total_count": contribution.total_count,
                "explanation": contribution.explanation,
            }
            for contribution in sorted(
                assessment.contributions, key=lambda c: str(c.family)
            )
        ],
        "missing_signals": missing,
        "families_with_no_available_signal": sorted(
            str(c.family)
            for c in assessment.contributions
            if c.present_count == 0
        ),
        "inference_levels": sorted(
            level.value for level in assessment.inference_levels
        ),
        "distinct_relationship_ids": sorted(assessment.distinct_relationship_ids),
        "edge_confidence_context": assessment.confidence.model_dump(mode="json"),
        "explanation": [
            {
                "kind": item.kind.value,
                "text": item.text,
                "evidence_ref_count": len(item.evidence),
            }
            for item in assessment.explanation
        ],
        "explanation_line_count": len(assessment.explanation),
    }


# --------------------------------------------------------------------------- #
# Dimension 8: dashboard payload
# --------------------------------------------------------------------------- #


def dimension_dashboard(data) -> Dict[str, Any]:
    """The dashboard payload as the Step 9A adapter builds it.

    Ordering here is the frontend contract, so it is recorded rather than
    re-sorted: sorted ids for lists the adapter sorts, and the adapter's own
    order for anything it sequences deliberately.
    """
    payload = data.model_dump(mode="json")
    return {
        "version": payload.get("version"),
        "generated_at": payload.get("generated_at"),
        "overview": payload.get("overview"),
        "signal_count": len(payload.get("signals", ())),
        "signal_status_counts": _tally(
            signal.get("status") for signal in payload.get("signals", ())
        ),
        "signal_ids_in_order": [
            signal.get("signal_id") for signal in payload.get("signals", ())
        ],
        "evidence_count": len(payload.get("evidence", ())),
        "evidence_relationship_ids": [
            item.get("relationship_id") for item in payload.get("evidence", ())
        ],
        "path_count": len(payload.get("paths", ())),
        "path_ids_in_order": [
            item.get("path_id") for item in payload.get("paths", ())
        ],
        "behavior": payload.get("behavior"),
        "provenance": payload.get("provenance"),
        "overview_keys": sorted(payload.get("overview", {}).keys()),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_all(shuffle: Optional[Sequence[int]] = None) -> EvaluationOutcome:
    """Run every dimension once and return the canonical result.

    *shuffle* reorders the observation list before the pipeline runs, so a
    caller can check that nothing downstream depends on the order records
    arrive in. Passing ``None`` is the reference run.

    Corpus positions are taken from the unshuffled order, so a shuffled run
    labels the same record with the same position rather than renumbering it.

    Two relationship sets are built, and they are kept apart on purpose.
    ``rule_relationships`` is whatever Step 4B derives from the corpus alone,
    and is what the correlation and graph dimensions report. ``levelled`` adds
    relationships an analyst would characterise at the other three inference
    levels, which no rule can produce. Path search, fusion and the dashboard
    run over the union, because carrying every level through those stages is
    one of the sixteen cases and cannot be demonstrated on CORRELATED alone.
    """
    observations, summary = corpus.load_observations()
    corpus_order = corpus.stable_order(observations)
    positions = {o.observation_id: index for index, o in enumerate(corpus_order)}

    if shuffle is None:
        ordered = list(corpus_order)
    else:
        ordered = [corpus_order[i] for i in shuffle]

    comparable = [
        o for o in corpus_order
        if o.timestamp is not None and o.timestamp.tzinfo is not None
    ]
    subject = comparable[corpus.SUBJECT_POSITION]

    rule_relationships = corpus.run_correlation(ordered)
    rule_graph = corpus.build_graph_from_relationships(rule_relationships)

    levelled = corpus.levelled_relationships(ordered)
    combined = list(rule_relationships) + list(levelled)
    combined_graph = corpus.build_graph_from_relationships(combined)

    temporal = corpus.temporal_probes(corpus_order)

    builder = BehaviorProfileBuilder()
    profile = builder.build([subject])
    population = corpus.population_profiles()

    start, target = corpus.choose_path_endpoints(combined_graph, corpus_order, subject)
    searches = {
        "shortest": corpus.search_paths(
            combined_graph, ordered, start=start, target=target
        ),
        "bounded_enumeration": corpus.search_paths(
            combined_graph,
            ordered,
            start=start,
            target=target,
            exhaustive=True,
        ),
    }

    context = corpus.build_fusion_context(
        ordered,
        combined,
        search_override=searches["shortest"],
        subject_override=subject,
    )
    assessment = FusionEngine().assess(context)
    dashboard = build_dashboard_data(assessment, context)

    dimensions: Dict[str, Any] = {
        "ingestion_normalization": dimension_ingestion(corpus_order, summary),
        "evidence_correlation": dimension_correlation(rule_relationships, positions),
        "evidence_graph": dimension_graph(
            rule_graph, rule_relationships, corpus_order, positions
        ),
        "behavioral_features": dimension_behavior([profile]),
        "ml_determinism": dimension_ml(population),
        "path_determinism": dimension_paths(searches, positions, temporal),
        "fusion_priority": dimension_fusion(assessment),
        "dashboard_payload": dimension_dashboard(dashboard),
    }

    invariants = _invariants(dimensions, combined_graph, population, searches)
    shuffled = _shuffle_checks(dimensions, corpus_order)

    canonical_dimensions = canonical(dimensions)
    return EvaluationOutcome(
        dimensions=canonical_dimensions,
        case_coverage=CASE_COVERAGE,
        shuffle_checks=shuffled,
        invariants=invariants,
        dimensions_digest=digest(canonical_dimensions),
        run_scoped_paths=tuple(find_run_scoped(canonical_dimensions)),
    )


def run_dimension(name: str) -> Dict[str, Any]:
    """Run one dimension by name, for a narrower investigation."""
    outcome = run_all()
    if name not in DIMENSIONS:
        raise KeyError(
            "unknown dimension %r; expected one of %s" % (name, ", ".join(DIMENSIONS))
        )
    return outcome.dimensions[name]


def _shuffle_checks(dimensions, corpus_order) -> Dict[str, Any]:
    """Re-run the order-sensitive stages on reordered input.

    Only stages that consume a list are shuffled. Behavioural features and the
    ML population are built from a stated population rather than from the
    corpus order, so reordering them would change nothing worth measuring;
    correlation, graph construction, path search and fusion all take a list,
    and a list's arrival order is not part of their contract.
    """
    count = len(corpus_order)
    orders = {
        "reversed": list(reversed(range(count))),
        "rotated_by_five": [(i + 5) % count for i in range(count)],
    }

    watched = (
        "evidence_correlation",
        "evidence_graph",
        "path_determinism",
        "fusion_priority",
    )

    reference = {name: digest(dimensions[name]) for name in watched}

    results: Dict[str, Any] = {}
    for label, order in orders.items():
        produced = _pipeline_digests(corpus_order, order)
        identical = {
            name: produced[name] == reference[name] for name in watched
        }
        results[label] = {
            "order": order,
            "identical": identical,
            "order_independent": all(identical.values()),
            "mismatched_dimensions": sorted(
                name for name, same in identical.items() if not same
            ),
        }
    return {
        "watched_dimensions": list(watched),
        "reference_digests": reference,
        "orders": results,
        "all_orders_independent": all(
            entry["order_independent"] for entry in results.values()
        ),
    }


def _watched_digest(payload: Dict[str, Any], name: str) -> str:
    """Digest one dimension.

    Nothing is excluded here. Every field in these four dimensions is either
    content-derived or a function of content-derived values, so a difference
    here is a real difference in behaviour rather than a run-scoped artefact.
    """
    del name
    return digest(payload)


def _pipeline_digests(corpus_order, order) -> Dict[str, str]:
    """Digests of the order-sensitive dimensions for one input order."""
    observations = [corpus_order[i] for i in order]
    positions = {
        observation.observation_id: index
        for index, observation in enumerate(corpus_order)
    }

    comparable = [
        observation
        for observation in corpus_order
        if observation.timestamp is not None
        and observation.timestamp.tzinfo is not None
    ]
    subject = comparable[corpus.SUBJECT_POSITION]

    rule_relationships = corpus.run_correlation(observations)
    rule_graph = corpus.build_graph_from_relationships(rule_relationships)
    combined = list(rule_relationships) + list(
        corpus.levelled_relationships(observations)
    )
    combined_graph = corpus.build_graph_from_relationships(combined)

    start, target = corpus.choose_path_endpoints(combined_graph, corpus_order, subject)
    searches = {
        "shortest": corpus.search_paths(
            combined_graph, observations, start=start, target=target
        ),
        "bounded_enumeration": corpus.search_paths(
            combined_graph,
            observations,
            start=start,
            target=target,
            exhaustive=True,
        ),
    }
    context = corpus.build_fusion_context(
        observations,
        combined,
        search_override=searches["shortest"],
        subject_override=subject,
    )
    assessment = FusionEngine().assess(context)

    # Temporal probes are built from the canonical order in both runs: they are
    # a property of the corpus, not of the order it was shuffled into.
    temporal = corpus.temporal_probes(corpus_order)

    return {
        "evidence_correlation": digest(
            dimension_correlation(rule_relationships, positions)
        ),
        "evidence_graph": digest(
            dimension_graph(rule_graph, rule_relationships, observations, positions)
        ),
        "path_determinism": digest(
            dimension_paths(searches, positions, temporal)
        ),
        "fusion_priority": digest(dimension_fusion(assessment)),
    }


def _invariants(dimensions, combined_graph, population, searches) -> Dict[str, Any]:
    """Properties that must hold for the evaluation to mean anything.

    Each one is a statement about the code, checked rather than assumed. A
    vacuous dimension would pass its own expected-output comparison while
    proving nothing; these assertions are what stop that.
    """
    correlation = dimensions["evidence_correlation"]
    graph_payload = dimensions["evidence_graph"]
    behavior = dimensions["behavioral_features"]
    ml = dimensions["ml_determinism"]
    paths = dimensions["path_determinism"]
    fusion = dimensions["fusion_priority"]
    dashboard = dimensions["dashboard_payload"]
    ingestion = dimensions["ingestion_normalization"]

    # Certainty is a statement about the match, not about the conclusion. The
    # shipped rules allow an exact deterministic match to carry confidence 1.0
    # while still recording the relationship as CORRELATED: the values are
    # certainly equal, and what that equality implies is still inferred. So the
    # rule is basis-aware, not a blanket ceiling:
    #
    #   * only an EXACT_MATCH basis may carry 1.0;
    #   * an inference or an unverified claim is never an exact match.
    #
    # A blanket "confidence below 1.0" check would have passed this by
    # forbidding the shipped rules' correct behaviour.
    certainty_without_exact_match = sorted(
        {
            (edge.rule_id, edge.inference_level.value, edge.confidence.basis)
            for edge in combined_graph.edges()
            if float(edge.confidence.value) >= 1.0
            and edge.confidence.basis != "EXACT_MATCH"
        }
    )
    inference_claiming_exact_match = sorted(
        {
            (edge.rule_id, edge.inference_level.value, edge.confidence.basis)
            for edge in combined_graph.edges()
            if edge.inference_level.value
            in ("INFERRED", "REQUIRES_VERIFICATION")
            and edge.confidence.basis == "EXACT_MATCH"
        }
    )
    confidence_by_basis = {}
    for edge in combined_graph.edges():
        confidence_by_basis.setdefault(
            edge.confidence.basis, set()
        ).add(edge.confidence.value)
    confidence_by_basis = {
        basis: sorted(values) for basis, values in sorted(confidence_by_basis.items())
    }

    shortest = paths["searches"]["shortest"]
    enumeration = paths["searches"]["bounded_enumeration"]
    pairwise = paths["temporal_comparison"]["pairwise"]

    # Evidence for the temporal and attribution invariants, so each verdict
    # arrives with the numbers behind it rather than as a bare claim.
    elapsed_cases = [
        {
            "case": case["case"],
            "relation": case["relation"],
            "comparable": case["comparable"],
            "elapsed_seconds": case["elapsed_seconds"],
        }
        for case in pairwise
        if case["elapsed_seconds"] is not None
    ]
    attribution_by_profile = {
        row["profile_id"]: row["describes_attribution"]
        for row in behavior["profiles"]
    }

    return {
        "corpus_is_non_trivial": {
            "holds": ingestion["summary"]["total_observations"] >= 12,
            "observation_count": ingestion["summary"]["total_observations"],
            "why": "a corpus too small to correlate would make every other check vacuous",
        },
        "corpus_exercises_every_format": {
            "holds": sorted(ingestion["summary"]["formats"])
            == ["csv", "json", "xml"],
            "formats": ingestion["summary"]["formats"],
            "why": "Step 2 supports three parsers and one would otherwise go untested",
        },
        "corpus_records_its_own_problems": {
            "holds": ingestion["summary"]["observations_with_warnings"] > 0
            and ingestion["summary"]["observations_without_timestamp"] > 0,
            "observations_with_warnings": ingestion["summary"][
                "observations_with_warnings"
            ],
            "observations_without_timestamp": ingestion["summary"][
                "observations_without_timestamp"
            ],
            "why": "a corpus where nothing is malformed cannot show that problems are reported",
        },
        "correlation_finds_relationships": {
            "holds": correlation["relationship_count"] > 0,
            "relationship_count": correlation["relationship_count"],
            "why": "a zero-relationship run cannot exercise graph, path or fusion",
        },
        "correlation_covers_many_evidence_types": {
            "holds": len(correlation["evidence_types_present"]) >= 4,
            "evidence_types_present": correlation["evidence_types_present"],
            "why": "one evidence type would not test the rule set",
        },
        "graph_preserves_every_relationship": {
            "holds": graph_payload["every_relationship_became_an_edge"]
            and graph_payload["edge_count"] == graph_payload["relationship_count"],
            "edge_count": graph_payload["edge_count"],
            "relationship_count": graph_payload["relationship_count"],
            "why": "losing an edge would silently drop evidence",
        },
        "graph_has_no_orphan_nodes": {
            "holds": graph_payload["isolated_node_count"] == 0,
            "isolated_node_count": graph_payload["isolated_node_count"],
            "why": "a node with no edge carries no evidence",
        },
        "behavioural_features_are_real": {
            "holds": behavior["profile_count"] > 0
            and behavior["total_features"] > 0
            and len(behavior["profiles"][0]["features"]) > 10,
            "feature_count": behavior["total_features"],
            "why": "a handful of features would not exercise the Step 5 builder",
        },
        "behavioural_profile_claims_no_attribution": {
            "holds": not behavior["any_profile_claims_attribution"],
            "attribution_flag_by_profile": attribution_by_profile,
            "why": "Step 5 measures behaviour and must not name an owner",
        },
        "ml_produced_scores_and_clusters": {
            "holds": ml["anomaly"]["score_max"] is not None
            and ml["clustering"]["cluster_count"] >= 2
            and len(ml["clustering"]["assignments"]) == len(population),
            "cluster_count": ml["clustering"]["cluster_count"],
            "band_counts": ml["anomaly"]["band_counts"],
            "why": "determinism over a constant would prove nothing",
        },
        "ml_recovered_the_declared_grouping": {
            "holds": ml["clustering"]["cluster_count"]
            == ml["declared_grouping"]["expected_cluster_count"],
            "cluster_count": ml["clustering"]["cluster_count"],
            "expected": ml["declared_grouping"]["expected_cluster_count"],
            "why": (
                "the corpus states two groups, so a third cluster means the "
                "algorithm drifted; this is a consistency check, not a claim "
                "that two clusters is correct"
            ),
        },
        "path_search_found_a_multi_hop_path": {
            "holds": paths["max_hop_count"] > 1,
            "max_hop_count": paths["max_hop_count"],
            "shortest_hops": shortest["search"]["shortest_hops"],
            "enumerated_paths": enumeration["search"]["path_count"],
            "why": "the multi-hop case is one of the sixteen",
        },
        "path_search_reaches_its_target": {
            "holds": shortest["search"]["status"] == "FOUND"
            and shortest["search"]["path_count"] > 0,
            "status": shortest["search"]["status"],
            "why": "a search that found nothing would pass a multi-hop check vacuously",
        },
        "comparable_temporal_data_yields_elapsed_time": {
            "holds": any(
                case["case"] == "comparable_temporal_data"
                and case["comparable"]
                and case["elapsed_seconds"] is not None
                for case in pairwise
            ),
            "comparisons_with_elapsed_time": elapsed_cases,
            "why": "refusing to compare two aware timestamps would be over-caution",
        },
        "incomparable_temporal_data_yields_no_elapsed_time": {
            "holds": any(
                case["case"] == "not_comparable_temporal_data"
                and not case["comparable"]
                and case["elapsed_seconds"] is None
                for case in pairwise
            )
            and any(
                case["case"] == "missing_temporal_data"
                and not case["comparable"]
                and case["elapsed_seconds"] is None
                for case in pairwise
            ),
            "pairs": pairwise,
            "why": "inventing elapsed time across incomparable timestamps is the failure mode",
        },
        "fusion_records_both_present_and_missing_signals": {
            "holds": fusion["present_signal_count"] > 0
            and len(fusion["missing_signals"]) > 0
            and fusion["signal_count"] == fusion["present_signal_count"]
            + len(fusion["missing_signals"]),
            "present_signal_count": fusion["present_signal_count"],
            "missing_signal_count": len(fusion["missing_signals"]),
            "why": "a priority that ignores unavailable evidence is not evidence-supported",
        },
        "fusion_carries_all_four_inference_levels": {
            "holds": len(fusion["inference_levels"]) == 4,
            "inference_levels": fusion["inference_levels"],
            "why": "carrying every level through fusion is one of the sixteen cases",
        },
        "fusion_explains_itself": {
            "holds": len(fusion["explanation"]) > 0
            and len(fusion["distinct_relationship_ids"]) > 0
            and all(item["text"] for item in fusion["explanation"]),
            "explanation_line_count": fusion["explanation_line_count"],
            "distinct_relationship_count": len(fusion["distinct_relationship_ids"]),
            "why": "the priority must cite recorded evidence to be worth reading",
        },
        "inference_does_not_claim_certainty": {
            "holds": not certainty_without_exact_match
            and not inference_claiming_exact_match,
            "certainty_without_exact_match_basis": certainty_without_exact_match,
            "inference_claiming_exact_match": inference_claiming_exact_match,
            "confidence_values_by_basis": confidence_by_basis,
            "why": (
                "certainty is reserved for an exact deterministic match; an "
                "inference claiming it, or a rule-derived judgement carrying "
                "it, would present a judgement as a fact"
            ),
        },
        "dashboard_matches_the_fusion_result": {
            "holds": dashboard["signal_count"] == fusion["signal_count"],
            "dashboard_signal_count": dashboard["signal_count"],
            "fusion_signal_count": fusion["signal_count"],
            "why": "the dashboard must not restate a different number",
        },
        "dashboard_preserves_dashboard_ordering": {
            "holds": dashboard["signal_ids_in_order"]
            == sorted(dashboard["signal_ids_in_order"])
            and dashboard["path_ids_in_order"]
            == sorted(dashboard["path_ids_in_order"]),
            "signal_ids_in_order": dashboard["signal_ids_in_order"],
            "path_ids_in_order": dashboard["path_ids_in_order"],
            "why": (
                "Step 9A guarantees sorted ids to the frontend; a report that "
                "hid the order would hide that guarantee"
            ),
        },
    }


def compare_to_expected(
    actual: EvaluationOutcome,
    expected_path: Path,
) -> Tuple[bool, List[str]]:
    """Compare a run against a stored expected output.

    The comparison is against the *compact projection*, not the full payload.
    The full payload carries every feature value of every profile and every
    signal explanation, which is right for a report and wrong for a fixture: a
    128 KB file nobody reads cannot be reviewed, so a wrong value in it goes
    unnoticed. The compact view keeps identifiers, types, counts, levels, order
    and statuses, and drops per-item measurements.

    Returns ``(matched, differences)``. Differences are reported as JSON paths
    so a reader knows *where* two runs parted company, not merely that they did.
    """
    import json

    if not expected_path.exists():
        return False, ["expected output file does not exist: %s" % expected_path]
    with open(expected_path, "r", encoding="utf-8") as stream:
        stored = json.load(stream)
    return diff_canonical(stored, compact_view(actual.dimensions), "$")


def compact_view(dimensions: Dict[str, Any]) -> Dict[str, Any]:
    """The reviewable projection of a full evaluation payload.

    What is kept: identifiers, types, counts, levels, ordering and statuses.
    What is dropped: individual measurements - per-feature values, per-signal
    raw numbers, per-edge confidences. Those are in ``reports/latest.json``.

    The split is not only about size. A stored expectation is only a useful
    guard if a person can read it end to end and see whether it is right. A
    fixture nobody can read is a fixture nobody maintains.
    """
    return {
        "ingestion_normalization": {
            "summary": dimensions["ingestion_normalization"]["summary"],
            "files": [
                {
                    "file": entry["file"],
                    "detected_format": entry["detected_format"],
                    "raw_record_count": entry["raw_record_count"],
                    "record_count": entry["record_count"],
                    "duplicate_count": entry["duplicate_count"],
                    "error_count": entry["error_count"],
                    "warning_count": entry["warning_count"],
                    "observation_types": entry["observation_types"],
                    "error_codes": entry["error_codes"],
                    "warning_codes": entry["warning_codes"],
                    "source_bytes_unchanged": entry["source_bytes_unchanged"],
                }
                for entry in dimensions["ingestion_normalization"]["files"]
            ],
            "observation_shape": [
                {
                    "corpus_index": row["corpus_index"],
                    "observation_type": row["observation_type"],
                    "source_format": row["source_format"],
                    "error_codes": row["error_codes"],
                    "warning_codes": row["warning_codes"],
                    "has_txid": row["txid"] is not None,
                    "has_src_ip": row["src_ip"] is not None,
                }
                for row in dimensions["ingestion_normalization"]["observations"]
            ],
            "awareness_counts": dimensions["ingestion_normalization"][
                "temporal_classification"
            ]["awareness_counts"],
        },
        "evidence_correlation": {
            "relationship_count": dimensions["evidence_correlation"][
                "relationship_count"
            ],
            "evidence_types_present": dimensions["evidence_correlation"][
                "evidence_types_present"
            ],
            "evidence_type_counts": dimensions["evidence_correlation"][
                "evidence_type_counts"
            ],
            "inference_levels_present": dimensions["evidence_correlation"][
                "inference_levels_present"
            ],
            "inference_level_counts": dimensions["evidence_correlation"][
                "inference_level_counts"
            ],
            "rules_that_fired": dimensions["evidence_correlation"]["rules_that_fired"],
            "relationship_labels": [
                row["label"] for row in dimensions["evidence_correlation"]["relationships"]
            ],
        },
        "evidence_graph": {
            "node_count": dimensions["evidence_graph"]["node_count"],
            "edge_count": dimensions["evidence_graph"]["edge_count"],
            "every_relationship_became_an_edge": dimensions["evidence_graph"][
                "every_relationship_became_an_edge"
            ],
            "isolated_node_count": dimensions["evidence_graph"]["isolated_node_count"],
            "degree_counts": dimensions["evidence_graph"]["degree_counts"],
            "nodes": [
                {
                    "position": row["position"],
                    "observation_type": row["observation_type"],
                    "degree": row["degree"],
                }
                for row in dimensions["evidence_graph"]["nodes"]
            ],
            "edge_labels": [
                row["label"] for row in dimensions["evidence_graph"]["edges"]
            ],
        },
        "behavioral_features": {
            "profile_count": dimensions["behavioral_features"]["profile_count"],
            "total_features": dimensions["behavioral_features"]["total_features"],
            "category_counts": dimensions["behavioral_features"]["category_counts"],
            "measurement_status_counts": dimensions["behavioral_features"][
                "measurement_status_counts"
            ],
            "excluded_measurement_count": dimensions["behavioral_features"][
                "excluded_measurement_count"
            ],
            "any_profile_claims_attribution": dimensions["behavioral_features"][
                "any_profile_claims_attribution"
            ],
            "feature_names_by_category": _features_by_category(dimensions),
        },
        "ml_determinism": {
            "population_size": dimensions["ml_determinism"]["population_size"],
            "feature_name_count": dimensions["ml_determinism"]["feature_name_count"],
            "anomaly": {
                "status": dimensions["ml_determinism"]["anomaly"]["status"],
                "algorithm": dimensions["ml_determinism"]["anomaly"]["algorithm"],
                "algorithm_version": dimensions["ml_determinism"]["anomaly"][
                    "algorithm_version"
                ],
                "feature_schema_id": dimensions["ml_determinism"]["anomaly"][
                    "feature_schema_id"
                ],
                "scaling": dimensions["ml_determinism"]["anomaly"]["scaling"],
                "imputation": dimensions["ml_determinism"]["anomaly"]["imputation"],
                "random_state": dimensions["ml_determinism"]["anomaly"]["random_state"],
                "band_boundaries": dimensions["ml_determinism"]["anomaly"][
                    "band_boundaries"
                ],
                "band_counts": dimensions["ml_determinism"]["anomaly"]["band_counts"],
                "bands_by_population_index": [
                    {"population_index": row["population_index"], "band": row["band"]}
                    for row in dimensions["ml_determinism"]["anomaly"]["results"]
                ],
            },
            "clustering": {
                "status": dimensions["ml_determinism"]["clustering"]["status"],
                "algorithm": dimensions["ml_determinism"]["clustering"]["algorithm"],
                "algorithm_version": dimensions["ml_determinism"]["clustering"][
                    "algorithm_version"
                ],
                "feature_schema_id": dimensions["ml_determinism"]["clustering"][
                    "feature_schema_id"
                ],
                "random_state": dimensions["ml_determinism"]["clustering"]["random_state"],
                "k": dimensions["ml_determinism"]["clustering"]["k"],
                "cluster_count": dimensions["ml_determinism"]["clustering"][
                    "cluster_count"
                ],
                "cluster_ids": dimensions["ml_determinism"]["clustering"]["cluster_ids"],
                "noise_profile_count": dimensions["ml_determinism"]["clustering"][
                    "noise_profile_count"
                ],
                "clusters": [
                    {
                        "cluster_id": cluster["cluster_id"],
                        "signature": cluster["signature"],
                        "member_population_indices": cluster["member_population_indices"],
                        "features_above_reference": cluster[
                            "features_above_reference"
                        ],
                    }
                    for cluster in dimensions["ml_determinism"]["clustering"]["clusters"]
                ],
                "cluster_by_population_index": [
                    {
                        "population_index": row["population_index"],
                        "cluster_id": row["cluster_id"],
                        "is_noise": row["is_noise"],
                    }
                    for row in dimensions["ml_determinism"]["clustering"]["assignments"]
                ],
            },
        },
        "path_determinism": {
            "max_hop_count": dimensions["path_determinism"]["max_hop_count"],
            "total_paths_found": dimensions["path_determinism"]["total_paths_found"],
            "searches": {
                name: {
                    "status": entry["search"]["status"],
                    "path_count": entry["search"]["path_count"],
                    "shortest_hops": entry["search"]["shortest_hops"],
                    "temporal_exclusions": entry["search"]["temporal_exclusions"],
                    "incomparable_exclusions": entry["search"][
                        "incomparable_exclusions"
                    ],
                    "paths": [
                        _compact_path(row) for row in entry["paths"]
                    ],
                }
                for name, entry in dimensions["path_determinism"]["searches"].items()
            },
            "temporal_comparison": dimensions["path_determinism"][
                "temporal_comparison"
            ]["pairwise"],
        },
        "fusion_priority": {
            "status": dimensions["fusion_priority"]["status"],
            "config_version": dimensions["fusion_priority"]["config_version"],
            "fused_score": dimensions["fusion_priority"]["fused_score"],
            "risk_band": dimensions["fusion_priority"]["risk_band"],
            "signal_count": dimensions["fusion_priority"]["signal_count"],
            "present_signal_count": dimensions["fusion_priority"][
                "present_signal_count"
            ],
            "interpretation": dimensions["fusion_priority"]["interpretation"],
            "signal_status_counts": dimensions["fusion_priority"][
                "signal_status_counts"
            ],
            "signal_source_steps": dimensions["fusion_priority"]["signal_source_steps"],
            "signals": [
                {
                    "signal_type": signal["signal_type"],
                    "family": signal["family"],
                    "role": signal["role"],
                    "status": signal["status"],
                    "source_step": signal["source_step"],
                    "inference_levels": signal["inference_levels"],
                    "evidence_ref_count": signal["evidence_ref_count"],
                }
                for signal in dimensions["fusion_priority"]["signals"]
            ],
            "contributions": [
                {
                    "family": contribution["family"],
                    "weight": contribution["weight"],
                    "availability": contribution["availability"],
                    "present_count": contribution["present_count"],
                    "total_count": contribution["total_count"],
                }
                for contribution in dimensions["fusion_priority"]["contributions"]
            ],
            "missing_signals": [
                {
                    "signal_type": item["signal_type"],
                    "family": item["family"],
                    "status": item["status"],
                    "source_step": item["source_step"],
                }
                for item in dimensions["fusion_priority"]["missing_signals"]
            ],
            "families_with_no_available_signal": dimensions["fusion_priority"][
                "families_with_no_available_signal"
            ],
            "inference_levels": dimensions["fusion_priority"]["inference_levels"],
            "distinct_relationship_count": len(
                dimensions["fusion_priority"]["distinct_relationship_ids"]
            ),
            "explanation": [
                {"kind": item["kind"], "text": item["text"]}
                for item in dimensions["fusion_priority"]["explanation"]
            ],
        },
        "dashboard_payload": {
            "version": dimensions["dashboard_payload"]["version"],
            "generated_at": dimensions["dashboard_payload"]["generated_at"],
            "signal_count": dimensions["dashboard_payload"]["signal_count"],
            "signal_status_counts": dimensions["dashboard_payload"][
                "signal_status_counts"
            ],
            "signal_ids_in_order": dimensions["dashboard_payload"][
                "signal_ids_in_order"
            ],
            "evidence_count": dimensions["dashboard_payload"]["evidence_count"],
            "evidence_relationship_ids": dimensions["dashboard_payload"][
                "evidence_relationship_ids"
            ],
            "path_count": dimensions["dashboard_payload"]["path_count"],
            "path_ids_in_order": dimensions["dashboard_payload"]["path_ids_in_order"],
            "overview_keys": dimensions["dashboard_payload"]["overview_keys"],
            "behavior_keys": sorted(dimensions["dashboard_payload"]["behavior"]),
            "provenance_keys": sorted(dimensions["dashboard_payload"]["provenance"]),
        },
    }


def _compact_path(row: Dict[str, Any]) -> Dict[str, Any]:
    """One path as a short, readable line.

    Each transition becomes ``from->to EVIDENCE/LEVEL/ORIENTATION RELATION
    comparable``. The full payload carries the relationship ids, rationales and
    provenance; none of those belong in a fixture, and dropping them is what
    makes 10 enumerated paths readable on one screen.
    """
    return {
        "hop_count": row["hop_count"],
        "nodes": "->".join(str(position) for position in row["node_positions"]),
        "temporal_status": row["temporal_status"],
        "duration_seconds": row["duration_seconds"],
        "transitions": [
            "%s->%s %s/%s/%s %s comparable=%s"
            % (
                transition["from_position"],
                transition["to_position"],
                transition["evidence_type"],
                transition["inference_level"],
                transition["orientation"],
                (transition["temporal"] or {}).get("relation"),
                (transition["temporal"] or {}).get("comparable"),
            )
            for transition in row["transitions"]
        ],
    }


def _features_by_category(dimensions: Dict[str, Any]) -> Dict[str, List[str]]:
    """Feature names grouped by category.

    The names are the contract: Step 5 emits the same set every time, so a
    missing or renamed feature is a real change and belongs in the fixture.

    A feature's kind is recorded only when it is not the plain observed kind,
    and its status only when it is not ``OBSERVED``. Both are recorded because
    they mark a measurement that was estimated or excluded, which is exactly
    the kind of thing a reviewer should be shown. Repeating them on all 73
    features would bury that.
    """
    grouped: Dict[str, Dict[str, str]] = {}
    for profile in dimensions["behavioral_features"]["profiles"]:
        for feature in profile["features"]:
            label = feature["name"]
            if feature["status"] != "OBSERVED":
                label += " [%s]" % feature["status"]
            if feature["kind"] != "OBSERVED":
                label += " (%s)" % feature["kind"]
            grouped.setdefault(feature["category"], {})[label] = ""
    return {
        category: sorted(names) for category, names in sorted(grouped.items())
    }


def diff_canonical(expected, actual, prefix: str = "") -> Tuple[bool, List[str]]:
    """Recursive structural comparison producing readable difference paths."""
    differences: List[str] = []
    _diff(expected, actual, prefix or "$", differences)
    return not differences, differences


def _diff(expected, actual, path: str, out: List[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = "%s.%s" % (path, key)
            if key not in expected:
                out.append("%s: unexpected in run (%s)" % (child, _short(actual[key])))
            elif key not in actual:
                out.append("%s: missing from run (expected %s)" % (child, _short(expected[key])))
            else:
                _diff(expected[key], actual[key], child, out)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(
                "%s: length %d expected, %d in run" % (path, len(expected), len(actual))
            )
            return
        for position, (left, right) in enumerate(zip(expected, actual)):
            _diff(left, right, "%s[%d]" % (path, position), out)
        return
    if canonical(expected) != canonical(actual):
        out.append(
            "%s: expected %s, run has %s" % (path, _short(expected), _short(actual))
        )


def _short(value, limit: int = 60) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."