/**
 * Test fixtures mirroring the Step 9A dashboard API contracts exactly.
 *
 * Field names, value spellings and enum members are taken from
 * `backend/app/dashboard/models.py` and from live responses of
 * `GET /api/dashboard/evidence` and `GET /api/dashboard/paths`. Nothing here is
 * invented: every key is a key the backend actually emits.
 */

/**
 * Evidence fixture with several evidence types and all four inference levels so
 * filtering by type and by level can be exercised with real AND semantics.
 */
export const EVIDENCE = [
  {
    relationship_id: "rel-observed-ip",
    source_observation_ids: ["obs-a1"],
    target_observation_ids: ["obs-b1"],
    evidence_type: "SAME_IP",
    inference_level: "OBSERVED",
    confidence_value: 1.0,
    confidence_basis: "EXACT_MATCH",
    rationale: "Both observations recorded from the same source IP 10.0.0.1.",
    rule_id: "rule.network.same_ip",
    rule_version: "1.0.0",
    provenance_refs: [
      { ingestion_id: "ing-1", source_file: "demo.csv", source_record: "row-obs-a1" },
      { ingestion_id: "ing-1", source_file: "demo.csv", source_record: "row-obs-b1" }
    ],
    created_at: "2024-06-01T00:00:00Z"
  },
  {
    relationship_id: "rel-observed-txid",
    source_observation_ids: ["obs-a2"],
    target_observation_ids: ["obs-b2"],
    evidence_type: "SAME_TXID",
    inference_level: "OBSERVED",
    confidence_value: 1.0,
    confidence_basis: "EXACT_MATCH",
    rationale: "Transaction demo-tx-1 appears in both observation records.",
    rule_id: "rule.blockchain.same_txid",
    rule_version: "1.0.0",
    provenance_refs: [
      { ingestion_id: "ing-1", source_file: "demo.csv", source_record: "row-obs-a2" }
    ],
    created_at: "2024-06-01T00:00:00Z"
  },
  {
    relationship_id: "rel-correlated-adjacency",
    source_observation_ids: ["obs-a1"],
    target_observation_ids: ["obs-c1"],
    evidence_type: "TEMPORAL_ADJACENCY",
    inference_level: "CORRELATED",
    confidence_value: 0.8,
    confidence_basis: "DETERMINISTIC_RULE",
    rationale: "Temporal adjacency within the same hour window.",
    rule_id: "rule.temporal.adjacency",
    rule_version: "1.0.0",
    provenance_refs: [],
    created_at: "2024-06-01T00:00:00Z"
  },
  {
    relationship_id: "rel-correlated-amount",
    source_observation_ids: ["obs-b2"],
    target_observation_ids: ["obs-c2"],
    evidence_type: "AMOUNT_MATCH",
    inference_level: "CORRELATED",
    confidence_value: 0.75,
    confidence_basis: "DETERMINISTIC_RULE",
    rationale: "Matching transaction amounts observed.",
    rule_id: "rule.blockchain.amount_match",
    rule_version: "1.0.0",
    provenance_refs: [],
    created_at: "2024-06-01T00:00:00Z"
  },
  {
    relationship_id: "rel-inferred-address",
    source_observation_ids: ["obs-a1"],
    target_observation_ids: ["obs-c2"],
    evidence_type: "SAME_ADDRESS",
    inference_level: "INFERRED",
    confidence_value: 0.6,
    confidence_basis: "DETERMINISTIC_RULE",
    rationale: "Repeated address usage pattern suggests common controller.",
    rule_id: "rule.behavioral.repeated_address",
    rule_version: "1.0.0",
    provenance_refs: [],
    created_at: "2024-06-01T00:00:00Z"
  },
  {
    relationship_id: "rel-requires-geo",
    source_observation_ids: ["obs-a2"],
    target_observation_ids: ["obs-c1"],
    evidence_type: "GEO_COINCIDENCE",
    inference_level: "REQUIRES_VERIFICATION",
    confidence_value: 0.4,
    confidence_basis: "DETERMINISTIC_RULE",
    rationale: "Same country code observed but requires verification of origin.",
    rule_id: "rule.network.geo_attribution",
    rule_version: "1.0.0",
    provenance_refs: [],
    created_at: "2024-06-01T00:00:00Z"
  }
];

/** Deterministic deep copy used to assert data immutability. */
export function snapshot(value) {
  return JSON.parse(JSON.stringify(value));
}

/**
 * Path A: comparable aware timestamps, so Step 7 reports COMPARABLE and the
 * `temporal_flow` strings carry real AFTER/BEFORE relations and elapsed values.
 */
export const PATH_A = {
  path_id: "path-alpha",
  source_observation_id: "obs-a1",
  target_observation_id: "obs-c2",
  hop_count: 2,
  node_count: 3,
  observation_ids: ["obs-a1", "obs-b1", "obs-c2"],
  relationship_ids: ["rel-observed-ip", "rel-correlated-amount"],
  temporal_status: "COMPARABLE",
  temporal_flow: [
    "obs-a1 -> obs-b1 (SAME_IP, AFTER, elapsed=3600.0s)",
    "obs-b1 -> obs-c2 (AMOUNT_MATCH, BEFORE, elapsed=-1800.0s)"
  ],
  duration_seconds: 1800.0,
  evidence_summary: {
    relationship_count: 2,
    evidence_types: ["AMOUNT_MATCH", "SAME_IP"],
    confidence_bounds: { min: 0.75, max: 1.0 }
  },
  search_status: "FOUND"
};

/**
 * Path B: mixed aware/naive timestamps, so Step 7 reports NOT_COMPARABLE and
 * no elapsed time is invented.
 */
export const PATH_B = {
  path_id: "path-beta",
  source_observation_id: "obs-a2",
  target_observation_id: "obs-c1",
  hop_count: 1,
  node_count: 2,
  observation_ids: ["obs-a2", "obs-c1"],
  relationship_ids: ["rel-requires-geo"],
  temporal_status: "NOT_COMPARABLE",
  temporal_flow: [
    "obs-a2 -> obs-c1 (GEO_COINCIDENCE, NOT_COMPARABLE, elapsed=Nones)"
  ],
  duration_seconds: null,
  evidence_summary: {
    relationship_count: 1,
    evidence_types: ["GEO_COINCIDENCE"],
    confidence_bounds: { min: 0.4, max: 0.4 }
  },
  search_status: "FOUND"
};

/** Path C: no timestamps carried at all, so Step 7 reports UNKNOWN. */
export const PATH_C = {
  path_id: "path-gamma",
  source_observation_id: "obs-b2",
  target_observation_id: "obs-c1",
  hop_count: 1,
  node_count: 2,
  observation_ids: ["obs-b2", "obs-c1"],
  relationship_ids: ["rel-correlated-adjacency"],
  temporal_status: "UNKNOWN",
  temporal_flow: [
    "obs-b2 -> obs-c1 (TEMPORAL_ADJACENCY, UNKNOWN, elapsed=Nones)"
  ],
  duration_seconds: null,
  evidence_summary: {
    relationship_count: 1,
    evidence_types: ["TEMPORAL_ADJACENCY"],
    confidence_bounds: { min: 0.8, max: 0.8 }
  },
  search_status: "FOUND"
};

export const PATHS = [PATH_A, PATH_B, PATH_C];

/** Observation ids that exist in the graph, for exact-id synchronisation. */
export const GRAPH_OBSERVATION_IDS = [
  "obs-a1", "obs-a2", "obs-b1", "obs-b2", "obs-c1", "obs-c2"
];