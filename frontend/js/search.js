import { escapeHtml, escapeRegExp } from "./formatters.js";
import { store } from "./state.js";
import { NAV_TARGETS, navigateToSection } from "./navigation.js";

/**
 * Global dashboard search (Step 9B Phase 1).
 *
 * Searches ONLY the data already loaded into the store by the existing Step 9A
 * dashboard API. No new endpoint, no request per keystroke, no external source,
 * no historical data. The index is derived data: the source store objects are
 * read but never mutated, and result ordering never depends on object or hash
 * iteration order.
 */

export const DEFAULT_RESULT_LIMIT = 50;

export const SEARCH_TYPES = Object.freeze([
  Object.freeze({ type: "observation", priority: 10, label: "Observation" }),
  Object.freeze({ type: "relationship", priority: 20, label: "Evidence relationship" }),
  Object.freeze({ type: "evidence-type", priority: 30, label: "Evidence type" }),
  Object.freeze({ type: "inference-level", priority: 40, label: "Inference level" }),
  Object.freeze({ type: "signal", priority: 50, label: "Signal" }),
  Object.freeze({ type: "path", priority: 60, label: "Investigation path" }),
  Object.freeze({ type: "provenance", priority: 70, label: "Provenance source" }),
  Object.freeze({ type: "behavior", priority: 80, label: "Behavioral / ML" }),
  Object.freeze({ type: "overview", priority: 90, label: "Overview" })
]);

const INFERENCE_DISPLAY_ORDER = ["OBSERVED", "CORRELATED", "INFERRED", "REQUIRES_VERIFICATION"];

const SECTION = Object.freeze({
  overview: "overview-content",
  whyLead: "why-lead-content",
  evidence: "evidence-chain-content",
  crossLayer: "cross-layer-content",
  paths: "paths-content",
  behavior: "behavior-ml-content",
  provenance: "provenance-content"
});

const TYPE_BY_NAME = new Map(SEARCH_TYPES.map(t => [t.type, t]));

function sectionLabel(id) {
  const found = NAV_TARGETS.find(t => t.id === id);
  return found ? found.label : id;
}

function typeMeta(type) {
  const meta = TYPE_BY_NAME.get(type);
  return meta ? meta : { type: type, priority: 999, label: type };
}

function text(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function joinValues(values) {
  return asArray(values).map(text).filter(v => v !== "").join(", ");
}

function sortedUnique(values) {
  const seen = new Set();
  values.forEach(v => {
    const s = text(v).trim();
    if (s !== "") seen.add(s);
  });
  return Array.from(seen).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

function makeRecord(type, id, primary, section, fields) {
  const meta = typeMeta(type);
  const values = [];
  const collect = v => {
    const s = text(v).trim();
    if (s !== "") values.push(s.toLowerCase());
  };
  collect(primary);
  fields.forEach(f => collect(f.value));
  return {
    key: meta.type + ":" + id,
    type: meta.type,
    typeLabel: meta.label,
    typePriority: meta.priority,
    primary: text(primary),
    section: section,
    sectionLabel: sectionLabel(section),
    fields: fields,
    values: values
  };
}

function provenanceLabel(ref) {
  const file = text(ref && ref.source_file);
  const record = text(ref && ref.source_record);
  if (file !== "" && record !== "") return file + ":" + record;
  if (file !== "") return file;
  return record;
}

function collectProvenanceRefs(evidence, signals) {
  const refs = [];
  asArray(evidence).forEach(item => {
    asArray(item && item.provenance_refs).forEach(ref => refs.push(ref));
  });
  asArray(signals).forEach(item => {
    asArray(item && item.provenance_refs).forEach(ref => refs.push(ref));
  });
  return refs;
}

// ---------------------------------------------------------------------------
// Index construction
// ---------------------------------------------------------------------------

function buildObservationRecords(overview, evidence, paths) {
  const subjectIds = new Set(
    asArray(overview && overview.subject_observation_ids).map(text).filter(v => v !== "")
  );
  const all = [];
  evidence.forEach(item => {
    asArray(item && item.source_observation_ids).forEach(id => all.push(text(id)));
    asArray(item && item.target_observation_ids).forEach(id => all.push(text(id)));
  });
  paths.forEach(item => asArray(item && item.observation_ids).forEach(id => all.push(text(id))));

  return sortedUnique(all).map(id => {
    const isSubject = subjectIds.has(id);
    return makeRecord("observation", id, id, isSubject ? SECTION.overview : SECTION.evidence, [
      { label: "Observation ID", value: id },
      { label: "Scope", value: isSubject ? "Subject observation" : "Linked observation" }
    ]);
  });
}

function buildRelationshipRecords(evidence) {
  return evidence.map(item => {
    const provenance = asArray(item && item.provenance_refs)
      .map(provenanceLabel)
      .filter(v => v !== "")
      .join(", ");
    return makeRecord("relationship", item && item.relationship_id, item && item.relationship_id, SECTION.evidence, [
      { label: "Relationship ID", value: item && item.relationship_id },
      { label: "Evidence type", value: item && item.evidence_type },
      { label: "Inference level", value: item && item.inference_level },
      { label: "Source observations", value: joinValues(item && item.source_observation_ids) },
      { label: "Target observations", value: joinValues(item && item.target_observation_ids) },
      { label: "Rationale", value: item && item.rationale },
      { label: "Confidence", value: item && item.confidence_value },
      { label: "Confidence basis", value: item && item.confidence_basis },
      { label: "Rule", value: text(item && item.rule_id) + " v" + text(item && item.rule_version) },
      { label: "Provenance source", value: provenance },
      { label: "Created at", value: item && item.created_at }
    ]);
  });
}

function buildEvidenceTypeRecords(evidence) {
  const counts = new Map();
  evidence.forEach(item => {
    const t = text(item && item.evidence_type).trim();
    if (t === "") return;
    counts.set(t, (counts.get(t) || 0) + 1);
  });
  return sortedUnique(Array.from(counts.keys())).map(type =>
    makeRecord("evidence-type", type, type, SECTION.evidence, [
      { label: "Evidence type", value: type },
      { label: "Relationship count", value: counts.get(type) }
    ])
  );
}

function buildInferenceLevelRecords(evidence) {
  const counts = new Map();
  evidence.forEach(item => {
    const level = text(item && item.inference_level).trim();
    if (level === "") return;
    counts.set(level, (counts.get(level) || 0) + 1);
  });
  const known = [];
  const other = [];
  sortedUnique(Array.from(counts.keys())).forEach(level => {
    if (INFERENCE_DISPLAY_ORDER.indexOf(level) === -1) other.push(level);
    else known.push(level);
  });
  known.sort((a, b) => INFERENCE_DISPLAY_ORDER.indexOf(a) - INFERENCE_DISPLAY_ORDER.indexOf(b));
  return known.concat(other).map(level =>
    makeRecord("inference-level", level, level, SECTION.evidence, [
      { label: "Inference level", value: level },
      { label: "Relationship count", value: counts.get(level) }
    ])
  );
}

function buildSignalRecords(signals) {
  return signals.map(item => {
    const context = asArray(item && item.context)
      .map(pair => (Array.isArray(pair) ? text(pair[0]) + "=" + text(pair[1]) : ""))
      .filter(v => v !== "")
      .join(", ");
    const provenance = asArray(item && item.provenance_refs)
      .map(provenanceLabel)
      .filter(v => v !== "")
      .join(", ");
    return makeRecord("signal", item && item.signal_id, item && item.signal_id, SECTION.whyLead, [
      { label: "Signal ID", value: item && item.signal_id },
      { label: "Signal type", value: item && item.signal_type },
      { label: "Family", value: item && item.family },
      { label: "Role", value: item && item.role },
      { label: "Status", value: item && item.status },
      { label: "Source step", value: item && item.source_step },
      { label: "Normalized value", value: item && item.normalized_value },
      { label: "Inference levels", value: joinValues(item && item.inference_levels) },
      { label: "Evidence references", value: joinValues(item && item.evidence_refs) },
      { label: "Lineage", value: joinValues(item && item.lineage) },
      { label: "Explanation", value: item && item.explanation },
      { label: "Context", value: context },
      { label: "Provenance source", value: provenance }
    ]);
  });
}

function buildPathRecords(paths) {
  return paths.map(item => {
    const summary = (item && item.evidence_summary) || {};
    const bounds = summary.confidence_bounds || {};
    return makeRecord("path", item && item.path_id, item && item.path_id, SECTION.paths, [
      { label: "Path ID", value: item && item.path_id },
      { label: "Source observation", value: item && item.source_observation_id },
      { label: "Target observation", value: item && item.target_observation_id },
      { label: "Hop count", value: item && item.hop_count },
      { label: "Node count", value: item && item.node_count },
      { label: "Observation IDs", value: joinValues(item && item.observation_ids) },
      { label: "Relationship IDs", value: joinValues(item && item.relationship_ids) },
      { label: "Temporal status", value: item && item.temporal_status },
      { label: "Search status", value: item && item.search_status },
      { label: "Duration (s)", value: item && item.duration_seconds },
      { label: "Evidence types", value: joinValues(summary.evidence_types) },
      { label: "Confidence min", value: bounds.min },
      { label: "Confidence max", value: bounds.max },
      { label: "Temporal flow", value: joinValues(item && item.temporal_flow) }
    ]);
  });
}

function buildProvenanceRecords(evidence, signals, provenance) {
  const refs = collectProvenanceRefs(evidence, signals);
  const seen = new Map();
  refs.forEach(ref => {
    const ingestionId = text(ref && ref.ingestion_id);
    const sourceFile = text(ref && ref.source_file);
    const sourceRecord = text(ref && ref.source_record);
    const key = ingestionId + " " + sourceFile + " " + sourceRecord;
    if (seen.has(key)) return;
    const primary = sourceFile !== "" ? sourceFile : ingestionId !== "" ? ingestionId : sourceRecord;
    seen.set(key, makeRecord("provenance", key, primary, SECTION.provenance, [
      { label: "Source file", value: sourceFile },
      { label: "Ingestion ID", value: ingestionId },
      { label: "Source record", value: sourceRecord },
      { label: "Cited by relationship", value: "Evidence relationship" }
    ]));
  });

  asArray(provenance && provenance.source_files).forEach(file => {
    const value = text(file);
    if (value === "") return;
    const key = "file " + value;
    if (seen.has(key)) return;
    seen.set(key, makeRecord("provenance", key, value, SECTION.provenance, [
      { label: "Source file", value: value },
      { label: "Declared by", value: "Provenance summary" }
    ]));
  });

  asArray(provenance && provenance.missing_signals).forEach(signal => {
    const value = text(signal);
    if (value === "") return;
    const key = "missing " + value;
    if (seen.has(key)) return;
    seen.set(key, makeRecord("provenance", key, value, SECTION.provenance, [
      { label: "Signal", value: value },
      { label: "Status", value: "Missing signal" }
    ]));
  });

  return Array.from(seen.values()).sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

function buildBehaviorRecords(behavior) {
  if (!behavior) return [];
  return [
    makeRecord("behavior", behavior.profile_id, behavior.profile_id, SECTION.behavior, [
      { label: "Profile ID", value: behavior.profile_id },
      { label: "Grouping", value: behavior.grouping },
      { label: "Feature count", value: behavior.feature_count },
      { label: "Feature categories", value: joinValues(behavior.feature_categories) },
      { label: "Anomaly band", value: behavior.anomaly_band },
      { label: "Anomaly normalized score", value: behavior.anomaly_normalized_score },
      { label: "Cluster ID", value: behavior.cluster_id },
      { label: "Cluster size", value: behavior.cluster_size },
      { label: "Cluster distance", value: behavior.cluster_distance },
      { label: "Explanation items", value: joinValues(behavior.explanation_items) }
    ])
  ];
}

function buildOverviewRecord(overview) {
  if (!overview) return [];
  const summary = overview.evidence_summary || {};
  return [
    makeRecord("overview", overview.subject_profile_id, overview.subject_profile_id, SECTION.overview, [
      { label: "Subject profile ID", value: overview.subject_profile_id },
      { label: "Assessment ID", value: overview.assessment_id },
      { label: "Config version", value: overview.config_version },
      { label: "Status", value: overview.status },
      { label: "Risk band", value: overview.risk_band },
      { label: "Investigative priority", value: overview.investigative_priority },
      { label: "Observation count", value: overview.observation_count },
      { label: "Evidence relationship count", value: overview.evidence_relationship_count },
      { label: "Investigation path count", value: overview.investigation_path_count },
      { label: "Evidence types", value: joinValues(summary.evidence_types) },
      { label: "Inference levels", value: joinValues(summary.inference_levels) },
      { label: "Interpretation", value: overview.interpretation }
    ])
  ];
}

/**
 * Builds the deterministic search index from already-loaded dashboard state.
 * Pure with respect to `state`: nothing in `state` is written or reordered.
 */
export function buildSearchIndex(state) {
  const source = state || {};
  const overview = source.overview || null;
  const signals = asArray(source.signals);
  const evidence = asArray(source.evidence);
  const paths = asArray(source.paths);
  const behavior = source.behavior || null;
  const provenance = source.provenance || null;

  const records = []
    .concat(buildObservationRecords(overview, evidence, paths))
    .concat(buildRelationshipRecords(evidence))
    .concat(buildEvidenceTypeRecords(evidence))
    .concat(buildInferenceLevelRecords(evidence))
    .concat(buildSignalRecords(signals))
    .concat(buildPathRecords(paths))
    .concat(buildProvenanceRecords(evidence, signals, provenance))
    .concat(buildBehaviorRecords(behavior))
    .concat(buildOverviewRecord(overview));

  return records;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

export function normalizeQuery(query) {
  return text(query).trim().toLowerCase();
}

/**
 * Canonical deterministic ordering:
 *   1. exact match before partial match
 *   2. deterministic type priority (SEARCH_TYPES order)
 *   3. lexical primary value (code-unit order, locale independent)
 *   4. record type then record key as a final stable tiebreak
 */
export function compareResults(a, b) {
  if (a.exact !== b.exact) return a.exact ? -1 : 1;
  if (a.typePriority !== b.typePriority) return a.typePriority - b.typePriority;
  const ap = a.primary.toLowerCase();
  const bp = b.primary.toLowerCase();
  if (ap !== bp) return ap < bp ? -1 : 1;
  if (a.type !== b.type) return a.type < b.type ? -1 : 1;
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
}

export function searchIndex(index, query, options) {
  const settings = options || {};
  const limit = typeof settings.limit === "number" ? settings.limit : DEFAULT_RESULT_LIMIT;
  const normalized = normalizeQuery(query);
  if (normalized === "") {
    return { query: text(query), normalized: "", total: 0, results: [] };
  }

  const matches = [];
  asArray(index).forEach(record => {
    let matched = false;
    for (let i = 0; i < record.values.length; i += 1) {
      if (record.values[i].indexOf(normalized) !== -1) {
        matched = true;
        break;
      }
    }
    if (!matched) return;
    matches.push({
      key: record.key,
      type: record.type,
      typeLabel: record.typeLabel,
      typePriority: record.typePriority,
      primary: record.primary,
      section: record.section,
      sectionLabel: record.sectionLabel,
      exact: record.primary.toLowerCase() === normalized,
      fields: record.fields
    });
  });

  matches.sort(compareResults);
  return {
    query: text(query),
    normalized: normalized,
    total: matches.length,
    results: matches.slice(0, limit)
  };
}

/**
 * Escapes `value`, then wraps case-insensitive matches of `query` in <mark>.
 * The query string itself is never emitted, so raw user input cannot become
 * markup. Match offsets come from a case-insensitive regex over the original
 * string, so indices stay aligned with the untransformed value.
 */
export function highlightMatch(value, query) {
  const source = text(value);
  const needle = text(query);
  if (needle === "") return escapeHtml(source);
  let pattern;
  try {
    pattern = new RegExp(escapeRegExp(needle), "gi");
  } catch {
    return escapeHtml(source);
  }
  let out = "";
  let cursor = 0;
  let match = pattern.exec(source);
  while (match !== null) {
    const start = match.index;
    const end = start + match[0].length;
    out += escapeHtml(source.slice(cursor, start));
    out += "<mark>" + escapeHtml(source.slice(start, end)) + "</mark>";
    cursor = end;
    if (match[0].length === 0) pattern.lastIndex += 1;
    match = pattern.exec(source);
  }
  out += escapeHtml(source.slice(cursor));
  return out;
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

let els = null;
let searchRecords = [];
let lastResult = { total: 0, results: [] };
let currentQuery = "";
let activeIndex = -1;
let unsubscribe = null;
let trackedState = null;

function detailMarkup(result, query) {
  const fields = result.fields
    .filter(f => text(f.value) !== "")
    .slice(0, 3)
    .map(f => '<span class="global-search__detail">' + escapeHtml(f.label) + ": " + highlightMatch(f.value, query) + "</span>")
    .join("");
  return '<span class="global-search__primary">' + highlightMatch(result.primary, query) + "</span>" +
    '<span class="global-search__meta">' + escapeHtml(result.typeLabel) + " · " + escapeHtml(result.sectionLabel) + "</span>" +
    fields;
}

function closeResults() {
  activeIndex = -1;
  if (!els) return;
  els.results.hidden = true;
  els.results.innerHTML = "";
  els.input.setAttribute("aria-expanded", "false");
  els.input.removeAttribute("aria-activedescendant");
}

function setActiveIndex(index) {
  const options = els.results.querySelectorAll('[role="option"]');
  if (options.length === 0) {
    activeIndex = -1;
    els.input.removeAttribute("aria-activedescendant");
    return;
  }
  if (index < 0) index = options.length - 1;
  if (index >= options.length) index = 0;
  activeIndex = index;
  options.forEach((option, i) => {
    const selected = i === activeIndex;
    option.setAttribute("aria-selected", selected ? "true" : "false");
    option.classList.toggle("is-active", selected);
  });
  const active = options[activeIndex];
  els.input.setAttribute("aria-activedescendant", active.id);
  if (typeof active.scrollIntoView === "function") {
    active.scrollIntoView({ block: "nearest" });
  }
}

function selectActiveResult() {
  const result = lastResult.results[activeIndex];
  if (!result) return;
  closeResults();
  navigateToSection(result.section);
}

function renderResults() {
  if (!els) return;
  const query = currentQuery;
  lastResult = searchIndex(searchRecords, query, { limit: DEFAULT_RESULT_LIMIT });
  const normalized = lastResult.normalized;

  if (normalized === "") {
    els.clear.hidden = true;
    els.count.textContent = "";
    closeResults();
    return;
  }

  els.clear.hidden = false;
  els.count.textContent = lastResult.total === 0
    ? "No results for " + query
    : lastResult.total + (lastResult.total === 1 ? " result" : " results") + " for " + query;

  if (lastResult.total === 0) {
    activeIndex = -1;
    els.results.hidden = false;
    els.results.innerHTML =
      '<li class="global-search__empty" role="presentation">No matches in the loaded dashboard data.</li>';
    els.input.setAttribute("aria-expanded", "true");
    els.input.removeAttribute("aria-activedescendant");
    return;
  }

  els.results.hidden = false;
  els.input.setAttribute("aria-expanded", "true");
  els.results.innerHTML = lastResult.results.map((result, i) =>
    '<li class="global-search__option" role="option" id="global-search-option-' + i + '" aria-selected="false" data-index="' + i + '" data-section="' + escapeHtml(result.section) + '">' +
    detailMarkup(result, query) + "</li>"
  ).join("");
  setActiveIndex(0);
}

function onInput() {
  currentQuery = els.input.value;
  renderResults();
}

function onClear() {
  els.input.value = "";
  currentQuery = "";
  els.input.focus();
  renderResults();
}

function onKeyDown(event) {
  const key = event.key;
  if (key === "ArrowDown") {
    if (els.results.hidden) return;
    event.preventDefault();
    setActiveIndex(activeIndex + 1);
  } else if (key === "ArrowUp") {
    if (els.results.hidden) return;
    event.preventDefault();
    setActiveIndex(activeIndex - 1);
  } else if (key === "Home") {
    if (els.results.hidden) return;
    event.preventDefault();
    setActiveIndex(0);
  } else if (key === "End") {
    if (els.results.hidden) return;
    event.preventDefault();
    setActiveIndex(lastResult.results.length - 1);
  } else if (key === "Enter") {
    if (els.results.hidden || activeIndex < 0) return;
    event.preventDefault();
    selectActiveResult();
  } else if (key === "Escape") {
    if (els.results.hidden) return;
    event.preventDefault();
    closeResults();
  }
}

function onResultsClick(event) {
  const option = event.target.closest('[role="option"]');
  if (!option) return;
  activeIndex = Number(option.dataset.index);
  selectActiveResult();
}

function onResultsPointerOver(event) {
  const option = event.target.closest('[role="option"]');
  if (!option) return;
  const index = Number(option.dataset.index);
  if (index !== activeIndex) setActiveIndex(index);
}

function rebuildIndex(state) {
  searchRecords = buildSearchIndex(state);
  trackedState = [state.overview, state.signals, state.evidence, state.paths, state.behavior, state.provenance];
}

function dataChanged(state) {
  if (!trackedState) return true;
  const next = [state.overview, state.signals, state.evidence, state.paths, state.behavior, state.provenance];
  for (let i = 0; i < next.length; i += 1) {
    if (next[i] !== trackedState[i]) return true;
  }
  return false;
}

export function initGlobalSearch() {
  els = {
    input: document.getElementById("global-search-input"),
    clear: document.getElementById("global-search-clear"),
    count: document.getElementById("global-search-count"),
    results: document.getElementById("global-search-results")
  };
  if (!els.input || !els.clear || !els.count || !els.results) {
    els = null;
    return;
  }

  rebuildIndex(store.state);
  currentQuery = "";
  renderResults();

  els.input.addEventListener("input", onInput);
  els.input.addEventListener("search", onInput);
  els.input.addEventListener("keydown", onKeyDown);
  els.clear.addEventListener("click", onClear);
  els.results.addEventListener("click", onResultsClick);
  els.results.addEventListener("pointerover", onResultsPointerOver);

  unsubscribe = store.subscribe(state => {
    if (!dataChanged(state)) return;
    rebuildIndex(state);
    if (els && currentQuery !== "") renderResults();
  });
}

export function cleanupGlobalSearch() {
  if (els) {
    els.input.removeEventListener("input", onInput);
    els.input.removeEventListener("search", onInput);
    els.input.removeEventListener("keydown", onKeyDown);
    els.clear.removeEventListener("click", onClear);
    els.results.removeEventListener("click", onResultsClick);
    els.results.removeEventListener("pointerover", onResultsPointerOver);
  }
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
  els = null;
  searchRecords = [];
  lastResult = { total: 0, results: [] };
  currentQuery = "";
  activeIndex = -1;
  trackedState = null;
}