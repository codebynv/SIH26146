/**
 * Evidence Chain interaction (Step 9B Phase 3).
 *
 * Two independent pieces of presentation state:
 *
 *   expansion  which evidence items are open, tracked by `relationship_id`
 *   selection  which relationship is currently selected
 *
 * Both live in plain objects that the renderer and the tests both consume, so
 * the state transitions are inspectable rather than implicit in the DOM. The DOM
 * is a projection of this state: activating an item updates the state and then
 * re-renders, and `aria-expanded` is written from the state, never the reverse.
 *
 * Every exported state function is pure and returns a new object. Dashboard
 * data is only ever read: `filterEvidence` returns a new array holding the same
 * record references and mutates nothing.
 *
 * Synchronisation with the graph is by exact `relationship_id` only. Nothing is
 * matched on display text, evidence type, or position in an array, and no
 * relationship is created that the data does not already carry.
 */

import { el, replaceChildrenWith, idSpan } from "./dom.js";
import { sortEvidenceByInference, getInferenceClass, escapeHtml } from "./formatters.js";

export const ALL_FILTER = "all";

export const INFERENCE_ORDER = Object.freeze([
  "OBSERVED",
  "CORRELATED",
  "INFERRED",
  "REQUIRES_VERIFICATION"
]);

export const EVIDENCE_TYPE_ORDER = Object.freeze([
  "SAME_SOURCE_RECORD",
  "SAME_TXID",
  "SAME_ADDRESS",
  "SAME_IP",
  "TEMPORAL_ADJACENCY",
  "AMOUNT_MATCH",
  "PROVENANCE_LINK",
  "GEO_COINCIDENCE"
]);

const KNOWN_TYPES = new Set(EVIDENCE_TYPE_ORDER);
const KNOWN_INFERENCE = new Set(INFERENCE_ORDER);

/**
 * Whether a key press means "activate", per the WAI-ARIA button pattern.
 *
 * Enter and Space are both activation keys. `Spacebar` is the legacy IE/Edge
 * spelling of the same key and is accepted so a keyboard that reports it is not
 * silently ignored.
 */
export function isActivationKey(key) {
  return key === "Enter" || key === " " || key === "Spacebar";
}

/** One evidence record's relationship id, or `null` when it carries none. */
export function relationshipIdOf(record) {
  if (!record || typeof record.relationship_id !== "string") return null;
  const id = record.relationship_id.trim();
  return id.length ? id : null;
}

function vocabularySort(seen, order, known) {
  return seen.sort((a, b) => {
    const ia = known.has(a) ? order.indexOf(a) : -1;
    const ib = known.has(b) ? order.indexOf(b) : -1;
    if (ia !== -1 && ib !== -1) return ia - ib;
    if (ia !== -1) return -1;
    if (ib !== -1) return 1;
    return a.localeCompare(b);
  });
}

/** The evidence types present, in Step 4A vocabulary order. */
export function evidenceTypesOf(evidence) {
  const seen = [];
  (evidence || []).forEach(record => {
    const type = record && record.evidence_type;
    if (typeof type === "string" && type && seen.indexOf(type) === -1) seen.push(type);
  });
  return vocabularySort(seen, EVIDENCE_TYPE_ORDER, KNOWN_TYPES);
}

/** The inference levels present, in Step 4A spectrum order. */
export function inferenceLevelsOf(evidence) {
  const seen = [];
  (evidence || []).forEach(record => {
    const level = record && record.inference_level;
    if (typeof level === "string" && level && seen.indexOf(level) === -1) seen.push(level);
  });
  return vocabularySort(seen, INFERENCE_ORDER, KNOWN_INFERENCE);
}

/** The complete filter state: no type filter, no inference filter. */
export function defaultFilters() {
  return { evidenceType: ALL_FILTER, inferenceLevel: ALL_FILTER };
}

/** True when neither filter narrows the collection. */
export function filtersAreDefault(filters) {
  const current = filters || defaultFilters();
  return current.evidenceType === ALL_FILTER && current.inferenceLevel === ALL_FILTER;
}

/**
 * Filter evidence by evidence type and inference level, with AND semantics.
 *
 * `ALL_FILTER` (or an absent filter) does not constrain its dimension, so the
 * two filters compose: a record must satisfy both when both are set. The result
 * is a new array of the original record references; neither the input array nor
 * the records in it are touched.
 */
export function filterEvidence(evidence, filters) {
  const source = Array.isArray(evidence) ? evidence : [];
  const current = filters || defaultFilters();
  const typeFilter = current.evidenceType === ALL_FILTER ? null : current.evidenceType;
  const levelFilter = current.inferenceLevel === ALL_FILTER ? null : current.inferenceLevel;
  if (typeFilter === null && levelFilter === null) return source.slice();
  return source.filter(record => {
    if (!record) return false;
    if (typeFilter !== null && record.evidence_type !== typeFilter) return false;
    if (levelFilter !== null && record.inference_level !== levelFilter) return false;
    return true;
  });
}

/** The initial interaction state: everything collapsed, nothing selected. */
export function defaultState() {
  return { expanded: [], selectedRelationshipId: null };
}

/** Whether one relationship is expanded. */
export function isExpanded(state, relationshipId) {
  if (!state || !relationshipId) return false;
  return (state.expanded || []).indexOf(relationshipId) !== -1;
}

/** Whether one relationship is the current selection. */
export function isSelected(state, relationshipId) {
  return !!state && !!relationshipId && state.selectedRelationshipId === relationshipId;
}

/**
 * Toggle one relationship's expansion and return new state.
 *
 * Selection is untouched: opening a card is independent of selecting its
 * evidence. An unknown id still toggles, so the function is total, but
 * `visibleEvidence` remains the authority on what is rendered.
 */
export function toggleExpanded(state, relationshipId) {
  const current = state || defaultState();
  const expanded = (current.expanded || []).slice();
  const index = expanded.indexOf(relationshipId);
  if (index === -1) expanded.push(relationshipId);
  else expanded.splice(index, 1);
  return { expanded, selectedRelationshipId: current.selectedRelationshipId || null };
}

/** Expand every relationship present in `evidence` and return new state. */
export function expandAll(state, evidence) {
  const current = state || defaultState();
  return {
    expanded: (evidence || []).map(relationshipIdOf).filter(id => id !== null),
    selectedRelationshipId: current.selectedRelationshipId || null
  };
}

/** Collapse every relationship and return new state. */
export function collapseAll(state) {
  const current = state || defaultState();
  return { expanded: [], selectedRelationshipId: current.selectedRelationshipId || null };
}

/**
 * Select a relationship, replacing any previous selection.
 *
 * Selecting the already-selected id clears the selection, so a second
 * activation deselects. Matching is on the exact id and nothing else.
 */
export function selectRelationship(state, relationshipId) {
  const current = state || defaultState();
  const next =
    current.selectedRelationshipId && relationshipId === current.selectedRelationshipId
      ? null
      : relationshipId || null;
  return { expanded: (current.expanded || []).slice(), selectedRelationshipId: next };
}

/** Clear the selection, leaving expansion as it is. */
export function clearSelection(state) {
  const current = state || defaultState();
  return { expanded: (current.expanded || []).slice(), selectedRelationshipId: null };
}

/**
 * Select a relationship without toggling.
 *
 * `selectRelationship` toggles because an investigator clicking the same record
 * twice means to deselect it. A selection arriving from elsewhere is not a click:
 * reporting the same id again must leave it selected rather than flip it off, so
 * this sets the id outright and is idempotent.
 */
export function setSelectedRelationship(state, relationshipId) {
  const current = state || defaultState();
  return {
    expanded: (current.expanded || []).slice(),
    selectedRelationshipId: relationshipId || null
  };
}

/**
 * The visible evidence: filtered, then ordered.
 *
 * Ordering is Step 4A's own — inference level in spectrum order (OBSERVED
 * first) and then `relationship_id` — so the list is deterministic and does not
 * depend on filter state.
 */
export function visibleEvidence(evidence, filters) {
  return sortEvidenceByInference(filterEvidence(evidence, filters));
}

/**
 * Find one evidence record by exact `relationship_id`.
 *
 * Returns `null` when there is no exact match. Never matches on text, on
 * evidence type, or on position in an array.
 */
export function findByRelationshipId(evidence, relationshipId) {
  if (!relationshipId) return null;
  const found = (evidence || []).find(
    record => record && record.relationship_id === relationshipId
  );
  return found === undefined ? null : found;
}

/** The record the state currently selects, or `null`. */
export function selectedEvidence(evidence, state) {
  if (!state || !state.selectedRelationshipId) return null;
  return findByRelationshipId(evidence, state.selectedRelationshipId);
}

/**
 * The detail rows for one evidence record.
 *
 * Every field is copied from the record, including the source and target
 * observation ids. Nothing is computed, ranked or summarised: the record is
 * already a Step 4A relationship and this only presents it.
 */
export function evidenceDetails(record) {
  if (!record) return null;
  return {
    relationshipId: record.relationship_id,
    evidenceType: record.evidence_type,
    inferenceLevel: record.inference_level,
    confidenceValue: record.confidence_value,
    confidenceBasis: record.confidence_basis,
    rationale: record.rationale,
    ruleId: record.rule_id,
    ruleVersion: record.rule_version,
    sources: (record.source_observation_ids || []).slice(),
    targets: (record.target_observation_ids || []).slice(),
    provenance: (record.provenance_refs || []).slice(),
    createdAt: record.created_at === undefined ? null : record.created_at
  };
}

/**
 * Exact-id synchronisation between an evidence selection and the graph.
 *
 * A relationship synchronises only when the graph holds a record carrying that
 * exact `relationship_id`. When it does not, `synchronized` is false and
 * `graphRecord` is `null`, so nothing downstream can act on a relationship that
 * was never recorded.
 */
export function syncToGraph(record, graphEvidence) {
  const relationshipId = record ? record.relationship_id || null : null;
  if (!relationshipId) {
    return { synchronized: false, relationshipId: null, graphRecord: null };
  }
  const graphRecord = findByRelationshipId(graphEvidence, relationshipId);
  return {
    synchronized: graphRecord !== null,
    relationshipId,
    graphRecord
  };
}

/**
 * The `aria-expanded` value for one relationship, as a string.
 *
 * Derived from state, so the attribute cannot disagree with the model.
 */
export function ariaExpandedFor(state, relationshipId) {
  return isExpanded(state, relationshipId) ? "true" : "false";
}

function inferenceBadge(record) {
  return el(
    "span",
    { class: "badge inference-badge " + getInferenceClass(record.inference_level) },
    String(record.inference_level || "")
  );
}

function detailRow(field, label, value) {
  return el(
    "div",
    { class: "evidence-item__detail", "data-field": field },
    [
      el("span", { class: "evidence-item__label" }, label),
      el("span", { class: "evidence-item__value" }, value)
    ]
  );
}

function idList(ids) {
  if (!ids || ids.length === 0) return "-";
  return el(
    "span",
    { class: "evidence-item__ids" },
    ids.map(id => idSpan(id, "mono evidence-item__id", 24))
  );
}

/** Build one evidence item, header plus body. */
export function renderEvidenceItem(record, state) {
  const relationshipId = record.relationship_id;
  const expanded = isExpanded(state, relationshipId);
  const selected = isSelected(state, relationshipId);
  const bodyId = "evidence-body-" + relationshipId;

  const toggle = el(
    "button",
    {
      type: "button",
      class: "evidence-item__toggle",
      "aria-expanded": ariaExpandedFor(state, relationshipId),
      "aria-controls": bodyId,
      "data-relationship-id": relationshipId
    },
    [
      el("span", { class: "evidence-item__type" }, String(record.evidence_type || "")),
      inferenceBadge(record),
      el("span", { class: "evidence-item__confidence mono" }, String(record.confidence_value))
    ]
  );

  const body = el(
    "div",
    {
      class: "evidence-item__body",
      id: bodyId,
      role: "region",
      "aria-label": "Evidence details for " + relationshipId,
      hidden: !expanded
    },
    [
      detailRow("relationship_id", "Relationship", relationshipId),
      detailRow("source_observation_ids", "Source", idList(record.source_observation_ids)),
      detailRow("target_observation_ids", "Target", idList(record.target_observation_ids)),
      detailRow(
        "confidence",
        "Confidence",
        String(record.confidence_value) + " (" + String(record.confidence_basis) + ")"
      ),
      detailRow("rule", "Rule", String(record.rule_id) + " v" + String(record.rule_version)),
      detailRow(
        "provenance",
        "Provenance",
        (record.provenance_refs || [])
          .map(ref => ref.source_file)
          .filter(Boolean)
          .join(", ") || "-"
      ),
      detailRow("rationale", "Rationale", String(record.rationale || ""))
    ]
  );

  return el(
    "article",
    {
      class: "evidence-item" + (selected ? " is-selected" : ""),
      "data-relationship-id": relationshipId,
      "data-expanded": expanded ? "true" : "false"
    },
    [el("h4", { class: "evidence-item__header" }, toggle), body]
  );
}

/**
 * Render the evidence chain into `container`.
 *
 * The container is replaced wholesale, so the DOM is always a fresh projection
 * of `state` and `filters`. Every item carries its `relationship_id` as a data
 * attribute, which is what selection and graph synchronisation match on. Returns
 * the records that were rendered, in render order.
 */
export function renderEvidenceChain(container, evidence, state, filters) {
  const source = Array.isArray(evidence) ? evidence : [];
  const records = visibleEvidence(source, filters);
  const nodes = [
    el(
      "div",
      { class: "evidence-chain__count", role: "status", "aria-live": "polite" },
      records.length + " of " + source.length + " relationships"
    )
  ];
  if (records.length === 0) {
    nodes.push(
      el(
        "div",
        { class: "evidence-chain__empty" },
        "No evidence matches the current filters."
      )
    );
  } else {
    records.forEach(record => nodes.push(renderEvidenceItem(record, state)));
  }
  replaceChildrenWith(container, nodes);
  return records;
}

/**
 * Render the selected evidence into the details panel.
 *
 * `sync` is the result of `syncToGraph`, so the panel states plainly whether
 * the selection also exists in the graph rather than implying that it does.
 * Returns the details object, or `null` when nothing is selected.
 */
export function renderSelectionDetails(container, record, sync) {
  if (!record) {
    replaceChildrenWith(container, [
      el("div", { class: "evidence-selection__empty" }, "No evidence selected.")
    ]);
    return null;
  }
  const details = evidenceDetails(record);
  const syncText = sync && sync.synchronized
    ? "Synchronized with graph relationship " + sync.relationshipId + "."
    : "No exact graph relationship for " + details.relationshipId + ".";
  const row = (field, label, value) =>
    el("div", { class: "evidence-selection__row", "data-field": field }, [
      el("span", { class: "evidence-selection__label" }, label),
      typeof value === "string"
        ? el("span", { class: "evidence-selection__value" }, value)
        : value
    ]);
  replaceChildrenWith(container, [
    el(
      "div",
      { class: "evidence-selection", "data-relationship-id": details.relationshipId },
      [
        row("relationship_id", "Relationship", details.relationshipId),
        row("evidence_type", "Evidence type", String(details.evidenceType)),
        row("inference_level", "Inference level", String(details.inferenceLevel)),
        row("source_observation_ids", "Source observations", idList(details.sources)),
        row("target_observation_ids", "Target observations", idList(details.targets)),
        row(
          "confidence",
          "Confidence",
          String(details.confidenceValue) + " (" + String(details.confidenceBasis) + ")"
        ),
        row("rule", "Rule", String(details.ruleId) + " v" + String(details.ruleVersion)),
        row("rationale", "Rationale", String(details.rationale)),
        row("graph_sync", "Graph", syncText)
      ]
    )
  ]);
  return details;
}

/**
 * Reconcile a rendered chain's `aria-expanded` and `hidden` attributes with
 * `state`, in place.
 *
 * Called after a toggle so assistive technology is told about the new state
 * even before the next render. Returns the relationship ids whose
 * `aria-expanded` value actually changed, which is what a state-transition
 * assertion can check.
 */
export function applyExpansionState(container, state) {
  const changed = [];
  const toggles = container.querySelectorAll(".evidence-item__toggle");
  toggles.forEach(toggle => {
    const relationshipId = toggle.getAttribute("data-relationship-id");
    const next = ariaExpandedFor(state, relationshipId);
    if (toggle.getAttribute("aria-expanded") !== next) {
      toggle.setAttribute("aria-expanded", next);
      changed.push(relationshipId);
    }
    const article = toggle.closest(".evidence-item");
    const body = article ? article.querySelector(".evidence-item__body") : null;
    if (!body) return;
    if (isExpanded(state, relationshipId)) body.removeAttribute("hidden");
    else body.setAttribute("hidden", "");
    if (article) article.setAttribute("data-expanded", isExpanded(state, relationshipId) ? "true" : "false");
  });
  return changed;
}

/**
 * Apply a selection to the rendered chain.
 *
 * Writes `is-selected` and `aria-pressed` on exactly one item and clears them on
 * every other, so the DOM cannot report two selected items or none.
 */
export function applySelectionState(container, state) {
  const items = container.querySelectorAll(".evidence-item");
  items.forEach(item => {
    const relationshipId = item.getAttribute("data-relationship-id");
    const selected = isSelected(state, relationshipId);
    item.classList.toggle("is-selected", selected);
    item.setAttribute("aria-pressed", selected ? "true" : "false");
  });
  const selected = selectedIdOf(state);
  return selected;
}

function selectedIdOf(state) {
  return state && state.selectedRelationshipId ? state.selectedRelationshipId : null;
}

/**
 * Wire the evidence chain section.
 *
 * Expansion and selection are both driven from one delegated click listener on
 * the container, and one delegated keydown listener that maps Enter and Space
 * onto the same toggle a click performs. Using a native `<button>` means the
 * keyboard path is genuine activation rather than a synthesised click, and
 * `aria-expanded` is written from state on every transition.
 *
 * `onSelect` receives `(record, syncResult)` so a caller can synchronise with
 * the graph. Returns a handle exposing the current state and filters for
 * inspection.
 */
export function initEvidenceChain(options) {
  const config = options || {};
  const container = document.getElementById("evidence-chain-content");
  const detailsContainer = document.getElementById("evidence-selection-details");
  const typeSelect = document.getElementById("evidence-filter-type");
  const inferenceSelect = document.getElementById("evidence-filter-inference");
  const clearButton = document.getElementById("evidence-clear-filters");
  const expandAllButton = document.getElementById("evidence-expand-all");
  const collapseAllButton = document.getElementById("evidence-collapse-all");

  let evidence = Array.isArray(config.evidence) ? config.evidence : [];
  let state = config.state || defaultState();
  let filters = config.filters || defaultFilters();

  function currentRecord() {
    return selectedEvidence(evidence, state);
  }

  /**
   * Offer exactly the evidence types and inference levels the loaded data holds.
   *
   * The selects ship with a single "all" option, so without this they can only
   * ever report "all" and the filters do nothing. The options are read from the
   * records themselves, in Step 4A vocabulary order, so the control can never
   * offer a value no record carries. A selected value is restored when it is
   * still present, and dropped to "all" when the new data no longer holds it.
   */
  function fillSelect(select, allLabel, values, selected) {
    if (!select) return null;
    const options = [el("option", { value: ALL_FILTER }, allLabel)];
    values.forEach(value => options.push(el("option", { value }, value)));
    replaceChildrenWith(select, options);
    const keep = values.indexOf(selected) !== -1 ? selected : ALL_FILTER;
    select.value = keep;
    select.setAttribute("value", keep);
    return keep;
  }

  function syncFilterOptions() {
    const next = { ...filters };
    const evidenceType = fillSelect(
      typeSelect,
      "All Types",
      evidenceTypesOf(evidence),
      filters.evidenceType
    );
    if (evidenceType !== null) next.evidenceType = evidenceType;
    const inferenceLevel = fillSelect(
      inferenceSelect,
      "All Levels",
      inferenceLevelsOf(evidence),
      filters.inferenceLevel
    );
    if (inferenceLevel !== null) next.inferenceLevel = inferenceLevel;
    // A value the new data no longer holds would filter everything away, so the
    // filter falls back to "all" rather than reporting an empty section. A
    // control that is not on the page cannot invalidate anything, so a filter
    // set programmatically is left exactly as it was given.
    filters = next;
  }

  function render() {
    if (!container) return visibleEvidence(evidence, filters);
    syncFilterOptions();
    const rendered = renderEvidenceChain(container, evidence, state, filters);
    if (detailsContainer) {
      const record = currentRecord();
      renderSelectionDetails(
        detailsContainer,
        record,
        syncToGraph(record, config.graphEvidence || evidence)
      );
    }
    return rendered;
  }

  function setFilters(next) {
    filters = { ...defaultFilters(), ...(next || {}) };
    render();
    return filters;
  }

  /**
 * Perform the expansion toggle a click or an Enter/Space keydown means.
 *
 * `preventDefault` on the keydown suppresses the click a browser synthesises
 * from Enter or Space on a `<button>`, so a keyboard activation toggles exactly
 * once rather than twice.
 */
function expandViaToggle(event) {
    const toggle = event.target.closest(".evidence-item__toggle");
    if (!toggle) return false;
    if (event.type === "keydown") event.preventDefault();
    // The container's delegated selection handlers below must not read this
    // same event as a selection activation.
    event.stopPropagation();
    const relationshipId = toggle.getAttribute("data-relationship-id");
    state = toggleExpanded(state, relationshipId);
    applyExpansionState(container, state);
    if (config.onToggle) config.onToggle(relationshipId, isExpanded(state, relationshipId));
    return true;
  }

  function onContainerClick(event) {
    expandViaToggle(event);
  }

  function onContainerKeydown(event) {
    if (!isActivationKey(event.key)) return;
    expandViaToggle(event);
  }

  function onSelectClick(event) {
    const item = event.target.closest(".evidence-item");
    if (!item) return;
    const toggle = event.target.closest(".evidence-item__toggle");
    // An expansion toggle is its own control; selecting is a separate
    // activation, so a click on the toggle must not also select.
    if (toggle && toggle.getAttribute("data-relationship-id") === item.getAttribute("data-relationship-id")) {
      return;
    }
    const relationshipId = item.getAttribute("data-relationship-id");
    applySelection(selectRelationship(state, relationshipId));
  }

  /**
   * Move to a new state, repaint, and report the selection.
   *
   * Every path that changes the selection goes through here, so the details panel
   * and the `onSelect` callback can never disagree with what the DOM shows.
   *
   * `report` is false when the change originated elsewhere — a selection the
   * graph reported back to us. Echoing it to `onSelect` would tell the graph about
   * the graph's own selection and loop forever.
   */
  function applySelection(nextState, report) {
    state = nextState;
    applySelectionState(container, state);
    const record = selectedEvidence(evidence, state);
    const sync = syncToGraph(record, config.graphEvidence || evidence);
    if (detailsContainer) renderSelectionDetails(detailsContainer, record, sync);
    if (report !== false && config.onSelect) config.onSelect(record, sync);
    return record;
  }

  function onSelectKeydown(event) {
    if (!isActivationKey(event.key)) return;
    const item = event.target.closest(".evidence-item");
    if (!item) return;
    // Suppress the click a browser synthesises from Enter/Space on a button, so
    // keyboard selection happens once, exactly as a click does.
    event.preventDefault();
    onSelectClick(event);
  }

  function onTypeChange() {
    setFilters({ ...filters, evidenceType: typeSelect ? typeSelect.value : ALL_FILTER });
  }

  function onInferenceChange() {
    setFilters({ ...filters, inferenceLevel: inferenceSelect ? inferenceSelect.value : ALL_FILTER });
  }

  function onClearFilters() {
    setFilters(defaultFilters());
  }

  function onExpandAll() {
    state = expandAll(state, visibleEvidence(evidence, filters));
    applyExpansionState(container, state);
  }

  function onCollapseAll() {
    state = collapseAll(state);
    applyExpansionState(container, state);
  }

  if (container) {
    // One delegated click and one delegated keydown on the container. Activation
    // inside a toggle is intercepted by `expandViaToggle`, which stops the event
    // from reaching the selection handlers; activating anywhere else in an item
    // selects it.
    container.addEventListener("click", onContainerClick);
    container.addEventListener("keydown", onContainerKeydown);
    container.addEventListener("click", onSelectClick);
    container.addEventListener("keydown", onSelectKeydown);
  }
  if (typeSelect) typeSelect.addEventListener("change", onTypeChange);
  if (inferenceSelect) inferenceSelect.addEventListener("change", onInferenceChange);
  if (clearButton) clearButton.addEventListener("click", onClearFilters);
  if (expandAllButton) expandAllButton.addEventListener("click", onExpandAll);
  if (collapseAllButton) collapseAllButton.addEventListener("click", onCollapseAll);

  render();

  return {
    getState: () => state,
    getFilters: () => filters,
    getEvidence: () => evidence,
    getSelectedRecord: currentRecord,
    getVisible: () => visibleEvidence(evidence, filters),
    setEvidence(next) {
      evidence = Array.isArray(next) ? next : [];
      // Keep the graph-side copy in step with the section. Holding on to the
      // array supplied at mount time would validate later selections against a
      // stale dataset and silently refuse ids the graph really does hold.
      config.graphEvidence = evidence;
      render();
    },
    setFilters,
    /**
     * Mark a relationship selected because the graph reported it as selected.
     * Only an id this section actually holds is selected; any other leaves the
     * section's own selection untouched rather than matching a near record.
     */
    highlightRelationship(relationshipId) {
      if (!findByRelationshipId(evidence, relationshipId)) return null;
      return applySelection(setSelectedRelationship(state, relationshipId), false);
    },
    clearHighlight() {
      return applySelection(clearSelection(state), false);
    },
    toggle(relationshipId) {
      state = toggleExpanded(state, relationshipId);
      applyExpansionState(container, state);
      return isExpanded(state, relationshipId);
    },
    select(relationshipId) {
      return applySelection(selectRelationship(state, relationshipId));
    },
    destroy() {
      if (container) {
        container.removeEventListener("click", onSelectClick);
        container.removeEventListener("keydown", onSelectKeydown);
      }
      if (typeSelect) typeSelect.removeEventListener("change", onTypeChange);
      if (inferenceSelect) inferenceSelect.removeEventListener("change", onInferenceChange);
      if (clearButton) clearButton.removeEventListener("click", onClearFilters);
      if (expandAllButton) expandAllButton.removeEventListener("click", onExpandAll);
      if (collapseAllButton) collapseAllButton.removeEventListener("click", onCollapseAll);
    }
  };
}

/** Teardown handle retained for the dashboard's own lifecycle. */
let activeChain = null;

export function mountEvidenceChain(options) {
  if (activeChain) activeChain.destroy();
  activeChain = initEvidenceChain(options);
  return activeChain;
}

export function unmountEvidenceChain() {
  if (activeChain) {
    activeChain.destroy();
    activeChain = null;
  }
}

export { escapeHtml };