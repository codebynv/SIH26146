/**
 * Behavioural verification of the Step 9B Phase 3 evidence chain.
 *
 * Every check below runs the real `frontend/js/evidence-chain.js` module against
 * the real DOM it renders into, then asserts on observed state transitions:
 * `aria-expanded` before and after activation, the collection a filter returns,
 * which record a selection resolves to, and whether the graph synchronised.
 * Nothing here inspects source text to conclude that behaviour exists.
 *
 * Run standalone:  node tests/js/run_evidence.mjs
 */

import assert from "node:assert/strict";
import { installDom } from "./minidom.mjs";
import { EVIDENCE, snapshot } from "./fixtures.mjs";

installDom();

const evidence = await import("../../frontend/js/evidence-chain.js");

const {
  ALL_FILTER,
  defaultFilters,
  defaultState,
  filterEvidence,
  visibleEvidence,
  evidenceTypesOf,
  inferenceLevelsOf,
  isExpanded,
  toggleExpanded,
  expandAll,
  collapseAll,
  selectRelationship,
  clearSelection,
  findByRelationshipId,
  selectedEvidence,
  evidenceDetails,
  syncToGraph,
  ariaExpandedFor,
  renderEvidenceChain,
  applyExpansionState,
  applySelectionState,
  initEvidenceChain
} = evidence;

const results = [];
let failures = 0;

function scenario(name, body) {
  try {
    const detail = body();
    results.push({ name, ok: true, detail });
  } catch (error) {
    failures += 1;
    results.push({ name, ok: false, error: error.message });
  }
}

function freshContainer() {
  document.body.replaceChildren();
  const container = document.createElement("div");
  container.setAttribute("id", "evidence-chain-content");
  const details = document.createElement("div");
  details.setAttribute("id", "evidence-selection-details");
  document.body.appendChild(container);
  document.body.appendChild(details);
  return { container, details };
}

function ariaExpandedOf(container, relationshipId) {
  const toggle = container
    .querySelectorAll(".evidence-item__toggle")
    .find(node => node.getAttribute("data-relationship-id") === relationshipId);
  return toggle ? toggle.getAttribute("aria-expanded") : null;
}

scenario("expansion toggles collapsed to expanded and back", () => {
  const { container } = freshContainer();
  const state = defaultState();
  renderEvidenceChain(container, EVIDENCE, state, defaultFilters());

  assert.equal(ariaExpandedOf(container, "rel-observed-ip"), "false");
  assert.equal(state.expanded.length, 0);

  const expanded = toggleExpanded(state, "rel-observed-ip");
  assert.equal(isExpanded(expanded, "rel-observed-ip"), true);
  applyExpansionState(container, expanded);
  assert.equal(ariaExpandedOf(container, "rel-observed-ip"), "true");

  const collapsed = toggleExpanded(expanded, "rel-observed-ip");
  assert.equal(isExpanded(collapsed, "rel-observed-ip"), false);
  applyExpansionState(container, collapsed);
  assert.equal(ariaExpandedOf(container, "rel-observed-ip"), "false");

  return {
    initialExpanded: isExpanded(state, "rel-observed-ip"),
    afterFirstClick: isExpanded(expanded, "rel-observed-ip"),
    afterSecondClick: isExpanded(collapsed, "rel-observed-ip"),
    ariaAfterFirstActivation: "true",
    ariaAfterSecondActivation: "false",
    finalExpandedList: collapsed.expanded
  };
});

scenario("aria-expanded is derived from state for every item", () => {
  const { container } = freshContainer();
  const state = toggleExpanded(defaultState(), "rel-observed-txid");
  renderEvidenceChain(container, EVIDENCE, state, defaultFilters());

  const observed = {};
  container.querySelectorAll(".evidence-item__toggle").forEach(toggle => {
    observed[toggle.getAttribute("data-relationship-id")] =
      toggle.getAttribute("aria-expanded");
  });

  const expected = {};
  container.querySelectorAll(".evidence-item__toggle").forEach(toggle => {
    const id = toggle.getAttribute("data-relationship-id");
    observed[id] = toggle.getAttribute("aria-expanded");
    expected[id] = ariaExpandedFor(state, id);
  });

  assert.equal(observed["rel-observed-txid"], "true");
  assert.equal(observed["rel-observed-ip"], "false");
  assert.equal(observed["rel-requires-geo"], "false");
  assert.equal(ariaExpandedFor(state, "rel-observed-txid"), "true");
  assert.equal(ariaExpandedFor(defaultState(), "rel-observed-txid"), "false");
  assert.equal(observed["rel-observed-ip"], ariaExpandedFor(state, "rel-observed-ip"));

  // aria-expanded must be present on every toggle and must never disagree with
  // the state that produced the render.
  const ids = Object.keys(observed);
  const missing = ids.filter(id => observed[id] === null || observed[id] === undefined);
  const stale = ids.filter(id => observed[id] !== expected[id]);
  assert.equal(missing.length, 0);
  assert.equal(stale.length, 0);

  return {
    itemCount: ids.length,
    expandedCount: ids.filter(id => observed[id] === "true").length,
    collapsedAria: ids.filter(id => observed[id] === "false").length,
    itemsWithoutAria: missing.length,
    staleAriaCount: stale.length,
    ariaByRelationship: observed
  };
});

scenario("expand all then collapse all moves every item", () => {
  const { container } = freshContainer();
  renderEvidenceChain(container, EVIDENCE, defaultState(), defaultFilters());

  const expanded = expandAll(defaultState(), EVIDENCE);
  applyExpansionState(container, expanded);
  assert.equal(expanded.expanded.length, EVIDENCE.length);
  const expandedAria = container
    .querySelectorAll(".evidence-item__toggle")
    .map(toggle => toggle.getAttribute("aria-expanded"));
  assert.ok(expandedAria.every(value => value === "true"));

  const collapsed = collapseAll(expanded);
  applyExpansionState(container, collapsed);
  assert.equal(collapsed.expanded.length, 0);
  const collapsedAria = container
    .querySelectorAll(".evidence-item__toggle")
    .map(toggle => toggle.getAttribute("aria-expanded"));
  assert.ok(collapsedAria.every(value => value === "false"));

  return {
    itemCount: expandedAria.length,
    expandedAfterExpandAll: expandedAria.filter(v => v === "true").length,
    expandedAfterCollapseAll: collapsedAria.filter(v => v === "true").length,
    allAriaAfterCollapse: collapsedAria.every(v => v === "false") ? "false" : "mixed",
    allExpandedThenCollapsed: true
  };
});

scenario("click activates expansion and reports the new state", () => {
  const { container } = freshContainer();
  let reported = null;
  const handle = initEvidenceChain({
    evidence: EVIDENCE,
    onToggle: (id, nowExpanded) => {
      reported = { id, nowExpanded };
    }
  });

  const toggle = container
    .querySelectorAll(".evidence-item__toggle")
    .find(node => node.getAttribute("data-relationship-id") === "rel-observed-ip");

  toggle.click();
  assert.deepEqual(reported, { id: "rel-observed-ip", nowExpanded: true });
  assert.equal(handle.getState().expanded.indexOf("rel-observed-ip") !== -1, true);
  assert.equal(ariaExpandedOf(container, "rel-observed-ip"), "true");

  toggle.click();
  assert.deepEqual(reported, { id: "rel-observed-ip", nowExpanded: false });
  assert.equal(ariaExpandedOf(container, "rel-observed-ip"), "false");

  handle.destroy();
  return {
    clicked: true,
    reported,
    reportedTransitions: ["expanded:true", "expanded:false"],
    ariaAfterSecondClick: ariaExpandedOf(container, "rel-observed-ip")
  };
});

scenario("Enter and Space both toggle, and neither double-toggles", () => {
  const { container } = freshContainer();
  let toggles = 0;
  const handle = initEvidenceChain({
    evidence: EVIDENCE,
    onToggle: () => {
      toggles += 1;
    }
  });

  const toggle = container
    .querySelectorAll(".evidence-item__toggle")
    .find(node => node.getAttribute("data-relationship-id") === "rel-correlated-adjacency");

  toggle.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
  assert.equal(toggles, 1);
  const enterExpanded = ariaExpandedOf(container, "rel-correlated-adjacency");
  assert.equal(enterExpanded, "true");
  assert.equal(handle.getState().expanded.length, 1);

  toggle.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
  const spaceAfterEnterExpanded = ariaExpandedOf(container, "rel-correlated-adjacency");
  assert.equal(toggles, 2);
  assert.equal(spaceAfterEnterExpanded, "false");
  assert.equal(handle.getState().expanded.length, 0);

  toggle.dispatchEvent(new KeyboardEvent("keydown", { key: "a" }));
  assert.equal(toggles, 2, "a non-activation key must not toggle");

  handle.destroy();
  return {
    keysTested: ["Enter", "Space", "a"],
    toggles,
    enterExpanded: enterExpanded,
    spaceAfterEnterExpanded: spaceAfterEnterExpanded,
    nonActivationKeyIgnored: toggles === 2
  };
});

scenario("filtering by evidence type returns only that type", () => {
  const { container } = freshContainer();
  const filtered = filterEvidence(EVIDENCE, {
    evidenceType: "SAME_IP",
    inferenceLevel: ALL_FILTER
  });
  assert.equal(filtered.length, 1);
  assert.ok(filtered.every(record => record.evidence_type === "SAME_IP"));
  assert.equal(filtered[0].relationship_id, "rel-observed-ip");

  renderEvidenceChain(container, EVIDENCE, defaultState(), {
    evidenceType: "SAME_IP",
    inferenceLevel: ALL_FILTER
  });
  const rendered = container.querySelectorAll(".evidence-item");
  assert.equal(rendered.length, 1);
  assert.equal(rendered[0].getAttribute("data-relationship-id"), "rel-observed-ip");

  const types = evidenceTypesOf(EVIDENCE);
  assert.equal(types.indexOf("SAME_IP") !== -1, true);
  assert.equal(types.length, new Set(EVIDENCE.map(r => r.evidence_type)).size);

  return {
    total: EVIDENCE.length,
    filter: "SAME_IP",
    filterValue: "SAME_IP",
    matched: filtered.length,
    matchedIds: filtered.map(record => record.relationship_id),
    matchedEvidenceTypes: filtered.map(record => record.evidence_type),
    renderedCount: rendered.length,
    distinctTypes: types
  };
});

scenario("filtering by inference level returns only that level", () => {
  const filtered = filterEvidence(EVIDENCE, {
    evidenceType: ALL_FILTER,
    inferenceLevel: "REQUIRES_VERIFICATION"
  });
  assert.equal(filtered.length, 1);
  assert.equal(filtered[0].relationship_id, "rel-requires-geo");
  assert.ok(filtered.every(record => record.inference_level === "REQUIRES_VERIFICATION"));

  const observed = filterEvidence(EVIDENCE, {
    evidenceType: ALL_FILTER,
    inferenceLevel: "OBSERVED"
  });
  assert.equal(observed.length, 2);
  assert.ok(observed.every(record => record.inference_level === "OBSERVED"));

  const levels = inferenceLevelsOf(EVIDENCE);
  assert.deepEqual(levels, [
    "OBSERVED",
    "CORRELATED",
    "INFERRED",
    "REQUIRES_VERIFICATION"
  ]);

  return {
    total: EVIDENCE.length,
    filterValue: "REQUIRES_VERIFICATION",
    matched: filtered.length,
    matchedIds: filtered.map(r => r.relationship_id),
    matchedInferenceLevels: filtered.map(r => r.inference_level),
    observedMatched: observed.length,
    requiresVerificationIds: filtered.map(r => r.relationship_id),
    observedIds: observed.map(r => r.relationship_id),
    levelOrder: levels
  };
});

scenario("combined filters use AND semantics", () => {
  const both = filterEvidence(EVIDENCE, {
    evidenceType: "SAME_IP",
    inferenceLevel: "CORRELATED"
  });
  assert.deepEqual(both, [], "SAME_IP is OBSERVED, so AND with CORRELATED is empty");

  const observedSameIp = filterEvidence(EVIDENCE, {
    evidenceType: "SAME_IP",
    inferenceLevel: "OBSERVED"
  });
  assert.equal(observedSameIp.length, 1);
  assert.equal(observedSameIp[0].relationship_id, "rel-observed-ip");

  // AND semantics means the combined result is exactly the intersection of the
  // two single-filter results, and is a subset of each.
  const sameIpOnly = filterEvidence(EVIDENCE, {
    evidenceType: "SAME_IP",
    inferenceLevel: ALL_FILTER
  });
  const correlatedOnly = filterEvidence(EVIDENCE, {
    evidenceType: ALL_FILTER,
    inferenceLevel: "CORRELATED"
  });
  assert.equal(sameIpOnly.length, 1);
  assert.equal(correlatedOnly.length, 2);

  const intersection = correlatedOnly.filter(record =>
    sameIpOnly.some(other => other.relationship_id === record.relationship_id)
  );
  assert.deepEqual(
    both.map(r => r.relationship_id),
    intersection.map(r => r.relationship_id)
  );
  assert.equal(both.length <= sameIpOnly.length, true);
  assert.equal(both.length <= correlatedOnly.length, true);

  return {
    sameIpAndCorrelated: both.length,
    sameIpAndObserved: observedSameIp.map(r => r.relationship_id),
    sameIpOnly: sameIpOnly.map(r => r.relationship_id),
    correlatedOnly: correlatedOnly.map(r => r.relationship_id),
    combinedEqualsIntersection: true
  };
});

scenario("clearing filters restores the complete evidence set", () => {
  const before = visibleEvidence(EVIDENCE, {
    evidenceType: "GEO_COINCIDENCE",
    inferenceLevel: "REQUIRES_VERIFICATION"
  });
  assert.equal(before.length, 1);

  const after = visibleEvidence(EVIDENCE, defaultFilters());
  assert.equal(after.length, EVIDENCE.length);
  assert.deepEqual(
    after.map(record => record.relationship_id).sort(),
    EVIDENCE.map(record => record.relationship_id).sort()
  );

  const { container } = freshContainer();
  const handle = initEvidenceChain({ evidence: EVIDENCE });
  assert.equal(handle.getVisible().length, EVIDENCE.length);
  handle.setFilters({ evidenceType: "GEO_COINCIDENCE", inferenceLevel: "REQUIRES_VERIFICATION" });
  assert.equal(handle.getVisible().length, 1);
  assert.equal(container.querySelectorAll(".evidence-item").length, 1);

  const clear = document.getElementById("evidence-clear-filters");
  assert.equal(handle.setFilters(defaultFilters()).evidenceType, ALL_FILTER);
  assert.equal(handle.getVisible().length, EVIDENCE.length);
  assert.equal(container.querySelectorAll(".evidence-item").length, EVIDENCE.length);
  handle.destroy();
  void clear;

  return {
    total: EVIDENCE.length,
    renderedWhenFiltered: 1,
    restoredCount: after.length,
    renderedAfterClear: container.querySelectorAll(".evidence-item").length,
    defaultFilters: defaultFilters()
  };
});

scenario("selecting a relationship resolves that exact record", () => {
  const state = selectRelationship(defaultState(), "rel-inferred-address");
  assert.equal(state.selectedRelationshipId, "rel-inferred-address");

  const record = selectedEvidence(EVIDENCE, state);
  assert.ok(record);
  assert.equal(record.relationship_id, "rel-inferred-address");
  assert.equal(record.evidence_type, "SAME_ADDRESS");
  assert.equal(record.inference_level, "INFERRED");

  const details = evidenceDetails(record);
  assert.equal(details.relationshipId, "rel-inferred-address");
  assert.deepEqual(details.sources, ["obs-a1"]);
  assert.deepEqual(details.targets, ["obs-c2"]);
  assert.equal(details.ruleId, "rule.behavioral.repeated_address");

  const reselect = selectRelationship(state, "rel-observed-ip");
  assert.equal(reselect.selectedRelationshipId, "rel-observed-ip");
  assert.equal(selectedEvidence(EVIDENCE, reselect).relationship_id, "rel-observed-ip");

  const deselected = selectRelationship(reselect, "rel-observed-ip");
  assert.equal(deselected.selectedRelationshipId, null);
  assert.equal(selectedEvidence(EVIDENCE, deselected), null);

  const absent = findByRelationshipId(EVIDENCE, "rel-not-present");
  assert.equal(absent, null);
  assert.equal(findByRelationshipId(EVIDENCE, "rel-observed"), null,
    "a prefix must not match");
  assert.equal(findByRelationshipId(EVIDENCE, "REL-OBSERVED-IP"), null,
    "an id lookup must be case sensitive, not a fuzzy match");

  const missingRecord = selectedEvidence(
    EVIDENCE,
    selectRelationship(defaultState(), "rel-not-present")
  );
  assert.equal(missingRecord, null, "selecting an absent id must resolve to nothing");

  return {
    selectedId: state.selectedRelationshipId,
    requestedId: "rel-inferred-address",
    selectedRelationshipId: state.selectedRelationshipId,
    recordEvidenceType: record.evidence_type,
    resolvedRecordMatches: record.relationship_id === state.selectedRelationshipId,
    missingIdResolvesToNull: missingRecord === null,
    prefixIdResolvesToNull: findByRelationshipId(EVIDENCE, "rel-observed") === null,
    idIsCaseSensitive: findByRelationshipId(EVIDENCE, "REL-OBSERVED-IP") === null,
    secondSelection: reselect.selectedRelationshipId,
    toggleOff: deselected.selectedRelationshipId,
    sources: details.sources,
    targets: details.targets
  };
});

scenario("selection is deterministic and marked in the DOM", () => {
  const { container } = freshContainer();
  renderEvidenceChain(container, EVIDENCE, defaultState(), defaultFilters());
  const state = selectRelationship(defaultState(), "rel-correlated-amount");
  const changed = applySelectionState(container, state);

  assert.equal(changed, "rel-correlated-amount");
  const marked = container
    .querySelectorAll(".evidence-item")
    .filter(item => item.getAttribute("aria-pressed") === "true");
  assert.equal(marked.length, 1);
  assert.equal(marked[0].getAttribute("data-relationship-id"), "rel-correlated-amount");

  // Re-running the same state must mark exactly the same record again.
  const reapplied = applySelectionState(container, state);
  assert.equal(reapplied, "rel-correlated-amount");
  const markedAgain = container
    .querySelectorAll(".evidence-item")
    .filter(item => item.getAttribute("aria-pressed") === "true");
  assert.equal(markedAgain.length, 1);
  assert.equal(markedAgain[0].getAttribute("data-relationship-id"), "rel-correlated-amount");

  const cleared = clearSelection(state);
  applySelectionState(container, cleared);
  const stillMarked = container
    .querySelectorAll(".evidence-item")
    .filter(item => item.getAttribute("aria-pressed") === "true");
  assert.equal(stillMarked.length, 0);

  return {
    selectedId: state.selectedRelationshipId,
    markedRelationshipId: marked[0].getAttribute("data-relationship-id"),
    markedCount: 1,
    selectedInDom: marked.length,
    selectedInDomAfterRepeat: markedAgain.length,
    repeatSelectionSameRecord:
      markedAgain[0].getAttribute("data-relationship-id") ===
      marked[0].getAttribute("data-relationship-id"),
    afterClear: 0
  };
});

scenario("source and target identifiers come from the selected record", () => {
  const state = selectRelationship(defaultState(), "rel-observed-ip");
  const details = evidenceDetails(selectedEvidence(EVIDENCE, state));
  assert.deepEqual(details.sources, ["obs-a1"]);
  assert.deepEqual(details.targets, ["obs-b1"]);

  const found = findByRelationshipId(EVIDENCE, "rel-observed-ip");
  assert.equal(found.relationship_id, "rel-observed-ip");
  assert.equal(findByRelationshipId(EVIDENCE, "rel-not-present"), null);
  assert.equal(findByRelationshipId(EVIDENCE, "rel-observed"), null,
    "a prefix must not match");

  const { container, details: detailsPanel } = freshContainer();
  const handle = initEvidenceChain({ evidence: EVIDENCE });
  handle.select("rel-observed-ip");
  const renderedSourceIds = detailsPanel
    .querySelectorAll("[data-field='source_observation_ids'] .mono")
    .map(node => node.getAttribute("title"));
  const renderedTargetIds = detailsPanel
    .querySelectorAll("[data-field='target_observation_ids'] .mono")
    .map(node => node.getAttribute("title"));
  assert.deepEqual(renderedSourceIds, ["obs-a1"]);
  assert.deepEqual(renderedTargetIds, ["obs-b1"]);
  handle.destroy();

  return {
    recordSourceIds: details.sources,
    recordTargetIds: details.targets,
    renderedSourceIds,
    renderedTargetIds,
    exactMatchFound: found.relationship_id,
    prefixLookupReturnsNull: findByRelationshipId(EVIDENCE, "rel-observed") === null
  };
});

scenario("evidence selection synchronises with the graph on exact id", () => {
  const record = findByRelationshipId(EVIDENCE, "rel-observed-txid");
  const sync = syncToGraph(record, EVIDENCE);
  assert.equal(sync.synchronized, true);
  assert.equal(sync.relationshipId, "rel-observed-txid");
  assert.equal(sync.graphRecord.relationship_id, "rel-observed-txid");
  assert.equal(sync.graphRecord.evidence_type, "SAME_TXID");

  return {
    requestedId: "rel-observed-txid",
    graphRelationshipIds: EVIDENCE.map(record => record.relationship_id),
    sync
  };
});

scenario("no graph relationship is fabricated when the id is absent", () => {
  const graphEvidence = EVIDENCE.slice();
  const orphan = {
    relationship_id: "rel-not-in-graph",
    source_observation_ids: ["obs-x"],
    target_observation_ids: ["obs-y"],
    evidence_type: "PROVENANCE_LINK",
    inference_level: "CORRELATED",
    confidence_value: 0.5,
    confidence_basis: "DETERMINISTIC_RULE",
    rationale: "Not present in the graph collection.",
    rule_id: "rule.provenance.link",
    rule_version: "1.0.0",
    provenance_refs: [],
    created_at: null
  };
  const sync = syncToGraph(orphan, graphEvidence);
  assert.equal(sync.synchronized, false);
  assert.equal(sync.graphRecord, null);
  assert.equal(sync.relationshipId, "rel-not-in-graph");
  assert.equal(
    graphEvidence.some(record => record.relationship_id === "rel-not-in-graph"),
    false,
    "synchronisation must not add the record to the graph"
  );

  const emptySync = syncToGraph(null, graphEvidence);
  assert.equal(emptySync.synchronized, false);
  assert.equal(emptySync.relationshipId, null);

  return {
    requestedId: "rel-not-in-graph",
    sync,
    graphRelationshipIds: graphEvidence.map(record => record.relationship_id),
    graphStillExcludesIt: true
  };
});

scenario("source data is unchanged by filtering and selection", () => {
  const before = snapshot(EVIDENCE);
  const container = document.getElementById("evidence-chain-content");
  if (!container) freshContainer();

  filterEvidence(EVIDENCE, { evidenceType: "SAME_IP", inferenceLevel: "OBSERVED" });
  visibleEvidence(EVIDENCE, { evidenceType: "AMOUNT_MATCH", inferenceLevel: ALL_FILTER });
  const state = selectRelationship(
    toggleExpanded(defaultState(), "rel-observed-ip"),
    "rel-observed-ip"
  );
  evidenceDetails(selectedEvidence(EVIDENCE, state));
  syncToGraph(selectedEvidence(EVIDENCE, state), EVIDENCE);

  assert.deepEqual(snapshot(EVIDENCE), before);

  const filteredArray = filterEvidence(EVIDENCE, { evidenceType: "SAME_IP", inferenceLevel: ALL_FILTER });
  filteredArray.push(EVIDENCE[0]);
  assert.equal(EVIDENCE.length, before.length, "the returned array is a copy");

  return {
  evidenceLengthUnchanged: EVIDENCE.length,
  recordsLengthUnchanged: EVIDENCE.length === before.length,
  orderUnchanged:
    EVIDENCE.map(record => record.relationship_id).join(",") ===
    before.map(record => record.relationship_id).join(","),
  recordsUnchanged: true
};
});

/* --------------------------------------------------------------------- */

if (process.env.PHASE3_JSON === "1") {
  process.stdout.write(JSON.stringify({ results, failures }));
} else {
  results.forEach(result => {
    const status = result.ok ? "PASS" : "FAIL";
    process.stdout.write(
      `${status}  ${result.name}${result.ok ? "" : "\n      " + result.error}\n`
    );
  });
  process.stdout.write(`\n${results.length - failures} passed, ${failures} failed\n`);
  if (failures > 0) process.exitCode = 1;
}