/**
 * Behavioural verification of the Step 9B Phase 3 paths section.
 *
 * Each check runs the real `frontend/js/paths.js` and `frontend/js/temporal.js`
 * modules against the DOM they render into, then asserts on observed behaviour:
 * which path is selected, which ordered sequences and transitions are displayed,
 * what a comparison contains, what temporal standing is reported, and whether a
 * path node synchronised with the graph.
 *
 * The fixtures mirror `DashboardPath` exactly and the temporal values come from
 * Step 7's own vocabulary. Where Step 7 declined to compare two timestamps, the
 * test asserts the frontend keeps that answer rather than manufacturing one.
 *
 * Run standalone:  node tests/js/run_paths.mjs
 */

import assert from "node:assert/strict";
import { installDom } from "./minidom.mjs";
import { PATHS, PATH_A, PATH_B, PATH_C, EVIDENCE, GRAPH_OBSERVATION_IDS, snapshot } from "./fixtures.mjs";

installDom();

const pathsModule = await import("../../frontend/js/paths.js");
const temporal = await import("../../frontend/js/temporal.js");

const {
  MAX_COMPARISON_PATHS,
  defaultState,
  findByPathId,
  isSelected,
  selectPath,
  clearSelection,
  selectedPath,
  pathDetails,
  orderedNodes,
  orderedTransitions,
  evidenceSummary,
  isInComparison,
  toggleComparison,
  comparisonPaths,
  buildComparison,
  resetComparison,
  hasComparison,
  syncNodeToGraph,
  syncRelationshipToGraph,
  initPaths,
  renderPaths,
  renderTimeline
} = pathsModule;

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
  container.setAttribute("id", "paths-content");
  document.body.appendChild(container);
  return container;
}

function cardFor(container, pathId) {
  return container
    .querySelectorAll(".path-card")
    .find(card => card.getAttribute("data-path-id") === pathId);
}

function headerFor(container, pathId) {
  const card = cardFor(container, pathId);
  return card ? card.querySelector(".path-card__header") : null;
}

function collectAttributes(root) {
  const names = [];
  const visit = node => {
    if (node.getAttributeNames) names.push(...node.getAttributeNames());
    node.querySelectorAll("*").forEach(visit);
  };
  visit(root);
  return names.join(" ");
}

function bodyVisible(container, pathId) {
  const body = cardFor(container, pathId).querySelector(".path-card__body");
  return !body.hasAttribute("hidden");
}

scenario("selecting a path selects exactly that path", () => {
  const state = selectPath(defaultState(), "path-alpha");
  assert.equal(state.selectedPathId, "path-alpha");
  assert.equal(isSelected(state, "path-alpha"), true);
  assert.equal(isSelected(state, "path-beta"), false);

  const record = selectedPath(PATHS, state);
  assert.ok(record);
  assert.equal(record.path_id, "path-alpha");
  assert.equal(findByPathId(PATHS, "path-missing"), null);
  assert.equal(findByPathId(PATHS, "path-"), null, "a prefix must not match");

  const moved = selectPath(state, "path-beta");
  assert.equal(moved.selectedPathId, "path-beta");
  assert.equal(selectedPath(PATHS, moved).path_id, "path-beta");
  assert.equal(isSelected(state, "path-alpha"), true, "the prior state is unchanged");

  const cleared = clearSelection(moved);
  assert.equal(cleared.selectedPathId, null);
  assert.equal(selectedPath(PATHS, cleared), null);

  return {
    firstSelection: "path-alpha",
    secondSelection: moved.selectedPathId,
    clearedSelection: cleared.selectedPathId
  };
});

scenario("the selected path's nodes and transitions are shown in order", () => {
  const container = freshContainer();
  const handle = initPaths({ paths: PATHS });
  handle.select("path-alpha");

  const card = cardFor(container, "path-alpha");
  assert.ok(card);
  assert.equal(card.getAttribute("data-selected"), "true");

  const renderedNodes = card
    .querySelector(".path-nodes")
    .querySelectorAll(".path-nodes__id")
    .map(node => node.getAttribute("title"));
  assert.deepEqual(renderedNodes, PATH_A.observation_ids);

  const renderedTransitions = card
    .querySelector(".path-transitions")
    .querySelectorAll(".path-transitions__id")
    .map(node => node.getAttribute("title"));
  assert.deepEqual(renderedTransitions, PATH_A.relationship_ids);

  assert.deepEqual(orderedNodes(PATH_A), ["obs-a1", "obs-b1", "obs-c2"]);
  assert.deepEqual(orderedTransitions(PATH_A), ["rel-observed-ip", "rel-correlated-amount"]);

  handle.destroy();
  return {
    pathId: "path-alpha",
    nodes: renderedNodes,
    transitions: renderedTransitions
  };
});

scenario("selecting a second path replaces the first and swaps the body", () => {
  const container = freshContainer();
  const handle = initPaths({ paths: PATHS });

  const headerA = headerFor(container, "path-alpha");
  headerA.click();
  assert.equal(handle.getState().selectedPathId, "path-alpha");
  assert.equal(bodyVisible(container, "path-alpha"), true);
  assert.equal(headerA.getAttribute("aria-expanded"), "true");
  assert.equal(bodyVisible(container, "path-beta"), false);

  headerFor(container, "path-beta").click();
  assert.equal(handle.getState().selectedPathId, "path-beta");
  assert.equal(bodyVisible(container, "path-beta"), true);
  assert.equal(bodyVisible(container, "path-alpha"), false);
  assert.equal(headerFor(container, "path-alpha").getAttribute("aria-expanded"), "false");
  assert.equal(headerFor(container, "path-beta").getAttribute("aria-expanded"), "true");
  assert.equal(cardFor(container, "path-alpha").getAttribute("data-selected"), "false");

  const betaNodes = cardFor(container, "path-beta")
    .querySelector(".path-nodes")
    .querySelectorAll(".path-nodes__id")
    .map(node => node.getAttribute("title"));
  assert.deepEqual(betaNodes, PATH_B.observation_ids);

  handle.destroy();
  return {
    switchedTo: "path-beta",
    ariaAlpha: headerFor(container, "path-alpha").getAttribute("aria-expanded"),
    ariaBeta: headerFor(container, "path-beta").getAttribute("aria-expanded"),
    betaNodes
  };
});

scenario("path details report evidence summary, temporal status and window", () => {
  const details = pathDetails(PATH_A);
  assert.equal(details.pathId, "path-alpha");
  assert.equal(details.hopCount, 2);
  assert.equal(details.nodeCount, 3);
  assert.equal(details.temporalStatus, "COMPARABLE");
  assert.equal(details.temporalWindow.known, true);
  assert.equal(details.temporalWindow.durationSeconds, 1800.0);
  assert.equal(details.evidenceSummary.relationshipCount, 2);
  assert.deepEqual(details.evidenceSummary.evidenceTypes, ["AMOUNT_MATCH", "SAME_IP"]);
  assert.equal(details.evidenceSummary.confidenceMin, 0.75);
  assert.equal(details.evidenceSummary.confidenceMax, 1.0);

  const unknown = pathDetails(PATH_C);
  assert.equal(unknown.temporalStatus, "UNKNOWN");
  assert.equal(unknown.temporalWindow.known, false);
  assert.equal(unknown.temporalWindow.durationSeconds, null);

  assert.equal(evidenceSummary(PATH_A).relationship_count, 2);
  assert.equal(pathDetails(null), null);

  const container = freshContainer();
  const handle = initPaths({ paths: PATHS });
  handle.select("path-alpha");
  const rendered = cardFor(container, "path-alpha");
  assert.equal(
    rendered.querySelector(".path-detail__temporal [data-field='temporal_status'] .path-detail__value").textContent,
    "COMPARABLE"
  );
  assert.equal(
    rendered.querySelector(".path-detail__summary").getAttribute("data-relationship-count"),
    "2"
  );
  handle.destroy();

  return {
    pathA: { status: details.temporalStatus, duration: details.temporalWindow.durationSeconds },
    pathC: { status: unknown.temporalStatus, known: unknown.temporalWindow.known }
  };
});

scenario("two distinct paths can be selected for comparison", () => {
  let state = defaultState();
  state = toggleComparison(state, "path-alpha");
  assert.deepEqual(state.comparePathIds, ["path-alpha"]);
  assert.equal(isInComparison(state, "path-alpha"), true);
  assert.equal(hasComparison(state), true);

  state = toggleComparison(state, "path-beta");
  assert.deepEqual(state.comparePathIds, ["path-alpha", "path-beta"]);
  assert.equal(state.comparePathIds.length, MAX_COMPARISON_PATHS);
  assert.equal(new Set(state.comparePathIds).size, 2);

  const resolved = comparisonPaths(PATHS, state);
  assert.equal(resolved.length, 2);
  assert.equal(resolved[0].path_id, "path-alpha");
  assert.equal(resolved[1].path_id, "path-beta");

  const container = freshContainer();
  const handle = initPaths({ paths: PATHS });
  handle.compare("path-alpha");
  handle.compare("path-beta");
  assert.deepEqual(handle.getComparison().map(path => path.path_id), ["path-alpha", "path-beta"]);
  assert.equal(container.querySelector(".path-compare").getAttribute("data-path-count"), "2");
  assert.equal(handle.getState().selectedPathId, null, "comparing does not select");

  handle.destroy();
  return {
    comparePathIds: state.comparePathIds,
    distinctCount: new Set(state.comparePathIds).size,
    selectionUnaffectedByCompare: true
  };
});

scenario("comparison output preserves both paths' sequences and evidence", () => {
  let state = defaultState();
  state = toggleComparison(state, "path-alpha");
  state = toggleComparison(state, "path-gamma");
  const comparison = buildComparison(PATHS, state);

  assert.ok(comparison);
  assert.deepEqual(comparison.pathIds, ["path-alpha", "path-gamma"]);

  const [alpha, gamma] = comparison.sides;
  assert.equal(alpha.pathId, "path-alpha");
  assert.equal(gamma.pathId, "path-gamma");

  assert.deepEqual(alpha.nodes, PATH_A.observation_ids);
  assert.deepEqual(gamma.nodes, PATH_C.observation_ids);
  assert.deepEqual(alpha.transitions, PATH_A.relationship_ids);
  assert.deepEqual(gamma.transitions, PATH_C.relationship_ids);

  assert.deepEqual(alpha.evidenceSummary.evidenceTypes, ["AMOUNT_MATCH", "SAME_IP"]);
  assert.deepEqual(gamma.evidenceSummary.evidenceTypes, ["TEMPORAL_ADJACENCY"]);
  assert.equal(alpha.temporalStatus, "COMPARABLE");
  assert.equal(gamma.temporalStatus, "UNKNOWN");

  assert.equal(alpha.timeline.length, 2);
  assert.equal(alpha.timeline[0].relation, "AFTER");
  assert.equal(alpha.timeline[0].elapsedSeconds, 3600.0);
  assert.equal(alpha.timeline[1].relation, "BEFORE");
  assert.equal(alpha.timeline[1].elapsedSeconds, -1800.0);
  assert.equal(gamma.timeline[0].relation, "UNKNOWN");
  assert.equal(gamma.timeline[0].elapsedSeconds, null);
  assert.equal(gamma.timeline[0].comparable, false);

  assert.equal(comparison.sharedObservations.length, 0);
  assert.equal(comparison.sharedRelationships.length, 0);

  return {
    pathIds: comparison.pathIds,
    alphaNodes: alpha.nodes,
    gammaNodes: gamma.nodes,
    alphaRelations: alpha.timeline.map(step => step.relation),
    gammaRelation: gamma.timeline[0].relation
  };
});

scenario("comparison produces no path-level score or ranking", () => {
  let state = defaultState();
  state = toggleComparison(state, "path-alpha");
  state = toggleComparison(state, "path-beta");
  const comparison = buildComparison(PATHS, state);
  const serialised = JSON.stringify(comparison).toLowerCase();

  ["score", "risk", "suspicious", "probability", "guilt", "malicious", "attribution", "priority"]
    .forEach(term => {
      assert.equal(serialised.indexOf(term), -1, "comparison must not carry " + term);
    });

  assert.equal(comparison.sides[0].pathId, "path-alpha", "selection order is preserved");
  assert.equal(comparison.sides[1].pathId, "path-beta");

  const reversed = buildComparison(PATHS, toggleComparison(toggleComparison(defaultState(), "path-beta"), "path-alpha"));
  assert.deepEqual(reversed.pathIds, ["path-beta", "path-alpha"], "order follows selection, not merit");

  return { pathIds: comparison.pathIds, reversedPathIds: reversed.pathIds, noScoreFields: true };
});

scenario("a comparison needs two distinct paths", () => {
  assert.equal(buildComparison(PATHS, defaultState()), null);

  const single = toggleComparison(defaultState(), "path-alpha");
  assert.equal(buildComparison(PATHS, single), null);

  // Toggling the same path again removes it rather than adding a duplicate, so
  // a set can never hold the same path twice.
  const same = toggleComparison(single, "path-alpha");
  assert.deepEqual(same.comparePathIds, []);
  assert.equal(new Set(same.comparePathIds).size, same.comparePathIds.length);
  assert.equal(buildComparison(PATHS, same), null);

  const twice = toggleComparison(toggleComparison(defaultState(), "path-alpha"), "path-alpha");
  assert.deepEqual(twice.comparePathIds, []);

  return { emptyComparisonIsNull: true, duplicateSelectionRemoves: true };
});

scenario("a third path is refused rather than silently replacing one", () => {
  let state = defaultState();
  state = toggleComparison(state, "path-alpha");
  state = toggleComparison(state, "path-beta");
  state = toggleComparison(state, "path-gamma");
  assert.deepEqual(state.comparePathIds, ["path-alpha", "path-beta"]);
  assert.equal(state.comparePathIds.length, MAX_COMPARISON_PATHS);

  return { comparePathIds: state.comparePathIds, thirdRefused: true };
});

scenario("clearing the comparison empties it and restores normal browsing", () => {
  const container = freshContainer();
  const handle = initPaths({ paths: PATHS });

  handle.select("path-alpha");
  handle.compare("path-alpha");
  handle.compare("path-beta");
  assert.equal(container.querySelector(".path-compare") !== null, true);
  assert.equal(container.querySelectorAll(".path-card").length, 0);
  assert.deepEqual(handle.getComparison().map(path => path.path_id), ["path-alpha", "path-beta"]);

  const resetButton = container.querySelector("[data-action='reset-comparison']");
  assert.ok(resetButton);
  assert.equal(resetButton.getAttribute("aria-label"), "Clear path comparison");
  resetButton.click();

  assert.deepEqual(handle.getState().comparePathIds, []);
  assert.equal(hasComparison(handle.getState()), false);
  assert.equal(handle.getComparison().length, 0);
  assert.equal(handle.getComparisonView(), null);
  assert.equal(container.querySelector(".path-compare"), null);
  assert.equal(container.querySelectorAll(".path-card").length, PATHS.length);
  assert.equal(handle.getState().selectedPathId, "path-alpha", "browsing state survives");
  assert.equal(cardFor(container, "path-alpha").getAttribute("data-selected"), "true");

  const pure = resetComparison(toggleComparison(toggleComparison(defaultState(), "path-alpha"), "path-beta"));
  assert.deepEqual(pure.comparePathIds, []);
  assert.equal(pure.selectedPathId, null);

  handle.destroy();
  return {
    comparePathIdsAfterReset: handle.getState().comparePathIds,
    cardsVisibleAfterReset: PATHS.length,
    selectionPreserved: handle.getState().selectedPathId
  };
});

scenario("keyboard selection activates a path and updates aria-expanded", () => {
  const container = freshContainer();
  const handle = initPaths({ paths: PATHS });
  const header = headerFor(container, "path-beta");

  assert.equal(header.getAttribute("aria-expanded"), "false");
  header.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
  assert.equal(handle.getState().selectedPathId, "path-beta");
  assert.equal(headerFor(container, "path-beta").getAttribute("aria-expanded"), "true");
  assert.equal(bodyVisible(container, "path-beta"), true);

  headerFor(container, "path-gamma").dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
  assert.equal(handle.getState().selectedPathId, "path-gamma");
  assert.equal(headerFor(container, "path-beta").getAttribute("aria-expanded"), "false");

  headerFor(container, "path-alpha").dispatchEvent(new KeyboardEvent("keydown", { key: "q" }));
  assert.equal(handle.getState().selectedPathId, "path-gamma", "a non-activation key must not select");

  handle.destroy();
  return {
    enterSelects: "path-beta",
    spaceSelects: "path-gamma",
    nonActivationKeyIgnored: true
  };
});

scenario("comparison controls are labelled and keyboard operable", () => {
  const container = freshContainer();
  const handle = initPaths({ paths: PATHS });

  const button = cardFor(container, "path-alpha").querySelector(".path-card__compare");
  assert.equal(button.getAttribute("aria-label"), "Compare path path-alpha");
  assert.equal(button.getAttribute("aria-pressed"), "false");

  button.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
  assert.deepEqual(handle.getComparison().map(path => path.path_id), ["path-alpha"]);
  const comparedButton = cardFor(container, "path-alpha").querySelector(".path-card__compare");
  assert.equal(comparedButton.getAttribute("aria-pressed"), "true");
  assert.equal(comparedButton.textContent, "In comparison");

  const second = cardFor(container, "path-beta").querySelector(".path-card__compare");
  second.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
  assert.deepEqual(handle.getComparison().map(path => path.path_id), ["path-alpha", "path-beta"]);

  const reset = container.querySelector("[data-action='reset-comparison']");
  reset.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
  assert.equal(handle.getState().comparePathIds.length, 0);

  handle.destroy();
  return {
    compareButtonLabel: "Compare path path-alpha",
    ariaPressedAfterCompare: true,
    resetViaKeyboard: true
  };
});

scenario("comparable aware timestamps keep chronological order", () => {
  const timeline = temporal.buildTimeline(PATH_A);
  assert.equal(timeline.status, "COMPARABLE");
  assert.equal(timeline.chronologicallyOrdered, true);
  assert.equal(timeline.steps.length, 2);
  assert.equal(timeline.steps[0].relation, "AFTER");
  assert.equal(timeline.steps[0].elapsedSeconds, 3600.0);
  assert.equal(timeline.steps[1].relation, "BEFORE");
  assert.equal(timeline.steps[1].elapsedSeconds, -1800.0);
  assert.ok(timeline.steps.every(step => step.comparable));
  assert.ok(timeline.steps.every(step => step.establishedOrder));

  const rendered = renderTimeline(PATH_A);
  assert.equal(rendered.getAttribute("data-chronological"), "true");
  assert.equal(rendered.querySelector(".timeline__note"), null);
  const rows = rendered.querySelectorAll(".timeline__step");
  assert.deepEqual(
    rows.map(row => row.getAttribute("data-relation")),
    ["AFTER", "BEFORE"]
  );
  assert.deepEqual(
    rows.map(row => row.getAttribute("data-elapsed")),
    ["3600", "-1800"]
  );

  return {
    status: timeline.status,
    relations: timeline.steps.map(step => step.relation),
    elapsed: timeline.steps.map(step => step.elapsedSeconds)
  };
});

scenario("NOT_COMPARABLE stays NOT_COMPARABLE with no elapsed time", () => {
  const timeline = temporal.buildTimeline(PATH_B);
  assert.equal(timeline.status, "NOT_COMPARABLE");
  assert.equal(timeline.chronologicallyOrdered, false);
  const step = timeline.steps[0];
  assert.equal(step.relation, "NOT_COMPARABLE");
  assert.equal(step.standing, "NOT_COMPARABLE");
  assert.equal(step.comparable, false);
  assert.equal(step.establishedOrder, false);
  assert.equal(step.elapsedSeconds, null);

  assert.equal(temporal.pathTemporalWindow(PATH_B).known, false);
  assert.equal(temporal.pathTemporalWindow(PATH_B).durationSeconds, null);

  const rendered = renderTimeline(PATH_B);
  assert.equal(rendered.getAttribute("data-chronological"), "false");
  const row = rendered.querySelector(".timeline__step");
  assert.equal(row.getAttribute("data-relation"), "NOT_COMPARABLE");
  assert.equal(row.getAttribute("data-comparable"), "false");
  assert.equal(row.getAttribute("data-elapsed"), "none");
  assert.ok(rendered.querySelector(".timeline__note"), "ordering must be stated as unavailable");

  return {
    status: timeline.status,
    relation: step.relation,
    elapsedSeconds: step.elapsedSeconds,
    windowKnown: temporal.pathTemporalWindow(PATH_B).known
  };
});

scenario("missing timestamps stay UNKNOWN", () => {
  const timeline = temporal.buildTimeline(PATH_C);
  assert.equal(timeline.status, "UNKNOWN");
  assert.equal(timeline.chronologicallyOrdered, false);
  const step = timeline.steps[0];
  assert.equal(step.relation, "UNKNOWN");
  assert.equal(step.comparable, false);
  assert.equal(step.elapsedSeconds, null);
  assert.equal(temporal.pathTemporalWindow(PATH_C).known, false);

  const noFlow = temporal.buildTimeline({
    path_id: "path-empty",
    temporal_status: "UNKNOWN",
    temporal_flow: [],
    relationship_ids: []
  });
  assert.equal(noFlow.steps.length, 0);
  assert.equal(noFlow.chronologicallyOrdered, false);

  return {
    status: timeline.status,
    relation: step.relation,
    elapsedSeconds: step.elapsedSeconds,
    emptyFlowIsNotOrdered: noFlow.chronologicallyOrdered === false
  };
});

scenario("a naive timestamp is not silently read as UTC", () => {
  const naiveOnly = temporal.parseTemporalFlowEntry(
    "obs-n1 -> obs-n2 (SAME_IP, AFTER, elapsed=60.0s)"
  );
  assert.equal(naiveOnly.relation, "AFTER");
  assert.equal(naiveOnly.elapsedSeconds, 60.0);
  assert.equal(
    Object.prototype.hasOwnProperty.call(naiveOnly, "normalizedToUtc"),
    false,
    "no UTC normalisation field may appear"
  );

  // An aware/naive pair is what Step 7 reports NOT_COMPARABLE for, and the
  // frontend must carry that through rather than resolving it.
  const mixed = temporal.parseTemporalFlowEntry(
    "obs-n1 -> obs-a1 (SAME_IP, NOT_COMPARABLE, elapsed=Nones)"
  );
  assert.equal(mixed.relation, "NOT_COMPARABLE");
  assert.equal(mixed.elapsedSeconds, null);
  assert.equal(temporal.isComparableStep(mixed), false);

  const path = {
    path_id: "path-mixed",
    temporal_status: "PARTIALLY_COMPARABLE",
    duration_seconds: null,
    relationship_ids: ["rel-a", "rel-b"],
    temporal_flow: [
      "obs-a1 -> obs-n1 (SAME_IP, NOT_COMPARABLE, elapsed=Nones)",
      "obs-n1 -> obs-a2 (SAME_IP, AFTER, elapsed=120.0s)"
    ]
  };
  const timeline = temporal.buildTimeline(path);
  assert.equal(timeline.status, "PARTIALLY_COMPARABLE");
  assert.equal(timeline.chronologicallyOrdered, false, "one incomparable step breaks the sequence");
  assert.deepEqual(
    timeline.steps.map(step => step.comparable),
    [false, true]
  );
  assert.equal(timeline.steps[0].elapsedSeconds, null);
  assert.equal(timeline.steps[1].elapsedSeconds, 120.0);

  return {
    naiveRelation: naiveOnly.relation,
    mixedRelation: mixed.relation,
    partialStatus: timeline.status,
    comparability: timeline.steps.map(step => step.comparable)
  };
});

scenario("no temporal ordering is fabricated for an unparsable entry", () => {
  const unparsable = temporal.parseTemporalFlowEntry("this is not a flow entry");
  assert.equal(unparsable, null);
  assert.equal(temporal.isComparableStep(unparsable), false);
  assert.equal(temporal.describeTemporalStanding(unparsable), "UNKNOWN");

  const invented = temporal.parseTemporalFlowEntry(
    "obs-a -> obs-b (SAME_IP, LATER, elapsed=5s)"
  );
  assert.equal(invented, null, "a relation outside Step 7's vocabulary must not parse");

  const path = {
    path_id: "path-garbled",
    temporal_status: "COMPARABLE",
    duration_seconds: 10,
    relationship_ids: ["rel-x"],
    temporal_flow: ["garbled"]
  };
  const timeline = temporal.buildTimeline(path);
  assert.equal(timeline.steps.length, 1);
  assert.equal(timeline.steps[0].unparsed, true);
  assert.equal(timeline.steps[0].relation, null);
  assert.equal(timeline.steps[0].elapsedSeconds, null);
  assert.equal(timeline.chronologicallyOrdered, false);

  return {
    unparsableIsNull: unparsable === null,
    inventedRelationRejected: temporal.parseTemporalFlowEntry(
      "obs-a -> obs-b (SAME_IP, LATER, elapsed=5s)"
    ) === null,
    garbledStepStanding: timeline.steps[0].standing
  };
});

scenario("a path node synchronises with the graph on exact observation_id", () => {
  const present = syncNodeToGraph("obs-c1", GRAPH_OBSERVATION_IDS);
  assert.equal(present.synchronized, true);
  assert.equal(present.graphNodeId, "obs-c1");

  const absent = syncNodeToGraph("obs-not-in-graph", GRAPH_OBSERVATION_IDS);
  assert.equal(absent.synchronized, false);
  assert.equal(absent.graphNodeId, null, "no graph node may be invented");

  const truncated = syncNodeToGraph("obs-c", GRAPH_OBSERVATION_IDS);
  assert.equal(truncated.synchronized, false, "a prefix must not match");

  const byPosition = syncNodeToGraph("0", GRAPH_OBSERVATION_IDS);
  assert.equal(byPosition.synchronized, false, "an index must not match");

  const relationship = syncRelationshipToGraph("rel-observed-ip", EVIDENCE);
  assert.equal(relationship.synchronized, true);
  assert.equal(syncRelationshipToGraph("rel-absent", EVIDENCE).synchronized, false);
  assert.equal(syncRelationshipToGraph("rel-absent", EVIDENCE).graphRecord, null);

  return {
    presentNode: present.graphNodeId,
    absentNode: absent.graphNodeId,
    prefixRejected: truncated.synchronized === false,
    indexRejected: byPosition.synchronized === false,
    relationshipSync: relationship.synchronized
  };
});

scenario("path data is unchanged by selection, comparison and rendering", () => {
  const before = snapshot(PATHS);
  const container = freshContainer();
  let state = defaultState();
  state = selectPath(state, "path-alpha");
  state = toggleComparison(toggleComparison(state, "path-beta"), "path-gamma");
  orderedNodes(PATH_A);
  orderedTransitions(PATH_A);
  pathDetails(PATH_A);
  buildComparison(PATHS, state);
  renderPaths(container, PATHS, state);
  temporal.buildTimeline(PATH_A);
  renderTimeline(PATH_C);

  assert.deepEqual(snapshot(PATHS), before);

  const nodes = orderedNodes(PATH_A);
  nodes.push("obs-injected");
  assert.equal(orderedNodes(PATH_A).length, PATH_A.observation_ids.length);

  return { pathsLengthUnchanged: PATHS.length, recordsUnchanged: true };
});

scenario("comparison rendering omits any score and keeps selection order", () => {
  const container = freshContainer();
  let state = toggleComparison(defaultState(), "path-gamma");
  state = toggleComparison(state, "path-alpha");
  const comparison = buildComparison(PATHS, state);
  renderPaths(container, PATHS, state);

  const panel = container.querySelector(".path-compare");
  assert.ok(panel);
  const grid = panel.querySelector(".path-compare__grid");
  assert.equal(grid.getAttribute("data-path-ids"), "path-gamma path-alpha");
  const sides = grid.querySelectorAll(".path-compare__side");
  assert.equal(sides.length, 2);
  assert.deepEqual(
    sides.map(side => side.getAttribute("data-path-id")),
    ["path-gamma", "path-alpha"]
  );
  // Scan attributes and labels rather than prose: the panel prints a note
  // explaining that no score is produced, and that wording must not be mistaken
  // for a score field.
  const attributes = collectAttributes(panel).toLowerCase();
  ["score", "risk", "suspicious", "probability"].forEach(term => {
    assert.equal(attributes.indexOf(term), -1, "comparison must not carry a " + term + " attribute");
  });
  const labels = panel
    .querySelectorAll(".path-detail__label")
    .map(node => node.textContent.toLowerCase());
  ["score", "risk", "probability", "guilt", "malicious", "attribution", "priority"].forEach(term => {
    assert.equal(
      labels.some(label => label.indexOf(term) !== -1),
      false,
      "comparison must not display a " + term + " field"
    );
  });
  assert.equal(labels.length > 0, true, "labels should exist so the check is meaningful");
  // Confidence appears only as recorded per-edge bounds, never as a path verdict.
  assert.equal(
    labels.some(label => label.indexOf("edge confidence bounds") !== -1),
    true
  );
  assert.equal(
    labels
      .filter(label => label.indexOf("confidence") !== -1)
      .every(label => label.indexOf("edge confidence bounds") !== -1),
    true,
    "confidence may only appear as recorded per-edge bounds"
  );
  assert.equal(labels.length, 18, "nine fields per side, for two sides");
  const timelines = panel.querySelectorAll(".path-compare__timeline");
  assert.equal(timelines[0].getAttribute("data-chronological"), "false");
  assert.equal(timelines[1].getAttribute("data-chronological"), "true");

  return {
    renderedOrder: comparison.pathIds,
    renderedSides: sides.length,
    noScoreInHtml: true
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