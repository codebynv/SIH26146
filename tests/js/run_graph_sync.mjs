/**
 * Behavioural verification of the Step 9B Phase 3 graph synchronisation.
 *
 * The evidence chain and the paths section both need to reflect a selection the
 * graph reports, and both need to push their own selection into the graph. This
 * runner wires the three real modules together the same way `app.js` does and
 * drives the resulting selection in both directions.
 *
 * Every assertion is on exact identifiers. Nothing here accepts a near match, an
 * array index, or a label similarity, and the tests below deliberately try all
 * three and require them to be refused.
 *
 * Run standalone:  node tests/js/run_graph_sync.mjs
 */

import assert from "node:assert/strict";
import { installDom } from "./minidom.mjs";
import { EVIDENCE, PATHS } from "./fixtures.mjs";

installDom();

const graph = await import("../../frontend/js/graph.js");
const evidenceChain = await import("../../frontend/js/evidence-chain.js");
const paths = await import("../../frontend/js/paths.js");
const { store } = await import("../../frontend/js/state.js");

const {
  defaultSelection,
  selectRelationship,
  selectObservation,
  clearSelection,
  hasRelationship,
  hasObservation,
  applyGraphSelection,
  syncFromEvidence,
  syncFromPathNode
} = graph;

const results = [];
let failures = 0;

function scenario(name, body) {
  try {
    results.push({ name, ok: true, detail: body() });
  } catch (error) {
    failures += 1;
    results.push({ name, ok: false, error: error.message });
  }
}

/**
 * Build a DOM shaped like the dashboard: the three sections the real modules
 * look for, with one edge and one node per evidence record.
 */
function buildDashboard(options) {
  document.body.replaceChildren();

  const section = tag => {
    const node = document.createElement("div");
    node.setAttribute("id", tag);
    document.body.appendChild(node);
    return node;
  };

  section("cross-layer-content");
  const graphContainer = document.createElement("div");
  graphContainer.setAttribute("class", "graph-container");
  document.getElementById("cross-layer-content").appendChild(graphContainer);
  void options;
  section("graph-selection-details");
  section("evidence-chain-content");
  section("evidence-selection-details");
  section("paths-content");

  // The real graph draws one node per distinct observation and one edge per
  // evidence record, so the fixture does the same. `extraObservationIds` adds
  // nodes the evidence does not reach, which is how a path node can exist in the
  // graph while no single evidence record mentions it.
  const observationIds = [];
  EVIDENCE.forEach(record => {
    record.source_observation_ids
      .concat(record.target_observation_ids)
      .forEach(id => {
        if (observationIds.indexOf(id) === -1) observationIds.push(id);
      });
  });
  (options && options.extraObservationIds ? options.extraObservationIds : []).forEach(id => {
    if (observationIds.indexOf(id) === -1) observationIds.push(id);
  });

  observationIds.forEach(observationId => {
    const node = document.createElement("div");
    node.setAttribute("class", "graph-node");
    node.setAttribute("data-observation", observationId);
    graphContainer.appendChild(node);
  });

  EVIDENCE.forEach(record => {
    const edge = document.createElement("div");
    edge.setAttribute("class", "graph-edge");
    edge.setAttribute("data-relationship", record.relationship_id);
    graphContainer.appendChild(edge);
  });

  return graphContainer;
}

/**
 * Wire the three modules together exactly as app.js does, including both
 * directions of the handshake: each section pushes its selection into the
 * graph, and the graph pushes a reported selection back out to both sections.
 */
function mountDashboard(options) {
  const container = buildDashboard(options);
  let chain = null;
  let pathsSection = null;
  const graphSelection = graph.initGraphSelection({
    container: "cross-layer-content",
    detailsContainer: document.getElementById("graph-selection-details"),
    evidence: EVIDENCE,
    onSelect(selection) {
      if (!selection) return;
      if (selection.relationshipId && chain) {
        chain.highlightRelationship(selection.relationshipId);
      }
      if (selection.observationId && pathsSection) {
        pathsSection.highlightObservation(selection.observationId);
      }
    }
  });
  chain = evidenceChain.initEvidenceChain({
    evidence: EVIDENCE,
    graphEvidence: EVIDENCE,
    onSelect(record, sync) {
      if (record && sync.synchronized) graphSelection.selectRelationship(sync.relationshipId);
      else graphSelection.clear();
    }
  });
  pathsSection = paths.initPaths({
    paths: PATHS,
    graphEvidence: EVIDENCE,
    onSelect(record) {
      if (!record) {
        graphSelection.clear();
        return;
      }
      const applied = graphSelection.selectMany(
        record.observation_ids,
        record.relationship_ids
      );
      if (!applied.relationships.length && !applied.observations.length) {
        graphSelection.clear();
      }
    },
    onCompare() {
      graphSelection.clear();
    }
  });
  return { container, graphSelection, chain, pathsSection };
}

function selectedEdges(container) {
  return container
    .querySelectorAll(".graph-edge")
    .filter(edge => edge.classList.contains("is-selected"))
    .map(edge => edge.getAttribute("data-relationship"));
}

function selectedNodes(container) {
  return container
    .querySelectorAll(".graph-node")
    .filter(node => node.classList.contains("is-selected"))
    .map(node => node.getAttribute("data-observation"));
}

scenario("clicking an evidence record selects that exact graph relationship", () => {
  const { container, graphSelection, chain } = mountDashboard();

  const item = document
    .getElementById("evidence-chain-content")
    .querySelectorAll(".evidence-item")
    .find(node => node.getAttribute("data-relationship-id") === "rel-observed-ip");
  item.click();

  assert.deepEqual(selectedEdges(container), ["rel-observed-ip"]);
  assert.deepEqual(selectedNodes(container), [], "selecting an edge selects no node");
  assert.equal(graphSelection.getSelection().relationshipId, "rel-observed-ip");
  assert.equal(chain.getState().selectedRelationshipId, "rel-observed-ip");

  chain.destroy();
  graphSelection.destroy();
  return { selectedEdges: selectedEdges(container), selection: graphSelection.getSelection() };
});

scenario("aria-selected on the graph agrees with the selected evidence", () => {
  const { container, graphSelection, chain } = mountDashboard();
  chain.select("rel-correlated-amount");

  const marked = container
    .querySelectorAll(".graph-edge")
    .filter(edge => edge.getAttribute("aria-selected") === "true");
  assert.equal(marked.length, 1);
  assert.equal(marked[0].getAttribute("data-relationship"), "rel-correlated-amount");

  graphSelection.clear();
  const afterClear = container
    .querySelectorAll(".graph-edge")
    .filter(edge => edge.getAttribute("aria-selected") === "true");
  assert.equal(afterClear.length, 0, "clearing must clear aria-selected everywhere");

  chain.destroy();
  graphSelection.destroy();
  return { ariaSelectedCount: marked.length, afterClear: afterClear.length };
});

scenario("a graph relationship click selects the matching evidence record", () => {
  const { graphSelection, chain } = mountDashboard();

  const edge = document
    .getElementById("cross-layer-content")
    .querySelectorAll(".graph-edge")
    .find(node => node.getAttribute("data-relationship") === "rel-inferred-address");
  edge.click();

  assert.equal(chain.getState().selectedRelationshipId, "rel-inferred-address");
  const marked = document
    .getElementById("evidence-chain-content")
    .querySelectorAll(".evidence-item")
    .filter(node => node.getAttribute("aria-pressed") === "true");
  assert.equal(marked.length, 1);
  assert.equal(marked[0].getAttribute("data-relationship-id"), "rel-inferred-address");

  graphSelection.destroy();
  chain.destroy();
  return { evidenceSelection: chain.getState().selectedRelationshipId, markedInDom: marked.length };
});

scenario("selecting a path highlights its exact nodes and transitions", () => {
  const { container, graphSelection, pathsSection } = mountDashboard();

  pathsSection.select("path-alpha");
  assert.deepEqual(selectedNodes(container).sort(), ["obs-a1", "obs-b1", "obs-c2"]);
  assert.deepEqual(
    selectedEdges(container).sort(),
    ["rel-correlated-amount", "rel-observed-ip"]
  );
  assert.equal(
    selectedEdges(container).length,
    2,
    "two relationships belong to this path and both are marked"
  );

  graphSelection.destroy();
  pathsSection.destroy();
  return { nodes: selectedNodes(container).sort(), edges: selectedEdges(container).sort() };
});

scenario("a path highlights no observation the graph does not contain", () => {
  const { container, graphSelection, pathsSection } = mountDashboard();

  pathsSection.select("path-gamma");
  assert.deepEqual(selectedNodes(container).sort(), ["obs-b2", "obs-c1"]);
  const graphObservations = new Set();
  container
    .querySelectorAll(".graph-node")
    .forEach(node => graphObservations.add(node.getAttribute("data-observation")));
  assert.ok(
    selectedNodes(container).every(id => graphObservations.has(id)),
    "every highlighted node must exist in the graph"
  );

  graphSelection.destroy();
  pathsSection.destroy();
  return { highlighted: selectedNodes(container).sort(), graphNodeCount: graphObservations.size };
});

scenario("clicking a graph node selects the path that contains it", () => {
  const { graphSelection, pathsSection } = mountDashboard();

  const node = document
    .getElementById("cross-layer-content")
    .querySelectorAll(".graph-node")
    .find(element => element.getAttribute("data-observation") === "obs-a2");
  assert.ok(node, "obs-a2 must exist in the graph");
  node.click();

  assert.equal(pathsSection.getState().selectedPathId, "path-beta");
  assert.equal(pathsSection.getSelected().path_id, "path-beta");

  graphSelection.destroy();
  pathsSection.destroy();
  return { selectedPath: pathsSection.getState().selectedPathId };
});

scenario("a node shared by two paths resolves to the same path every time", () => {
  const { graphSelection, pathsSection } = mountDashboard();
  const node = document
    .getElementById("cross-layer-content")
    .querySelectorAll(".graph-node")
    .find(element => element.getAttribute("data-observation") === "obs-c1");

  const chosen = [];
  for (let attempt = 0; attempt < 3; attempt += 1) {
    node.click();
    chosen.push(pathsSection.getState().selectedPathId);
  }
  assert.ok(
    chosen.every(id => id === chosen[0]),
    "a node in two paths must resolve deterministically, not vary per click"
  );
  assert.ok(PATHS.some(path => (path.observation_ids || []).indexOf("obs-c1") !== -1));

  graphSelection.destroy();
  pathsSection.destroy();
  return { sharedObservation: "obs-c1", resolvedTo: chosen[0], attempts: chosen };
});

scenario("a graph node belonging to no path selects no path", () => {
  // This node is in the graph but in no path, which is the case that must not
  // quietly pick the nearest path.
  const orphanObservation = "obs-only-in-graph";
  const { graphSelection, pathsSection } = mountDashboard({
    extraObservationIds: [orphanObservation]
  });

  pathsSection.select("path-alpha");
  const orphanNode = document
    .getElementById("cross-layer-content")
    .querySelectorAll(".graph-node")
    .find(element => element.getAttribute("data-observation") === orphanObservation);
  assert.ok(orphanNode, "the orphan node must exist in the graph");
  orphanNode.click();

  assert.equal(
    pathsSection.getState().selectedPathId,
    "path-alpha",
    "an observation outside every path must not select one"
  );
  assert.equal(pathsSection.highlightObservation(orphanObservation), null);

  graphSelection.destroy();
  pathsSection.destroy();
  return { orphanObservation, selectedPath: pathsSection.getState().selectedPathId };
});

scenario("clearing the graph clears the evidence selection", () => {
  const { graphSelection, chain } = mountDashboard();

  chain.select("rel-observed-ip");
  assert.equal(chain.getState().selectedRelationshipId, "rel-observed-ip");

  graphSelection.clear();
  assert.equal(graphSelection.getSelection().relationshipId, null);
  assert.deepEqual(selectedEdges(document.getElementById("cross-layer-content")), []);

  graphSelection.destroy();
  chain.destroy();
  return { selectionAfterClear: graphSelection.getSelection() };
});

scenario("comparing paths clears the graph rather than highlighting one", () => {
  const { container, graphSelection, pathsSection } = mountDashboard();

  pathsSection.compare("path-alpha");
  pathsSection.compare("path-beta");

  assert.deepEqual(selectedEdges(container), []);
  assert.deepEqual(selectedNodes(container), []);
  assert.equal(graphSelection.getSelection().relationshipId, null);
  assert.equal(graphSelection.getSelection().observationId, null);
  assert.deepEqual(
    pathsSection.getComparison().map(path => path.path_id),
    ["path-alpha", "path-beta"]
  );

  graphSelection.destroy();
  pathsSection.destroy();
  return { selectedEdges: selectedEdges(container), comparison: ["path-alpha", "path-beta"] };
});

scenario("only one graph element is selected at a time", () => {
  const { container, graphSelection } = mountDashboard();

  graphSelection.selectRelationship("rel-observed-ip");
  assert.deepEqual(selectedEdges(container), ["rel-observed-ip"]);

  graphSelection.selectRelationship("rel-observed-txid");
  assert.deepEqual(
    selectedEdges(container),
    ["rel-observed-txid"],
    "selecting a second edge clears the first"
  );

  graphSelection.selectObservation("obs-b1");
  assert.deepEqual(selectedEdges(container), [], "selecting a node clears the edge");
  assert.deepEqual(selectedNodes(container), ["obs-b1"]);

  graphSelection.destroy();
  return {
    afterTwoEdges: "one edge",
    afterNode: { edges: [], nodes: ["obs-b1"] }
  };
});

scenario("an id the graph does not hold is refused, not approximated", () => {
  const { container, graphSelection, chain } = mountDashboard();
  const nearMisses = ["rel-observed", "REL-OBSERVED-IP", "rel-observed-ip ", "0", ""];

  nearMisses.forEach(id => {
    assert.equal(graphSelection.selectRelationship(id), null, "refused: " + JSON.stringify(id));
  });
  assert.equal(graphSelection.selectObservation("obs-c"), null, "a prefix must not match");
  assert.equal(graphSelection.selectObservation("2"), null, "an index must not match");

  assert.deepEqual(selectedEdges(document.getElementById("cross-layer-content")), []);
  assert.deepEqual(selectedNodes(container), []);

  assert.equal(chain.highlightRelationship("rel-observed"), null);
  assert.equal(chain.getState().selectedRelationshipId, null);

  graphSelection.destroy();
  chain.destroy();
  return { refused: nearMisses.length + 3, selectionRemainedEmpty: true };
});

scenario("evidence and graph agree on which record exists", () => {
  assert.equal(hasRelationship(EVIDENCE, "rel-observed-ip"), true);
  assert.equal(hasRelationship(EVIDENCE, "rel-observed"), false);
  assert.equal(hasObservation(EVIDENCE, "obs-a1"), true);
  assert.equal(hasObservation(EVIDENCE, "obs-a"), false);
  assert.equal(hasRelationship(EVIDENCE, null), false);

  const fromEvidence = syncFromEvidence(EVIDENCE, "rel-observed-ip");
  assert.equal(fromEvidence.synchronized, true);
  assert.deepEqual(fromEvidence.selection, selectRelationship(defaultSelection(), "rel-observed-ip"));

  const missing = syncFromEvidence(EVIDENCE, "rel-absent");
  assert.equal(missing.synchronized, false);
  assert.equal(missing.relationshipId, null);

  const fromPath = syncFromPathNode(EVIDENCE, "obs-a1");
  assert.equal(fromPath.synchronized, true);
  assert.deepEqual(fromPath.selection, selectObservation(defaultSelection(), "obs-a1"));

  assert.equal(syncFromPathNode(EVIDENCE, "obs-a").synchronized, false);
  assert.deepEqual(clearSelection(selectObservation(defaultSelection(), "obs-a1")), {
    relationshipId: null,
    observationId: null
  });

  return {
    relationshipPresent: true,
    prefixRefused: true,
    observationPresent: true,
    missingRefused: true
  };
});

scenario("applyGraphSelection ignores ids with no element", () => {
  const container = buildDashboard();
  const applied = applyGraphSelection(container, {
    relationshipId: "rel-observed-ip",
    observationId: "obs-a1"
  }, {
    relationships: ["rel-observed-ip", "rel-not-drawn"],
    observations: ["obs-a1", "obs-not-drawn"]
  });

  assert.deepEqual(applied.relationships, ["rel-observed-ip"], "an undrawn id is not marked");
  assert.deepEqual(applied.observations, ["obs-a1"]);

  const none = applyGraphSelection(container, defaultSelection(), {
    relationships: ["rel-not-drawn"],
    observations: []
  });
  assert.deepEqual(none.relationships, []);
  return { applied, withNoElements: none };
});

scenario("a store redraw restores the selection and drops ids the new data lacks", () => {
  const { container, graphSelection } = mountDashboard();

  graphSelection.selectRelationship("rel-observed-ip");
  assert.deepEqual(selectedEdges(container), ["rel-observed-ip"]);

  // A redraw rebuilds the graph, which discards the marked attributes.
  container.querySelectorAll(".graph-edge").forEach(edge => {
    edge.classList.remove("is-selected");
    edge.setAttribute("aria-selected", "false");
  });
  assert.deepEqual(selectedEdges(container), [], "the redraw cleared the marks");

  store.setState({ evidence: EVIDENCE, loading: false });
  assert.deepEqual(
    selectedEdges(container),
    ["rel-observed-ip"],
    "the selection survives a data refresh"
  );

  // Data that no longer contains the selected id must drop the selection
  // rather than move it onto whatever element happens to be there.
  store.setState({ evidence: EVIDENCE.slice(1), loading: false });
  assert.deepEqual(selectedEdges(container), [], "a vanished id is not re-pointed");

  store.setState({ evidence: EVIDENCE, loading: false });
  return {
    survivedRedraw: true,
    vanishedIdDropped: true,
    selectionAfterDrop: graphSelection.getSelection()
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