import { getOverview, getSignals, getEvidence, getPaths, getBehavior, getProvenance, getAll, getHealth } from "./api.js";
import { store } from "./state.js";
import { initOverview, cleanupOverview } from "./overview.js";
import { initWhyThisLead, cleanupWhyThisLead } from "./why-this-lead.js";
import { mountEvidenceChain, unmountEvidenceChain } from "./evidence-chain.js";
import { initCrossLayer, cleanupCrossLayer } from "./cross-layer.js";
import { mountPaths, unmountPaths } from "./paths.js";
import { initBehaviorML, cleanupBehaviorML } from "./behavior-ml.js";
import { initProvenance, cleanupProvenance } from "./provenance.js";
import { initLegend, cleanupLegend } from "./legend.js";
import { initGraph, cleanupGraph, initGraphSelection } from "./graph.js";
import { initNavigation, cleanupNavigation } from "./navigation.js";
import { initGlobalSearch, cleanupGlobalSearch } from "./search.js";

let currentAbortController = null;
let retryHandlers = {};
let evidenceChain = null;
let pathsSection = null;
let graphSelection = null;
let dataUnsubscribe = null;

async function loadDashboard() {
  store.setState({ loading: true, errors: {} });
  
  try {
    const [overview, signals, evidence, paths, behavior, provenance] = await Promise.allSettled([
      getOverview(),
      getSignals(),
      getEvidence(),
      getPaths(),
      getBehavior(),
      getProvenance()
    ]);
    
    const errors = {};
    const data = {};
    
    if (overview.status === "fulfilled") data.overview = overview.value;
    else { errors.overview = overview.reason; console.error("Overview error:", overview.reason); }
    
    if (signals.status === "fulfilled") data.signals = signals.value;
    else { errors.signals = signals.reason; console.error("Signals error:", signals.reason); }
    
    if (evidence.status === "fulfilled") data.evidence = evidence.value;
    else { errors.evidence = evidence.reason; console.error("Evidence error:", evidence.reason); }
    
    if (paths.status === "fulfilled") data.paths = paths.value;
    else { errors.paths = paths.reason; console.error("Paths error:", paths.reason); }
    
    if (behavior.status === "fulfilled") data.behavior = behavior.value;
    else { errors.behavior = behavior.reason; console.error("Behavior error:", behavior.reason); }
    
    if (provenance.status === "fulfilled") data.provenance = provenance.value;
    else { errors.provenance = provenance.reason; console.error("Provenance error:", provenance.reason); }
    
store.setState({ ...data, loading: false, errors });
  } catch (error) {
    console.error("Dashboard load failed:", error);
    store.setState({ loading: false, errors: { global: error } });
  }
}

function startDataSubscription() {
  if (dataUnsubscribe) return;
  dataUnsubscribe = store.subscribe(state => {
    if (state.loading) return;
    onDashboardData(state);
  });
}

/** Interactive sections render once, then follow the store. */
function startInteractiveSections() {
  initInteractiveSections();
  startDataSubscription();
}

function setupRetryHandlers() {
  const sections = ["overview", "signals", "evidence", "paths", "behavior", "provenance"];
  
  sections.forEach(section => {
    const handler = () => retrySection(section);
    retryHandlers[section] = handler;
    window.addEventListener(`retry-${section}`, handler);
  });
}

function cleanupRetryHandlers() {
  Object.entries(retryHandlers).forEach(([section, handler]) => {
    window.removeEventListener(`retry-${section}`, handler);
  });
  retryHandlers = {};
}

async function retrySection(section) {
  store.setState({ errors: { ...store.state.errors, [section]: null } });
  
  try {
    let data;
    switch (section) {
      case "overview": data = await getOverview(); break;
      case "signals": data = { signals: await getSignals() }; break;
      case "evidence": data = { evidence: await getEvidence() }; break;
      case "paths": data = { paths: await getPaths() }; break;
      case "behavior": data = { behavior: await getBehavior() }; break;
      case "provenance": data = { provenance: await getProvenance() }; break;
    }
    store.setState({ ...data, errors: { ...store.state.errors, [section]: null } });
  } catch (error) {
    console.error(`Retry ${section} failed:`, error);
    store.setState({ errors: { ...store.state.errors, [section]: error } });
  }
}

function initAll() {
  initOverview();
  initWhyThisLead();
  initCrossLayer();
  initBehaviorML();
  initProvenance();
initLegend();
  initGraph();
  initNavigation();
  initGlobalSearch();
  setupRetryHandlers();

  document.getElementById("btn-refresh").addEventListener("click", loadDashboard);
  document.getElementById("btn-print").addEventListener("click", () => window.print());

  loadDashboard();
  startInteractiveSections();
}

/*
 * Step 9B Phase 3: the evidence chain and the paths section are interactive, so
 * they are wired to the store here rather than self-subscribing. Each section
 * owns its own expansion/selection state and re-renders when the data changes;
 * the store still only ever holds read-only dashboard data.
 */
function initInteractiveSections() {
  const evidence = store.state.evidence || [];

  evidenceChain = mountEvidenceChain({
    evidence,
    graphEvidence: evidence,
    onSelect(record, sync) {
      syncGraphSelection(record ? record.relationship_id : null, null, sync);
    }
  });

  pathsSection = mountPaths({
    paths: store.state.paths || [],
    graphEvidence: evidence,
    onSelect(record) {
      // A path is a sequence, not a single element, so selecting one highlights
      // its nodes and edges in the graph without choosing any one of them here.
      if (!record) {
        graphSelection.clear();
        return;
      }
      applyGraphSelection(record.observation_ids, record.relationship_ids);
    },
    onCompare() {
      // Comparing two paths does not imply anything about the graph.
      graphSelection.clear();
    }
  });

  graphSelection = initGraphSelection({
    container: "cross-layer-content",
    detailsContainer: document.getElementById("graph-selection-details"),
    evidence,
    onSelect(selection, applied) {
      onGraphSelectionChanged(selection, applied);
    }
  });
}

/**
 * Push an evidence chain selection into the graph by exact id.
 *
 * `sync.synchronized` is only true when the graph holds that exact
 * `relationship_id`. When it does not, the graph is cleared rather than
 * highlighted on a near match.
 */
function syncGraphSelection(relationshipId, observationId, sync) {
  if (!graphSelection) return;
  if (relationshipId && sync && sync.synchronized) {
    graphSelection.selectRelationship(relationshipId);
    return;
  }
  if (observationId) {
    graphSelection.selectObservation(observationId);
    return;
  }
  graphSelection.clear();
}

/** Mark every exact id a path contains, without inventing or picking one. */
function applyGraphSelection(observationIds, relationshipIds) {
  if (!graphSelection) return;
  const applied = graphSelection.selectMany(
    observationIds || [],
    relationshipIds || []
  );
  if (!applied.relationships.length && !applied.observations.length) {
    graphSelection.clear();
  }
}

/** Reflect a graph-initiated selection back into the two Phase 3 sections. */
function onGraphSelectionChanged(selection, applied) {
  if (!selection) return;
  if (selection.relationshipId && evidenceChain) {
    evidenceChain.highlightRelationship(selection.relationshipId);
  }
  if (selection.observationId && pathsSection) {
    pathsSection.highlightObservation(selection.observationId);
  }
  void applied;
}

function onDashboardData(state) {
  const evidence = state.evidence || [];
  if (evidenceChain) evidenceChain.setEvidence(evidence);
  if (pathsSection) pathsSection.setPaths(state.paths || [], evidence);
  if (graphSelection) graphSelection.setEvidence(evidence);
}

function cleanupAll() {
  cleanupOverview();
  cleanupWhyThisLead();
  cleanupCrossLayer();
  cleanupBehaviorML();
  cleanupProvenance();
  cleanupLegend();
  cleanupGraph();
  cleanupGlobalSearch();
  cleanupNavigation();
  cleanupRetryHandlers();
  unmountEvidenceChain();
  unmountPaths();
  if (graphSelection) {
    graphSelection.destroy();
    graphSelection = null;
  }
  if (dataUnsubscribe) {
    dataUnsubscribe();
    dataUnsubscribe = null;
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initAll);
} else {
  initAll();
}

window.addEventListener("beforeunload", cleanupAll);

export { loadDashboard };
