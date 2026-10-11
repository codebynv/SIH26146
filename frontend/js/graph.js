import { escapeHtml, truncateId, getInferenceClass } from "./formatters.js";
import { store } from "./state.js";

let unsubscribe = null;
let svg = null;
let zoomLevel = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let lastPanX = 0;
let lastPanY = 0;

const NODE_RADIUS = 18;
const SUBJECT_NODE_RADIUS = 22;
const LABEL_OFFSET = 28;
const VERTICAL_SPACING = 100;
const HORIZONTAL_SPACING = 180;

export function initGraph() {
  unsubscribe = store.subscribe(renderGraph);
  renderGraph(store.state);
}

function renderGraph(state) {
  const container = document.getElementById("cross-layer-content");
  if (!container) return;
  
  const graphContainer = container.querySelector(".graph-container");
  if (!graphContainer) return;
  
  const { evidence, loading, errors } = state;
  
  if (loading || evidence.length === 0) {
    graphContainer.innerHTML = '<div class="loading-skeleton" style="width: 100%; height: 100%;"></div>';
    return;
  }
  
  if (errors.evidence) {
    graphContainer.innerHTML = renderError("evidence", errors.evidence);
    return;
  }
  
  const layout = computeLayout(evidence);
  drawGraph(graphContainer, layout, evidence);
}

function computeLayout(evidence) {
  const subjectNodes = new Set();
  const neighborNodes = new Set();
  const edges = [];
  
  for (const e of evidence) {
    for (const src of e.source_observation_ids || []) {
      subjectNodes.add(src);
    }
    for (const tgt of e.target_observation_ids || []) {
      if (!subjectNodes.has(tgt)) neighborNodes.add(tgt);
    }
    edges.push(e);
  }
  
  const subjectList = Array.from(subjectNodes).sort();
  const neighborList = Array.from(neighborNodes).sort();
  
  const positions = new Map();
  
  const centerX = 400;
  const centerY = 200;
  
  subjectList.forEach((id, i) => {
    const x = centerX + (i - (subjectList.length - 1) / 2) * HORIZONTAL_SPACING;
    positions.set(id, { x, y: centerY, isSubject: true });
  });
  
  neighborList.forEach((id, i) => {
    const row = Math.floor(i / 4);
    const col = i % 4;
    const x = centerX + (col - 1.5) * HORIZONTAL_SPACING;
    const y = centerY + (row - 0.5) * VERTICAL_SPACING * 1.5;
    positions.set(id, { x, y, isSubject: false });
  });
  
  const layoutEdges = [];
  for (const e of edges) {
    for (const src of e.source_observation_ids || []) {
      for (const tgt of e.target_observation_ids || []) {
        const srcPos = positions.get(src);
        const tgtPos = positions.get(tgt);
        if (srcPos && tgtPos) {
          layoutEdges.push({
            source: src,
            target: tgt,
            sourcePos: srcPos,
            targetPos: tgtPos,
            evidence: e,
            isIncident: subjectNodes.has(src) && subjectNodes.has(tgt)
          });
        }
      }
    }
  }
  
  return { nodes: positions, edges: layoutEdges };
}

function drawGraph(container, layout, evidence) {
  const { nodes, edges } = layout;
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 400;
  
  let html = `
    <svg class="graph-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
        </marker>
      </defs>
      <g class="graph-layer" transform="translate(${panX}, ${panY}) scale(${zoomLevel})">
  `;
  
  for (const edge of edges) {
    const { sourcePos, targetPos, evidence: e, isIncident } = edge;
    const x1 = sourcePos.x;
    const y1 = sourcePos.y;
    const x2 = targetPos.x;
    const y2 = targetPos.y;
    
    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist === 0) continue;
    
    const r1 = sourcePos.isSubject ? SUBJECT_NODE_RADIUS : NODE_RADIUS;
    const r2 = targetPos.isSubject ? SUBJECT_NODE_RADIUS : NODE_RADIUS;
    
    const nx = dx / dist;
    const ny = dy / dist;
    
    const sx = x1 + nx * r1;
    const sy = y1 + ny * r1;
    const tx = x2 - nx * r2;
    const ty = y2 - ny * r2;
    
    const mx = (sx + tx) / 2;
    const my = (sy + ty) / 2;
    const curveOffset = 30;
    const cx = mx - ny * curveOffset;
    const cy = my + nx * curveOffset;
    
    const inferenceClass = getInferenceClass(e.inference_level);
    const edgeClass = `graph-edge ${inferenceClass} ${isIncident ? "graph-edge--incident" : "graph-edge--context"}`;
    
// Step 9B Phase 3: the edge is a focusable, labelled control so it can be
    // selected from the keyboard as well as the pointer, and it carries its exact
    // relationship id plus an explicit aria-selected="false" baseline that
    // selection state is written over.
    html += `
      <path class="${edgeClass}" 
            d="M ${sx} ${sy} Q ${cx} ${cy} ${tx} ${ty}" 
            marker-end="url(#arrowhead)"
            tabindex="0"
            role="button"
            aria-selected="false"
            aria-label="${escapeHtml(e.evidence_type)} relationship ${escapeHtml(e.relationship_id)}"
            data-relationship="${escapeHtml(e.relationship_id)}"
            data-evidence="${escapeHtml(JSON.stringify(e).replace(/"/g, ""))}"></path>
    `;
  }
  
  for (const [id, pos] of nodes) {
    const isSubject = pos.isSubject;
    const r = isSubject ? SUBJECT_NODE_RADIUS : NODE_RADIUS;
    const nodeClass = `graph-node ${isSubject ? "graph-node--subject" : "graph-node--neighbor"}`;
    const labelY = pos.y + r + LABEL_OFFSET;
    
    // Step 9B Phase 3: as with edges, a node is focusable and exposes its exact
    // observation id as both its accessible name and its selection key.
    html += `
      <g class="${nodeClass}" tabindex="0" role="button" aria-selected="false" aria-label="Observation ${escapeHtml(id)}" data-observation="${escapeHtml(id)}" transform="translate(${pos.x}, ${pos.y})">
        <circle class="graph-node__circle" r="${r}"></circle>
        <text class="graph-node__label" x="0" y="${LABEL_OFFSET}" text-anchor="middle">${escapeHtml(truncateId(id, 14))}</text>
      </g>
    `;
  }
  
  html += `
      </g>
    </svg>
    <div class="graph-controls">
      <button class="graph-zoom-btn" data-action="zoom-in" aria-label="Zoom in">+</button>
      <button class="graph-zoom-btn" data-action="zoom-out" aria-label="Zoom out">−</button>
      <button class="graph-zoom-btn" data-action="reset" aria-label="Reset view">⌂</button>
    </div>
    <div class="graph-zoom">Zoom: ${Math.round(zoomLevel * 100)}%</div>
    <div class="graph-legend">
      <div class="graph-legend__item"><span class="graph-legend__swatch" style="background: var(--inference-observed);"></span><span>Observed</span></div>
      <div class="graph-legend__item"><span class="graph-legend__swatch" style="background: var(--inference-correlated);"></span><span>Correlated</span></div>
      <div class="graph-legend__item"><span class="graph-legend__swatch" style="background: repeating-linear-gradient(90deg, var(--inference-inferred), var(--inference-inferred) 4px, transparent 4px, transparent 8px);"></span><span>Inferred</span></div>
      <div class="graph-legend__item"><span class="graph-legend__swatch" style="background: repeating-linear-gradient(90deg, var(--inference-requires), var(--inference-requires) 2px, transparent 2px, transparent 4px);"></span><span>Requires Verification</span></div>
    </div>
  `;
  
  container.innerHTML = html;
  
  attachGraphInteractions(container, layout);
}

function attachGraphInteractions(container, layout) {
  const svgEl = container.querySelector(".graph-svg");
  const layer = container.querySelector(".graph-layer");
  const zoomDisplay = container.querySelector(".graph-zoom");
  const tooltip = document.getElementById("global-tooltip");
  
  if (!svgEl || !layer) return;
  
  container.querySelectorAll(".graph-zoom-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.action;
      if (action === "zoom-in") zoomLevel = Math.min(3, zoomLevel * 1.2);
      else if (action === "zoom-out") zoomLevel = Math.max(0.3, zoomLevel / 1.2);
      else { zoomLevel = 1; panX = 0; panY = 0; }
      layer.setAttribute("transform", `translate(${panX}, ${panY}) scale(${zoomLevel})`);
      zoomDisplay.textContent = `Zoom: ${Math.round(zoomLevel * 100)}%`;
    });
  });
  
  svgEl.addEventListener("mousedown", (e) => {
    if (e.target.closest(".graph-zoom-btn")) return;
    isPanning = true;
    lastPanX = e.clientX - panX;
    lastPanY = e.clientY - panY;
    svgEl.style.cursor = "grabbing";
  });
  
  window.addEventListener("mousemove", (e) => {
    if (!isPanning) return;
    panX = e.clientX - lastPanX;
    panY = e.clientY - lastPanY;
    layer.setAttribute("transform", `translate(${panX}, ${panY}) scale(${zoomLevel})`);
  });
  
  window.addEventListener("mouseup", () => {
    if (isPanning) {
      isPanning = false;
      svgEl.style.cursor = "grab";
    }
  });
  
  svgEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    zoomLevel = Math.max(0.3, Math.min(3, zoomLevel * delta));
    layer.setAttribute("transform", `translate(${panX}, ${panY}) scale(${zoomLevel})`);
    zoomDisplay.textContent = `Zoom: ${Math.round(zoomLevel * 100)}%`;
  }, { passive: false });
  
  container.querySelectorAll(".graph-edge").forEach(path => {
    path.addEventListener("mouseenter", (e) => {
      try {
        const evidence = JSON.parse(path.dataset.evidence);
        tooltip.innerHTML = `
          <div class="tooltip__title">${escapeHtml(evidence.relationship_id)}</div>
          <div class="tooltip__row"><span class="tooltip__label">Type</span><span class="tooltip__value">${escapeHtml(evidence.evidence_type)}</span></div>
          <div class="tooltip__row"><span class="tooltip__label">Inference</span><span class="tooltip__value">${escapeHtml(evidence.inference_level)}</span></div>
          <div class="tooltip__row"><span class="tooltip__label">Confidence</span><span class="tooltip__value">${(evidence.confidence_value * 100).toFixed(1)}% (${escapeHtml(evidence.confidence_basis)})</span></div>
          <div class="tooltip__row"><span class="tooltip__label">Rule</span><span class="tooltip__value">${escapeHtml(evidence.rule_id)} v${escapeHtml(evidence.rule_version)}</span></div>
          <div class="tooltip__row"><span class="tooltip__label">Source</span><span class="tooltip__value">${escapeHtml(evidence.provenance_refs?.map(r => r.source_file).join(", ") || "—")}</span></div>
          <div class="tooltip__row"><span class="tooltip__label">Rationale</span><span class="tooltip__value" style="max-width: 200px; white-space: normal;">${escapeHtml(evidence.rationale)}</span></div>
        `;
        tooltip.classList.add("visible");
        positionTooltip(e, tooltip);
      } catch {}
    });
    path.addEventListener("mousemove", (e) => positionTooltip(e, tooltip));
    path.addEventListener("mouseleave", () => tooltip.classList.remove("visible"));
  });
  
  container.querySelectorAll(".graph-node").forEach(node => {
    node.addEventListener("mouseenter", (e) => {
      const obsId = node.dataset.observation;
      tooltip.innerHTML = `
        <div class="tooltip__title">${escapeHtml(obsId)}</div>
        <div class="tooltip__row"><span class="tooltip__label">Type</span><span class="tooltip__value">${node.classList.contains("graph-node--subject") ? "Subject / Incident" : "1-Hop Neighbor"}</span></div>
      `;
      tooltip.classList.add("visible");
      positionTooltip(e, tooltip);
    });
    node.addEventListener("mousemove", (e) => positionTooltip(e, tooltip));
    node.addEventListener("mouseleave", () => tooltip.classList.remove("visible"));
  });
}

function positionTooltip(event, tooltip) {
  const padding = 12;
  let x = event.clientX + padding;
  let y = event.clientY + padding;
  
  const rect = tooltip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - padding) x = event.clientX - rect.width - padding;
  if (y + rect.height > window.innerHeight - padding) y = event.clientY - rect.height - padding;
  
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function renderError(section, error) {
  return `
<div class="section-error">
      <span class="section-error__icon">?</span>
      <div>
        <div style="font-weight: 600;">Investigation data could not be loaded.</div>
        <button class="primary" onclick="window.dispatchEvent(new CustomEvent('retry-${section}'))">Retry</button>
      </div>
    </div>
  `;
}

/**
 * Graph selection (Step 9B Phase 3).
 *
 * Selection is by exact identifier and nothing else: an edge is selected by its
 * `relationship_id` and a node by its `observation_id`. Display text, array
 * index and label similarity are never used, and selecting an id the graph does
 * not hold leaves the graph unselected rather than matching a near neighbour.
 * Every function here is pure; `applyGraphSelection` only writes presentation
 * attributes on elements that already exist in the DOM.
 */

/** The initial graph selection state: nothing selected. */
export function defaultSelection() {
  return { relationshipId: null, observationId: null };
}

/**
 * Select one edge by exact `relationship_id`.
 *
 * Passing `null` clears the edge selection. Selecting a new edge clears the node
 * selection, so the graph reports one selection rather than two.
 */
export function selectRelationship(selection, relationshipId) {
  const current = selection || defaultSelection();
  return {
    relationshipId: relationshipId || null,
    observationId: null
  };
}

/**
 * Select one node by exact `observation_id`.
 *
 * Passing `null` clears the node selection. Selecting a node clears the edge
 * selection.
 */
export function selectObservation(selection, observationId) {
  const current = selection || defaultSelection();
  return {
    relationshipId: null,
    observationId: observationId || null
  };
}

/** Clear both selections. */
export function clearSelection(selection) {
  return defaultSelection();
}

/**
 * Whether a graph edge with this exact `relationship_id` exists.
 *
 * `false` means there is no exact match, and a caller must not select anything
 * on the strength of a near-miss.
 */
export function hasRelationship(evidence, relationshipId) {
  if (!relationshipId) return false;
  return (evidence || []).some(
    record => record && record.relationship_id === relationshipId
  );
}

/**
 * Whether a graph node with this exact `observation_id` exists.
 *
 * A node exists when the id is a source or a target of some recorded
 * relationship, which is exactly how the renderer builds its node set.
 */
export function hasObservation(evidence, observationId) {
  if (!observationId) return false;
  return (evidence || []).some(
    record =>
      record &&
      ((record.source_observation_ids || []).indexOf(observationId) !== -1 ||
        (record.target_observation_ids || []).indexOf(observationId) !== -1)
  );
}

/**
 * Resolve an evidence selection into the graph, by exact `relationship_id`.
 *
 * Returns `synchronized: false` and no relationship id when the graph holds no
 * relationship with that id, so no caller can act on a relationship that was
 * never recorded.
 */
export function syncFromEvidence(evidence, relationshipId) {
  if (!hasRelationship(evidence, relationshipId)) {
    return { synchronized: false, relationshipId: null, selection: defaultSelection() };
  }
  return {
    synchronized: true,
    relationshipId,
    selection: selectRelationship(defaultSelection(), relationshipId)
  };
}

/**
 * Resolve a path node into the graph, by exact `observation_id`.
 */
export function syncFromPathNode(evidence, observationId) {
  if (!hasObservation(evidence, observationId)) {
    return { synchronized: false, observationId: null, selection: defaultSelection() };
  }
  return {
    synchronized: true,
    observationId,
    selection: selectObservation(defaultSelection(), observationId)
  };
}

/**
 * Write a selection into a rendered graph.
 *
 * By default this marks exactly one element with `is-selected` and
 * `aria-selected`, and clears those attributes on every other, so the DOM cannot
 * report two selected elements. When `highlight.relationships` or
 * `highlight.observations` are supplied, every exact id in them is marked
 * instead; that is how a path highlights a whole sequence without any one of its
 * elements becoming "the" selection. Ids with no element are ignored, never
 * approximated.
 *
 * Returns the ids that were actually marked.
 */
export function applyGraphSelection(container, selection, highlight) {
  if (!container) return { relationships: [], observations: [] };
  const manyRelationships = highlight && highlight.relationships ? highlight.relationships : null;
  const manyObservations = highlight && highlight.observations ? highlight.observations : null;
  const relationships = [];
  const observations = [];
  container.querySelectorAll(".graph-edge").forEach(edge => {
    const id = edge.getAttribute("data-relationship");
    const selected = manyRelationships
      ? manyRelationships.indexOf(id) !== -1
      : !!selection && selection.relationshipId === id;
    edge.classList.toggle("is-selected", selected);
    edge.setAttribute("aria-selected", selected ? "true" : "false");
    if (selected) relationships.push(id);
  });
  container.querySelectorAll(".graph-node").forEach(node => {
    const id = node.getAttribute("data-observation");
    const selected = manyObservations
      ? manyObservations.indexOf(id) !== -1
      : !!selection && selection.observationId === id;
    node.classList.toggle("is-selected", selected);
    node.setAttribute("aria-selected", selected ? "true" : "false");
    if (selected) observations.push(id);
  });
  return { relationships, observations };
}

/**
 * Render the graph selection details panel.
 *
 * States plainly which relationship or observation is selected, and says so
 * explicitly when nothing is. When several ids are highlighted without any one
 * being the selection, it says how many. Returns the detail object that was
 * rendered, so a caller can assert on it without scraping HTML.
 */
export function renderSelectionDetails(container, selection, highlight) {
  if (!container) return null;
  const highlightedRelationships = highlight && highlight.relationshipIds ? highlight.relationshipIds : [];
  const highlightedObservations = highlight && highlight.observationIds ? highlight.observationIds : [];
  const hasSingle = selection && (selection.relationshipId || selection.observationId);

  if (!hasSingle && !highlightedRelationships.length && !highlightedObservations.length) {
    container.innerHTML = "No graph element selected.";
    return null;
  }

  const parts = [`<div class="graph-selection"`];
  if (hasSingle && selection.relationshipId) {
    parts.push(` data-relationship-id="${escapeHtml(selection.relationshipId)}"`);
  }
  if (hasSingle && selection.observationId) {
    parts.push(` data-observation-id="${escapeHtml(selection.observationId)}"`);
  }
  parts.push(">");

  if (hasSingle && selection.relationshipId) {
    parts.push(
      `<div class="graph-selection__row" data-field="relationship_id"><span>Relationship</span><span class="mono">${escapeHtml(selection.relationshipId)}</span></div>`
    );
  }
  if (hasSingle && selection.observationId) {
    parts.push(
      `<div class="graph-selection__row" data-field="observation_id"><span>Observation</span><span class="mono">${escapeHtml(selection.observationId)}</span></div>`
    );
  }
  if (highlightedRelationships.length) {
    parts.push(
      `<div class="graph-selection__row" data-field="highlighted_relationships"><span>Highlighted relationships</span><span class="mono">${highlightedRelationships.length}</span></div>`
    );
  }
  if (highlightedObservations.length) {
    parts.push(
      `<div class="graph-selection__row" data-field="highlighted_observations"><span>Highlighted observations</span><span class="mono">${highlightedObservations.length}</span></div>`
    );
  }
  parts.push("</div>");
  container.innerHTML = parts.join("");

  return {
    kind: hasSingle
      ? selection.relationshipId
        ? "relationship"
        : "observation"
      : "highlighted",
    relationshipId: hasSingle ? selection.relationshipId : null,
    observationId: hasSingle ? selection.observationId : null,
    highlightedRelationships,
    highlightedObservations
  };
}

/**
 * Find the element that holds the graph and receives selection events.
 *
 * A caller may pass an element directly or the id of one. Anything else falls
 * back to the cross-layer section, which is where the graph is rendered.
 */
function resolveContainer(container) {
  if (container && typeof container.querySelector === "function") return container;
  if (typeof container === "string") return document.getElementById(container);
  return document.getElementById("cross-layer-content");
}

/**
 * Wire the graph section's selection.
 *
 * A click on an edge selects that edge's exact `relationship_id`; a click on a
 * node selects that node's exact `observation_id`. Selection is also available
 * from the keyboard, via Enter and Space on a focusable element.
 *
 * `config.evidence` is the collection the graph was drawn from, and it is the
 * only thing an id is allowed to match against.
 */
export function initGraphSelection(options) {
  const config = options || {};
  const container = resolveContainer(config.container);
  const detailsContainer = config.detailsContainer || document.getElementById("graph-selection-details");
  let selection = config.selection || defaultSelection();

  function render() {
    const applied = applyGraphSelection(container, selection);
    const rendered = renderSelectionDetails(detailsContainer, selection);
    if (config.onSelect) config.onSelect(selection, applied, rendered);
    return rendered;
  }

  /**
   * Re-mark the current selection after the graph has been redrawn.
   *
   * The graph is rebuilt from the store on every data change, which discards the
   * `is-selected` / `aria-selected` attributes written earlier. This reapplies
   * them without notifying `onSelect`, so a data refresh cannot look like a user
   * click and cannot feed back into the sections that requested the selection.
   * A selection whose id the new data no longer holds is dropped, never moved
   * onto a near element.
   */
  function reapply() {
    const evidence = config.evidence;
    if (selection.relationshipId && !(evidence && hasRelationship(evidence, selection.relationshipId))) {
      selection = clearSelection(selection);
    }
    if (selection.observationId && !(evidence && hasObservation(evidence, selection.observationId))) {
      selection = clearSelection(selection);
    }
    return applyGraphSelection(container, selection);
  }

  function idFromEvent(event) {
    const edge = event.target.closest(".graph-edge");
    if (edge) {
      return { kind: "relationship", id: edge.getAttribute("data-relationship") };
    }
    const node = event.target.closest(".graph-node");
    if (node) {
      return { kind: "observation", id: node.getAttribute("data-observation") };
    }
    return null;
  }

  function activate(event) {
    if (event.type === "keydown") {
      if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") return;
      event.preventDefault();
    }
    const target = idFromEvent(event);
    if (!target) return;
    if (target.kind === "relationship") {
      if (!config.evidence || !hasRelationship(config.evidence, target.id)) return;
      selection = selectRelationship(selection, target.id);
    } else {
      if (!config.evidence || !hasObservation(config.evidence, target.id)) return;
      selection = selectObservation(selection, target.id);
    }
    render();
  }

  if (container) {
    container.addEventListener("click", activate);
    container.addEventListener("keydown", activate);
  }

  // Subscribed after initGraph, so this runs after the redraw it must correct.
  const unsubscribeRedraw = store.subscribe(state => {
    if (state.loading) return;
    config.evidence = state.evidence || [];
    reapply();
  });

  render();

  return {
    getSelection: () => selection,
    selectRelationship(relationshipId) {
      if (!config.evidence || !hasRelationship(config.evidence, relationshipId)) return null;
      selection = selectRelationship(selection, relationshipId);
      return render();
    },
    selectObservation(observationId) {
      if (!config.evidence || !hasObservation(config.evidence, observationId)) return null;
      selection = selectObservation(selection, observationId);
      return render();
    },
    /**
     * Highlight a set of exact ids without making any one of them "the"
     * selection. Used by the paths section, whose nodes and edges belong to a
     * sequence rather than to a single element. Ids the graph does not hold are
     * simply not highlighted.
     */
    selectMany(observationIds, relationshipIds) {
      const observations = (observationIds || []).filter(id =>
        config.evidence && hasObservation(config.evidence, id)
      );
      const relationships = (relationshipIds || []).filter(id =>
        config.evidence && hasRelationship(config.evidence, id)
      );
      selection = {
        relationshipId: relationships.length === 1 ? relationships[0] : null,
        observationId: observations.length === 1 ? observations[0] : null
      };
      const applied = applyGraphSelection(container, selection, {
        relationships,
        observations
      });
      renderSelectionDetails(detailsContainer, selection, {
        relationshipIds: relationships,
        observationIds: observations
      });
      if (config.onSelect) config.onSelect(selection, applied, null);
      return applied;
    },
    setEvidence(evidence) {
      config.evidence = evidence || [];
      return render();
    },
    clear() {
      selection = clearSelection(selection);
      return render();
    },
    destroy() {
      if (container) {
        container.removeEventListener("click", activate);
        container.removeEventListener("keydown", activate);
      }
      unsubscribeRedraw();
    }
  };
}

export function cleanupGraph() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
}
