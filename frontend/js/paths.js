/**
 * Investigation Paths interaction (Step 9B Phase 3).
 *
 * Selection and comparison are presentation state over the Step 7 paths the API
 * already returns. Every exported state function is pure, returns new state, and
 * reads the path collection without mutating it.
 *
 * Two deliberate omissions, both from the Step 8 semantics this project holds:
 *
 *   no risk figure     A path is a structure over recorded evidence. No
 *                      path-level score, risk number or priority is computed
 *                      anywhere in this module, and the two paths in a comparison
 *                      are never ordered by suspicion: the comparison preserves
 *                      the order they were selected in.
 *   no invented times  Temporal readings come from Step 7 unchanged. A
 *                      NOT_COMPARABLE or UNKNOWN transition carries no elapsed
 *                      time and no ordering, and a path whose timestamps could
 *                      not be compared is never re-sorted into a chronology.
 *
 * Graph synchronisation is by exact `observation_id` / `relationship_id` only.
 */

import { el, replaceChildrenWith, idSpan } from "./dom.js";
import {
  formatNumber,
  formatDuration,
  truncateId,
  getInferenceClass
} from "./formatters.js";
import { isActivationKey } from "./evidence-chain.js";
import {
  buildTimeline,
  pathTemporalStatus,
  pathTemporalWindow,
  PATH_TEMPORAL_STATUS,
  TEMPORAL_ORDER
} from "./temporal.js";

export {
  buildTimeline,
  pathTemporalStatus,
  pathTemporalWindow,
  PATH_TEMPORAL_STATUS,
  TEMPORAL_ORDER
};

/** The most paths a comparison holds. */
export const MAX_COMPARISON_PATHS = 2;

/**
 * Order paths by hop count, then by `path_id`.
 *
 * The id tiebreak matters: two paths with the same hop count must not be
 * ordered by array position or by object identity, or the dashboard's ordering
 * would depend on the order the API happened to serialise.
 */
export function sortPaths(paths) {
  return (Array.isArray(paths) ? paths.slice() : []).sort((a, b) => {
    const ha = typeof a.hop_count === "number" ? a.hop_count : 0;
    const hb = typeof b.hop_count === "number" ? b.hop_count : 0;
    if (ha !== hb) return ha - hb;
    return String(a.path_id).localeCompare(String(b.path_id));
  });
}

/** The initial interaction state: nothing selected, no comparison. */
export function defaultState() {
  return { selectedPathId: null, comparePathIds: [] };
}

/**
 * Find one path by exact `path_id`.
 *
 * Returns `null` when there is no exact match. Nothing is matched on displayed
 * text, on hop count, or on position in an array.
 */
export function findByPathId(paths, pathId) {
  if (!pathId) return null;
  const found = (Array.isArray(paths) ? paths : []).find(
    path => path && path.path_id === pathId
  );
  return found === undefined ? null : found;
}

/**
 * Select a path without toggling.
 *
 * `selectPath` toggles because a second click on the same header means to close
 * it. A selection arriving from the graph is not a click, so this sets the id
 * outright and is idempotent.
 */
export function setSelectedPath(state, pathId) {
  const current = state || defaultState();
  return {
    selectedPathId: pathId || null,
    comparePathIds: (current.comparePathIds || []).slice()
  };
}

/** Whether one path is the current selection. */
export function isSelected(state, pathId) {
  return !!state && !!pathId && state.selectedPathId === pathId;
}

/** The path the state currently selects, or `null`. */
export function selectedPath(paths, state) {
  if (!state || !state.selectedPathId) return null;
  return findByPathId(paths, state.selectedPathId);
}

/**
 * Select a path, replacing any previous selection.
 *
 * Selecting the already-selected id clears the selection, so a second
 * activation deselects. The comparison set is untouched: comparing and browsing
 * are independent.
 */
export function selectPath(state, pathId) {
  const current = state || defaultState();
  const next =
    current.selectedPathId && pathId === current.selectedPathId ? null : pathId || null;
  return {
    selectedPathId: next,
    comparePathIds: (current.comparePathIds || []).slice()
  };
}

/** Clear the selection, leaving the comparison set as it is. */
export function clearSelection(state) {
  const current = state || defaultState();
  return { selectedPathId: null, comparePathIds: (current.comparePathIds || []).slice() };
}

/** Whether a path is in the comparison set. */
export function isInComparison(state, pathId) {
  return !!state && (state.comparePathIds || []).indexOf(pathId) !== -1;
}

/**
 * Add or remove a path from the comparison set.
 *
 * The set holds at most `MAX_COMPARISON_PATHS` distinct paths. Asking for a
 * third returns the state unchanged rather than silently dropping one of the two
 * already chosen: which path leaves a comparison is the investigator's decision,
 * not the code's.
 *
 * Order of addition is preserved, so the comparison reads in the order the paths
 * were chosen and never re-ranks them.
 */
export function toggleComparison(state, pathId) {
  const current = state || defaultState();
  const comparePathIds = (current.comparePathIds || []).slice();
  if (!pathId) return { selectedPathId: current.selectedPathId || null, comparePathIds };
  const index = comparePathIds.indexOf(pathId);
  if (index !== -1) {
    comparePathIds.splice(index, 1);
    return { selectedPathId: current.selectedPathId || null, comparePathIds };
  }
  if (comparePathIds.length >= MAX_COMPARISON_PATHS) {
    return { selectedPathId: current.selectedPathId || null, comparePathIds };
  }
  comparePathIds.push(pathId);
  return { selectedPathId: current.selectedPathId || null, comparePathIds };
}

/** The comparison paths, resolved against `paths`, in selection order. */
export function comparisonPaths(paths, state) {
  const ids = (state && state.comparePathIds) || [];
  return ids.map(id => findByPathId(paths, id)).filter(path => path !== null);
}

/** Empty the comparison set, leaving the current selection as it is. */
export function resetComparison(state) {
  const current = state || defaultState();
  return { selectedPathId: current.selectedPathId || null, comparePathIds: [] };
}

/** Whether a comparison is active. */
export function hasComparison(state) {
  return !!state && (state.comparePathIds || []).length > 0;
}

/** The ordered node sequence of a path, as a new array. */
export function orderedNodes(path) {
  if (!path) return [];
  return (path.observation_ids || []).slice();
}

/** The ordered transition (relationship) sequence of a path, as a new array. */
export function orderedTransitions(path) {
  if (!path) return [];
  return (path.relationship_ids || []).slice();
}

/** The evidence summary the API carries for a path. */
export function evidenceSummary(path) {
  if (!path || !path.evidence_summary) return null;
  return path.evidence_summary;
}

/**
 * The per-path detail view: identity, ordered nodes and transitions, evidence
 * summary and temporal standing.
 *
 * Every value is copied from the path. No score is derived, and the ordered
 * sequences are the path's own — they are not re-sorted by timestamp.
 */
export function pathDetails(path) {
  if (!path) return null;
  const summary = evidenceSummary(path);
  const window = pathTemporalWindow(path);
  return {
    pathId: path.path_id,
    hopCount: path.hop_count,
    nodeCount: path.node_count,
    searchStatus: path.search_status,
    nodes: orderedNodes(path),
    transitions: orderedTransitions(path),
    sourceObservationId: path.source_observation_id,
    targetObservationId: path.target_observation_id,
    evidenceSummary: summary
      ? {
          relationshipCount: summary.relationship_count,
          evidenceTypes: (summary.evidence_types || []).slice(),
          confidenceMin: summary.confidence_bounds
            ? summary.confidence_bounds.min
            : null,
          confidenceMax: summary.confidence_bounds
            ? summary.confidence_bounds.max
            : null
        }
      : null,
    temporalStatus: pathTemporalStatus(path),
    temporalWindow: {
      status: window.status,
      durationSeconds: window.durationSeconds,
      known: window.known
    }
  };
}

/**
 * Exact-id synchronisation between a path node and the graph.
 *
 * A node synchronises only when the graph holds a node carrying that exact
 * `observation_id`. When it does not, `synchronized` is false and
 * `graphNodeId` is `null`, so nothing downstream can highlight a node the graph
 * never built. Synchronisation never falls back to display text, array index or
 * label similarity.
 */
export function syncNodeToGraph(observationId, graphObservationIds) {
  if (!observationId) {
    return { synchronized: false, observationId: null, graphNodeId: null };
  }
  const present = (Array.isArray(graphObservationIds) ? graphObservationIds : []).some(
    id => id === observationId
  );
  return {
    synchronized: present,
    observationId,
    graphNodeId: present ? observationId : null
  };
}

/**
 * Exact-id synchronisation between a path transition and the graph.
 */
export function syncRelationshipToGraph(relationshipId, graphEvidence) {
  if (!relationshipId) {
    return { synchronized: false, relationshipId: null, graphRecord: null };
  }
  const graphRecord = (Array.isArray(graphEvidence) ? graphEvidence : []).find(
    record => record && record.relationship_id === relationshipId
  );
  return {
    synchronized: !!graphRecord,
    relationshipId,
    graphRecord: graphRecord || null
  };
}

/**
 * Build the comparison of the paths in the comparison set.
 *
 * The result is a side-by-side of what each path carries: its id, its ordered
 * node sequence, its ordered transition sequence, its evidence summary and its
 * temporal standing. Both paths' orderings are preserved verbatim.
 *
 * Deliberately absent: any aggregate score, any ranking, and any "more
 * suspicious" ordering. The paths stay in the order they were selected, and
 * nothing is combined across them into a new figure.
 *
 * Returns `null` when fewer than two distinct paths are in the set, since a
 * one-path comparison is not a comparison.
 */
export function buildComparison(paths, state) {
  const selected = comparisonPaths(paths, state);
  if (selected.length < MAX_COMPARISON_PATHS) return null;
  const distinct = new Set(selected.map(path => path.path_id));
  if (distinct.size !== MAX_COMPARISON_PATHS) return null;

  const sides = selected.map(path => {
    const details = pathDetails(path);
    const timeline = buildTimeline(path);
    return {
      pathId: details.pathId,
      hopCount: details.hopCount,
      nodeCount: details.nodeCount,
      nodes: details.nodes,
      transitions: details.transitions,
      evidenceSummary: details.evidenceSummary,
      temporalStatus: details.temporalStatus,
      temporalWindow: details.temporalWindow,
      timeline: timeline.steps.map(step => ({
        index: step.index,
        fromObservationId: step.fromObservationId,
        toObservationId: step.toObservationId,
        relationshipId: step.relationshipId,
        evidenceType: step.evidenceType,
        relation: step.relation,
        comparable: step.comparable,
        elapsedSeconds: step.elapsedSeconds
      })),
      chronologicallyOrdered: timeline.chronologicallyOrdered
    };
  });

  const [left, right] = sides;
  return {
    pathIds: sides.map(side => side.pathId),
    sides,
    sharedObservations: left.nodes.filter(id => right.nodes.indexOf(id) !== -1),
    sharedRelationships: left.transitions.filter(
      id => right.transitions.indexOf(id) !== -1
    )
  };
}

function inferenceBadge(level) {
  return el(
    "span",
    { class: "badge inference-badge " + getInferenceClass(level) },
    String(level || "")
  );
}

function detailRow(field, label, value) {
  return el("div", { class: "path-detail__row", "data-field": field }, [
    el("span", { class: "path-detail__label" }, label),
    typeof value === "string" || typeof value === "number"
      ? el("span", { class: "path-detail__value" }, String(value))
      : value
  ]);
}

function idList(ids, truncate) {
  if (!ids || ids.length === 0) return "-";
  return el(
    "span",
    { class: "path-detail__ids" },
    ids.map(id => idSpan(id, "mono path-detail__id", truncate))
  );
}

function metaItem(label, value) {
  return el("span", { class: "path-card__meta-item" }, [
    el("span", { class: "path-card__meta-label" }, label + " "),
    typeof value === "string" || typeof value === "number"
      ? el("span", { class: "path-card__meta-value" }, String(value))
      : value
  ]);
}

/** Build the ordered node strip for a path. */
export function renderNodeStrip(path) {
  const nodes = orderedNodes(path);
  return el(
    "div",
    { class: "path-nodes", "data-count": String(nodes.length) },
    nodes.map((observationId, index) =>
      el("span", { class: "path-nodes__step" }, [
        el("span", { class: "path-nodes__index" }, String(index + 1)),
        idSpan(observationId, "mono path-nodes__id", 20)
      ])
    )
  );
}

/** Build the ordered transition strip for a path. */
export function renderTransitionStrip(path) {
  const transitions = orderedTransitions(path);
  return el(
    "div",
    { class: "path-transitions", "data-count": String(transitions.length) },
    transitions.map((relationshipId, index) =>
      el("span", { class: "path-transitions__step" }, [
        el("span", { class: "path-transitions__index" }, String(index + 1)),
        idSpan(relationshipId, "mono path-transitions__id", 20)
      ])
    )
  );
}

/**
 * Build the rendered timeline for a path.
 *
 * One row per transition, in the path's own order. A row whose temporal standing
 * is NOT_COMPARABLE or UNKNOWN is marked as such and shows no elapsed time, so
 * the panel cannot be read as asserting a chronology the evidence does not
 * carry. When the path as a whole is not chronologically ordered, that is stated
 * once, above the rows.
 */
export function renderTimeline(path) {
  const timeline = buildTimeline(path);
  const rows = timeline.steps.map(step =>
    el(
      "li",
      {
        class:
          "timeline__step" +
          (step.comparable ? " is-comparable" : " is-incomparable"),
        "data-index": String(step.index),
        "data-relation": String(step.relation),
        "data-comparable": step.comparable ? "true" : "false",
        "data-standing": step.standing,
        "data-elapsed": step.elapsedSeconds === null ? "none" : String(step.elapsedSeconds),
        "data-relationship-id": step.relationshipId || ""
      },
      [
        el("span", { class: "timeline__node mono" }, step.fromObservationId || "?"),
        el("span", { class: "timeline__edge" }, step.evidenceType || "?"),
        el(
          "span",
          { class: "timeline__relation", title: "Step 7 temporal relation" },
          step.standing
        ),
        el("span", { class: "timeline__node mono" }, step.toObservationId || "?"),
        el(
          "span",
          { class: "timeline__time mono" },
          step.elapsedSeconds === null ? "—" : formatDuration(step.elapsedSeconds)
        )
      ]
    )
  );
  const orderedNote = timeline.chronologicallyOrdered
    ? null
    : el(
        "p",
        { class: "timeline__note", "data-note": "ordering-unavailable" },
        "Temporal ordering is unavailable for at least one transition, so this " +
          "sequence is shown in path order only."
      );
  return el(
    "div",
    {
      class: "timeline",
      "data-status": timeline.status,
      "data-chronological": timeline.chronologicallyOrdered ? "true" : "false"
    },
    [
      orderedNote,
      el("ol", { class: "timeline__steps" }, rows)
    ]
  );
}

/** Build one path card, including its body. */
export function renderPathCard(path, state) {
  const selected = isSelected(state, path.path_id);
  const compared = isInComparison(state, path.path_id);
  const bodyId = "path-body-" + path.path_id;
  const expanded = selected;

  const header = el(
    "button",
    {
      type: "button",
      class: "path-card__header",
      "aria-expanded": expanded ? "true" : "false",
      "aria-controls": bodyId,
      "data-path-id": path.path_id
    },
    [
      el("span", { class: "path-card__id mono" }, truncateId(path.path_id, 24)),
      el("span", { class: "badge badge--context" }, path.hop_count + " hops"),
      el("span", { class: "path-card__meta" }, [
        metaItem("Nodes:", String(path.node_count)),
        metaItem("Status:", String(path.search_status)),
        metaItem("Duration:", formatDuration(path.duration_seconds)),
        metaItem("Temporal:", pathTemporalStatus(path))
      ])
    ]
  );

  const compareButton = el(
    "button",
    {
      type: "button",
      class: "path-card__compare",
      "aria-pressed": compared ? "true" : "false",
      "aria-label": "Compare path " + path.path_id,
      "data-path-id": path.path_id
    },
    compared ? "In comparison" : "Compare"
  );

  const summary = evidenceSummary(path);
  const body = el(
    "div",
    {
      class: "path-card__body",
      id: bodyId,
      role: "region",
      "aria-label": "Path details for " + path.path_id,
      hidden: !expanded
    },
    [
      el("div", { class: "path-card__actions" }, compareButton),
      el("div", { class: "path-detail__section" }, [
        el("h5", { class: "path-detail__title" }, "Ordered observations"),
        renderNodeStrip(path)
      ]),
      el("div", { class: "path-detail__section" }, [
        el("h5", { class: "path-detail__title" }, "Ordered transitions"),
        renderTransitionStrip(path)
      ]),
      el("div", { class: "path-detail__section" }, [
        el("h5", { class: "path-detail__title" }, "Evidence summary"),
        el(
          "div",
          { class: "path-detail__summary", "data-relationship-count": String(
            summary ? summary.relationship_count : 0
          ) },
          [
            detailRow(
              "relationships",
              "Relationships",
              String(summary ? summary.relationship_count : 0)
            ),
            detailRow(
              "evidence_types",
              "Evidence types",
              (summary && summary.evidence_types ? summary.evidence_types : []).join(", ") || "-"
            ),
            detailRow(
              "confidence_bounds",
              "Edge confidence bounds",
              summary && summary.confidence_bounds
                ? formatNumber(summary.confidence_bounds.min) +
                  " – " +
                  formatNumber(summary.confidence_bounds.max)
                : "-"
            )
          ]
        )
      ]),
      el("div", { class: "path-detail__section" }, [
        el("h5", { class: "path-detail__title" }, "Temporal"),
        el("div", { class: "path-detail__temporal" }, [
          detailRow("temporal_status", "Temporal status", pathTemporalStatus(path)),
          detailRow(
            "temporal_window",
            "Temporal window",
            pathTemporalWindow(path).known
              ? formatDuration(pathTemporalWindow(path).durationSeconds)
              : "Not established"
          )
        ])
      ]),
      el("div", { class: "path-detail__section" }, [
        el("h5", { class: "path-detail__title" }, "Temporal sequence"),
        renderTimeline(path)
      ])
    ]
  );

  return el(
    "article",
    {
      class:
        "path-card" + (selected ? " is-selected" : "") + (compared ? " is-compared" : ""),
      "data-path-id": path.path_id,
      "data-selected": selected ? "true" : "false",
      "data-compared": compared ? "true" : "false"
    },
    [header, body]
  );
}

/**
 * Render the paths section into `container`.
 *
 * When a comparison is active the section renders the comparison instead of the
 * list, so the two never compete for the same space. Returns the paths that were
 * rendered.
 */
export function renderPaths(container, paths, state) {
  const source = Array.isArray(paths) ? paths : [];
  const comparison = hasComparison(state) ? buildComparison(source, state) : null;
  const nodes = [];
  if (comparison) {
    nodes.push(renderComparison(comparison, state));
  } else {
    nodes.push(
      el(
        "div",
        { class: "paths__count", role: "status", "aria-live": "polite" },
        source.length + " investigation paths"
      )
    );
    sortPaths(source).forEach(path => nodes.push(renderPathCard(path, state)));
    if (source.length === 0) {
      nodes.push(el("div", { class: "paths__empty" }, "No investigation paths available."));
    }
  }
  replaceChildrenWith(container, nodes);
  return comparison ? comparison.pathIds : sortPaths(source).map(path => path.path_id);
}

/** Build the side-by-side comparison panel. */
export function renderComparison(comparison, state) {
  const resetButton = el(
    "button",
    {
      type: "button",
      class: "path-compare__reset",
      "data-action": "reset-comparison",
      "aria-label": "Clear path comparison"
    },
    "Clear comparison"
  );

  const side = data =>
    el(
      "div",
      { class: "path-compare__side", "data-path-id": data.pathId },
      [
        el("h5", { class: "path-compare__title mono" }, data.pathId),
        detailRow("hop_count", "Hops", String(data.hopCount)),
        detailRow("node_count", "Nodes", String(data.nodeCount)),
        detailRow(
          "temporal_status",
          "Temporal status",
          data.temporalStatus
        ),
        detailRow(
          "temporal_window",
          "Temporal window",
          data.temporalWindow.known
            ? formatDuration(data.temporalWindow.durationSeconds)
            : "Not established"
        ),
        detailRow("nodes", "Ordered observations", idList(data.nodes, 20)),
        detailRow("transitions", "Ordered transitions", idList(data.transitions, 20)),
        detailRow(
          "evidence_types",
          "Evidence types",
          data.evidenceSummary
            ? data.evidenceSummary.evidenceTypes.join(", ") || "-"
            : "-"
        ),
        detailRow(
          "confidence_bounds",
          "Edge confidence bounds",
          data.evidenceSummary
            ? formatNumber(data.evidenceSummary.confidenceMin) +
              " – " +
              formatNumber(data.evidenceSummary.confidenceMax)
            : "-"
        ),
        detailRow(
          "timeline",
          "Temporal sequence",
          el(
            "ol",
            {
              class: "path-compare__timeline",
              "data-chronological": data.chronologicallyOrdered ? "true" : "false"
            },
            data.timeline.map(step =>
              el(
                "li",
                {
                  class: "path-compare__timeline-step",
                  "data-relation": String(step.relation),
                  "data-comparable": step.comparable ? "true" : "false",
                  "data-elapsed": step.elapsedSeconds === null ? "none" : String(step.elapsedSeconds)
                },
                [
                  el("span", { class: "mono" }, (step.fromObservationId || "?") + " → " + (step.toObservationId || "?")),
                  el("span", { class: "path-compare__timeline-relation" }, step.relation || "?"),
                  el(
                    "span",
                    { class: "mono" },
                    step.elapsedSeconds === null ? "—" : formatDuration(step.elapsedSeconds)
                  )
                ]
              )
            )
          )
        )
      ]
    );

  return el(
    "section",
    {
      class: "path-compare",
      role: "group",
      "aria-label": "Path comparison",
      "data-path-count": String(comparison.sides.length)
    },
    [
      el("div", { class: "path-compare__header" }, [
        el("h4", { class: "path-compare__heading" }, "Path comparison"),
        resetButton
      ]),
      el(
        "div",
        { class: "path-compare__grid", "data-path-ids": comparison.pathIds.join(" ") },
        comparison.sides.map(side)
      ),
      el(
        "p",
        { class: "path-compare__note" },
        "Both paths are shown with their own recorded evidence and temporal " +
          "readings. No combined or path-level score is produced."
      )
    ]
  );
}

/**
 * Wire the paths section.
 *
 * Selection is a delegated click on a card header, toggling that path's body and
 * `aria-expanded`. Comparison is a separate control inside the card, so adding a
 * path to a comparison never changes which path is being browsed. Enter and
 * Space activate both, with `preventDefault` on the keydown so the browser's
 * synthesised click cannot double-apply the transition.
 */
export function initPaths(options) {
  const config = options || {};
  const container = document.getElementById("paths-content");
  let paths = Array.isArray(config.paths) ? config.paths : [];
  let state = config.state || defaultState();

  function render() {
    if (!container) return [];
    return renderPaths(container, paths, state);
  }

  function applyState() {
    if (!container) return;
    container.querySelectorAll(".path-card").forEach(card => {
      const pathId = card.getAttribute("data-path-id");
      const selected = isSelected(state, pathId);
      const compared = isInComparison(state, pathId);
      card.classList.toggle("is-selected", selected);
      card.setAttribute("data-selected", selected ? "true" : "false");
      card.classList.toggle("is-compared", compared);
      card.setAttribute("data-compared", compared ? "true" : "false");
      const header = card.querySelector(".path-card__header");
      if (header) header.setAttribute("aria-expanded", selected ? "true" : "false");
      const body = card.querySelector(".path-card__body");
      if (body) {
        if (selected) body.removeAttribute("hidden");
        else body.setAttribute("hidden", "");
      }
      const compareButton = card.querySelector(".path-card__compare");
      if (compareButton) {
        compareButton.setAttribute("aria-pressed", compared ? "true" : "false");
        compareButton.textContent = compared ? "In comparison" : "Compare";
      }
    });
  }

  function select(pathId) {
    state = selectPath(state, pathId);
    applyState();
    const record = selectedPath(paths, state);
    if (config.onSelect) config.onSelect(record);
    return record;
  }

  function compare(pathId) {
    state = toggleComparison(state, pathId);
    render();
    if (config.onCompare) config.onCompare(comparisonPaths(paths, state));
    return comparisonPaths(paths, state);
  }

  function onClick(event) {
    const reset = event.target.closest("[data-action='reset-comparison']");
    if (reset) {
      state = resetComparison(state);
      render();
      if (config.onCompare) config.onCompare([]);
      return;
    }
    const compareButton = event.target.closest(".path-card__compare");
    if (compareButton) {
      event.stopPropagation();
      compare(compareButton.getAttribute("data-path-id"));
      return;
    }
    const header = event.target.closest(".path-card__header");
    if (header) {
      select(header.getAttribute("data-path-id"));
      return;
    }
    const card = event.target.closest(".path-card");
    if (card) select(card.getAttribute("data-path-id"));
  }

  function onKeydown(event) {
    if (!isActivationKey(event.key)) return;
    const reset = event.target.closest("[data-action='reset-comparison']");
    if (reset) {
      event.preventDefault();
      reset.click();
      return;
    }
    const compareButton = event.target.closest(".path-card__compare");
    if (compareButton) {
      event.preventDefault();
      compare(compareButton.getAttribute("data-path-id"));
      return;
    }
    const header = event.target.closest(".path-card__header");
    if (header) {
      event.preventDefault();
      select(header.getAttribute("data-path-id"));
    }
  }

  if (container) {
    container.addEventListener("click", onClick);
    container.addEventListener("keydown", onKeydown);
  }

  render();

  return {
    getState: () => state,
    getPaths: () => paths,
    getSelected: () => selectedPath(paths, state),
    getComparison: () => comparisonPaths(paths, state),
    getComparisonView: () => buildComparison(paths, state),
    setPaths(next, graphEvidence) {
      paths = Array.isArray(next) ? next : [];
      if (Array.isArray(graphEvidence)) config.graphEvidence = graphEvidence;
      render();
    },
    /**
     * Select a path because the graph reported one of its nodes as selected.
     * The path is only selected when it actually contains that exact
     * `observation_id`; otherwise nothing changes here.
     */
    highlightObservation(observationId) {
      const holder = paths.find(path =>
        (path.observation_ids || []).indexOf(observationId) !== -1
      );
      if (!holder) return null;
      // `onSelect` is not called: this change came from the graph, and telling
      // the graph about the graph's own selection would loop.
      state = setSelectedPath(state, holder.path_id);
      applyState();
      return selectedPath(paths, state);
    },
    select,
    compare,
    reset() {
      state = resetComparison(state);
      render();
      return state;
    },
    destroy() {
      if (container) {
        container.removeEventListener("click", onClick);
        container.removeEventListener("keydown", onKeydown);
      }
    }
  };
}

/** Teardown handle retained for the dashboard's own lifecycle. */
let activePaths = null;

export function mountPaths(options) {
  if (activePaths) activePaths.destroy();
  activePaths = initPaths(options);
  return activePaths;
}

export function unmountPaths() {
  if (activePaths) {
    activePaths.destroy();
    activePaths = null;
  }
}