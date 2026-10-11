/**
 * Temporal semantics for the dashboard (Step 9B Phase 3).
 *
 * Step 7 (`backend/app/investigation/temporal.py`) is the only layer allowed to
 * decide whether two timestamps can be compared. This module does not
 * re-decide anything: it preserves what Step 7 reported and refuses to invent
 * what Step 7 declined to establish. In particular it never converts a naive
 * timestamp to UTC, never attaches an elapsed time to a pair Step 7 marked
 * `NOT_COMPARABLE` or `UNKNOWN`, and never reorders a path whose transitions
 * could not be ordered.
 *
 * A pure module: no DOM, no store, no side effects. The renderer in
 * `paths.js` consumes these functions, and the Phase 3 tests import them
 * directly.
 */

/** Step 7's `TemporalOrder` vocabulary, verbatim. */
export const TEMPORAL_ORDER = Object.freeze({
  BEFORE: "BEFORE",
  SAME_TIME: "SAME_TIME",
  AFTER: "AFTER",
  NOT_COMPARABLE: "NOT_COMPARABLE",
  UNKNOWN: "UNKNOWN"
});

/** Step 7's `PathTemporalStatus` vocabulary, verbatim. */
export const PATH_TEMPORAL_STATUS = Object.freeze({
  COMPARABLE: "COMPARABLE",
  PARTIALLY_COMPARABLE: "PARTIALLY_COMPARABLE",
  NOT_COMPARABLE: "NOT_COMPARABLE",
  UNKNOWN: "UNKNOWN"
});

const ORDER_VALUES = Object.freeze(Object.values(TEMPORAL_ORDER));
const STATUS_VALUES = Object.freeze(Object.values(PATH_TEMPORAL_STATUS));

/**
 * One entry of the `temporal_flow` tuple the dashboard API emits:
 *
 *   "<from> -> <to> (<EVIDENCE_TYPE>, <RELATION>, elapsed=<seconds|Nones>s)"
 *
 * The relation is matched only against Step 7's own vocabulary, so an
 * unexpected relation yields no match rather than a guess.
 */
const FLOW_PATTERN = new RegExp(
  "^(.+) -> (.+) \\(([^,]+), (" + ORDER_VALUES.join("|") +
  "), elapsed=(None|-?[0-9.]+)s\\)$"
);

/**
 * Parse one `temporal_flow` string into typed fields.
 *
 * Returns `null` when the text does not match the documented shape. `null`
 * means "not understood", and every consumer of a `null` step reports UNKNOWN
 * rather than reconstructing a value. `elapsedSeconds` is `null` unless the
 * source carried a number.
 */
export function parseTemporalFlowEntry(text) {
  if (typeof text !== "string") return null;
  const match = FLOW_PATTERN.exec(text.trim());
  if (!match) return null;
  const rawElapsed = match[5];
  return {
    fromObservationId: match[1],
    toObservationId: match[2],
    evidenceType: match[3],
    relation: match[4],
    elapsedSeconds: rawElapsed === "None" ? null : Number(rawElapsed)
  };
}

/**
 * Parse a whole `temporal_flow` tuple, preserving path order.
 *
 * An entry that does not parse is kept as `{ unparsed: true, raw: text }` rather
 * than dropped, so a rendered timeline still accounts for every transition the
 * path carries.
 */
export function parseTemporalFlow(flow) {
  const entries = Array.isArray(flow) ? flow : [];
  return entries.map(text => {
    const parsed = parseTemporalFlowEntry(text);
    return parsed === null ? { unparsed: true, raw: String(text) } : parsed;
  });
}

/**
 * Whether a parsed step established an ordering.
 *
 * `NOT_COMPARABLE` (mixed timestamp awareness) and `UNKNOWN` (a timestamp is
 * missing) never did, so neither yields an elapsed time or a before/after
 * reading anywhere in this module.
 */
export function isComparableStep(step) {
  if (!step || step.unparsed) return false;
  return step.relation !== TEMPORAL_ORDER.NOT_COMPARABLE &&
    step.relation !== TEMPORAL_ORDER.UNKNOWN;
}

/**
 * A short, honest label for one step's temporal standing. Preserves Step 7's
 * own vocabulary instead of collapsing everything into a yes/no.
 */
export function describeTemporalStanding(step) {
  if (!step || step.unparsed) return TEMPORAL_ORDER.UNKNOWN;
  if (step.relation === TEMPORAL_ORDER.NOT_COMPARABLE) return TEMPORAL_ORDER.NOT_COMPARABLE;
  if (step.relation === TEMPORAL_ORDER.UNKNOWN) return TEMPORAL_ORDER.UNKNOWN;
  return step.relation;
}

/**
 * The path's own `temporal_status`, preserved verbatim. An unrecognised value
 * is reported as UNKNOWN rather than passed through as something unknown to
 * Step 7.
 */
export function pathTemporalStatus(path) {
  const status = path && path.temporal_status;
  return STATUS_VALUES.indexOf(status) !== -1 ? status : PATH_TEMPORAL_STATUS.UNKNOWN;
}

/**
 * Whether a duration is usable: a finite number and nothing else.
 *
 * `null` (Step 7 could not compare the endpoints), `undefined` and any
 * non-numeric value are all unusable.
 */
export function isComparableDuration(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * The path's overall temporal window.
 *
 * `duration_seconds` is passed through only when Step 7 actually established
 * it. A path reporting `NOT_COMPARABLE` or `UNKNOWN` carries `null` and is
 * returned as `durationSeconds: null, known: false`; the window is never
 * reconstructed from the individual steps.
 */
export function pathTemporalWindow(path) {
  const status = pathTemporalStatus(path);
  const durationSeconds = path ? path.duration_seconds : null;
  const comparable = isComparableDuration(durationSeconds);
  return {
    status,
    durationSeconds: comparable ? durationSeconds : null,
    known: comparable
  };
}

/**
 * Whether a path's node sequence may be read as a chronological ordering.
 *
 * Only when *every* transition established an ordering. One `NOT_COMPARABLE` or
 * `UNKNOWN` step makes the sequence non-chronological as a whole, and the
 * sequence is then shown in path order with that stated, never re-sorted into a
 * chronology the evidence does not support. An empty sequence is not ordered.
 */
export function isChronologicallyOrdered(steps) {
  const list = Array.isArray(steps) ? steps : [];
  if (list.length === 0) return false;
  return list.every(step => isComparableStep(step));
}

/**
 * Build the rendered timeline for a path.
 *
 * Transition order is the path's own order. `establishedOrder` is false for a
 * NOT_COMPARABLE or UNKNOWN step and `elapsedSeconds` is then `null`, so the UI
 * cannot imply a chronology the evidence does not carry.
 */
export function buildTimeline(path) {
  const parsed = parseTemporalFlow(path ? path.temporal_flow : []);
  const relationshipIds = path && Array.isArray(path.relationship_ids)
    ? path.relationship_ids
    : [];
  return {
    pathId: path ? path.path_id : null,
    status: pathTemporalStatus(path),
    chronologicallyOrdered: isChronologicallyOrdered(parsed),
    steps: parsed.map((step, index) => {
      const comparable = isComparableStep(step);
      return {
        index,
        fromObservationId: step.fromObservationId === undefined ? null : step.fromObservationId,
        toObservationId: step.toObservationId === undefined ? null : step.toObservationId,
        relationshipId:
          relationshipIds[index] === undefined ? null : relationshipIds[index],
        evidenceType: step.evidenceType === undefined ? null : step.evidenceType,
        relation: step.relation === undefined ? null : step.relation,
        comparable,
        establishedOrder: comparable,
        standing: describeTemporalStanding(step),
        elapsedSeconds: comparable && typeof step.elapsedSeconds === "number"
          ? step.elapsedSeconds
          : null,
        unparsed: step.unparsed === true,
        raw: step.raw === undefined ? null : step.raw
      };
    })
  };
}