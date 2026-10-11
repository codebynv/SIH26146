/**
 * Parity harness for `frontend/js/temporal.js` against Step 7.
 *
 * The frontend must not re-decide temporal semantics; it must report exactly
 * what `backend/app/investigation/temporal.py` established. The cases are built
 * by the Python test using the real Step 7 functions, so a divergence in
 * relation, comparability or elapsed time fails here rather than being papered
 * over by a hand-written fixture.
 *
 * Reads a JSON array of cases on stdin and writes a JSON array of the frontend's
 * reading of each case on stdout.
 *
 *   echo '[{"label":"...","flow":["..."],"expected":{...}}]' | node run_temporal_parity.mjs
 */

import {
  TEMPORAL_ORDER,
  PATH_TEMPORAL_STATUS,
  parseTemporalFlowEntry,
  parseTemporalFlow,
  isComparableStep,
  describeTemporalStanding,
  pathTemporalStatus,
  pathTemporalWindow,
  isChronologicallyOrdered,
  buildTimeline
} from "../../frontend/js/temporal.js";

function readStdin() {
  return new Promise(resolve => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", chunk => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", () => resolve(""));
  });
}

const raw = (await readStdin()).trim();
const cases = raw ? JSON.parse(raw) : [];

const report = cases.map(testCase => {
  const flow = Array.isArray(testCase.flow) ? testCase.flow : [];
  const parsed = parseTemporalFlow(flow);
  const entries = parsed.map(step => ({
    fromObservationId: step.unparsed ? null : step.fromObservationId,
    toObservationId: step.unparsed ? null : step.toObservationId,
    evidenceType: step.unparsed ? null : step.evidenceType,
    relation: step.unparsed ? null : step.relation,
    elapsedSeconds: step.unparsed ? null : step.elapsedSeconds,
    comparable: isComparableStep(step),
    standing: describeTemporalStanding(step),
    unparsed: step.unparsed === true
  }));

  // Every single-entry flow string must also parse on its own, so a caller that
  // reads one transition at a time cannot see a different answer.
  const single = flow.length === 1 ? parseTemporalFlowEntry(flow[0]) : null;

  const path = {
    path_id: testCase.label || "path-parity",
    observation_ids: testCase.observationIds || [],
    relationship_ids: testCase.relationshipIds || [],
    temporal_flow: flow,
    temporal_status: testCase.temporalStatus || PATH_TEMPORAL_STATUS.UNKNOWN,
    duration_seconds: testCase.durationSeconds === undefined ? null : testCase.durationSeconds
  };

  return {
    label: testCase.label || null,
    entries,
    singleEntryMatches: single === null ? null : single.relation === entries[0].relation,
    chronologicallyOrdered: isChronologicallyOrdered(parsed),
    status: pathTemporalStatus(path),
    window: pathTemporalWindow(path),
    timelineSteps: buildTimeline(path).steps.map(step => ({
      relation: step.relation,
      comparable: step.comparable,
      standing: step.standing,
      elapsedSeconds: step.elapsedSeconds,
      unparsed: step.unparsed
    })),
    // Vocabulary must be Step 7's, not a frontend invention.
    orderVocabulary: Object.values(TEMPORAL_ORDER).slice(),
    statusVocabulary: Object.values(PATH_TEMPORAL_STATUS).slice()
  };
});

process.stdout.write(JSON.stringify(report));