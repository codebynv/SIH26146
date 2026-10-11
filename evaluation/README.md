# BitTrace Step 10 evaluation

A deterministic, offline functional evaluation of the whole pipeline. It runs
Steps 1 through 9A end to end against a fixed corpus and reports eight
dimensions of behaviour, sixteen required functional cases, and twenty-one
invariants.

```
python -m evaluation.run_evaluation
```

```
corpus        12 observations from 3 files (csv, json, xml)
correlation   40 relationships, 6 evidence types
graph         10 nodes, 40 edges, 0 orphan nodes
paths         11 found, deepest 5 hops
ml            2 clusters, anomaly bands LOW=8, MODERATE=4
priority      score 0.6884615384615384615384615384615385, band HIGH, 9/11 signals present
levels        CORRELATED, INFERRED, OBSERVED, REQUIRES_VERIFICATION

determinism   in-process identical, cross-process identical, order-independent yes
invariants    21/21 hold
expected      matches evaluation/expected/expected_output.json
digest        9f4d5ef8667008c99f54152d99818bc3275ca82bb85ec8ae25d23055542ffd12
```

Exit code `0` means every invariant held, both runs matched each other, a fresh
interpreter produced the same bytes, input order made no difference, and the run
matched the stored expectation.

## What it measures

| Dimension | Question it answers |
| --- | --- |
| `ingestion_normalization` | Do all three formats ingest to the same unified model, with problems reported rather than dropped? |
| `behavioral_features` | Does Step 5 produce a full feature set without claiming attribution? |
| `evidence_correlation` | Which evidence types and inference levels does the rule engine actually produce? |
| `evidence_graph` | Does every relationship survive into the graph, with no orphan nodes? |
| `path_determinism` | Are paths reproducible, and does temporal comparison refuse to invent elapsed time? |
| `ml_determinism` | Are anomaly scores and clusters reproducible, and did clustering recover the grouping the corpus declares? |
| `fusion_priority` | Does fusion carry all four inference levels, cite evidence, and report missing signals? |
| `dashboard_payload` | Does the dashboard payload match the fusion result and preserve sorted id ordering? |

The sixteen functional cases are declared in `harness.CASE_COVERAGE`, each
naming the dimension that exercises it and the source file that implements it.
`tests/test_step10_evaluation.py` checks the declaration is complete and that
every referenced file exists.

## Metrics deliberately not computed

`accuracy`, `precision`, `recall`, `f1`, `auc`, `detection_rate`,
`false_positive_rate`.

The corpus is synthetic and carries no ground-truth labels, so every one of
these numbers would be invented. A test asserts both halves of this: that no key
anywhere in the payload is a performance metric, and that no corpus file
contains a label column. What is measured instead is repeatability, order
independence, and whether each stage reported what it actually had.

The fused score is labelled as neither a probability nor an accuracy, and
certainty is reserved for exact deterministic matches.

## Determinism

Two runs in the same process produce identical bytes. So does a fresh
interpreter: the runner re-executes itself in a subprocess and compares the
canonical output, because a module-level cache or memo would survive an
in-process repeat while hiding exactly the drift this is meant to find.

Input order is checked by re-running the pipeline with reversed and rotated
observation lists. Correlation, graph, paths and fusion must be identical.

Canonicalisation sorts keys, renders `Decimal` as exact text and `datetime` as
UTC ISO-8601, so the digest depends on content rather than on formatting.

### Run-scoped substitutions

Two values cannot be content-derived: the ingestion run identifier and the
library version. Rather than deleting them, the canonical form substitutes
`<run-scoped>` and the report names every path it touched:

- `behavioral_features.profiles[0].features[22].value`
- `dashboard_payload.provenance.ingestion_id`

Those paths are recorded in the report and enumerated in the tests. A missing
substitution list is treated as a change worth investigating, not as a clean
result.

Metadata such as `generated_at` lives in the report only, never inside
`dimensions`, so the clock cannot cause a comparison failure.

## The stored expectation

`expected/expected_output.json` is a compact projection, not a dump. It keeps
counts, evidence types, inference levels, representative identifiers, ordering
and statuses. It deliberately omits per-feature values, raw anomaly scores and
edge confidences, all of which live in the full report.

The test suite enforces both directions of this: a size ceiling so a dump cannot
be reintroduced, and assertions that the kept fields are still present.

Regenerating it is deliberate and visible in review:

```
python -m evaluation.run_evaluation --write-expected
```

When a stored expectation is absent the run fails rather than skipping. The
diff reports the failing path (`$.fusion_priority.fused_score`), not just a
mismatch count.

## Layout

```
evaluation/
  scenarios/flow_combined.csv   8 primary records
  scenarios/flow_records.json   2 records, one with a naive timestamp
  scenarios/flow_records.xml    2 records, one with no timestamp
  canonical.py                  canonical form, digests, run-scoped rules
  corpus.py                     ingestion, ordering, correlation config
  harness.py                    dimensions, cases, invariants
  run_evaluation.py             CLI, report, expected-output management
  expected/expected_output.json compact stored expectation
  reports/latest.json           full report
```

## Options

| Flag | Effect |
| --- | --- |
| `--write-expected` | Overwrite the stored expectation and verify against it. |
| `--no-expected` | Skip the comparison. Use when measuring a run you expect to change. |
| `--report PATH` | Write the report elsewhere. |
| `--quiet` | Suppress the summary. |

## The corpus

Twelve observations across CSV, JSON and XML, containing deliberately awkward
records: a naive timestamp, a missing timestamp, a duplicate, a low-information
record, and a four-member high-burst group that differs from the rest in amount
alone. Nothing tells the clustering stage that those four belong together, so
recovering the split is the algorithm working rather than merely running.

Ingestion is asserted to be read-only over the caller's files by hashing each
input before and after a run.

## Offline

The evaluation opens no socket. `tests/test_step10_evaluation.py` enforces this
by disabling `socket.socket` and `socket.create_connection` for the duration of
a full run, and by scanning the package for network client imports. A functional
evaluator that reached the network could return different numbers on a connected
machine than on an air-gapped one, which would make the determinism result
worthless.