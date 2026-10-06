# BitTrace AI

**SIH26146 - AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**

### Project Description

BitTrace AI is an offline, open-source investigative intelligence platform built for the
**Smart India Hackathon 2026** challenge *SIH26146*. It ingests raw network and blockchain
data, validates and normalises it, correlates network traffic against on-chain activity,
derives behavioural profiles, scores anomaly risk, and serves the result to a dashboard
API. Every stage runs on-device with no cloud dependency.

### Development Status

| Step | Component | Status |
|------|-----------|--------|
| 1 | Foundation: FastAPI app, `/`, `/health`, package layout | Implemented |
| 2 | Ingestion engine: content-based format detection, CSV/JSON/XML parsers, aliases, per-field validation, provenance, conservative deduplication, secure uploads, `POST /ingest` | Implemented |
| 3 | Unified data model: immutable Pydantic models, observation classification, `normalized_to_unified` adapter | Implemented |
| 4A | Evidence identity: canonical evidence model and content-derived identity | Implemented |
| 4B | Correlation engine: rule definitions and evaluation, provenance recording | Implemented |
| 4C | Evidence graph: graph models and builder | Implemented |
| 5 | Behavioural analysis: feature construction, models, profiles | Implemented |
| 6 | ML: preprocessing, anomaly scoring, clustering, explanations | Implemented |
| 7 | Investigation engine: path search, temporal comparison | Implemented |
| 8 | Fusion and explainability: signals, monotonic priority fusion, explanations | Implemented |
| 9A | Dashboard API: read-only endpoints over the fused result | Implemented |
| 9B | Frontend: navigation, search, graph, evidence, paths | Implemented |
| 10 | Evaluation: deterministic offline harness, expected outputs, reports, container, Linux and security checks | Implemented |

### Architecture

```
  raw bytes
      |
      v
  [ format detection ]  content-based, never the file extension
      |
      v
  [ CSV / JSON / XML parser ]  -> raw records + locator ("row 2", "$[0]")
      |
      v
  [ alias resolution + per-field validation ]  -> NormalizedRecord
      |                                          (+ provenance, + issues)
      v
  [ conservative deduplication ]  identical evidence only
      |                              (canonical fields + unmapped extras + issues)
      v
  [ normalized_to_unified ]  -> UnifiedObservation
                                   (NETWORK | BLOCKCHAIN | COMBINED | UNKNOWN)
      |
      v
  [ Step 4 correlation rules ]  -> EvidenceRelationship (OBSERVED / CORRELATED / INFERRED)
      |                              (+ analyst-levelled REQUIRES_VERIFICATION)
      v
  [ Step 4C evidence graph ]  -> graph nodes + edges, provenance on every edge
      |
      +--> [ Step 5 behavioural features ]  -> profiles, no attribution
      +--> [ Step 6 anomaly + clustering ]  -> scores, bands, clusters
      +--> [ Step 7 path search + temporal ]  -> multi-hop paths, comparability
      |
      v
  [ Step 8 fusion ]  -> signals, monotonic priority score, explanation
      |
      v
  [ Step 9A dashboard API ]  -> read-only, sorted ids, deterministic payload
```

### Canonical SIH26146 fields

`timestamp`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `txid`, `input_addresses[]`,
`output_addresses[]`, `input_amounts[]`, `output_amounts[]`, `fee`, `script_type`,
`geo_country`, `asn`.

### Design decisions worth knowing

- **Timestamps.** Timezone-aware values are converted to UTC. A naive timestamp is
  preserved *as naive* and flagged with a `NAIVE_TIMESTAMP` warning; it is never
  silently stamped with the host's local timezone. Bare numeric epoch values are
  refused because seconds-vs-milliseconds is undecidable.
- **Deduplication is deliberately weak.** Two records collapse only when they are
  the *same evidence*: every canonical field, every unmapped field, and every
  reported issue identical. Same TXID, same IP pair or same fee is never a
  duplicate, and neither is the same fourteen SIH columns carrying a different
  `block_height` alongside. Only provenance may be dropped by a merge, which is
  what makes a merge lossless rather than destructive.
- **Unmapped fields are evidence, not decoration.** Any source column with no
  canonical equivalent is preserved in `extra`, and it takes part in the
  deduplication identity *and* in the derived `observation_id`, so distinct
  unmapped values can never be collapsed away. An unmapped value whose type has
  no defined equality (anything outside JSON-like data) is treated as
  **non-deduplicable**: it never merges with anything and never shares an
  `observation_id`. Inventing equality for a type we cannot compare would
  silently merge unrelated evidence, which is the one failure a forensic
  pipeline cannot recover from.
- **Repeated XML siblings are never overwritten.** Two `<script_type>` elements
  become a list; a value that cannot map onto a canonical scalar is reported as a
  validation error with both values retained rather than silently resolved to one.
- **`txid` is not an observation identifier.** `observation_id` is a separate,
  deterministic UUIDv5 derived from observation content.
- **Preserving data is not the same as claiming identity.** A `geo_country` or
  `asn` supplied with no IP address anywhere is still canonical SIH data, so it
  is kept on the `NetworkObservation` rather than discarded or demoted to
  `extra`. That does **not** make the record a network observation:
  `has_network_identity()` still requires an IP address, so such a record
  classifies as `UNKNOWN` (or `BLOCKCHAIN` if it also carries transaction
  structure). No address, owner or location is ever inferred; a preserved
  `geo_country` is reproduced exactly as supplied.
- **Reading evidence is never destructive.** `ingest_path()` leaves a caller's
  source file untouched unless the caller explicitly declares it owns the file
  (`owned=True`), which is how the `/ingest` endpoint disposes of its own staged
  upload. Ownership is never inferred from a filename.
- **No attribution.** A record containing both an IP and a TXID is classified
  `COMBINED`, which is a statement about the record's *contents* only. It is not
  evidence that the IP belongs to the wallet. Behavioural profiles describe
  measured behaviour and never name an owner.
- **Inference is labelled, never laundered.** `OBSERVED` is reserved for direct
  observation, `CORRELATED` for rule-derived links, `INFERRED` for derived
  judgements, and `REQUIRES_VERIFICATION` for analyst-levelled links. Certainty
  (confidence 1.0) is reserved for exact deterministic matches; an inference
  claiming it would present a judgement as a fact.
- **A fused priority score is not a probability.** It is a monotonic score over
  the signals that were available, and the explanation names both the signals
  that contributed and the ones that were missing.
- **Uploads.** The client filename is reduced to a bare display name and never
  used to build a path; uploads are staged under server-generated temporary names
  and always cleaned up. The format is decided by content, and a contradicting
  `declared_format` is rejected rather than trusted.

### Repository Layout

```
BitTrace/
├── backend/
│   ├── requirements.txt          pinned runtime dependencies
│   ├── requirements-dev.txt      test-only dependencies
│   └── app/
│       ├── main.py               FastAPI app, /, /health, POST /ingest
│       ├── domain/               Step 3: enums, validation, models, adapters
│       ├── ingestion/            Step 2: aliases, dedup, models, uploads,
│       │                         service, parsers/{base,csv,json,xml}
│       ├── correlation/          Step 4B: enums, models, rules, engine, provenance
│       ├── graph/                Step 4C: models, builder
│       ├── behavior/             Step 5: features, models, profiles
│       ├── ml/                   Step 6: preprocessing, anomaly, clustering,
│       │                         explanations
│       ├── investigation/        Step 7: models, engine, paths, temporal
│       ├── explainability/       Step 8: signals, fusion, models
│       └── dashboard/            Step 9A: adapter, models, api
├── evaluation/                   Step 10: scenarios, harness, CLI, expectations
│   ├── README.md                 how to run it and what it does not claim
│   ├── scenarios/                fixed CSV/JSON/XML corpus
│   ├── canonical.py              canonical form, digests, run-scoped rules
│   ├── corpus.py                 ingestion, ordering, correlation configuration
│   ├── harness.py                dimensions, cases, invariants
│   ├── run_evaluation.py         CLI, report, expected-output management
│   ├── expected/                 the reviewed contract fixture
│   └── reports/                  generated full report (git-ignored)
├── frontend/
├── tests/                        2214 tests across Steps 1-10
├── Dockerfile                    two-stage, non-root runtime image
├── .dockerignore
└── pytest.ini
```

### Technology Stack

- **Backend**: Python 3.13, FastAPI, Pydantic v2, Uvicorn
- **Testing**: pytest, httpx (through FastAPI's `TestClient`)
- **Packaging**: pinned `requirements.txt`, multi-stage `Dockerfile`
- **Evaluation**: no third-party dependency; runs on the standard library plus the
  application itself
- **Frontend**: static HTML/CSS/JS with no build step required to serve

### Offline Operation

The platform is designed to run locally on Linux (or WSL on Windows). No cloud
services are used. The Step 10 evaluation asserts this rather than assuming it:
the full harness runs with `socket.socket.connect` disabled, so an accidental
network dependency fails the suite instead of quietly changing the numbers on a
connected machine.

### Setup Instructions

1. **Clone the repository** (already initialized).

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Linux / macOS
   # .venv\Scripts\activate           # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt -r backend/requirements-dev.txt
   ```

4. **Run the FastAPI server**
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
   ```
   Then open `http://127.0.0.1:8000/docs` for the API browser, or serve
   `frontend/index.html` against the same origin.

5. **Run the test suite**
   ```bash
   pytest
   ```

6. **Run the Step 10 evaluation**
   ```bash
   python -m evaluation.run_evaluation
   ```
   See [`evaluation/README.md`](evaluation/README.md) for what it measures, what it
   deliberately does not claim, and how to regenerate the stored expectation.

### Container

```bash
docker build -t bittrace:latest .
docker run --rm -p 8000:8000 bittrace:latest
```

The image is a two-stage build: dependencies are resolved and byte-compiled in the
builder, and the runtime stage copies only the virtual environment, `backend/` and
`evaluation/`. It runs as an unprivileged user (`uid 10001`, `nologin`), sets
`PYTHONHASHSEED=0` so set iteration order cannot break determinism, and declares a
`HEALTHCHECK` against `/health`. Tests, the frontend and documentation are excluded
from the image via `.dockerignore`.

To verify the evaluation runs inside the image with no network at all:

```bash
docker run --rm --network none --entrypoint python bittrace:latest \
    -m evaluation.run_evaluation --quiet
```

### Determinism

The evaluation digest is stable across processes and independent of input order.
Where a value genuinely cannot be content-derived, it is substituted with a visible
`<run-scoped>` placeholder and the path is listed in the report rather than being
quietly dropped. The two known cases are the ingestion run identifier and the library
version.

---

*Repository maintained by **Brain Bytes** for SIH 2026.*