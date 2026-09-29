<div align="center">

# 🧠 BitTrace AI

### Cross-Layer AI Platform for Bitcoin Transaction Investigation

**Smart India Hackathon 2026 · SIH26146 · Brain Bytes · Team ID 131594**

[![SIH 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-0A66C2?style=for-the-badge)](https://www.sih.gov.in/)
[![PS](https://img.shields.io/badge/PS-SIH26146-6C5CE7?style=for-the-badge)](https://www.sih.gov.in/sih2026PS)
[![Theme](https://img.shields.io/badge/Theme-Blockchain%20%26%20Cybersecurity-00A86B?style=for-the-badge)](https://www.sih.gov.in/)
[![Software](https://img.shields.io/badge/Category-Software-F39C12?style=for-the-badge)](https://www.sih.gov.in/)
[![Linux](https://img.shields.io/badge/Platform-Linux%20%7C%20Offline-111827?style=for-the-badge&logo=linux)](https://www.linux.org/)

# 🎥 LIVE DEMO · YOUTUBE

### ▶️ YouTube Video Demo

**The live working demonstration of BitTrace AI will be published here.**

**YouTube Demo:** https://youtube.com/@coding-w5z?si=GVbCpjPuLZawp9lP

</div>

---

# 🏆 Smart India Hackathon 2026

| | Details |
|---|---|
| 🎯 Problem Statement | **SIH26146** |
| 📌 Title | **AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic** |
| 🧩 Theme | **Blockchain & Cybersecurity** |
| 💻 Category | **Software** |
| 🏢 Organisation | **National Technical Research Organisation (NTRO)** |
| 👥 Team ID | **131594** |
| 🚀 Team | **Brain Bytes** |
| 🧠 Solution | **BitTrace AI** |
| 🐧 Deployment | **Linux / Offline-first** |

> BitTrace AI is an offline-first investigation platform for correlating network-layer observations with Bitcoin blockchain metadata, constructing a provenance-aware evidence graph, applying AI/ML and graph intelligence, and generating explainable, prioritized investigative leads.

The SIH problem calls for bulk ingestion of transaction/network metadata, cross-layer correlation, an entity/transaction graph, a working AI/ML detection component, ranked explainable alerts, and dashboard/link-analysis visualization.

---

# 🔎 What is BitTrace AI?

**BitTrace AI** is an evidence-centric Bitcoin transaction investigation platform built around one principle:

> ## Don't just flag suspicious activity — connect the evidence and explain why it matters.

The platform brings together:

- 🌐 Network telemetry
- ₿ Bitcoin transaction metadata
- 👛 Wallet addresses
- 🔗 Transaction IDs
- 🕒 Temporal behaviour
- 🧩 Entity relationships
- 🧠 AI/ML signals
- 🕸️ Graph patterns
- 📍 IP / ASN / geolocation context
- 📑 Evidence provenance

### Core journey

**Raw Metadata → Normalization → Cross-Layer Correlation → Evidence Graph → AI/ML → Risk Scoring → Investigation Leads**

---

# 🚨 Problem We Address

SIH26146 focuses on monitoring and analysing Bitcoin transaction traffic where illicit actors may move, layer, and cash out funds while attempting to avoid conventional financial surveillance.

Relevant metadata includes:

- Timestamp
- Source IP / destination IP
- Source / destination port
- TXID
- Input wallet addresses
- Output wallet addresses
- Input amounts
- Output amounts
- Fee
- Script type
- ASN / geolocation context

The challenge is not simply storing this information. The harder problem is **connecting it across layers and extracting investigative meaning**.

| Challenge | BitTrace AI Approach |
|---|---|
| Disconnected network + blockchain data | 🔗 Cross-layer correlation |
| Large metadata volume | ⚙️ Structured ingestion + normalization |
| Hidden relationships | 🕸️ Evidence graph |
| Unusual behaviour | 🧠 Anomaly detection |
| Related entities | 🧩 Entity clustering |
| Time-based behaviour | 🕒 Temporal analysis |
| Complex transaction structures | 📊 Graph pattern analysis |
| Multiple risk signals | 🎯 Evidence fusion |
| Black-box alerts | 🔍 Explainable evidence contribution |
| Manual exploration | 🧭 Prioritized investigation paths |
| Restricted environment | 🐧 Offline Linux-first design |

---

# 💡 Core Approach

BitTrace AI follows a 7-stage investigation pipeline:

~~~text
1. DATA INGESTION
   CSV / JSON / XML
          ↓
2. NORMALIZATION
   Clean + Validate + Deduplicate
          ↓
3. CROSS-LAYER CORRELATION
   IP ↔ TXID ↔ Wallet ↔ Time
          ↓
4. EVIDENCE GRAPH
   Typed Entities + Relationships + Provenance
          ↓
5. AI / ML + GRAPH INTELLIGENCE
   Anomaly + Clustering + Temporal + Graph
          ↓
6. EVIDENCE FUSION + RISK SCORING
   Multi-signal + Explainability
          ↓
7. INVESTIGATIVE LEADS
   Ranked Entities + Transactions + Paths
~~~

---

# 🧬 What Makes BitTrace AI Different?

### 🔗 1. Cross-Layer Investigation

Network-layer and blockchain-layer information are correlated rather than analysed as isolated datasets.

Example:

**IP → TXID → Wallet → Transaction → Related Wallet → Entity**

### 🕸️ 2. Evidence-First Graph

The graph is built around typed relationships and provenance.

Important relationships can retain:

- Source
- Timestamp
- Relationship type
- Observation context
- Correlation basis

### 🧠 3. Multi-Signal Intelligence

Detection can combine:

- Anomaly signals
- Entity clustering
- Temporal behaviour
- Graph patterns
- Transaction characteristics
- Cross-layer relationships

### 🔍 4. Explainable Risk Scoring

Risk is accompanied by the evidence contributing to the flag rather than presenting an unexplained number.

### 🎯 5. Prioritized Investigative Leads

The system is designed to organize results into prioritized:

- Entities
- Transactions
- Clusters
- Network relationships
- Investigation paths

### 🐧 6. Offline-First

The core workflow is designed for Linux and controlled environments using local/open-source components.

---

# 🏗️ Solution Architecture

~~~mermaid
flowchart LR
A["📥 Network Data<br/>CSV / JSON / XML"]
B["₿ Blockchain Data<br/>TXID / Wallet / Amount"]
C["⚙️ Normalization<br/>Validation + Deduplication"]
D["🔗 Cross-Layer<br/>Correlation"]
E["🕸️ Evidence Graph<br/>Entities + Relationships"]
F["🧠 AI / ML + Graph<br/>Intelligence"]
G["🎯 Evidence Fusion<br/>Risk Scoring"]
H["🔎 Investigation Layer<br/>Leads + Evidence"]
I["📊 Dashboard<br/>Graph + Reports"]

A --> C
B --> C
C --> D
D --> E
E --> F
F --> G
G --> H
H --> I
~~~

---

# 🔬 Technical Data Flow

## 01 · Data Ingestion

Supported input direction:

- CSV
- JSON
- XML

Expected fields:

**IP · Port · Timestamp · ASN · TXID · Wallet · Amount · Fee · Script Type**

## 02 · Unified Data Model

Raw inputs are transformed into a common internal representation through:

- Schema mapping
- Validation
- Cleaning
- Entity normalization
- Deduplication
- Missing-value handling
- Type normalization

Planned tooling: **Pandas · NumPy · FastAPI · SQLite · DuckDB**

## 03 · Cross-Layer Correlation

~~~text
Network IP
   │
   ├── observed near transaction
   ↓
TXID
   │
   ├── references
   ↓
Wallet
   │
   ├── participates in
   ↓
Transaction Flow
   │
   └── connects to
   ↓
Related Wallet / Entity
~~~

Correlation dimensions:

- IP ↔ TXID
- Wallet ↔ IP
- Transaction ↔ Time
- Transaction ↔ Amount
- Wallet ↔ Wallet
- Multi-signal temporal relationships

## 04 · Evidence Graph

### Entities

🌐 IP Address · 🔌 Port · ₿ Transaction · 👛 Wallet · 🧩 Cluster · 🏷️ ASN · 📍 Geo Context

### Relationships

OBSERVED_FROM · RELATED_TO · INPUT_OF · OUTPUT_OF · CONNECTED_TO · OCCURRED_NEAR · CLUSTERED_WITH

The graph is intended to preserve **source + time + relationship semantics**, not merely display connections.

---

# 🧠 AI / ML Intelligence

SIH26146 highlights entity clustering, anomaly detection, peeling-chain / mixing detection and risk scoring.

## 🚨 Anomaly Detection

Identify statistically unusual transaction or flow behaviour using signals such as:

- Unusual transaction amounts
- Unusual frequency
- Abnormal temporal activity
- Unexpected interaction patterns
- Unusual graph connectivity

## 🧩 Entity Clustering

Group wallets or related entities using behavioural and structural characteristics such as:

- Common input/output patterns
- Shared relationships
- Transaction behaviour
- Graph structural features
- Temporal similarity

## 🕒 Temporal Analysis

Investigate sequences and movement patterns over time.

~~~text
Event A
  ↓
short time gap
  ↓
Event B
  ↓
short time gap
  ↓
Event C
~~~

## 🕸️ Graph Pattern Analysis

Potential patterns include:

- Peeling-chain-like flows
- Mixing-like structures
- Dense wallet relationships
- Rapid multi-hop movement
- Cluster-level behavioural patterns

## 🎯 Risk Scoring

Conceptually:

~~~text
Risk Score =
    Behavioural Signal
  + Temporal Signal
  + Graph Signal
  + Cross-Layer Signal
  + Entity/Cluster Signal
  + Evidence Strength
~~~

The exact weighting/model will be validated experimentally during implementation.

> **Risk scores are investigation-support signals, not automatic accusations or final determinations.**

---

# 🔍 Explainability & Evidence Contribution

A core design principle:

> **Every important alert should have an evidence trail.**

Instead of only showing:

~~~text
Wallet XYZ
Risk: High
~~~

the investigation view should expose:

~~~text
Wallet XYZ

Why flagged?
✓ Cross-layer relationship
✓ Related transaction cluster
✓ Temporal pattern
✓ Graph connectivity
✓ Anomaly contribution

Evidence
├── Network observation
├── Transaction relationship
├── Wallet relationship
├── Temporal relationship
└── Graph path
~~~

This supports analyst review and makes the output traceable.

---

# 🎯 Prioritized Investigative Leads

| Lead Type | Purpose |
|---|---|
| 👛 Entity Lead | Wallet/entity requiring review |
| ₿ Transaction Lead | Transaction requiring investigation |
| 🧩 Cluster Lead | Related group of entities |
| 🌐 Network Lead | Relevant IP/network relationship |
| 🕒 Temporal Lead | Important sequence/timing |
| 🕸️ Graph Lead | Important relationship path |
| 📑 Evidence Lead | Evidence supporting a flag |

Each lead can expose:

- Risk score
- Evidence contribution
- Related entities
- Related transactions
- Investigation path
- Source information
- Time context

---

# 🖥️ Investigation Dashboard

### 📊 Overview

- Total entities
- Transactions processed
- Alerts generated
- High-priority leads
- Risk distribution
- Activity trends

### 🕸️ Graph View

Interactive visualization of:

- Wallets
- Transactions
- IPs
- Clusters
- Relationships
- Evidence paths

### 🔎 Entity Details

For a selected entity:

- Entity/type
- Related transactions
- Related IPs
- Related wallets
- Cluster information
- Risk score
- Evidence contributions

### 🧭 Investigation Path

~~~text
IP
 ↓
TXID
 ↓
Wallet A
 ↓
Wallet B
 ↓
Cluster C
 ↓
Prioritized Lead
~~~

### 📑 Reports

Investigation summaries containing:

- Alert details
- Risk score
- Evidence
- Relationships
- Investigation path
- Supporting metadata

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| 🐍 Backend | **Python** |
| ⚡ API | **FastAPI** |
| 📦 Data Processing | **Pandas, NumPy** |
| 🕸️ Graph Analysis | **NetworkX** |
| 🗄️ Graph Storage | **Neo4j (local)** |
| 🤖 ML | **Scikit-learn** |
| 🧠 Deep Learning | **PyTorch** |
| 💾 Local Database | **SQLite / DuckDB** |
| ⚛️ Frontend | **React + TypeScript + Vite** |
| 🕸️ Graph UI | **Cytoscape.js** |
| 📈 Visualization | **Plotly** |
| 🐳 Deployment | **Docker** |
| 🐧 Target Platform | **Linux / Offline** |

---

# 📁 Planned Repository Structure

~~~text
SIH26146/
├── backend/
│   ├── api/
│   ├── ingestion/
│   ├── normalization/
│   ├── correlation/
│   ├── graph/
│   ├── detection/
│   ├── risk/
│   └── investigation/
├── frontend/
│   ├── src/
│   ├── components/
│   ├── pages/
│   ├── graph/
│   └── services/
├── data/
│   ├── sample/
│   └── schemas/
├── models/
├── notebooks/
├── docs/
├── tests/
├── scripts/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
~~~

> This is the intended development architecture. Components will be added as implementation progresses.

---

# 🔐 Security & Investigation Principles

### 🛡️ Privacy & Data Handling

- Offline-first processing
- Local data storage
- Controlled deployment
- No unnecessary external data transmission
- Dataset separation from application code
- Configurable access boundaries

### 👤 Human-in-the-Loop

~~~text
AI Signal
   ↓
Evidence
   ↓
Risk Score
   ↓
Analyst Review
   ↓
Investigation Decision
~~~

The system supports analysts; model output remains a decision-support signal.

---

# ⚠️ Risks & Mitigation

| Risk | Mitigation |
|---|---|
| Synthetic / incomplete data | Multiple datasets + scenario-based validation |
| False positives | Precision/recall + human review |
| Class imbalance | Suitable sampling + anomaly-focused evaluation |
| Incorrect clustering | Graph context + temporal consistency |
| Resource constraints | Lightweight models + batching + filtering |
| Model overconfidence | Explainable evidence contribution |
| Weak traceability | Provenance-aware graph relationships |
| Offline limitations | Local/open-source architecture |
| Dataset bias | Multiple behavioural scenarios |

---

# 📏 Evaluation Strategy

### Machine Learning

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC where applicable
- Anomaly detection quality

### Investigation

- Alert prioritization quality
- Evidence coverage
- Investigation-path completeness
- Cross-layer correlation accuracy
- Entity clustering quality
- Analyst review efficiency

### System

- Ingestion throughput
- Query latency
- Graph response time
- Memory usage
- Offline deployment reliability

---

# 🧪 Dataset Strategy

The SIH problem statement specifies synthetic data modelled on real Bitcoin P2P / transaction fields.

Development can use controlled datasets containing:

~~~text
timestamp
src_ip
dst_ip
src_port
dst_port
txid
input_addresses[]
output_addresses[]
input_amounts[]
output_amounts[]
fee
script_type
geo / ASN context
~~~

External research datasets used for experimentation will be clearly separated from SIH-provided or synthetic evaluation data.

---

# 🚀 Build Roadmap

~~~text
PHASE 01 ─ Foundation
   Repository · Environment · Schemas · Sample Data

PHASE 02 ─ Data Layer
   CSV/JSON/XML · Validation · Normalization · Storage

PHASE 03 ─ Correlation
   Network ↔ Blockchain · Temporal Correlation · Entity Normalization

PHASE 04 ─ Evidence Graph
   Entities · Relationships · Provenance · Graph Queries

PHASE 05 ─ Intelligence
   Anomaly Detection · Clustering · Temporal · Graph Patterns

PHASE 06 ─ Risk & Investigation
   Evidence Fusion · Explainable Scoring · Lead Prioritization

PHASE 07 ─ Dashboard
   Overview · Graph · Entity Explorer · Alerts · Reports

PHASE 08 ─ Validation
   Benchmarks · False Positives · Performance · Offline Deployment

PHASE 09 ─ Demo
   End-to-End Scenario · Working Demo · Documentation · YouTube
~~~

---

# 🏁 SIH26146 Requirement Coverage

| SIH Requirement | BitTrace AI Coverage |
|---|---|
| Bulk metadata ingestion | ✅ CSV / JSON / XML pipeline |
| Network metadata | ✅ IP / port / timestamp / ASN |
| Blockchain metadata | ✅ TXID / wallet / amount / fee / script |
| Cross-layer correlation | ✅ IP ↔ TXID ↔ Wallet ↔ Time |
| Entity / transaction graph | ✅ Evidence graph |
| AI/ML model | 🔧 Intelligence layer |
| Entity clustering | 🔧 Planned |
| Anomaly detection | 🔧 Planned |
| Peeling / mixing analysis | 🔧 Planned |
| Risk scoring | 🔧 Evidence fusion |
| Explainable alerts | 🔧 Evidence contribution |
| Ranked investigative leads | 🔧 Investigation layer |
| Dashboard | 🔧 React + Cytoscape.js + Plotly |
| Link analysis | 🔧 Interactive graph |
| Offline Linux solution | 🎯 Core architecture |

**Legend:** ✅ addressed in architecture · 🔧 implementation planned/in progress · 🎯 core requirement

---

# 👥 Team Brain Bytes

### Smart India Hackathon 2026 · Team ID 131594

| Role | Name | Stream | Year |
|---|---|---|---|
| 👑 **Team Leader** | **Nirav Vala** | B.Sc. IT | 2nd Year |
| 👤 Team Member | **Sammi Yadav** | B.Sc. IT | 3rd Year |
| 👤 Team Member | **Maimuna Vala** | B.Sc. IT | 3rd Year |
| 👤 Team Member | **Naaz Khalani** | B.Sc. IT | 3rd Year |
| 👤 Team Member | **Vaishaliba Mahendrasinh** | B.Sc. IT | 3rd Year |
| 👤 Team Member | **Archiben Nagar** | B.Sc. IT | 3rd Year |

**Team:** Brain Bytes  
**Team ID:** 131594  
**Problem Statement:** SIH26146

---

# 🌐 SIH & Research Resources

### 🇮🇳 Smart India Hackathon

- [SIH Official Website](https://www.sih.gov.in/)
- [SIH 2026 Problem Statements](https://www.sih.gov.in/sih2026PS)
- [SIH26146 Official Problem Statement](https://www.sih.gov.in/sih2026PS)

### 🏛️ Organisation

- [National Technical Research Organisation (NTRO)](https://ntro.gov.in/)

### ₿ Bitcoin

- [Bitcoin Whitepaper — Satoshi Nakamoto](https://bitcoin.org/bitcoin.pdf)

### 📊 Dataset

- [Elliptic Bitcoin Dataset — Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

### 📚 Research

- [Weber et al. — Anti-Money Laundering in Bitcoin](https://arxiv.org/abs/1908.02591)

### 🔬 Industry Research

- [Chainalysis — Crypto Crime Research](https://www.chainalysis.com/blog/)

### 🌍 Geolocation

- [MaxMind GeoLite2](https://dev.maxmind.com/geoip/geolite2/)

---

# 🎬 LIVE DEMO

<div align="center">

## ▶️ BitTrace AI — Live Working Demonstration

**Dataset → Ingestion → Normalization → Correlation → Evidence Graph → AI/ML → Risk Scoring → Investigation Dashboard**

### 📺 YouTube

**Live demo / project channel:** https://youtube.com/@coding-w5z?si=GVbCpjPuLZawp9lP

[🔴 YouTube — Coding](https://youtube.com/@coding-w5z?si=GVbCpjPuLZawp9lP)

</div>

---

# 📌 Project Status

> 🚧 **Active Development**

- [ ] Repository foundation
- [ ] Dataset schemas
- [ ] Data ingestion
- [ ] Normalization
- [ ] Cross-layer correlation
- [ ] Evidence graph
- [ ] AI/ML detection
- [ ] Entity clustering
- [ ] Risk scoring
- [ ] Investigation dashboard
- [ ] Evidence-backed reports
- [ ] End-to-end offline demo
- [ ] YouTube live demo

---

# 🤝 Development Principles

- Modular components
- Reproducible experiments
- Clear data schemas
- Testable analytical modules
- Explainable outputs
- Evidence traceability
- Offline-compatible dependencies
- Documentation alongside implementation

---

# 📜 Disclaimer

BitTrace AI is an **investigative decision-support prototype** developed for Smart India Hackathon 2026.

Model outputs, anomaly scores and risk scores are analytical signals. They are not, by themselves, proof of wrongdoing or automatic determinations of criminal activity. Human review and appropriate investigative procedures remain essential.

---

<div align="center">

# 🧠 BitTrace AI

### Cross-Layer AI Platform for Bitcoin Transaction Investigation

**Built by Brain Bytes for Smart India Hackathon 2026**

[![GitHub](https://img.shields.io/badge/GitHub-SIH26146-181717?style=for-the-badge&logo=github)](https://github.com/codebynv/SIH26146)
[![SIH](https://img.shields.io/badge/SIH-2026-0A66C2?style=for-the-badge)](https://www.sih.gov.in/)
[![PS](https://img.shields.io/badge/PS-SIH26146-6C5CE7?style=for-the-badge)](https://www.sih.gov.in/sih2026PS)

**Trace the signal. Connect the evidence. Prioritize the investigation.**

</div>
