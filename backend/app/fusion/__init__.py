"""Deterministic evidence fusion, risk bands and explainability (SIH26146 - Step 8).

Step 8 is the single disclosed table where the results of earlier layers - the
Step 4C recorded-evidence graph, Step 5 behavioural profiles, Step 6 anomaly and
cluster results, and Step 7 investigation paths with their temporal structure -
are combined into one priority-ranking figure per subject profile, with the
weights, thresholds and normalisation caps fully disclosed and re-playable from
the assessment itself.

What it contains
----------------
* ``fusion.models`` -- the typed vocabulary: signals, families, statuses, risk
  bands, contributions, confidence context, the frozen configuration and the
  self-verifying assessment.
* ``fusion.signals`` -- the adapters that read Steps 4-7 outputs and emit the
  11 typed signals about one profile.
* ``fusion.fusion`` -- the engine that weights and averages the signals under a
  configuration.
* ``fusion.explain`` -- the deterministic explanation sentences.

Boundaries that matter
----------------------
The fused score is a **priority-ranking number**, not a probability, not an
accuracy or detection figure, and not an attribution: it asserts nothing about
who owns, controls or is behind anything, and ``explains_attribution()``
returns ``False`` everywhere in this package.  Missing signals are never
replaced by zeros.  Inference levels and edge confidence are carried verbatim
as context and never upgrade or multiply into the score.  Cluster membership
is descriptive geometric context only.

Determinism
-----------
Signal ids and assessment ids are UUIDv5 derivations over content; every
arithmetic step runs under the pinned 34-digit decimal context, so the same
inputs always produce the same assessment, byte for byte.

What this package depends on
----------------------------
The correlation package (inference levels, confidence bases, provenance
citations), the graph package (recorded-evidence edges), the behavior package
(profiles and feature categories), the ml package (anomaly and clustering run
results), and the investigation package (paths and temporal structure).  It
does not import the Step 1-3 substrate.
"""

from __future__ import annotations

from .explain import build_explanation
from .fusion import FusionEngine
from .models import (
    FUSION_DECIMAL_CONTEXT,
    FUSION_NAMESPACE,
    FUSION_SIGNAL_NAMESPACE,
    ConfidenceContext,
    EVIDENCE_COUNT_CAP_DEFAULT,
    ExplainabilityKind,
    FusionAssessment,
    FusionConfig,
    FusionError,
    FusionExplanation,
    FusionSignal,
    FusionStatus,
    MissingSignal,
    PATH_COUNT_CAP_DEFAULT,
    PROVENANCE_FILE_CAP_DEFAULT,
    RiskBand,
    SignalContribution,
    SignalFamily,
    SignalRole,
    SignalStatus,
    band_for_score,
    derive_assessment_id,
    derive_signal_id,
)
from .signals import FusionContext, build_signals

__all__ = [
    "ConfidenceContext",
    "EVIDENCE_COUNT_CAP_DEFAULT",
    "ExplainabilityKind",
    "FUSION_DECIMAL_CONTEXT",
    "FUSION_NAMESPACE",
    "FUSION_SIGNAL_NAMESPACE",
    "FusionAssessment",
    "FusionConfig",
    "FusionContext",
    "FusionEngine",
    "FusionError",
    "FusionExplanation",
    "FusionSignal",
    "FusionStatus",
    "MissingSignal",
    "PATH_COUNT_CAP_DEFAULT",
    "PROVENANCE_FILE_CAP_DEFAULT",
    "RiskBand",
    "SignalContribution",
    "SignalFamily",
    "SignalRole",
    "SignalStatus",
    "band_for_score",
    "build_explanation",
    "build_signals",
    "derive_assessment_id",
    "derive_signal_id",
]