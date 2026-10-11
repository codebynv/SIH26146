"""Dashboard data contracts (SIH26146 - Step 9A).

Read-only data models for the investigator dashboard. These models expose the
already-existing Step 1-8 intelligence in a clear, deterministic, evidence-aware
interface.

The dashboard is an INVESTIGATION INTERFACE. It is NOT an automated attribution
system.
"""

from __future__ import annotations

import decimal
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DashboardOverview",
    "DashboardSignal",
    "DashboardEvidence",
    "DashboardPath",
    "DashboardBehaviorContext",
    "DashboardProvenance",
    "DashboardData",
    "SignalRoleDisplay",
    "SignalStatusDisplay",
    "InferenceLevelDisplay",
]


class SignalRoleDisplay(str, Enum):
    """Display role for signals in the dashboard."""
    PRIORITY_SIGNAL = "PRIORITY_SIGNAL"
    EVIDENCE_STRENGTH = "EVIDENCE_STRENGTH"
    CONTEXT_ONLY = "CONTEXT_ONLY"


class SignalStatusDisplay(str, Enum):
    """Display status for signals in the dashboard."""
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class InferenceLevelDisplay(str, Enum):
    """Display inference levels preserved from earlier steps."""
    OBSERVED = "OBSERVED"
    CORRELATED = "CORRELATED"
    INFERRED = "INFERRED"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"


class DashboardSignal(BaseModel):
    """One typed signal exposed to the dashboard.

    Preserves all signal metadata including role, status, and inference levels.
    CONTEXT_ONLY signals are explicitly marked so the UI never implies they
    contribute to the fused priority score.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str
    signal_type: str
    family: str
    role: SignalRoleDisplay
    status: SignalStatusDisplay
    source_step: int
    source_id: str
    subject_id: str
    raw_value: Optional[float] = None
    normalized_value: Optional[float] = None
    explanation: str
    evidence_refs: Tuple[str, ...] = ()
    inference_levels: Tuple[InferenceLevelDisplay, ...] = ()
    provenance_refs: Tuple[dict, ...] = ()
    lineage: Tuple[str, ...] = ()
    context: Tuple[Tuple[str, str], ...] = ()

    def is_priority_signal(self) -> bool:
        return self.role == SignalRoleDisplay.PRIORITY_SIGNAL

    def is_evidence_strength(self) -> bool:
        return self.role == SignalRoleDisplay.EVIDENCE_STRENGTH

    def is_context_only(self) -> bool:
        return self.role == SignalRoleDisplay.CONTEXT_ONLY


class DashboardEvidence(BaseModel):
    """Evidence relationship summary for the dashboard."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    relationship_id: str
    source_observation_ids: Tuple[str, ...]
    target_observation_ids: Tuple[str, ...]
    evidence_type: str
    inference_level: InferenceLevelDisplay
    confidence_value: float
    confidence_basis: str
    rationale: str
    rule_id: str
    rule_version: str
    provenance_refs: Tuple[dict, ...] = ()
    created_at: Optional[datetime] = None


class DashboardPath(BaseModel):
    """Investigation path summary for the dashboard."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    path_id: str
    source_observation_id: str
    target_observation_id: str
    hop_count: int
    node_count: int
    observation_ids: Tuple[str, ...]
    relationship_ids: Tuple[str, ...]
    temporal_status: str
    temporal_flow: Tuple[str, ...] = ()
    duration_seconds: Optional[float] = None
    evidence_summary: dict = {}
    search_status: str


class DashboardBehaviorContext(BaseModel):
    """Behavioral and ML context for the dashboard."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str
    grouping: str
    feature_count: int
    feature_categories: Tuple[str, ...] = ()
    anomaly_band: Optional[str] = None
    anomaly_normalized_score: Optional[float] = None
    anomaly_is_in_sample: Optional[bool] = None
    cluster_id: Optional[str] = None
    cluster_size: Optional[int] = None
    cluster_distance: Optional[float] = None
    cluster_normalized: Optional[float] = None
    explanation_items: Tuple[str, ...] = ()


class DashboardProvenance(BaseModel):
    """Provenance and data quality summary for the dashboard."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ingestion_id: str
    source_files: Tuple[str, ...] = ()
    source_records: Tuple[str, ...] = ()
    evidence_citation_ratio: Optional[float] = None
    cited_edge_count: Optional[int] = None
    incident_edge_count: Optional[int] = None
    distinct_source_files: Optional[int] = None
    temporal_comparability: Optional[float] = None
    temporal_forward_flow: Optional[float] = None
    missing_signals: Tuple[str, ...] = ()


class DashboardOverview(BaseModel):
    """Investigation overview for the dashboard.

    Uses "Investigative Priority" or "Evidence-Supported Investigative Priority"
    terminology. Never presents fused value as probability of crime/fraud/etc.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    subject_profile_id: str
    subject_observation_ids: Tuple[str, ...] = ()
    investigative_priority: Optional[float] = None
    risk_band: Optional[str] = None
    status: str
    evidence_summary: dict = {}
    observation_count: int = 0
    evidence_relationship_count: int = 0
    investigation_path_count: int = 0
    anomaly_context: dict = {}
    cluster_context: dict = {}
    provenance_context: dict = {}
    temporal_context: dict = {}
    assessment_id: Optional[str] = None
    config_version: Optional[str] = None
    interpretation: Optional[str] = None


class DashboardData(BaseModel):
    """Complete dashboard data payload.

    All data is read-only. The same backend state must produce the same
    dashboard data (deterministic ordering enforced).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    overview: DashboardOverview
    signals: Tuple[DashboardSignal, ...] = ()
    evidence: Tuple[DashboardEvidence, ...] = ()
    paths: Tuple[DashboardPath, ...] = ()
    behavior: Optional[DashboardBehaviorContext] = None
    provenance: Optional[DashboardProvenance] = None
    generated_at: datetime
    version: str = "STEP9A/1"


# Helper functions for deterministic ordering

def _sort_by_id(items: Sequence[Any], key_attr: str = "signal_id") -> Tuple[Any, ...]:
    """Sort items by a string ID attribute for deterministic output."""
    return tuple(sorted(items, key=lambda x: getattr(x, key_attr)))


def _sort_strings(items: Sequence[str]) -> Tuple[str, ...]:
    """Sort strings for deterministic output."""
    return tuple(sorted(items))