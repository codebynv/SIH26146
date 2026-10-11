"""Dashboard package (SIH26146 - Step 9A).

Investigator-facing dashboard foundation exposing Step 1-8 intelligence.
"""

from .adapter import build_dashboard_data
from .api import router
from .models import (
    DashboardBehaviorContext,
    DashboardData,
    DashboardEvidence,
    DashboardOverview,
    DashboardPath,
    DashboardProvenance,
    DashboardSignal,
    InferenceLevelDisplay,
    SignalRoleDisplay,
    SignalStatusDisplay,
)

__all__ = [
    "DashboardBehaviorContext",
    "DashboardData",
    "DashboardEvidence",
    "DashboardOverview",
    "DashboardPath",
    "DashboardProvenance",
    "DashboardSignal",
    "InferenceLevelDisplay",
    "SignalRoleDisplay",
    "SignalStatusDisplay",
    "build_dashboard_data",
    "router",
]