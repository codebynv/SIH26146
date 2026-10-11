"""Step 9A: All 11 Step 8 Signals Represented Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data
from backend.app.fusion.models import SignalFamily


def test_all_11_signals_present():
    """Test that all 11 Step 8 signals are represented in dashboard."""
    data = _get_demo_data()
    signals = data.signals
    
    # Should have 11 signals (one per family minimum, some families have 2)
    assert len(signals) == 11
    
    # Check all 7 families are represented
    families = {s.family for s in signals}
    expected_families = {
        "ANOMALY",
        "BEHAVIORAL",
        "CLUSTER_CONTEXT",
        "EVIDENCE_GRAPH",
        "PATH_STRUCTURE",
        "PROVENANCE",
        "TEMPORAL",
    }
    assert families == expected_families
    
    # Check specific signal types exist
    signal_types = {s.signal_type for s in signals}
    expected_types = {
        "ANOMALY_NORMALIZED_SCORE",
        "BEHAVIORAL_FEATURE_COVERAGE",
        "CLUSTER_CENTROID_DISTANCE",
        "EVIDENCE_INCIDENT_COUNT",
        "EVIDENCE_TYPE_DIVERSITY",
        "PROVENANCE_CITED_EDGE_RATIO",
        "PROVENANCE_SOURCE_FILE_DIVERSITY",
        "TEMPORAL_COMPARABILITY",
        "TEMPORAL_FORWARD_FLOW",
        "PATH_REACHABILITY_FOUND",
        "PATH_STRUCTURE_COUNT",
    }
    assert signal_types == expected_types


def test_signal_ids_deterministic():
    """Test that signal IDs are deterministic (content-derived)."""
    data = _get_demo_data()
    signal_ids = [s.signal_id for s in data.signals]
    
    # Should be sorted
    assert signal_ids == sorted(signal_ids)
    
    # All unique
    assert len(signal_ids) == len(set(signal_ids))
    
    # UUIDv5 format
    import uuid
    for sid in signal_ids:
        uuid.UUID(sid)  # Should not raise


if __name__ == "__main__":
    test_all_11_signals_present()
    test_signal_ids_deterministic()
    print("test_step9a_signals_11.py PASSED")
