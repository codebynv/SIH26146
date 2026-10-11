"""Step 9A: Signal Roles Preserved Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data
from backend.app.dashboard.models import SignalRoleDisplay


def test_signal_roles_preserved():
    """Test that PRIORITY_SIGNAL, EVIDENCE_STRENGTH, CONTEXT_ONLY roles are preserved."""
    data = _get_demo_data()
    signals = data.signals
    
    # Count roles
    priority_count = sum(1 for s in signals if s.role == SignalRoleDisplay.PRIORITY_SIGNAL)
    evidence_count = sum(1 for s in signals if s.role == SignalRoleDisplay.EVIDENCE_STRENGTH)
    context_count = sum(1 for s in signals if s.role == SignalRoleDisplay.CONTEXT_ONLY)
    
    # Should have all three role types
    assert priority_count > 0, "Should have PRIORITY_SIGNAL signals"
    assert evidence_count > 0, "Should have EVIDENCE_STRENGTH signals"
    assert context_count > 0, "Should have CONTEXT_ONLY signals"
    
    # Total should be 11
    assert priority_count + evidence_count + context_count == 11
    
    # Verify specific signals have correct roles
    signal_roles = {s.signal_type: s.role for s in signals}
    
    # PRIORITY_SIGNAL signals
    assert signal_roles["ANOMALY_NORMALIZED_SCORE"] == SignalRoleDisplay.PRIORITY_SIGNAL
    assert signal_roles["EVIDENCE_INCIDENT_COUNT"] == SignalRoleDisplay.PRIORITY_SIGNAL
    assert signal_roles["CLUSTER_CENTROID_DISTANCE"] == SignalRoleDisplay.PRIORITY_SIGNAL
    assert signal_roles["PATH_REACHABILITY_FOUND"] == SignalRoleDisplay.PRIORITY_SIGNAL
    assert signal_roles["PATH_STRUCTURE_COUNT"] == SignalRoleDisplay.PRIORITY_SIGNAL
    
    # EVIDENCE_STRENGTH signals
    assert signal_roles["BEHAVIORAL_FEATURE_COVERAGE"] == SignalRoleDisplay.EVIDENCE_STRENGTH
    assert signal_roles["EVIDENCE_TYPE_DIVERSITY"] == SignalRoleDisplay.EVIDENCE_STRENGTH
    assert signal_roles["PROVENANCE_CITED_EDGE_RATIO"] == SignalRoleDisplay.EVIDENCE_STRENGTH
    assert signal_roles["PROVENANCE_SOURCE_FILE_DIVERSITY"] == SignalRoleDisplay.EVIDENCE_STRENGTH
    assert signal_roles["TEMPORAL_COMPARABILITY"] == SignalRoleDisplay.EVIDENCE_STRENGTH
    
    # CONTEXT_ONLY signals
    assert signal_roles["TEMPORAL_FORWARD_FLOW"] == SignalRoleDisplay.CONTEXT_ONLY


def test_role_display_values():
    """Test that role display values match expected enum."""
    data = _get_demo_data()
    for signal in data.signals:
        assert signal.role in (
            SignalRoleDisplay.PRIORITY_SIGNAL,
            SignalRoleDisplay.EVIDENCE_STRENGTH,
            SignalRoleDisplay.CONTEXT_ONLY,
        )


if __name__ == "__main__":
    test_signal_roles_preserved()
    test_role_display_values()
    print("test_step9a_signal_roles.py PASSED")
