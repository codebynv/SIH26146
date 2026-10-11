"""Step 9A: CONTEXT_ONLY Signals Preserved Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data
from backend.app.dashboard.models import SignalRoleDisplay, SignalStatusDisplay
from backend.app.fusion.models import SignalRole


def test_context_only_signals_explicit():
    """Test that CONTEXT_ONLY signals are explicitly marked and don't contribute to score."""
    data = _get_demo_data()
    signals = data.signals
    
    context_signals = [s for s in signals if s.role == SignalRoleDisplay.CONTEXT_ONLY]
    
    # Should have at least one CONTEXT_ONLY signal
    assert len(context_signals) >= 1
    
    # TEMPORAL_FORWARD_FLOW should be CONTEXT_ONLY
    forward_flow = next((s for s in signals if s.signal_type == "TEMPORAL_FORWARD_FLOW"), None)
    assert forward_flow is not None
    assert forward_flow.role == SignalRoleDisplay.CONTEXT_ONLY
    
    # CONTEXT_ONLY signals should have normalized_value but not contribute to fused score
    # (This is verified by the fusion engine, but we verify the signal carries the right role)
    assert forward_flow.normalized_value is not None
    assert 0.0 <= forward_flow.normalized_value <= 1.0


def test_context_only_not_in_priority_calculation():
    """Verify CONTEXT_ONLY signals are distinguishable from score-contributing signals."""
    data = _get_demo_data()
    signals = data.signals
    
    for signal in signals:
        if signal.role == SignalRoleDisplay.CONTEXT_ONLY:
            # Should be explicitly marked as context
            assert signal.is_context_only()
            assert not signal.is_priority_signal()
            assert not signal.is_evidence_strength()
        elif signal.role == SignalRoleDisplay.PRIORITY_SIGNAL:
            assert signal.is_priority_signal()
            assert not signal.is_context_only()
        elif signal.role == SignalRoleDisplay.EVIDENCE_STRENGTH:
            assert signal.is_evidence_strength()
            assert not signal.is_context_only()


if __name__ == "__main__":
    test_context_only_signals_explicit()
    test_context_only_not_in_priority_calculation()
    print("test_step9a_context_only.py PASSED")
