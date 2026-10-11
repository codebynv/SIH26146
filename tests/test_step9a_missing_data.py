"""Step 9A: Missing Data Handling Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data
from backend.app.dashboard.models import SignalStatusDisplay


def test_missing_data_not_zero():
    """Test that missing signals are not silently converted to zero."""
    data = _get_demo_data()
    signals = data.signals
    
    # Find signals with non-PRESENT status
    non_present = [s for s in signals if s.status != SignalStatusDisplay.PRESENT]
    
    for signal in non_present:
        # Non-PRESENT signals should NOT have numeric values
        assert signal.raw_value is None, f"Signal {signal.signal_type} should have raw_value=None"
        assert signal.normalized_value is None, f"Signal {signal.signal_type} should have normalized_value=None"
        
        # Should have explicit status
        assert signal.status in (
            SignalStatusDisplay.ABSENT,
            SignalStatusDisplay.UNAVAILABLE,
            SignalStatusDisplay.NOT_APPLICABLE,
        )
        
        # Should have explanation for why missing
        assert signal.explanation is not None
        assert len(signal.explanation) > 10


def test_status_distinctions():
    """Test that PRESENT/ABSENT/UNAVAILABLE/NOT_APPLICABLE are distinguished."""
    data = _get_demo_data()
    signals = data.signals
    
    statuses = {s.status for s in signals}
    
    # Should have at least PRESENT and some non-PRESENT
    assert SignalStatusDisplay.PRESENT in statuses
    
    # Check we have the distinction
    for signal in signals:
        if signal.status == SignalStatusDisplay.PRESENT:
            # Present signals must have values
            assert signal.normalized_value is not None
        else:
            # Non-present signals must NOT have values
            assert signal.raw_value is None
            assert signal.normalized_value is None


def test_provenance_missing_signals():
    """Test that provenance reports missing signals."""
    data = _get_demo_data()
    prov = data.provenance
    
    if prov and prov.missing_signals:
        for sig in prov.missing_signals:
            assert isinstance(sig, str)
            assert len(sig) > 0
            # Should match actual missing signal types
            assert sig in [
                "TEMPORAL_COMPARABILITY",
                "TEMPORAL_FORWARD_FLOW",
                "ANOMALY_NORMALIZED_SCORE",
                "CLUSTER_CENTROID_DISTANCE",
            ]


if __name__ == "__main__":
    test_missing_data_not_zero()
    test_status_distinctions()
    test_provenance_missing_signals()
    print("test_step9a_missing_data.py PASSED")
