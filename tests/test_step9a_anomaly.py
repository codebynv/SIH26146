"""Step 9A: Anomaly Context Preserved Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data


def test_anomaly_context():
    """Test that anomaly context is preserved."""
    data = _get_demo_data()
    overview = data.overview
    
    anomaly_ctx = overview.anomaly_context
    assert isinstance(anomaly_ctx, dict)
    
    if anomaly_ctx:
        # Should have band, normalized_score, is_in_sample
        if "band" in anomaly_ctx:
            assert anomaly_ctx["band"] in ("LOW", "MODERATE", "HIGH", "EXTREME", None)
        
        if "normalized_score" in anomaly_ctx:
            score = anomaly_ctx["normalized_score"]
            assert score is None or (0.0 < score <= 1.0)
        
        if "is_in_sample" in anomaly_ctx:
            assert isinstance(anomaly_ctx["is_in_sample"], bool)


def test_anomaly_terminology():
    """Test that anomaly uses 'unusualness' not 'fraud'."""
    data = _get_demo_data()
    overview = data.overview
    
    anomaly_ctx = overview.anomaly_context
    
    # No fraud terminology in keys or values
    for key, value in anomaly_ctx.items():
        assert "fraud" not in key.lower()
        assert "fraud" not in str(value).lower()
        assert "malicious" not in key.lower()
        assert "malicious" not in str(value).lower()


if __name__ == "__main__":
    test_anomaly_context()
    test_anomaly_terminology()
    print("test_step9a_anomaly.py PASSED")

