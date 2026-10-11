"""Step 9A: Behavioral Features Preserved Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data


def test_behavioral_context():
    """Test that behavioral context is preserved."""
    data = _get_demo_data()
    behavior = data.behavior
    
    assert behavior is not None
    import uuid; assert uuid.UUID(behavior.profile_id); assert behavior.grouping in ("ADDRESS", "ENTITY", "NETWORK", "SUPPLIED_COLLECTION")
    assert behavior.grouping in ("ADDRESS", "ENTITY", "NETWORK", "SUPPLIED_COLLECTION")
    assert behavior.feature_count > 0
    assert isinstance(behavior.feature_categories, tuple)
    assert len(behavior.feature_categories) > 0
    
    # Expected categories
    expected_cats = {"ACTIVITY_VOLUME", "TEMPORAL_ACTIVITY", "TRANSACTION_AMOUNTS", "NETWORK_BEHAVIOR", "ADDRESS_PARTICIPATION", "EVIDENCE_GRAPH", "PROVENANCE_DIVERSITY"}
    actual_cats = set(behavior.feature_categories)
    assert actual_cats.issubset(expected_cats) or expected_cats.issubset(actual_cats)


def test_behavioral_no_fraud_terminology():
    """Test that behavioral context uses 'unusualness' not 'fraud'."""
    data = _get_demo_data()
    behavior = data.behavior
    
    # Check field names don't contain fraud terminology
    assert not hasattr(behavior, "fraud_probability")
    assert not hasattr(behavior, "fraud_score")
    assert not hasattr(behavior, "maliciousness")
    
    # Uses anomaly_band, anomaly_normalized_score
    if behavior.anomaly_band is not None:
        assert behavior.anomaly_band in ("LOW", "MODERATE", "HIGH", "EXTREME")
    
    if behavior.anomaly_normalized_score is not None:
        assert 0.0 < behavior.anomaly_normalized_score <= 1.0


if __name__ == "__main__":
    test_behavioral_context()
    test_behavioral_no_fraud_terminology()
    print("test_step9a_behavioral.py PASSED")



