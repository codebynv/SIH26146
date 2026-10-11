"""Step 9A: Cluster Context Preserved Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data


def test_cluster_context():
    """Test that cluster context is preserved as descriptive only."""
    data = _get_demo_data()
    overview = data.overview
    
    cluster_ctx = overview.cluster_context
    assert isinstance(cluster_ctx, dict)
    
    if cluster_ctx:
        # Should have cluster_id, cluster_size, distance_to_centroid
        if "cluster_id" in cluster_ctx:
            assert cluster_ctx["cluster_id"] is not None
        
        if "cluster_size" in cluster_ctx:
            assert isinstance(cluster_ctx["cluster_size"], int)
            assert cluster_ctx["cluster_size"] > 0
        
        if "distance_to_centroid" in cluster_ctx:
            dist = cluster_ctx["distance_to_centroid"]
            assert dist is None or (isinstance(dist, (int, float)) and dist >= 0)


def test_cluster_context_only():
    """Test that cluster membership is context only, not identity."""
    data = _get_demo_data()
    behavior = data.behavior
    
    if behavior.cluster_id is not None:
        # Cluster context fields exist
        assert behavior.cluster_size is not None
        assert behavior.cluster_distance is not None
        
        # But normalized is explicitly labeled as context
        if behavior.cluster_normalized is not None:
            assert 0.0 <= behavior.cluster_normalized <= 1.0
    
    # No identity attribution fields
    assert not hasattr(behavior, "identity_group")
    assert not hasattr(behavior, "owner")
    assert not hasattr(behavior, "person_id")


if __name__ == "__main__":
    test_cluster_context()
    test_cluster_context_only()
    print("test_step9a_cluster.py PASSED")
