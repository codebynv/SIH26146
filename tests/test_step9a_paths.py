"""Step 9A: Investigation Paths Preserved Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data


def test_paths_structure():
    """Test that investigation paths have correct structure."""
    data = _get_demo_data()
    paths = data.paths
    
    assert len(paths) >= 1
    
    for path in paths:
        assert path.path_id is not None
        assert path.source_observation_id is not None
        assert path.target_observation_id is not None
        assert path.hop_count >= 1
        assert path.node_count == path.hop_count + 1
        assert isinstance(path.observation_ids, tuple)
        assert isinstance(path.relationship_ids, tuple)
        assert len(path.observation_ids) == path.node_count
        assert len(path.relationship_ids) == path.hop_count
        assert path.temporal_status in ("COMPARABLE", "PARTIALLY_COMPARABLE", "NOT_COMPARABLE", "UNKNOWN")
        assert isinstance(path.temporal_flow, tuple)
        assert isinstance(path.evidence_summary, dict)
        assert path.search_status == "FOUND"
        
        # Evidence summary
        es = path.evidence_summary
        assert "relationship_count" in es
        assert "evidence_types" in es
        assert "confidence_bounds" in es


def test_temporal_flow_preserved():
    """Test that temporal flow steps are preserved as formatted strings."""
    data = _get_demo_data()
    paths = data.paths
    
    for path in paths:
        if path.temporal_flow:
            for step_str in path.temporal_flow:
                # temporal_flow is Tuple[str, ...] with formatted strings
                assert isinstance(step_str, str)
                assert "->" in step_str
                assert "evidence_type" not in step_str  # it'"'"'s formatted, not JSON


def test_path_observation_chain():
    """Test that path forms a valid observation chain."""
    data = _get_demo_data()
    paths = data.paths
    
    for path in paths:
        obs_ids = path.observation_ids
        rel_ids = path.relationship_ids
        
        # Chain should be continuous
        for i, rel_id in enumerate(rel_ids):
            assert obs_ids[i] != obs_ids[i + 1]
        
        # No cycles in simple path
        assert len(set(obs_ids)) == len(obs_ids)
        assert len(set(rel_ids)) == len(rel_ids)


if __name__ == "__main__":
    test_paths_structure()
    test_temporal_flow_preserved()
    test_path_observation_chain()
    print("test_step9a_paths.py PASSED")
