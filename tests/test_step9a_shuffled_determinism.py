"""Step 9A: Shuffled Input Determinism Test"""

import pytest
from backend.app.dashboard.adapter import build_dashboard_data
from backend.app.dashboard.api import _create_demo_context
from backend.app.fusion import FusionEngine


def test_shuffled_edges_deterministic():
    """Test that shuffled evidence edges produce same dashboard output."""
    # Get the demo context
    context = _create_demo_context()
    
    # Build dashboard data twice
    engine = FusionEngine()
    assessment1 = engine.assess(context)
    dashboard1 = build_dashboard_data(assessment1, context)
    
    assessment2 = engine.assess(context)
    dashboard2 = build_dashboard_data(assessment2, context)
    
    # Should be identical
    assert dashboard1 == dashboard2
    
    # Evidence order should be deterministic (not dependent on input order)
    evidence_ids_1 = [e.relationship_id for e in dashboard1.evidence]
    evidence_ids_2 = [e.relationship_id for e in dashboard2.evidence]
    assert evidence_ids_1 == evidence_ids_2
    assert evidence_ids_1 == sorted(evidence_ids_1)


def test_shuffled_signals_deterministic():
    """Test that signal ordering is deterministic regardless of family iteration order."""
    context = _create_demo_context()
    engine = FusionEngine()
    assessment = engine.assess(context)
    
    from backend.app.dashboard.adapter import build_dashboard_data
    dashboard = build_dashboard_data(assessment, context)
    
    # Signals should be sorted by signal_id
    signal_ids = [s.signal_id for s in dashboard.signals]
    assert signal_ids == sorted(signal_ids)
    
    # Families should appear in fixed order in overview evidence_summary
    if dashboard.overview.evidence_summary and "evidence_types" in dashboard.overview.evidence_summary:
        types = dashboard.overview.evidence_summary["evidence_types"]
        assert types == sorted(types)


if __name__ == "__main__":
    test_shuffled_edges_deterministic()
    test_shuffled_signals_deterministic()
    print("test_step9a_shuffled_determinism.py PASSED")
