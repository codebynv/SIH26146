"""Step 9A: Evidence Inference Levels Preserved Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data
from backend.app.dashboard.models import InferenceLevelDisplay


def test_inference_levels_preserved():
    """Test that OBSERVED, CORRELATED, INFERRED, REQUIRES_VERIFICATION are preserved."""
    data = _get_demo_data()
    evidence = data.evidence
    
    # Should have evidence with all four inference levels
    inference_levels = {e.inference_level for e in evidence}
    
    assert InferenceLevelDisplay.OBSERVED in inference_levels
    assert InferenceLevelDisplay.CORRELATED in inference_levels
    assert InferenceLevelDisplay.INFERRED in inference_levels
    assert InferenceLevelDisplay.REQUIRES_VERIFICATION in inference_levels
    
    # Count each level
    observed_count = sum(1 for e in evidence if e.inference_level == InferenceLevelDisplay.OBSERVED)
    correlated_count = sum(1 for e in evidence if e.inference_level == InferenceLevelDisplay.CORRELATED)
    inferred_count = sum(1 for e in evidence if e.inference_level == InferenceLevelDisplay.INFERRED)
    requires_count = sum(1 for e in evidence if e.inference_level == InferenceLevelDisplay.REQUIRES_VERIFICATION)
    
    assert observed_count >= 3  # At least 3 OBSERVED
    assert correlated_count >= 2  # At least 2 CORRELATED
    assert inferred_count >= 1  # At least 1 INFERRED
    assert requires_count >= 1  # At least 1 REQUIRES_VERIFICATION


def test_inference_levels_not_upgraded():
    """Test that inference levels are never upgraded (correlation != identity)."""
    data = _get_demo_data()
    evidence = data.evidence
    
    for e in evidence:
        # No inference level should be "IDENTITY" or "OWNERSHIP"
        assert e.inference_level != "IDENTITY"
        assert e.inference_level != "OWNERSHIP"
        assert e.inference_level != "ATTRIBUTION"
        
        # Verify valid levels only
        assert e.inference_level in (
            InferenceLevelDisplay.OBSERVED,
            InferenceLevelDisplay.CORRELATED,
            InferenceLevelDisplay.INFERRED,
            InferenceLevelDisplay.REQUIRES_VERIFICATION,
        )


def test_rationale_preserved():
    """Test that rationale text is preserved for each evidence relationship."""
    data = _get_demo_data()
    evidence = data.evidence
    
    for e in evidence:
        assert e.rationale is not None
        assert len(e.rationale) > 10  # Should be descriptive
        
        # Check specific rationales for our demo data
        if e.inference_level == InferenceLevelDisplay.REQUIRES_VERIFICATION:
            assert "verification" in e.rationale.lower() or "requires" in e.rationale.lower()
        elif e.inference_level == InferenceLevelDisplay.INFERRED:
            assert "inferred" in e.rationale.lower() or "pattern" in e.rationale.lower()


if __name__ == "__main__":
    test_inference_levels_preserved()
    test_inference_levels_not_upgraded()
    test_rationale_preserved()
    print("test_step9a_evidence_inference.py PASSED")
