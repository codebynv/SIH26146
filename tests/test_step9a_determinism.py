"""Step 9A: Deterministic Dashboard Output Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data


def test_deterministic_output():
    """Test that same input produces identical dashboard data."""
    data1 = _get_demo_data()
    data2 = _get_demo_data()
    
    # Should be exactly equal (including generated_at via deterministic timestamp)
    assert data1 == data2
    
    # Check specific fields
    assert data1.overview == data2.overview
    assert data1.signals == data2.signals
    assert data1.evidence == data2.evidence
    assert data1.paths == data2.paths
    assert data1.behavior == data2.behavior
    assert data1.provenance == data2.provenance
    assert data1.generated_at == data2.generated_at
    assert data1.version == data2.version


def test_deterministic_ordering():
    """Test that collections are deterministically ordered."""
    data = _get_demo_data()
    
    # Signals ordered by signal_id
    signal_ids = [s.signal_id for s in data.signals]
    assert signal_ids == sorted(signal_ids)
    
    # Evidence ordered by relationship_id
    evidence_ids = [e.relationship_id for e in data.evidence]
    assert evidence_ids == sorted(evidence_ids)
    
    # Paths ordered by path_id
    path_ids = [p.path_id for p in data.paths]
    assert path_ids == sorted(path_ids)
    
    # Provenance source_files sorted
    if data.provenance and data.provenance.source_files:
        assert list(data.provenance.source_files) == sorted(data.provenance.source_files)
    
    # Missing signals sorted
    if data.provenance and data.provenance.missing_signals:
        assert list(data.provenance.missing_signals) == sorted(data.provenance.missing_signals)


if __name__ == "__main__":
    test_deterministic_output()
    test_deterministic_ordering()
    print("test_step9a_determinism.py PASSED")
