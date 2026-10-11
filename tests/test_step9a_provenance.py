"""Step 9A: Provenance Preserved Test"""

import pytest
from backend.app.dashboard.api import _get_demo_data


def test_provenance_fields():
    """Test that provenance fields are preserved."""
    data = _get_demo_data()
    prov = data.provenance
    
    assert prov is not None
    assert prov.ingestion_id == "demo-ingestion"
    assert isinstance(prov.source_files, tuple)
    assert "demo.csv" in prov.source_files
    assert isinstance(prov.source_records, tuple)
    
    # Evidence citation ratio
    if prov.evidence_citation_ratio is not None:
        assert 0.0 <= prov.evidence_citation_ratio <= 1.0
    
    if prov.cited_edge_count is not None:
        assert prov.cited_edge_count >= 0
    
    if prov.incident_edge_count is not None:
        assert prov.incident_edge_count >= prov.cited_edge_count
    
    if prov.distinct_source_files is not None:
        assert prov.distinct_source_files >= 1
    
    # Temporal comparability
    if prov.temporal_comparability is not None:
        assert 0.0 <= prov.temporal_comparability <= 1.0
    
    if prov.temporal_forward_flow is not None:
        assert 0.0 <= prov.temporal_forward_flow <= 1.0
    
    # Missing signals
    assert isinstance(prov.missing_signals, tuple)
    for sig in prov.missing_signals:
        assert isinstance(sig, str)
        assert len(sig) > 0


def test_provenance_source_references():
    """Test that evidence relationships carry provenance references."""
    data = _get_demo_data()
    evidence = data.evidence
    
    for e in evidence:
        assert isinstance(e.provenance_refs, tuple)
        assert len(e.provenance_refs) >= 1
        
        for ref in e.provenance_refs:
            assert "ingestion_id" in ref
            assert "source_file" in ref
            assert "source_record" in ref
            assert ref["ingestion_id"] == "demo-ingestion"
            assert ref["source_file"] == "demo.csv"
            assert ref["source_record"].startswith("row-")


if __name__ == "__main__":
    test_provenance_fields()
    test_provenance_source_references()
    print("test_step9a_provenance.py PASSED")
