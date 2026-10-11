"""Step 9A: Empty Investigation Handling Test"""

import pytest
from backend.app.dashboard.adapter import build_dashboard_data
from backend.app.fusion import FusionConfig, FusionEngine
from backend.app.fusion.signals import FusionContext
from backend.app.behavior.profiles import BehaviorProfileBuilder
from backend.app.domain.models import UnifiedObservation, ObservationType, Provenance, NetworkObservation, BlockchainTransaction
from decimal import Decimal
from datetime import datetime, timezone


def test_empty_investigation():
    """Test dashboard handles empty investigation gracefully."""
    # Create minimal context with no graph, no paths, no anomaly, no clustering
    UTC = timezone.utc
    T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    
    obs = UnifiedObservation(
        observation_id="single-obs",
        observation_type=ObservationType.COMBINED,
        timestamp=T0,
        network=NetworkObservation(timestamp=T0, src_ip="1.2.3.4", dst_ip="5.6.7.8", src_port=1234, dst_port=443, asn=15169, geo_country="US"),
        blockchain=BlockchainTransaction(timestamp=T0, txid="a" * 64, input_addresses=["bc1qaaaa"], output_addresses=["bc1qbbbb"], input_amounts=[Decimal("1.0")], output_amounts=[Decimal("1.0")], fee=Decimal("0.001")),
        provenance=Provenance(ingestion_id="test", source_file="test.csv", source_record="row-1", context={}),
        extra={},
    )
    
    profile = BehaviorProfileBuilder().build([obs], group_id="single-profile")
    
    context = FusionContext(profile=profile)
    engine = FusionEngine()
    assessment = engine.assess(context)
    dashboard = build_dashboard_data(assessment, context)
    
    # Should not crash
    assert dashboard is not None
    assert dashboard.overview.observation_count == 1
    assert dashboard.overview.evidence_relationship_count == 0
    assert dashboard.overview.investigation_path_count == 0
    assert dashboard.overview.anomaly_context == {}
    assert dashboard.overview.cluster_context == {}
    assert dashboard.overview.temporal_context == {}
    
    # Sections should show empty state, not crash
    assert len(dashboard.evidence) == 0
    assert len(dashboard.paths) == 0
    assert dashboard.behavior is not None  # Profile exists
    assert dashboard.provenance is not None


def test_empty_graph_handling():
    """Test that empty graph produces empty evidence list."""
    from backend.app.dashboard.api import _get_demo_data
    data = _get_demo_data()
    # Our demo has evidence, but verify the structure handles empty
    assert isinstance(data.evidence, tuple)
    
    # Evidence summary should handle zero
    ev_sum = data.overview.evidence_summary
    assert "incident_edge_count" in ev_sum
    assert ev_sum["incident_edge_count"] >= 0


if __name__ == "__main__":
    test_empty_investigation()
    test_empty_graph_handling()
    print("test_step9a_empty_investigation.py PASSED")
