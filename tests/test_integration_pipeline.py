"""Integration - ingestion (Step 2) into the unified model (Step 3).

These tests exercise the full path: raw bytes -> parser -> normalised record
-> unified observation, with no mocking in between.
"""

from __future__ import annotations

from decimal import Decimal



from backend.app.domain.adapters import normalized_to_unified
from backend.app.domain.enums import ObservationType, SourceFormat
from backend.app.ingestion.uploads import store_upload
from tests.conftest import ADDR_1, ADDR_3, TXID_A, TXID_B, csv_bytes, json_bytes

COMBINED_ROW = ",".join(
    [
        "2024-05-01T12:00:00Z",
        "203.0.113.5",
        "198.51.100.9",
        "51000",
        "443",
        TXID_A,
        ADDR_1,
        ADDR_3,
        "1.5",
        "1.49",
        "0.00005",
        "P2WPKH",
        "IN",
        "AS15169",
    ]
)
NETWORK_ROW = (
    "2024-05-01T12:00:05Z,203.0.113.5,198.51.100.9,51001,443,,,,,,,0.00001,P2PKH,US,AS64500"
)
BLOCKCHAIN_ROW = (
    "2024-05-01T12:00:10,,,,,%s,%s,%s,1.5,1.49,0.00001,P2SH,," % (TXID_B, ADDR_1, ADDR_3)
)


def test_csv_end_to_end_produces_three_observation_types(service):
    result = service.ingest_bytes(
        csv_bytes(COMBINED_ROW, NETWORK_ROW, BLOCKCHAIN_ROW), source_file="mixed.csv"
    )
    observations = [normalized_to_unified(record) for record in result.records]
    assert [o.observation_type for o in observations] == [
        ObservationType.COMBINED,
        ObservationType.NETWORK,
        ObservationType.BLOCKCHAIN,
    ]


def test_end_to_end_preserves_the_full_sih_field_set(service):
    result = service.ingest_bytes(csv_bytes(COMBINED_ROW), source_file="mixed.csv")
    observation = normalized_to_unified(result.records[0])
    assert str(observation.network.src_ip) == "203.0.113.5"
    assert str(observation.network.dst_ip) == "198.51.100.9"
    assert observation.network.src_port == 51000
    assert observation.network.dst_port == 443
    assert observation.network.geo_country == "IN"
    assert observation.network.asn == 15169
    assert observation.blockchain.txid == TXID_A
    assert observation.blockchain.input_addresses == [ADDR_1]
    assert observation.blockchain.output_addresses == [ADDR_3]
    assert observation.blockchain.fee == Decimal("0.00005")
    assert observation.blockchain.script_type == "P2WPKH"
    assert observation.timestamp == result.records[0].timestamp


def test_end_to_end_preserves_provenance(service):
    result = service.ingest_bytes(csv_bytes(COMBINED_ROW), source_file="mixed.csv")
    observation = normalized_to_unified(result.records[0])
    assert observation.provenance.source_file == "mixed.csv"
    assert observation.provenance.source_format is SourceFormat.CSV
    assert observation.provenance.source_record == "row 2"
    assert observation.provenance.ingestion_id == result.ingestion_id


def test_json_end_to_end(service):
    payload = [
        {
            "timestamp": "2024-05-01T12:00:00Z",
            "txid": TXID_A,
            "input_addresses": [ADDR_1],
            "output_addresses": [ADDR_3],
            "input_amounts": [1.5],
            "output_amounts": [1.49],
            "fee": 0.00005,
            "script_type": "P2WPKH",
        }
    ]
    result = service.ingest_bytes(json_bytes(payload), source_file="tx.json")
    observation = normalized_to_unified(result.records[0])
    assert observation.observation_type is ObservationType.BLOCKCHAIN
    assert observation.blockchain.fee == Decimal("0.00005")
    assert observation.provenance.source_format is SourceFormat.JSON


def test_xml_end_to_end(service):
    content = (
        "<records><record>"
        "<timestamp>2024-05-01T12:00:00Z</timestamp>"
        "<src_ip>203.0.113.5</src_ip>"
        "<txid>%s</txid>"
        "<input_addresses><address>%s</address></input_addresses>"
        "<fee>0.0001</fee><script_type>P2TR</script_type>"
        "</record></records>" % (TXID_A, ADDR_1)
    ).encode()
    result = service.ingest_bytes(content, source_file="flows.xml")
    observation = normalized_to_unified(result.records[0])
    assert observation.observation_type is ObservationType.COMBINED
    assert str(observation.network.src_ip) == "203.0.113.5"
    assert observation.blockchain.input_addresses == [ADDR_1]
    assert observation.blockchain.script_type == "P2TR"
    assert observation.provenance.source_format is SourceFormat.XML


def test_aliases_flow_through_to_the_observation(service):
    content = b"time,client_ip,server_port,tx_hash,from_addresses,txn_fee,country_code\n2024-05-01T12:00:00Z,203.0.113.5,443,%s,%s,0.0002,IN\n" % (
        TXID_A.encode(),
        ADDR_1.encode(),
    )
    observation = normalized_to_unified(service.ingest_bytes(content).records[0])
    assert str(observation.network.src_ip) == "203.0.113.5"
    assert observation.network.dst_port == 443
    assert observation.blockchain.txid == TXID_A
    assert observation.blockchain.fee == Decimal("0.0002")
    assert observation.network.geo_country == "IN"


def test_unmapped_geo_column_stays_in_extra(service):
    """'code' is not a recognised country alias, so it must not be guessed."""
    content = b"time,client_ip,code\n2024-05-01T12:00:00Z,203.0.113.5,IN\n"
    observation = normalized_to_unified(service.ingest_bytes(content).records[0])
    assert observation.network.geo_country is None
    assert observation.extra == {"code": "IN"}


def test_unmapped_fields_survive_ingestion_to_observation(service):
    content = b"timestamp,txid,input_addresses,block_height,confirmations\n2024-05-01T12:00:00Z,%s,%s,840000,6\n" % (
        TXID_A.encode(),
        ADDR_1.encode(),
    )
    observation = normalized_to_unified(service.ingest_bytes(content).records[0])
    assert observation.extra == {"block_height": "840000", "confirmations": "6"}


def test_validation_error_surfaces_on_the_observation(service):
    content = csv_bytes("2024-05-01T12:00:00Z,bad-ip,1.2.3.5,,,,,,,,,,,")
    observation = normalized_to_unified(service.ingest_bytes(content).records[0])
    assert not observation.is_valid
    assert observation.errors[0].field == "src_ip"
    assert observation.errors[0].code == "INVALID_FIELD"


def test_naive_timestamp_warning_surfaces_on_the_observation(service):
    content = csv_bytes("2024-05-01T12:00:00,1.2.3.4,1.2.3.5,,,,,,,,,,,")
    observation = normalized_to_unified(service.ingest_bytes(content).records[0])
    assert observation.warnings[0].code == "NAIVE_TIMESTAMP"
    assert observation.timestamp.tzinfo is None


def test_deduplication_happens_before_adaptation(service):
    result = service.ingest_bytes(csv_bytes(COMBINED_ROW, COMBINED_ROW, COMBINED_ROW))
    observations = [normalized_to_unified(record) for record in result.records]
    assert len(observations) == 1
    assert result.duplicate_count == 2
    # The same observation yields the same id.
    assert len({o.observation_id for o in observations}) == 1


def test_distinct_records_yield_distinct_observation_ids(service):
    result = service.ingest_bytes(csv_bytes(COMBINED_ROW, NETWORK_ROW, BLOCKCHAIN_ROW))
    ids = {normalized_to_unified(record).observation_id for record in result.records}
    assert len(ids) == len(result.records)


def test_ingest_endpoint_to_observation_round_trip(client, service):
    """The API response and the domain model agree on the same data."""
    response = client.post(
        "/ingest",
        files={"file": ("mixed.csv", csv_bytes(COMBINED_ROW), "text/csv")},
    )
    body = response.json()
    result = service.ingest_bytes(csv_bytes(COMBINED_ROW), source_file="mixed.csv")
    observation = normalized_to_unified(result.records[0])

    api_record = body["records"][0]["record"]
    assert api_record["txid"] == observation.blockchain.txid
    assert api_record["script_type"] == observation.blockchain.script_type
    assert api_record["src_ip"] == str(observation.network.src_ip)
    assert api_record["provenance"]["source_file"] == observation.provenance.source_file
    assert api_record["provenance"]["source_record"] == observation.provenance.source_record
    # Each run has its own id, but both must be populated.
    assert api_record["provenance"]["ingestion_id"]
    assert observation.provenance.ingestion_id


def test_upload_to_observation_via_staging_file(service, tmp_path):
    stored = store_upload(
        csv_bytes(COMBINED_ROW), client_filename="../../evil.csv", destination=str(tmp_path)
    )
    result = service.ingest_path(
        stored.path, context={"client_filename": stored.display_filename}
    )
    observation = normalized_to_unified(result.records[0])
    assert observation.observation_type is ObservationType.COMBINED
    assert observation.provenance.source_file == "evil.csv"


def test_empty_dataset_produces_no_observations(service):
    result = service.ingest_bytes(b"")
    assert result.records == []
    assert [normalized_to_unified(r) for r in result.records] == []


def _imported_modules(module_name: str) -> set[str]:
    """Return every module name actually imported by *module_name*.

    Parsed from the AST rather than grepped from the source, so a passing
    reference in a docstring is not mistaken for a dependency.
    """
    import importlib
    import ast

    module = importlib.import_module(module_name)
    source = open(module.__file__, encoding="utf-8").read()
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(
                "%s.%s" % (node.module, alias.name)
                for alias in node.names
            )
    return names


def test_step3_does_not_depend_on_the_ingestion_package():
    """Layering: the domain package must not import the ingestion package."""
    for module_name in (
        "backend.app.domain",
        "backend.app.domain.adapters",
        "backend.app.domain.models",
        "backend.app.domain.validation",
        "backend.app.domain.enums",
    ):
        imported = _imported_modules(module_name)
        offenders = [n for n in imported if "backend.app.ingestion" in n]
        assert offenders == [], "%s imports %s" % (module_name, offenders)


def test_ingestion_may_depend_on_the_domain_package():
    """Layering: the dependency runs one way only."""
    imported = _imported_modules("backend.app.ingestion.models")
    assert any("backend.app.domain" in name for name in imported)


def test_adapter_works_with_a_non_ingestion_record_object():
    """The adapter depends on structure, not on the ingestion class."""

    class DuckRecord:
        timestamp = "2024-05-01T12:00:00Z"
        src_ip = "203.0.113.5"
        dst_ip = None
        src_port = None
        dst_port = 443
        txid = None
        input_addresses: list = []
        output_addresses: list = []
        input_amounts: list = []
        output_amounts: list = []
        fee = None
        script_type = None
        geo_country = "IN"
        asn = 15169
        extra: dict = {}
        issues: tuple = ()

        def __init__(self, provenance):
            self.provenance = provenance

    from backend.app.domain.adapters import normalized_to_unified

    observation = normalized_to_unified(DuckRecord(_provenance()))
    assert observation.observation_type is ObservationType.NETWORK
    assert str(observation.network.src_ip) == "203.0.113.5"


# ------------------------------------------------- evidence-loss regressions
def test_repeated_xml_siblings_survive_into_the_unified_model():
    """A repeated unmapped XML tag must reach the observation intact.

    The full chain -- bytes, parser, normalisation, adapter -- must not lose a
    sibling value on the way, otherwise the evidence substrate Step 4 will
    correlate over is already incomplete.
    """
    from backend.app.ingestion.service import IngestionService

    payload = (
        "<records><record><txid>%s</txid>"
        "<script_type>P2WPKH</script_type>"
        "<script_type>P2TR</script_type>"
        "<out><a>1</a></out><out><a>2</a></out>"
        "</record></records>" % TXID_A
    ).encode()
    record = IngestionService().ingest_bytes(payload).records[0]
    observation = normalized_to_unified(record)

    # Both script types were seen, so the field is rejected and reported.
    assert observation.blockchain is not None
    assert observation.blockchain.script_type is None
    assert any(issue.field == "script_type" for issue in observation.errors)
    # Both unmapped values survive untouched.
    assert observation.extra["out"] == ["1", "2"]


def test_differing_unmapped_evidence_survives_the_whole_pipeline():
    """Two rows differing only in an unmapped column become two observations."""
    from backend.app.ingestion.service import IngestionService

    result = IngestionService().ingest_bytes(
        b"timestamp,txid,block_height\n"
        b"2024-05-01T00:00:00Z," + TXID_A.encode() + b",840000\n"
        b"2024-05-01T00:00:00Z," + TXID_A.encode() + b",840001\n"
    )
    observations = [normalized_to_unified(r) for r in result.records]
    assert len(observations) == 2
    assert {o.extra["block_height"] for o in observations} == {"840000", "840001"}
    # Distinct evidence must not collapse into one observation identity.
    assert len({o.observation_id for o in observations}) == 2


def test_a_caller_owned_evidence_file_survives_ingestion(tmp_path):
    """End-to-end proof that reading evidence is non-destructive."""
    import os

    from backend.app.ingestion.service import IngestionService

    source = tmp_path / "case_42_blockchain.csv"
    content = csv_bytes(
        ",".join(["2024-05-01T12:00:00Z", TXID_A, ADDR_1, "", "1.5", "", "0.0001",
                  "P2WPKH", "IN", "AS1", "", "", "", ""])
    )
    source.write_bytes(content)

    result = IngestionService().ingest_path(str(source))

    assert result.record_count == 1
    assert os.path.exists(source)
    assert source.read_bytes() == content


def _provenance():
    from datetime import datetime, timezone

    from backend.app.domain.models import Provenance

    return Provenance(
        ingestion_id="duck",
        source_file="duck.csv",
        source_format="csv",
        source_record="row 1",
        ingested_at=datetime(2024, 5, 1, tzinfo=timezone.utc),
        context={},
    )
