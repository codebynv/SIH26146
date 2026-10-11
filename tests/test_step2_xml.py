"""Step 2 - XML parsing, namespaces, safety and controlled errors."""

from __future__ import annotations

import pytest

from backend.app.domain.enums import SourceFormat
from backend.app.ingestion.parsers import IngestionError, detect_format, parse_xml

from tests.conftest import ADDR_1, ADDR_2, TXID_A

FLAT_RECORD = (
    "<record>"
    "<timestamp>2024-05-01T12:00:00Z</timestamp>"
    "<txid>%s</txid>"
    "<fee>0.00012</fee>"
    "<script_type>P2SH</script_type>"
    "</record>" % TXID_A
)

NESTED_RECORD = (
    "<record>"
    "<timestamp>2024-05-01T12:00:00Z</timestamp>"
    "<input_addresses><address>%s</address><address>%s</address></input_addresses>"
    "<input_amounts><amount>1.5</amount><amount>0.25</amount></input_amounts>"
    "</record>" % (ADDR_1, ADDR_2)
)


def test_detects_xml_from_content():
    assert detect_format(b"<records><record/></records>") is SourceFormat.XML


def test_parses_repeated_flat_elements(service):
    result = service.ingest_bytes(
        ("<records>%s%s</records>" % (FLAT_RECORD, FLAT_RECORD)).encode()
    )
    assert result.raw_record_count == 2


def test_parses_nested_wrapper(service):
    content = ("<dataset><data>%s</data><data>%s</data></dataset>" % (FLAT_RECORD, FLAT_RECORD)).encode()
    assert service.ingest_bytes(content).raw_record_count == 2


def test_single_record_at_document_root(service):
    content = FLAT_RECORD.encode()
    result = service.ingest_bytes(content)
    assert result.raw_record_count == 1
    assert result.records[0].txid == TXID_A


def test_nested_list_element_becomes_a_list(service):
    content = ("<records>%s</records>" % NESTED_RECORD).encode()
    record = service.ingest_bytes(content).records[0]
    assert record.input_addresses == [ADDR_1, ADDR_2]
    assert [float(v) for v in record.input_amounts] == [1.5, 0.25]


def test_single_child_list_element_becomes_a_list(service):
    content = (
        "<records><record><input_addresses><address>%s</address>"
        "</input_addresses></record></records>" % ADDR_1
    ).encode()
    record = service.ingest_bytes(content).records[0]
    assert record.input_addresses == [ADDR_1]


def test_namespaces_are_stripped(service):
    content = (
        '<records xmlns="http://example.com/ns">'
        "<record><timestamp>2024-05-01T00:00:00Z</timestamp>"
        "<txid>%s</txid></record></records>" % TXID_A
    ).encode()
    record = service.ingest_bytes(content).records[0]
    assert record.txid == TXID_A


def test_locator_is_unique_per_record(service):
    content = ("<records>%s%s</records>" % (FLAT_RECORD, FLAT_RECORD)).encode()
    result = service.ingest_bytes(content, deduplicate_records=False)
    locators = [r.provenance.source_record for r in result.records]
    assert len(set(locators)) == 2
    assert locators == ["<record>[0]", "<record>[1]"]


def test_empty_document_is_an_empty_dataset(service):
    assert service.ingest_bytes(b"").raw_record_count == 0


def test_root_with_no_records_is_an_empty_dataset(service):
    assert service.ingest_bytes(b"<records></records>").raw_record_count == 0


def test_malformed_xml_raises_controlled_error(service):
    with pytest.raises(IngestionError) as excinfo:
        service.ingest_bytes(b"<records><record></records>")
    assert excinfo.value.code == "MALFORMED_XML"


def test_unclosed_tag_raises_controlled_error(service):
    with pytest.raises(IngestionError) as excinfo:
        service.ingest_bytes(b"<records><record><txid>abc</record></records>")
    assert excinfo.value.code == "MALFORMED_XML"


def test_doctype_is_refused_to_block_entity_expansion(service):
    payload = (
        b'<?xml version="1.0"?>'
        b"<!DOCTYPE lolz [<!ENTITY lol \"lol\">]>"
        b"<records><record><txid>&lol;</txid></record></records>"
    )
    with pytest.raises(IngestionError) as excinfo:
        service.ingest_bytes(payload)
    assert excinfo.value.code == "UNSAFE_XML"


def test_entity_declaration_is_refused(service):
    payload = b'<!DOCTYPE r [<!ENTITY x "y">]><records><record/></records>'
    with pytest.raises(IngestionError) as excinfo:
        service.ingest_bytes(payload)
    assert excinfo.value.code == "UNSAFE_XML"


def test_parser_yields_raw_records():
    records = list(parse_xml(("<records>%s</records>" % FLAT_RECORD).encode()))
    assert records[0].data["script_type"] == "P2SH"


def test_xml_with_bom_is_parsed(service):
    content = ("﻿<records>%s</records>" % FLAT_RECORD).encode("utf-8")
    assert service.ingest_bytes(content).raw_record_count == 1


# ------------------------------------------- repeated siblings: no data loss
def test_repeated_scalar_siblings_are_both_preserved():
    """Two ``<script_type>`` siblings must not collapse to the last one.

    Overwriting here would silently assert that the record said only ``P2TR``
    when it said both, which is unacceptable in a forensic pipeline.
    """
    payload = (
        "<records><record><txid>%s</txid>"
        "<script_type>P2WPKH</script_type>"
        "<script_type>P2TR</script_type>"
        "</record></records>" % TXID_A
    ).encode()
    data = dict(list(parse_xml(payload))[0].data)
    assert data["script_type"] == ["P2WPKH", "P2TR"]


def test_repeated_structured_siblings_are_both_preserved():
    payload = (
        "<records><record><txid>%s</txid>"
        "<out><a>1</a></out>"
        "<out><a>2</a></out>"
        "</record></records>" % TXID_A
    ).encode()
    data = dict(list(parse_xml(payload))[0].data)
    assert data["out"] == ["1", "2"]


def test_three_repeated_siblings_are_all_preserved():
    payload = (
        "<records><record><txid>%s</txid>"
        "<k>1</k><k>2</k><k>3</k>"
        "</record></records>" % TXID_A
    ).encode()
    data = dict(list(parse_xml(payload))[0].data)
    assert data["k"] == ["1", "2", "3"]


def test_repeated_canonical_list_field_concatenates_rather_than_nests(service):
    """Two ``<input_addresses>`` wrappers yield one flat address list."""
    payload = (
        "<records><record><txid>%s</txid>"
        "<input_addresses><address>%s</address></input_addresses>"
        "<input_addresses><address>%s</address></input_addresses>"
        "</record></records>" % (TXID_A, ADDR_1, ADDR_2)
    ).encode()
    record = service.ingest_bytes(payload).records[0]
    assert record.input_addresses == [ADDR_1, ADDR_2]
    assert not record.errors


def test_mixed_single_and_repeated_list_siblings_flatten(service):
    """A one-element wrapper next to a two-element wrapper stays flat."""
    payload = (
        "<records><record><txid>%s</txid>"
        "<input_amounts><amount>1.5</amount></input_amounts>"
        "<input_amounts><amount>0.25</amount><amount>0.75</amount></input_amounts>"
        "</record></records>" % TXID_A
    ).encode()
    record = service.ingest_bytes(payload).records[0]
    assert [str(a) for a in record.input_amounts] == ["1.5", "0.25", "0.75"]
    assert not record.errors


def test_repeated_scalar_canonical_field_is_reported_not_silently_resolved(service):
    """A repeat that cannot map onto a scalar is reported with both values.

    The information is neither dropped nor arbitrarily resolved: the field is
    rejected and the offending values are carried on the issue.
    """
    payload = (
        "<records><record><txid>%s</txid>"
        "<script_type>P2WPKH</script_type>"
        "<script_type>P2TR</script_type>"
        "</record></records>" % TXID_A
    ).encode()
    result = service.ingest_bytes(payload)
    record = result.records[0]
    assert not record.is_valid
    assert record.script_type is None
    issues = [i for i in record.issues if i.field == "script_type"]
    assert len(issues) == 1
    assert issues[0].raw_value == ["P2WPKH", "P2TR"]


def test_single_sibling_is_still_a_plain_scalar():
    """Merging repeats must not turn a lone value into a one-element list."""
    data = dict(list(parse_xml(("<records>%s</records>" % FLAT_RECORD).encode()))[0].data)
    assert data["script_type"] == "P2SH"
    assert data["txid"] == TXID_A


def test_nested_duplicate_handling_still_works():
    """The pre-existing nested merge behaviour must be unaffected."""
    payload = (
        "<records><record><txid>%s</txid>"
        "<o><a>1</a><a>2</a></o>"
        "</record></records>" % TXID_A
    ).encode()
    data = dict(list(parse_xml(payload))[0].data)
    assert data["o"] == ["1", "2"]


def test_repeated_siblings_do_not_leak_between_records(service):
    """Merging must not leak a sibling value from one record into the next."""
    payload = (
        "<records>"
        "<record><txid>%s</txid><k>1</k><k>2</k></record>"
        "<record><txid>%s</txid><k>3</k></record>"
        "</records>" % (TXID_A, "b" * 64)
    ).encode()
    result = service.ingest_bytes(payload)
    assert result.raw_record_count == 2
    first, second = [dict(r.data) for r in parse_xml(payload)]
    assert first["k"] == ["1", "2"]
    assert second["k"] == "3"
