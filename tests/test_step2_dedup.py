"""Step 2 - conservative deduplication.

The central guarantee: deduplication must never merge two observations that
are merely *similar*.  Many of these tests assert that records SURVIVE.
"""

from __future__ import annotations



from backend.app.domain.models import Provenance
from backend.app.ingestion.dedup import deduplicate, record_fingerprint
from backend.app.ingestion.models import NormalizedRecord
from tests.conftest import (
    ADDR_1,
    ADDR_3,
    TXID_A,
    TXID_B,
    csv_bytes,
    csv_row,
    make_record,
)


def prov(record: str = "row 2", source: str = "a.csv", run: str = "run-1") -> Provenance:
    return Provenance(
        ingestion_id=run,
        source_file=source,
        source_format="csv",
        source_record=record,
        context={},
    )


# ------------------------------------------------------------------ merging
def test_byte_identical_records_are_deduplicated(provenance):
    records = [
        make_record(provenance, src_ip="1.2.3.4", dst_port=443),
        make_record(provenance, src_ip="1.2.3.4", dst_port=443),
    ]
    result = deduplicate(records)
    assert result.unique_count == 1
    assert result.duplicate_count == 1


def test_same_observation_from_two_files_collapses(provenance):
    records = [
        make_record(prov("row 2", "a.csv"), src_ip="1.2.3.4", dst_port=443),
        make_record(prov("row 2", "b.csv", "run-2"), src_ip="1.2.3.4", dst_port=443),
    ]
    result = deduplicate(records)
    assert result.unique_count == 1
    assert result.duplicate_count == 1


def test_duplicate_group_reports_the_merged_locator(provenance):
    records = [
        make_record(prov("row 2", "a.csv"), src_ip="1.2.3.4"),
        make_record(prov("row 9", "b.csv", "run-2"), src_ip="1.2.3.4"),
    ]
    result = deduplicate(records)
    assert result.groups[0].retained_locator == "row 2"
    assert result.groups[0].merged == ("row 9",)
    assert result.groups[0].sources == ("b.csv",)


def test_fingerprint_ignores_provenance(provenance):
    a = make_record(prov("row 2", "a.csv"), src_ip="1.2.3.4")
    b = make_record(prov("row 77", "z.csv", "other-run"), src_ip="1.2.3.4")
    assert record_fingerprint(a) == record_fingerprint(b)


def test_first_seen_order_is_preserved(provenance):
    records = [
        make_record(prov("row 2"), src_ip="1.1.1.1"),
        make_record(prov("row 3"), src_ip="2.2.2.2"),
        make_record(prov("row 4"), src_ip="1.1.1.1"),
    ]
    result = deduplicate(records)
    assert [str(r.src_ip) for r in result.records] == ["1.1.1.1", "2.2.2.2"]


def test_retained_record_keeps_its_own_provenance(provenance):
    records = [
        make_record(prov("row 2", "a.csv"), src_ip="1.1.1.1"),
        make_record(prov("row 9", "b.csv", "run-2"), src_ip="1.1.1.1"),
    ]
    result = deduplicate(records)
    assert result.records[0].provenance.source_file == "a.csv"


# ------------------------------------------- records that must NOT be merged
def test_same_txid_different_amounts_are_both_kept(provenance):
    records = [
        make_record(provenance, txid=TXID_A, fee="0.001"),
        make_record(provenance, txid=TXID_A, fee="0.002"),
    ]
    assert deduplicate(records).unique_count == 2


def test_same_txid_different_timestamps_are_both_kept(provenance):
    records = [
        make_record(provenance, timestamp="2024-05-01T00:00:00Z", txid=TXID_A),
        make_record(provenance, timestamp="2024-05-01T00:00:01Z", txid=TXID_A),
    ]
    assert deduplicate(records).unique_count == 2


def test_same_txid_different_script_types_are_both_kept(provenance):
    records = [
        make_record(provenance, txid=TXID_A, script_type="P2PKH"),
        make_record(provenance, txid=TXID_A, script_type="P2SH"),
    ]
    assert deduplicate(records).unique_count == 2


def test_same_ip_different_ports_are_both_kept(provenance):
    records = [
        make_record(provenance, src_ip="1.2.3.4", src_port=1000),
        make_record(provenance, src_ip="1.2.3.4", src_port=2000),
    ]
    assert deduplicate(records).unique_count == 2


def test_same_ip_different_timestamps_are_both_kept(provenance):
    records = [
        make_record(provenance, timestamp="2024-05-01T00:00:00Z", src_ip="1.2.3.4"),
        make_record(provenance, timestamp="2024-05-01T00:00:05Z", src_ip="1.2.3.4"),
    ]
    assert deduplicate(records).unique_count == 2


def test_different_txids_are_both_kept(provenance):
    records = [
        make_record(provenance, txid=TXID_A, fee="0.001"),
        make_record(provenance, txid=TXID_B, fee="0.001"),
    ]
    assert deduplicate(records).unique_count == 2


def test_records_differing_only_in_unmapped_extra_are_kept(provenance):
    """Unmapped extras are evidence, so a differing extra keeps both records.

    Two rows can agree on all fourteen canonical SIH columns and still be two
    pieces of evidence: ``block_height`` 840000 and 840001 describe different
    observations.  Dedup must key on the unmapped fields too, otherwise one of
    them is silently destroyed.
    """
    a = NormalizedRecord.from_raw(
        {"txid": TXID_A, "block_height": 840000}, prov("row 2")
    )
    b = NormalizedRecord.from_raw(
        {"txid": TXID_A, "block_height": 840001}, prov("row 3")
    )
    assert a.txid == b.txid
    assert a.canonical_payload() == b.canonical_payload()
    assert a.extra != b.extra

    # The unmapped difference must reach the deduplication identity.
    assert record_fingerprint(a) != record_fingerprint(b)

    result = deduplicate([a, b])
    assert result.unique_count == 2
    assert result.duplicate_count == 0
    assert result.groups == ()
    # Both pieces of evidence remain readable.
    assert sorted(r.extra["block_height"] for r in result.records) == [840000, 840001]


def test_identical_records_with_identical_extras_are_merged():
    a = NormalizedRecord.from_raw({"txid": TXID_A, "confirmations": 6}, prov("row 2"))
    b = NormalizedRecord.from_raw({"txid": TXID_A, "confirmations": 6}, prov("row 3"))
    assert deduplicate([a, b]).unique_count == 1


def test_extras_matching_in_a_different_key_order_still_merge():
    """Key order in an unmapped mapping is not a difference in evidence."""
    a = NormalizedRecord.from_raw(
        {"txid": TXID_A, "meta": {"height": 840000, "confirmations": 6}},
        prov("row 2"),
    )
    b = NormalizedRecord.from_raw(
        {"txid": TXID_A, "meta": {"confirmations": 6, "height": 840000}},
        prov("row 3"),
    )
    assert a.extra == b.extra
    assert record_fingerprint(a) == record_fingerprint(b)
    assert deduplicate([a, b]).unique_count == 1


def test_unmapped_list_order_is_treated_as_evidence():
    """Sequence order is meaningful, so a reordered list must not merge."""
    a = NormalizedRecord.from_raw({"txid": TXID_A, "path": [1, 2]}, prov("row 2"))
    b = NormalizedRecord.from_raw({"txid": TXID_A, "path": [2, 1]}, prov("row 3"))
    assert record_fingerprint(a) != record_fingerprint(b)
    assert deduplicate([a, b]).unique_count == 2


def test_unmapped_value_types_are_distinguished():
    """``1``, ``"1"`` and ``True`` are three different claims, not one."""
    records = [
        NormalizedRecord.from_raw({"txid": TXID_A, "v": 1}, prov("row 2")),
        NormalizedRecord.from_raw({"txid": TXID_A, "v": "1"}, prov("row 3")),
        NormalizedRecord.from_raw({"txid": TXID_A, "v": True}, prov("row 4")),
    ]
    fingerprints = {record_fingerprint(r) for r in records}
    assert len(fingerprints) == 3
    assert deduplicate(records).unique_count == 3


def test_nested_unmapped_differences_are_detected():
    """A difference nested inside an unmapped mapping must prevent a merge."""
    a = NormalizedRecord.from_raw(
        {"txid": TXID_A, "meta": {"block": {"height": 840000}}}, prov("row 2")
    )
    b = NormalizedRecord.from_raw(
        {"txid": TXID_A, "meta": {"block": {"height": 840001}}}, prov("row 3")
    )
    assert record_fingerprint(a) != record_fingerprint(b)
    assert deduplicate([a, b]).unique_count == 2


def test_fingerprint_is_deterministic_across_ingestion_runs(service):
    """The same evidence must fingerprint identically every time."""
    first = service.ingest_bytes(csv_bytes(csv_row(txid=TXID_A), csv_row(txid=TXID_A)))
    second = service.ingest_bytes(csv_bytes(csv_row(txid=TXID_A), csv_row(txid=TXID_A)))
    assert [record_fingerprint(r) for r in first.records] == [
        record_fingerprint(r) for r in second.records
    ]


def test_provenance_alone_does_not_prevent_a_merge(provenance):
    """Provenance is the one thing a merge may drop, and it never blocks one."""
    a = make_record(prov("row 2", "a.csv", "run-1"), src_ip="1.2.3.4", fee="0.1")
    b = make_record(prov("row 9", "b.csv", "run-2"), src_ip="1.2.3.4", fee="0.1")
    assert record_fingerprint(a) == record_fingerprint(b)
    result = deduplicate([a, b])
    assert result.unique_count == 1
    assert result.duplicate_count == 1
    # The dropped copy is still auditable through the group.
    assert result.groups[0].merged == ("row 9",)
    assert result.groups[0].sources == ("b.csv",)


def test_same_addresses_different_output_amounts_are_kept(provenance):
    records = [
        make_record(provenance, input_addresses=[ADDR_1], input_amounts=["1.0"]),
        make_record(provenance, input_addresses=[ADDR_1], input_amounts=["2.0"]),
    ]
    assert deduplicate(records).unique_count == 2


def test_network_and_blockchain_records_with_overlap_are_kept(provenance):
    records = [
        make_record(provenance, src_ip="1.2.3.4", txid=TXID_A),
        make_record(provenance, src_ip="1.2.3.4", txid=TXID_A, fee="0.5"),
    ]
    assert deduplicate(records).unique_count == 2


# ---------------------------------------------------------------- safety
def test_invalid_records_are_never_merged(provenance):
    """Two different malformed records must not be collapsed together."""
    a = make_record(prov("row 2"), src_ip="bad-one", dst_ip="1.1.1.1")
    b = make_record(prov("row 3"), src_ip="bad-two", dst_ip="1.1.1.1")
    assert not a.is_valid and not b.is_valid
    result = deduplicate([a, b])
    assert result.unique_count == 2


def test_invalid_record_does_not_absorb_a_valid_lookalike(provenance):
    a = make_record(prov("row 2"), src_ip="1.1.1.1")
    b = make_record(prov("row 3"), src_ip="1.1.1.1", fee="notanumber")
    assert deduplicate([a, b]).unique_count == 2


def test_empty_input_is_safe(provenance):
    result = deduplicate([])
    assert result.unique_count == 0
    assert result.records == []


def test_single_record_is_unchanged(provenance):
    records = [make_record(provenance, src_ip="1.2.3.4")]
    result = deduplicate(records)
    assert result.unique_count == 1
    assert result.duplicate_count == 0


def test_input_count_is_the_sum_of_unique_and_duplicates(provenance):
    records = [
        make_record(prov("row 2"), src_ip="1.1.1.1"),
        make_record(prov("row 3"), src_ip="1.1.1.1"),
        make_record(prov("row 4"), src_ip="2.2.2.2"),
    ]
    result = deduplicate(records)
    assert result.input_count == 3
    assert result.unique_count + result.duplicate_count == 3


def test_three_way_duplicate_is_collapsed_to_one(provenance):
    records = [
        make_record(prov("row 2"), src_ip="1.1.1.1"),
        make_record(prov("row 3", "b.csv"), src_ip="1.1.1.1"),
        make_record(prov("row 4", "c.csv", "run-3"), src_ip="1.1.1.1"),
    ]
    result = deduplicate(records)
    assert result.unique_count == 1
    assert result.duplicate_count == 2


# ---------------------------------------------------------------- service
def test_service_deduplicates_by_default(service):
    row = csv_row(timestamp="2024-05-01T00:00:00Z", src_ip="1.2.3.4",
                  dst_ip="1.2.3.5", src_port=1000, dst_port=443)
    result = service.ingest_bytes(csv_bytes(row, row))
    assert result.raw_record_count == 2
    assert result.record_count == 1
    assert result.duplicate_count == 1
    # A structurally sound row must not raise a RAGGED_RECORD complaint.
    assert result.issues == ()


def test_deduplication_can_be_disabled(service):
    row = csv_row(timestamp="2024-05-01T00:00:00Z", src_ip="1.2.3.4",
                  dst_ip="1.2.3.5", src_port=1000, dst_port=443)
    result = service.ingest_bytes(csv_bytes(row, row), deduplicate_records=False)
    assert result.record_count == 2
    assert result.duplicate_count == 0
    assert result.issues == ()


def test_whole_row_duplicates_including_fees_collapse(service):
    row = csv_row(timestamp="2024-05-01T00:00:00Z", txid=TXID_A,
                  input_addresses=[ADDR_1], output_addresses=[ADDR_3],
                  input_amounts=["1.5"], output_amounts=["1.49"],
                  fee="0.0001", script_type="P2WPKH", geo_country="IN", asn="AS1")
    result = service.ingest_bytes(csv_bytes(row, row))
    assert result.record_count == 1
    assert result.issues == ()


def test_fee_difference_prevents_merge_end_to_end(service):
    first = csv_row(timestamp="2024-05-01T00:00:00Z", txid=TXID_A,
                    fee="0.0001", script_type="P2WPKH", geo_country="IN", asn="AS1")
    second = csv_row(timestamp="2024-05-01T00:00:00Z", txid=TXID_A,
                     fee="0.0002", script_type="P2WPKH", geo_country="IN", asn="AS1")
    result = service.ingest_bytes(csv_bytes(first, second))
    assert result.record_count == 2
    assert result.issues == ()


def test_end_to_end_differing_unmapped_column_keeps_both_rows(service):
    """Two rows differing only in an unmapped column must both survive."""
    header = "timestamp,txid,block_height"
    result = service.ingest_bytes(
        csv_bytes(
            "2024-05-01T00:00:00Z,%s,840000" % TXID_A,
            "2024-05-01T00:00:00Z,%s,840001" % TXID_A,
            header=header,
        )
    )
    assert result.raw_record_count == 2
    assert result.record_count == 2
    assert result.duplicate_count == 0
    assert result.issues == ()
    # CSV cells are text, so the unmapped values are preserved verbatim.
    assert sorted(r.extra["block_height"] for r in result.records) == [
        "840000",
        "840001",
    ]
