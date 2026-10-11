"""How unmapped source evidence becomes a record's identity.

Every unmapped field is *evidence*, not decoration, so it takes part in both
the Step 2 deduplication fingerprint and the Step 3 ``observation_id``.  Two
records that agree on every canonical SIH field but differ in an unmapped field
are two pieces of evidence and must never collapse into one.

The invariant these tests defend:

    UNKNOWN A != UNKNOWN B

unless a deterministic, justified equality representation proves they are
equivalent.  Inventing equality for a type we cannot compare is how a forensic
pipeline silently destroys evidence.
"""

from __future__ import annotations

from backend.app.domain.adapters import normalized_to_unified
from backend.app.domain.models import Provenance
from backend.app.domain.validation import canonicalize_evidence
from backend.app.ingestion.dedup import deduplicate, record_fingerprint
from backend.app.ingestion.models import NormalizedRecord
from tests.conftest import TXID_A


def _prov(locator: str = "row 1") -> Provenance:
    return Provenance(
        ingestion_id="evidence-identity",
        source_file="evidence.csv",
        source_format="csv",
        source_record=locator,
        context={},
    )


def _record(**extra) -> NormalizedRecord:
    return NormalizedRecord.from_raw({"txid": TXID_A, **extra}, _prov())


class Opaque:
    """A type with no defined equality semantics of its own."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __repr__(self) -> str:
        return f"Opaque({self.marker!r})"


# ------------------------------------------- 1. opaque values never collide
def test_two_distinct_opaque_values_of_the_same_type_do_not_collide():
    """The defect this file exists to prevent.

    Both values are ``Opaque``.  Rendering either from its type name alone made
    the two canonicalize identically, so deduplication would have merged two
    unrelated pieces of evidence into one record.
    """
    left = canonicalize_evidence(Opaque("a"))
    right = canonicalize_evidence(Opaque("b"))

    assert left != right, "distinct opaque values collapsed to the same canonical form"
    # Same type name, different identity.
    assert left[:2] == right[:2] == ["opaque", "Opaque"]


def test_distinct_opaque_values_produce_distinct_fingerprints():
    a = _record(blob=Opaque("a"))
    b = _record(blob=Opaque("b"))
    assert record_fingerprint(a) != record_fingerprint(b)


def test_opaque_records_are_never_merged_by_deduplication():
    result = deduplicate([_record(blob=Opaque("a")), _record(blob=Opaque("b"))])
    assert result.unique_count == 2
    assert result.duplicate_count == 0
    # Neither record may absorb the other's evidence.
    assert [r.extra["blob"].marker for r in result.records] == ["a", "b"]


def test_even_the_same_opaque_object_is_not_merged():
    """Without proven equality, the safe answer is 'not the same'.

    Merging is only justified by a deterministic representation, and none
    exists for this type, so a repeated reference stays a separate record.
    """
    shared = Opaque("same")
    result = deduplicate([_record(blob=shared), _record(blob=shared)])
    assert result.unique_count == 2
    assert result.duplicate_count == 0


def test_opaque_nested_in_a_container_is_still_unique():
    """The uniqueness must hold at depth, not just at the top level."""
    a = canonicalize_evidence({"meta": [Opaque("a")]})
    b = canonicalize_evidence({"meta": [Opaque("b")]})
    assert a != b


def test_opaque_evidence_yields_distinct_observation_ids():
    a = normalized_to_unified(_record(blob=Opaque("a")))
    b = normalized_to_unified(_record(blob=Opaque("b")))
    assert a.observation_id != b.observation_id


def test_opaque_type_name_is_still_reported():
    """The token must not erase the only useful description we have."""
    canonical = canonicalize_evidence(Opaque("a"))
    assert canonical[0] == "opaque"
    assert canonical[1] == "Opaque"
    assert len(canonical) == 3


# ------------------------------- 2. JSON-like determinism is NOT weakened
def test_equal_dicts_canonicalize_identically():
    assert canonicalize_evidence({"a": 1, "b": [2, 3]}) == canonicalize_evidence(
        {"a": 1, "b": [2, 3]}
    )


def test_dictionary_key_order_does_not_matter():
    assert canonicalize_evidence({"a": 1, "b": 2}) == canonicalize_evidence(
        {"b": 2, "a": 1}
    )


def test_list_order_still_matters():
    assert canonicalize_evidence([1, 2]) != canonicalize_evidence([2, 1])


def test_scalar_type_differences_remain_meaningful():
    """1, "1" and True are three separate claims, not one."""
    assert canonicalize_evidence(1) != canonicalize_evidence("1")
    assert canonicalize_evidence(1) != canonicalize_evidence(True)
    assert canonicalize_evidence("1") != canonicalize_evidence(True)
    # int and float are different types with different forensic meaning.
    assert canonicalize_evidence(1) != canonicalize_evidence(1.0)


def test_known_scalar_types_are_not_treated_as_opaque():
    for value in (None, True, 1, 1.5, "s", b"b", [1], {"a": 1}, (1, 2), {1, 2}):
        assert canonicalize_evidence(value)[0] != "opaque", value


def test_json_like_fingerprints_stay_reproducible_across_calls():
    """Determinism must survive the opaque change for normal evidence."""
    a = _record(meta={"h": 1, "c": 2}, tags=["x", "y"])
    b = _record(meta={"h": 1, "c": 2}, tags=["x", "y"])
    assert record_fingerprint(a) == record_fingerprint(b)
    assert record_fingerprint(a) == record_fingerprint(a)


def test_normal_deduplication_still_collapses_identical_records():
    result = deduplicate([_record(meta={"h": 1}), _record(meta={"h": 1})])
    assert result.unique_count == 1
    assert result.duplicate_count == 1


def test_normal_deduplication_still_keeps_distinct_json_like_records():
    result = deduplicate(
        [_record(meta={"h": 1}), _record(meta={"h": 2}), _record(meta={"h": 1})]
    )
    assert result.unique_count == 2
    assert result.duplicate_count == 1


# --------------------------------- 3. dedup identity and observation_id agree
def test_dedup_and_observation_id_agree_on_distinct_extra():
    records = [_record(block_height=840000), _record(block_height=840001)]
    assert record_fingerprint(records[0]) != record_fingerprint(records[1])
    assert len({normalized_to_unified(r).observation_id for r in records}) == 2
    # And they are not merged.
    assert deduplicate(records).unique_count == 2


def test_dedup_and_observation_id_agree_on_identical_extra():
    records = [_record(block_height=840000), _record(block_height=840000)]
    assert record_fingerprint(records[0]) == record_fingerprint(records[1])
    assert len({normalized_to_unified(r).observation_id for r in records}) == 1
    assert deduplicate(records).unique_count == 1


def test_observation_id_is_stable_for_repeated_normal_records():
    a = _record(meta={"h": 1, "c": 2})
    b = _record(meta={"h": 1, "c": 2})
    assert normalized_to_unified(a).observation_id == normalized_to_unified(b).observation_id


def test_canonical_fields_alone_still_separate_records():
    """A canonical SIH field must separate records in dedup and identity."""
    a = _record(geo_country="IN", src_ip="203.0.113.5")
    b = _record(geo_country="US", src_ip="203.0.113.5")
    assert record_fingerprint(a) != record_fingerprint(b)
    assert normalized_to_unified(a).observation_id != normalized_to_unified(b).observation_id


def test_canonical_network_attribute_without_an_ip_still_affects_identity():
    """Regression: a preserved canonical field must reach the identity.

    ``geo_country`` with no IP address is no longer discarded with the rest of
    the sub-object, so it reaches the observation and therefore the derived
    ``observation_id``.  Two records differing only in it must not collide.
    """
    a = _record(geo_country="IN")
    b = _record(geo_country="US")

    assert normalized_to_unified(a).network is not None
    assert normalized_to_unified(a).network.geo_country == "IN"
    assert normalized_to_unified(b).network.geo_country == "US"
    assert normalized_to_unified(a).observation_id != normalized_to_unified(b).observation_id
    # Preserved, not reclassified: still no network identity was claimed.
    assert normalized_to_unified(a).network.has_network_identity is False


def test_preserved_network_field_is_not_pushed_into_extra():
    """A canonical field must not be demoted to an unmapped one."""
    observation = normalized_to_unified(_record(geo_country="IN", asn="AS15169"))
    assert observation.extra == {}
    assert observation.network.geo_country == "IN"
    assert observation.network.asn == 15169
