"""Step 4A - the evidence relationship semantic model.

These tests cover the *vocabulary* of evidence-aware correlation: what a
relationship means, what must accompany one, and -- just as importantly -- what
it must never mean.

The relationship model is not an engine.  Nothing here fires a rule, matches
records, scores risk or builds a graph; every relationship is constructed
explicitly.  What is being pinned down is the meaning of the parts, so that the
engine built later cannot quietly redefine them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from backend.app.correlation import (
    Confidence,
    ConfidenceBasis,
    EvidenceRelationship,
    EvidenceType,
    InferenceLevel,
    ProvenanceRef,
    TemporalRelation,
    TemporalWindow,
    derive_relationship_id,
)
from backend.app.domain.models import OBSERVATION_NAMESPACE, UnifiedObservation

UTC = timezone.utc
NOW = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
OBS_A = "0d2f6a1e-0000-5000-8000-000000000001"
OBS_B = "0d2f6a1e-0000-5000-8000-000000000002"


def _ref(ingestion_id: str = "ing-1") -> ProvenanceRef:
    return ProvenanceRef(
        ingestion_id=ingestion_id, source_file="evidence.csv", source_record="row 2"
    )


def _relationship(**overrides) -> EvidenceRelationship:
    payload = {
        "relationship_id": derive_relationship_id({"test": 1}),
        "source_observation_ids": (OBS_A,),
        "target_observation_ids": (OBS_B,),
        "evidence_type": EvidenceType.SAME_IP,
        "inference_level": InferenceLevel.CORRELATED,
        "confidence": Confidence.exact_match(),
        "rationale": "Both observations carry src_ip 203.0.113.5 as recorded.",
        "rule_id": "rule.same_ip",
        "rule_version": "1.0.0",
        "provenance_refs": (_ref(),),
        "created_at": NOW,
    }
    payload.update(overrides)
    return EvidenceRelationship(**payload)


# ------------------------------------------------------------------ enums
class TestInferenceLevel:
    def test_every_documented_level_exists(self):
        assert [level.value for level in InferenceLevel] == [
            "OBSERVED",
            "CORRELATED",
            "INFERRED",
            "REQUIRES_VERIFICATION",
        ]

    def test_it_is_an_enum_not_a_boolean(self):
        import enum

        assert issubclass(InferenceLevel, enum.Enum)
        # A bool-like model would have exactly two states; this has four.
        assert len(InferenceLevel) > 2
        assert not isinstance(InferenceLevel.CORRELATED, bool)

    def test_values_are_strings_and_round_trip(self):
        assert InferenceLevel.REQUIRES_VERIFICATION == "REQUIRES_VERIFICATION"
        assert InferenceLevel("INFERRED") is InferenceLevel.INFERRED

    def test_unknown_level_is_rejected(self):
        with pytest.raises(ValueError):
            InferenceLevel("PROVEN")

    def test_verification_remains_an_explicit_state(self):
        """It is a first-class level, not a fallback or an afterthought."""
        assert "REQUIRES_VERIFICATION" in InferenceLevel._value2member_map_


class TestEvidenceType:
    def test_every_documented_type_exists(self):
        assert {member.value for member in EvidenceType} == {
            "SAME_SOURCE_RECORD",
            "SAME_TXID",
            "SAME_ADDRESS",
            "SAME_IP",
            "TEMPORAL_ADJACENCY",
            "AMOUNT_MATCH",
            "PROVENANCE_LINK",
            "GEO_COINCIDENCE",
        }

    def test_it_is_an_enum(self):
        import enum

        assert issubclass(EvidenceType, enum.Enum)

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError):
            EvidenceType("SAME_PERSON")

    def test_evidence_type_and_inference_level_are_separate_axes(self):
        """Never merge them into one enum."""
        assert set(EvidenceType) != set(InferenceLevel)
        assert not set(m.name for m in EvidenceType) & set(
            m.name for m in InferenceLevel
        )
        # The same evidence kind can sit at different inference levels.
        for level in InferenceLevel:
            rel = _relationship(inference_level=level)
            assert rel.evidence_type is EvidenceType.SAME_IP
            assert rel.inference_level is level


class TestConfidenceBasis:
    def test_no_machine_learned_basis_exists_yet(self):
        assert {b.value for b in ConfidenceBasis} == {
            "EXACT_MATCH",
            "DETERMINISTIC_RULE",
            "HUMAN_ASSESSMENT",
        }
        assert not any("LEARN" in b.value or "MODEL" in b.value for b in ConfidenceBasis)


# -------------------------------------------------------------- confidence
class TestConfidence:
    @pytest.mark.parametrize("value", [0.0, 0.001, 0.5, 0.999, 1.0])
    def test_accepts_values_in_the_unit_interval(self, value):
        assert Confidence(value=value, basis=ConfidenceBasis.DETERMINISTIC_RULE).value == value

    @pytest.mark.parametrize("value", [-0.0001, -1.0, 1.0001, 2.0, 100.0])
    def test_rejects_values_outside_the_unit_interval(self, value):
        with pytest.raises(ValidationError):
            Confidence(value=value, basis=ConfidenceBasis.DETERMINISTIC_RULE)

    def test_rejects_nan(self):
        with pytest.raises(ValidationError):
            Confidence(value=float("nan"), basis=ConfidenceBasis.DETERMINISTIC_RULE)

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_rejects_infinities(self, value):
        with pytest.raises(ValidationError):
            Confidence(value=value, basis=ConfidenceBasis.DETERMINISTIC_RULE)

    def test_boundaries_are_inclusive(self):
        for value in (0.0, 1.0):
            Confidence(value=value, basis=ConfidenceBasis.EXACT_MATCH)

    def test_requires_a_documented_basis(self):
        with pytest.raises(ValidationError):
            Confidence(value=0.5)

    def test_unknown_basis_is_rejected(self):
        with pytest.raises(ValidationError):
            Confidence(value=0.5, basis="MADE_UP")

    def test_is_a_typed_model_not_a_bare_float(self):
        """A bare float cannot explain where its number came from."""
        confidence = Confidence.from_rule(0.75)
        assert isinstance(confidence, Confidence)
        assert not isinstance(confidence, float)
        assert confidence.basis is ConfidenceBasis.DETERMINISTIC_RULE

    def test_named_constructors_document_their_basis(self):
        assert Confidence.exact_match().value == 1.0
        assert Confidence.exact_match().basis is ConfidenceBasis.EXACT_MATCH
        assert Confidence.from_rule(0.4).basis is ConfidenceBasis.DETERMINISTIC_RULE
        assert Confidence.from_assessment(0.9).basis is ConfidenceBasis.HUMAN_ASSESSMENT

    def test_named_constructors_still_validate(self):
        with pytest.raises(ValidationError):
            Confidence.from_rule(1.5)

    def test_is_immutable(self):
        confidence = Confidence.exact_match()
        with pytest.raises(ValidationError):
            confidence.value = 0.1

    def test_is_serialisable(self):
        assert Confidence.exact_match().model_dump() == {
            "value": 1.0,
            "basis": ConfidenceBasis.EXACT_MATCH,
        }

    def test_extra_fields_are_refused(self):
        with pytest.raises(ValidationError):
            Confidence(value=0.5, basis=ConfidenceBasis.DETERMINISTIC_RULE, model="x")


# ------------------------------------------------------------- provenance
class TestProvenanceRef:
    def test_carries_the_minimum_locator(self):
        ref = _ref()
        assert ref.ingestion_id == "ing-1"
        assert ref.source_file == "evidence.csv"
        assert ref.source_record == "row 2"

    def test_source_file_and_record_are_optional(self):
        ref = ProvenanceRef(ingestion_id="ing-1")
        assert ref.source_file is None
        assert ref.source_record is None

    def test_ingestion_id_is_required(self):
        with pytest.raises(ValidationError):
            ProvenanceRef()

    @pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
    def test_blank_ingestion_id_is_rejected(self, value):
        with pytest.raises(ValidationError):
            ProvenanceRef(ingestion_id=value)

    @pytest.mark.parametrize("field", ["source_file", "source_record"])
    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_optional_locators_are_rejected(self, field, value):
        with pytest.raises(ValidationError):
            ProvenanceRef(ingestion_id="ing-1", **{field: value})

    def test_is_immutable(self):
        ref = _ref()
        with pytest.raises(ValidationError):
            ref.ingestion_id = "other"

    def test_is_serialisable_and_round_trips(self):
        ref = _ref()
        assert ProvenanceRef.model_validate_json(ref.model_dump_json()) == ref
        assert json.loads(ref.model_dump_json())["ingestion_id"] == "ing-1"

    def test_does_not_duplicate_whole_observation_provenance(self):
        """A reference, not a copy of the substrate."""
        assert set(ProvenanceRef.model_fields) == {
            "ingestion_id",
            "source_file",
            "source_record",
        }
        for forbidden in ("context", "ingested_at", "source_format", "network",
                          "blockchain", "extra"):
            assert forbidden not in ProvenanceRef.model_fields

    def test_references_are_retained_on_the_relationship(self):
        refs = (_ref("ing-1"), _ref("ing-2"), ProvenanceRef(ingestion_id="ing-3"))
        assert _relationship(provenance_refs=refs).provenance_refs == refs


# --------------------------------------------------------------- temporal
class TestTemporalRelation:
    def test_every_documented_relation_exists(self):
        assert [relation.value for relation in TemporalRelation] == [
            "BEFORE",
            "AFTER",
            "WITHIN",
            "OVERLAPS",
        ]

    def test_unknown_relation_is_rejected(self):
        with pytest.raises(ValueError):
            TemporalRelation("DURING")


class TestTemporalWindow:
    def test_accepts_an_explicit_aware_window(self):
        window = TemporalWindow(
            start=datetime(2024, 5, 1, tzinfo=UTC), end=datetime(2024, 5, 2, tzinfo=UTC)
        )
        assert window.duration_seconds == 86400.0

    def test_allows_a_zero_length_window(self):
        moment = datetime(2024, 5, 1, tzinfo=UTC)
        assert TemporalWindow(start=moment, end=moment).duration_seconds == 0.0

    def test_rejects_start_after_end(self):
        with pytest.raises(ValidationError):
            TemporalWindow(
                start=datetime(2024, 5, 2, tzinfo=UTC),
                end=datetime(2024, 5, 1, tzinfo=UTC),
            )

    def test_rejects_naive_bounds(self):
        with pytest.raises(ValidationError):
            TemporalWindow(start=datetime(2024, 5, 1), end=datetime(2024, 5, 2, tzinfo=UTC))
        with pytest.raises(ValidationError):
            TemporalWindow(start=datetime(2024, 5, 1, tzinfo=UTC), end=datetime(2024, 5, 2))

    def test_rejects_missing_bounds(self):
        with pytest.raises(ValidationError):
            TemporalWindow(start=datetime(2024, 5, 1, tzinfo=UTC))

    def test_non_utc_timezones_are_preserved_not_silently_shifted(self):
        tz = timezone(timedelta(hours=5, minutes=30))
        window = TemporalWindow(
            start=datetime(2024, 5, 1, tzinfo=tz), end=datetime(2024, 5, 1, 1, tzinfo=tz)
        )
        assert window.start.utcoffset() == timedelta(hours=5, minutes=30)
        assert window.duration_seconds == 3600.0

    def test_compares_correctly_across_timezones(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        utc_window = TemporalWindow(
            start=datetime(2024, 5, 1, 0, 0, tzinfo=UTC), end=datetime(2024, 5, 1, 1, 0, tzinfo=UTC)
        )
        ist_window = TemporalWindow(
            start=datetime(2024, 5, 1, 5, 30, tzinfo=ist), end=datetime(2024, 5, 1, 6, 30, tzinfo=ist)
        )
        assert utc_window.overlaps(ist_window)

    def test_contains_is_inclusive_and_rejects_naive_moments(self):
        window = TemporalWindow(
            start=datetime(2024, 5, 1, tzinfo=UTC), end=datetime(2024, 5, 2, tzinfo=UTC)
        )
        assert window.contains(datetime(2024, 5, 1, tzinfo=UTC))
        assert window.contains(datetime(2024, 5, 2, tzinfo=UTC))
        assert not window.contains(datetime(2024, 5, 3, tzinfo=UTC))
        with pytest.raises(ValueError):
            window.contains(datetime(2024, 5, 1))

    def test_is_immutable(self):
        window = TemporalWindow(
            start=datetime(2024, 5, 1, tzinfo=UTC), end=datetime(2024, 5, 2, tzinfo=UTC)
        )
        with pytest.raises(ValidationError):
            window.start = NOW

    def test_is_serialisable_and_round_trips(self):
        window = TemporalWindow(
            start=datetime(2024, 5, 1, tzinfo=UTC), end=datetime(2024, 5, 2, tzinfo=UTC)
        )
        assert TemporalWindow.model_validate_json(window.model_dump_json()) == window

    def test_temporal_relation_requires_a_window(self):
        with pytest.raises(ValidationError):
            _relationship(temporal_relation=TemporalRelation.WITHIN)

    def test_temporal_pair_is_retained(self):
        window = TemporalWindow(
            start=datetime(2024, 5, 1, tzinfo=UTC), end=datetime(2024, 5, 2, tzinfo=UTC)
        )
        rel = _relationship(
            temporal_window=window, temporal_relation=TemporalRelation.BEFORE
        )
        assert rel.temporal_window == window
        assert rel.temporal_relation is TemporalRelation.BEFORE


# ----------------------------------------------------------- relationship
class TestEvidenceRelationship:
    def test_keeps_all_required_components(self):
        rel = _relationship()
        assert rel.evidence_type is EvidenceType.SAME_IP
        assert rel.inference_level is InferenceLevel.CORRELATED
        assert rel.confidence.value == 1.0
        assert rel.rule_id == "rule.same_ip"
        assert rel.rule_version == "1.0.0"
        assert rel.rationale
        assert rel.provenance_refs == (_ref(),)
        assert rel.created_at == NOW

    def test_source_and_target_ids_are_explicit(self):
        rel = _relationship(
            source_observation_ids=("obs-a", "obs-b"), target_observation_ids=("obs-c",)
        )
        assert rel.source_observation_ids == ("obs-a", "obs-b")
        assert rel.target_observation_ids == ("obs-c",)
        assert set(rel.all_observation_ids) == {"obs-a", "obs-b", "obs-c"}

    @pytest.mark.parametrize("field", ["source_observation_ids", "target_observation_ids"])
    def test_rejects_empty_id_collections(self, field):
        with pytest.raises(ValidationError):
            _relationship(**{field: ()})

    @pytest.mark.parametrize("field", ["source_observation_ids", "target_observation_ids"])
    @pytest.mark.parametrize("blank", ["", "   "])
    def test_rejects_blank_ids(self, field, blank):
        with pytest.raises(ValidationError):
            _relationship(**{field: (blank,)})

    @pytest.mark.parametrize("field", ["source_observation_ids", "target_observation_ids"])
    def test_rejects_duplicate_ids(self, field):
        with pytest.raises(ValidationError):
            _relationship(**{field: ("obs-a", "obs-a")})

    def test_accepts_many_ids_on_both_sides(self):
        rel = _relationship(
            source_observation_ids=("a1", "a2", "a3"), target_observation_ids=("b1", "b2")
        )
        assert len(rel.source_observation_ids) == 3
        assert len(rel.target_observation_ids) == 2

    def test_rejects_invalid_confidence(self):
        with pytest.raises(ValidationError):
            _relationship(confidence=1.5)
        with pytest.raises(ValidationError):
            _relationship(confidence=float("nan"))

    def test_rejects_blank_rationale(self):
        for blank in ("", "   ", "\n"):
            with pytest.raises(ValidationError):
                _relationship(rationale=blank)

    def test_rejects_a_rationale_that_does_not_explain(self):
        with pytest.raises(ValidationError):
            _relationship(rationale="because")

    def test_rejects_missing_rule_id(self):
        payload = _relationship().model_dump()
        del payload["rule_id"]
        with pytest.raises(ValidationError):
            EvidenceRelationship(**payload)

    def test_rejects_missing_rule_version(self):
        payload = _relationship().model_dump()
        del payload["rule_version"]
        with pytest.raises(ValidationError):
            EvidenceRelationship(**payload)

    @pytest.mark.parametrize("field", ["rule_id", "rule_version", "rationale"])
    def test_rejects_blank_rule_and_rationale_fields(self, field):
        with pytest.raises(ValidationError):
            _relationship(**{field: "   "})

    def test_rejects_invalid_evidence_type(self):
        with pytest.raises(ValidationError):
            _relationship(evidence_type="SAME_PERSON")

    def test_rejects_invalid_inference_level(self):
        with pytest.raises(ValidationError):
            _relationship(inference_level="PROVEN")

    def test_rejects_naive_created_at(self):
        with pytest.raises(ValidationError):
            _relationship(created_at=datetime(2024, 5, 1))

    def test_rejects_invalid_provenance_refs(self):
        with pytest.raises(ValidationError):
            _relationship(provenance_refs=({"ingestion_id": ""},))
        with pytest.raises(ValidationError):
            _relationship(provenance_refs=("not-a-ref",))

    def test_provenance_refs_default_to_empty_but_never_drop_silent_ones(self):
        assert _relationship(provenance_refs=()).provenance_refs == ()
        assert len(_relationship().provenance_refs) == 1

    def test_is_immutable(self):
        rel = _relationship()
        for field, value in (
            ("rationale", "something else entirely"),
            ("rule_id", "other"),
            ("inference_level", InferenceLevel.INFERRED),
        ):
            with pytest.raises(ValidationError):
                setattr(rel, field, value)

    def test_is_serialisable_and_round_trips(self):
        rel = _relationship()
        assert EvidenceRelationship.model_validate_json(rel.model_dump_json()) == rel
        assert json.loads(rel.model_dump_json())["evidence_type"] == "SAME_IP"

    def test_refuses_unknown_extra_fields(self):
        payload = _relationship().model_dump()
        payload["wallet_ownership"] = "some-wallet"
        with pytest.raises(ValidationError):
            EvidenceRelationship(**payload)

    def test_self_reference_is_reported_not_rejected(self):
        """Reported, because only a rule can decide whether it is meaningful."""
        rel = _relationship(
            source_observation_ids=("obs-a",), target_observation_ids=("obs-a",)
        )
        assert rel.is_self_referential is True
        assert _relationship().is_self_referential is False

    def test_deterministic_identity_for_the_same_semantic_content(self):
        first = derive_relationship_id({"evidence_type": "SAME_IP", "rule": "r1"})
        second = derive_relationship_id({"rule": "r1", "evidence_type": "SAME_IP"})
        assert first == second

    def test_identity_differs_for_different_content(self):
        assert derive_relationship_id({"evidence_type": "SAME_IP"}) != derive_relationship_id(
            {"evidence_type": "SAME_TXID"}
        )

    def test_relationship_id_cannot_collide_with_an_observation_id(self):
        payload = {"type": "SAME_IP", "network": None, "extra": {}}
        assert derive_relationship_id(payload) != str(
            __import__("uuid").uuid5(OBSERVATION_NAMESPACE, json.dumps(payload, sort_keys=True))
        )


# --------------------------------------------------------- anti-attribution
class TestNoAttribution:
    """The guarantee that makes the model defensible."""

    FORBIDDEN_FIELDS = (
        "attribution",
        "attributed_to",
        "wallet_ownership",
        "wallet_owner",
        "owner",
        "owner_address",
        "belongs_to",
        "person",
        "person_id",
        "identity",
        "entity",
        "entity_id",
        "suspect",
        "suspect_address",
        "risk_score",
        "suspicion_score",
        "fraud",
        "guilty",
        "controlled_by",
    )

    def test_relationship_has_no_attribution_field(self):
        for field in self.FORBIDDEN_FIELDS:
            assert field not in EvidenceRelationship.model_fields, field

    def test_confidence_has_no_attribution_field(self):
        assert set(Confidence.model_fields) == {"value", "basis"}

    def test_a_same_ip_relationship_carries_no_attribution_claim(self):
        rel = _relationship(
            evidence_type=EvidenceType.SAME_IP, inference_level=InferenceLevel.CORRELATED
        )
        dumped = rel.model_dump()
        for field in self.FORBIDDEN_FIELDS:
            assert field not in dumped, field
        # Serialised output is equally clean.
        rendered = rel.model_dump_json().lower()
        for field in self.FORBIDDEN_FIELDS:
            assert field not in rendered, field

    def test_explains_attribution_is_always_false(self):
        for evidence_type in EvidenceType:
            for level in InferenceLevel:
                rel = _relationship(evidence_type=evidence_type, inference_level=level)
                assert rel.explains_attribution() is False

    def test_same_txid_does_not_create_attribution(self):
        """A shared TXID is not evidence that any IP belongs to the wallet."""
        rel = _relationship(
            evidence_type=EvidenceType.SAME_TXID,
            inference_level=InferenceLevel.REQUIRES_VERIFICATION,
        )
        assert rel.explains_attribution() is False
        assert "attribution" not in rel.model_dump_json().lower()
        assert "ownership" not in rel.model_dump_json().lower()

    def test_geo_coincidence_makes_no_location_claim(self):
        rel = _relationship(evidence_type=EvidenceType.GEO_COINCIDENCE)
        rendered = rel.model_dump_json().lower()
        assert "location" not in rendered
        assert "physical" not in rendered
        assert rel.explains_attribution() is False

    def test_certain_comparison_is_still_not_attribution(self):
        """Even maximum confidence asserts a match, never an owner."""
        rel = _relationship(confidence=Confidence.exact_match())
        assert rel.confidence.value == 1.0
        assert rel.explains_attribution() is False
        assert "owner" not in rel.model_dump_json().lower()

    def test_evidence_types_carry_no_ownership_semantics(self):
        for member in EvidenceType:
            name = member.name.lower()
            for banned in ("owner", "own", "wallet", "person", "entity", "suspect"):
                assert banned not in name, member

    def test_step_1_to_3_substrate_is_unmodified_and_unreferenced(self):
        """Step 4A reads nothing and mutates nothing from the substrate.

        Checked against the parsed import graph rather than the raw text: the
        module docstrings legitimately *name* ``UnifiedObservation`` to explain
        the boundary, and a substring search would flag that prose as if it
        were a dependency.
        """
        import ast
        import pathlib

        import backend.app.correlation as package

        forbidden_prefixes = ("backend.app.domain", "backend.app.ingestion")
        offenders = []
        for path in pathlib.Path(package.__file__).parent.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith(forbidden_prefixes):
                        offenders.append((path.name, node.module))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(forbidden_prefixes):
                            offenders.append((path.name, alias.name))
        assert offenders == []

    def test_step_1_to_3_sources_are_untouched_by_step_4a(self):
        """No file in Steps 1-3 mentions the correlation package."""
        import pathlib

        import backend.app as app_root

        for path in pathlib.Path(app_root.__file__).parent.rglob("*.py"):
            if "correlation" in path.parts:
                continue
            assert "app.correlation" not in path.read_text(encoding="utf-8"), path.name

    def test_correlation_package_defines_no_engine(self):
        """No rule, registry, graph, store or scoring entry point."""
        import backend.app.correlation as package

        forbidden = (
            "build_relationships",
            "correlate",
            "run_rule",
            "RuleRegistry",
            "Graph",
            "Node",
            "Edge",
            "score",
            "detect_anomalies",
            "cluster",
            "persist",
            "store",
            "save",
            "router",
            "app",
            "APIRouter",
        )
        exported = set(package.__all__)
        for name in forbidden:
            assert not any(name.lower() in item.lower() for item in exported), name
