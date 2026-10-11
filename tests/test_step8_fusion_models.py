"""Step 8 -- the fusion contract vocabulary (SIH26146 - Step 8).

These tests pin the model layer of Step 8: the enums, the deterministic
identifier derivations, the band mapping, and every validator that keeps a
contribution honest.  A contribution that does not equal ``weight * value *
availability`` is a contradiction that must be caught at construction time, not
explained away later; missing signals are refused any numeric value; and the
configuration is the complete, replayable policy behind a score.
"""

from __future__ import annotations

import decimal
import json
import uuid
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.app.correlation.enums import ConfidenceBasis, InferenceLevel
from backend.app.fusion import (
    FusionAssessment,
    FusionConfig,
    FusionEngine,
    FusionError,
    FusionExplanation,
    FusionSignal,
    FusionStatus,
    MissingSignal,
    RiskBand,
    SignalContribution,
    SignalFamily,
    SignalRole,
    SignalStatus,
    band_for_score,
    derive_assessment_id,
    derive_signal_id,
)
from backend.app.fusion.models import (
    EVIDENCE_COUNT_CAP_DEFAULT,
    FUSION_DECIMAL_CONTEXT,
    FUSION_NAMESPACE,
    FUSION_SIGNAL_NAMESPACE,
    PATH_COUNT_CAP_DEFAULT,
    PROVENANCE_FILE_CAP_DEFAULT,
    ExplainabilityKind,
    canonical_json,
    ConfidenceContext,
)
from backend.app.fusion import FusionContext

from tests.test_step8_fusion import _all_weights, _pipeline

DEFAULT_CAP_EXPECTATIONS = (
    EVIDENCE_COUNT_CAP_DEFAULT,
    PATH_COUNT_CAP_DEFAULT,
    PROVENANCE_FILE_CAP_DEFAULT,
)


def _signal(
    signal_type: str = "TEST_SIGNAL",
    family: SignalFamily = SignalFamily.ANOMALY,
    role: SignalRole = SignalRole.PRIORITY_SIGNAL,
    status: SignalStatus = SignalStatus.PRESENT,
    source_step: int = 8,
    subject_id: str = "prof-subject-1",
    source_id: str = "source-1",
    raw_value: float | None = 0.5,
    normalized_value: float | None = 0.5,
    evidence_refs: tuple = (),
    inference_levels: tuple = (),
    context: tuple = (),
    explanation: str = "A real explanation sentence that is at least ten letters.",
) -> FusionSignal:
    return FusionSignal(
        signal_id=derive_signal_id(subject_id, signal_type, source_step, source_id),
        signal_type=signal_type,
        family=family,
        role=role,
        status=status,
        source_step=source_step,
        source_id=source_id,
        subject_id=subject_id,
        raw_value=raw_value,
        normalized_value=normalized_value,
        explanation=explanation,
        evidence_refs=evidence_refs,
        inference_levels=inference_levels,
        context=context,
    )


def _computed_assessment(config=None):
    subject, _population, graph, anomaly, clustering, paths = _pipeline()
    return FusionEngine(config).assess(
        FusionContext(
            profile=subject,
            graph=graph,
            anomaly=anomaly,
            clustering=clustering,
            paths=paths,
        )
    )


def _insufficient_assessment():
    subject = _pipeline()[0]
    config = FusionConfig(
        weights=_all_weights(
            anomaly="1.0",
            behavioral="0",
            cluster="0",
            evidence="0",
            path="0",
            provenance="0",
            temporal="0",
        )
    )
    return FusionEngine(config).assess(FusionContext(profile=subject))


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class TestVocabulary:
    def test_signal_family_has_exactly_seven_members(self):
        assert {item.value for item in SignalFamily} == {
            "ANOMALY",
            "BEHAVIORAL",
            "CLUSTER_CONTEXT",
            "EVIDENCE_GRAPH",
            "PATH_STRUCTURE",
            "PROVENANCE",
            "TEMPORAL",
        }

    def test_signal_status_has_exactly_four_members(self):
        assert {item.value for item in SignalStatus} == {
            "PRESENT",
            "ABSENT",
            "UNAVAILABLE",
            "NOT_APPLICABLE",
        }

    def test_risk_band_has_the_four_ascending_operating_points(self):
        assert [item.value for item in RiskBand] == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_fusion_status_has_exactly_two_members(self):
        assert {item.value for item in FusionStatus} == {"COMPUTED", "INSUFFICIENT_SIGNALS"}

    def test_explainability_kinds_are_a_fixed_set(self):
        assert {item.value for item in ExplainabilityKind} == {
            "TOP_CONTRIBUTOR",
            "SIGNAL_CONTRIBUTION",
            "MISSING_SIGNAL",
            "CLUSTER_CONTEXT_NOTE",
            "INFERENCE_LEVEL_NOTE",
            "OVERLAP_NOTE",
            "INTERPRETATION_NOTE",
        }

    def test_the_namespaces_are_fixed_uuids_unlike_every_other_step(self):
        assert str(FUSION_NAMESPACE) != str(FUSION_SIGNAL_NAMESPACE)
        assert str(uuid.UUID(str(FUSION_NAMESPACE))) == str(FUSION_NAMESPACE)
        assert str(uuid.UUID(str(FUSION_SIGNAL_NAMESPACE))) == str(FUSION_SIGNAL_NAMESPACE)


# --------------------------------------------------------------------------- #
# Deterministic identifiers
# --------------------------------------------------------------------------- #


class TestDeriveSignalId:
    def test_the_same_payload_produces_the_same_id(self):
        first = derive_signal_id("p1", "SAMPLE", 5, "src")
        second = derive_signal_id("p1", "SAMPLE", 5, "src")
        assert first == second
        assert first != derive_signal_id("p1", "SAMPLE", 5, "other")

    def test_every_dimension_changes_the_id(self):
        base = derive_signal_id("p1", "SAMPLE", 5, "src")
        assert base != derive_signal_id("p2", "SAMPLE", 5, "src")
        assert base != derive_signal_id("p1", "SAMPLE", 6, "src")
        assert base != derive_signal_id("p1", "SAMPLE", 5, "other")
        assert base != derive_signal_id("p1", "OTHER", 5, "src")

    def test_blank_and_malformed_inputs_are_refused(self):
        with pytest.raises(FusionError):
            derive_signal_id("   ", "SAMPLE", 5, "src")
        with pytest.raises(FusionError):
            derive_signal_id("p1", "lowercase", 5, "src")
        with pytest.raises(FusionError):
            derive_signal_id("p1", "SAMPLE", True, "src")
        with pytest.raises(FusionError):
            derive_signal_id("p1", "SAMPLE", 5, "")


class TestDeriveAssessmentId:
    def test_the_same_content_produces_the_same_id(self):
        ids = ("a", "b", "c")
        assert derive_assessment_id("p1", "{}", ids) == derive_assessment_id(
            "p1", "{}", ids
        )

    def test_signal_id_order_and_duplicates_do_not_matter(self):
        assert derive_assessment_id("p1", "{}", ("b", "a")) == derive_assessment_id(
            "p1", "{}", ("a", "b", "b")
        )

    def test_the_configuration_is_part_of_the_identity(self):
        assert derive_assessment_id("p1", "{}", ("a",)) != derive_assessment_id(
            "p1", '{"w": 2}', ("a",)
        )


class TestCanonicalJson:
    def test_key_order_does_not_change_the_json(self):
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_non_plain_values_serialise_via_str(self):
        assert canonical_json({"family": SignalFamily.ANOMALY}) == '{"family":"ANOMALY"}'


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #


class TestBands:
    H = (Decimal("0.40"), Decimal("0.60"), Decimal("0.80"))

    def test_each_score_lands_on_the_disclosed_band(self):
        assert band_for_score(Decimal("0.00"), self.H) is RiskBand.LOW
        assert band_for_score(Decimal("0.39"), self.H) is RiskBand.LOW
        assert band_for_score(Decimal("0.40"), self.H) is RiskBand.MEDIUM
        assert band_for_score(Decimal("0.59"), self.H) is RiskBand.MEDIUM
        assert band_for_score(Decimal("0.60"), self.H) is RiskBand.HIGH
        assert band_for_score(Decimal("0.80"), self.H) is RiskBand.CRITICAL
        assert band_for_score(Decimal("1.00"), self.H) is RiskBand.CRITICAL

    def test_thresholds_must_be_three_ascending_values_inside_the_unit_interval(self):
        with pytest.raises(ValueError):
            band_for_score(Decimal("0.5"), (Decimal("0.2"), Decimal("0.6")))
        with pytest.raises(ValueError):
            band_for_score(Decimal("0.5"), (Decimal("0.6"), Decimal("0.4"), Decimal("0.2")))
        with pytest.raises(ValueError):
            band_for_score(Decimal("0.5"), (Decimal("0.0"), Decimal("0.5"), Decimal("0.9")))


# --------------------------------------------------------------------------- #
# FusionSignal
# --------------------------------------------------------------------------- #


class TestFusionSignal:
    def test_a_present_signal_carries_its_numbers(self):
        signal = _signal()
        assert signal.status is SignalStatus.PRESENT
        assert signal.normalized_value == 0.5
        assert signal.signal_id == derive_signal_id("prof-subject-1", "TEST_SIGNAL", 8, "source-1")

    def test_present_requires_a_number(self):
        with pytest.raises(ValidationError):
            _signal(raw_value=None, normalized_value=None)

    def test_missing_is_not_zero(self):
        with pytest.raises(ValidationError):
            _signal(status=SignalStatus.ABSENT, normalized_value=0.0)
        with pytest.raises(ValidationError):
            _signal(status=SignalStatus.UNAVAILABLE, raw_value=0.0, normalized_value=0.0)

    def test_a_non_present_signal_is_valid_without_numbers(self):
        signal = _signal(
            status=SignalStatus.ABSENT,
            raw_value=None,
            normalized_value=None,
            explanation="No anomaly result was supplied for this subject, so no number exists.",
        )
        assert signal.normalized_value is None
        assert signal.raw_value is None

    def test_the_id_must_match_the_derived_id(self):
        with pytest.raises(ValidationError):
            FusionSignal(
                signal_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "wrong")),
                signal_type="TEST_SIGNAL",
                family=SignalFamily.ANOMALY,
                status=SignalStatus.PRESENT,
                source_step=8,
                source_id="source-1",
                subject_id="prof-subject-1",
                raw_value=0.5,
                normalized_value=0.5,
                explanation="A real explanation sentence that is long enough.",
            )

    def test_normalized_value_stays_in_unit_interval(self):
        with pytest.raises(ValidationError):
            _signal(normalized_value=1.1)
        with pytest.raises(ValidationError):
            _signal(normalized_value=-0.1)

    def test_non_finite_values_are_refused(self):
        with pytest.raises(ValidationError):
            _signal(normalized_value=float("nan"))
        with pytest.raises(ValidationError):
            _signal(raw_value=float("inf"))

    def test_signal_type_must_be_uppercase_snake(self):
        with pytest.raises((ValidationError, FusionError)):
            _signal(signal_type="camelCase")

    def test_context_must_be_ascending_and_unique_by_key(self):
        with pytest.raises(ValidationError):
            _signal(context=(("z", "1"), ("a", "2")))
        with pytest.raises(ValidationError):
            _signal(context=(("k", "1"), ("k", "2")))

    def test_evidence_refs_and_lineage_must_be_sorted_and_unique(self):
        with pytest.raises(ValidationError):
            _signal(evidence_refs=("b", "a"))
        with pytest.raises(ValidationError):
            _signal(evidence_refs=("a", "a"))

    def test_inference_levels_must_be_ascending(self):
        with pytest.raises(ValidationError):
            _signal(inference_levels=(InferenceLevel.OBSERVED, InferenceLevel.CORRELATED))
        with pytest.raises(ValidationError):
            _signal(inference_levels=(InferenceLevel.OBSERVED, InferenceLevel.OBSERVED))

    def test_a_signal_never_explains_attribution(self):
        assert _signal().explains_attribution() is False


# --------------------------------------------------------------------------- #
# SignalContribution
# --------------------------------------------------------------------------- #


def _contribution(**overrides):
    payload = dict(
        family=SignalFamily.ANOMALY,
        weight=Decimal("0.30"),
        normalized_value=Decimal("0.5"),
        availability=Decimal("1.0"),
        contribution=Decimal("0.15"),
        present_count=1,
        total_count=1,
        signal_ids=(derive_signal_id("p1", "TEST_SIGNAL", 8, "s"),),
        explanation="Anomaly over recorded values contributed 0.30 times 0.50 times 1.00 equals 0.15.",
    )
    payload.update(**overrides)
    return SignalContribution(**payload)


class TestSignalContribution:
    def test_the_contribution_is_exactly_weight_times_value_times_availability(self):
        contribution = _contribution()
        assert contribution.contribution == (
            contribution.weight * contribution.normalized_value * contribution.availability
        )

    def test_a_wrong_contribution_is_refused(self):
        with pytest.raises(ValidationError):
            _contribution(contribution=Decimal("0.99"))

    def test_counts_and_signal_ids_must_agree(self):
        with pytest.raises(ValidationError):
            _contribution(present_count=2, total_count=1)
        with pytest.raises(ValidationError):
            _contribution(present_count=1, signal_ids=())

    def test_decimals_stay_non_negative_and_in_range(self):
        with pytest.raises(ValidationError):
            _contribution(weight=Decimal("-0.1"))
        with pytest.raises(ValidationError):
            _contribution(normalized_value=Decimal("1.01"))
        with pytest.raises(ValidationError):
            _contribution(availability=Decimal("1.01"))

    def test_the_contribution_is_frozen(self):
        contribution = _contribution()
        with pytest.raises(ValidationError):
            contribution.contribution = Decimal("0.0")


# --------------------------------------------------------------------------- #
# MissingSignal + FusionExplanation + ConfidenceContext
# --------------------------------------------------------------------------- #


class TestMissingSignal:
    def test_a_missing_signal_is_never_present(self):
        with pytest.raises(ValidationError):
            MissingSignal(
                signal_id=derive_signal_id("p1", "TEST_SIGNAL", 8, "s"),
                signal_type="TEST_SIGNAL",
                family=SignalFamily.ANOMALY,
                status=SignalStatus.PRESENT,
                source_step=8,
                explanation="A real explanation sentence that is at least ten letters.",
            )
        record = MissingSignal(
            signal_id=derive_signal_id("p1", "TEST_SIGNAL", 8, "s"),
            signal_type="TEST_SIGNAL",
            family=SignalFamily.ANOMALY,
            status=SignalStatus.ABSENT,
            source_step=8,
            explanation="No anomaly result was supplied, so no number was measured.",
        )
        assert record.status is SignalStatus.ABSENT


class TestFusionExplanation:
    def test_evidence_is_sorted_and_text_is_a_real_sentence(self):
        item = FusionExplanation(
            kind=ExplainabilityKind.TOP_CONTRIBUTOR,
            text="The behavioural family contributed the most at 0.32.",
            evidence=("a", "b"),
        )
        assert item.evidence == ("a", "b")
        with pytest.raises(ValidationError):
            FusionExplanation(
                kind=ExplainabilityKind.TOP_CONTRIBUTOR,
                text="The behavioural family contributed the most at 0.32.",
                evidence=("b", "a"),
            )
        with pytest.raises(ValidationError):
            FusionExplanation(kind=ExplainabilityKind.OVERLAP_NOTE, text="short")


class TestConfidenceContext:
    def test_the_empty_context_is_the_default(self):
        context = ConfidenceContext()
        assert context.considered_edge_count == 0
        assert context.min_edge_confidence is None
        assert context.max_edge_confidence is None
        assert context.bases == ()

    def test_a_non_empty_set_needs_both_bounds(self):
        with pytest.raises(ValidationError):
            ConfidenceContext(
                min_edge_confidence=Decimal("1.0"), max_edge_confidence=None, considered_edge_count=1
            )
        with pytest.raises(ValidationError):
            ConfidenceContext(
                min_edge_confidence=None, max_edge_confidence=None, considered_edge_count=1
            )

    def test_bounds_must_be_ordered_and_inside_the_unit_interval(self):
        with pytest.raises(ValidationError):
            ConfidenceContext(
                min_edge_confidence=Decimal("0.9"),
                max_edge_confidence=Decimal("0.8"),
                considered_edge_count=2,
            )
        with pytest.raises(ValidationError):
            ConfidenceContext(
                min_edge_confidence=Decimal("1.1"),
                max_edge_confidence=Decimal("1.1"),
                considered_edge_count=1,
            )

    def test_bases_are_ascending_and_unique(self):
        with pytest.raises(ValidationError):
            ConfidenceContext(
                min_edge_confidence=Decimal("0.5"),
                max_edge_confidence=Decimal("0.5"),
                considered_edge_count=1,
                bases=(ConfidenceBasis.HUMAN_ASSESSMENT, ConfidenceBasis.EXACT_MATCH),
            )

    def test_the_standing_note_never_claims_a_role_in_the_score(self):
        context = ConfidenceContext()
        assert "never multiplied" in context.note


# --------------------------------------------------------------------------- #
# FusionConfig
# --------------------------------------------------------------------------- #


class TestFusionConfig:
    def test_the_default_configuration_is_complete_and_ascending(self):
        config = FusionConfig()
        assert [family.value for family, _ in config.weights] == sorted(
            family.value for family, _ in config.weights
        )
        assert set(family for family, _ in config.weights) == set(SignalFamily)
        weights = dict(config.weights)
        assert weights[SignalFamily.ANOMALY] == Decimal("0.30")
        assert weights[SignalFamily.BEHAVIORAL] == Decimal("0.15")
        assert weights[SignalFamily.CLUSTER_CONTEXT] == Decimal("0.05")
        assert weights[SignalFamily.EVIDENCE_GRAPH] == Decimal("0.20")
        assert weights[SignalFamily.PATH_STRUCTURE] == Decimal("0.10")
        assert weights[SignalFamily.PROVENANCE] == Decimal("0.10")
        assert weights[SignalFamily.TEMPORAL] == Decimal("0.10")

    def test_default_caps_are_the_disclosed_constants(self):
        config = FusionConfig()
        assert (
            config.evidence_count_cap,
            config.path_count_cap,
            config.provenance_file_cap,
        ) == DEFAULT_CAP_EXPECTATIONS

    def test_weights_must_cover_every_family_exactly_once(self):
        with pytest.raises(ValidationError):
            FusionConfig(weights=(("ANOMALY", "1.0"),))
        with pytest.raises(ValidationError):
            FusionConfig(weights=_all_weights() + (("ANOMALY", "0.4"),))

    def test_at_least_one_weight_must_be_non_zero(self):
        with pytest.raises(ValidationError):
            FusionConfig(
                weights=_all_weights(
                    anomaly="0",
                    behavioral="0",
                    cluster="0",
                    evidence="0",
                    path="0",
                    provenance="0",
                    temporal="0",
                )
            )

    def test_bands_must_be_ascending_and_inside_the_unit_interval(self):
        with pytest.raises(ValidationError):
            FusionConfig(band_low="0.6", band_medium="0.5", band_high="0.8")
        with pytest.raises(ValidationError):
            FusionConfig(band_high="1.0")

    def test_caps_are_positive_integers(self):
        with pytest.raises(ValidationError):
            FusionConfig(evidence_count_cap=0)

    def test_the_json_record_is_stable_and_replayable(self):
        config = FusionConfig()
        assert config.to_json() == config.to_json()
        payload = json.loads(config.to_json())
        assert payload["version"] == "FUSION/1"
        assert payload["band"]["low"] == "0.40"
        assert len(payload["weights"]) == 7

    def test_weight_of_maps_every_family(self):
        config = FusionConfig()
        assert config.weight_of[SignalFamily.TEMPORAL] == Decimal("0.10")

    def test_the_config_is_frozen(self):
        config = FusionConfig()
        with pytest.raises(ValidationError):
            config.version = "FUSION/2"


# --------------------------------------------------------------------------- #
# FusionAssessment
# --------------------------------------------------------------------------- #


class TestFusionAssessment:
    def test_an_engine_assessment_revalidates_cleanly(self):
        assessment = _computed_assessment()
        FusionAssessment.model_validate(assessment.model_dump())

    def test_tampered_scores_and_counts_are_refused_on_replay(self):
        assessment = _computed_assessment()
        with pytest.raises(ValidationError):
            assessment.model_copy(update={"fused_score": Decimal("0.99")})
        with pytest.raises(ValidationError):
            assessment.model_copy(update={"present_signal_count": 1})

    def test_tampered_identity_is_refused(self):
        assessment = _computed_assessment()
        with pytest.raises(ValidationError):
            assessment.model_copy(update={"assessment_id": str(uuid.uuid4())})
        with pytest.raises(ValidationError):
            assessment.model_copy(update={"config_version": "FUSION/9"})

    def test_removing_a_signal_breaks_the_overlap_bookkeeping(self):
        assessment = _computed_assessment()
        fewer = assessment.signals[:-1]
        with pytest.raises(ValidationError):
            assessment.model_copy(update={"signals": fewer})

    def test_an_insufficient_assessment_carries_no_score_or_band(self):
        assessment = _insufficient_assessment()
        assert assessment.status is FusionStatus.INSUFFICIENT_SIGNALS
        assert assessment.fused_score is None
        assert assessment.risk_band is None

    def test_adding_a_score_to_an_insufficient_assessment_is_refused(self):
        assessment = _insufficient_assessment()
        with pytest.raises(ValidationError):
            assessment.model_copy(update={"fused_score": Decimal("0.5")})

    def test_each_family_has_a_contribution(self):
        assessment = _computed_assessment()
        assert {item.family for item in assessment.contributions} == set(SignalFamily)

    def test_contributions_family_order_is_ascending(self):
        assessment = _computed_assessment()
        assert [item.family.value for item in assessment.contributions] == sorted(
            item.family.value for item in assessment.contributions
        )

    def test_the_assessment_never_explains_attribution(self):
        assert _computed_assessment().explains_attribution() is False

    def test_to_dict_is_stable_and_json_friendly(self):
        assessment = _computed_assessment()
        payload = assessment.to_dict()
        assert payload == assessment.to_dict()
        assert payload["fused_score"] is None or isinstance(payload["fused_score"], str)
        assert payload["status"] in {"COMPUTED", "INSUFFICIENT_SIGNALS"}
        assert isinstance(payload["config_json"], str)


class TestDecimalDiscipline:
    def test_decimal_context_is_pinned(self):
        assert FUSION_DECIMAL_CONTEXT.prec == 34
        assert FUSION_DECIMAL_CONTEXT.rounding == decimal.ROUND_HALF_EVEN


class TestSignalRoleVocabulary:
    def test_signal_role_has_three_mutually_exclusive_members(self):
        assert {item.value for item in SignalRole} == {
            "PRIORITY_SIGNAL",
            "EVIDENCE_STRENGTH",
            "CONTEXT_ONLY",
        }

    def test_fusion_signal_role_methods(self):
        p = _signal(role=SignalRole.PRIORITY_SIGNAL)
        assert p.is_priority_signal() is True
        assert p.is_evidence_strength() is False
        assert p.is_context_only() is False

        e = _signal(role=SignalRole.EVIDENCE_STRENGTH)
        assert e.is_priority_signal() is False
        assert e.is_evidence_strength() is True
        assert e.is_context_only() is False

        c = _signal(role=SignalRole.CONTEXT_ONLY)
        assert c.is_priority_signal() is False
        assert c.is_evidence_strength() is False
        assert c.is_context_only() is True

    def test_assessment_role_properties_partition_signals(self):
        assessment = _computed_assessment()
        assert len(assessment.priority_signals) == 5
        assert len(assessment.evidence_strength_signals) == 5
        assert len(assessment.context_signals) == 1
        assert len(assessment.signals) == 11
        assert (
            set(assessment.priority_signals)
            | set(assessment.evidence_strength_signals)
            | set(assessment.context_signals)
        ) == set(assessment.signals)