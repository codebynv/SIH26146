"""Step 8 -- determinism and completeness of the explanation set (SIH26146 - Step 8).

The explanation is not prose a writer composes after the fact; it is generated
from the very objects recorded on the assessment, so a sentence cannot drift
from the number it describes.  These tests pin the fixed set of items, their
ordering, the standing disclosure notes, and the rule that explains the largest
contributor rather than bestowing a verdict.
"""

from __future__ import annotations

from collections import Counter

from backend.app.fusion import FusionEngine, ExplainabilityKind
from backend.app.fusion.explain import build_explanation

from tests.test_step8_fusion import _assessment

NOTE_KINDS = {
    ExplainabilityKind.CLUSTER_CONTEXT_NOTE,
    ExplainabilityKind.INFERENCE_LEVEL_NOTE,
    ExplainabilityKind.OVERLAP_NOTE,
    ExplainabilityKind.INTERPRETATION_NOTE,
}


class TestFullFixtureExplanation:
    def test_the_full_fixture_produces_exactly_twelve_items(self):
        assessment = _assessment()
        kinds = Counter(item.kind for item in assessment.explanation)
        assert kinds == Counter(
            {
                ExplainabilityKind.TOP_CONTRIBUTOR: 1,
                ExplainabilityKind.SIGNAL_CONTRIBUTION: 7,
                ExplainabilityKind.CLUSTER_CONTEXT_NOTE: 1,
                ExplainabilityKind.INFERENCE_LEVEL_NOTE: 1,
                ExplainabilityKind.OVERLAP_NOTE: 1,
                ExplainabilityKind.INTERPRETATION_NOTE: 1,
            }
        )

    def test_items_are_ordered_by_kind_then_text(self):
        assessment = _assessment()
        items = assessment.explanation
        assert [item.kind.value for item in items] == sorted(
            item.kind.value for item in items
        )
        for a, b in zip(items, items[1:]):
            assert (a.kind.value, a.text) <= (b.kind.value, b.text)

    def test_the_top_contributor_names_the_largest_term(self):
        assessment = _assessment()
        top = next(
            item
            for item in assessment.explanation
            if item.kind is ExplainabilityKind.TOP_CONTRIBUTOR
        )
        best = max(assessment.contributions, key=lambda item: item.contribution)
        assert best.family.value in top.text
        assert top.evidence == best.signal_ids

    def test_every_family_gets_its_own_contribution_sentence(self):
        assessment = _assessment()
        sentences = [
            item
            for item in assessment.explanation
            if item.kind is ExplainabilityKind.SIGNAL_CONTRIBUTION
        ]
        assert {item.evidence for item in sentences} == {
            contribution.signal_ids for contribution in assessment.contributions
        }

    def test_the_standing_notes_carry_the_exact_disclosures(self):
        assessment = _assessment()
        by_kind = {item.kind: item.text for item in assessment.explanation}
        assert "never upgraded" in by_kind[ExplainabilityKind.INFERENCE_LEVEL_NOTE]
        assert "descriptive context only" in by_kind[ExplainabilityKind.CLUSTER_CONTEXT_NOTE]
        assert "not assumed independent" in by_kind[ExplainabilityKind.OVERLAP_NOTE]
        assert "not a probability" in by_kind[ExplainabilityKind.INTERPRETATION_NOTE]

    def test_the_interpretation_note_is_the_configurations_own_statement(self):
        assessment = _assessment()
        interpretation = next(
            item
            for item in assessment.explanation
            if item.kind is ExplainabilityKind.INTERPRETATION_NOTE
        )
        assert interpretation.text == FusionEngine().config.interpretation


class TestMissingExplanation:
    def test_every_missing_signal_is_explained(self):
        assessment = _assessment(producers=())
        missing_items = [
            item
            for item in assessment.explanation
            if item.kind is ExplainabilityKind.MISSING_SIGNAL
        ]
        assert len(missing_items) == assessment.signal_count - assessment.present_signal_count
        reported = {
            item.text.split(" was ", 1)[0].replace("Signal ", "") for item in missing_items
        }
        missing_types = {item.signal_type for item in assessment.missing_signals}
        assert reported == missing_types

    def test_a_missing_signal_item_explains_the_refusal(self):
        assessment = _assessment(producers=())
        item = next(
            item
            for item in assessment.explanation
            if item.kind is ExplainabilityKind.MISSING_SIGNAL
        )
        assert "was ABSENT" in item.text
        assert "never replaced by a zero" in item.text
        assert item.evidence == ()

    def test_notes_are_always_emitted_even_when_few_signals_are_present(self):
        assessment = _assessment(producers=())
        kinds = {item.kind for item in assessment.explanation}
        assert NOTE_KINDS <= kinds


class TestEngineAndBuilderAgree:
    def test_the_engine_records_exactly_what_the_builder_would_write(self):
        assessment = _assessment()
        rebuilt = build_explanation(
            contributions=assessment.contributions,
            signals=assessment.signals,
            config=FusionEngine().config,
        )
        assert rebuilt == assessment.explanation
        assert [item.to_dict() for item in rebuilt] == [item.to_dict() for item in assessment.explanation]

    def test_the_builder_is_deterministic(self):
        assessment = _assessment()
        rebuilt = build_explanation(
            contributions=assessment.contributions,
            signals=assessment.signals,
            config=FusionEngine().config,
        )
        again = build_explanation(
            contributions=assessment.contributions,
            signals=assessment.signals,
            config=FusionEngine().config,
        )
        assert rebuilt == again


class TestPackageSurface:
    def test_explainability_kind_is_exported(self):
        import backend.app.fusion as fusion

        assert "ExplainabilityKind" in fusion.__all__

    def test_the_kind_enum_is_stable(self):
        assert [item.value for item in ExplainabilityKind] == [
            "TOP_CONTRIBUTOR",
            "SIGNAL_CONTRIBUTION",
            "MISSING_SIGNAL",
            "CLUSTER_CONTEXT_NOTE",
            "INFERENCE_LEVEL_NOTE",
            "OVERLAP_NOTE",
            "INTERPRETATION_NOTE",
        ]