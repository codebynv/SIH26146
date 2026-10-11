"""Step 6 -- explaining an unusualness number without overclaiming.

An isolation forest computes its score by averaging path lengths through random
splits.  There is no arithmetic relationship between that number and any single
column, so apportioning the score across columns would be invented arithmetic.
These tests hold the implementation to describing a *position* instead: how far
a column sits from the population it was fitted against, in that population's
own units, and nothing more.
"""

import math

import pytest
from pydantic import ValidationError

from backend.app.ml.anomaly import IsolationForestAnomalyDetector
from backend.app.ml.explanations import (
    EXPLANATION_METHOD_DETAIL,
    explain_profile,
    explain_profile_vector,
)
from backend.app.ml.models import (
    AnomalyExplanation,
    AnomalyFeatureContribution,
    DeviationDirection,
    ExplanationMethod,
    describe_explanation_methods,
)
from backend.app.ml.preprocessing import ProfileFeatureVectorizer
from tests.test_step6_ml_models import (
    make_distinct_identity_population,
    make_outlier,
    make_population,
)

POPULATION_SIZE = 8


@pytest.fixture()
def population():
    return make_population(POPULATION_SIZE) + [make_outlier()]


@pytest.fixture()
def preprocessor(population):
    return ProfileFeatureVectorizer().fit(population)


class TestWhatAnExplanationClaims:
    def test_every_explanation_names_its_method(self, preprocessor, population):
        for profile in population:
            explanation = explain_profile(preprocessor, profile)
            assert explanation.method is ExplanationMethod.STATISTICAL_REFERENCE

    def test_the_method_detail_denies_the_claims_it_cannot_support(
        self, preprocessor, population
    ):
        """The prose is part of the output, not a comment on it.  A reader who
        skips the numbers must still be unable to mistake this for an
        attribution."""
        explanation = explain_profile(preprocessor, population[0])
        detail = explanation.method_detail.lower()
        assert "does not decompose" in detail
        assert "contributed" in detail
        assert "causal" in detail
        assert detail == EXPLANATION_METHOD_DETAIL.lower()

    def test_the_method_detail_is_a_constant_not_a_formatted_sentence(
        self, preprocessor, population
    ):
        """Two profiles, one sentence.  A reader comparing runs should not be
        asked to diff prose."""
        first = explain_profile(preprocessor, population[0])
        second = explain_profile(preprocessor, population[1])
        assert first.method_detail == second.method_detail == EXPLANATION_METHOD_DETAIL

    def test_no_contribution_is_expressed_as_a_share_of_the_score(
        self, preprocessor, population
    ):
        """A percentage of an unexplained total is the shape of a claim this
        step refuses to make, so the magnitudes are signed deviations in fitted
        spreads -- they may be negative, and they carry no unit of share."""
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        signed = [
            contribution.deviation
            for result in run.results
            for contribution in result.explanation.contributions
        ]
        assert signed
        assert any(value < 0.0 for value in signed), (
            "a signed deviation set with no negatives is a rank, not a deviation"
        )
        for result in run.results:
            for contribution in result.explanation.contributions:
                assert contribution.absolute_deviation == pytest.approx(
                    abs(contribution.deviation)
                )

    def test_the_deviations_do_not_sum_to_the_score(self, preprocessor, population):
        """Checked deliberately.  If they summed to anything meaningful the
        module would be making an attribution claim, and this would fail."""
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        for result in run.results:
            total = sum(
                contribution.deviation
                for contribution in result.explanation.contributions
            )
            assert total != pytest.approx(result.normalized_score, abs=1e-6)

    def test_the_unavailable_methods_are_documented_as_unavailable(self):
        """Auditable from code rather than from a claim in a report."""
        described = describe_explanation_methods()
        assert set(described) == {member.value for member in ExplanationMethod}
        assert described[ExplanationMethod.STATISTICAL_REFERENCE.value].startswith(
            "AVAILABLE"
        )
        for unavailable in (
            ExplanationMethod.MODEL_NATIVE,
            ExplanationMethod.DERIVED_APPROXIMATION,
        ):
            assert described[unavailable.value].startswith("UNAVAILABLE")

    def test_no_module_produces_an_unavailable_method(self, preprocessor, population):
        available = {
            key
            for key, text in describe_explanation_methods().items()
            if text.startswith("AVAILABLE")
        }
        for profile in population:
            method = explain_profile(preprocessor, profile).method
            assert method.value in available


class TestDeviations:
    def test_a_deviation_is_the_distance_from_the_fitted_median(self, preprocessor):
        population = make_population(6)
        profile = population[3]
        explanation = explain_profile(preprocessor, profile)
        raw = dict(preprocessor.raw_values_for(profile))
        for contribution in explanation.contributions:
            expected = (
                raw[contribution.feature_name]
                - preprocessor.center_of(contribution.feature_name)
            ) / preprocessor.scale_of(contribution.feature_name)
            assert contribution.deviation == pytest.approx(expected)
            assert contribution.reference_value == pytest.approx(
                preprocessor.center_of(contribution.feature_name)
            )
            assert contribution.observed_value == pytest.approx(
                raw[contribution.feature_name]
            )

    def test_the_direction_follows_the_sign_of_the_deviation(self, preprocessor):
        population = make_population(6)
        for profile in population:
            for contribution in explain_profile(preprocessor, profile).contributions:
                if contribution.deviation > 0:
                    assert contribution.direction is DeviationDirection.ABOVE_REFERENCE
                else:
                    assert contribution.direction is DeviationDirection.BELOW_REFERENCE

    def test_contributions_are_ranked_by_absolute_deviation(self, preprocessor):
        population = make_population(9, amount=4)
        for profile in population:
            contributions = explain_profile(preprocessor, profile).contributions
            absolute = [c.absolute_deviation for c in contributions]
            assert absolute == sorted(absolute, reverse=True)
            assert [c.rank for c in contributions] == list(
                range(1, len(contributions) + 1)
            )

    def test_every_deviation_is_finite(self, preprocessor, population):
        for profile in population:
            for contribution in explain_profile(preprocessor, profile).contributions:
                assert math.isfinite(contribution.deviation)
                assert math.isfinite(contribution.observed_value)
                assert math.isfinite(contribution.reference_value)

    def test_the_explanation_uses_the_measured_units(self, preprocessor, population):
        """Seconds are seconds.  Without the unit a reader cannot tell a
        duration from a count."""
        units = {
            contribution.feature_name: contribution.reference_unit
            for profile in population
            for contribution in explain_profile(preprocessor, profile).contributions
        }
        assert any(unit for unit in units.values())
        for name, unit in units.items():
            assert unit == preprocessor.unit_of(name)

    def test_the_explanation_names_the_profile_it_describes(self, preprocessor, population):
        for profile in population:
            assert (
                explain_profile(preprocessor, profile).profile_id == profile.profile_id
            )

    def test_a_profile_at_the_population_median_deviates_by_nothing(self, preprocessor):
        population = make_distinct_identity_population(3)
        fitted = ProfileFeatureVectorizer().fit(population)
        explanation = explain_profile(fitted, population[0])
        for contribution in explanation.contributions:
            assert contribution.deviation == pytest.approx(0.0)

    def test_a_large_outlier_is_described_in_its_own_units(self, preprocessor):
        """The biggest deviations of the outlier are amounts, and they are
        reported in the currency Step 5 recorded."""
        explanation = explain_profile(preprocessor, make_outlier())
        assert explanation.contributions
        assert all(c.absolute_deviation > 0.0 for c in explanation.contributions)
        assert any(c.reference_unit for c in explanation.contributions)


class TestLimit:
    def test_the_limit_is_honoured(self, preprocessor, population):
        for limit in (1, 3, 5, 10):
            explanation = explain_profile(preprocessor, population[0], limit=limit)
            assert len(explanation.contributions) == min(
                limit, len(preprocessor.feature_names)
            )

    def test_truncation_is_reported(self, preprocessor, population):
        narrow = explain_profile(preprocessor, population[0], limit=1)
        wide = explain_profile(preprocessor, population[0], limit=len(preprocessor.feature_names))
        assert narrow.truncated is True
        assert wide.truncated is False

    def test_a_limit_below_the_column_count_truncates(self, preprocessor, population):
        explanation = explain_profile(preprocessor, population[0], limit=2)
        assert len(explanation.contributions) == 2
        assert explanation.truncated is True

    def test_a_limit_of_zero_reports_nothing_and_says_it_truncated(
        self, preprocessor, population
    ):
        explanation = explain_profile(preprocessor, population[0], limit=0)
        assert explanation.contributions == ()
        assert explanation.truncated is True

    def test_a_limit_wider_than_the_schema_does_not_pad(self, preprocessor, population):
        explanation = explain_profile(preprocessor, population[0], limit=10_000)
        assert len(explanation.contributions) == len(preprocessor.feature_names)
        assert explanation.truncated is False

    def test_a_negative_limit_is_refused(self, preprocessor, population):
        with pytest.raises(ValueError):
            explain_profile(preprocessor, population[0], limit=-1)

    def test_the_widest_explanation_starts_where_the_narrow_one_ends(self, preprocessor):
        population = make_population(9, amount=4)
        narrow = explain_profile(preprocessor, population[0], limit=3)
        wide = explain_profile(preprocessor, population[0], limit=50)
        assert [c.feature_name for c in narrow.contributions] == [
            c.feature_name for c in wide.contributions[:3]
        ]


class TestVectorEntryPoint:
    def test_the_vector_path_gives_the_same_answer_as_the_profile_path(
        self, preprocessor, population
    ):
        """Two entry points, one ranking.  If they disagreed, a stored vector
        and a stored profile would explain the same profile differently."""
        profile = population[2]
        assert explain_profile_vector(
            preprocessor,
            profile.profile_id,
            preprocessor.raw_values_for(profile),
        ).model_dump() == explain_profile(preprocessor, profile).model_dump()

    def test_the_vector_path_needs_every_schema_column(self, preprocessor, population):
        profile = population[0]
        raw = preprocessor.raw_values_for(profile)[:-1]
        with pytest.raises(ValueError) as caught:
            explain_profile_vector(preprocessor, profile.profile_id, raw)
        assert "missing schema columns" in str(caught.value)

    def test_the_vector_path_refuses_anything_but_a_fitted_preprocessor(self, population):
        """The reference values are read off a fitted artefact, so an object
        that is not one has no reference to compare against."""
        profile = population[0]
        raw = dict(ProfileFeatureVectorizer().fit(population).raw_values_for(profile))
        with pytest.raises(TypeError):
            explain_profile_vector(None, profile.profile_id, tuple(raw.items()))
        with pytest.raises(TypeError):
            explain_profile_vector(object(), profile.profile_id, tuple(raw.items()))


class TestDeterminism:
    def test_two_explanations_of_one_profile_agree(self, preprocessor, population):
        first = explain_profile(preprocessor, population[0])
        second = explain_profile(preprocessor, population[0])
        assert first.model_dump() == second.model_dump()

    def test_ties_are_broken_by_feature_name(self, preprocessor):
        """A population of identical vectors has every deviation tied, so the
        ranking falls back to the column name -- a total order either way."""
        population = make_distinct_identity_population(3)
        fitted = ProfileFeatureVectorizer().fit(population)
        explanation = explain_profile(fitted, population[0], limit=6)
        names = [c.feature_name for c in explanation.contributions]
        assert names == sorted(names)

    def test_the_explanation_survives_a_json_round_trip(self, preprocessor, population):
        explanation = explain_profile(preprocessor, population[0])
        reloaded = AnomalyExplanation.model_validate_json(explanation.model_dump_json())
        assert reloaded.model_dump() == explanation.model_dump()

    def test_the_fitted_preprocessor_can_be_stored_with_the_explanation(
        self, population
    ):
        """An explanation is only useful if the reference it was computed
        against can be kept alongside it."""
        from backend.app.ml.preprocessing import FittedFeaturePreprocessor

        fitted = ProfileFeatureVectorizer().fit(population)
        reloaded = FittedFeaturePreprocessor.model_validate_json(
            fitted.model_dump_json()
        )
        assert explain_profile(reloaded, population[0]).model_dump() == explain_profile(
            fitted, population[0]
        ).model_dump()


class TestContributionShape:
    def test_a_contribution_needs_a_feature_and_a_rank(self):
        with pytest.raises(ValidationError):
            AnomalyFeatureContribution(
                feature_name="observation_count",
                observed_value=1.0,
                reference_value=0.0,
                deviation=1.0,
                absolute_deviation=1.0,
                direction=DeviationDirection.ABOVE_REFERENCE,
            )

    def test_a_contribution_refuses_an_inconsistent_absolute_deviation(self):
        with pytest.raises(ValidationError):
            AnomalyFeatureContribution(
                rank=1,
                feature_name="observation_count",
                observed_value=1.0,
                reference_value=0.0,
                deviation=1.0,
                absolute_deviation=0.5,
                direction=DeviationDirection.ABOVE_REFERENCE,
            )

    def test_a_contribution_refuses_a_direction_that_contradicts_the_sign(self):
        with pytest.raises(ValidationError):
            AnomalyFeatureContribution(
                rank=1,
                feature_name="observation_count",
                observed_value=1.0,
                reference_value=0.0,
                deviation=1.0,
                absolute_deviation=1.0,
                direction=DeviationDirection.BELOW_REFERENCE,
            )

    def test_a_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            AnomalyFeatureContribution(
                rank=0,
                feature_name="observation_count",
                observed_value=1.0,
                reference_value=0.0,
                deviation=1.0,
                absolute_deviation=1.0,
                direction=DeviationDirection.ABOVE_REFERENCE,
            )

    def test_an_explanation_with_a_method_must_carry_its_detail(self):
        with pytest.raises(ValidationError):
            AnomalyExplanation(
                profile_id="00000000-0000-5000-8000-000000000000",
                method=ExplanationMethod.STATISTICAL_REFERENCE,
                method_detail="",
                contributions=(),
                truncated=False,
            )


class TestIntegrationWithTheAnomalyRun:
    def test_the_explanation_is_attached_to_the_result_it_explains(
        self, population
    ):
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        by_id = {profile.profile_id: profile for profile in population}
        for result in run.results:
            assert result.explanation is not None
            assert result.explanation.profile_id == result.profile_id
            assert result.explanation.method is ExplanationMethod.STATISTICAL_REFERENCE

    def test_the_top_ranked_result_explains_its_own_score(self, population):
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        top = run.results[0]
        assert top.band.value in {"HIGH", "EXTREME"}
        assert top.explanation.contributions
        assert all(c.absolute_deviation > 0.0 for c in top.explanation.contributions)

    def test_the_explanation_limit_follows_the_configuration(self, population):
        from backend.app.ml.models import AnomalyModelConfig

        _, run = IsolationForestAnomalyDetector(
            AnomalyModelConfig(explanation_limit=2)
        ).fit_score(population)
        for result in run.results:
            assert len(result.explanation.contributions) <= 2

    def test_an_out_of_sample_result_still_explains_itself(self, population):
        model = IsolationForestAnomalyDetector().fit(population)
        out = model.score_new(make_population(4, amount=25, tag="later"))
        for result in out.results:
            assert result.explanation is not None
            assert result.explanation.contributions
            assert result.is_in_sample is False
