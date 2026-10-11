"""Step 6 -- isolation-forest unusualness over Step 5 profiles.

The score is a *ranking*, not a verdict.  What these tests hold the
implementation to is the three things that make a ranking publishable: the
ranking is reproducible, the direction of the ranking is the documented one,
and the model says so plainly when it has nothing to rank.  Nothing here
asserts that anything is wrong -- a profile at the top of this list is only
unusual relative to the population it was fit on.
"""

import json
import math

import pytest
from pydantic import ValidationError

from backend.app.ml.anomaly import (
    ANOMALY_ALGORITHM_VERSION,
    IsolationForestAnomalyDetector,
    average_unsuccessful_path_length,
    band_for_score,
)
from backend.app.ml.models import (
    AnomalyModelConfig,
    AnomalyResult,
    DeviationDirection,
    ExplanationMethod,
    ModelStatus,
    UnusualnessBand,
)
from tests.test_step6_ml_models import (
    make_distinct_identity_population,
    make_outlier,
    make_population,
)

POPULATION_SIZE = 8
BOUNDARIES = (0.5, 0.7, 0.9)


@pytest.fixture()
def population():
    """A gradient, plus one profile far outside it."""
    return make_population(POPULATION_SIZE) + [make_outlier()]


@pytest.fixture()
def fitted_run(population):
    return IsolationForestAnomalyDetector().fit_score(population)


class TestScoreSemantics:
    def test_a_normalized_score_is_always_a_fraction(self, fitted_run):
        """``2 ** -raw`` over a non-negative raw score, so the range is ``(0, 1]``
        and the direction is fixed: higher means fewer isolating splits, which
        means further from the fitted population."""
        for result in fitted_run[1].results:
            assert 0.0 < result.normalized_score <= 1.0, result
            assert result.raw_score >= 0.0, result

    def test_the_normalization_is_the_documented_formula(self, fitted_run):
        for result in fitted_run[1].results:
            assert result.normalized_score == pytest.approx(
                2.0 ** (-result.raw_score), rel=1e-12
            )

    def test_a_higher_raw_score_lowers_the_normalized_score(self, fitted_run):
        results = fitted_run[1].results
        by_raw = sorted(results, key=lambda item: item.raw_score)
        for lower, higher in zip(by_raw, by_raw[1:]):
            if lower.raw_score == higher.raw_score:
                assert lower.normalized_score == higher.normalized_score
            else:
                assert lower.normalized_score > higher.normalized_score

    def test_a_result_never_claims_a_score_it_does_not_have(self, fitted_run):
        for result in fitted_run[1].results:
            if result.raw_score is None:
                assert result.normalized_score is None
                assert result.band is None
            else:
                assert result.normalized_score is not None
                assert result.band is not None

    def test_the_run_reports_the_boundaries_it_used(self, fitted_run):
        assert fitted_run[1].band_boundaries == BOUNDARIES
        low, moderate, high = BOUNDARIES
        assert 0.0 < low < moderate < high < 1.0

    def test_a_band_is_the_band_its_score_falls_in(self, fitted_run):
        for result in fitted_run[1].results:
            assert result.band is band_for_score(result.normalized_score, BOUNDARIES)

    def test_the_band_boundaries_are_ascending(self):
        for boundaries in ((0.9, 0.7, 0.5), (0.5, 0.5, 0.9), (0.0, 0.5, 0.7)):
            with pytest.raises(ValidationError):
                AnomalyModelConfig(band_boundaries=boundaries)

    @pytest.mark.parametrize(
        "score,band",
        [
            (0.999, UnusualnessBand.EXTREME),
            (0.9, UnusualnessBand.EXTREME),
            (0.7, UnusualnessBand.HIGH),
            (0.5, UnusualnessBand.MODERATE),
            (0.001, UnusualnessBand.LOW),
        ],
    )
    def test_the_bands_are_half_open_at_their_boundaries(self, score, band):
        """Each boundary belongs to the band above it, so a score on a boundary
        reads the same way in a report and in a run."""
        assert band_for_score(score, BOUNDARIES) is band

    @pytest.mark.parametrize("score", [0.0, -0.001, 1.001, float("nan")])
    def test_a_score_outside_the_range_is_refused(self, score):
        with pytest.raises(ValueError):
            band_for_score(score, BOUNDARIES)

    def test_a_band_is_not_a_verdict(self, fitted_run):
        """The vocabulary carries no judgement: a band is a position in a
        ranking, and the enum has no member that names a conclusion."""
        for member in UnusualnessBand:
            assert member.name not in {"FRAUD", "MALICIOUS", "GUILTY", "SUSPICIOUS"}
        assert {member.name for member in UnusualnessBand} == {
            "LOW",
            "MODERATE",
            "HIGH",
            "EXTREME",
        }


class TestRankingDirection:
    def test_the_outlier_ranks_first(self, population):
        """The whole point of the ranking, checked on a profile built to be
        outside: 120 observations and 500x the amounts of the gradient."""
        outlier = make_outlier()
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        worst = max(run.results, key=lambda item: item.normalized_score)
        assert worst.profile_id == outlier.profile_id
        assert worst.observation_ids
        assert worst.band in {UnusualnessBand.HIGH, UnusualnessBand.EXTREME}

    def test_the_typical_profiles_rank_below_it(self, population):
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        outlier_score = max(result.normalized_score for result in run.results)
        others = [
            result
            for result in run.results
            if result.normalized_score < outlier_score
        ]
        assert others, "an outlier that ties the whole population ranks nothing"

    def test_results_are_ordered_by_score_then_id(self, fitted_run):
        """The ranking is the output, so it is what the order expresses:
        most unusual first, ties broken by profile id so the order is total."""
        results = fitted_run[1].results
        keys = [(-result.normalized_score, result.profile_id) for result in results]
        assert keys == sorted(keys)
        assert len({result.profile_id for result in results}) == len(results)

    def test_the_scores_are_not_all_identical_for_a_real_population(self, population):
        """A ranking where everything ties is not a ranking.  Checked so that
        a broken normalizer cannot pass by producing a constant."""
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        assert len({round(result.normalized_score, 12) for result in run.results}) > 1

    def test_the_scores_stay_inside_the_population_rather_than_blowing_up(
        self, population
    ):
        """Isolation forest scores are bounded by construction: a point cannot be
        infinitely far from a finite population."""
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        for result in run.results:
            assert 0.0 < result.normalized_score <= 1.0
            assert math.isfinite(result.normalized_score)


class TestDegeneratePopulations:
    def test_one_profile_is_insufficient_and_says_why(self, population):
        model, run = IsolationForestAnomalyDetector().fit_score(population[:1])
        assert model.status is ModelStatus.INSUFFICIENT_DATA
        assert run.status is ModelStatus.INSUFFICIENT_DATA
        assert run.insufficient_reason
        assert "1" in run.insufficient_reason
        assert run.results == ()

    def test_no_profiles_is_insufficient(self):
        model, run = IsolationForestAnomalyDetector().fit_score([])
        assert model.status is ModelStatus.INSUFFICIENT_DATA
        assert run.insufficient_reason
        assert run.results == ()

    def test_profiles_that_measure_identically_all_tie(self):
        """Different identities, identical measurements.

        This is the case that proves the vector carries no identity: four
        profiles with four different ids, four different networks, four
        different address sets, and not one varying column.
        """
        population = make_distinct_identity_population(4)
        assert len({profile.profile_id for profile in population}) == 4

        model, run = IsolationForestAnomalyDetector().fit_score(population)
        assert model.status is ModelStatus.FITTED
        assert model.varying_feature_count == 0
        assert run.status is ModelStatus.FITTED
        assert not run.insufficient_reason

        scores = {round(result.normalized_score, 12) for result in run.results}
        assert len(scores) == 1, scores
        assert run.results[0].band is UnusualnessBand.LOW

    def test_a_degenerate_run_still_scores_rather_than_failing(self):
        """An honest 'nothing separates these' is a result.  Raising would push
        the decision back onto the caller."""
        population = make_distinct_identity_population(3)
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        assert len(run.results) == 3
        assert all(result.raw_score is not None for result in run.results)

    def test_a_degenerate_explanation_deviates_from_nothing(self):
        """Every deviation is exactly zero, so every direction it reports is
        'equal to the reference'.  The explanation is present -- there is a
        score -- but it names no feature as standing out."""
        population = make_distinct_identity_population(3)
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        for result in run.results:
            assert result.explanation is not None
            for contribution in result.explanation.contributions:
                assert contribution.deviation == 0.0
                assert contribution.absolute_deviation == 0.0

    def test_a_minimum_population_is_configurable(self):
        config = AnomalyModelConfig(min_profiles=6)
        population = make_population(4)
        _, run = IsolationForestAnomalyDetector(config).fit_score(population)
        assert run.status is ModelStatus.INSUFFICIENT_DATA
        assert "4" in run.insufficient_reason

        _, ok = IsolationForestAnomalyDetector(config).fit_score(make_population(6))
        assert ok.status is ModelStatus.FITTED

    def test_insufficient_data_carries_no_fitted_trees(self):
        model, _ = IsolationForestAnomalyDetector().fit_score(make_population(1))
        assert model.trees == ()
        assert model.subsample_size == 0


class TestDeterminism:
    def test_two_runs_over_the_same_population_are_identical(self, population):
        _, first = IsolationForestAnomalyDetector().fit_score(population)
        _, second = IsolationForestAnomalyDetector().fit_score(population)
        assert first.model_dump() == second.model_dump()

    def test_shuffling_the_population_changes_nothing(self, population):
        _, forward = IsolationForestAnomalyDetector().fit_score(population)
        _, backward = IsolationForestAnomalyDetector().fit_score(
            list(reversed(population))
        )
        assert forward.model_dump() == backward.model_dump()

    def test_the_seed_changes_the_forest_and_nothing_else(self, population):
        plain_model, plain = IsolationForestAnomalyDetector(
            AnomalyModelConfig(random_state=0)
        ).fit_score(population)
        other_model, other = IsolationForestAnomalyDetector(
            AnomalyModelConfig(random_state=7)
        ).fit_score(population)
        assert plain_model.model_id != other_model.model_id
        assert plain.metadata.algorithm == other.metadata.algorithm
        assert plain.metadata.algorithm_version == other.metadata.algorithm_version
        assert len(plain.results) == len(other.results)

    def test_the_algorithm_version_is_published(self, fitted_run):
        assert fitted_run[0].algorithm_version == ANOMALY_ALGORITHM_VERSION
        assert fitted_run[1].metadata.algorithm_version == ANOMALY_ALGORITHM_VERSION

    def test_the_model_id_depends_on_the_seed_and_the_schema(self, population):
        first = IsolationForestAnomalyDetector().fit(population)
        second = IsolationForestAnomalyDetector(
            AnomalyModelConfig(random_state=3)
        ).fit(population)
        assert first.model_id != second.model_id
        assert first.model_id == IsolationForestAnomalyDetector().fit(population).model_id


class TestFitAndScoreSeparation:
    def test_a_fitted_model_scores_a_later_population(self, fitted_run, population):
        """Fit and score are separate operations so a model fitted once can be
        reused -- which is the only way a stored model is worth storing."""
        model, _ = fitted_run
        later = make_population(4, amount=20)
        run = model.score(later)
        assert run.status is ModelStatus.FITTED
        assert run.metadata.scoring_population_size == len(later)
        assert run.metadata.fitting_population_size == len(population)

    def test_out_of_sample_scoring_is_labelled_as_such(self, fitted_run, population):
        model, in_sample = fitted_run
        assert in_sample.metadata.scoring_in_sample is True
        assert all(result.is_in_sample for result in in_sample.results)

        out = model.score_new(make_population(4, amount=20))
        assert out.metadata.scoring_in_sample is False
        assert all(result.is_in_sample is False for result in out.results)

    def test_scoring_an_unfitted_model_produces_nothing_and_says_why(self):
        """A model that never fitted has no trees to isolate anything with.
        Raising would force the caller to branch; reporting the status keeps the
        decision where the data is."""
        model = IsolationForestAnomalyDetector().fit(make_population(1))
        assert model.status is ModelStatus.INSUFFICIENT_DATA

        run = model.score(make_population(3))
        assert run.status is ModelStatus.INSUFFICIENT_DATA
        assert run.insufficient_reason
        assert run.results == ()

    def test_scoring_nothing_with_a_fitted_model_says_so(self, fitted_run):
        model, _ = fitted_run
        run = model.score([])
        assert run.status is ModelStatus.INSUFFICIENT_DATA
        assert run.results == ()
        assert "no profiles" in run.insufficient_reason

    def test_a_raw_score_can_be_taken_for_one_vector(self, fitted_run, population):
        """The public path for scoring something that is not a profile."""
        model, _ = fitted_run
        vector = model.preprocessor.transform_one(population[0])
        assert model.raw_score_of(vector.values) == pytest.approx(
            model.raw_score_of(vector.values)
        )

    def test_the_fitted_model_remembers_its_schema(self, fitted_run, population):
        model, run = fitted_run
        assert model.schema_id == run.metadata.feature_schema_id
        assert model.feature_names == run.feature_names

    def test_a_fitted_model_survives_a_json_round_trip(self, fitted_run, population):
        from backend.app.ml.anomaly import FittedAnomalyModel

        model, _ = fitted_run
        reloaded = FittedAnomalyModel.model_validate_json(model.model_dump_json())
        assert reloaded.model_id == model.model_id
        assert reloaded.model_dump() == model.model_dump()
        assert reloaded.score(population).model_dump() == model.score(population).model_dump()


class TestConfiguration:
    def test_the_defaults_are_published_and_frozen(self, population):
        detector = IsolationForestAnomalyDetector()
        assert detector.config.algorithm.value == "ISOLATION_FOREST"
        assert detector.config.random_state == 0
        assert detector.config.n_estimators == 100
        with pytest.raises(ValidationError):
            detector.config.random_state = 5

    def test_the_configuration_refuses_impossible_settings(self):
        for kwargs in (
            {"n_estimators": 0},
            {"max_samples": 0},
            {"min_profiles": 1},
            {"max_depth": 0},
            {"explanation_limit": 101},
        ):
            with pytest.raises(ValidationError):
                AnomalyModelConfig(**kwargs)

    def test_an_explanation_limit_of_zero_is_allowed_and_means_no_explanation(
        self, population
    ):
        """Zero is a real choice -- a caller that stores results and explains
        them later may not want the payload -- so it is not refused, but it is
        honoured."""
        _, run = IsolationForestAnomalyDetector(
            AnomalyModelConfig(explanation_limit=0)
        ).fit_score(population)
        for result in run.results:
            assert result.explanation is not None
            assert result.explanation.contributions == ()

    def test_the_population_bounds_the_subsample(self, population):
        """A forest cannot sample more rows than exist, and it must not pretend
        to."""
        model = IsolationForestAnomalyDetector(
            AnomalyModelConfig(max_samples=1000)
        ).fit(population)
        assert model.subsample_size <= len(population)

    def test_a_depth_bounded_forest_is_built(self, population):
        model = IsolationForestAnomalyDetector(
            AnomalyModelConfig(max_depth=1)
        ).fit(population)
        assert model.status is ModelStatus.FITTED
        assert model.trees

    def test_the_tree_count_is_the_configured_count(self, population):
        for count in (1, 5, 37):
            model = IsolationForestAnomalyDetector(
                AnomalyModelConfig(n_estimators=count)
            ).fit(population)
            assert len(model.trees) == count

    def test_fewer_trees_still_produce_a_ranking(self, population):
        _, run = IsolationForestAnomalyDetector(
            AnomalyModelConfig(n_estimators=1)
        ).fit_score(population)
        assert run.status is ModelStatus.FITTED
        assert all(result.normalized_score is not None for result in run.results)

    def test_the_unexpected_path_length_follows_the_published_definition(self):
        """The standard average unsuccessful path length of size *n*.

        Published so a reader can check the normalization instead of trusting
        it: ``2 * (ln(n-1) + gamma) - 2 * (n-1) / n`` for ``n > 2``, and the
        exact values 0 and 1 for the two cases the closed form does not cover.
        """
        assert average_unsuccessful_path_length(0) == 0.0
        assert average_unsuccessful_path_length(1) == 0.0
        assert average_unsuccessful_path_length(2) == 1.0
        for size in (3, 8, 256):
            expected = 2.0 * (math.log(size - 1) + 0.5772156649015329) - (
                2.0 * (size - 1)
            ) / size
            assert average_unsuccessful_path_length(size) == pytest.approx(
                expected, rel=1e-9
            )

    def test_the_path_length_grows_with_the_subsample(self):
        """A larger subsample is a deeper search, so a failed search takes
        longer.  This is the property the normalizer depends on."""
        lengths = [average_unsuccessful_path_length(size) for size in (4, 16, 64, 256)]
        assert lengths == sorted(lengths)
        assert lengths[0] < lengths[-1]

    def test_the_normalizer_is_the_path_length_of_the_actual_subsample(
        self, population
    ):
        """The normalizer must match the subsample the trees were built on.  A
        mismatch would shift every score in the run."""
        model = IsolationForestAnomalyDetector().fit(population)
        assert model.normalizer == pytest.approx(
            average_unsuccessful_path_length(model.subsample_size)
        )


class TestMetadata:
    def test_the_run_metadata_names_the_algorithm_and_its_origin(self, fitted_run):
        metadata = fitted_run[1].metadata
        assert metadata.algorithm == "ISOLATION_FOREST"
        assert metadata.algorithm_version == ANOMALY_ALGORITHM_VERSION
        assert metadata.library == "python-stdlib"
        assert metadata.random_state == 0
        assert metadata.fitting_population_size == len(fitted_run[1].results)
        assert metadata.scoring_population_size == len(fitted_run[1].results)

    def test_the_metadata_carries_the_whole_configuration(self, fitted_run):
        """Stored as canonical JSON text, so a stored run reproduces its own
        settings without a second lookup table."""
        config = json.loads(fitted_run[1].metadata.config_json)
        assert config["n_estimators"] == 100
        assert config["max_samples"] == 256
        assert config["min_profiles"] == 2
        assert config["band_boundaries"] == [0.5, 0.7, 0.9]
        assert config["random_state"] == 0

    def test_the_metadata_names_the_schema_it_was_fit_on(self, fitted_run):
        metadata = fitted_run[1].metadata
        assert metadata.feature_schema_id == fitted_run[0].schema_id
        assert tuple(metadata.feature_names) == fitted_run[1].feature_names

    def test_the_metadata_states_how_the_columns_were_prepared(self, fitted_run):
        metadata = fitted_run[1].metadata
        assert metadata.scaling.value == "ROBUST_MAD"
        assert metadata.imputation.value == "FITTED_MEDIAN"

    def test_the_model_id_is_reproducible_from_its_contents(self, population):
        model = IsolationForestAnomalyDetector().fit(population)
        assert model.model_id.startswith("ANOMALY_ISOLATION_FOREST_")
        assert model.model_id == IsolationForestAnomalyDetector().fit(
            population
        ).model_id

    def test_the_model_id_appears_in_the_run_metadata(self, fitted_run):
        model, run = fitted_run
        assert run.metadata.model_id == model.model_id


class TestResultShape:
    def test_a_result_reports_the_schema_and_the_observations(self, fitted_run):
        model, run = fitted_run
        for result in run.results:
            assert isinstance(result, AnomalyResult)
            assert result.schema_id == model.schema_id
            assert result.observation_ids
            assert result.observation_ids == tuple(sorted(result.observation_ids))

    def test_the_observations_come_from_the_step5_profile(self, population):
        by_id = {profile.profile_id: profile for profile in population}
        _, run = IsolationForestAnomalyDetector().fit_score(population)
        for result in run.results:
            assert tuple(result.observation_ids) == tuple(
                by_id[result.profile_id].observation_ids
            )

    def test_a_result_refuses_to_pair_a_score_with_a_missing_band(self):
        with pytest.raises(ValidationError):
            AnomalyResult(
                profile_id="00000000-0000-5000-8000-000000000000",
                schema_id="SCHEMA_X",
                raw_score=0.3,
                normalized_score=0.8,
                band=None,
                is_in_sample=True,
            )

    def test_a_result_refuses_a_score_outside_the_range(self):
        for normalized in (0.0, -0.5, 1.5):
            with pytest.raises(ValidationError):
                AnomalyResult(
                    profile_id="00000000-0000-5000-8000-000000000000",
                    schema_id="SCHEMA_X",
                    raw_score=0.3,
                    normalized_score=normalized,
                    band=UnusualnessBand.LOW,
                    is_in_sample=True,
                )

    def test_a_fitted_run_without_results_is_refused(self):
        """A run that reports FITTED and nothing else would be a claim with no
        evidence behind it."""
        from tests.test_step6_ml_models import make_metadata

        with pytest.raises(ValidationError):
            from backend.app.ml.models import AnomalyRunResult

            AnomalyRunResult(
                metadata=make_metadata(),
                status=ModelStatus.FITTED,
                feature_names=("observation_count",),
            )
