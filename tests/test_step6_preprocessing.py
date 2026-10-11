"""Step 6 -- turning Step 5 profiles into a matrix.

The adapter is where the locked Step 5 contract meets this step's own rules, so
these tests are mostly about *refusals*: what is deliberately left out, what is
imputed, and what happens when there is not enough to impute from.  The
guarantee underneath all of it is that a measurement is a measurement -- no
identifier, no label and no judgement reaches a column.
"""

import math

import pytest
from pydantic import ValidationError

from backend.app.ml.models import (
    ExcludedFeature,
    ExclusionReason,
    FeatureSchema,
    ImputationStrategy,
    MLFeatureVector,
    ScalingMethod,
    identifier_like_feature_name,
    is_numeric_value_kind,
)
from backend.app.ml.preprocessing import (
    MLPreprocessorError,
    PreprocessorConfig,
    ProfileFeatureVectorizer,
    derive_schema_id,
    median,
    median_absolute_deviation,
)
from backend.app.behavior.profiles import BehaviorProfileBuilder
from tests.test_step6_ml_models import (
    make_distinct_identity_population,
    make_observation,
    make_population,
)

POPULATION_SIZE = 8


@pytest.fixture()
def population():
    return make_population(POPULATION_SIZE)


@pytest.fixture()
def schema(population):
    return ProfileFeatureVectorizer().select_schema(population)


@pytest.fixture()
def fitted(population):
    return ProfileFeatureVectorizer().fit(population)


def _network_only_profile():
    """One profile with no blockchain and no evidence: every amount, graph and
    evidence measurement is unavailable for it, which is the only honest way to
    exercise imputation against the real Step 5 extractor."""
    from tests.test_step6_ml_models import ObservationType

    builder = BehaviorProfileBuilder()
    records = [
        make_observation(
            "net-%02d" % step,
            hour=step,
            txid=None,
            amount=None,
            observation_type=ObservationType.NETWORK,
        )
        for step in range(3)
    ]
    return builder.build(records, group_id="network-only")


def _rename_feature(profile, name):
    """Return *profile* with the *source* feature renamed to *name*.

    Used to publish an identifier-shaped measurement that the locked Step 5
    extractor does not currently emit, so the refusal path is exercised rather
    than merely asserted.  ``unique_src_ip_count`` is renamed rather than a new
    feature appended because Step 5 requires features to stay sorted by
    ``(category, name)`` and unique, and this keeps the forged profile a valid
    Step 5 object.
    """
    from backend.app.behavior.profiles import BehavioralProfile

    renamed = [
        feature.model_copy(update={"name": name, "feature_id": name})
        if feature.name == "unique_src_ip_count"
        else feature
        for feature in profile.features
    ]
    # Step 5 requires (category, name) order; renaming moves the name, so the
    # sequence is re-sorted rather than patched in place.
    renamed.sort(key=lambda feature: (feature.category.value, feature.name))
    return BehavioralProfile(
        profile_id=profile.profile_id,
        grouping=profile.grouping,
        group_id=profile.group_id,
        observation_ids=profile.observation_ids,
        features=tuple(renamed),
        evidence_relationship_ids=profile.evidence_relationship_ids,
        provenance_ingestion_ids=profile.provenance_ingestion_ids,
        provenance_source_files=profile.provenance_source_files,
    )


class TestColumnSelection:
    def test_only_numeric_measurements_become_columns(self, schema, population):
        """A count and a decimal are measurements; everything else is not."""
        names = schema.feature_names
        assert names
        assert list(names) == sorted(names), "column order must be deterministic"
        assert len(set(names)) == len(names), "no duplicate column"

        kinds = {
            feature.name: feature.value.kind
            for profile in population
            for feature in profile.features
        }
        for name in names:
            assert name in kinds, name
            assert is_numeric_value_kind(kinds[name]), name

    def test_every_step5_feature_is_either_a_column_or_a_recorded_refusal(
        self, schema, population
    ):
        """73 in, 52 kept plus 21 refused, with nothing unaccounted for.

        This is the test that would catch a Step 5 feature arriving with a kind
        the adapter had never seen.  It would show up as a *missing* exclusion
        rather than as an error, which is exactly the failure worth guarding.
        """
        published = {
            feature.name for profile in population for feature in profile.features
        }
        kept = set(schema.feature_names)
        refused = {excluded.name for excluded in schema.excluded_features}
        assert kept | refused == published
        assert kept.isdisjoint(refused)

    def test_every_dropped_column_records_a_reason(self, schema):
        """Nothing is silently dropped.  Silence would be indistinguishable
        from a bug, and a reader could not tell a refusal from an oversight."""
        assert schema.excluded_features
        for excluded in schema.excluded_features:
            assert isinstance(excluded, ExcludedFeature)
            assert excluded.name, excluded
            assert isinstance(excluded.reason, ExclusionReason), excluded
            assert excluded.detail.strip(), "%s has no explanation" % excluded.name
        names = [excluded.name for excluded in schema.excluded_features]
        assert len(set(names)) == len(names), "a column cannot be dropped twice"

        kept = set(schema.feature_names)
        assert not (kept & set(names)), "a column is either kept or dropped"

    def test_no_excluded_name_overlaps_a_kept_column(self, schema):
        kept = set(schema.feature_names)
        dropped = {excluded.name for excluded in schema.excluded_features}
        assert kept.isdisjoint(dropped)

    def test_label_and_timestamp_features_are_dropped_with_their_own_reason(
        self, schema
    ):
        by_name = {excluded.name: excluded for excluded in schema.excluded_features}
        reasons = {name: excluded.reason for name, excluded in by_name.items()}

        assert ExclusionReason.CATEGORICAL_LABEL in reasons.values()
        assert ExclusionReason.CATEGORICAL_LABEL_SET in reasons.values()
        assert ExclusionReason.INSTANT_TIMESTAMP in reasons.values()
        assert ExclusionReason.IDENTIFIER_LIKE_NAME not in reasons.values()

    def test_the_refusal_reasons_match_the_value_kinds_that_caused_them(self, schema):
        """The stated reason and the actual kind must agree, or the reason is
        decoration.  A label refused as a timestamp would send a reader looking
        in the wrong place."""
        expected = {
            "LABEL": ExclusionReason.CATEGORICAL_LABEL,
            "LABEL_SET": ExclusionReason.CATEGORICAL_LABEL_SET,
            "TIMESTAMP": ExclusionReason.INSTANT_TIMESTAMP,
        }
        for excluded in schema.excluded_features:
            if excluded.value_kind in expected:
                assert excluded.reason is expected[excluded.value_kind], excluded

    def test_no_identifier_like_column_is_ever_kept(self, schema):
        """The structural guarantee, checked against the real Step 5 output."""
        for name in schema.feature_names:
            assert not identifier_like_feature_name(name), name

    def test_the_identifier_refusal_works_when_step5_publishes_one(self):
        """Proved on a synthetic profile, because Step 5 does not currently
        publish an identifier-shaped measurement -- so testing only the real
        output would leave the guard unexercised."""
        population = make_population(3)
        forged = _rename_feature(population[0], "primary_address")
        schema = ProfileFeatureVectorizer().select_schema([forged])
        reasons = {
            excluded.name: excluded.reason for excluded in schema.excluded_features
        }
        assert reasons["primary_address"] is ExclusionReason.IDENTIFIER_LIKE_NAME
        assert "primary_address" not in schema.feature_names
        assert "primary_address" not in schema.feature_names

    def test_the_identifier_refusal_survives_a_whole_population(self):
        """The guard is in the per-feature path, so it holds for every profile
        in the population, not just the first."""
        population = [_rename_feature(p, "primary_address") for p in make_population(4)]
        schema = ProfileFeatureVectorizer().select_schema(population)
        reasons = {
            excluded.name: excluded.reason for excluded in schema.excluded_features
        }
        assert reasons["primary_address"] is ExclusionReason.IDENTIFIER_LIKE_NAME

    def test_the_schema_id_is_derived_from_the_column_names(self, schema):
        assert schema.schema_id == derive_schema_id(schema.feature_names)
        assert schema.schema_id.startswith("SCHEMA_")

    def test_the_schema_id_ignores_everything_but_the_columns(self):
        """Two populations with the same columns share a schema id.

        This is what lets a model fit on one run score a later run, and it is
        the reason the id is a function of the columns alone.
        """
        first = ProfileFeatureVectorizer().select_schema(make_population(4))
        second = ProfileFeatureVectorizer().select_schema(
            make_population(9, amount=3)
        )
        assert first.schema_id == second.schema_id

    def test_a_different_column_set_yields_a_different_schema_id(self, schema):
        assert derive_schema_id(schema.feature_names) != derive_schema_id(
            schema.feature_names[:-1]
        )

    def test_the_schema_states_its_own_scaling_and_imputation(self, schema):
        assert isinstance(schema.scaling, ScalingMethod)
        assert isinstance(schema.imputation, ImputationStrategy)
        assert schema.scaling is not ScalingMethod.NONE
        assert schema.imputation is ImputationStrategy.FITTED_MEDIAN

    def test_a_schema_refuses_an_identifier_shaped_column(self):
        for name in ("src_ip", "observation_id", "primary_address", "txid"):
            with pytest.raises(ValidationError):
                FeatureSchema(feature_names=(name,))

    def test_a_schema_requires_unique_non_empty_column_names(self):
        with pytest.raises(ValidationError):
            FeatureSchema(feature_names=("observation_count", "observation_count"))
        with pytest.raises(ValidationError):
            FeatureSchema(feature_names=("",))

    def test_selecting_a_schema_needs_the_population(self, population):
        """Selection reads the published feature list, so an empty input is a
        caller error rather than an empty schema."""
        assert ProfileFeatureVectorizer().select_schema(population).feature_names
        with pytest.raises(MLPreprocessorError):
            ProfileFeatureVectorizer().select_schema([])

    def test_the_adapter_refuses_anything_that_is_not_a_step5_profile(self, population):
        with pytest.raises(MLPreprocessorError):
            ProfileFeatureVectorizer().fit(population + [object()])

    def test_a_population_may_not_repeat_a_profile(self, population):
        """Step 5 derives an id from content, so a repeat is a copy.  Counting
        one profile twice would quietly weight it double."""
        with pytest.raises(MLPreprocessorError):
            ProfileFeatureVectorizer().fit(population + population[:1])


class TestImputationAndScaling:
    def test_every_population_value_is_finite(self, fitted, population):
        matrix = fitted.transform(population)
        assert len(matrix.rows) == len(population)
        for row in matrix.rows:
            assert len(row) == len(matrix.feature_names)
            for value in row:
                assert isinstance(value, float)
                assert not math.isnan(value)
                assert math.isfinite(value)

    def test_a_missing_measurement_is_imputed_from_the_fitted_median(self):
        """A measurement Step 5 could not compute becomes the fitted median,
        and the vector names it rather than pretending it was observed.

        The population here is one network-only profile, so every amount and
        evidence feature is genuinely unavailable for it.
        """
        unavailable = _network_only_profile()
        fitted = ProfileFeatureVectorizer().fit([unavailable])

        assert fitted.unavailable_count_of("total_input_amount") == 1
        assert fitted.observed_count_of("total_input_amount") == 0

        vector = fitted.transform_one(unavailable)
        assert "total_input_amount" in vector.imputed_feature_names

        index = list(vector.feature_names).index("total_input_amount")
        center = fitted.center_of("total_input_amount")
        scale = fitted.scale_of("total_input_amount")
        expected = 0.0 if fitted.is_constant("total_input_amount") else (
            center - center
        ) / scale
        assert vector.values[index] == pytest.approx(expected)

    def test_an_observed_measurement_is_not_reported_as_imputed(self, fitted, population):
        for profile in population:
            vector = fitted.transform_one(profile)
            for name in vector.imputed_feature_names:
                assert fitted.unavailable_count_of(name) > 0, name

    def test_imputation_does_not_poison_the_fitted_median(self):
        """The median is taken over observed values only.  Imputing first and
        then fitting would let a missing value re-enter as if it were data."""
        whole = make_population(6)
        holed = whole + [_network_only_profile()]

        reference = ProfileFeatureVectorizer().fit(whole)
        with_holed = ProfileFeatureVectorizer().fit(holed)

        for name in ("total_input_amount", "mean_amount"):
            assert with_holed.center_of(name) == reference.center_of(name), name
        assert with_holed.fitted_profile_count == len(holed)
        assert reference.fitted_profile_count == len(whole)

    def test_a_constant_column_is_reported_and_left_at_zero(self, population):
        fitted = ProfileFeatureVectorizer().fit(
            make_distinct_identity_population(4)
        )
        assert fitted.constant_feature_names
        matrix = fitted.transform(make_distinct_identity_population(4))
        for name in fitted.constant_feature_names:
            index = list(matrix.feature_names).index(name)
            for row in matrix.rows:
                assert row[index] == 0.0, name

    def test_the_scale_ladder_falls_back_until_it_is_positive(self, population):
        """A median absolute deviation of zero must not become a divide-by-zero."""
        fitted = ProfileFeatureVectorizer().fit(population)
        for name in fitted.feature_names:
            assert fitted.scale_of(name) > 0, name
            if not fitted.is_constant(name):
                assert fitted.scale_of(name) > 0, name

    def test_a_constant_column_still_reports_a_positive_scale(self, population):
        fitted = ProfileFeatureVectorizer().fit(population)
        for name in fitted.constant_feature_names:
            assert fitted.scale_of(name) > 0, name

    def test_fitting_needs_at_least_one_profile(self):
        with pytest.raises(MLPreprocessorError):
            ProfileFeatureVectorizer().fit([])

    def test_fitting_reports_what_it_saw(self, fitted, population):
        assert fitted.fitted_profile_count == len(population)
        for name, count in fitted.observed_value_counts:
            assert count <= len(population), name
        for name, _ in fitted.unavailable_value_counts:
            assert name in fitted.feature_names, name

    def test_units_are_recorded_for_the_features_that_have_them(
        self, fitted, population
    ):
        """Seconds are seconds.  Without the unit a reader cannot tell a
        duration from a count, and the numbers are equally plausible either way."""
        units = dict(fitted.value_units)
        assert set(units) == set(fitted.feature_names)
        assert any(unit for unit in units.values())
        for name, unit in units.items():
            assert name in fitted.feature_names, name

    def test_the_vector_reports_which_values_were_imputed(self, fitted, population):
        vector = fitted.transform_one(population[0])
        assert isinstance(vector, MLFeatureVector)
        assert set(vector.imputed_feature_names) <= set(vector.feature_names)
        assert len(vector.values) == len(vector.feature_names)

    def test_scaling_can_be_turned_off_and_is_then_still_usable(self, population):
        plain = ProfileFeatureVectorizer(PreprocessorConfig(scaling=ScalingMethod.NONE))
        fitted = plain.fit(population)
        matrix = fitted.transform(population)
        assert fitted.config.scaling is ScalingMethod.NONE
        for row in matrix.rows:
            assert all(math.isfinite(value) for value in row)

    def test_turning_scaling_off_leaves_the_raw_magnitudes(self, population):
        """Without scaling a count is the count Step 5 reported -- which is why
        a distance across features becomes meaningless, and why it is not the
        default."""
        fitted = ProfileFeatureVectorizer(
            PreprocessorConfig(scaling=ScalingMethod.NONE)
        ).fit(population)
        raw = dict(fitted.raw_values_for(population[0]))
        vector = fitted.transform_one(population[0])
        for index, name in enumerate(vector.feature_names):
            assert vector.values[index] == pytest.approx(raw[name]), name

    def test_robust_scaling_changes_the_numbers_but_not_the_ordering(self, population):
        """Scaling may move a value; it may not reorder a population."""
        scaled = ProfileFeatureVectorizer().fit(population).transform(population)
        plain = ProfileFeatureVectorizer(
            PreprocessorConfig(scaling=ScalingMethod.NONE)
        ).fit(population).transform(population)

        for name in ("total_input_amount", "observation_count"):
            index = list(scaled.feature_names).index(name)
            left = [row[index] for row in scaled.rows]
            right = [row[index] for row in plain.rows]
            assert sorted(range(len(left)), key=lambda i: left[i]) == sorted(
                range(len(right)), key=lambda i: right[i]
            ), name

    def test_a_vector_carries_the_schema_it_was_built_for(self, fitted, population):
        vector = fitted.transform_one(population[0])
        assert vector.schema_id == fitted.schema_id
        assert tuple(vector.feature_names) == tuple(fitted.feature_names)


class TestMatrixShape:
    def test_the_matrix_orders_the_population_by_profile_id(self, fitted, population):
        """Canonical order, not the order the caller happened to supply.

        Every fitted statistic and every random draw downstream is taken over
        this order, so a shuffled input produces a byte-identical model.  This
        is the single place that guarantee is established.
        """
        matrix = fitted.transform(population)
        assert list(matrix.profile_ids) == sorted(matrix.profile_ids)
        assert matrix.profile_ids == tuple(
            profile.profile_id for profile in sorted(population, key=lambda p: p.profile_id)
        )

    def test_shuffling_the_population_does_not_change_the_matrix(self, fitted, population):
        forward = fitted.transform(population)
        backward = fitted.transform(list(reversed(population)))
        assert forward.rows == backward.rows
        assert forward.profile_ids == backward.profile_ids

    def test_each_row_is_the_vector_of_the_profile_named_beside_it(
        self, fitted, population
    ):
        """A row that drifted from its id would be worse than no matrix: every
        later number would describe the wrong profile."""
        matrix = fitted.transform(population)
        by_id = {profile.profile_id: profile for profile in population}
        for profile_id, row in zip(matrix.profile_ids, matrix.rows):
            assert fitted.transform_one(by_id[profile_id]).values == row, profile_id

    def test_transforming_a_different_population_keeps_the_schema(
        self, fitted, population
    ):
        """A model fit on Monday must accept Tuesday without being refit."""
        fresh = make_population(3, amount=7)
        matrix = fitted.transform(fresh)
        assert matrix.schema_id == fitted.schema_id
        assert matrix.feature_names == fitted.feature_names
        assert matrix.profile_ids == tuple(
            profile.profile_id for profile in sorted(fresh, key=lambda p: p.profile_id)
        )

    def test_the_same_profiles_produce_the_same_matrix_twice(self, fitted, population):
        assert fitted.transform(population).rows == fitted.transform(population).rows

    def test_transforming_nothing_is_refused(self, fitted):
        """An empty matrix would look like a population of no measurements,
        which is a different claim from 'you gave me nothing'."""
        with pytest.raises(MLPreprocessorError):
            fitted.transform([])

    def test_the_median_of_an_even_count_averages_the_middle_pair(self):
        assert median([3.0, 1.0, 2.0, 4.0]) == 2.5
        assert median([5.0]) == 5.0
        assert median([1.0, 3.0, 2.0]) == 2.0

    def test_the_median_of_an_empty_sequence_is_refused(self):
        with pytest.raises(MLPreprocessorError):
            median([])

    def test_the_median_absolute_deviation_is_the_median_of_the_deviations(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert median_absolute_deviation(values) == median(
            [abs(value - median(values)) for value in values]
        )

    def test_the_median_absolute_deviation_of_identical_values_is_zero(self):
        assert median_absolute_deviation([2.0, 2.0, 2.0]) == 0.0

    def test_the_median_absolute_deviation_of_nothing_is_refused(self):
        with pytest.raises(MLPreprocessorError):
            median_absolute_deviation([])


class TestDeterminism:
    def test_two_fits_over_the_same_population_agree(self, population):
        first = ProfileFeatureVectorizer().fit(population)
        second = ProfileFeatureVectorizer().fit(list(population))
        assert first.centers == second.centers
        assert first.scales == second.scales
        assert first.schema_id == second.schema_id
        assert first.transform(population).rows == second.transform(population).rows

    def test_fitting_the_population_in_reverse_produces_the_same_fit(self, population):
        forward = ProfileFeatureVectorizer().fit(population)
        backward = ProfileFeatureVectorizer().fit(list(reversed(population)))
        assert forward.centers == backward.centers
        assert forward.scales == backward.scales
        assert forward.constant_feature_names == backward.constant_feature_names

    def test_a_vector_does_not_depend_on_the_order_of_the_population(
        self, population
    ):
        fitted = ProfileFeatureVectorizer().fit(population)
        reference = fitted.transform_one(population[0])
        reverse = ProfileFeatureVectorizer().fit(list(reversed(population)))
        assert reverse.transform_one(population[0]).values == reference.values

    def test_a_fitted_preprocessor_survives_a_json_round_trip(self, population):
        """A preprocessor is a fitted artefact.  If it cannot be stored and
        reloaded, every result would depend on living in one process."""
        from backend.app.ml.preprocessing import FittedFeaturePreprocessor

        fitted = ProfileFeatureVectorizer().fit(population)
        reloaded = FittedFeaturePreprocessor.model_validate_json(
            fitted.model_dump_json()
        )
        assert reloaded.centers == fitted.centers
        assert reloaded.scales == fitted.scales
        assert reloaded.transform(population).rows == fitted.transform(population).rows
