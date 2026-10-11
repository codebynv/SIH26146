"""Step 6 -- k-means grouping of Step 5 profiles.

A cluster here is a *group of similar measurements*, not a category and not a
finding.  The tests below hold the implementation to the properties that make
such a grouping publishable: the groups are reproducible, the labels are
content-derived rather than fitted-order artifacts, and the model reports its
own structure instead of implying more than the data supports.
"""

import json
import math

import pytest
from pydantic import ValidationError

from backend.app.ml.clustering import (
    CLUSTERING_ALGORITHM_VERSION,
    KMeansBehaviorClusterer,
    silhouette_score,
)
from backend.app.ml.models import (
    ClusterAssignment,
    ClusterModelConfig,
    ClusterSummary,
    ModelStatus,
    NOISE_CLUSTER_ID,
    cluster_label,
)
from tests.test_step6_ml_models import (
    make_distinct_identity_population,
    make_outlier,
    make_population,
)

POPULATION_SIZE = 9


@pytest.fixture()
def population():
    return make_population(POPULATION_SIZE)


@pytest.fixture()
def fitted(population):
    return KMeansBehaviorClusterer().fit(population)


@pytest.fixture()
def result(fitted):
    return fitted.result()


class TestPartitionShape:
    def test_a_fit_reports_its_status_and_its_k(self, fitted, population):
        assert fitted.status is ModelStatus.FITTED
        assert fitted.fitting_population_size == len(population)
        assert fitted.k is not None
        assert fitted.k >= 2
        assert len(fitted.labels) == fitted.k
        assert len(fitted.centroids) == fitted.k

    def test_every_profile_lands_in_exactly_one_cluster(self, result, population):
        ids = [assignment.profile_id for assignment in result.assignments]
        assert len(ids) == len(population)
        assert len(set(ids)) == len(population)
        assert set(ids) == {profile.profile_id for profile in population}

    def test_the_cluster_members_and_the_assignments_agree(self, result):
        """Two views of one partition.  If they disagree, a reader of the
        summary and a reader of the assignment list get different answers."""
        from_assignments = {}
        for assignment in result.assignments:
            from_assignments.setdefault(assignment.cluster_id, []).append(
                assignment.profile_id
            )
        for cluster in result.clusters:
            assert set(cluster.member_profile_ids) == set(
                from_assignments[cluster.cluster_id]
            )

    def test_the_clusters_partition_the_population(self, result, population):
        members = [
            profile_id
            for cluster in result.clusters
            for profile_id in cluster.member_profile_ids
        ]
        assert len(members) == len(population)
        assert len(set(members)) == len(members)
        assert set(members) == {profile.profile_id for profile in population}

    def test_every_member_is_closer_to_its_own_centroid_than_to_others(
        self, result, fitted
    ):
        """The defining property of a k-means partition, checked rather than
        assumed."""
        by_id = {
            profile_id: index
            for index, profile_id in enumerate(fitted.profile_ids)
        }
        centroids = fitted.centroids
        for assignment in result.assignments:
            index = by_id[assignment.profile_id]
            row = fitted.rows[index]
            own = _distance(row, centroids[assignment_label(fitted, assignment)])
            for label, centroid in enumerate(centroids):
                if label == assignment_label(fitted, assignment):
                    continue
                assert own <= _distance(row, centroid) + 1e-9, assignment.profile_id

    def test_a_cluster_is_never_empty(self, result, fitted):
        for cluster in result.clusters:
            assert cluster.member_profile_ids
        assert len(result.clusters) == fitted.k

    def test_every_centroid_is_the_mean_of_its_members(self, fitted, result):
        """Otherwise the reported centroid is not the thing the distance was
        measured to."""
        by_id = {profile_id: index for index, profile_id in enumerate(fitted.profile_ids)}
        for cluster in result.clusters:
            label = fitted.labels.index(cluster.cluster_id)
            indices = [by_id[profile_id] for profile_id in cluster.member_profile_ids]
            for column in range(len(fitted.feature_names)):
                expected = math.fsum(
                    fitted.rows[index][column] for index in indices
                ) / len(indices)
                assert fitted.centroids[label][column] == pytest.approx(
                    expected, abs=1e-9
                ), (cluster.cluster_id, fitted.feature_names[column])

    def test_the_matrix_uses_the_same_schema_as_the_anomaly_step(self, fitted):
        assert fitted.feature_names
        assert fitted.schema_id.startswith("SCHEMA_")


def assignment_label(fitted, assignment):
    return fitted.labels.index(assignment.cluster_id)


def _distance(row, centroid):
    return math.sqrt(
        math.fsum((left - right) ** 2 for left, right in zip(row, centroid))
    )


class TestLabels:
    def test_the_labels_are_the_published_shape(self, fitted):
        for index, label in enumerate(fitted.labels):
            assert label == cluster_label(index)
            assert label == "CLUSTER_%03d" % index

    def test_the_labels_are_derived_from_content_not_from_fit_order(self):
        """Two runs that find the same groups must publish the same label for
        the same group, however the search happened to reach it."""
        population = make_population(POPULATION_SIZE)
        forward = KMeansBehaviorClusterer().fit(population).result()
        backward = KMeansBehaviorClusterer().fit(
            list(reversed(population))
        ).result()
        assert {c.cluster_id: set(c.member_profile_ids) for c in forward.clusters} == {
            c.cluster_id: set(c.member_profile_ids) for c in backward.clusters
        }

    def test_the_labels_survive_a_reordering_of_the_population(self, population):
        forward = KMeansBehaviorClusterer().fit(population).result()
        rotated = KMeansBehaviorClusterer().fit(population[3:] + population[:3]).result()
        assert [c.cluster_id for c in forward.clusters] == [
            c.cluster_id for c in rotated.clusters
        ]
        for left, right in zip(forward.clusters, rotated.clusters):
            assert set(left.member_profile_ids) == set(right.member_profile_ids)

    def test_each_cluster_signature_is_a_digest_of_its_membership(self, result):
        """A signature is a content digest, so two clusters claiming the same
        members are visibly the same cluster."""
        seen = {}
        for cluster in result.clusters:
            assert cluster.signature
            if cluster.signature in seen:
                assert set(seen[cluster.signature]) == set(cluster.member_profile_ids)
            seen[cluster.signature] = cluster.member_profile_ids

    def test_two_different_clusters_never_share_a_signature(self, result):
        signatures = [cluster.signature for cluster in result.clusters]
        assert len(set(signatures)) == len(signatures)

    def test_a_cluster_label_beyond_the_fitted_k_is_refused(self):
        assert cluster_label(0) == "CLUSTER_000"
        assert cluster_label(9999) == "CLUSTER_9999"
        for index in (-1, 10000, True, 1.0, "0"):
            with pytest.raises(ValueError):
                cluster_label(index)

    def test_kmeans_publishes_no_noise(self, result, fitted):
        """K-means assigns every row to a centroid, so there is no such thing
        as an unassigned member.  The field exists only so a later step that
        does allow noise has a place to put it, and it is empty here rather than
        invented."""
        assert fitted.algorithm_version == CLUSTERING_ALGORITHM_VERSION
        assert result.noise_profile_ids == ()
        assert all(assignment.is_noise is False for assignment in result.assignments)
        assert all(assignment.cluster_id is not None for assignment in result.assignments)
        assert NOISE_CLUSTER_ID not in {c.cluster_id for c in result.clusters}


class TestDispersion:
    def test_dispersion_is_the_mean_distance_to_the_centroid(self, fitted, result):
        for cluster in result.clusters:
            assert cluster.dispersion >= 0.0
            assert math.isfinite(cluster.dispersion)

    def test_a_singleton_cluster_has_zero_dispersion(self, result):
        singletons = [
            cluster for cluster in result.clusters if len(cluster.member_profile_ids) == 1
        ]
        for cluster in singletons:
            assert cluster.dispersion == pytest.approx(0.0)

    def test_the_dispersion_detail_says_what_the_number_is(self, result):
        for cluster in result.clusters:
            assert "distance" in cluster.dispersion_detail.lower()
            assert "centroid" in cluster.dispersion_detail.lower()
            assert "mean" in cluster.dispersion_detail.lower()

    def test_dispersion_equals_the_mean_distance_to_the_reported_centroid(
        self, fitted, result
    ):
        """Recomputed from the model's own rows and centroids, so the published
        number cannot drift from the geometry it describes."""
        by_id = {
            profile_id: index for index, profile_id in enumerate(fitted.profile_ids)
        }
        for cluster in result.clusters:
            label = fitted.labels.index(cluster.cluster_id)
            centroid = fitted.centroids[label]
            distances = [
                _distance(fitted.rows[by_id[profile_id]], centroid)
                for profile_id in cluster.member_profile_ids
            ]
            assert cluster.dispersion == pytest.approx(
                math.fsum(distances) / len(distances)
            )

    def test_a_group_of_identical_measurements_has_zero_dispersion(self):
        """The vector demonstrably carries no identity, so profiles that differ
        only in identity are the same point, and a group of them is a point."""
        population = make_distinct_identity_population(3)
        result = KMeansBehaviorClusterer().fit(population).result()
        for cluster in result.clusters:
            assert cluster.dispersion == pytest.approx(0.0)

    def test_a_group_of_distinct_measurements_has_positive_dispersion(self, result):
        """The counterpart to the singleton and all-identical cases: a group
        holding more than one distinct vector cannot have zero spread.

        Located by search rather than assumed, so the test does not depend on
        which partition k-means happens to find.
        """
        spread = [
            cluster
            for cluster in result.clusters
            if len(cluster.member_profile_ids) > 1
        ]
        assert spread, "the population produced no group with more than one member"
        for cluster in spread:
            assert cluster.dispersion > 0.0, cluster.cluster_id


def _two_groups():
    """Two populations that genuinely differ: one barely active, one very."""
    return make_population(5, amount=1, tag="quiet") + make_population(
        5, amount=40, tag="busy", observations=3
    )


class TestModelSelection:
    def test_the_selection_scores_cover_every_admissible_k(self, fitted, population):
        scores = dict(fitted.selection_scores)
        assert set(scores) == set(range(2, min(8, len(population) - 1) + 1))
        assert fitted.k in scores
        assert scores[fitted.k] == max(scores.values())

    def test_the_chosen_k_is_the_best_scoring_admissible_k(self, fitted):
        scores = dict(fitted.selection_scores)
        best = max(scores.values())
        assert scores[fitted.k] == best

    def test_an_explicit_k_is_honoured_exactly(self, population):
        for k in (2, 3, 4):
            model = KMeansBehaviorClusterer(
                ClusterModelConfig(k=k, k_min=2, k_max=8)
            ).fit(population)
            assert model.k == k
            assert len(model.labels) == k
            assert model.result().status is ModelStatus.FITTED
            assert model.result().k == k
            assert len(model.result().clusters) == k

    def test_a_fixed_k_does_not_score_alternative_k(self, population):
        model = KMeansBehaviorClusterer(ClusterModelConfig(k=3)).fit(population)
        assert [k for k, _ in model.selection_scores] == [3]
        assert model.result().selection_scores == model.selection_scores

    def test_the_silhouette_of_a_real_partition_is_in_range(self, fitted):
        assert -1.0 <= fitted.selection_scores[0][1] <= 1.0

    def test_the_silhouette_of_one_group_is_neutral(self, population):
        """Undefined, reported as zero rather than as a fabricated score."""
        fitted = KMeansBehaviorClusterer().fit(population)
        matrix = _matrix_of(fitted)
        assert silhouette_score(matrix, [0] * len(fitted.rows)) == 0.0

    def test_the_silhouette_of_too_few_points_is_neutral(self):
        assert silhouette_score(_tiny_matrix(), [0, 1]) == 0.0

    def test_a_well_separated_partition_scores_above_a_mixed_one(self):
        matrix = _matrix_of(KMeansBehaviorClusterer().fit(_two_groups()))
        separated = silhouette_score(matrix, [0] * 5 + [1] * 5)
        mixed = silhouette_score(matrix, [index % 2 for index in range(10)])
        assert separated > mixed


def _matrix_of(fitted):
    from backend.app.ml.models import FeatureMatrix

    return FeatureMatrix(
        schema_id=fitted.schema_id,
        feature_names=fitted.feature_names,
        profile_ids=fitted.profile_ids,
        rows=fitted.rows,
    )


def _tiny_matrix():
    from backend.app.ml.models import FeatureMatrix

    return FeatureMatrix(
        schema_id="SCHEMA_TINY",
        feature_names=("a",),
        profile_ids=("p1", "p2"),
        rows=((0.0,), (1.0,)),
    )


class TestDegeneratePopulations:
    def test_no_projects_reports_insufficient_data(self):
        model = KMeansBehaviorClusterer().fit([])
        assert model.status is ModelStatus.INSUFFICIENT_DATA
        assert model.insufficient_reason
        assert model.k is None
        assert model.result().status is ModelStatus.INSUFFICIENT_DATA

    def test_one_or_two_profiles_reports_insufficient_data(self):
        for count in (1, 2):
            model = KMeansBehaviorClusterer().fit(make_population(count))
            assert model.status is ModelStatus.INSUFFICIENT_DATA
            assert model.insufficient_reason
            assert str(count) in model.insufficient_reason

    def test_an_impossible_k_range_reports_no_clusters(self, population):
        """A distinct status from insufficient data: the data may be fine, the
        request is not satisfiable.  Collapsing the two would tell a reader the
        population was too small when it was the request."""
        model = KMeansBehaviorClusterer(
            ClusterModelConfig(k_min=6, k_max=8)
        ).fit(make_population(5))
        assert model.status is ModelStatus.NO_CLUSTERS
        assert model.insufficient_reason
        assert "no admissible" in model.insufficient_reason

    def test_no_clusters_still_produces_a_readable_result(self, population):
        model = KMeansBehaviorClusterer(
            ClusterModelConfig(k_min=6, k_max=8)
        ).fit(make_population(5))
        result = model.result()
        assert result.status is ModelStatus.NO_CLUSTERS
        assert result.k is None
        assert result.clusters == ()
        assert result.assignments == ()
        assert result.insufficient_reason

    def test_identical_profiles_collapse_to_one_group(self):
        """Every row coincides, so no partition can separate them.

        The reported ``k`` is 1 even though ``k_min`` is 2: a centroid that
        claimed no members is not a group, so the model reports the one group
        that exists rather than the number that was asked for.  The silhouette
        of that partition is 0.0, which is why it was never going to be chosen
        over a real one.
        """
        population = make_distinct_identity_population(4)
        model = KMeansBehaviorClusterer().fit(population)
        assert model.status is ModelStatus.FITTED
        result = model.result()

        assert model.k == 1
        assert model.labels == ("CLUSTER_000",)
        assert len(result.clusters) == 1
        assert result.clusters[0].dispersion == pytest.approx(0.0)

        assert len(result.assignments) == 4
        for assignment in result.assignments:
            assert assignment.distance_to_centroid == pytest.approx(0.0)
            assert assignment.cluster_id == "CLUSTER_000"

    def test_a_minimum_population_is_configurable(self):
        config = ClusterModelConfig(min_profiles=6)
        model = KMeansBehaviorClusterer(config).fit(make_population(4))
        assert model.status is ModelStatus.INSUFFICIENT_DATA
        assert model.status is not KMeansBehaviorClusterer(config).fit(
            make_population(6)
        ).status

    def test_an_outlier_gets_a_group_of_its_own(self):
        """K-means has no noise label, so an extreme profile is assigned to its
        nearest centroid.  With one profile far outside the rest, the nearest
        centroid is the one seeded on it, so the outlier forms a singleton
        rather than being pulled into the bulk."""
        population = make_population(8) + [make_outlier()]
        result = KMeansBehaviorClusterer().fit(population).result()
        outlier = make_outlier()
        assignment = next(
            item for item in result.assignments if item.profile_id == outlier.profile_id
        )
        assert assignment.cluster_id is not None
        assert assignment.is_noise is False
        assert assignment.distance_to_centroid == pytest.approx(0.0)

        singleton = next(
            cluster
            for cluster in result.clusters
            if cluster.cluster_id == assignment.cluster_id
        )
        assert singleton.member_profile_ids == (outlier.profile_id,)

    def test_the_outlier_singleton_is_not_merged_into_the_bulk(self):
        """A cluster id cannot express 'far', so the separation is checked as a
        partition: the extreme profile is claimed by a centroid of its own."""
        population = make_population(8) + [make_outlier()]
        model = KMeansBehaviorClusterer(ClusterModelConfig(k=2)).fit(population)
        result = model.result()
        outlier = make_outlier()

        holder = next(
            cluster
            for cluster in result.clusters
            if outlier.profile_id in cluster.member_profile_ids
        )
        assert holder.member_profile_ids == (outlier.profile_id,)
        assert holder.dispersion == pytest.approx(0.0)

        bulk = next(
            cluster for cluster in result.clusters if cluster.cluster_id != holder.cluster_id
        )
        assert len(bulk.member_profile_ids) == 8
        assert bulk.dispersion > 0.0


class TestDeterminism:
    def test_two_fits_over_the_same_population_agree(self, population):
        first = KMeansBehaviorClusterer().fit(population)
        second = KMeansBehaviorClusterer().fit(population)
        assert first.model_dump() == second.model_dump()

    def test_shuffling_the_population_changes_nothing(self, population):
        forward = KMeansBehaviorClusterer().fit(population)
        backward = KMeansBehaviorClusterer().fit(list(reversed(population)))
        assert forward.model_dump() == backward.model_dump()

    def test_restarts_are_reported_and_finite(self, fitted):
        assert fitted.iterations > 0
        assert math.isfinite(fitted.inertia)
        assert fitted.inertia >= 0.0

    def test_more_restarts_cannot_make_the_objective_worse(self, population):
        few = KMeansBehaviorClusterer(ClusterModelConfig(n_init=1)).fit(population)
        many = KMeansBehaviorClusterer(ClusterModelConfig(n_init=12)).fit(population)
        assert many.inertia <= few.inertia + 1e-9

    def test_the_algorithm_version_is_published(self, fitted, result):
        assert fitted.algorithm_version == CLUSTERING_ALGORITHM_VERSION
        assert result.metadata.algorithm_version == CLUSTERING_ALGORITHM_VERSION
        assert result.metadata.algorithm == "K_MEANS"

    def test_a_fitted_model_survives_a_json_round_trip(self, population):
        from backend.app.ml.clustering import FittedClusteringModel

        model = KMeansBehaviorClusterer().fit(population)
        reloaded = FittedClusteringModel.model_validate_json(model.model_dump_json())
        assert reloaded.model_dump() == model.model_dump()
        assert reloaded.result().model_dump() == model.result().model_dump()


class TestPredict:
    def test_a_fitted_profile_predicts_into_its_own_cluster(self, fitted, population):
        for profile in population:
            assignment = fitted.predict_cluster(profile)
            assert isinstance(assignment, ClusterAssignment)
            assert assignment.profile_id == profile.profile_id
            assert assignment.cluster_id in fitted.labels
            assert assignment.distance_to_centroid >= 0.0

    def test_a_new_profile_predicts_without_refitting(self, fitted):
        fresh = make_population(2, amount=60, tag="fresh", observations=6)[1]
        assignment = fitted.predict_cluster(fresh)
        assert assignment.cluster_id in fitted.labels
        assert assignment.is_noise is False
        assert assignment.is_in_sample is False

    def test_a_prediction_reports_the_observations_it_used(self, fitted, population):
        assignment = fitted.predict_cluster(population[0])
        assert assignment.profile_id == population[0].profile_id
        assert population[0].observation_ids

    def test_a_prediction_goes_to_the_nearest_centroid(self, fitted, population):
        """Recomputed from the model's own centroids, so the reported distance
        and the reported cluster cannot disagree."""
        preprocessor = fitted.preprocessor
        for profile in population:
            assignment = fitted.predict_cluster(profile)
            vector = preprocessor.transform_one(profile)
            distances = [
                _distance(vector.values, centroid) for centroid in fitted.centroids
            ]
            nearest = min(range(len(distances)), key=lambda index: distances[index])
            assert assignment.cluster_id == fitted.labels[nearest]
            assert assignment.distance_to_centroid == pytest.approx(
                distances[nearest], abs=1e-9
            )

    def test_a_prediction_is_out_of_sample_for_an_unseen_profile(self, fitted):
        """A profile the model never saw is labelled as such on the assignment,
        because a stored assignment is otherwise indistinguishable from a
        re-scored one."""
        fitted_ids = set(fitted.profile_ids)
        fresh = next(
            profile
            for profile in make_population(9, amount=70, tag="unseen", observations=5)
            if profile.profile_id not in fitted_ids
        )
        assignment = fitted.predict_cluster(fresh)
        assert assignment.is_in_sample is False
        assert assignment.profile_id == fresh.profile_id

    def test_predicting_with_an_unfitted_model_is_refused(self):
        model = KMeansBehaviorClusterer().fit(make_population(2))
        assert model.status is ModelStatus.INSUFFICIENT_DATA
        with pytest.raises(ValueError):
            model.predict_cluster(make_population(1)[0])


class TestConfiguration:
    def test_the_defaults_are_published_and_frozen(self):
        clusterer = KMeansBehaviorClusterer()
        assert clusterer.config.algorithm.value == "K_MEANS"
        assert clusterer.config.k is None
        assert clusterer.config.k_min == 2
        assert clusterer.config.k_max == 8
        assert clusterer.config.n_init == 10
        assert clusterer.config.selection_metric == "SILHOUETTE"
        with pytest.raises(ValidationError):
            clusterer.config.n_init = 1

    def test_the_configuration_refuses_impossible_settings(self):
        for kwargs in (
            {"k": 1},
            {"k_min": 1},
            {"k_max": 0},
            {"n_init": 0},
            {"max_iterations": 0},
            {"min_profiles": 2},
            {"tolerance": -1.0},
        ):
            with pytest.raises(ValidationError):
                ClusterModelConfig(**kwargs)

    def test_a_k_below_k_min_is_refused(self):
        with pytest.raises(ValidationError):
            ClusterModelConfig(k=1, k_min=2, k_max=8)

    def test_a_k_above_k_max_is_refused(self):
        with pytest.raises(ValidationError):
            ClusterModelConfig(k=9, k_min=2, k_max=8)

    def test_an_inverted_k_range_is_refused(self):
        with pytest.raises(ValidationError):
            ClusterModelConfig(k_min=8, k_max=3)

    def test_an_iteration_cap_of_one_still_produces_a_partition(self, population):
        model = KMeansBehaviorClusterer(ClusterModelConfig(max_iterations=1)).fit(
            population
        )
        assert model.status is ModelStatus.FITTED
        assert model.iterations <= 1
        assert len(model.result().assignments) == len(population)

    def test_a_single_restart_still_produces_a_partition(self, population):
        model = KMeansBehaviorClusterer(ClusterModelConfig(n_init=1)).fit(population)
        assert model.status is ModelStatus.FITTED
        assert model.result().clusters

    def test_the_metadata_carries_the_whole_configuration(self, result):
        config = json.loads(result.metadata.config_json)
        assert config["k_min"] == 2
        assert config["k_max"] == 8
        assert config["n_init"] == 10
        assert config["selection_metric"] == "SILHOUETTE"
        assert config["random_state"] == 0

    def test_the_metadata_names_the_schema_it_was_fit_on(self, fitted, result):
        assert result.metadata.feature_schema_id == fitted.schema_id
        assert tuple(result.metadata.feature_names) == tuple(result.feature_names)

    def test_the_model_id_is_reproducible(self, population):
        first = KMeansBehaviorClusterer().fit(population)
        second = KMeansBehaviorClusterer().fit(population)
        assert first.model_id == second.model_id
        assert first.model_id.startswith("CLUSTERING_K_MEANS_")


class TestResultShape:
    def test_a_summary_requires_members(self):
        with pytest.raises(ValidationError):
            ClusterSummary(
                cluster_id="CLUSTER_000",
                signature="abc",
                member_profile_ids=(),
                centroid=(("observation_count", 0.0),),
                dispersion=0.0,
                dispersion_detail="mean distance to the centroid",
            )

    def test_a_summary_refuses_a_negative_dispersion(self):
        with pytest.raises(ValidationError):
            ClusterSummary(
                cluster_id="CLUSTER_000",
                signature="abc",
                member_profile_ids=("00000000-0000-5000-8000-000000000000",),
                centroid=(("observation_count", 0.0),),
                dispersion=-1.0,
                dispersion_detail="mean distance to the centroid",
            )

    def test_an_assignment_refuses_a_negative_distance(self):
        with pytest.raises(ValidationError):
            ClusterAssignment(
                profile_id="00000000-0000-5000-8000-000000000000",
                cluster_id="CLUSTER_000",
                is_noise=False,
                distance_to_centroid=-0.1,
                is_in_sample=True,
            )

    def test_an_assignment_needs_a_profile_id(self):
        with pytest.raises(ValidationError):
            ClusterAssignment(
                profile_id="",
                cluster_id="CLUSTER_000",
                is_noise=False,
                distance_to_centroid=0.1,
                is_in_sample=True,
            )

    def test_every_cluster_id_in_a_result_is_one_the_model_published(
        self, fitted, result
    ):
        """The cross-check the result model cannot make for itself, done here."""
        for assignment in result.assignments:
            assert assignment.cluster_id in fitted.labels
        for cluster in result.clusters:
            assert cluster.cluster_id in fitted.labels

    def test_a_result_reports_the_selection_scores_it_used(self, fitted, result):
        assert result.selection_scores == fitted.selection_scores
        assert result.k == fitted.k
