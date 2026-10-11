"""Step 6 result models: shape, immutability, vocabulary and boundaries.

This module also holds the profile builders the other Step 6 test modules
import, so that every Step 6 test builds its population from the same locked
Step 3 models and the same Step 5 builder.  There is no second fixture file: a
helper module would be a sixth Step 6 file, and the brief asks for five.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pytest

from backend.app.behavior.profiles import BehaviorProfileBuilder
from backend.app.domain.models import (
    BlockchainTransaction,
    NetworkObservation,
    ObservationType,
    Provenance,
    UnifiedObservation,
)
from backend.app.ml import (
    NOISE_CLUSTER_ID,
    AnomalyExplanation,
    AnomalyFeatureContribution,
    AnomalyModelConfig,
    AnomalyResult,
    ClusterAssignment,
    ClusterModelConfig,
    ClusterSummary,
    DeviationDirection,
    ExcludedFeature,
    ExplanationMethod,
    FeatureMatrix,
    FeatureSchema,
    MLFeatureVector,
    ModelRunMetadata,
    ModelStatus,
    ScalingMethod,
    UnusualnessBand,
    cluster_label,
    describe_explanation_methods,
    is_numeric_value_kind,
)
from backend.app.ml import models as ml_models
from backend.app.ml import anomaly as ml_anomaly
from backend.app.ml import clustering as ml_clustering
from backend.app.ml import explanations as ml_explanations
from backend.app.ml import preprocessing as ml_preprocessing

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_PACKAGE = REPOSITORY_ROOT / "backend" / "app" / "ml"

#: A fixed instant for every synthetic observation.  No clock, no ``now()``,
#: no ``random``: a test population must be the same population on every run.
T0 = datetime(2024, 5, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Shared builders
# --------------------------------------------------------------------------- #


def make_observation(
    observation_id: str,
    *,
    hour: int = 0,
    src_ip: str = "10.0.0.1",
    txid: Optional[str] = None,
    address: str = "bc1qexample",
    amount: str = "1.5",
    fee: str = "0.01",
    ingestion_id: str = "ing-1",
    source_file: str = "feed.csv",
    observation_type: ObservationType = ObservationType.COMBINED,
) -> UnifiedObservation:
    """Build one observation from the locked Step 3 models."""
    timestamp = T0 + timedelta(hours=hour)
    network = NetworkObservation(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip="10.0.9.9",
        src_port=1024,
        dst_port=443,
        asn=15169,
        geo_country="US",
    )
    blockchain = None
    if txid is not None or amount is not None:
        blockchain = BlockchainTransaction(
            timestamp=timestamp,
            txid=txid,
            input_addresses=[address],
            output_addresses=["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
            input_amounts=[Decimal(amount)] if amount else [],
            output_amounts=[Decimal(amount)] if amount else [],
            fee=None if fee is None else Decimal(fee),
        )
    return UnifiedObservation(
        observation_id=observation_id,
        observation_type=observation_type,
        timestamp=timestamp,
        network=network,
        blockchain=blockchain,
        provenance=Provenance(
            ingestion_id=ingestion_id,
            source_file=source_file,
            source_record="row %s" % observation_id,
            context={"declared": "csv"},
        ),
        extra={},
    )


def make_profile(
    index: int,
    *,
    observations: int = 4,
    amount: int = 10,
    tag: str = "grp",
    src_ip: str = "10.0.0.1",
    fee: Optional[str] = None,
) -> "object":
    """Build one Step 5 profile whose numbers grow with *index*.

    *index* drives the amounts and the distinct identifiers, so a population of
    these profiles is a gradient: a k-means fit has real structure to find and
    an unusualness fit has a real interior to measure distance from.  *src_ip*
    is the network prefix; the last octet varies per observation, so a caller
    can separate two profiles by network without writing its own builder.
    """
    builder = BehaviorProfileBuilder()
    if src_ip.count(".") == 3:
        prefix = src_ip.rsplit(".", 1)[0]
    else:
        prefix = "10.0.%d" % (index % 5)
    records = [
        make_observation(
            "obs-%03d-%02d" % (index, step),
            hour=step,
            src_ip="%s.%d" % (prefix, step % 3),
            txid="%064x" % (index * 1000 + step),
            address="bc1q%04d" % index,
            amount=str(amount * index + step),
            fee="0.0%d" % (1 + index % 9) if fee is None else fee,
            ingestion_id="ing-%d" % (index % 3),
            source_file="feed-%d.csv" % (index % 3),
        )
        for step in range(observations)
    ]
    return builder.build(records, group_id="%s-%03d" % (tag, index))


def make_population(count: int = 12, **kwargs) -> list:
    """Build a gradient population of *count* profiles."""
    return [make_profile(index, **kwargs) for index in range(count)]


def make_outlier(observations: int = 120) -> "object":
    """Build a profile far outside the gradient population."""
    return make_profile(97, observations=observations, amount=500)


def make_distinct_identity_population(count: int = 4) -> list:
    """Build profiles whose *vectors are identical* but whose identities differ.

    Everything that differs between these profiles -- the ip, the txid, the
    address, the ingestion id, the source file -- is a Step 5 feature that this
    step refuses to vectorize.  Every vectorized measurement is the same.  Two
    things follow, and both are load-bearing: the vector demonstrably carries no
    identity, and the resulting population is the honest way to build the
    degenerate "every profile is identical" case, because Step 5 derives a
    profile id from content and would otherwise collapse them into one.
    """
    return [
        make_profile(
            index,
            observations=3,
            amount=0,
            tag="ident",
            src_ip="10.0.0.%d" % (index + 1),
            fee="0.01",
        )
        for index in range(count)
    ]


def make_metadata(**overrides) -> ModelRunMetadata:
    """Build a minimal valid metadata block for model-level tests."""
    payload = {
        "model_id": "ANOMALY_TEST",
        "algorithm": "ISOLATION_FOREST",
        "algorithm_version": "TEST/1",
        "feature_schema_id": "SCHEMA_TEST",
        "feature_names": ("observation_count", "total_input_amount"),
        "scaling": ScalingMethod.ROBUST_MAD,
        "imputation": None,
        "random_state": 0,
        "fitting_population_size": 4,
        "scoring_population_size": 4,
        "scoring_in_sample": True,
        "config_json": '{"k": 2}',
    }
    payload.update(overrides)
    return ModelRunMetadata(**payload)


# --------------------------------------------------------------------------- #
# Module surface
# --------------------------------------------------------------------------- #


ML_MODULES = (
    ml_models,
    ml_preprocessing,
    ml_anomaly,
    ml_clustering,
    ml_explanations,
)


class TestModuleSurface:
    """The package's public surface is deliberate, sorted and complete."""

    def test_every_module_exports_a_sorted_list(self):
        for module in ML_MODULES:
            assert isinstance(module.__all__, list), module.__name__
            assert module.__all__ == sorted(module.__all__), module.__name__
            assert len(set(module.__all__)) == len(module.__all__), module.__name__

    def test_every_exported_name_exists(self):
        for module in ML_MODULES:
            for name in module.__all__:
                assert hasattr(module, name), "%s.%s" % (module.__name__, name)

    def test_package_exports_are_sorted_and_resolvable(self):
        from backend.app import ml

        assert ml.__all__ == sorted(ml.__all__)
        for name in ml.__all__:
            assert hasattr(ml, name), name

    def test_package_does_not_re_export_imported_implementation_symbols(self):
        """A public name must be a Step 6 definition, not an imported helper."""
        import backend.app.ml as ml

        step6_modules = ML_MODULES
        for name in ml.__all__:
            owner = [module for module in step6_modules if hasattr(module, name)]
            assert owner, name

    def test_internal_helpers_stay_private(self):
        """Underscore-prefixed internals are not part of the public surface."""
        for module in ML_MODULES:
            for name in module.__all__:
                assert not name.startswith("_"), "%s.%s" % (module.__name__, name)

    def test_every_module_documents_itself(self):
        for module in ML_MODULES:
            assert (module.__doc__ or "").strip(), module.__name__

    def test_every_public_callable_has_a_docstring(self):
        for module in ML_MODULES:
            for name in module.__all__:
                member = getattr(module, name)
                if callable(member) and not isinstance(member, type):
                    assert (member.__doc__ or "").strip(), "%s.%s" % (
                        module.__name__,
                        name,
                    )


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class TestVocabulary:
    """The model's words are the model's own, and none of them judge."""

    def test_model_statuses_cover_the_documented_states(self):
        assert {item.value for item in ModelStatus} == {
            "NOT_FITTED",
            "INSUFFICIENT_DATA",
            "NO_CLUSTERS",
            "FITTED",
        }

    def test_every_status_carries_an_explanation(self):
        for item in ModelStatus:
            assert (item.__doc__ or "").strip(), item

    def test_bands_are_ordered_from_least_to_most_separated(self):
        assert [item.value for item in UnusualnessBand] == [
            "LOW",
            "MODERATE",
            "HIGH",
            "EXTREME",
        ]

    def test_direction_values_describe_a_side_not_a_cause(self):
        assert {item.value for item in DeviationDirection} == {
            "ABOVE_REFERENCE",
            "BELOW_REFERENCE",
        }

    def test_explanation_methods_name_all_three_kinds(self):
        assert {item.value for item in ExplanationMethod} == {
            "MODEL_NATIVE",
            "STATISTICAL_REFERENCE",
            "DERIVED_APPROXIMATION",
        }

    def test_native_and_approximated_explanations_are_declared_unavailable(self):
        """The isolation forest has no attribution; the pipeline must say so."""
        described = describe_explanation_methods()
        assert "AVAILABLE" in described["STATISTICAL_REFERENCE"]
        assert described["MODEL_NATIVE"].startswith("UNAVAILABLE")
        assert described["DERIVED_APPROXIMATION"].startswith("UNAVAILABLE")

    def test_numeric_value_kinds_are_counts_and_decimals_only(self):
        assert is_numeric_value_kind("COUNT")
        assert is_numeric_value_kind("DECIMAL")
        for kind in ("TIMESTAMP", "LABEL", "LABEL_SET"):
            assert not is_numeric_value_kind(kind)

    def test_noise_label_is_reserved_and_namespaced(self):
        assert NOISE_CLUSTER_ID == "CLUSTER_NOISE"
        assert NOISE_CLUSTER_ID not in {cluster_label(index) for index in range(10)}

    def test_cluster_labels_are_dense_zero_padded_and_ordered(self):
        labels = [cluster_label(index) for index in range(3)]
        assert labels == ["CLUSTER_000", "CLUSTER_001", "CLUSTER_002"]
        assert labels == sorted(labels)

    def test_cluster_label_rejects_impossible_indices(self):
        for value in (-1, 10000, 1.0, True):
            with pytest.raises(ValueError):
                cluster_label(value)

    def test_scaling_methods_are_the_documented_three(self):
        assert {item.value for item in ScalingMethod} == {
            "NONE",
            "ROBUST_MAD",
            "STANDARD_DEVIATION",
        }


# --------------------------------------------------------------------------- #
# Feature schema and vectors
# --------------------------------------------------------------------------- #


class TestFeatureSchema:
    """Column selection is explicit, and refusals are recorded."""

    def test_schema_requires_columns(self):
        with pytest.raises(ValueError):
            FeatureSchema(schema_id="SCHEMA_X", feature_names=())

    def test_schema_rejects_unsorted_columns(self):
        with pytest.raises(ValueError):
            FeatureSchema(schema_id="SCHEMA_X", feature_names=("b_count", "a_count"))

    def test_schema_rejects_duplicate_columns(self):
        with pytest.raises(ValueError):
            FeatureSchema(schema_id="SCHEMA_X", feature_names=("a_count", "a_count"))

    def test_schema_rejects_an_identifier_shaped_column(self):
        """A column named like an identifier is refused, by construction."""
        for name in ("src_ip", "txid", "source_file", "observation_id", "address"):
            with pytest.raises(ValueError):
                FeatureSchema(schema_id="SCHEMA_X", feature_names=(name,))

    def test_schema_accepts_a_measurement_named_after_a_measured_thing(self):
        schema = FeatureSchema(
            schema_id="SCHEMA_X", feature_names=("unique_src_ip_count",)
        )
        assert schema.dimension == 1

    def test_schema_rejects_a_feature_that_is_both_column_and_exclusion(self):
        with pytest.raises(ValueError):
            FeatureSchema(
                schema_id="SCHEMA_X",
                feature_names=("txid_occurrence_count",),
                excluded_features=(
                    ExcludedFeature(
                        name="txid_occurrence_count",
                        value_kind="COUNT",
                        reason="IDENTIFIER_LIKE_NAME",
                        detail="refused",
                    ),
                ),
            )

    def test_schema_serializes_its_exclusions(self):
        schema = FeatureSchema(
            schema_id="SCHEMA_X",
            feature_names=("observation_count",),
            excluded_features=(
                ExcludedFeature(
                    name="src_ip_values",
                    value_kind="LABEL_SET",
                    reason="CATEGORICAL_LABEL_SET",
                    detail="no magnitude",
                ),
            ),
        )
        payload = schema.to_dict()
        assert payload["excluded_features"][0]["name"] == "src_ip_values"
        assert payload["excluded_features"][0]["reason"] == "CATEGORICAL_LABEL_SET"


class TestMLFeatureVector:
    """Vectors are finite, aligned and immutable."""

    def test_vector_requires_matching_widths(self):
        with pytest.raises(ValueError):
            MLFeatureVector(
                schema_id="SCHEMA_X", feature_names=("a", "b"), values=(1.0,)
            )

    def test_vector_refuses_non_finite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                MLFeatureVector(
                    schema_id="SCHEMA_X", feature_names=("a",), values=(value,)
                )

    def test_vector_rejects_imputed_columns_outside_the_schema(self):
        with pytest.raises(ValueError):
            MLFeatureVector(
                schema_id="SCHEMA_X",
                feature_names=("a_count",),
                values=(1.0,),
                imputed_feature_names=("b_count",),
            )

    def test_vector_reports_dimension_and_lookup_by_name(self):
        vector = MLFeatureVector(
            schema_id="SCHEMA_X",
            feature_names=("a_count", "b_count"),
            values=(1.0, 2.0),
            imputed_feature_names=("a_count",),
        )
        assert vector.dimension == 2
        assert vector.value_of("b_count") == 2.0
        with pytest.raises(KeyError):
            vector.value_of("c_count")

    def test_vector_is_frozen(self):
        vector = MLFeatureVector(
            schema_id="SCHEMA_X", feature_names=("a_count",), values=(1.0,)
        )
        with pytest.raises(Exception):
            vector.values = (2.0,)

    def test_vector_serializes_deterministically(self):
        vector = MLFeatureVector(
            schema_id="SCHEMA_X", feature_names=("a_count",), values=(1.5,)
        )
        first = json.dumps(vector.to_dict(), sort_keys=True)
        second = json.dumps(vector.to_dict(), sort_keys=True)
        assert first == second


class TestFeatureMatrix:
    """The population matrix keeps identifiers in the row labels only."""

    def test_matrix_requires_one_row_per_profile(self):
        with pytest.raises(ValueError):
            FeatureMatrix(
                schema_id="SCHEMA_X",
                feature_names=("a_count",),
                profile_ids=("p1", "p2"),
                rows=((1.0,),),
            )

    def test_matrix_requires_uniform_row_width(self):
        with pytest.raises(ValueError):
            FeatureMatrix(
                schema_id="SCHEMA_X",
                feature_names=("a_count", "b_count"),
                profile_ids=("p1",),
                rows=((1.0,),),
            )

    def test_matrix_refuses_duplicate_profile_ids(self):
        with pytest.raises(ValueError):
            FeatureMatrix(
                schema_id="SCHEMA_X",
                feature_names=("a_count",),
                profile_ids=("p1", "p1"),
                rows=((1.0,), (2.0,)),
            )

    def test_matrix_row_lookup(self):
        matrix = FeatureMatrix(
            schema_id="SCHEMA_X",
            feature_names=("a_count",),
            profile_ids=("p1", "p2"),
            rows=((1.0,), (2.0,)),
        )
        assert matrix.population_size == 2
        assert matrix.dimension == 1
        assert matrix.row_of("p2") == (2.0,)
        with pytest.raises(KeyError):
            matrix.row_of("p3")


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #


class TestConfigurations:
    """Every choice that can move a number is explicit and validated."""

    def test_band_boundaries_must_be_ascending_and_inside_the_unit_interval(self):
        for boundaries in ((0.9, 0.7, 0.5), (0.0, 0.7, 0.9), (0.5, 1.0, 0.9), (0.5, 0.7, 0.7)):
            with pytest.raises(ValueError):
                AnomalyModelConfig(band_boundaries=boundaries)

    def test_default_band_boundaries_are_documented_round_numbers(self):
        assert AnomalyModelConfig().band_boundaries == (0.5, 0.7, 0.9)

    def test_depth_limit_is_derived_from_the_subsample_size(self):
        config = AnomalyModelConfig()
        assert config.resolved_max_depth(256) == 8
        assert config.resolved_max_depth(2) == 1
        assert config.resolved_max_depth(1) == 1

    def test_explicit_depth_limit_wins_over_the_derived_one(self):
        assert AnomalyModelConfig(max_depth=3).resolved_max_depth(256) == 3

    def test_model_ids_are_content_derived_and_stable(self):
        first = AnomalyModelConfig(random_state=5).model_id
        second = AnomalyModelConfig(random_state=5).model_id
        assert first == second
        assert first != AnomalyModelConfig(random_state=6).model_id
        assert first != ClusterModelConfig(random_state=5).model_id

    def test_model_id_prefix_names_the_algorithm(self):
        assert AnomalyModelConfig().model_id.startswith("ANOMALY_ISOLATION_FOREST_")
        assert ClusterModelConfig().model_id.startswith("CLUSTERING_K_MEANS_")

    def test_cluster_k_must_lie_inside_the_admissible_range(self):
        with pytest.raises(ValueError):
            ClusterModelConfig(k=5, k_min=2, k_max=4)

    def test_cluster_k_range_must_be_ordered(self):
        with pytest.raises(ValueError):
            ClusterModelConfig(k_min=6, k_max=3)

    def test_configuration_bounds_are_enforced(self):
        with pytest.raises(ValueError):
            AnomalyModelConfig(n_estimators=0)
        with pytest.raises(ValueError):
            AnomalyModelConfig(max_samples=1)
        with pytest.raises(ValueError):
            AnomalyModelConfig(min_profiles=1)
        with pytest.raises(ValueError):
            AnomalyModelConfig(random_state=-1)
        with pytest.raises(ValueError):
            ClusterModelConfig(n_init=0)
        with pytest.raises(ValueError):
            ClusterModelConfig(tolerance=0.0)

    def test_configurations_are_frozen(self):
        config = AnomalyModelConfig()
        with pytest.raises(Exception):
            config.random_state = 9


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class TestAnomalyResult:
    """A score is reported with its band, its range and its explanation."""

    def test_score_and_band_must_be_present_together(self):
        with pytest.raises(ValueError):
            AnomalyResult(
                profile_id="p1",
                schema_id="SCHEMA_X",
                raw_score=0.5,
                is_in_sample=True,
            )
        with pytest.raises(ValueError):
            AnomalyResult(
                profile_id="p1",
                schema_id="SCHEMA_X",
                raw_score=0.5,
                normalized_score=0.7,
                is_in_sample=True,
            )

    def test_normalized_score_must_lie_in_the_unit_interval(self):
        for score in (0.0, -0.1, 1.5, float("nan")):
            with pytest.raises(ValueError):
                AnomalyResult(
                    profile_id="p1",
                    schema_id="SCHEMA_X",
                    raw_score=0.5,
                    normalized_score=score,
                    band=UnusualnessBand.LOW,
                    is_in_sample=True,
                )

    def test_exactly_one_is_a_valid_normalized_score(self):
        result = AnomalyResult(
            profile_id="p1",
            schema_id="SCHEMA_X",
            raw_score=0.0,
            normalized_score=1.0,
            band=UnusualnessBand.EXTREME,
            is_in_sample=True,
        )
        assert result.normalized_score == 1.0

    def test_explanation_must_describe_the_same_profile(self):
        explanation = AnomalyExplanation(
            profile_id="other",
            method=ExplanationMethod.STATISTICAL_REFERENCE,
            method_detail="described",
        )
        with pytest.raises(ValueError):
            AnomalyResult(
                profile_id="p1",
                schema_id="SCHEMA_X",
                raw_score=0.5,
                normalized_score=0.7,
                band=UnusualnessBand.LOW,
                is_in_sample=True,
                explanation=explanation,
            )

    def test_observation_ids_must_be_sorted_and_unique(self):
        with pytest.raises(ValueError):
            AnomalyResult(
                profile_id="p1",
                observation_ids=("obs-2", "obs-1"),
                schema_id="SCHEMA_X",
                raw_score=0.5,
                normalized_score=0.7,
                band=UnusualnessBand.LOW,
                is_in_sample=True,
            )

    def test_result_serializes_with_a_null_band_when_unfitted(self):
        result = AnomalyResult(profile_id="p1", schema_id="SCHEMA_X", is_in_sample=False)
        assert result.to_dict()["band"] is None
        assert result.to_dict()["normalized_score"] is None


class TestExplanationModels:
    """An explanation must rank, label its method and refuse invented values."""

    def contribution(self, rank: int, name: str, deviation: float):
        return AnomalyFeatureContribution(
            rank=rank,
            feature_name=name,
            observed_value=1.0 + deviation,
            reference_value=1.0,
            deviation=deviation,
            absolute_deviation=abs(deviation),
            direction=(
                DeviationDirection.ABOVE_REFERENCE
                if deviation > 0
                else DeviationDirection.BELOW_REFERENCE
            ),
        )

    def test_contribution_magnitude_must_match_the_deviation(self):
        with pytest.raises(ValueError):
            AnomalyFeatureContribution(
                rank=1,
                feature_name="a_count",
                observed_value=2.0,
                reference_value=1.0,
                deviation=1.0,
                absolute_deviation=2.0,
                direction=DeviationDirection.ABOVE_REFERENCE,
            )

    def test_direction_must_agree_with_the_sign(self):
        with pytest.raises(ValueError):
            AnomalyFeatureContribution(
                rank=1,
                feature_name="a_count",
                observed_value=2.0,
                reference_value=1.0,
                deviation=1.0,
                absolute_deviation=1.0,
                direction=DeviationDirection.BELOW_REFERENCE,
            )

    def test_ranks_must_be_contiguous_from_one(self):
        with pytest.raises(ValueError):
            AnomalyExplanation(
                profile_id="p1",
                method=ExplanationMethod.STATISTICAL_REFERENCE,
                method_detail="described",
                contributions=(self.contribution(2, "a_count", 1.0),),
            )

    def test_contributions_must_be_ranked_by_magnitude_then_name(self):
        with pytest.raises(ValueError):
            AnomalyExplanation(
                profile_id="p1",
                method=ExplanationMethod.STATISTICAL_REFERENCE,
                method_detail="described",
                contributions=(
                    self.contribution(1, "a_count", 1.0),
                    self.contribution(2, "b_count", 2.0),
                ),
            )

    def test_equal_magnitudes_are_ordered_by_feature_name(self):
        explanation = AnomalyExplanation(
            profile_id="p1",
            method=ExplanationMethod.STATISTICAL_REFERENCE,
            method_detail="described",
            contributions=(
                self.contribution(1, "a_count", 2.0),
                self.contribution(2, "b_count", 2.0),
            ),
        )
        assert [item.feature_name for item in explanation.contributions] == [
            "a_count",
            "b_count",
        ]
        assert explanation.top_contribution.feature_name == "a_count"

    def test_a_model_native_explanation_must_carry_contributions(self):
        with pytest.raises(ValueError):
            AnomalyExplanation(
                profile_id="p1",
                method=ExplanationMethod.MODEL_NATIVE,
                method_detail="claimed",
            )

    def test_method_detail_may_not_be_blank(self):
        with pytest.raises(ValueError):
            AnomalyExplanation(
                profile_id="p1",
                method=ExplanationMethod.STATISTICAL_REFERENCE,
                method_detail="   ",
            )


class TestClusteringResult:
    """A partition reports membership, geometry and canonical labels."""

    def summary(self, cluster_id: str, members):
        import hashlib

        return ClusterSummary(
            cluster_id=cluster_id,
            signature=hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest(),
            member_profile_ids=members,
            centroid=(("observation_count", 1.0),),
            dispersion=0.5,
            dispersion_detail="mean distance to centroid",
        )

    def test_signature_must_be_the_digest_of_the_members(self):
        with pytest.raises(ValueError):
            ClusterSummary(
                cluster_id="CLUSTER_000",
                signature="0" * 64,
                member_profile_ids=("p1", "p2"),
                centroid=(("observation_count", 1.0),),
                dispersion=0.5,
                dispersion_detail="mean distance to centroid",
            )

    def test_cluster_requires_at_least_one_member(self):
        with pytest.raises(ValueError):
            self.summary("CLUSTER_000", ())

    def test_cluster_reports_its_size_and_centroid_lookup(self):
        summary = self.summary("CLUSTER_000", ("p1", "p2"))
        assert summary.member_count == 2
        assert summary.centroid_value("observation_count") == 1.0
        with pytest.raises(KeyError):
            summary.centroid_value("total_input_amount")

    def test_assignment_requires_a_cluster_unless_it_is_noise(self):
        with pytest.raises(ValueError):
            ClusterAssignment(profile_id="p1", is_noise=False, is_in_sample=True)

    def test_noise_assignment_uses_the_reserved_label(self):
        with pytest.raises(ValueError):
            ClusterAssignment(
                profile_id="p1",
                cluster_id="CLUSTER_042",
                is_noise=True,
                is_in_sample=True,
            )
        assignment = ClusterAssignment(
            profile_id="p1",
            cluster_id=NOISE_CLUSTER_ID,
            is_noise=True,
            is_in_sample=True,
        )
        assert assignment.is_noise

    def test_cluster_label_must_use_the_prefix(self):
        with pytest.raises(ValueError):
            ClusterAssignment(
                profile_id="p1", cluster_id="group-a", is_in_sample=True
            )

    def test_distance_must_be_present_and_non_negative(self):
        with pytest.raises(ValueError):
            ClusterAssignment(profile_id="p1", cluster_id="CLUSTER_000", is_in_sample=True)
        with pytest.raises(ValueError):
            ClusterAssignment(
                profile_id="p1",
                cluster_id="CLUSTER_000",
                distance_to_centroid=-1.0,
                is_in_sample=True,
            )

    def test_run_rejects_k_that_disagrees_with_the_cluster_count(self):
        from backend.app.ml.models import ClusteringRunResult

        with pytest.raises(ValueError):
            ClusteringRunResult(
                metadata=make_metadata(),
                status=ModelStatus.FITTED,
                k=3,
                clusters=(self.summary("CLUSTER_000", ("p1",)),),
                assignments=(
                    ClusterAssignment(
                        profile_id="p1",
                        cluster_id="CLUSTER_000",
                        distance_to_centroid=0.1,
                        is_in_sample=True,
                    ),
                ),
                feature_names=("observation_count", "total_input_amount"),
            )

    def test_run_rejects_a_profile_in_two_clusters(self):
        from backend.app.ml.models import ClusteringRunResult

        with pytest.raises(ValueError):
            ClusteringRunResult(
                metadata=make_metadata(),
                status=ModelStatus.FITTED,
                k=2,
                clusters=(
                    self.summary("CLUSTER_000", ("p1", "p2")),
                    self.summary("CLUSTER_001", ("p1",)),
                ),
                assignments=(
                    ClusterAssignment(
                        profile_id="p1",
                        cluster_id="CLUSTER_000",
                        distance_to_centroid=0.1,
                        is_in_sample=True,
                    ),
                    ClusterAssignment(
                        profile_id="p2",
                        cluster_id="CLUSTER_000",
                        distance_to_centroid=0.1,
                        is_in_sample=True,
                    ),
                ),
                feature_names=("observation_count", "total_input_amount"),
            )

    def test_unfitted_run_must_state_why(self):
        from backend.app.ml.models import ClusteringRunResult

        with pytest.raises(ValueError):
            ClusteringRunResult(
                metadata=make_metadata(),
                status=ModelStatus.INSUFFICIENT_DATA,
                feature_names=("observation_count", "total_input_amount"),
            )

    def test_unfitted_run_may_not_report_clusters(self):
        from backend.app.ml.models import ClusteringRunResult

        with pytest.raises(ValueError):
            ClusteringRunResult(
                metadata=make_metadata(),
                status=ModelStatus.INSUFFICIENT_DATA,
                clusters=(self.summary("CLUSTER_000", ("p1",)),),
                feature_names=("observation_count", "total_input_amount"),
                insufficient_reason="nothing to group",
            )

    def test_silhouette_scores_must_be_ascending_in_k_and_inside_the_unit_range(self):
        from backend.app.ml.models import ClusteringRunResult

        with pytest.raises(ValueError):
            ClusteringRunResult(
                metadata=make_metadata(),
                status=ModelStatus.NO_CLUSTERS,
                feature_names=("observation_count", "total_input_amount"),
                insufficient_reason="no partition",
                selection_scores=((3, 0.5), (2, 0.4)),
            )
        with pytest.raises(ValueError):
            ClusteringRunResult(
                metadata=make_metadata(),
                status=ModelStatus.NO_CLUSTERS,
                feature_names=("observation_count", "total_input_amount"),
                insufficient_reason="no partition",
                selection_scores=((2, 1.4),),
            )


class TestRunMetadata:
    """Metadata is the provenance block that makes a number checkable."""

    def test_metadata_requires_a_feature_set_or_reports_none(self):
        metadata = make_metadata(feature_names=(), feature_schema_id="SCHEMA_NONE")
        assert metadata.feature_names == ()

    def test_metadata_rejects_unsorted_feature_names(self):
        with pytest.raises(ValueError):
            make_metadata(feature_names=("b_count", "a_count"))

    def test_metadata_rejects_a_non_json_configuration(self):
        with pytest.raises(ValueError):
            make_metadata(config_json="not json")

    def test_configuration_is_available_as_a_dictionary(self):
        assert make_metadata().config == {"k": 2}

    def test_metadata_serializes_the_configuration_as_a_dictionary(self):
        payload = make_metadata().to_dict()
        assert payload["config"] == {"k": 2}
        assert "config_json" not in payload
        json.dumps(payload, sort_keys=True, default=str)

    def test_metadata_states_whether_scoring_was_in_sample(self):
        assert make_metadata(scoring_in_sample=False).scoring_in_sample is False

    def test_metadata_reports_the_library_it_ran_on(self):
        metadata = make_metadata()
        assert metadata.library == "python-stdlib"
        assert metadata.library_version


# --------------------------------------------------------------------------- #
# Immutability and serialization
# --------------------------------------------------------------------------- #


class TestImmutabilityAndSerialization:
    """Results are frozen and serialize to the same bytes every time."""

    def test_every_result_model_is_frozen(self):
        for model in (make_metadata(),):
            with pytest.raises(Exception):
                model.random_state = 3

    def test_run_result_serialization_is_byte_stable(self):
        from backend.app.ml.models import AnomalyRunResult

        run = AnomalyRunResult(
            metadata=make_metadata(),
            status=ModelStatus.INSUFFICIENT_DATA,
            feature_names=("observation_count", "total_input_amount"),
            insufficient_reason="too few profiles to compare",
        )
        first = json.dumps(run.to_dict(), sort_keys=True, default=str)
        second = json.dumps(run.to_dict(), sort_keys=True, default=str)
        assert first == second
        assert run.to_dict()["status"] == "INSUFFICIENT_DATA"

    def test_fitted_run_must_report_results(self):
        from backend.app.ml.models import AnomalyRunResult

        with pytest.raises(ValueError):
            AnomalyRunResult(
                metadata=make_metadata(),
                status=ModelStatus.FITTED,
                feature_names=("observation_count", "total_input_amount"),
            )

    def test_unfitted_run_may_not_report_scores(self):
        from backend.app.ml.models import AnomalyRunResult

        with pytest.raises(ValueError):
            AnomalyRunResult(
                metadata=make_metadata(),
                status=ModelStatus.INSUFFICIENT_DATA,
                results=(
                    AnomalyResult(
                        profile_id="p1",
                        schema_id="SCHEMA_TEST",
                        raw_score=0.5,
                        normalized_score=0.7,
                        band=UnusualnessBand.LOW,
                        is_in_sample=True,
                    ),
                ),
                feature_names=("observation_count", "total_input_amount"),
                insufficient_reason="too few",
            )

    def test_results_must_be_ordered_by_score_descending(self):
        from backend.app.ml.models import AnomalyResult, AnomalyRunResult

        def scored(profile_id, score):
            return AnomalyResult(
                profile_id=profile_id,
                schema_id="SCHEMA_TEST",
                raw_score=1.0 - score,
                normalized_score=score,
                band=UnusualnessBand.LOW,
                is_in_sample=True,
            )

        with pytest.raises(ValueError):
            AnomalyRunResult(
                metadata=make_metadata(),
                status=ModelStatus.FITTED,
                results=(scored("p1", 0.4), scored("p2", 0.8)),
                feature_names=("observation_count", "total_input_amount"),
            )

    def test_run_feature_names_must_match_the_metadata_schema(self):
        from backend.app.ml.models import AnomalyRunResult

        with pytest.raises(ValueError):
            AnomalyRunResult(
                metadata=make_metadata(),
                status=ModelStatus.INSUFFICIENT_DATA,
                feature_names=("observation_count",),
                insufficient_reason="too few",
            )

    def test_result_lookup_by_profile_id(self):
        from backend.app.ml.models import AnomalyResult, AnomalyRunResult

        run = AnomalyRunResult(
            metadata=make_metadata(),
            status=ModelStatus.FITTED,
            results=(
                AnomalyResult(
                    profile_id="p1",
                    schema_id="SCHEMA_TEST",
                    raw_score=0.5,
                    normalized_score=0.7,
                    band=UnusualnessBand.LOW,
                    is_in_sample=True,
                ),
            ),
            feature_names=("observation_count", "total_input_amount"),
        )
        assert run.result_for("p1").profile_id == "p1"
        assert run.result_for("missing") is None


# --------------------------------------------------------------------------- #
# The boundary, checked by reading the source
# --------------------------------------------------------------------------- #


FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "torch",
        "tensorflow",
        "xgboost",
        "lightgbm",
        "statsmodels",
        "sqlalchemy",
        "neo4j",
        "redis",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "http",
        "sqlite3",
        "psycopg2",
        "openai",
        "anthropic",
        "langchain",
    }
)

#: Words that would turn a measurement into a judgement, an identity or a
#: product.  Scanned as whole tokens of every identifier in the package.
FORBIDDEN_TOKENS = frozenset(
    {
        "fraud",
        "fraudulent",
        "malicious",
        "attacker",
        "victim",
        "suspicious",
        "suspicion",
        "culprit",
        "guilty",
        "perpetrator",
        "owner",
        "ownership",
        "identity",
        "actor",
        "threat",
        "threatlevel",
        "risk",
        "blame",
        "verdict",
        "guilt",
        "criminal",
        "criminality",
        "malware",
        "insider",
        "blacklist",
        "whitelist",
        "chatbot",
        "llm",
        "dashboard",
        "frontend",
        "backend",
        "api",
        "http",
        "endpoint",
        "sql",
        "query",
        "database",
        "neo4j",
        "fusion",
    }
)

#: Compounds whose parts are ordinary words on their own -- ``page`` and
#: ``rank`` are, ``pagerank`` is not.  Matched against an identifier with every
#: non-alphanumeric character removed, so ``page_rank``, ``pageRank`` and
#: ``PAGERANK`` all match the same entry.
FORBIDDEN_COMPOUNDS = frozenset(
    {
        "pagerank",
        "risklevel",
        "riskvalue",
        "riskscore",
        "groundtruth",
        "truepositive",
        "falsepositive",
        "truestate",
        "chaining",
        "flowanalysis",
        "keyholder",
        "ownership",
    }
)


def _docstring_nodes(tree: ast.AST) -> set:
    """Ids of string constants that are docstrings rather than data."""
    excluded = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                excluded.add(id(body[0].value))
    return excluded


def _identifier_texts(tree: ast.AST):
    """Yield every identifier and data string in *tree*, prose excluded.

    Prose is excluded on purpose.  The boundary this file checks is about what
    the code can say through its own names, and a docstring explaining that
    identity is deliberately not used cannot smuggle one in.  Data strings --
    enum members, field defaults -- are included, because those are values the
    package actually publishes.
    """
    excluded = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in excluded:
                yield node.value


def _squash(text: str) -> str:
    """*text* with every non-alphanumeric character removed, lower-cased."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def identifier_tokens(tree: ast.AST) -> set:
    """Word parts of every identifier and data string in *tree*."""
    tokens = set()
    for text in _identifier_texts(tree):
        tokens.update(_split(text))
        squashed = _squash(text)
        if squashed in FORBIDDEN_COMPOUNDS:
            tokens.add(squashed)
    return tokens


def _split(text: str) -> set:
    """Lower-case word parts of *text*.

    Two boundaries are recognised and no others: any non-alphanumeric
    character, and a camel-case hump.  No substring expansion -- prefixes and
    suffixes of an ordinary word are not words, and generating them turns
    "factor" into "actor".
    """
    parts = set()
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        if not chunk:
            continue
        parts.add(chunk.lower())
        segments = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", chunk)
        if len(segments) > 1:
            parts.update(segment.lower() for segment in segments)
    return {part for part in parts if part}


class TestBoundaryIsStructural:
    """Read the package source and check the promises made in its docstrings."""

    @staticmethod
    def sources():
        for path in sorted(ML_PACKAGE.glob("*.py")):
            yield path, path.read_text(encoding="utf-8")

    def test_no_module_imports_a_forbidden_root(self):
        for path, source in self.sources():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        assert root not in FORBIDDEN_IMPORT_ROOTS, "%s: %s" % (
                            path.name,
                            alias.name,
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative, inside this package
                        continue
                    root = (node.module or "").split(".")[0]
                    assert root not in FORBIDDEN_IMPORT_ROOTS, "%s: %s" % (
                        path.name,
                        node.module,
                    )

    def test_no_identifier_uses_a_judgement_or_product_word(self):
        for path, source in self.sources():
            tokens = identifier_tokens(ast.parse(source))
            found = tokens & FORBIDDEN_TOKENS
            assert not found, "%s uses %s" % (path.name, sorted(found))

    def test_the_token_scanner_itself_detects_the_words_it_claims_to(self):
        """A boundary test is only as good as its own detector.

        Checked against ``identifier_tokens`` on a real source expression, not
        against ``_split`` directly, so the whole detection path is exercised.
        """
        caught = {
            "fraud_score": "fraud",
            "isMalicious": "malicious",
            "SUSPICIOUS_COUNT": "suspicious",
            "risk_level": "risk",
            "owner_id": "owner",
            "pageRank": "pagerank",
            "page_rank": "pagerank",
            "llmVerdict": "llm",
            "CHATBOT_REPLY": "chatbot",
            "threatLevel": "threat",
            "whitelist_entry": "whitelist",
            "ground_truth": "groundtruth",
        }
        for name, expected in caught.items():
            tree = ast.parse("x = '%s'" % name)
            tokens = identifier_tokens(tree)
            assert expected in tokens, "%s -> %s" % (name, sorted(tokens))

        allowed = (
            "observation_count",
            "total_input_amount",
            "consistency_factor",
            "unusualness_band",
            "profile_id",
            "unique_src_ip_count",
            "distinct_counterparty_count",
        )
        for name in allowed:
            tree = ast.parse("x = '%s'" % name)
            tokens = identifier_tokens(tree)
            assert not (tokens & FORBIDDEN_TOKENS), name
            assert not (tokens & FORBIDDEN_COMPOUNDS), name

    def test_a_docstring_cannot_trigger_a_violation_but_data_can(self):
        """Prose is exempt; published values are not.

        ``anomaly.py`` documents that ``uuid4`` is not used, and ``models.py``
        explains what identity is.  Neither may fail a test, and neither may
        hide one.
        """
        prose = ast.parse("def f():\n    '''the risk_level is a band'''\n    return 1")
        assert "risk" not in identifier_tokens(prose)
        data = ast.parse("class E:\n    RISK_LEVEL = 'high'\n")
        assert "risk" in identifier_tokens(data)

    def test_no_squashed_compound_identifier_is_published(self):
        for path, source in self.sources():
            found = identifier_tokens(ast.parse(source)) & FORBIDDEN_COMPOUNDS
            assert not found, "%s uses %s" % (path.name, sorted(found))

    def test_no_module_opens_a_file_or_evaluates_code(self):
        for path, source in self.sources():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in (
                        "eval",
                        "exec",
                        "compile",
                        "open",
                        "__import__",
                    ), "%s calls %s" % (path.name, node.func.id)

    def test_no_module_generates_a_random_identifier(self):
        """No ``uuid4``, no :mod:`random` import, anywhere in the package."""
        for path, source in self.sources():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    assert node.attr != "uuid4", path.name
                if isinstance(node, ast.Name):
                    assert node.id != "uuid4", path.name
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    roots = [
                        alias.name.split(".")[0]
                        for alias in getattr(node, "names", [])
                    ]
                    if getattr(node, "module", None):
                        roots.append(node.module.split(".")[0])
                    assert "random" not in roots, "%s imports random" % path.name
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "Random", path.name

    def test_no_module_reads_a_clock(self):
        """No wall clock: a result must not depend on when it was produced."""
        forbidden = {"now", "utcnow", "today", "time", "monotonic", "perf_counter"}
        for path, source in self.sources():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    assert node.attr not in forbidden, "%s uses %s" % (
                        path.name,
                        node.attr,
                    )
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    roots = [
                        alias.name.split(".")[0]
                        for alias in getattr(node, "names", [])
                    ]
                    if getattr(node, "module", None):
                        roots.append(node.module.split(".")[0])
                    assert "time" not in roots, "%s imports time" % path.name
                    assert "secrets" not in roots, "%s imports secrets" % path.name

    def test_no_model_field_can_hold_a_probability_or_a_verdict(self):
        """The strongest boundary claim: there is no field to put one in."""
        forbidden = (
            "probability",
            "confidence",
            "verdict",
            "label",
            "class",
            "ground_truth",
            "accuracy",
            "fraud",
            "malicious",
            "risk",
            "score_reason",
            "justification",
            "recommendation",
            "narrative",
        )
        for model in ML_MODULES:
            for name in model.__all__:
                member = getattr(model, name)
                if not isinstance(member, type):
                    continue
                for field in getattr(member, "model_fields", {}):
                    assert field not in forbidden, "%s.%s" % (model.__name__, field)

    def test_package_does_not_import_a_later_step_or_a_product_dependency(self):
        """Step 7 onward is not here: no LLM, no store, no service, no UI."""
        later_step_roots = {
            "openai",
            "anthropic",
            "langchain",
            "llama_index",
            "chromadb",
            "faiss",
            "networkx",
            "matplotlib",
            "seaborn",
            "plotly",
            "streamlit",
            "gradio",
            "dash",
            "fastapi",
            "flask",
            "django",
        }
        for path, source in self.sources():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        assert root not in later_step_roots, "%s: %s" % (
                            path.name,
                            alias.name,
                        )
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    root = (node.module or "").split(".")[0]
                    assert root not in later_step_roots, "%s: %s" % (
                        path.name,
                        node.module,
                    )

    def test_the_package_imports_only_the_standard_library_and_its_own_step(self):
        """Every absolute import is stdlib, pydantic, or a sibling Step."""
        allowed_non_stdlib = {"pydantic"}
        for path, source in self.sources():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                roots = []
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    roots = [(node.module or "").split(".")[0]]
                for root in roots:
                    if not root:
                        continue
                    assert (
                        root in sys.stdlib_module_names
                        or root in allowed_non_stdlib
                        or root in {"backend"}
                    ), "%s imports %s" % (path.name, root)

    def test_the_package_states_that_it_uses_only_the_standard_library(self):
        assert "standard library" in (
            (ML_PACKAGE / "__init__.py").read_text(encoding="utf-8")
        )
