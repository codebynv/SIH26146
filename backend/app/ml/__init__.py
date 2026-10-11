"""Offline machine learning over Step 5 behavioural profiles (SIH26146 - Step 6).

Step 5 measured behaviour and stopped.  Step 6 computes two things from those
measurements, and only these two things:

Unusualness
    How far a profile sits from the population it is compared with, in a stated
    numeric geometry.  :class:`IsolationForestAnomalyDetector` fits the
    population and
    :meth:`FittedAnomalyModel.score <backend.app.ml.anomaly.FittedAnomalyModel.score>`
    reports a score; :mod:`backend.app.ml.explanations` describes the columns
    that place it there.

Grouping
    Which profiles sit near the same centroid.
    :class:`KMeansBehaviorClusterer` fits a partition and
    :meth:`FittedClusteringModel.predict_cluster <backend.app.ml.clustering.FittedClusteringModel.predict_cluster>`
    places a profile in it, under labels derived from content so they do not
    depend on input order.

Both use only the standard library.  That is a deliberate consequence of the
project's dependency policy rather than a claim that no numerical library
exists: the isolated-environment check found none declared in
``backend/requirements.txt`` and none installed, and installing one is not
permitted here.  So the two algorithms are implemented directly, with their
formulas written out in the module docstrings, and the exact estimator version
travels in every result.

What this package will not do
-----------------------------
It does not classify a profile, and it cannot: there is no field in any model
here for a label, a verdict, a score of blame, or an identity.  Unusualness and
group membership are positions in a measurement, and nothing in Step 6 may be
read as more than that.  There is no database, no service, no network call and
no model download: input is Step 5 profile objects, output is Step 6 result
objects, and both are frozen.

What is deliberately absent
---------------------------
No class weights, no probability, no ground truth, no accuracy claim.  A profile
was never labelled, so no accuracy can be computed, and inventing one would be
a fabricated measurement.  ``unusual`` is reported as a position and a band,
``cluster`` as membership and a distance, and an explanation as a column
deviation from a fitted reference -- each with its method stated.
"""

from __future__ import annotations

from .anomaly import (
    ANOMALY_ALGORITHM_VERSION,
    FittedAnomalyModel,
    IsolationForestAnomalyDetector,
    average_unsuccessful_path_length,
    band_for_score,
)
from .clustering import (
    CLUSTERING_ALGORITHM_VERSION,
    FittedClusteringModel,
    KMeansBehaviorClusterer,
    silhouette_score,
)
from .explanations import (
    EXPLANATION_METHOD_DETAIL,
    explain_profile,
    explain_profile_vector,
)
from .models import (
    AnomalyAlgorithm,
    AnomalyExplanation,
    AnomalyFeatureContribution,
    AnomalyModelConfig,
    AnomalyResult,
    AnomalyRunResult,
    ClusterAssignment,
    ClusterModelConfig,
    ClusterSummary,
    ClusteringAlgorithm,
    ClusteringRunResult,
    DeviationDirection,
    ExcludedFeature,
    ExclusionReason,
    ExplanationMethod,
    FeatureMatrix,
    FeatureSchema,
    ImputationStrategy,
    MLFeatureVector,
    ModelRunMetadata,
    ModelStatus,
    NOISE_CLUSTER_ID,
    ScalingMethod,
    UnusualnessBand,
    cluster_label,
    describe_explanation_methods,
    identifier_like_feature_name,
    is_numeric_value_kind,
)
from .preprocessing import (
    FittedFeaturePreprocessor,
    MLPreprocessorError,
    PreprocessorConfig,
    ProfileFeatureVectorizer,
    derive_schema_id,
    median,
    median_absolute_deviation,
)

__all__ = [
    "ANOMALY_ALGORITHM_VERSION",
    "AnomalyAlgorithm",
    "AnomalyExplanation",
    "AnomalyFeatureContribution",
    "AnomalyModelConfig",
    "AnomalyResult",
    "AnomalyRunResult",
    "CLUSTERING_ALGORITHM_VERSION",
    "ClusterAssignment",
    "ClusterModelConfig",
    "ClusterSummary",
    "ClusteringAlgorithm",
    "ClusteringRunResult",
    "DeviationDirection",
    "EXPLANATION_METHOD_DETAIL",
    "ExcludedFeature",
    "ExclusionReason",
    "ExplanationMethod",
    "FeatureMatrix",
    "FeatureSchema",
    "FittedAnomalyModel",
    "FittedClusteringModel",
    "FittedFeaturePreprocessor",
    "ImputationStrategy",
    "IsolationForestAnomalyDetector",
    "KMeansBehaviorClusterer",
    "MLFeatureVector",
    "MLPreprocessorError",
    "ModelRunMetadata",
    "ModelStatus",
    "NOISE_CLUSTER_ID",
    "PreprocessorConfig",
    "ProfileFeatureVectorizer",
    "ScalingMethod",
    "UnusualnessBand",
    "average_unsuccessful_path_length",
    "band_for_score",
    "cluster_label",
    "derive_schema_id",
    "describe_explanation_methods",
    "explain_profile",
    "explain_profile_vector",
    "identifier_like_feature_name",
    "is_numeric_value_kind",
    "median",
    "median_absolute_deviation",
    "silhouette_score",
]
