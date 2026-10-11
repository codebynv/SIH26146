"""Typed, immutable offline machine-learning models (SIH26146 - Step 6).

Step 5 measured behaviour and deliberately stopped short of meaning.  Step 6 is
where interpretation begins, and this module is the boundary where it is
written down.  Every field below can hold a *number about a profile*; none of
them can hold a verdict about a person, a wallet, an address, an account or an
actor, because no such field exists here to put one in.

The vocabulary, in order
------------------------
:class:`FeatureSchema`
    Which Step 5 feature names became vector columns, in which order, under
    which version, and -- just as important -- which names were *left out* and
    why.  An exclusion is recorded, never silent.

:class:`MLFeatureVector` / :class:`FeatureMatrix`
    Numbers ready for arithmetic, and the same numbers as a rectangular
    population.  A profile identifier appears only in the row label, never as a
    column.

:class:`AnomalyModelConfig` / :class:`ClusterModelConfig`
    Every choice that can change a number: algorithm, seed, sample count,
    iteration budget, band boundaries.  Frozen, so a fitted model can be
    replayed exactly.

:class:`AnomalyResult` / :class:`AnomalyRunResult`
    How unusual a profile is relative to the population it was fitted on, and
    the metadata a reader needs to judge that number.

:class:`ClusterAssignment` / :class:`ClusterSummary` / :class:`ClusteringRunResult`
    Which profiles sit together under a stated geometry, how far each is from
    its centroid, and a content-derived cluster label that does not depend on
    input order.

:class:`ModelRunMetadata`
    The shared provenance block: algorithm and version, model id, feature
    schema, scaling and imputation, seed, and both population sizes.

What "unusual" and "cluster" mean here
--------------------------------------
``unusual`` means "sits far from the majority of the population it was fitted
on, in a documented numeric geometry".  It is a position, not a property of the
entity, and it says nothing about intent.  A profile can be unusual and sit
inside a cluster, or sit alone near a cluster edge; the two signals are
computed independently and are not allowed to be merged into one verdict.

Determinism
-----------
Every identifier in this module is derived from content with SHA-256.  There is
no :func:`uuid.uuid4`, no clock and no :mod:`random` module anywhere in Step 6:
the same population and the same configuration produce byte-identical
serialization on any machine, in any Python version, in any order.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import uuid
from enum import Enum
from typing import Any, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AnomalyAlgorithm",
    "AnomalyExplanation",
    "AnomalyFeatureContribution",
    "AnomalyModelConfig",
    "AnomalyResult",
    "AnomalyRunResult",
    "ClusterAssignment",
    "ClusterModelConfig",
    "ClusterSummary",
    "ClusteringAlgorithm",
    "ClusteringRunResult",
    "DeviationDirection",
    "ExcludedFeature",
    "ExclusionReason",
    "ExplanationMethod",
    "FeatureMatrix",
    "FeatureSchema",
    "ImputationStrategy",
    "MLFeatureVector",
    "ML_FEATURE_NAMESPACE",
    "ModelRunMetadata",
    "ModelStatus",
    "NOISE_CLUSTER_ID",
    "ScalingMethod",
    "UnusualnessBand",
    "cluster_label",
    "describe_explanation_methods",
    "identifier_like_feature_name",
    "is_numeric_value_kind",
]

#: Namespace for the UUIDv5 identifiers Step 6 derives from content.  Fixed
#: forever: changing it would change every derived id and silently break
#: comparison against previously written results.
ML_FEATURE_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

#: Cluster label reserved for a profile that belongs to no cluster.  K-means,
#: the algorithm selected here, assigns every profile it was fitted on, so this
#: label is part of the *vocabulary* rather than a prediction; it exists so a
#: later density-based algorithm can express "belongs to nothing" without a
#: breaking change.  See :class:`ClusterAssignment`.
NOISE_CLUSTER_ID = "CLUSTER_NOISE"

#: A feature name that would smuggle an identifier into the vector as a number.
#: Step 5 features are *measurements*, and measurements are named for what was
#: measured (``unique_src_ip_count``), never for the identifier itself
#: (``src_ip``).  This pattern refuses the second kind by name, in both the
#: suffixed form and the bare form, so a future Step 5 feature cannot quietly
#: introduce leakage.
_IDENTIFIER_LIKE_SUFFIXES = (
    "id",
    "ids",
    "uuid",
    "uids",
    "txid",
    "txids",
    "ip",
    "ips",
    "address",
    "addresses",
    "file",
    "files",
    "path",
    "paths",
    "label",
    "labels",
)

_IDENTIFIER_LIKE_NAME_RE = re.compile(
    r"^(?:%s)$|_(?:%s)$" % ("|".join(_IDENTIFIER_LIKE_SUFFIXES), "|".join(_IDENTIFIER_LIKE_SUFFIXES))
)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Value kinds Step 5 defines that carry a directly comparable magnitude.
NUMERIC_VALUE_KINDS: frozenset[str] = frozenset({"COUNT", "DECIMAL"})


def _require_non_blank(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-blank string" % field)
    return value


def _require_sorted_unique(values: Tuple[str, ...], field: str) -> Tuple[str, ...]:
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("%s entries must be non-blank strings" % field)
    if len(set(values)) != len(values):
        raise ValueError("%s entries must be unique" % field)
    if list(values) != sorted(values):
        raise ValueError("%s entries must be in ascending order" % field)
    return values


def _require_all_finite(values: Tuple[float, ...], field: str) -> Tuple[float, ...]:
    for item in values:
        if item is None or not math.isfinite(float(item)):
            raise ValueError("%s entries must be finite numbers" % field)
    return values


def canonical_json(payload: Any) -> str:
    """Serialize *payload* to a stable string.

    Sorted keys, no insignificant whitespace, no locale formatting.  Two equal
    configurations must produce equal strings, because configuration equality
    is what decides whether two runs are comparable.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True
    )


def _short_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def identifier_like_feature_name(name: str) -> bool:
    """Whether *name* denotes an identifier rather than a measurement.

    Used twice, deliberately.  :class:`FeatureSchema` refuses such a column
    outright, and the adapter records it as an exclusion with a reason, so a
    name of this shape can never reach the vector -- whether it arrives as a
    schema or as a Step 5 feature someone extended.
    """
    return bool(_IDENTIFIER_LIKE_NAME_RE.search(name))


def is_numeric_value_kind(kind: str) -> bool:
    """Whether a Step 5 value kind carries a comparable magnitude.

    Only ``COUNT`` and ``DECIMAL`` qualify.  ``TIMESTAMP`` is an instant, which
    has no natural origin: turning one into a number would invent an epoch and a
    timezone policy that no reader asked for.  ``LABEL`` and ``LABEL_SET`` are
    categories, and assigning them integers would invent an ordering -- it
    would silently claim that ``"IN"`` is "closer" to ``"US"`` than to
    ``"ZZ"``.
    """
    return kind in NUMERIC_VALUE_KINDS


class AnomalyAlgorithm(str, Enum):
    """How an unusualness number was produced.

    ``ISOLATION_FOREST`` measures how quickly a profile separates from a
    reference sample under random axis-aligned splits.  It is a *position*
    statistic with no notion of a class, a label or an intent.
    """

    ISOLATION_FOREST = "ISOLATION_FOREST"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class ClusteringAlgorithm(str, Enum):
    """How profiles were grouped.

    ``K_MEANS`` partitions a numeric geometry into ``k`` groups of similar
    coordinates.  It does not name, describe or rank the groups it finds.
    """

    K_MEANS = "K_MEANS"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class ScalingMethod(str, Enum):
    """How columns were brought onto a comparable range."""

    NONE = "NONE"
    ROBUST_MAD = "ROBUST_MAD"
    STANDARD_DEVIATION = "STANDARD_DEVIATION"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class ImputationStrategy(str, Enum):
    """How a measurement that was never available was filled in.

    Only one strategy exists on purpose.  An absent measurement is replaced by
    the median of the *observed* values for that same column in the fitted
    population, which is the neutral choice: it adds no direction and no
    magnitude of its own.  The substituted columns are reported per row, so a
    reader can always discount them.
    """

    FITTED_MEDIAN = "FITTED_MEDIAN"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class ExclusionReason(str, Enum):
    """Why a Step 5 feature did not become a vector column."""

    CATEGORICAL_LABEL = "CATEGORICAL_LABEL"
    CATEGORICAL_LABEL_SET = "CATEGORICAL_LABEL_SET"
    INSTANT_TIMESTAMP = "INSTANT_TIMESTAMP"
    IDENTIFIER_LIKE_NAME = "IDENTIFIER_LIKE_NAME"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class ModelStatus(str, Enum):
    """State of a fit attempt.

    ``NOT_FITTED``
        No model exists yet, or scoring was attempted before fitting.
    ``INSUFFICIENT_DATA``
        The population is too small for the selected algorithm, and no number
        was produced.  Absent by design: a score computed from one profile is
        the profile's own distance to itself, which is not evidence.
    ``NO_CLUSTERS``
        A clustering fit completed but the configuration cannot produce a
        partition -- for example ``k_max`` below two.
    ``FITTED``
        A model was produced and the numbers in its results are real.
    """

    NOT_FITTED = "NOT_FITTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_CLUSTERS = "NO_CLUSTERS"
    FITTED = "FITTED"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class UnusualnessBand(str, Enum):
    """A fixed operating point on the normalized unusualness score.

    The bands are *thresholds chosen in advance*, not fitted cut points and
    not judgements.  ``LOW`` means "not separated from this population by this
    method", and ``EXTREME`` means "separated most strongly of anything in this
    population".  Neither says anything about what caused the separation or
    what ought to be done about it.
    """

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value

class DeviationDirection(str, Enum):
    """Which side of the fitted reference a profile fell on."""

    ABOVE_REFERENCE = "ABOVE_REFERENCE"
    BELOW_REFERENCE = "BELOW_REFERENCE"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class ExplanationMethod(str, Enum):
    """Where an explanation's numbers come from.

    ``MODEL_NATIVE``
        The model itself attributes the score to features.  Isolation forest
        does not, so nothing in Step 6 may claim this value.
    ``STATISTICAL_REFERENCE``
        Each column is compared with that column's fitted reference
        distribution.  This is what Step 6 produces, and it is explicitly a
        deviation description, not a decomposition of the model score.
    ``DERIVED_APPROXIMATION``
        A post-hoc approximation to a model-native attribution, for algorithms
        that offer none.  It must be labelled as an approximation.  Step 6
        does not produce this value either: with no native attribution to
        approximate, an approximation would be decoration.
    """

    MODEL_NATIVE = "MODEL_NATIVE"
    STATISTICAL_REFERENCE = "STATISTICAL_REFERENCE"
    DERIVED_APPROXIMATION = "DERIVED_APPROXIMATION"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


def describe_explanation_methods() -> dict:
    """Return, for each explanation method, whether Step 6 can produce it.

    Exposed so the claim is auditable from code rather than from prose: the
    selected anomaly model offers no per-feature attribution, so
    :data:`ExplanationMethod.MODEL_NATIVE` and
    :data:`ExplanationMethod.DERIVED_APPROXIMATION` are documented as
    unavailable rather than quietly filled with invented numbers.
    """
    return {
        ExplanationMethod.MODEL_NATIVE.value: (
            "UNAVAILABLE: the selected isolation-forest model provides no "
            "per-feature attribution."
        ),
        ExplanationMethod.STATISTICAL_REFERENCE.value: (
            "AVAILABLE: per-column deviation from the fitted reference "
            "distribution."
        ),
        ExplanationMethod.DERIVED_APPROXIMATION.value: (
            "UNAVAILABLE: with no native attribution available, a post-hoc "
            "approximation would be an invented decomposition."
        ),
    }


def cluster_label(index: int) -> str:
    """Return the canonical label for the cluster at *index*.

    Labels are dense and zero-padded so that lexical order matches numeric
    order, and they are assigned only after canonical re-ordering -- see
    :mod:`backend.app.ml.clustering`.  The label carries no meaning beyond
    "this group, in this run".
    """
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("cluster index must be a non-negative integer")
    if index > 9999:
        raise ValueError("cluster index must not exceed 9999")
    return "CLUSTER_%03d" % index


class ExcludedFeature(BaseModel):
    """A Step 5 feature that deliberately did not become a vector column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Step 5 feature name.")
    value_kind: str = Field(description="Step 5 value kind that was refused.")
    reason: ExclusionReason = Field(description="Why the column was refused.")
    detail: str = Field(description="Human-readable justification.")

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        return _require_non_blank(value, "name")

    @field_validator("value_kind")
    @classmethod
    def _check_value_kind(cls, value: str) -> str:
        return _require_non_blank(value, "value_kind")

    @field_validator("detail")
    @classmethod
    def _check_detail(cls, value: str) -> str:
        return _require_non_blank(value, "detail")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value_kind": self.value_kind,
            "reason": self.reason.value,
            "detail": self.detail,
        }


class FeatureSchema(BaseModel):
    """The ordered column set of a vector, with its exclusions recorded.

    Column order is alphabetical by Step 5 feature name.  Alphabetical order
    is chosen deliberately: it is independent of the order Step 5 happened to
    emit features in, of the number of profiles, and of input shuffling, so the
    same set of features always yields the same column positions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: str = Field(description="Derived identifier of this column set.")
    feature_names: Tuple[str, ...] = Field(description="Vector columns, ascending.")
    excluded_features: Tuple[ExcludedFeature, ...] = Field(
        default=(), description="Step 5 features left out, with reasons."
    )
    scaling: ScalingMethod = Field(
        default=ScalingMethod.ROBUST_MAD, description="Scaling applied to columns."
    )
    imputation: ImputationStrategy = Field(
        default=ImputationStrategy.FITTED_MEDIAN,
        description="Rule used for measurements that were unavailable.",
    )

    @field_validator("feature_names")
    @classmethod
    def _check_feature_names(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if not value:
            raise ValueError("feature_names must not be empty")
        _require_sorted_unique(value, "feature_names")
        for name in value:
            if not _NAME_RE.match(name):
                raise ValueError("feature name %r is not lower snake_case" % name)
            if _IDENTIFIER_LIKE_NAME_RE.search(name):
                raise ValueError(
                    "feature name %r looks like an identifier and is refused as a "
                    "numeric feature" % name
                )
        return value

    @field_validator("excluded_features")
    @classmethod
    def _check_excluded(cls, value: Tuple[ExcludedFeature, ...]) -> Tuple[ExcludedFeature, ...]:
        excluded = tuple(item.name for item in value)
        _require_sorted_unique(excluded, "excluded_features")
        return value

    @model_validator(mode="after")
    def _check_disjoint(self) -> "FeatureSchema":
        overlap = sorted(
            set(self.feature_names)
            & {item.name for item in self.excluded_features}
        )
        if overlap:
            raise ValueError(
                "a feature cannot be both a column and excluded: %s" % overlap
            )
        return self

    @property
    def dimension(self) -> int:
        """Number of vector columns."""
        return len(self.feature_names)

    def to_dict(self) -> dict:
        return {
            "schema_id": self.schema_id,
            "dimension": self.dimension,
            "feature_names": list(self.feature_names),
            "excluded_features": [item.to_dict() for item in self.excluded_features],
            "scaling": self.scaling.value,
            "imputation": self.imputation.value,
        }


class MLFeatureVector(BaseModel):
    """One profile as a fixed-width row of finite numbers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: str = Field(description="Schema the values were produced under.")
    feature_names: Tuple[str, ...] = Field(description="Vector columns, ascending.")
    values: Tuple[float, ...] = Field(description="Values aligned to feature_names.")
    imputed_feature_names: Tuple[str, ...] = Field(
        default=(),
        description="Columns filled from the fitted reference because the "
        "measurement was unavailable.",
    )

    @field_validator("feature_names")
    @classmethod
    def _check_feature_names(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if not value:
            raise ValueError("feature_names must not be empty")
        _require_sorted_unique(value, "feature_names")
        return value

    @field_validator("values")
    @classmethod
    def _check_values(cls, value: Tuple[float, ...]) -> Tuple[float, ...]:
        return _require_all_finite(value, "values")

    @field_validator("imputed_feature_names")
    @classmethod
    def _check_imputed(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return _require_sorted_unique(value, "imputed_feature_names")

    @model_validator(mode="after")
    def _check_consistency(self) -> "MLFeatureVector":
        if len(self.values) != len(self.feature_names):
            raise ValueError(
                "values length %d does not match feature_names length %d"
                % (len(self.values), len(self.feature_names))
            )
        unknown = sorted(set(self.imputed_feature_names) - set(self.feature_names))
        if unknown:
            raise ValueError("imputed columns are not in the schema: %s" % unknown)
        return self

    @property
    def dimension(self) -> int:
        """Number of vector columns."""
        return len(self.values)

    def value_of(self, feature_name: str) -> float:
        """Return the value of one column by name."""
        if feature_name not in self.feature_names:
            raise KeyError(feature_name)
        return self.values[self.feature_names.index(feature_name)]

    def to_dict(self) -> dict:
        return {
            "schema_id": self.schema_id,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "imputed_feature_names": list(self.imputed_feature_names),
        }


class FeatureMatrix(BaseModel):
    """A population of vectors, rows aligned to ``profile_ids``.

    Profile identifiers are row labels only.  They are deliberately *not* a
    column: a distance computed with an identifier column would be a statement
    about how similar two names are.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: str = Field(description="Schema shared by every row.")
    feature_names: Tuple[str, ...] = Field(description="Vector columns, ascending.")
    profile_ids: Tuple[str, ...] = Field(description="Row labels, ascending.")
    rows: Tuple[Tuple[float, ...], ...] = Field(description="Row vectors.")

    @field_validator("feature_names")
    @classmethod
    def _check_feature_names(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if not value:
            raise ValueError("feature_names must not be empty")
        _require_sorted_unique(value, "feature_names")
        return value

    @field_validator("profile_ids")
    @classmethod
    def _check_profile_ids(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return _require_sorted_unique(value, "profile_ids")

    @field_validator("rows")
    @classmethod
    def _check_rows(cls, value: Tuple[Tuple[float, ...], ...]) -> Tuple[Tuple[float, ...], ...]:
        return tuple(_require_all_finite(row, "rows") for row in value)

    @model_validator(mode="after")
    def _check_consistency(self) -> "FeatureMatrix":
        if len(self.rows) != len(self.profile_ids):
            raise ValueError(
                "rows length %d does not match profile_ids length %d"
                % (len(self.rows), len(self.profile_ids))
            )
        width = len(self.feature_names)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    "row %d has %d values but the schema has %d columns"
                    % (index, len(row), width)
                )
        return self

    @property
    def population_size(self) -> int:
        """Number of rows."""
        return len(self.rows)

    @property
    def dimension(self) -> int:
        """Number of columns."""
        return len(self.feature_names)

    def row_of(self, profile_id: str) -> Tuple[float, ...]:
        """Return the row labelled by *profile_id*."""
        if profile_id not in self.profile_ids:
            raise KeyError(profile_id)
        return self.rows[self.profile_ids.index(profile_id)]

    def to_dict(self) -> dict:
        return {
            "schema_id": self.schema_id,
            "feature_names": list(self.feature_names),
            "profile_ids": list(self.profile_ids),
            "rows": [list(row) for row in self.rows],
        }


class AnomalyModelConfig(BaseModel):
    """Every choice that can change an unusualness number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: AnomalyAlgorithm = AnomalyAlgorithm.ISOLATION_FOREST
    random_state: int = Field(default=0, ge=0, le=4294967295)
    n_estimators: int = Field(default=100, ge=1, le=4096)
    max_samples: int = Field(
        default=256,
        ge=2,
        le=65536,
        description="Reference subsample size per tree; the tree depth limit "
        "is derived from it.",
    )
    min_profiles: int = Field(
        default=2,
        ge=2,
        description="Below this population size no model is fitted and no "
        "score is produced.",
    )
    max_depth: Optional[int] = Field(
        default=None,
        ge=1,
        le=64,
        description="Depth limit per tree; None derives it from the "
        "subsample size.",
    )
    band_boundaries: Tuple[float, float, float] = Field(
        default=(0.5, 0.7, 0.9),
        description="Ascending cut points on the normalized score that split "
        "it into the four unusualness bands.",
    )
    explanation_limit: int = Field(
        default=5, ge=0, le=100, description="Features reported per explanation."
    )

    @field_validator("band_boundaries")
    @classmethod
    def _check_bands(cls, value: Tuple[float, float, float]) -> Tuple[float, float, float]:
        bounds = tuple(float(item) for item in value)
        for item in bounds:
            if not math.isfinite(item) or not 0.0 < item < 1.0:
                raise ValueError("band boundaries must lie strictly inside (0, 1)")
        if list(bounds) != sorted(bounds) or len(set(bounds)) != len(bounds):
            raise ValueError("band boundaries must be strictly ascending")
        return bounds  # type: ignore[return-value]

    def resolved_max_depth(self, subsample_size: int) -> int:
        """Depth limit for a tree fitted on *subsample_size* samples.

        Unbalanced binary search trees that isolate a uniform sample of ``psi``
        points have expected depth ``ceil(log2(psi))``; stopping there is what
        makes the path length a usable anomaly statistic.
        """
        if self.max_depth is not None:
            return self.max_depth
        return max(1, math.ceil(math.log2(max(subsample_size, 2))))

    @property
    def model_id(self) -> str:
        """Content-derived identifier: equal configuration, equal id."""
        payload = self.model_dump(mode="json")
        return "ANOMALY_%s_%s" % (
            self.algorithm.value,
            _short_digest(payload)[:16].upper(),
        )

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class ClusterModelConfig(BaseModel):
    """Every choice that can change a clustering result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.K_MEANS
    random_state: int = Field(default=0, ge=0, le=4294967295)
    k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Fixed number of clusters; None selects k by score.",
    )
    k_min: int = Field(default=2, ge=2)
    k_max: int = Field(default=8, ge=2)
    n_init: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Independent initializations; the lowest-inertia fit is "
        "kept, which is what makes k-means stable rather than seed-lucky.",
    )
    max_iterations: int = Field(default=100, ge=1, le=10000)
    tolerance: float = Field(
        default=1e-6, gt=0.0, lt=1.0, description="Centroid movement that stops a fit."
    )
    selection_metric: str = Field(
        default="SILHOUETTE",
        description="Score used to choose k when it is not supplied.",
    )
    min_profiles: int = Field(
        default=3,
        ge=3,
        description="Silhouette needs at least three points to be defined.",
    )

    @model_validator(mode="after")
    def _check_k_range(self) -> "ClusterModelConfig":
        if self.k is not None and not self.k_min <= self.k <= self.k_max:
            raise ValueError(
                "k=%d lies outside the permitted range [%d, %d]"
                % (self.k, self.k_min, self.k_max)
            )
        if self.k_min > self.k_max:
            raise ValueError("k_min must not exceed k_max")
        return self

    @property
    def model_id(self) -> str:
        """Content-derived identifier: equal configuration, equal id."""
        payload = self.model_dump(mode="json")
        return "CLUSTERING_%s_%s" % (
            self.algorithm.value,
            _short_digest(payload)[:16].upper(),
        )

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class AnomalyFeatureContribution(BaseModel):
    """One column's distance from its fitted reference.

    This is a *description* of where a profile sits, ranked by magnitude of
    deviation.  It is not a share of the anomaly score: the columns need not
    sum to anything, and no such decomposition is claimed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    feature_name: str = Field(description="Step 5 feature name.")
    observed_value: float = Field(description="Raw value used for this column.")
    reference_value: float = Field(description="Fitted reference for this column.")
    deviation: float = Field(
        description="Signed, scaled distance from the reference."
    )
    absolute_deviation: float = Field(ge=0.0, description="Magnitude of the deviation.")
    direction: DeviationDirection
    reference_unit: Optional[str] = Field(
        default=None, description="Unit reported by Step 5, if any."
    )

    @field_validator("feature_name")
    @classmethod
    def _check_feature_name(cls, value: str) -> str:
        return _require_non_blank(value, "feature_name")

    @field_validator("absolute_deviation")
    @classmethod
    def _check_magnitude(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("absolute_deviation must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "AnomalyFeatureContribution":
        if not math.isfinite(self.deviation) or not math.isfinite(self.observed_value):
            raise ValueError("deviation and observed_value must be finite")
        if not math.isfinite(self.reference_value):
            raise ValueError("reference_value must be finite")
        if not math.isclose(abs(self.deviation), self.absolute_deviation, rel_tol=1e-9):
            raise ValueError("absolute_deviation must match abs(deviation)")
        expected = (
            DeviationDirection.ABOVE_REFERENCE
            if self.deviation > 0
            else DeviationDirection.BELOW_REFERENCE
        )
        if self.direction is not expected:
            raise ValueError("direction must agree with the sign of deviation")
        return self

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "feature_name": self.feature_name,
            "observed_value": self.observed_value,
            "reference_value": self.reference_value,
            "deviation": self.deviation,
            "absolute_deviation": self.absolute_deviation,
            "direction": self.direction.value,
            "reference_unit": self.reference_unit,
        }


class AnomalyExplanation(BaseModel):
    """Ranked per-column deviations for one profile.

    The ``method`` field is mandatory and is not decorative: a reader must be
    able to tell a description of a number from a claim about a model, and from
    an approximation of one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(description="Profile the explanation describes.")
    method: ExplanationMethod
    method_detail: str = Field(
        description="Explicit statement of what the numbers are and are not."
    )
    contributions: Tuple[AnomalyFeatureContribution, ...] = Field(
        default=(), description="Ranked deviations, strongest first."
    )
    truncated: bool = Field(
        default=False, description="True when more columns deviated than were reported."
    )

    @field_validator("profile_id")
    @classmethod
    def _check_profile_id(cls, value: str) -> str:
        return _require_non_blank(value, "profile_id")

    @field_validator("method_detail")
    @classmethod
    def _check_detail(cls, value: str) -> str:
        return _require_non_blank(value, "method_detail")

    @model_validator(mode="after")
    def _check_ranking(self) -> "AnomalyExplanation":
        ranks = [item.rank for item in self.contributions]
        if ranks != list(range(1, len(self.contributions) + 1)):
            raise ValueError("contribution ranks must be 1..n in order")
        keys = [(-item.absolute_deviation, item.feature_name) for item in self.contributions]
        if keys != sorted(keys):
            raise ValueError(
                "contributions must be ordered by absolute deviation descending, "
                "then feature name ascending"
            )
        if self.method is ExplanationMethod.MODEL_NATIVE and not self.contributions:
            raise ValueError("a model-native explanation must report contributions")
        return self

    @property
    def top_contribution(self) -> Optional[AnomalyFeatureContribution]:
        """The strongest single deviation, if any was reported."""
        return self.contributions[0] if self.contributions else None

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "method": self.method.value,
            "method_detail": self.method_detail,
            "truncated": self.truncated,
            "contributions": [item.to_dict() for item in self.contributions],
        }


class ModelRunMetadata(BaseModel):
    """Provenance shared by anomaly and clustering results.

    Carried inside every run result so a single serialized object is enough to
    reproduce the numbers: no hidden fit state, no implicit preprocessing, no
    ambient decimal or floating-point context.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Content-derived model identifier.")
    algorithm: str = Field(description="Algorithm that produced the numbers.")
    algorithm_version: str = Field(description="Version of that algorithm here.")
    library: str = Field(
        default="python-stdlib",
        description="Implementation source. Step 6 uses only the standard "
        "library, so there is no third-party numerical runtime to pin.",
    )
    library_version: str = Field(
        default_factory=lambda: platform.python_version(),
        description="Interpreter version the run used.",
    )
    feature_schema_id: str = Field(
        description="Feature schema the run used; SCHEMA_NONE when no schema "
        "was fitted, which is the only case where feature_names is empty."
    )
    feature_names: Tuple[str, ...] = Field(
        description="Vector columns, ascending; empty only when no model was fitted."
    )
    scaling: Optional[ScalingMethod] = Field(
        default=None,
        description="Scaling the run used; None when no model was fitted, so "
        "no scaling was ever applied.",
    )
    imputation: Optional[ImputationStrategy] = Field(
        default=None,
        description="Imputation rule the run used; None when no model was fitted.",
    )
    random_state: int = Field(ge=0, le=4294967295)
    fitting_population_size: int = Field(ge=0)
    scoring_population_size: int = Field(ge=0)
    scoring_in_sample: bool = Field(
        description="True when the scored population is the fitted population. "
        "Stated rather than implied, because an in-sample number is a "
        "description of a population, not a generalization to unseen data.",
    )
    config_json: str = Field(
        description="Canonical JSON of the full configuration."
    )

    @field_validator("feature_names")
    @classmethod
    def _check_feature_names(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        # Deliberately permissive: an empty column set is the honest answer for
        # a run where no model was fitted, and the schema id says so.
        return _require_sorted_unique(value, "feature_names")

    @field_validator("config_json")
    @classmethod
    def _check_config_json(cls, value: str) -> str:
        _require_non_blank(value, "config_json")
        try:
            json.loads(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("config_json must be valid JSON: %s" % exc) from exc
        return value

    @property
    def config(self) -> dict:
        """The configuration as a plain dictionary."""
        return json.loads(self.config_json)

    def to_dict(self) -> dict:
        payload = self.model_dump(mode="json")
        payload["config"] = self.config
        del payload["config_json"]
        return payload


class AnomalyResult(BaseModel):
    """How far one profile sits from the population it was scored against.

    ``raw_score``
        Model-native and model-specific: mean path length through the fitted
        isolation trees divided by the expected path length of a failed search
        in an equally sized reference sample.  A *small* value means the profile
        separates quickly, and a short path is what "unusual" means for this
        algorithm.
    ``normalized_score``
        The documented monotone remap onto ``(0, 1]``, ``2 ** (-raw_score)``.
        It rises towards 1 as a profile separates more quickly and falls
        towards 0 as it takes longer to separate.  The ordering is the same as
        the raw score's, reversed, and no ranking is changed by the transform.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str
    observation_ids: Tuple[str, ...] = Field(
        default=(), description="Step 5 observations behind this profile."
    )
    schema_id: str
    raw_score: Optional[float] = Field(
        default=None, description="Model-native score; None when no model was fitted."
    )
    normalized_score: Optional[float] = Field(
        default=None, description="Normalized score in (0, 1]; None when unfitted."
    )
    band: Optional[UnusualnessBand] = Field(
        default=None, description="Fixed operating point the score falls in."
    )
    is_in_sample: bool
    explanation: Optional[AnomalyExplanation] = Field(default=None)

    @field_validator("profile_id")
    @classmethod
    def _check_profile_id(cls, value: str) -> str:
        return _require_non_blank(value, "profile_id")

    @field_validator("observation_ids")
    @classmethod
    def _check_observation_ids(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        _require_sorted_unique(value, "observation_ids")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "AnomalyResult":
        if (self.raw_score is None) != (self.normalized_score is None):
            raise ValueError("raw_score and normalized_score must be set together")
        if self.raw_score is not None and not math.isfinite(self.raw_score):
            raise ValueError("raw_score must be finite")
        if self.normalized_score is not None and not (
            0.0 < self.normalized_score <= 1.0
        ):
            raise ValueError("normalized_score must lie inside (0, 1]")
        if (self.normalized_score is None) != (self.band is None):
            raise ValueError("band must be set exactly when a score was produced")
        if self.explanation is not None and self.explanation.profile_id != self.profile_id:
            raise ValueError("explanation must describe the same profile")
        return self

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "observation_ids": list(self.observation_ids),
            "schema_id": self.schema_id,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "band": self.band.value if self.band else None,
            "is_in_sample": self.is_in_sample,
            "explanation": self.explanation.to_dict() if self.explanation else None,
        }


class AnomalyRunResult(BaseModel):
    """A scored population, with everything needed to read the numbers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: ModelRunMetadata
    status: ModelStatus
    results: Tuple[AnomalyResult, ...] = Field(
        default=(), description="Per-profile results, most unusual first."
    )
    feature_names: Tuple[str, ...] = Field(description="Vector columns, ascending.")
    band_boundaries: Tuple[float, float, float] = Field(
        default=(0.5, 0.7, 0.9), description="Cut points used for the bands."
    )
    insufficient_reason: Optional[str] = Field(
        default=None, description="Why no model was produced, when status says so."
    )

    @field_validator("feature_names")
    @classmethod
    def _check_feature_names(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if not value:
            return value
        return _require_sorted_unique(value, "feature_names")

    @model_validator(mode="after")
    def _check_consistency(self) -> "AnomalyRunResult":
        if self.status is ModelStatus.FITTED and not self.feature_names:
            raise ValueError("a fitted run must report its feature names")
        if self.status is ModelStatus.FITTED and not self.results:
            raise ValueError("a fitted run must report at least one result")
        if self.status is not ModelStatus.FITTED and self.results:
            raise ValueError("an unfitted run must not report scores")
        if self.status is not ModelStatus.FITTED and not (
            self.insufficient_reason or ""
        ).strip():
            raise ValueError("an unfitted run must state why")
        if list(self.feature_names) != list(self.metadata.feature_names):
            raise ValueError("feature_names must match the metadata schema")
        scores = [
            item.normalized_score
            for item in self.results
            if item.normalized_score is not None
        ]
        if scores != sorted(scores, reverse=True):
            raise ValueError("results must be ordered by normalized score descending")
        return self

    def result_for(self, profile_id: str) -> Optional[AnomalyResult]:
        """Return the result for *profile_id*, if it was scored."""
        for item in self.results:
            if item.profile_id == profile_id:
                return item
        return None

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "status": self.status.value,
            "insufficient_reason": self.insufficient_reason,
            "feature_names": list(self.feature_names),
            "band_boundaries": list(self.band_boundaries),
            "results": [item.to_dict() for item in self.results],
        }


class ClusterAssignment(BaseModel):
    """Which group a profile was placed in, and how far it sits from it.

    ``is_noise`` exists because "belongs to no group" is a real state for
    density-based methods, which the prompt allows for a later step.  The
    k-means implementation here assigns every profile it was fitted on, so
    this flag is always ``False`` in current results; it is part of the
    vocabulary, not a prediction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str
    cluster_id: Optional[str] = Field(
        default=None, description="None only when the profile is noise."
    )
    is_noise: bool = False
    distance_to_centroid: Optional[float] = Field(
        default=None, description="Euclidean distance to the assigned centroid."
    )
    is_in_sample: bool

    @field_validator("profile_id")
    @classmethod
    def _check_profile_id(cls, value: str) -> str:
        return _require_non_blank(value, "profile_id")

    @model_validator(mode="after")
    def _check_consistency(self) -> "ClusterAssignment":
        if self.is_noise:
            if self.cluster_id is not None and self.cluster_id != NOISE_CLUSTER_ID:
                raise ValueError(
                    "a noise assignment must use the reserved label %s"
                    % NOISE_CLUSTER_ID
                )
            return self
        if self.cluster_id is None:
            raise ValueError("a non-noise assignment requires a cluster_id")
        if not self.cluster_id.startswith("CLUSTER_"):
            raise ValueError("cluster_id must use the CLUSTER_ prefix")
        if self.distance_to_centroid is None or not math.isfinite(self.distance_to_centroid):
            raise ValueError("a non-noise assignment requires a finite distance")
        if self.distance_to_centroid < 0.0:
            raise ValueError("distance_to_centroid must be non-negative")
        return self

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "cluster_id": self.cluster_id,
            "is_noise": self.is_noise,
            "distance_to_centroid": self.distance_to_centroid,
            "is_in_sample": self.is_in_sample,
        }


class ClusterSummary(BaseModel):
    """One group: who is in it, where it sits, how tightly packed it is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: str
    signature: str = Field(
        description="SHA-256 over the sorted member profile ids: the basis "
        "used to order groups before labels are assigned."
    )
    member_profile_ids: Tuple[str, ...] = Field(description="Members, ascending.")
    centroid: Tuple[Tuple[str, float], ...] = Field(
        description="Named centroid coordinates, ascending by feature name."
    )
    dispersion: float = Field(
        ge=0.0, description="Mean Euclidean distance of members to the centroid."
    )
    dispersion_detail: str = Field(
        description="Explicit statement of what the dispersion number is."
    )

    @field_validator("cluster_id")
    @classmethod
    def _check_cluster_id(cls, value: str) -> str:
        return _require_non_blank(value, "cluster_id")

    @field_validator("member_profile_ids")
    @classmethod
    def _check_members(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if not value:
            raise ValueError("a cluster must have at least one member")
        return _require_sorted_unique(value, "member_profile_ids")

    @field_validator("centroid")
    @classmethod
    def _check_centroid(cls, value: Tuple[Tuple[str, float], ...]) -> Tuple[Tuple[str, float], ...]:
        if not value:
            raise ValueError("centroid must not be empty")
        names = [item[0] for item in value]
        _require_sorted_unique(tuple(names), "centroid names")
        for _, number in value:
            if not math.isfinite(float(number)):
                raise ValueError("centroid coordinates must be finite")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "ClusterSummary":
        expected = hashlib.sha256("\n".join(self.member_profile_ids).encode("utf-8")).hexdigest()
        if self.signature != expected:
            raise ValueError("signature must be the digest of the sorted member ids")
        return self

    @property
    def member_count(self) -> int:
        """Number of profiles assigned to this group."""
        return len(self.member_profile_ids)

    def centroid_value(self, feature_name: str) -> float:
        """Return one centroid coordinate by feature name."""
        for name, number in self.centroid:
            if name == feature_name:
                return number
        raise KeyError(feature_name)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "signature": self.signature,
            "member_count": self.member_count,
            "member_profile_ids": list(self.member_profile_ids),
            "centroid": {name: number for name, number in self.centroid},
            "dispersion": self.dispersion,
            "dispersion_detail": self.dispersion_detail,
        }


class ClusteringRunResult(BaseModel):
    """A fitted partition, with the selection evidence kept alongside it.

    ``selection_scores`` is the record behind an automatic ``k`` choice: every
    candidate ``k`` with the score it achieved.  A reader can disagree with the
    choice, and can see exactly what it was chosen over.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: ModelRunMetadata
    status: ModelStatus
    k: Optional[int] = Field(
        default=None, description="Number of groups; None when none were produced."
    )
    clusters: Tuple[ClusterSummary, ...] = Field(
        default=(), description="Groups in canonical order."
    )
    assignments: Tuple[ClusterAssignment, ...] = Field(
        default=(), description="Per-profile assignment, ascending by profile id."
    )
    feature_names: Tuple[str, ...] = Field(description="Vector columns, ascending.")
    selection_scores: Tuple[Tuple[int, float], ...] = Field(
        default=(), description="Candidate k and the score it achieved."
    )
    noise_profile_ids: Tuple[str, ...] = Field(
        default=(), description="Profiles in no group; empty for k-means."
    )
    insufficient_reason: Optional[str] = Field(
        default=None, description="Why no partition was produced, when status says so."
    )

    @field_validator("feature_names")
    @classmethod
    def _check_feature_names(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if not value:
            return value
        return _require_sorted_unique(value, "feature_names")

    @model_validator(mode="after")
    def _check_consistency(self) -> "ClusteringRunResult":
        if list(self.feature_names) != list(self.metadata.feature_names):
            raise ValueError("feature_names must match the metadata schema")
        if self.status is ModelStatus.FITTED and not self.feature_names:
            raise ValueError("a fitted run must report its feature names")
        labels = [item.cluster_id for item in self.clusters]
        if labels != sorted(labels):
            raise ValueError("clusters must be listed in ascending label order")
        if self.k is not None and len(self.clusters) != self.k:
            raise ValueError("k must equal the number of reported clusters")
        if len(labels) != len(set(labels)):
            raise ValueError("cluster labels must be unique")
        if self.status is ModelStatus.FITTED:
            if not self.clusters or not self.assignments:
                raise ValueError("a fitted run must report clusters and assignments")
        elif self.clusters or self.assignments:
            raise ValueError("an unfitted run must not report clusters or assignments")
        if self.status is not ModelStatus.FITTED and not (self.insufficient_reason or "").strip():
            raise ValueError("an unfitted run must state why")
        members = [
            profile_id
            for summary in self.clusters
            for profile_id in summary.member_profile_ids
        ]
        if sorted(members) != sorted(
            item.profile_id for item in self.assignments if not item.is_noise
        ):
            raise ValueError("cluster members and non-noise assignments must agree")
        if len(members) != len(set(members)):
            raise ValueError("a profile may belong to exactly one cluster")
        if self.noise_profile_ids != tuple(
            sorted(item.profile_id for item in self.assignments if item.is_noise)
        ):
            raise ValueError("noise_profile_ids must match the noise assignments")
        for index, (candidate, score) in enumerate(self.selection_scores):
            if not isinstance(candidate, int) or candidate < 2:
                raise ValueError("selection scores must be reported for k >= 2")
            if not math.isfinite(score) or score < -1.0 or score > 1.0:
                raise ValueError("silhouette scores must lie inside [-1, 1]")
            if index and candidate <= self.selection_scores[index - 1][0]:
                raise ValueError("selection scores must ascend by k")
        return self

    def cluster_of(self, profile_id: str) -> Optional[ClusterAssignment]:
        """Return the assignment for *profile_id*, if it was assigned."""
        for item in self.assignments:
            if item.profile_id == profile_id:
                return item
        return None

    def cluster_summary(self, cluster_id: str) -> Optional[ClusterSummary]:
        """Return the summary for *cluster_id*, if it exists."""
        for item in self.clusters:
            if item.cluster_id == cluster_id:
                return item
        return None

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "status": self.status.value,
            "insufficient_reason": self.insufficient_reason,
            "k": self.k,
            "feature_names": list(self.feature_names),
            "selection_scores": [
                {"k": candidate, "score": score} for candidate, score in self.selection_scores
            ],
            "clusters": [item.to_dict() for item in self.clusters],
            "assignments": [item.to_dict() for item in self.assignments],
            "noise_profile_ids": list(self.noise_profile_ids),
        }
