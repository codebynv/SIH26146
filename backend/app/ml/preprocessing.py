"""Turning Step 5 profiles into numeric vectors (SIH26146 - Step 6).

This module is the only place in Step 6 that reads a Step 5 object.  It answers
three questions, once, in writing:

Which Step 5 features may become numbers?
    Counts and decimal measurements only.  A label set is a *set of strings*
    and has no magnitude; putting an integer in its column would invent an
    ordering and a distance between ``"US"`` and ``"IN"`` that nobody measured.
    A timestamp is an *instant*, and an instant has no natural origin -- writing
    it as seconds since an arbitrary epoch would silently introduce a policy
    about which clock and which timezone.  So both are refused, and every
    refusal is reported in :attr:`FeatureSchema.excluded_features` rather than
    being applied quietly.  See :func:`is_numeric_value_kind`.

What happens to a measurement that was never available?
    Step 5 distinguishes "measured as zero" from "could not be measured", and
    this module preserves that distinction instead of blurring it.  An
    unavailable measurement is replaced by the median of the *observed* values
    for that same column across the fitted population, which is the neutral
    fill: it adds no direction and no magnitude.  The substituted columns are
    listed per row in :attr:`MLFeatureVector.imputed_feature_names`, so a
    reader can discount them, and a column with no observed value at all falls
    back to ``0.0`` and is reported as constant.

How are columns put on a comparable scale?
    Median and median-absolute-deviation by default, by a three-step ladder.
    Step 5 columns are counts and sums, which are right-skewed, and the whole
    point of this step is to measure the profiles that sit far from the rest.
    Mean and standard deviation would be pulled toward exactly those profiles,
    so the outliers would set the scale that is then used to judge them.  The
    median and the MAD are not.  But the MAD is zero as soon as half the
    population shares one value, and Step 5 populations are full of such
    columns -- every profile built from four observations has an
    ``observation_count`` of four.  So the ladder is:

    1. ``1.4826 x MAD`` -- the spread most than half the population cannot move;
    2. failing that, ``1.2533 x mean absolute deviation from the median``,
       which is not zero merely because most values agree, and which is far
       less dominated by a single extreme profile than a standard deviation
       would be;
    3. failing that, the column genuinely never varied.  Its scale is set to
       ``1.0`` so it contributes nothing to any distance instead of dividing
       by zero, and it is reported in
       :attr:`FittedFeaturePreprocessor.constant_feature_names`.

    The ladder changes no ordering within a column.  It changes the *unit* each
    column is measured in, which is why both constants are written down here
    rather than being left inside the arithmetic.

Why Decimal is converted the way it is
--------------------------------------
Step 5 produces :class:`~decimal.Decimal` amounts.  Those are converted through
their exact decimal string at a pinned :data:`decimal.Decimal` context, never
through a lossy shortcut, and never through a caller-supplied context: a vector
must not change because a caller left a different rounding mode in place.  A
value that cannot be represented as a finite float -- ``NaN``, an infinity, a
magnitude beyond the float range -- is refused rather than clamped, because a
clamped number would be a fabricated measurement.
"""

from __future__ import annotations

import decimal
import hashlib
import math
from decimal import Decimal
from typing import Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..behavior.models import (
    BehavioralFeature,
    CountValue,
    DecimalValue,
    LabelSetValue,
    LabelValue,
    TimestampValue,
)
from ..behavior.profiles import BehavioralProfile
from .models import (
    ExcludedFeature,
    ExclusionReason,
    FeatureMatrix,
    FeatureSchema,
    ImputationStrategy,
    MLFeatureVector,
    ScalingMethod,
    identifier_like_feature_name,
    is_numeric_value_kind,
)

__all__ = [
    "FittedFeaturePreprocessor",
    "MLPreprocessorError",
    "PREPROCESSING_DECIMAL_CONTEXT",
    "PreprocessorConfig",
    "ProfileFeatureVectorizer",
    "derive_schema_id",
    "median",
    "median_absolute_deviation",
]

#: Precision used for every decimal-to-float conversion in this module.  Pinned
#: here so an ambient :data:`decimal` context cannot change a vector.
PREPROCESSING_DECIMAL_CONTEXT = decimal.Context(
    prec=34, rounding=decimal.ROUND_HALF_EVEN
)

#: Multiplier that makes the median absolute deviation a consistent estimator
#: of the standard deviation for normally distributed data.  Documented rather
#: than hidden: it changes the *unit* of the scale, not the *order* of any
#: distance or ranking computed with it.
_MAD_CONSISTENCY_FACTOR = 1.4826

#: Matching consistency factor for the mean absolute deviation, used only when
#: the median absolute deviation is zero.  Step 5 populations are full of
#: columns where most profiles agree, so this fallback is not an edge case here;
#: it is the common path.
_MEAN_ABSOLUTE_CONSISTENCY_FACTOR = 1.2533141373155001


class MLPreprocessorError(ValueError):
    """Raised when a profile cannot be turned into a vector at all."""


def median(values: Sequence[float]) -> float:
    """Median of *values*, averaging the middle pair for an even count."""
    if not values:
        raise MLPreprocessorError("median of an empty sequence is undefined")
    ordered = sorted(float(item) for item in values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def median_absolute_deviation(values: Sequence[float]) -> float:
    """Median absolute deviation of *values* about their median.

    A spread estimate that at least half the population cannot move.  Reported
    unscaled; :data:`_MAD_CONSISTENCY_FACTOR` is applied by the caller, where
    the choice of estimator belongs.
    """
    if not values:
        raise MLPreprocessorError(
            "median absolute deviation of an empty sequence is undefined"
        )
    centre = median(values)
    return median([abs(float(item) - centre) for item in values])


def _population_standard_deviation(values: Sequence[float]) -> float:
    """Population (not sample) standard deviation, using exact summation.

    Dividing by ``n`` rather than ``n - 1`` keeps the estimator defined for a
    single value, which matters because a one-profile population must fail with
    a stated status rather than raise a statistics error here.
    """
    if not values:
        return 0.0
    count = len(values)
    mean = math.fsum(float(item) for item in values) / count
    variance = math.fsum((float(item) - mean) ** 2 for item in values) / count
    return math.sqrt(variance) if variance > 0.0 else 0.0


def _mean_absolute_deviation(values: Sequence[float], centre: float) -> float:
    """Mean absolute distance from *centre*.

    Used as the second step of the scale ladder because it stays useful when
    the median absolute deviation is zero -- which happens as soon as half the
    population shares one value, common for Step 5 counts -- while still being
    far less dominated by a single extreme profile than a standard deviation.
    """
    if not values:
        return 0.0
    return math.fsum(abs(float(item) - centre) for item in values) / len(values)


def derive_schema_id(feature_names: Sequence[str]) -> str:
    """Content-derived identifier for an ordered column set.

    SHA-256 over the joined column names, so the same columns always carry the
    same schema id and any change to the column set changes it.
    """
    joined = "\n".join(feature_names)
    return "SCHEMA_" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16].upper()


def _to_float(value: object, feature_name: str) -> float:
    """Convert a Step 5 magnitude to a finite float, exactly.

    ``Decimal`` goes through its exact string form under a pinned context, so
    the conversion matches the precision the extractor used rather than taking
    a binary shortcut.  Anything that is not a magnitude is refused.
    """
    if isinstance(value, Decimal):
        with decimal.localcontext(PREPROCESSING_DECIMAL_CONTEXT):
            try:
                number = float(value)
            except (OverflowError, ValueError) as exc:
                raise MLPreprocessorError(
                    "feature %r holds a decimal magnitude with no finite float "
                    "representation" % feature_name
                ) from exc
    elif isinstance(value, int) and not isinstance(value, bool):
        number = float(value)
    else:
        raise MLPreprocessorError(
            "feature %r holds a %s, which is not a measurable magnitude"
            % (feature_name, type(value).__name__)
        )
    if not math.isfinite(number):
        raise MLPreprocessorError(
            "feature %r holds a value that is not finite" % feature_name
        )
    return number


def _exclusion_for(feature: BehavioralFeature) -> Optional[ExcludedFeature]:
    """Return why *feature* is not a vector column, or ``None`` if it is."""
    value = feature.value
    if isinstance(value, LabelSetValue):
        return ExcludedFeature(
            name=feature.name,
            value_kind=value.kind,
            reason=ExclusionReason.CATEGORICAL_LABEL_SET,
            detail=(
                "A distinct-value list has no magnitude. An integer encoding "
                "would invent an ordering between strings that were never "
                "ordered."
            ),
        )
    if isinstance(value, LabelValue):
        return ExcludedFeature(
            name=feature.name,
            value_kind=value.kind,
            reason=ExclusionReason.CATEGORICAL_LABEL,
            detail=(
                "A category has no magnitude, and encoding it as an integer "
                "would assert an ordering the source data does not contain."
            ),
        )
    if isinstance(value, TimestampValue):
        return ExcludedFeature(
            name=feature.name,
            value_kind=value.kind,
            reason=ExclusionReason.INSTANT_TIMESTAMP,
            detail=(
                "An instant has no natural origin. Encoding it as seconds "
                "since an epoch would add an unstated clock, timezone and "
                "epoch policy to the data."
            ),
        )
    if not is_numeric_value_kind(value.kind):
        return ExcludedFeature(
            name=feature.name,
            value_kind=value.kind,
            reason=ExclusionReason.CATEGORICAL_LABEL,
            detail="Only count and decimal measurements are magnitudes.",
        )
    if identifier_like_feature_name(feature.name):
        return ExcludedFeature(
            name=feature.name,
            value_kind=value.kind,
            reason=ExclusionReason.IDENTIFIER_LIKE_NAME,
            detail=(
                "The column name denotes an identifier rather than a "
                "measurement, so a distance over it would compare lookup keys."
            ),
        )
    return None


def _numeric_value(feature: BehavioralFeature) -> Optional[float]:
    """Return the measured magnitude of *feature*, or ``None`` if unavailable.

    ``None`` means Step 5 reported the scalar as unavailable, which is a
    different fact from a measured zero, and the two are kept apart all the way
    to the reported ``imputed_feature_names``.
    """
    value = feature.value
    if not isinstance(value, (CountValue, DecimalValue)):
        raise MLPreprocessorError(
            "feature %r is not a numeric Step 5 value" % feature.name
        )
    if value.value is None:
        return None
    return _to_float(value.value, feature.name)


def _has_feature(profile: BehavioralProfile, name: str) -> bool:
    return any(feature.name == name for feature in profile.features)


def _selected_names(profile: BehavioralProfile) -> Tuple[str, ...]:
    """Numeric column names of *profile*, ascending.

    Selection is read off one profile's feature set rather than the union of
    the population, so the schema is a property of the Step 5 feature contract
    and not of who happened to be in this batch.
    """
    names = tuple(
        sorted(
            feature.name
            for feature in profile.features
            if _exclusion_for(feature) is None
        )
    )
    if not names:
        raise MLPreprocessorError(
            "profile %s carries no measurable Step 5 feature" % profile.profile_id
        )
    return names


def _measurements(
    profile: BehavioralProfile, names: Sequence[str]
) -> Tuple[Tuple[str, Optional[float]], ...]:
    """Ordered ``(name, magnitude or None)`` pairs for one profile."""
    available = {feature.name: feature for feature in profile.features}
    pairs = []
    for name in names:
        feature = available.get(name)
        if feature is None:
            raise MLPreprocessorError(
                "profile %s has no feature %r" % (profile.profile_id, name)
            )
        pairs.append((name, _numeric_value(feature)))
    return tuple(pairs)


def _canonical_profiles(profiles: Sequence[BehavioralProfile]) -> Tuple[BehavioralProfile, ...]:
    """Order profiles by id and refuse duplicates.

    Every fitted statistic and every random draw in Step 6 is taken over this
    order, which is what makes a shuffled input produce identical output.
    """
    if not profiles:
        raise MLPreprocessorError("at least one profile is required to fit")
    for profile in profiles:
        if not isinstance(profile, BehavioralProfile):
            raise MLPreprocessorError(
                "expected Step 5 BehavioralProfile objects, received a %s"
                % type(profile).__name__
            )
    ids = [profile.profile_id for profile in profiles]
    if len(set(ids)) != len(ids):
        raise MLPreprocessorError("profile ids must be unique within a population")
    return tuple(sorted(profiles, key=lambda item: item.profile_id))


class PreprocessorConfig(BaseModel):
    """The two preprocessing choices that change every vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scaling: ScalingMethod = Field(
        default=ScalingMethod.ROBUST_MAD,
        description="How columns were centred and scaled. ROBUST_MAD centres "
        "on the median and scales by 1.4826 x MAD, falling back to the "
        "population standard deviation when the MAD is zero.",
    )
    imputation: ImputationStrategy = Field(
        default=ImputationStrategy.FITTED_MEDIAN,
        description="How an unavailable measurement is filled: the fitted "
        "median of that same column, which adds no direction of its own.",
    )

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class FittedFeaturePreprocessor(BaseModel):
    """An immutable, fitted column set with its reference statistics.

    Holds everything a later transform needs -- schema, centres, scales,
    constant columns -- so that transforming a *new* profile requires nothing
    but the object itself.  No fit state is hidden anywhere else.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_schema: FeatureSchema
    config: PreprocessorConfig
    centers: Tuple[Tuple[str, float], ...] = Field(
        description="Fitted reference per column, ascending by feature name."
    )
    scales: Tuple[Tuple[str, float], ...] = Field(
        description="Fitted spread per column, ascending by feature name, "
        "always strictly positive."
    )
    constant_feature_names: Tuple[str, ...] = Field(
        default=(), description="Columns whose spread is zero across the fit."
    )
    value_units: Tuple[Tuple[str, Optional[str]], ...] = Field(
        default=(),
        description="Unit label Step 5 reported per column, so an explanation "
        "can say what a number is counting.",
    )
    observed_value_counts: Tuple[Tuple[str, int], ...] = Field(
        default=(), description="Measured values seen per column while fitting."
    )
    unavailable_value_counts: Tuple[Tuple[str, int], ...] = Field(
        default=(), description="Unavailable measurements seen per column."
    )
    fitted_profile_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "FittedFeaturePreprocessor":
        names = self.feature_schema.feature_names
        for label, mapping in (("centers", self.centers), ("scales", self.scales)):
            keys = tuple(item[0] for item in mapping)
            if keys != names:
                raise ValueError("%s must cover exactly the schema columns" % label)
        for _, scale in self.scales:
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("every fitted scale must be finite and positive")
        for _, value in self.centers:
            if not math.isfinite(value):
                raise ValueError("every fitted centre must be finite")
        if list(self.constant_feature_names) != sorted(self.constant_feature_names):
            raise ValueError("constant_feature_names must be ascending")
        if not set(self.constant_feature_names) <= set(names):
            raise ValueError("constant_feature_names must name schema columns")
        if tuple(item[0] for item in self.value_units) != names:
            raise ValueError("value_units must cover exactly the schema columns")
        return self

    @property
    def schema_id(self) -> str:
        """Identifier of the fitted column set."""
        return self.feature_schema.schema_id

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Vector columns, ascending."""
        return self.feature_schema.feature_names

    def center_of(self, feature_name: str) -> float:
        """Fitted reference for one column, in the column's own units."""
        for name, value in self.centers:
            if name == feature_name:
                return value
        raise KeyError(feature_name)

    def scale_of(self, feature_name: str) -> float:
        """Fitted spread for one column, in the column's own units."""
        for name, value in self.scales:
            if name == feature_name:
                return value
        raise KeyError(feature_name)

    def observed_count_of(self, feature_name: str) -> int:
        """How many measured values were seen for one column while fitting."""
        for name, value in self.observed_value_counts:
            if name == feature_name:
                return value
        raise KeyError(feature_name)

    def unavailable_count_of(self, feature_name: str) -> int:
        """How many unavailable measurements were seen for one column."""
        for name, value in self.unavailable_value_counts:
            if name == feature_name:
                return value
        raise KeyError(feature_name)

    def is_constant(self, feature_name: str) -> bool:
        """Whether one column carried no variation across the fitted rows."""
        return feature_name in self.constant_feature_names

    def unit_of(self, feature_name: str) -> Optional[str]:
        """Unit label Step 5 reported for one column, if any."""
        for name, unit in self.value_units:
            if name == feature_name:
                return unit
        raise KeyError(feature_name)

    def _to_vector_value(self, measured: float, name: str) -> float:
        """Map one measured magnitude onto its vector position.

        Under :attr:`ScalingMethod.NONE` the measured magnitude is passed
        through untouched -- that is what the method name promises, and a column
        set left in mixed units is why it is not the default.  Imputation still
        substitutes the fitted median for an unavailable measurement, so an
        unscaled vector is centred on its own missing values.
        """
        if self.config.scaling is ScalingMethod.NONE:
            return float(measured)
        return (measured - self.center_of(name)) / self.scale_of(name)

    def _measurements_for(
        self, profile: BehavioralProfile
    ) -> Tuple[Tuple[str, Optional[float]], ...]:
        return _measurements(profile, self.feature_names)

    def raw_values_for(self, profile: BehavioralProfile) -> Tuple[Tuple[str, float], ...]:
        """Measured values in their own units, with unavailable ones filled.

        This is the human-readable view used by explanations: a reader wants to
        see the count that was actually there, not the scaled number.
        """
        filled = []
        for name, measured in self._measurements_for(profile):
            filled.append(
                (name, self.center_of(name) if measured is None else measured)
            )
        return tuple(filled)

    def transform_one(self, profile: BehavioralProfile) -> MLFeatureVector:
        """Return one profile as a scaled, imputed, finite-valued vector."""
        if not isinstance(profile, BehavioralProfile):
            raise MLPreprocessorError(
                "expected a Step 5 BehavioralProfile, received a %s"
                % type(profile).__name__
            )
        values = []
        imputed = []
        for name, measured in self._measurements_for(profile):
            if measured is None:
                measured = self.center_of(name)
                imputed.append(name)
            scaled = self._to_vector_value(measured, name)
            if not math.isfinite(scaled):
                raise MLPreprocessorError(
                    "feature %r did not produce a finite scaled value" % name
                )
            values.append(scaled)
        return MLFeatureVector(
            schema_id=self.schema_id,
            feature_names=self.feature_names,
            values=tuple(values),
            imputed_feature_names=tuple(sorted(imputed)),
        )

    def transform(self, profiles: Sequence[BehavioralProfile]) -> FeatureMatrix:
        """Return a whole population as a matrix in canonical profile order."""
        ordered = _canonical_profiles(profiles)
        vectors = [self.transform_one(profile) for profile in ordered]
        return FeatureMatrix(
            schema_id=self.schema_id,
            feature_names=self.feature_names,
            profile_ids=tuple(profile.profile_id for profile in ordered),
            rows=tuple(vector.values for vector in vectors),
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.feature_schema.to_dict(),
            "config": self.config.to_dict(),
            "centers": {name: value for name, value in self.centers},
            "scales": {name: value for name, value in self.scales},
            "constant_feature_names": list(self.constant_feature_names),
            "value_units": {name: unit for name, unit in self.value_units},
            "observed_value_counts": dict(self.observed_value_counts),
            "unavailable_value_counts": dict(self.unavailable_value_counts),
            "fitted_profile_count": self.fitted_profile_count,
        }


class ProfileFeatureVectorizer:
    """Fits and reports the Step 5 profile to vector adapter.

    Stateless and free of configuration ambiguity: one vectorizer, one schema,
    one set of reference statistics, all of them derived from the population it
    was asked to fit and from nothing else.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self._config = config or PreprocessorConfig()

    @property
    def config(self) -> PreprocessorConfig:
        """The preprocessing configuration in force."""
        return self._config

    def select_schema(self, profiles: Sequence[BehavioralProfile]) -> FeatureSchema:
        """Choose the column set from a population.

        A column is any Step 5 feature that every profile carries as a count or
        a decimal measurement.  A feature that some profile lacks is refused
        rather than partially filled, because a column that exists for some rows
        only is a column with a hole in it that no reader can see.
        """
        ordered = _canonical_profiles(profiles)
        names = _selected_names(ordered[0])
        for profile in ordered[1:]:
            missing = [name for name in names if not _has_feature(profile, name)]
            if missing:
                raise MLPreprocessorError(
                    "profile %s lacks schema columns %s" % (profile.profile_id, missing)
                )
        excluded: dict = {}
        for profile in ordered:
            for feature in profile.features:
                reason = _exclusion_for(feature)
                if reason is not None:
                    excluded.setdefault(feature.name, reason)
        return FeatureSchema(
            schema_id=derive_schema_id(names),
            feature_names=names,
            excluded_features=tuple(excluded[name] for name in sorted(excluded)),
            scaling=self._config.scaling,
            imputation=self._config.imputation,
        )

    def fit(self, profiles: Sequence[BehavioralProfile]) -> FittedFeaturePreprocessor:
        """Fit the schema and its reference statistics on *profiles*."""
        ordered = _canonical_profiles(profiles)
        schema = self.select_schema(ordered)
        measured: dict = {name: [] for name in schema.feature_names}
        unavailable: dict = {name: 0 for name in schema.feature_names}
        units: dict = {}
        for profile in ordered:
            for feature in profile.features:
                if feature.name in schema.feature_names:
                    units.setdefault(feature.name, feature.unit)
            for name, value in _measurements(profile, schema.feature_names):
                if value is None:
                    unavailable[name] += 1
                else:
                    measured[name].append(value)
        centers = []
        scales = []
        constants = []
        observed_counts = []
        unavailable_counts = []
        for name in schema.feature_names:
            observed = measured[name]
            centre = median(observed) if observed else 0.0
            spread = 0.0
            if self._config.scaling is not ScalingMethod.NONE and observed:
                if self._config.scaling is ScalingMethod.ROBUST_MAD:
                    spread = _MAD_CONSISTENCY_FACTOR * median_absolute_deviation(observed)
                if spread <= 0.0:
                    if self._config.scaling is ScalingMethod.ROBUST_MAD:
                        spread = _MEAN_ABSOLUTE_CONSISTENCY_FACTOR * (
                            _mean_absolute_deviation(observed, centre)
                        )
                    else:
                        spread = _population_standard_deviation(observed)
                if spread <= 0.0:
                    constants.append(name)
            centers.append((name, centre))
            scales.append((name, spread if spread > 0.0 else 1.0))
            observed_counts.append((name, len(observed)))
            unavailable_counts.append((name, unavailable[name]))
        return FittedFeaturePreprocessor(
            feature_schema=schema,
            config=self._config,
            centers=tuple(centers),
            scales=tuple(scales),
            constant_feature_names=tuple(sorted(constants)),
            value_units=tuple((name, units.get(name)) for name in schema.feature_names),
            observed_value_counts=tuple(observed_counts),
            unavailable_value_counts=tuple(unavailable_counts),
            fitted_profile_count=len(ordered),
        )

