"""Unusualness of a profile relative to a fitted population (SIH26146 - Step 6).

The algorithm, and why this one
------------------------------
An isolation forest.  A population is described by many random binary searches
that repeatedly cut it on one random column at a random point.  A profile that
falls out after few cuts is in a part of the space where there is little else,
and one that survives many cuts sits in a crowded neighbourhood.  The expected
path length of such a search is the natural unit of "crowded", and the score
below is exactly a comparison of one profile's path lengths against that unit.
The algorithm has no class labels, no training target and no notion of what a
profile is *for*; it only measures crowding, which is the most it can honestly
be asked for.

One documented deviation from the textbook algorithm
---------------------------------------------------
The cut column is drawn uniformly from the columns that *vary* in the fitted
population, not from all of them.  Step 5 vectors are sparse -- build every
profile from four observations and most of the fifty-odd columns are identical
across the whole population -- and a uniform draw over all columns would spend
most of its cuts on columns that cannot separate anything, degenerate the trees
into leaves, and hand the unusual profile nearly the same number as the typical
ones.  :func:`_varying_features` states the rule and the reason in full, and
the estimator version is reported as ``ISOLATION_FOREST_STDLIB/2`` so a stored
result can be traced to the variant that produced it.

The numbers, defined exactly
----------------------------
``c(n)``
    The expected path length of an unsuccessful search in a binary search tree
    over ``n`` uniform points: ``2 * (ln(n - 1) + gamma) - 2 * (n - 1) / n``,
    with the two small cases given their exact values (0 and 1).  This is the
    denominator that makes scores from different subsample sizes comparable.
``raw_score``
    Mean path length of the profile through the fitted trees, divided by
    ``c(psi)``.  Model-native, and normally in ``(0, 1]``.
``normalized_score``
    ``2 ** (-raw_score)``, a strictly decreasing remap onto ``(0, 1]``.  It
    rises towards 1 as a profile separates quickly from the population and
    falls towards 0 as it takes long to separate.  The direction matters: a
    *short* path is what "unusual" means for this algorithm, so the remap must
    grow as the path shrinks.  One is attained exactly when a single cut
    isolates the profile from every reference point, and zero is only ever
    approached.  The transform changes no ordering, which is why it is safe to
    report and easy to state.

Where the randomness comes from
-------------------------------
There is no :mod:`random` module and no :func:`uuid.uuid4` in this file.  Every
draw -- which column, where to cut, which points form a tree's subsample -- is
a SHA-256 digest of the configured ``random_state`` together with the tree
number, the node number and the row index.  The same seed therefore produces the
same forest on any machine, in any Python version, from any installation, which
is a stronger guarantee than seeding :mod:`random` would give.

Small and degenerate populations
--------------------------------
The model refuses to invent a number it cannot support, and says so through
:data:`~backend.app.ml.models.ModelStatus`:

* no profiles, or fewer than ``min_profiles`` -- ``INSUFFICIENT_DATA``, with a
  stated reason and no scores at all;
* a population where every profile is identical -- fitted normally, with
  :attr:`FittedAnomalyModel.varying_feature_count` reported as zero.  No column
  can be cut on, so every tree is a root leaf, every raw score is equal, and the
  run reports that equality rather than pretending to rank identical profiles.

Scoring out of sample
---------------------
:meth:`IsolationForestAnomalyDetector.fit` and
:meth:`FittedAnomalyModel.score` are separate calls, and
:meth:`FittedAnomalyModel.score_new` marks its results as out of sample.  A
baseline population can therefore be fitted now and new profiles scored later,
with the distinction recorded in
:attr:`~backend.app.ml.models.ModelRunMetadata.scoring_in_sample` rather than
left to the reader to guess.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..behavior.profiles import BehavioralProfile
from .explanations import explain_profile
from .models import (
    AnomalyModelConfig,
    AnomalyResult,
    AnomalyRunResult,
    FeatureMatrix,
    ModelRunMetadata,
    ModelStatus,
    UnusualnessBand,
    canonical_json,
)
from .preprocessing import (
    FittedFeaturePreprocessor,
    PreprocessorConfig,
    ProfileFeatureVectorizer,
    _canonical_profiles,
)

__all__ = [
    "ANOMALY_ALGORITHM_VERSION",
    "FittedAnomalyModel",
    "IsolationForestAnomalyDetector",
    "average_unsuccessful_path_length",
    "band_for_score",
]

#: Version tag of the estimator implemented here, reported in run metadata so a
#: stored result can be traced to the exact formula that produced it.  The
#: ``/2`` marks the documented cut-column variant described in
#: :func:`_varying_features`.
ANOMALY_ALGORITHM_VERSION = "ISOLATION_FOREST_STDLIB/2"

#: Schema identifier reported when no model was fitted and therefore no column
#: set exists.  A distinct literal, so an empty column list is never mistaken
#: for a population that was scored with zero columns.
_NO_SCHEMA = "SCHEMA_NONE"

_EULER_GAMMA = 0.5772156649015329

#: Digest width used for every derived draw.  Eight bytes is 64 bits, which is
#: far more than the entropy a 256-row subsample or a 52-column choice needs,
#: and it keeps the derivation identical on every platform.
_DIGEST_BYTES = 8
_DIGEST_MODULUS = 1 << (8 * _DIGEST_BYTES)


def _derive_draw(*parts: object) -> int:
    """Return a deterministic integer derived from *parts*."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:_DIGEST_BYTES], "big")


def average_unsuccessful_path_length(size: int) -> float:
    """Expected path length of a failed binary search over *size* points.

    ``2 * (ln(n - 1) + gamma) - 2 * (n - 1) / n`` for ``n > 2``, and the exact
    values 0 and 1 for the two cases the formula does not cover.
    """
    if size <= 1:
        return 0.0
    if size == 2:
        return 1.0
    return 2.0 * (math.log(size - 1) + _EULER_GAMMA) - (2.0 * (size - 1)) / size


def band_for_score(
    normalized_score: float, boundaries: Tuple[float, float, float]
) -> UnusualnessBand:
    """Return the fixed operating point *normalized_score* falls in.

    The boundaries are operating points chosen in advance, not fitted cut
    points.  The first of them, ``0.5``, is where this algorithm's own
    reference convention calls a profile unusual; the rest are documented round
    numbers above it.  Nothing here estimates a distribution, and no band
    asserts anything about a profile beyond where its number sits.
    """
    if not 0.0 < normalized_score <= 1.0:
        raise ValueError("normalized_score must lie in (0, 1]")
    low, moderate, high = boundaries
    if normalized_score < low:
        return UnusualnessBand.LOW
    if normalized_score < moderate:
        return UnusualnessBand.MODERATE
    if normalized_score < high:
        return UnusualnessBand.HIGH
    return UnusualnessBand.EXTREME


class _IsolationNode(BaseModel):
    """One node of a fitted tree: either a cut, or a terminal sample set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    size: int = Field(ge=0, description="Samples reaching this node.")
    depth: int = Field(ge=0)
    feature_index: Optional[int] = Field(
        default=None, description="Column cut on; None marks a leaf."
    )
    split_value: Optional[float] = Field(default=None)
    left: Optional["_IsolationNode"] = Field(default=None)
    right: Optional["_IsolationNode"] = Field(default=None)

    @property
    def is_leaf(self) -> bool:
        """Whether this node stops the search."""
        return self.feature_index is None

    def to_dict(self) -> dict:
        payload = {"size": self.size, "depth": self.depth, "is_leaf": self.is_leaf}
        if not self.is_leaf:
            payload["feature_index"] = self.feature_index
            payload["split_value"] = self.split_value
            payload["left"] = self.left.to_dict() if self.left else None
            payload["right"] = self.right.to_dict() if self.right else None
        return payload


_IsolationNode.model_rebuild()


def _subsample_indices(seed: int, tree: int, population: int, size: int) -> Tuple[int, ...]:
    """Choose *size* row indices for one tree, deterministically.

    Indices are ranked by a digest of the seed, the tree number and the row
    index, and the chosen set is then sorted so the tree is built over a
    canonical row order.  Selecting a *set* by content rather than by position
    is what keeps a shuffled input from producing a different forest.
    """
    if size >= population:
        return tuple(range(population))
    ranked = sorted(range(population), key=lambda row: _derive_draw(seed, "subsample", tree, row))
    return tuple(sorted(ranked[:size]))


def _varying_features(matrix: FeatureMatrix) -> Tuple[int, ...]:
    """Columns that vary somewhere in the fitted population, ascending.

    Step 5 vectors are sparse: build every profile from four observations and
    some forty of the fifty-odd columns are identical across the whole
    population.  A textbook isolation forest picks its cut column uniformly
    from *all* columns, so on this data most of its cuts land on a column that
    cannot separate anything, the node becomes a leaf, and the path length
    collapses towards its expected value for every profile alike.  The
    unusual profile and the typical ones then receive nearly the same number,
    and the method stops doing the only thing it is here to do.

    So the cut column is drawn uniformly from the columns that vary in the
    fitted population.  This is a deliberate, documented variant of the
    algorithm, and it is the reason the estimator version is reported as
    ``ISOLATION_FOREST_STDLIB/2``: a result can be traced to the exact variant
    that produced it.  A node whose chosen column happens to be constant *within
    that node's* rows still becomes a leaf, as it must.
    """
    varying = []
    for column in range(matrix.dimension):
        values = [row[column] for row in matrix.rows]
        if max(values) > min(values):
            varying.append(column)
    return tuple(varying)


def _build_node(
    seed: int,
    tree: int,
    node: int,
    rows: Sequence[int],
    matrix: FeatureMatrix,
    varying: Sequence[int],
    depth: int,
    max_depth: int,
) -> _IsolationNode:
    """Grow one subtree over the row indices in *rows*."""
    size = len(rows)
    if size <= 1 or depth >= max_depth or not varying:
        return _IsolationNode(size=size, depth=depth)
    feature_index = varying[_derive_draw(seed, "feature", tree, node) % len(varying)]
    values = [matrix.rows[row][feature_index] for row in rows]
    low = min(values)
    high = max(values)
    if not high > low:
        return _IsolationNode(size=size, depth=depth)
    unit = _derive_draw(seed, "split", tree, node) % _DIGEST_MODULUS
    split_value = low + (high - low) * (unit / _DIGEST_MODULUS)
    left_rows = [row for row in rows if matrix.rows[row][feature_index] < split_value]
    right_rows = [row for row in rows if matrix.rows[row][feature_index] >= split_value]
    if not left_rows or not right_rows:
        return _IsolationNode(size=size, depth=depth)
    return _IsolationNode(
        size=size,
        depth=depth,
        feature_index=feature_index,
        split_value=split_value,
        left=_build_node(
            seed, tree, node * 2, left_rows, matrix, varying, depth + 1, max_depth
        ),
        right=_build_node(
            seed, tree, node * 2 + 1, right_rows, matrix, varying, depth + 1, max_depth
        ),
    )


def _path_length(node: _IsolationNode, vector: Tuple[float, ...], depth: int) -> float:
    """Length of the path a vector takes through a fitted tree."""
    if node.is_leaf:
        return depth + average_unsuccessful_path_length(node.size)
    if vector[node.feature_index] < node.split_value:
        return _path_length(node.left, vector, depth + 1)
    return _path_length(node.right, vector, depth + 1)


class FittedAnomalyModel(BaseModel):
    """An immutable fitted isolation forest plus the preprocessor it used.

    Frozen, so a scored result cannot be changed by anything that happens to
    the model afterwards, and self-describing: the schema, the seed, the
    population size and the algorithm version all travel with it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: AnomalyModelConfig
    preprocessor: Optional[FittedFeaturePreprocessor] = Field(
        default=None, description="Absent when no model was fitted."
    )
    status: ModelStatus
    trees: Tuple[_IsolationNode, ...] = Field(default=())
    subsample_size: int = Field(default=0, ge=0)
    normalizer: float = Field(
        default=0.0,
        ge=0.0,
        description="c(psi): the path length a reference profile would "
        "expect, and the denominator of every raw score.",
    )
    varying_feature_count: int = Field(
        default=0,
        ge=0,
        description="Columns that varied in the fitted population and were "
        "therefore eligible to be cut on. Zero means every profile was "
        "identical, which is reported rather than hidden.",
    )
    fitting_population_size: int = Field(default=0, ge=0)
    insufficient_reason: Optional[str] = Field(default=None)
    algorithm_version: str = ANOMALY_ALGORITHM_VERSION

    @model_validator(mode="after")
    def _check_consistency(self) -> "FittedAnomalyModel":
        if self.status is ModelStatus.FITTED:
            if self.preprocessor is None or not self.trees:
                raise ValueError("a fitted model requires a preprocessor and trees")
            if self.subsample_size < 1 or self.normalizer <= 0.0:
                raise ValueError("a fitted model requires a usable normalizer")
        else:
            if self.trees or self.preprocessor is not None:
                raise ValueError("an unfitted model must carry no fit state")
            if not (self.insufficient_reason or "").strip():
                raise ValueError("an unfitted model must state why")
        return self

    @property
    def model_id(self) -> str:
        """Content-derived model identifier."""
        return self.config.model_id

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Vector columns the model was fitted on."""
        return () if self.preprocessor is None else self.preprocessor.feature_names

    @property
    def schema_id(self) -> str:
        """Feature schema identifier, or :data:`_NO_SCHEMA` when unfitted."""
        return _NO_SCHEMA if self.preprocessor is None else self.preprocessor.schema_id

    def raw_score_of(self, vector: Sequence[float]) -> float:
        """Model-native score for one vector: mean path length over c(psi)."""
        if self.status is not ModelStatus.FITTED:
            raise ValueError("the model was not fitted, so it has no scores")
        if len(vector) != len(self.feature_names):
            raise ValueError("vector width does not match the fitted schema")
        row = tuple(float(item) for item in vector)
        total = math.fsum(_path_length(tree, row, 0) for tree in self.trees)
        return total / len(self.trees) / self.normalizer

    def metadata(self, scoring_population_size: int, in_sample: bool) -> ModelRunMetadata:
        """Build the run metadata block for a scoring call."""
        return ModelRunMetadata(
            model_id=self.config.model_id,
            algorithm=self.config.algorithm.value,
            algorithm_version=self.algorithm_version,
            feature_schema_id=self.schema_id,
            feature_names=self.feature_names,
            scaling=(
                self.preprocessor.config.scaling if self.preprocessor is not None else None
            ),
            imputation=(
                self.preprocessor.config.imputation
                if self.preprocessor is not None
                else None
            ),
            random_state=self.config.random_state,
            fitting_population_size=self.fitting_population_size,
            scoring_population_size=scoring_population_size,
            scoring_in_sample=in_sample,
            config_json=canonical_json(self.config.to_dict()),
        )

    def _degenerate_run(
        self, status: ModelStatus, reason: str, profiles: Sequence[BehavioralProfile]
    ) -> AnomalyRunResult:
        return AnomalyRunResult(
            metadata=self.metadata(len(profiles), False),
            status=status,
            results=(),
            feature_names=self.feature_names,
            band_boundaries=self.config.band_boundaries,
            insufficient_reason=reason,
        )

    def score(
        self, profiles: Sequence[BehavioralProfile], in_sample: bool = True
    ) -> AnomalyRunResult:
        """Score *profiles* against this model.

        Results are ordered by normalized score descending, then profile id
        ascending, so a run is byte-identical for identical input.
        """
        if self.status is not ModelStatus.FITTED:
            return self._degenerate_run(
                self.status,
                self.insufficient_reason or "the model was not fitted",
                profiles,
            )
        if not profiles:
            return self._degenerate_run(
                ModelStatus.INSUFFICIENT_DATA,
                "no profiles were supplied to score, so no score was produced",
                profiles,
            )
        assert self.preprocessor is not None
        ordered = _canonical_profiles(profiles)
        results = []
        for profile in ordered:
            vector = self.preprocessor.transform_one(profile)
            raw = self.raw_score_of(vector.values)
            normalized = 2.0 ** (-raw)
            results.append(
                AnomalyResult(
                    profile_id=profile.profile_id,
                    observation_ids=profile.observation_ids,
                    schema_id=self.schema_id,
                    raw_score=raw,
                    normalized_score=normalized,
                    band=band_for_score(normalized, self.config.band_boundaries),
                    is_in_sample=in_sample,
                    explanation=explain_profile(
                        self.preprocessor, profile, self.config.explanation_limit
                    ),
                )
            )
        results.sort(key=lambda item: (-float(item.normalized_score), item.profile_id))
        return AnomalyRunResult(
            metadata=self.metadata(len(ordered), in_sample),
            status=ModelStatus.FITTED,
            results=tuple(results),
            feature_names=self.feature_names,
            band_boundaries=self.config.band_boundaries,
            insufficient_reason=None,
        )

    def score_new(self, profiles: Sequence[BehavioralProfile]) -> AnomalyRunResult:
        """Score profiles that were not part of the fitted population.

        Identical arithmetic to :meth:`score`; the difference is recorded in
        the result, because a number computed against unseen data and a number
        computed against the fitted rows are not the same kind of claim.
        """
        return self.score(profiles, in_sample=False)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "algorithm_version": self.algorithm_version,
            "insufficient_reason": self.insufficient_reason,
            "config": self.config.to_dict(),
            "subsample_size": self.subsample_size,
            "normalizer": self.normalizer,
            "fitting_population_size": self.fitting_population_size,
            "feature_schema_id": self.schema_id,
            "feature_names": list(self.feature_names),
            "tree_count": len(self.trees),
            "preprocessor": self.preprocessor.to_dict() if self.preprocessor else None,
        }


class IsolationForestAnomalyDetector:
    """Fits and applies the isolation-forest unusualness model.

    One detector, one configuration, one fitted model.  Fitting and scoring are
    separate methods so that a baseline population and a later scoring
    population stay distinct, and so that a reader can see which of the two a
    given number came from.
    """

    def __init__(
        self,
        config: Optional[AnomalyModelConfig] = None,
        preprocessor_config: Optional[PreprocessorConfig] = None,
    ) -> None:
        self._config = config or AnomalyModelConfig()
        self._vectorizer = ProfileFeatureVectorizer(preprocessor_config)

    @property
    def config(self) -> AnomalyModelConfig:
        """The configuration in force."""
        return self._config

    def fit(self, profiles: Sequence[BehavioralProfile]) -> FittedAnomalyModel:
        """Fit the forest on *profiles*.

        Returns a model whose ``status`` says whether a model was produced.  An
        under-populated input is not an exception: the caller is owed a
        machine-readable refusal, not a traceback.
        """
        if not profiles:
            return FittedAnomalyModel(
                config=self._config,
                status=ModelStatus.INSUFFICIENT_DATA,
                insufficient_reason=(
                    "no profiles were supplied, so there is no population to "
                    "measure an unusual profile against"
                ),
            )
        ordered = _canonical_profiles(profiles)
        if len(ordered) < self._config.min_profiles:
            return FittedAnomalyModel(
                config=self._config,
                status=ModelStatus.INSUFFICIENT_DATA,
                insufficient_reason=(
                    "%d profile(s) were supplied but the isolation forest needs "
                    "at least %d: a single profile can only be compared with "
                    "itself, which yields no separation"
                    % (len(ordered), self._config.min_profiles)
                ),
            )
        preprocessor = self._vectorizer.fit(ordered)
        matrix = preprocessor.transform(ordered)
        subsample_size = min(self._config.max_samples, matrix.population_size)
        max_depth = self._config.resolved_max_depth(subsample_size)
        varying = _varying_features(matrix)
        trees = tuple(
            _build_node(
                seed=self._config.random_state,
                tree=tree,
                node=1,
                rows=_subsample_indices(
                    self._config.random_state, tree, matrix.population_size, subsample_size
                ),
                matrix=matrix,
                varying=varying,
                depth=0,
                max_depth=max_depth,
            )
            for tree in range(self._config.n_estimators)
        )
        return FittedAnomalyModel(
            config=self._config,
            preprocessor=preprocessor,
            status=ModelStatus.FITTED,
            trees=trees,
            subsample_size=subsample_size,
            normalizer=average_unsuccessful_path_length(subsample_size),
            varying_feature_count=len(varying),
            fitting_population_size=matrix.population_size,
            insufficient_reason=None,
        )

    def fit_score(
        self, profiles: Sequence[BehavioralProfile]
    ) -> Tuple[FittedAnomalyModel, AnomalyRunResult]:
        """Fit on *profiles* and immediately score them in sample.

        A convenience for the common single-population case.  The result is
        marked in sample, so nothing here is mistaken for a generalization to
        data the model has not seen.
        """
        model = self.fit(profiles)
        return model, model.score(profiles, in_sample=True)
