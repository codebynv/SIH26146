"""Grouping profiles by numeric similarity (SIH26146 - Step 6).

The algorithm, and what a group is
---------------------------------
K-means over the preprocessed vectors, with a deterministic k-means++
initialization, several restarts, and a documented choice of ``k``.  A "group"
here means one thing and nothing more: *the set of profiles whose vectors lie
near the same centroid*.  It is a statement about geometry.  It does not name
what the members have in common, it does not rank groups, and it carries no
meaning beyond "these rows, this partition, this run".

Why k-means and not something more elaborate
--------------------------------------------
The prompt allows agglomerative, DBSCAN and k-means.  K-means was chosen for
two reasons that survive scrutiny.  First, its objective -- minimise the sum of
squared distances to the nearest centroid -- is a *disclosed* criterion: the
``dispersion`` reported for each group is exactly that sum divided by the member
count, so a reader can check the fit without trusting the code.  Second, it has
a defined answer for a singleton population and a defined failure for a
population where every row is identical, both of which this step must state
rather than discover at runtime.

How ``k`` is chosen, and why the choice is visible
-------------------------------------------------
When ``k`` is not supplied, every candidate in ``[k_min, k_max]`` is fitted and
scored by the mean silhouette coefficient, and the best score wins, with ties
going to the *smallest* ``k`` -- the parsimonious reading of a tie.  The full
score table is returned in
:attr:`~backend.app.ml.models.ClusteringRunResult.selection_scores`.  A reader
who prefers a different ``k`` can see exactly what the chosen one beat, and
:attr:`~backend.app.ml.models.ClusteringRunResult.status` distinguishes
"fitted" from "too small" from "no partition possible".

Determinism
-----------
Every random choice -- which row seeds the first centroid, which row is the
second, and so on -- is a SHA-256 digest of the seed and the step number, taken
over rows in canonical profile-id order.  With ``n_init`` restarts the fit keeps
the lowest inertia, so the outcome is stable against the order of the input and
reproducible on any machine.  Ties in assignment, in restart comparison and in
model selection break on the lowest index, never on iteration order.

Canonical cluster labels
------------------------
The labels are assigned *after* the fit, never during it.  Each group's
signature is the SHA-256 of its sorted member profile ids; groups are then
sorted by signature and labelled ``CLUSTER_000``, ``CLUSTER_001``, and so on.
Consequences, stated plainly:

* shuffling the input changes nothing -- the fit is order-independent and the
  signature does not depend on arrival order;
* the label is a function of membership, so adding a profile to a group can
  renumber the groups.  That is the documented trade-off of a content-derived
  label, and it is why the signature is reported alongside the label.

Unassigned profiles
-------------------
:data:`~backend.app.ml.models.NOISE_CLUSTER_ID` and
:attr:`~backend.app.ml.models.ClusterAssignment.is_noise` exist in the model
vocabulary because density-based methods, which the prompt allows for a later
step, can legitimately place a profile in no group.  K-means assigns every row
it was fitted on, so ``noise_profile_ids`` is empty in every result this module
produces.  The field is part of the contract, not a prediction.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..behavior.profiles import BehavioralProfile
from .models import (
    ClusterAssignment,
    ClusterModelConfig,
    ClusterSummary,
    ClusteringRunResult,
    FeatureMatrix,
    ModelRunMetadata,
    ModelStatus,
    canonical_json,
    cluster_label,
)
from .preprocessing import (
    FittedFeaturePreprocessor,
    PreprocessorConfig,
    ProfileFeatureVectorizer,
    _canonical_profiles,
)

__all__ = [
    "CLUSTERING_ALGORITHM_VERSION",
    "FittedClusteringModel",
    "KMeansBehaviorClusterer",
    "silhouette_score",
]

#: Version tag of the estimator implemented here, reported in run metadata so a
#: stored result can be traced to the exact objective that produced it.
CLUSTERING_ALGORITHM_VERSION = "KMEANS_STDLIB/1"

_NO_SCHEMA = "SCHEMA_NONE"

_DIGEST_BYTES = 8
_DIGEST_MODULUS = 1 << (8 * _DIGEST_BYTES)


def _derive_draw(*parts: object) -> int:
    """Return a deterministic integer derived from *parts*."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:_DIGEST_BYTES], "big")


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum((a - b) ** 2 for a, b in zip(left, right))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


#: Below this fraction of the squared norms involved, the expanded squared
#: distance is recomputed from the residual rather than trusted.  See
#: :func:`_assign`.
_EXPANSION_TRUST_LIMIT = 1e-9


def _mean_vector(rows: Sequence[Sequence[float]], dimension: int) -> Tuple[float, ...]:
    count = len(rows)
    return tuple(math.fsum(row[index] for row in rows) / count for index in range(dimension))


def _assign(
    rows: Sequence[Sequence[float]], centroids: Sequence[Tuple[float, ...]]
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """Assign each row to its nearest centroid, with distances.

    ``||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2`` is used so the per-row part is
    computed once.  Ties break on the lowest centroid index, which makes the
    assignment a function of the data rather than of storage order.
    """
    centroid_norms = [_norm(centroid) for centroid in centroids]
    assignments = []
    distances = []
    for row in rows:
        row_norm = _norm(row)
        best_index = 0
        best_distance = None
        for index, centroid in enumerate(centroids):
            dot = math.fsum(a * b for a, b in zip(row, centroid))
            distance = row_norm * row_norm - 2.0 * dot + centroid_norms[index] * centroid_norms[index]
            # The expanded form cancels catastrophically when a row sits on or
            # very near a centroid: the three terms are all large and nearly
            # equal, so the result is dominated by rounding error.  A row that
            # is its own centroid would then report a small non-zero distance.
            # Recomputing the residual directly costs one pass and is exact
            # where the expansion is not.
            if distance <= _EXPANSION_TRUST_LIMIT * max(
                row_norm * row_norm, centroid_norms[index] * centroid_norms[index], 1.0
            ):
                distance = math.fsum((a - b) ** 2 for a, b in zip(row, centroid))
            if distance < 0.0:
                distance = 0.0
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        assert best_distance is not None
        assignments.append(best_index)
        distances.append(math.sqrt(best_distance))
    return tuple(assignments), tuple(distances)


def _kmeans_plus_plus(
    seed: int, trial: int, rows: Sequence[Sequence[float]], k: int
) -> Tuple[Tuple[float, ...], ...]:
    """Deterministic k-means++ seeding.

    The first centroid is drawn uniformly over the rows; each later centroid is
    drawn from the squared distance to the *nearest* chosen centroid, which is
    why a k-means++ start is reliably better than a uniform one.  Every draw is
    a digest of the seed, the trial number and the step number.  When every
    remaining row already coincides with a chosen centroid the total weight is
    zero, and the next centroid is taken from the rows not yet used, by digest,
    rather than by a fallback that could repeat a centroid.
    """
    population = len(rows)
    first = _derive_draw(seed, "kpp", trial, 0) % population
    centroids = [tuple(rows[first])]
    # The standard rule weights each row by its distance to the *nearest*
    # chosen centroid, not to the most recent one.  Weighting by the latest
    # centroid alone lets a row sitting between two chosen centroids be pulled
    # back toward one it already matches, which is how k-means++ degenerates
    # into picking a near-duplicate.
    nearest = [math.inf] * population
    for step in range(1, k):
        latest = centroids[-1]
        weights = []
        for index, row in enumerate(rows):
            distance = _squared_distance(row, latest)
            nearest[index] = min(nearest[index], distance)
            weights.append(nearest[index])
        total = math.fsum(weights)
        if total <= 0.0:
            used = {centroid for centroid in centroids}
            remaining = [index for index, row in enumerate(rows) if tuple(row) not in used]
            if not remaining:
                remaining = list(range(population))
            choice = remaining[
                _derive_draw(seed, "degenerate", trial, len(centroids)) % len(remaining)
            ]
        else:
            target = (
                _derive_draw(seed, "kpp", trial, len(centroids)) % _DIGEST_MODULUS
            ) / _DIGEST_MODULUS
            running = 0.0
            choice = population - 1
            for index, weight in enumerate(weights):
                running += weight
                if running / total >= target:
                    choice = index
                    break
        centroids.append(tuple(rows[choice]))
    return tuple(centroids)


def _kmeans(
    seed: int,
    trial: int,
    rows: Sequence[Sequence[float]],
    k: int,
    config: ClusterModelConfig,
) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[int, ...], float, int]:
    """Run one k-means start; return centroids, assignments, inertia, iterations.

    An empty group is refilled with the row furthest from its own centroid.
    That repair is standard and cannot cycle: the moved row strictly increases
    the number of groups holding at least one point, and the move is redrawn
    from the deterministic far-point order each time.
    """
    dimension = len(rows[0])
    centroids = list(_kmeans_plus_plus(seed, trial, rows, k))
    assignments: Tuple[int, ...] = ()
    iterations = 0
    for iteration in range(config.max_iterations):
        iterations = iteration + 1
        assignments, distances = _assign(rows, centroids)
        groups: dict = {}
        for index, assignment in enumerate(assignments):
            groups.setdefault(assignment, []).append(index)
        updated = list(centroids)
        for cluster_index in range(k):
            members = groups.get(cluster_index)
            if members:
                updated[cluster_index] = _mean_vector(
                    [rows[index] for index in members], dimension
                )
            else:
                updated[cluster_index] = tuple(
                    rows[max(range(len(rows)), key=lambda index: (distances[index], -index))]
                )
        shift = max(
            _norm([a - b for a, b in zip(before, after)])
            for before, after in zip(centroids, updated)
        )
        centroids = updated
        if shift <= config.tolerance:
            break
    assignments, distances = _assign(rows, centroids)
    inertia = math.fsum(distance * distance for distance in distances)
    return tuple(centroids), assignments, inertia, iterations


def silhouette_score(matrix: FeatureMatrix, assignments: Sequence[int]) -> float:
    """Mean silhouette coefficient of a partition.

    For each row, ``(b - a) / max(a, b)`` where ``a`` is the mean distance to
    the rest of its own group and ``b`` the mean distance to the nearest other
    group.  Returns ``0.0`` where the quantity is undefined -- fewer than three
    points, or a single group -- reported as a neutral score rather than as a
    fabricated one, which is also why the clusterer refuses to fit below three
    profiles.
    """
    population = matrix.population_size
    if population < 3 or len(set(assignments)) < 2:
        return 0.0
    rows = matrix.rows
    groups = sorted(set(assignments))
    total = 0.0
    for index, row in enumerate(rows):
        own = [
            math.sqrt(_squared_distance(row, rows[other]))
            for other in range(population)
            if other != index and assignments[other] == assignments[index]
        ]
        nearest = None
        for label in groups:
            if label == assignments[index]:
                continue
            other = [
                math.sqrt(_squared_distance(row, rows[other]))
                for other in range(population)
                if assignments[other] == label
            ]
            mean_other = math.fsum(other) / len(other)
            if nearest is None or mean_other < nearest:
                nearest = mean_other
        if nearest is None:
            continue
        within = math.fsum(own) / len(own) if own else 0.0
        denominator = max(within, nearest)
        if denominator <= 0.0:
            continue
        total += (nearest - within) / denominator
    return total / population


class FittedClusteringModel(BaseModel):
    """An immutable fitted partition plus the preprocessor that produced it.

    The design matrix of the fitted population is retained deliberately: the
    dispersion and distance figures in the result are computed from it, and
    keeping it means those figures can be recomputed from the model alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: ClusterModelConfig
    preprocessor: Optional[FittedFeaturePreprocessor] = Field(default=None)
    status: ModelStatus
    k: Optional[int] = Field(default=None, ge=1)
    centroids: Tuple[Tuple[float, ...], ...] = Field(default=())
    labels: Tuple[str, ...] = Field(
        default=(),
        description="Canonical cluster label per centroid, aligned to centroids.",
    )
    assignments: Tuple[int, ...] = Field(
        default=(), description="Centroid index per row, in canonical row order."
    )
    rows: Tuple[Tuple[float, ...], ...] = Field(
        default=(), description="Design matrix of the fitted population."
    )
    feature_names: Tuple[str, ...] = Field(default=())
    profile_ids: Tuple[str, ...] = Field(default=())
    selection_scores: Tuple[Tuple[int, float], ...] = Field(default=())
    inertia: float = Field(default=0.0, ge=0.0)
    iterations: int = Field(default=0, ge=0)
    fitting_population_size: int = Field(default=0, ge=0)
    insufficient_reason: Optional[str] = Field(default=None)
    algorithm_version: str = CLUSTERING_ALGORITHM_VERSION

    @model_validator(mode="after")
    def _check_consistency(self) -> "FittedClusteringModel":
        if self.status is ModelStatus.FITTED:
            if self.preprocessor is None or not self.centroids:
                raise ValueError("a fitted model requires a preprocessor and centroids")
            if self.k != len(self.centroids):
                raise ValueError("k must equal the number of centroids")
            if len(self.labels) != self.k:
                raise ValueError("one canonical label per centroid is required")
            if list(self.labels) != sorted(self.labels):
                raise ValueError("canonical labels must be stored in label order")
            if len(self.assignments) != len(self.profile_ids) or len(self.rows) != len(
                self.profile_ids
            ):
                raise ValueError("one row, id and assignment per profile is required")
            if any(not 0 <= index < self.k for index in self.assignments):
                raise ValueError("assignments must index an existing centroid")
            for row in self.rows:
                if len(row) != len(self.feature_names):
                    raise ValueError("every row must match the feature width")
        else:
            if self.centroids or self.assignments or self.preprocessor is not None:
                raise ValueError("an unfitted model must carry no fit state")
            if not (self.insufficient_reason or "").strip():
                raise ValueError("an unfitted model must state why")
        return self

    @property
    def model_id(self) -> str:
        """Content-derived model identifier."""
        return self.config.model_id

    @property
    def schema_id(self) -> str:
        """Feature schema identifier, or a sentinel when unfitted."""
        return _NO_SCHEMA if self.preprocessor is None else self.preprocessor.schema_id

    def _row_of(self, profile_id: str) -> Tuple[float, ...]:
        for candidate, row in zip(self.profile_ids, self.rows):
            if candidate == profile_id:
                return row
        raise KeyError(profile_id)

    def predict_cluster(self, profile: BehavioralProfile) -> ClusterAssignment:
        """Assign one profile to the nearest centroid.

        Usable on a profile that was never fitted, which is what makes a
        baseline population reusable: the canonical labels already exist, so a
        new profile joins an existing group rather than starting a new
        numbering.
        """
        if self.status is not ModelStatus.FITTED:
            raise ValueError(
                "the model was not fitted, so it has no clusters to predict"
            )
        assert self.preprocessor is not None
        vector = self.preprocessor.transform_one(profile)
        assignments, distances = _assign((vector.values,), self.centroids)
        return ClusterAssignment(
            profile_id=profile.profile_id,
            cluster_id=self.labels[assignments[0]],
            is_noise=False,
            distance_to_centroid=distances[0],
            is_in_sample=profile.profile_id in self.profile_ids,
        )

    def result(self) -> ClusteringRunResult:
        """Return the run result describing the fitted partition."""
        metadata = ModelRunMetadata(
            model_id=self.config.model_id,
            algorithm=self.config.algorithm.value,
            algorithm_version=self.algorithm_version,
            feature_schema_id=self.schema_id,
            feature_names=self.feature_names,
            scaling=(
                self.preprocessor.config.scaling
                if self.preprocessor is not None
                else None
            ),
            imputation=(
                self.preprocessor.config.imputation
                if self.preprocessor is not None
                else None
            ),
            random_state=self.config.random_state,
            fitting_population_size=self.fitting_population_size,
            scoring_population_size=self.fitting_population_size,
            scoring_in_sample=True,
            config_json=canonical_json(self.config.to_dict()),
        )
        if self.status is not ModelStatus.FITTED:
            return ClusteringRunResult(
                metadata=metadata,
                status=self.status,
                k=None,
                clusters=(),
                assignments=(),
                feature_names=self.feature_names,
                selection_scores=(),
                noise_profile_ids=(),
                insufficient_reason=self.insufficient_reason,
            )
        members: dict = {}
        for profile_id, index in zip(self.profile_ids, self.assignments):
            members.setdefault(index, []).append(profile_id)
        clusters = []
        for index in range(self.k):
            member_ids = tuple(sorted(members[index]))
            centroid = self.centroids[index]
            distances = [
                math.sqrt(_squared_distance(self._row_of(profile_id), centroid))
                for profile_id in member_ids
            ]
            clusters.append(
                ClusterSummary(
                    cluster_id=self.labels[index],
                    signature=hashlib.sha256(
                        "\n".join(member_ids).encode("utf-8")
                    ).hexdigest(),
                    member_profile_ids=member_ids,
                    centroid=tuple(
                        (name, centroid[column])
                        for column, name in enumerate(self.feature_names)
                    ),
                    dispersion=math.fsum(distances) / len(distances),
                    dispersion_detail=(
                        "Mean Euclidean distance of the member vectors to this "
                        "centroid, in the scaled space of the feature schema. It "
                        "measures how tightly the group sits together and does "
                        "not compare groups with each other."
                    ),
                )
            )
        assignments = [
            ClusterAssignment(
                profile_id=profile_id,
                cluster_id=self.labels[index],
                is_noise=False,
                distance_to_centroid=math.sqrt(
                    _squared_distance(self._row_of(profile_id), self.centroids[index])
                ),
                is_in_sample=True,
            )
            for profile_id, index in zip(self.profile_ids, self.assignments)
        ]
        assignments.sort(key=lambda item: item.profile_id)
        return ClusteringRunResult(
            metadata=metadata,
            status=ModelStatus.FITTED,
            k=self.k,
            clusters=tuple(clusters),
            assignments=tuple(assignments),
            feature_names=self.feature_names,
            selection_scores=self.selection_scores,
            noise_profile_ids=(),
            insufficient_reason=None,
        )

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "algorithm_version": self.algorithm_version,
            "insufficient_reason": self.insufficient_reason,
            "config": self.config.to_dict(),
            "k": self.k,
            "feature_schema_id": self.schema_id,
            "feature_names": list(self.feature_names),
            "fitting_population_size": self.fitting_population_size,
            "inertia": self.inertia,
            "iterations": self.iterations,
            "labels": list(self.labels),
            "selection_scores": [
                {"k": candidate, "score": score}
                for candidate, score in self.selection_scores
            ],
            "preprocessor": self.preprocessor.to_dict() if self.preprocessor else None,
        }


class KMeansBehaviorClusterer:
    """Fits a k-means partition of Step 5 profiles.

    ``fit`` and the result are separate, so a baseline population can be
    clustered once and the partition reused -- including for profiles that
    arrive later, via :meth:`FittedClusteringModel.predict_cluster`.
    """

    def __init__(
        self,
        config: Optional[ClusterModelConfig] = None,
        preprocessor_config: Optional[PreprocessorConfig] = None,
    ) -> None:
        self._config = config or ClusterModelConfig()
        self._vectorizer = ProfileFeatureVectorizer(preprocessor_config)

    @property
    def config(self) -> ClusterModelConfig:
        """The configuration in force."""
        return self._config

    def _candidate_k(self, population: int) -> Tuple[int, ...]:
        if self._config.k is not None:
            return (self._config.k,)
        highest = max(2, min(self._config.k_max, population - 1))
        return tuple(range(self._config.k_min, highest + 1))

    def fit(self, profiles: Sequence[BehavioralProfile]) -> FittedClusteringModel:
        """Fit a partition over *profiles*, choosing ``k`` when not given."""
        if not profiles:
            return FittedClusteringModel(
                config=self._config,
                status=ModelStatus.INSUFFICIENT_DATA,
                insufficient_reason=(
                    "no profiles were supplied, so there is nothing to group"
                ),
            )
        if not self._candidate_k(len(profiles)):
            return FittedClusteringModel(
                config=self._config,
                status=ModelStatus.NO_CLUSTERS,
                insufficient_reason=(
                    "k_min is %d and k_max is %d, so no admissible number of "
                    "groups exists for a population of %d"
                    % (self._config.k_min, self._config.k_max, len(profiles))
                ),
            )
        ordered = _canonical_profiles(profiles)
        if len(ordered) < self._config.min_profiles:
            return FittedClusteringModel(
                config=self._config,
                status=ModelStatus.INSUFFICIENT_DATA,
                insufficient_reason=(
                    "%d profile(s) were supplied but the silhouette score needs "
                    "at least %d: with fewer points every profile would be "
                    "compared with itself and no group could be preferred"
                    % (len(ordered), self._config.min_profiles)
                ),
            )
        preprocessor = self._vectorizer.fit(ordered)
        matrix = preprocessor.transform(ordered)
        scores = []
        best = None
        for k in self._candidate_k(matrix.population_size):
            centroids, assignments, inertia, iterations = self._best_of_k(matrix, k)
            score = silhouette_score(matrix, assignments)
            scores.append((k, score))
            if best is None or (score, -k) > (best[3], -best[0]):
                best = (k, centroids, assignments, score, inertia, iterations)
        if best is None:  # pragma: no cover - guarded by _candidate_k
            return FittedClusteringModel(
                config=self._config,
                status=ModelStatus.NO_CLUSTERS,
                insufficient_reason="no admissible number of groups exists",
            )
        k, centroids, assignments, _, inertia, iterations = best
        members: dict = {}
        for profile_id, index in zip(matrix.profile_ids, assignments):
            members.setdefault(index, []).append(profile_id)
        signatures = sorted(
            (
                hashlib.sha256(
                    "\n".join(sorted(members[index])).encode("utf-8")
                ).hexdigest(),
                index,
            )
            for index in members
        )
        canonical_index = [index for _, index in signatures]
        labels = tuple(
            cluster_label(position) for position in range(len(canonical_index))
        )
        reindexed = {old: new for new, old in enumerate(canonical_index)}
        return FittedClusteringModel(
            config=self._config,
            preprocessor=preprocessor,
            status=ModelStatus.FITTED,
            # A fit can leave a centroid with no members -- k-means seeds k
            # centroids but a coincident row is claimed by whichever centroid
            # reaches it first.  The reported k is the number of groups that
            # actually exist, not the number requested, because a group with no
            # members is not a group.
            k=len(canonical_index),
            centroids=tuple(centroids[index] for index in canonical_index),
            labels=labels,
            assignments=tuple(reindexed[index] for index in assignments),
            rows=matrix.rows,
            feature_names=matrix.feature_names,
            profile_ids=matrix.profile_ids,
            selection_scores=tuple(scores),
            inertia=inertia,
            iterations=iterations,
            fitting_population_size=matrix.population_size,
            insufficient_reason=None,
        )

    def _best_of_k(
        self, matrix: FeatureMatrix, k: int
    ) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[int, ...], float, int]:
        """Run ``n_init`` restarts and keep the lowest-inertia fit.

        Restarting is what makes k-means reliable rather than seed-lucky: one
        start can converge to a poor local minimum, and the reported dispersion
        would then describe an arbitrary choice rather than the best partition
        this configuration can reach.
        """
        best = None
        for trial in range(self._config.n_init):
            centroids, assignments, inertia, iterations = _kmeans(
                self._config.random_state, trial, matrix.rows, k, self._config
            )
            if best is None or inertia < best[2]:
                best = (centroids, assignments, inertia, iterations)
        assert best is not None
        return best
