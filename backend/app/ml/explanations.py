"""Explaining an unusualness number (SIH26146 - Step 6).

What an explanation is here
---------------------------
Each column of a profile's vector is compared with the fitted reference for
that same column, and the deviations are reported strongest first.  That is the
whole of it.  It is a *description of a position*: "this count is far above the
middle of the fitted population" is a true statement about a number, and it is
the only kind of statement this module makes.

What an explanation is not
--------------------------
It is not a decomposition of the anomaly score, and the ratios are not claimed
to be shares of anything.  An isolation forest computes a score by averaging
path lengths through random splits; no arithmetic relationship exists between
that number and any individual column, so apportioning it across columns would
be arithmetic theatre.  For the same reason:

* no causal claim -- a high ``total_input_amount`` is not shown to have caused
  the profile to be unusual;
* no importance claim -- a column at the top of this list is not shown to be
  what the model "relied on";
* no SHAP-style value -- there is no game-theoretic attribution here, and
  :data:`~backend.app.ml.models.ExplanationMethod.MODEL_NATIVE` is documented as
  unavailable for this model rather than approximated with invented numbers.

The method is recorded on every explanation
--------------------------------------------
:data:`ExplanationMethod.STATISTICAL_REFERENCE` is set on everything produced
here, and :meth:`AnomalyExplanation.method_detail` states in words what the
numbers are and are not.  A reader who skips the numbers should still be unable
to mistake the output for an attribution.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from .models import (
    AnomalyExplanation,
    AnomalyFeatureContribution,
    DeviationDirection,
    ExplanationMethod,
)
from .preprocessing import FittedFeaturePreprocessor

__all__ = [
    "EXPLANATION_METHOD_DETAIL",
    "explain_profile",
    "explain_profile_vector",
]

#: The sentence attached to every explanation this module produces.  It is a
#: constant rather than a formatted string so that a reader comparing two runs
#: is not asked to diff prose.
EXPLANATION_METHOD_DETAIL = (
    "Each feature is compared with the median of the population it was fitted "
    "on and scaled by that population's spread. The ranking describes where "
    "this profile sits on those columns. It does not decompose the unusualness "
    "score, does not measure how much each feature contributed to it, and "
    "supports no causal conclusion."
)


def _contribution(
    rank: int,
    feature_name: str,
    observed: float,
    reference: float,
    scale: float,
    unit: Optional[str],
) -> AnomalyFeatureContribution:
    deviation = (observed - reference) / scale
    if not math.isfinite(deviation):
        raise ValueError(
            "feature %r produced a non-finite deviation" % feature_name
        )
    direction = (
        DeviationDirection.ABOVE_REFERENCE
        if deviation > 0
        else DeviationDirection.BELOW_REFERENCE
    )
    return AnomalyFeatureContribution(
        rank=rank,
        feature_name=feature_name,
        observed_value=observed,
        reference_value=reference,
        deviation=deviation,
        absolute_deviation=abs(deviation),
        direction=direction,
        reference_unit=unit,
    )


def _unit_for(preprocessor: FittedFeaturePreprocessor, feature_name: str) -> Optional[str]:
    try:
        return preprocessor.unit_of(feature_name)
    except KeyError:  # pragma: no cover - defensive
        return None


def explain_profile_vector(
    preprocessor: FittedFeaturePreprocessor,
    profile_id: str,
    observed_values: Sequence[Tuple[str, float]],
    limit: int = 5,
) -> AnomalyExplanation:
    """Rank one profile's column deviations against the fitted reference.

    *observed_values* are the profile's values in their own units, aligned to
    the fitted schema, with unavailable measurements already filled from the
    fitted reference.  Ties in absolute deviation are broken by feature name so
    that two runs of the same data rank the columns identically.
    """
    if limit < 0:
        raise ValueError("limit must not be negative")
    if not isinstance(preprocessor, FittedFeaturePreprocessor):
        raise TypeError("preprocessor must be a FittedFeaturePreprocessor")
    supplied = dict(observed_values)
    missing = [name for name in preprocessor.feature_names if name not in supplied]
    if missing:
        raise ValueError("observed values are missing schema columns: %s" % missing)
    scored = []
    for name in preprocessor.feature_names:
        reference = preprocessor.center_of(name)
        scale = preprocessor.scale_of(name)
        observed = float(supplied[name])
        deviation = (observed - reference) / scale
        scored.append((-abs(deviation), name, observed, reference, scale))
    scored.sort()
    total = len(scored)
    reported = scored[:limit]
    contributions = tuple(
        _contribution(
            rank=index,
            feature_name=name,
            observed=observed,
            reference=reference,
            scale=scale,
            unit=_unit_for(preprocessor, name),
        )
        for index, (_, name, observed, reference, scale) in enumerate(reported, start=1)
    )
    return AnomalyExplanation(
        profile_id=profile_id,
        method=ExplanationMethod.STATISTICAL_REFERENCE,
        method_detail=EXPLANATION_METHOD_DETAIL,
        contributions=contributions,
        truncated=len(contributions) < total,
    )


def explain_profile(
    preprocessor: FittedFeaturePreprocessor,
    profile: object,
    limit: int = 5,
) -> AnomalyExplanation:
    """Rank one Step 5 profile's column deviations against the fit.

    Reads the profile through the fitted preprocessor, so the values compared
    here are exactly the values the model scored.
    """
    return explain_profile_vector(
        preprocessor,
        profile.profile_id,
        preprocessor.raw_values_for(profile),
        limit=limit,
    )
