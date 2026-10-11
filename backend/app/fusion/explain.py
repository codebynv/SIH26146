"""Deterministic explanation items for a fusion assessment (SIH26146 - Step 8).

Every sentence a Step 8 assessment can produce is generated here from the
numbers themselves, so an explanation can never drift away from the score it
describes.  The fixed notes - cluster context, inference levels, overlap, and
the interpretation statement - are standing disclosures, not judgements:
they say what the score is and is not, and they are always emitted so the
reader is never left to infer the limits from an absence.
"""

from __future__ import annotations

from typing import Tuple

from ..fusion.models import (
    ExplainabilityKind,
    FusionConfig,
    FusionExplanation,
    FusionSignal,
    SignalContribution,
    SignalStatus,
)

__all__ = ["build_explanation"]

_CLUSTER_CONTEXT_NOTE = (
    "Cluster membership is descriptive context only: it states which group a "
    "profile sits in geometrically and never asserts that its members are the "
    "same actor; no relationship or identity claim is created from it."
)

_INFERENCE_LEVEL_NOTE = (
    "Inference levels recorded by earlier steps (OBSERVED, CORRELATED, "
    "INFERRED, REQUIRES_VERIFICATION) are carried verbatim through this "
    "assessment and are never upgraded because signals agree."
)

_OVERLAP_NOTE = (
    "A relationship id may appear in more than one family's lineage; the "
    "families are not assumed independent, so counts are reported per family "
    "and the same id may be cited in several signals."
)


def _contribution_sentence(contribution: SignalContribution) -> str:
    return (
        "Family %s contributed %s to the fused score under the disclosed "
        "weights (value %s, availability %s, %d of %d scoring signals present)."
        % (
            contribution.family.value,
            "{:.6f}".format(contribution.contribution),
            "{:.6f}".format(contribution.normalized_value),
            "{:.6f}".format(contribution.availability),
            contribution.present_count,
            contribution.total_count,
        )
    )


def build_explanation(
    *,
    contributions: Tuple[SignalContribution, ...],
    signals: Tuple[FusionSignal, ...],
    config: FusionConfig,
) -> Tuple[FusionExplanation, ...]:
    """Build the full, ordered explanation set for one assessment.

    ``contributions`` and ``signals`` are the assessment's own recorded
    objects, so the sentences always match the arithmetic.
    """
    items: list[FusionExplanation] = []

    measured = tuple(
        contribution
        for contribution in contributions
        if contribution.present_count > 0
    )
    if measured:
        top = max(measured, key=lambda item: item.contribution)
        items.append(
            FusionExplanation(
                kind=ExplainabilityKind.TOP_CONTRIBUTOR,
                text=(
                    "The %s family contributed most to this score, %s; this "
                    "names the largest disclosed term, not a verdict about the "
                    "subject."
                    % (top.family.value, "{:.6f}".format(top.contribution))
                ),
                evidence=top.signal_ids,
            )
        )

    for contribution in contributions:
        items.append(
            FusionExplanation(
                kind=ExplainabilityKind.SIGNAL_CONTRIBUTION,
                text=_contribution_sentence(contribution),
                evidence=contribution.signal_ids,
            )
        )

    for signal in signals:
        if signal.status is not SignalStatus.PRESENT:
            items.append(
                FusionExplanation(
                    kind=ExplainabilityKind.MISSING_SIGNAL,
                    text=(
                        "Signal %s was %s and therefore contributed nothing; "
                        "a missing signal is never replaced by a zero.  %s"
                        % (
                            signal.signal_type,
                            signal.status.value,
                            signal.explanation,
                        )
                    ),
                )
            )

    items.append(
        FusionExplanation(
            kind=ExplainabilityKind.CLUSTER_CONTEXT_NOTE, text=_CLUSTER_CONTEXT_NOTE
        )
    )
    items.append(
        FusionExplanation(
            kind=ExplainabilityKind.INFERENCE_LEVEL_NOTE,
            text=_INFERENCE_LEVEL_NOTE,
        )
    )
    items.append(
        FusionExplanation(
            kind=ExplainabilityKind.OVERLAP_NOTE, text=_OVERLAP_NOTE
        )
    )
    items.append(
        FusionExplanation(
            kind=ExplainabilityKind.INTERPRETATION_NOTE, text=config.interpretation
        )
    )

    items.sort(key=lambda item: (item.kind.value, item.text))
    return tuple(items)