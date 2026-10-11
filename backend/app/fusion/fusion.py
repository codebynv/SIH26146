"""The weighted evidence-fusion engine (SIH26146 - Step 8).

``FusionEngine`` is deliberately small.  The measuring of earlier layers lives
in :mod:`..fusion.signals`, the arithmetic vocabulary in
:mod:`..fusion.models`, and the sentences in :mod:`..fusion.explain`.  What is
left here is the single deterministic reduction: read the signals, weight and
average them under the disclosed configuration, and produce a frozen,
self-verifying :class:`~backend.app.fusion.models.FusionAssessment`.

The fused score is ``sum(weight * value * availability) / sum(weight *
availability)``.  The only divider between "a number was computed" and "no
number is invented" is the weighted-availability denominator: when every
family that carries weight produced no PRESENT signal, the status is
``INSUFFICIENT_SIGNALS`` and no score or band is reported.
"""

from __future__ import annotations

import decimal
from typing import Optional, Tuple

from ..correlation.enums import InferenceLevel
from ..fusion.explain import build_explanation
from ..fusion.models import (
    FUSION_DECIMAL_CONTEXT,
    ConfidenceContext,
    FusionAssessment,
    FusionConfig,
    FusionError,
    FusionStatus,
    FusionSignal,
    MissingSignal,
    SignalContribution,
    SignalFamily,
    SignalRole,
    SignalStatus,
    band_for_score,
    derive_assessment_id,
)
from ..fusion.signals import FusionContext, build_signals, incident_edges

__all__ = ["FusionEngine"]


class FusionEngine:
    """Assess one profile against the producers of Steps 4-7 under one policy.

    The engine owns no mutable state between calls besides its immutable
    configuration: assessing the same context twice produces the same
    assessment, byte for byte.
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        if config is None:
            self._config = FusionConfig()
        elif isinstance(config, FusionConfig):
            self._config = config
        else:
            self._config = FusionConfig.model_validate(config)

    @property
    def config(self) -> FusionConfig:
        """The immutable policy this engine assesses under."""
        return self._config

    def assess(self, context: FusionContext) -> FusionAssessment:
        """Produce the full, frozen assessment for *context*.

        :raises FusionError: if the context is not a
            :class:`FusionContext` or the adapters collide.
        """
        if not isinstance(context, FusionContext):
            raise FusionError(
                "FusionEngine.assess needs a FusionContext, got %s"
                % type(context).__name__
            )
        config = self._config
        profile = context.profile
        signals = build_signals(
            context,
            evidence_count_cap=config.evidence_count_cap,
            path_count_cap=config.path_count_cap,
            provenance_file_cap=config.provenance_file_cap,
        )
        contributions = self._contributions(signals, config)
        with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
            numerator = sum(
                (item.contribution for item in contributions), decimal.Decimal("0")
            )
            denominator = sum(
                (item.weight * item.availability for item in contributions),
                decimal.Decimal("0"),
            )
        if denominator <= 0:
            status = FusionStatus.INSUFFICIENT_SIGNALS
            fused_score: Optional[decimal.Decimal] = None
            risk_band = None
        else:
            status = FusionStatus.COMPUTED
            with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
                fused_score = numerator / denominator
            risk_band = band_for_score(
                fused_score,
                (config.band_low, config.band_medium, config.band_high),
            )

        present = tuple(
            signal for signal in signals if signal.status is SignalStatus.PRESENT
        )
        missing_signals = tuple(
            MissingSignal(
                signal_id=signal.signal_id,
                signal_type=signal.signal_type,
                family=signal.family,
                status=signal.status,
                source_step=signal.source_step,
                explanation=signal.explanation,
            )
            for signal in signals
            if signal.status is not SignalStatus.PRESENT
        )
        inference_levels: Tuple[InferenceLevel, ...] = tuple(
            sorted(
                {
                    level
                    for signal in present
                    for level in signal.inference_levels
                },
                key=lambda level: level.value,
            )
        )
        distinct_relationship_ids: Tuple[str, ...] = tuple(
            sorted(
                {
                    relationship_id
                    for signal in present
                    for relationship_id in signal.evidence_refs
                }
            )
        )
        confidence = self._confidence_context(context)
        explanation = build_explanation(
            contributions=contributions, signals=signals, config=config
        )

        config_json = config.to_json()
        signal_ids = [signal.signal_id for signal in signals]
        assessment_id = derive_assessment_id(
            profile.profile_id, config_json, signal_ids
        )
        return FusionAssessment(
            assessment_id=assessment_id,
            subject_profile_id=profile.profile_id,
            subject_observation_ids=profile.observation_ids,
            config_version=config.version,
            config_json=config_json,
            status=status,
            fused_score=fused_score,
            risk_band=risk_band,
            interpretation=config.interpretation,
            signal_count=len(signals),
            present_signal_count=len(present),
            signals=signals,
            contributions=contributions,
            missing_signals=missing_signals,
            inference_levels=inference_levels,
            confidence=confidence,
            distinct_relationship_ids=distinct_relationship_ids,
            explanation=explanation,
        )

    @staticmethod
    def _contributions(
        signals: Tuple[FusionSignal, ...], config: FusionConfig
    ) -> Tuple[SignalContribution, ...]:
        """One per-family breakdown, ascending by family.

        ``normalized_value`` is the mean of the family's PRESENT normalized
        values, ``availability`` the PRESENT count over the family's emitted
        count, and ``contribution`` their product with the configured weight.
        """
        weight_of = config.weight_of
        contributions: list[SignalContribution] = []
        for family in sorted(SignalFamily, key=lambda item: item.value):
            family_signals = tuple(
                signal for signal in signals if signal.family is family
            )
            scoring_signals = tuple(
                signal
                for signal in family_signals
                if signal.role is not SignalRole.CONTEXT_ONLY
            )
            present = tuple(
                signal
                for signal in scoring_signals
                if signal.status is SignalStatus.PRESENT
            )
            total_count = len(scoring_signals)
            present_count = len(present)
            with decimal.localcontext(FUSION_DECIMAL_CONTEXT):
                if present:
                    value = sum(
                        (
                            decimal.Decimal(str(signal.normalized_value))
                            for signal in present
                        ),
                        decimal.Decimal("0"),
                    ) / decimal.Decimal(len(present))
                else:
                    value = decimal.Decimal("0")
                availability = decimal.Decimal(present_count) / decimal.Decimal(
                    total_count
                )
                weight = weight_of[family]
                contribution = weight * value * availability
            signal_ids = tuple(sorted(signal.signal_id for signal in present))
            contributions.append(
                SignalContribution(
                    family=family,
                    weight=weight,
                    normalized_value=value,
                    availability=availability,
                    contribution=contribution,
                    present_count=present_count,
                    total_count=total_count,
                    signal_ids=signal_ids,
                    explanation=(
                        "Weighted family aggregate: configured weight %s times "
                        "the mean %s of its PRESENT scoring signals times "
                        "availability %s (%d of %d scoring signals present)."
                        % (
                            "{:.6f}".format(weight),
                            "{:.6f}".format(value),
                            "{:.6f}".format(availability),
                            present_count,
                            total_count,
                        )
                    ),
                )
            )
        return tuple(contributions)

    @staticmethod
    def _confidence_context(context: FusionContext) -> ConfidenceContext:
        """Edge confidence of the subject's incident edges, as context only.

        The bounds never enter the fused-score arithmetic; they are reported
        for the reader together with the standing note that they are
        explainability context.
        """
        profile = context.profile
        if context.graph is None:
            return ConfidenceContext()
        edges = incident_edges(context.graph, profile.observation_ids)
        if not edges:
            return ConfidenceContext()
        values = [float(edge.confidence.value) for edge in edges]
        bases = tuple(
            sorted(
                {edge.confidence.basis for edge in edges},
                key=lambda basis: basis.value,
            )
        )
        return ConfidenceContext(
            min_edge_confidence=decimal.Decimal(str(min(values))),
            max_edge_confidence=decimal.Decimal(str(max(values))),
            considered_edge_count=len(edges),
            bases=bases,
        )