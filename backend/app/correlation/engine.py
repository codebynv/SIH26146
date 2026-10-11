"""The deterministic correlation engine (SIH26146 - Step 4B).

What this module is
-------------------
The single entry point that turns a batch of observations into a list of
:class:`~backend.app.correlation.models.EvidenceRelationship`.  It owns three
things the individual rules deliberately do not: the shape of a group, the
guarantees that hold across all rules at once, and the final ordering.

Grouping
--------
For a shared-key rule, a group of *n* observations sharing a key produces
**one** relationship whose source is the lowest ``observation_id`` and whose
targets are the remaining *n-1* observations.  The source and target sets are
disjoint by construction, so a relationship can never reference an observation
against itself, and a group of fifty records costs one relationship instead of
1 225.  The temporal rule is pairwise, because a distance is not a shared key
and there is nothing to group on.

Merging
-------
When one rule matches the same anchor and the same target set through more than
one key -- the same source IP *and* the same destination IP, say -- that is one
relationship, not two.  The matching facts are merged into a single rationale
so the extra evidence is recorded rather than lost.  Two relationships are
therefore never emitted for the same ``(rule_id, source, targets)``.

Determinism
-----------
Every output order, identifier and piece of text is derived from sorted
semantic content, never from the order the caller happened to supply:

* observations are sorted by ``observation_id`` before any rule runs, so
  shuffling the input cannot change a single result;
* groups are built from dictionaries but emitted in sorted key order, so
  insertion order cannot leak into the result;
* provenance references are de-duplicated and sorted;
* facts are sorted before a rationale is assembled;
* the final list is sorted by ``(rule_id, source, targets)``.

``created_at`` is recorded from the injected clock but is **excluded** from the
derived relationship identity, so re-running the same batch tomorrow yields
the same relationship ids.  Two runs of one input are therefore equivalent in
every field except the recorded wall-clock time.

What it is not
--------------
This is not a graph, and it stores nothing.  There is no node or edge type here,
no database, no traversal, no path finding, no scoring, no clustering and no
learned model.  Each relationship is an independent statement about a
deterministic field match, and none of them asserts identity, ownership or
attribution.  The observations handed in are read and never written to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional, Sequence

from backend.app.correlation.enums import (
    EvidenceType,
    InferenceLevel,
    TemporalRelation,
)
from backend.app.correlation.models import (
    EvidenceRelationship,
    TemporalWindow,
    derive_relationship_id,
)
from backend.app.correlation.provenance import ProvenanceRef
from backend.app.correlation.rules import (
    CorrelationConfig,
    CorrelationRule,
    MatchGroup,
    SupportsObservation,
    build_default_rules,
    _is_aware,
)

__all__ = [
    "CorrelationEngine",
]


@dataclass
class _Accumulator:
    """Everything merged into one relationship, before it is built."""

    rule: CorrelationRule
    source_observation_ids: tuple[str, ...]
    target_observation_ids: tuple[str, ...]
    facts: set[str] = field(default_factory=set)
    members: set[str] = field(default_factory=set)


class CorrelationEngine:
    """Applies the deterministic rules to a batch of observations.

    Construct once and reuse; an instance holds no per-run state, so the same
    engine may be used for many batches without interfering with itself.
    """

    def __init__(
        self,
        config: Optional[CorrelationConfig] = None,
        rules: Optional[Sequence[CorrelationRule]] = None,
    ) -> None:
        self._config = config if config is not None else CorrelationConfig()
        if rules is None:
            resolved: tuple[CorrelationRule, ...] = build_default_rules(self._config)
        else:
            resolved = tuple(rules)
            if not resolved:
                raise ValueError("a correlation engine needs at least one rule")
            for rule in resolved:
                if not rule.rule_id:
                    raise ValueError(
                        "every rule must carry a rule_id; a relationship that "
                        "cannot name its rule cannot be reproduced"
                    )
        self._rules = resolved

    @property
    def config(self) -> CorrelationConfig:
        return self._config

    @property
    def rules(self) -> tuple[CorrelationRule, ...]:
        return self._rules

    # -- input handling ---------------------------------------------------- #

    @staticmethod
    def _index_observations(
        observations: Iterable[SupportsObservation],
    ) -> dict[str, SupportsObservation]:
        """Index by ``observation_id``, keeping the first of any duplicate.

        Duplicate identifiers are a caller-side problem -- Step 3 derives the
        id from content, so two records with the same id are the same record.
        Indexing rather than failing keeps one malformed row from discarding a
        whole batch, and keeps a repeated observation from correlating with
        itself.
        """
        index: dict[str, SupportsObservation] = {}
        for position, observation in enumerate(observations):
            if observation is None:
                raise ValueError("observation at position %d is None" % position)
            observation_id = getattr(observation, "observation_id", None)
            if not isinstance(observation_id, str) or not observation_id.strip():
                raise ValueError(
                    "observation at position %d has no usable observation_id"
                    % position
                )
            index.setdefault(observation_id, observation)
        return index

    # -- provenance -------------------------------------------------------- #

    @staticmethod
    def _provenance_for(
        observation: SupportsObservation,
    ) -> Optional[ProvenanceRef]:
        """Build a citation from *observation*, or ``None`` if it has none.

        A blank locator is treated as absent rather than passed through,
        because :class:`ProvenanceRef` refuses a blank field and a citation
        pointing nowhere is worse than no citation.
        """
        provenance = getattr(observation, "provenance", None)
        if provenance is None:
            return None
        ingestion_id = getattr(provenance, "ingestion_id", None)
        if not isinstance(ingestion_id, str) or not ingestion_id.strip():
            return None

        def optional(name: str) -> Optional[str]:
            value = getattr(provenance, name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None

        return ProvenanceRef(
            ingestion_id=ingestion_id.strip(),
            source_file=optional("source_file"),
            source_record=optional("source_record"),
        )

    @staticmethod
    def _collect_provenance(
        members: Iterable[SupportsObservation],
    ) -> tuple[ProvenanceRef, ...]:
        """Deduplicate and sort every citation the members can offer."""
        seen: dict[tuple[str, str, str], ProvenanceRef] = {}
        for observation in members:
            reference = CorrelationEngine._provenance_for(observation)
            if reference is None:
                continue
            key = (
                reference.ingestion_id,
                reference.source_file or "",
                reference.source_record or "",
            )
            seen.setdefault(key, reference)
        return tuple(seen[key] for key in sorted(seen))

    # -- temporal detail --------------------------------------------------- #

    def _temporal_detail(
        self,
        rule: CorrelationRule,
        source_id: str,
        target_id: str,
        index: dict[str, SupportsObservation],
    ) -> tuple[Optional[TemporalWindow], Optional[TemporalRelation]]:
        """Build the window and relation for a temporal relationship.

        The window spans the two observations' own timestamps, so it states
        exactly how far apart they were rather than restating the configured
        bound.  The relation is read from the *source* side's point in time
        relative to the target's: each observation is its own instantaneous
        window, so the source precedes, follows, or coincides with the target.
        Coincident timestamps are ``WITHIN`` because each point window contains
        the other; ``OVERLAPS`` is reserved for a genuine partial overlap,
        which two instants cannot produce.
        """
        if rule.evidence_type is not EvidenceType.TEMPORAL_ADJACENCY:
            return None, None
        source = getattr(index.get(source_id), "timestamp", None)
        target = getattr(index.get(target_id), "timestamp", None)
        if not _is_aware(source) or not _is_aware(target):
            return None, None
        window = TemporalWindow(start=min(source, target), end=max(source, target))
        if source < target:
            relation = TemporalRelation.BEFORE
        elif source > target:
            relation = TemporalRelation.AFTER
        else:
            relation = TemporalRelation.WITHIN
        return window, relation

    # -- the pass ---------------------------------------------------------- #

    def run(
        self, observations: Iterable[SupportsObservation]
    ) -> list[EvidenceRelationship]:
        """Correlate *observations* and return the relationships found.

        The input is read only.  A relationship is produced for every match
        every rule reports, in a deterministic order, with no duplicates and no
        self-references.
        """
        index = self._index_observations(observations)
        ordered = tuple(index[observation_id] for observation_id in sorted(index))

        merged: dict[tuple[str, tuple[str, ...], tuple[str, ...]], _Accumulator] = {}
        for rule in self._rules:
            for group in rule.evaluate(ordered):
                accumulator = self._accumulate(group, rule, merged)
                if accumulator is None:
                    continue
                accumulator.facts.update(group.facts)
                accumulator.members.update(group.observation_ids)

        relationships: list[EvidenceRelationship] = []
        for key in sorted(merged):
            relationships.append(self._build(merged[key], index))
        return relationships

    @staticmethod
    def _accumulate(
        group: MatchGroup,
        rule: CorrelationRule,
        merged: dict,
    ) -> Optional[_Accumulator]:
        """Fold *group* into the merge table, returning the entry it landed in.

        The lowest identifier in the group anchors the source side and the rest
        become targets, which keeps the two sides disjoint and makes the merge
        key independent of the order the group was discovered in.
        """
        members = sorted(dict.fromkeys(group.observation_ids))
        if len(members) < 2:
            return None
        anchor = members[0]
        targets = tuple(members[1:])
        key = (rule.rule_id, (anchor,), targets)
        accumulator = merged.get(key)
        if accumulator is None:
            accumulator = _Accumulator(
                rule=rule,
                source_observation_ids=(anchor,),
                target_observation_ids=targets,
            )
            merged[key] = accumulator
        return accumulator

    def _build(
        self,
        accumulator: _Accumulator,
        index: dict[str, SupportsObservation],
    ) -> EvidenceRelationship:
        """Turn one merged entry into an :class:`EvidenceRelationship`."""
        rule = accumulator.rule
        source_ids = accumulator.source_observation_ids
        target_ids = accumulator.target_observation_ids
        rationale = rule.describe(sorted(accumulator.facts))

        window, relation = self._temporal_detail(
            rule, source_ids[0], target_ids[0], index
        )

        provenance_refs = self._collect_provenance(
            index[observation_id]
            for observation_id in sorted(accumulator.members)
        )

        # Identity covers what the relationship *asserts* and never the moment
        # it was recorded, so the same batch always yields the same ids.
        relationship_id = derive_relationship_id(
            {
                "evidence_type": rule.evidence_type.value,
                "inference_level": InferenceLevel.CORRELATED.value,
                "rule_id": rule.rule_id,
                "rule_version": rule.rule_version,
                "source_observation_ids": list(source_ids),
                "target_observation_ids": list(target_ids),
                "confidence": {
                    "value": rule.confidence.value,
                    "basis": rule.confidence.basis.value,
                },
                "rationale": rationale,
            }
        )

        return EvidenceRelationship(
            relationship_id=relationship_id,
            source_observation_ids=source_ids,
            target_observation_ids=target_ids,
            evidence_type=rule.evidence_type,
            inference_level=InferenceLevel.CORRELATED,
            confidence=rule.confidence,
            rationale=rationale,
            rule_id=rule.rule_id,
            rule_version=rule.rule_version,
            provenance_refs=provenance_refs,
            created_at=self._config.now,
            temporal_window=window,
            temporal_relation=relation,
        )
