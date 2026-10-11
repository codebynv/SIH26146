"""Deterministic behavioral profiles (SIH26146 - Step 5).

A :class:`BehavioralProfile` is a descriptive aggregate over a stated set of
observations: the observations, their features, the recorded relationships that
name them, and the ingestion runs they came from.  That is all it is.  It does
not name who the observations belong to, whether the behaviour is unusual, or
what should be done about it.

Grouping safety
---------------
A profile is built for one of exactly two groupings, both of which already
exist before this layer runs:

``SINGLE_OBSERVATION``
    one record.

``SUPPLIED_COLLECTION``
    a collection the caller handed over, for its own reasons, on its own
    terms.  This layer measures it.  It does not decide who belongs together
    and it has no procedure for finding out: no entity resolution, no
    identity inference, no matching heuristic, no name, no account, no
    address-owner lookup.

``GRAPH_STRUCTURAL_GROUP``
    a component of the recorded-evidence graph, obtained by
    :func:`graph_structural_groups`.  Connectivity there means exactly one
    thing: *these observations are joined by recorded evidence
    relationships*.  Two records on the same source IP, two records naming the
    same address, two records cited in the same file -- that is the whole of
    what a component asserts, and it is asserted nowhere in this module in
    stronger words.  A component is not a person, an organisation, a wallet
    owner or an actor, and this layer cannot tell you it is.

Determinism
-----------
The profile identifier is a UUIDv5 over the sorted set of observation
identifiers and the grouping basis, so it depends on *which observations* a
profile covers and *why they were grouped*, never on the order they arrived
in, the process, or the clock.  Shuffled input yields an equal profile;
building twice yields an equal profile.
"""

from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

from ..domain.models import UnifiedObservation
from ..graph.models import EvidenceGraph
from .features import BehaviorFeatureExtractor
from .models import (
    BEHAVIOR_PROFILE_NAMESPACE,
    BehavioralFeature,
    FeatureCategory,
    FeatureScope,
)

__all__ = [
    "BehaviorProfileBuilder",
    "BehaviorProfileError",
    "BehavioralProfile",
    "ProfileGrouping",
    "derive_profile_id",
    "graph_structural_groups",
]

_PROFILE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_GROUP_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/-]*$")

#: Words refused in a caller-supplied ``group_id``.  A label is the one place
#: on a profile where free text lands, so it is the one place a verdict could
#: sneak in.  Matched on whole words with explicit boundaries, so "brisk" is
#: not mistaken for "risky" while "risk-scores" still is caught.  The check
#: applies to the label only: source data is never rewritten, because
#: preserving evidence as the source stated it is not this layer's to revise.
_EVALUATIVE_LABEL_WORDS = (
    "anomalies",
    "anomalous",
    "anomaly",
    "attacker",
    "benign",
    "criminal",
    "criminals",
    "fraud",
    "fraudulent",
    "identity",
    "illicit",
    "laundering",
    "malicious",
    "maliciousness",
    "owner",
    "ownership",
    "person",
    "risk",
    "risky",
    "scam",
    "suspect",
    "suspicious",
    "threat",
)
_EVALUATIVE_LABEL_RE = re.compile(
    r"(?<![a-z])(%s)(?![a-z])" % "|".join(_EVALUATIVE_LABEL_WORDS)
)


class ProfileGrouping(str, Enum):
    """Why a set of observations was put in one profile.

    Recorded on the profile so a reader is never left guessing.  All three
    describe a grouping that already existed; none of them is this layer's
    opinion about who belongs together.
    """

    SINGLE_OBSERVATION = "SINGLE_OBSERVATION"
    SUPPLIED_COLLECTION = "SUPPLIED_COLLECTION"
    GRAPH_STRUCTURAL_GROUP = "GRAPH_STRUCTURAL_GROUP"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class BehaviorProfileError(ValueError):
    """Raised when a profile cannot be built as requested.

    A :class:`ValueError`, for the same reason the feature extractor uses one:
    the caller handed over something that is not what was asked for, and
    quietly building a profile from half-understood input would produce a
    description that looks trustworthy and is not.
    """


def derive_profile_id(
    grouping: ProfileGrouping, observation_ids: Sequence[str]
) -> str:
    """Derive a profile identifier from the observations and the grouping basis.

    The payload is the sorted, deduplicated set of observation identifiers plus
    the grouping that produced it.  Sorting is what makes the result
    order-independent, and including the grouping means the same observations
    grouped for two different stated reasons are two different profiles rather
    than one ambiguously-shared identifier.

    Deliberately derived from the *membership* and not from the feature
    values.  A profile is a description of a set, and two descriptions of the
    same set are the same profile; the description itself is right there in
    ``features``, where it can change without the identity of what is being
    described moving underneath it.
    """
    payload = json.dumps(
        {
            "grouping": ProfileGrouping(grouping).value,
            "observation_ids": sorted(set(observation_ids)),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(BEHAVIOR_PROFILE_NAMESPACE, payload))


def graph_structural_groups(
    graph: EvidenceGraph,
) -> tuple[tuple[str, ...], ...]:
    """Partition the graph's observations into connected components.

    Structural and nothing else.  Two observations land in the same component
    when a recorded evidence relationship names both, directly or through
    others; that is the entire claim, and it is a statement about recorded
    evidence, not about anything in the world.  Two records joined because they
    shared a source IP are in one component because the correlation layer
    recorded a relationship between them -- not because anything was decided
    about who they are.

    Every observation that appears in the graph gets a component, including
    isolated ones, which appear alone.  The traversal ignores edge direction
    and runs once over the nodes and once over the edges, so the cost is linear
    in the size of the graph.

    The result is sorted by member tuple, so the same graph always yields the
    same groups in the same order.
    """
    if not isinstance(graph, EvidenceGraph):
        raise BehaviorProfileError(
            "graph must be an EvidenceGraph, not %r" % type(graph).__name__
        )
    parent: dict[str, str] = {
        node.observation_id: node.observation_id for node in graph.nodes()
    }

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        # Attach the lexicographically larger root to the smaller one, so the
        # resulting structure does not depend on the order edges are walked.
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for edge in graph.edges():
        members = edge.observation_ids()
        anchor = members[0]
        for other in members[1:]:
            union(anchor, other)

    components: dict[str, set[str]] = {}
    for observation_id in sorted(parent):
        components.setdefault(find(observation_id), set()).add(observation_id)
    return tuple(
        sorted((tuple(sorted(group)) for group in components.values()), key=lambda g: g)
    )


class BehavioralProfile:
    """An immutable, deterministic descriptive aggregate over observations.

    Frozen at every level: the collections are tuples and the features are
    frozen models, so a consumer cannot reorder, extend or edit a profile in
    place.  Equality is structural, which is what lets a test assert that two
    builds of the same input are the same profile.

    What is deliberately missing is the point.  There is no field for a name, a
    label, a score, a severity, a verdict or a party: this object describes a
    set of records, and the question "is this behaviour concerning?" has
    nowhere to be recorded even if someone wanted to answer it here.
    """

    __slots__ = (
        "_profile_id",
        "_grouping",
        "_group_id",
        "_observation_ids",
        "_features",
        "_evidence_relationship_ids",
        "_provenance_ingestion_ids",
        "_provenance_source_files",
    )

    def __init__(
        self,
        profile_id: str,
        grouping: ProfileGrouping,
        group_id: Optional[str],
        observation_ids: tuple[str, ...],
        features: tuple[BehavioralFeature, ...],
        evidence_relationship_ids: tuple[str, ...],
        provenance_ingestion_ids: tuple[str, ...],
        provenance_source_files: tuple[str, ...],
    ) -> None:
        if not _PROFILE_ID_RE.match(profile_id or ""):
            raise BehaviorProfileError("profile_id must be a UUID, got %r" % (profile_id,))
        object.__setattr__(self, "_profile_id", profile_id)
        object.__setattr__(self, "_grouping", ProfileGrouping(grouping))
        object.__setattr__(self, "_group_id", self._check_label(group_id))
        object.__setattr__(self, "_observation_ids", self._check_ids(observation_ids))
        object.__setattr__(self, "_features", self._check_features(features))
        object.__setattr__(
            self, "_evidence_relationship_ids", self._check_ids(evidence_relationship_ids)
        )
        object.__setattr__(
            self,
            "_provenance_ingestion_ids",
            self._check_ids(provenance_ingestion_ids),
        )
        object.__setattr__(
            self, "_provenance_source_files", self._check_ids(provenance_source_files)
        )
        expected = derive_profile_id(self._grouping, self._observation_ids)
        if self._profile_id != expected:
            raise BehaviorProfileError(
                "profile_id %s does not match the identifier derived from "
                "grouping %s and %d observation(s) (expected %s)"
                % (profile_id, self._grouping.value, len(self._observation_ids), expected)
            )
        self._check_consistency()

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every attribute assignment after construction.

        A profile is a statement about a set of records.  Letting a consumer
        rewrite a feature or an identifier in place would turn that statement
        into something else while leaving the identifier standing, so the
        object refuses to change at all.
        """
        raise AttributeError(
            "BehavioralProfile is immutable; %r cannot be assigned" % name
        )

    # -- validation ------------------------------------------------------- #

    @staticmethod
    def _check(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise BehaviorProfileError("%s must be a non-empty string" % field)
        return value

    @staticmethod
    def _check_label(value: Optional[str]) -> Optional[str]:
        """Validate a caller-supplied group label, if any.

        A label is a convenience for the caller's own bookkeeping, so free
        text is allowed.  It is refused if it reads like a verdict, because a
        label is the one place on this object where free text lands, and
        "the suspicious wallet group" would carry a judgement into a report
        where only a description belongs.
        """
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise BehaviorProfileError("group_id must be null or a non-blank string")
        if not _GROUP_LABEL_RE.match(value.strip()):
            raise BehaviorProfileError(
                "group_id %r contains characters that are not allowed in a label"
                % value
            )
        match = _EVALUATIVE_LABEL_RE.search(value.lower())
        if match is not None:
            raise BehaviorProfileError(
                "group_id %r is refused: this layer produces descriptive "
                "aggregates, and a label that judges the behaviour (%r) would "
                "put a verdict where only a description belongs"
                % (value, match.group(1))
            )
        return value.strip()

    @staticmethod
    def _check_ids(value: tuple[str, ...]) -> tuple[str, ...]:
        candidate = tuple(value)
        for item in candidate:
            if not isinstance(item, str) or not item.strip():
                raise BehaviorProfileError(
                    "identifiers must be non-empty strings, got %r" % (item,)
                )
        if tuple(sorted(set(candidate))) != candidate:
            raise BehaviorProfileError(
                "identifier tuples must be sorted and free of duplicates"
            )
        return candidate

    def _check_features(
        self, features: tuple[BehavioralFeature, ...]
    ) -> tuple[BehavioralFeature, ...]:
        candidate = tuple(features)
        for feature in candidate:
            if not isinstance(feature, BehavioralFeature):
                raise BehaviorProfileError(
                    "features must be BehavioralFeature, got %r"
                    % type(feature).__name__
                )
        keys = tuple(
            (feature.category.value, feature.name) for feature in candidate
        )
        if tuple(sorted(keys)) != keys:
            raise BehaviorProfileError(
                "features must be sorted by (category, name) and unique"
            )
        return candidate

    def _check_consistency(self) -> None:
        """Refuse a profile whose parts disagree with each other.

        Two invariants, both about the profile not claiming something its own
        contents do not support:

        * every feature is profile-scoped and points back at *this* profile;
        * the recorded relationships the profile lists are exactly the ones its
          features cite, in both directions.

        Note what is deliberately *not* required: that a feature's origin names
        only observations inside the profile.  A feature drawn from a recorded
        relationship legitimately cites the record on the far side of that
        relationship, which is precisely what "incident" means.  What must
        hold is that reaching it went through a listed relationship, and that
        is what the relationship check enforces.
        """
        for feature in self._features:
            if feature.scope is not FeatureScope.PROFILE:
                raise BehaviorProfileError(
                    "a profile carries profile-scoped features only; %r is %s"
                    % (feature.name, feature.scope.value)
                )
            if feature.profile_id != self._profile_id:
                raise BehaviorProfileError(
                    "feature %r references profile %s, not this profile %s"
                    % (feature.name, feature.profile_id, self._profile_id)
                )
        cited = {
            relationship_id
            for feature in self._features
            for relationship_id in feature.origin.relationship_ids
        }
        listed = set(self._evidence_relationship_ids)
        if not cited.issubset(listed):
            raise BehaviorProfileError(
                "a feature cites a recorded relationship the profile does not "
                "list: %s" % sorted(cited - listed)[0]
            )
        if not listed.issubset(cited):
            raise BehaviorProfileError(
                "the profile lists a recorded relationship no feature cites: %s"
                % sorted(listed - cited)[0]
            )

    # -- public surface --------------------------------------------------- #

    @property
    def profile_id(self) -> str:
        """Deterministic identifier derived from membership and grouping."""
        return self._profile_id

    @property
    def grouping(self) -> ProfileGrouping:
        """Why these observations were grouped, as stated by the caller."""
        return self._grouping

    @property
    def group_id(self) -> Optional[str]:
        """The caller's own label for the group, if it supplied one."""
        return self._group_id

    @property
    def observation_ids(self) -> tuple[str, ...]:
        """The observations this profile describes, sorted."""
        return self._observation_ids

    @property
    def features(self) -> tuple[BehavioralFeature, ...]:
        """The descriptive features, ordered by category then name."""
        return self._features

    @property
    def evidence_relationship_ids(self) -> tuple[str, ...]:
        """Recorded relationships naming these observations, sorted."""
        return self._evidence_relationship_ids

    @property
    def provenance_ingestion_ids(self) -> tuple[str, ...]:
        """Ingestion runs the described observations came from, sorted."""
        return self._provenance_ingestion_ids

    @property
    def provenance_source_files(self) -> tuple[str, ...]:
        """Source files the described observations came from, sorted."""
        return self._provenance_source_files

    @property
    def observation_count(self) -> int:
        return len(self._observation_ids)

    @property
    def feature_count(self) -> int:
        return len(self._features)

    def categories(self) -> tuple[FeatureCategory, ...]:
        """The categories actually carrying at least one feature."""
        present = {feature.category for feature in self._features}
        return tuple(
            category for category in FeatureCategory if category in present
        )

    def feature(self, name: str) -> BehavioralFeature:
        """Return the named feature.

        :raises KeyError: if this profile has no such feature, which is always
            a caller mistake: the extractor emits the same names every time.
        """
        for candidate in self._features:
            if candidate.name == name:
                return candidate
        raise KeyError(
            "no feature named %r; this profile measures %s"
            % (name, ", ".join(sorted(f.name for f in self._features)))
        )

    def value_of(self, name: str) -> Any:
        """The raw value of the named feature."""
        return self.feature(name).value.value

    def describes_attribution(self) -> bool:
        """Always ``False``.

        A profile states which observations it describes and what was measured
        about them.  It does not state who they belong to, and it has no field
        that could hold such a statement -- the presence of this method is what
        makes that absence checkable rather than merely asserted in prose.
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping with a fixed key order."""
        return {
            "profile_id": self._profile_id,
            "grouping": self._grouping.value,
            "group_id": self._group_id,
            "observation_count": self.observation_count,
            "observation_ids": list(self._observation_ids),
            "feature_count": self.feature_count,
            "features": [feature.to_dict() for feature in self._features],
            "evidence_relationship_ids": list(self._evidence_relationship_ids),
            "provenance_ingestion_ids": list(self._provenance_ingestion_ids),
            "provenance_source_files": list(self._provenance_source_files),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BehavioralProfile):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(self._profile_id)

    def __repr__(self) -> str:
        return "BehavioralProfile(profile_id=%s, grouping=%s, observations=%d, features=%d)" % (
            self._profile_id,
            self._grouping.value,
            self.observation_count,
            self.feature_count,
        )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.__repr__()


class BehaviorProfileBuilder:
    """Builds deterministic :class:`BehavioralProfile` aggregates.

    The builder owns no state beyond the graph it was given.  It does not
    decide membership: a caller names a collection, or asks for the graph's
    structural components, and the builder measures whichever it was handed.

    Duplicate observation identifiers are refused rather than merged.  Two
    records with the same identifier would be two descriptions of one thing,
    and picking either silently would discard a record without saying so.
    """

    def __init__(self, graph: Optional[EvidenceGraph] = None) -> None:
        if graph is not None and not isinstance(graph, EvidenceGraph):
            raise BehaviorProfileError(
                "graph must be an EvidenceGraph or None, not %r"
                % type(graph).__name__
            )
        self._graph = graph
        self._extractor = BehaviorFeatureExtractor(graph)

    @property
    def graph(self) -> Optional[EvidenceGraph]:
        """The evidence graph the builder measures against, if any."""
        return self._graph

    def build_empty(self, group_id: Optional[str] = None) -> BehavioralProfile:
        """A profile over no observations.

        Legitimate and meaningful: a caller who asked about a set and found it
        empty gets a real, fully-formed profile whose features are all
        ``ABSENT`` or zero, rather than ``None`` standing in for an answer.
        """
        return self._build(
            (),
            grouping=ProfileGrouping.SUPPLIED_COLLECTION,
            group_id=group_id,
        )

    def build_for_observation(
        self,
        observation: UnifiedObservation,
        group_id: Optional[str] = None,
    ) -> BehavioralProfile:
        """A profile covering exactly one record."""
        return self._build(
            (observation,),
            grouping=ProfileGrouping.SINGLE_OBSERVATION,
            group_id=group_id,
        )

    def build(
        self,
        observations: Iterable[UnifiedObservation],
        group_id: Optional[str] = None,
    ) -> BehavioralProfile:
        """A profile over a collection the caller supplied."""
        return self._build(
            observations,
            grouping=ProfileGrouping.SUPPLIED_COLLECTION,
            group_id=group_id,
        )

    def build_for_graph_structural_group(
        self,
        observation_ids: Iterable[str],
        observations: Iterable[UnifiedObservation],
        group_id: Optional[str] = None,
    ) -> BehavioralProfile:
        """A profile over a member list of a structural component.

        The members are taken as given; whether that list really is a
        component is the caller's business, and re-deriving it here would mean
        this layer getting a say in what belongs together.  The records behind
        the members must be supplied, because a profile that listed five
        observations while describing three would be quietly wrong, and a
        missing record is refused rather than absorbed.
        """
        lookup = self._index(observations)
        members = tuple(observation_ids)
        missing = sorted(set(members) - set(lookup))
        if missing:
            raise BehaviorProfileError(
                "no record was supplied for observation_id %s, so this profile "
                "could not describe it" % missing[0]
            )
        return self._build_ids(
            members,
            grouping=ProfileGrouping.GRAPH_STRUCTURAL_GROUP,
            group_id=group_id,
            lookup=lookup,
        )

    def build_for_graph_structural_groups(
        self, observations: Iterable[UnifiedObservation]
    ) -> tuple[BehavioralProfile, ...]:
        """One profile per connected component of the recorded-evidence graph.

        The grouping is structural by construction: two observations share a
        profile when a recorded evidence relationship names both, directly or
        through a chain of others.  That is a statement about recorded
        evidence and about nothing else.  A component does not indicate a
        person, an organisation, a wallet owner or an actor, and the label
        attached to each profile says only "structural group".

        Every observation the graph names must have a supplied record, and no
        record may appear twice.  A missing record is an error rather than a
        smaller profile, so a component can never be described by fewer
        observations than it contains.

        An observation the graph does not name is in no component and
        therefore gets no profile from this method: an unconnected record is
        not a group, and putting it in one would be inventing a grouping.
        Use :meth:`build` or :meth:`build_for_observation` for those.
        """
        if self._graph is None:
            raise BehaviorProfileError(
                "structural groups need a recorded-evidence graph; none was supplied"
            )
        lookup = self._index(observations)
        missing = sorted(
            {node.observation_id for node in self._graph.nodes()} - set(lookup)
        )
        if missing:
            raise BehaviorProfileError(
                "the graph names observation_id %s but no record was supplied "
                "for it; supply every observation in the graph so the component "
                "is described in full" % missing[0]
            )
        return tuple(
            self._build_ids(
                members,
                grouping=ProfileGrouping.GRAPH_STRUCTURAL_GROUP,
                group_id="structural group %d" % (index + 1),
                lookup=lookup,
            )
            for index, members in enumerate(graph_structural_groups(self._graph))
        )

    # -- internals -------------------------------------------------------- #

    def _index(
        self, observations: Iterable[UnifiedObservation]
    ) -> dict[str, UnifiedObservation]:
        """Index supplied records by identifier, refusing duplicates.

        Two records sharing an identifier are two records the model has failed
        to keep apart; picking either one silently would discard a record
        without saying so, so it is refused.
        """
        lookup: dict[str, UnifiedObservation] = {}
        for observation in self._materialise(observations):
            if observation.observation_id in lookup:
                raise BehaviorProfileError(
                    "observation_id %s appears more than once in the input"
                    % observation.observation_id
                )
            lookup[observation.observation_id] = observation
        return lookup

    def _materialise(
        self, observations: Iterable[UnifiedObservation]
    ) -> tuple[UnifiedObservation, ...]:
        if isinstance(observations, (str, bytes)):
            raise BehaviorProfileError(
                "observations must be a sequence of UnifiedObservation, not text"
            )
        try:
            candidate = tuple(observations)
        except TypeError as exc:
            raise BehaviorProfileError(
                "observations must be an iterable of UnifiedObservation: %s" % exc
            ) from exc
        for position, item in enumerate(candidate):
            if not isinstance(item, UnifiedObservation):
                raise BehaviorProfileError(
                    "observation at position %d is %r, not a UnifiedObservation"
                    % (position, type(item).__name__)
                )
        return candidate

    def _build(
        self,
        observations: Iterable[UnifiedObservation],
        grouping: ProfileGrouping,
        group_id: Optional[str],
    ) -> BehavioralProfile:
        lookup = self._index(observations)
        return self._build_ids(
            tuple(lookup), grouping=grouping, group_id=group_id, lookup=lookup
        )

    def _build_ids(
        self,
        observation_ids: tuple[str, ...],
        grouping: ProfileGrouping,
        group_id: Optional[str],
        lookup: dict[str, UnifiedObservation],
    ) -> BehavioralProfile:
        """Assemble a profile from a member list and the records behind it.

        The member list is sorted and deduplicated before the identifier is
        derived, so input order cannot reach the identifier at all.
        """
        ordered = tuple(sorted(set(observation_ids)))
        present = [lookup[member] for member in ordered]
        profile_id = derive_profile_id(grouping, ordered)
        features = self._extractor.extract_observations(present, profile_id=profile_id)
        relationship_ids = tuple(
            sorted(
                {
                    relationship_id
                    for feature in features
                    for relationship_id in feature.origin.relationship_ids
                }
            )
        )
        ingestion_ids = tuple(
            sorted(
                {
                    ingestion_id
                    for feature in features
                    for ingestion_id in feature.origin.ingestion_ids
                }
            )
        )
        source_files = tuple(
            sorted(
                {
                    source_file
                    for feature in features
                    for source_file in feature.origin.source_files
                }
            )
        )
        return BehavioralProfile(
            profile_id=profile_id,
            grouping=grouping,
            group_id=group_id,
            observation_ids=ordered,
            features=features,
            evidence_relationship_ids=relationship_ids,
            provenance_ingestion_ids=ingestion_ids,
            provenance_source_files=source_files,
        )
