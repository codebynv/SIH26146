"""BitTrace AI behavioral intelligence (SIH26146 - Step 5).

What this layer is for
----------------------
Step 5 reads the observations and the recorded-evidence graph and *describes*
what is there: how many records there are, how much they moved, how many
distinct addresses and IP addresses appear, over what span of time, how many
relationships the correlation layer recorded, and which sources they came from.

What this layer is not for
--------------------------
It does not interpret.  There is no anomaly score, no risk score, no
maliciousness measure, no clustering, no classification, no prediction, no
path finding, no ranking of observations, no graph centrality, and no claim
about who or what any record belongs to.  Those belong to Step 6 and later.

That boundary is not a matter of wording.  There is no field anywhere in this
package that could hold a verdict, so a verdict cannot be recorded here even by
accident, and the tests assert the absence of the vocabulary as well as the
presence of the numbers.

The pipeline
------------
::

    UnifiedObservation[]  +  EvidenceGraph
                |
                v
    BehaviorFeatureExtractor      -- descriptive, typed measurements
                |
                v
    BehavioralFeature[]
                |
                v
    BehaviorProfileBuilder         -- aggregates over an already-defined group
                |
                v
    BehavioralProfile[]

Grouping is never invented here.  A profile covers a collection the caller
supplied, a single record, or one connected component of the recorded-evidence
graph -- three groupings that all exist before this layer runs.  Connectivity
means only that observations are joined by recorded evidence relationships, and
it is not evidence that they share a person, an organisation, a wallet owner or
any other party.
"""

from __future__ import annotations

from .features import (
    BEHAVIORAL_CATEGORIES,
    BehaviorFeatureError,
    BehaviorFeatureExtractor,
)
from .models import (
    AMOUNT_STATISTICS_CONTEXT,
    BEHAVIOR_FEATURE_NAMESPACE,
    BEHAVIOR_PROFILE_NAMESPACE,
    AmountKind,
    BehavioralFeature,
    CountValue,
    DecimalValue,
    FeatureCategory,
    FeatureOrigin,
    FeatureScope,
    FeatureValue,
    LabelSetValue,
    LabelValue,
    MeasurementStatus,
    TimestampKind,
    TimestampState,
    TimestampValue,
    derive_feature_id,
    divide_exactly,
)
from .profiles import (
    BehaviorProfileError,
    BehaviorProfileBuilder,
    BehavioralProfile,
    ProfileGrouping,
    derive_profile_id,
    graph_structural_groups,
)

__all__ = [
    "AMOUNT_STATISTICS_CONTEXT",
    "AmountKind",
    "BEHAVIORAL_CATEGORIES",
    "BEHAVIOR_FEATURE_NAMESPACE",
    "BEHAVIOR_PROFILE_NAMESPACE",
    "BehaviorFeatureError",
    "BehaviorFeatureExtractor",
    "BehaviorProfileBuilder",
    "BehaviorProfileError",
    "BehavioralFeature",
    "BehavioralProfile",
    "CountValue",
    "DecimalValue",
    "FeatureCategory",
    "FeatureOrigin",
    "FeatureScope",
    "FeatureValue",
    "LabelSetValue",
    "LabelValue",
    "MeasurementStatus",
    "ProfileGrouping",
    "TimestampKind",
    "TimestampState",
    "TimestampValue",
    "derive_feature_id",
    "derive_profile_id",
    "divide_exactly",
    "graph_structural_groups",
]
