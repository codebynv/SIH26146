"""Step 5 tests: the behavioral feature models.

These cover the typed value objects, the empty-set policy they commit to, the
derived identifiers, serialisation, immutability, and the safeguards that keep
the vocabulary of this layer descriptive.

The last two classes are the ones that matter most for the boundary.  A model
that quietly grew an ``owner`` field, or a package that quietly grew an import
of a machine-learning library, would still pass every arithmetic test above.
These fail on that.
"""

from __future__ import annotations

import ast
import copy
import decimal
import json
import pathlib
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import pytest
from pydantic import ValidationError

from backend.app import behavior as behavior_package
from backend.app.behavior import features as behavior_features
from backend.app.behavior import models as behavior_models
from backend.app.behavior import profiles as behavior_profiles
from backend.app.behavior.models import (
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
    LabelSetValue,
    LabelValue,
    MeasurementStatus,
    TimestampKind,
    TimestampState,
    TimestampValue,
    derive_feature_id,
    divide_exactly,
)

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2024, 5, 2, 12, 0, tzinfo=UTC)

# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #

#: Every field name that would let this layer record something other than a
#: description.  Checked against the real model fields, not against prose.
BANNED_FIELD_WORDS = frozenset(
    {
        "actor",
        "anomalies",
        "anomalous",
        "anomaly",
        "attacker",
        "attributed",
        "attribution",
        "criminal",
        "fraud",
        "fraudulent",
        "identities",
        "identity",
        "illicit",
        "malicious",
        "maliciousness",
        "owner",
        "ownership",
        "person",
        "risk",
        "risky",
        "score",
        "scores",
        "scoring",
        "severity",
        "suspicious",
        "threat",
        "verdict",
    }
)

#: Modules that would mean a later stage had leaked backwards.  Checked across
#: every application module, not only this one.
FORBIDDEN_GLOBAL_MODULES = frozenset(
    {
        "airflow",
        "anthropic",
        "celery",
        "dbt",
        "duckdb",
        "graphblas",
        "igraph",
        "kafka",
        "langchain",
        "lightgbm",
        "llama",
        "matplotlib",
        "neo4j",
        "networkx",
        "numpy",
        "openai",
        "pandas",
        "plotly",
        "psycopg2",
        "pymongo",
        "pymysql",
        "redis",
        "scipy",
        "seaborn",
        "sklearn",
        "sqlalchemy",
        "statsmodels",
        "tensorflow",
        "torch",
        "transformers",
        "xgboost",
    }
)

#: Modules that would mean the behavioural layer had grown an interface or a
#: store.  Checked in this package and its tests only: the locked Step 1
#: health endpoint legitimately imports an ASGI framework, and that is not this
#: layer's business.
FORBIDDEN_HERE_MODULES = FORBIDDEN_GLOBAL_MODULES | frozenset(
    {
        "aiohttp",
        "django",
        "fastapi",
        "flask",
        "httpx",
        "jinja2",
        "requests",
        "socket",
        "sqlite3",
        "starlette",
        "uvicorn",
    }
)

#: Identifiers allowed despite containing a guarded word.  Each is a negative
#: assertion -- a method that answers "no" to the question -- and exists so the
#: absence of such a claim is callable rather than merely documented.  This
#: package has exactly one.
ALLOWED_NEGATIVE_ASSERTIONS = frozenset({"describes_attribution"})

_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+")
_SPLIT_RE = re.compile(r"[^0-9A-Za-z]+")


def identifier_words(name: str) -> tuple[str, ...]:
    """Split an identifier into lowercase words.

    Both boundaries matter.  ``snake_case`` is split on separators, and
    ``camelCase`` is split on case transitions, so ``unique_src_ip_count``
    yields three words and ``graphNodeCount`` yields three.  Without the second
    rule a camel-cased banned word would pass straight through.
    """
    words: list[str] = []
    for chunk in _SPLIT_RE.split(name):
        if not chunk:
            continue
        words.extend(match.group(0).lower() for match in _CAMEL_RE.finditer(chunk))
    return tuple(words)


def iter_identifiers(tree: ast.AST):
    """Yield every identifier an AST defines, names or reads.

    Deliberately broad -- definitions, parameters, names, attribute reads,
    import aliases, keyword names and globals -- because a banned concept
    introduced as a parameter name is just as much a problem as one introduced
    as a class.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.alias):
            yield node.asname or node.name
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            yield node.names


def imported_roots(tree: ast.AST) -> set[str]:
    """Top-level module names imported by an AST."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            roots.add(node.module.split(".")[0])
    return roots


PACKAGE_FILES = sorted(
    pathlib.Path(behavior_models.__file__).resolve().parent.glob("*.py")
)
STEP5_TEST_FILES = sorted(
    pathlib.Path(__file__).resolve().parent.glob("test_step5_behavior_*.py")
)
ALL_SOURCE_FILES = sorted(pathlib.Path("backend").resolve().rglob("*.py"))

assert PACKAGE_FILES, "the behavioural package must contain modules to scan"


def make_feature(
    name: str = "observation_count",
    *,
    scope: FeatureScope = FeatureScope.OBSERVATION,
    observation_id: Optional[str] = "obs-1",
    profile_id: Optional[str] = None,
    value: Any = None,
    category: FeatureCategory = FeatureCategory.ACTIVITY_VOLUME,
) -> BehavioralFeature:
    """Build a valid feature, deriving the identifier the model will insist on."""
    if value is None:
        value = CountValue(value=1)
    return BehavioralFeature(
        feature_id=derive_feature_id(name, scope, observation_id, profile_id),
        name=name,
        category=category,
        scope=scope,
        value=value,
        observation_id=observation_id,
        profile_id=profile_id,
    )


# --------------------------------------------------------------------------- #
# Valid features
# --------------------------------------------------------------------------- #


class TestValidFeature:
    def test_a_count_feature_is_accepted(self):
        feature = make_feature()
        assert feature.name == "observation_count"
        assert feature.scope is FeatureScope.OBSERVATION
        assert feature.observation_id == "obs-1"
        assert feature.profile_id is None
        assert feature.value.value == 1
        assert feature.status is MeasurementStatus.OBSERVED

    def test_every_value_type_can_carry_a_feature(self):
        for value in (
            CountValue(value=3),
            DecimalValue(value=Decimal("0.42"), unit="BTC", amount_kind=AmountKind.MEAN),
            TimestampValue(value=T0, timestamp_kind=TimestampKind.AWARE),
            LabelValue(value=TimestampState.MIXED.value),
            LabelSetValue(value=("10.0.0.1", "10.0.0.2")),
        ):
            feature = make_feature(value=value)
            assert feature.value.kind == value.kind
            assert feature.status is MeasurementStatus.OBSERVED

    def test_profile_scoped_feature_carries_the_profile(self):
        scope = FeatureScope.PROFILE
        feature = make_feature(
            "unique_src_ip_count",
            scope=scope,
            observation_id=None,
            profile_id="a" * 8 + "-0000-5000-8000-000000000000",
            category=FeatureCategory.NETWORK_BEHAVIOR,
            value=CountValue(value=4),
        )
        assert feature.observation_id is None
        assert feature.profile_id is not None

    def test_units_and_origins_are_carried(self):
        feature = make_feature(
            value=CountValue(value=2, unit="observations"),
        ).model_copy(
            update={
                "origin": FeatureOrigin(
                    observation_ids=("obs-1", "obs-2"),
                    ingestion_ids=("ing-1",),
                    source_files=("data.csv",),
                    source_records=("row 1", "row 2"),
                    relationship_ids=("rel-1",),
                )
            }
        )
        assert feature.origin.ingestion_ids == ("ing-1",)
        assert feature.origin.relationship_ids == ("rel-1",)
        assert feature.to_dict()["origin"]["source_records"] == ["row 1", "row 2"]

    def test_identifiers_are_namespaces_we_own(self):
        assert BEHAVIOR_FEATURE_NAMESPACE != BEHAVIOR_PROFILE_NAMESPACE
        assert derive_feature_id("observation_count", FeatureScope.OBSERVATION, "x") != (
            derive_feature_id(
                "observation_count", FeatureScope.PROFILE, None, "y"
            )
        )


# --------------------------------------------------------------------------- #
# Invalid features
# --------------------------------------------------------------------------- #


class TestInvalidFeature:
    @pytest.mark.parametrize(
        "name",
        ["Observation_Count", "observation count", "observation-count", "", "  ", "1count"],
    )
    def test_a_feature_name_must_be_lower_snake_case(self, name):
        with pytest.raises(ValidationError):
            make_feature(name)

    def test_a_feature_id_must_be_a_uuid(self):
        with pytest.raises(ValidationError):
            make_feature().model_copy(update={})
            BehavioralFeature(
                feature_id="not-a-uuid",
                name="observation_count",
                category=FeatureCategory.ACTIVITY_VOLUME,
                scope=FeatureScope.OBSERVATION,
                value=CountValue(value=1),
                observation_id="obs-1",
            )

    def test_a_feature_id_must_match_what_it_measures(self):
        """A feature cannot claim an identity that is not its own."""
        with pytest.raises(ValidationError) as excinfo:
            BehavioralFeature(
                feature_id=derive_feature_id("something_else", FeatureScope.OBSERVATION, "obs-1"),
                name="observation_count",
                category=FeatureCategory.ACTIVITY_VOLUME,
                scope=FeatureScope.OBSERVATION,
                value=CountValue(value=1),
                observation_id="obs-1",
            )
        assert "does not match the identifier derived" in str(excinfo.value)

    def test_observation_scope_requires_an_observation(self):
        with pytest.raises(ValidationError) as excinfo:
            make_feature(observation_id=None)
        assert "needs an observation_id" in str(excinfo.value)

    def test_observation_scope_refuses_a_profile(self):
        with pytest.raises(ValidationError):
            make_feature(profile_id="b" * 8 + "-0000-5000-8000-000000000000")

    def test_profile_scope_refuses_an_observation(self):
        with pytest.raises(ValidationError) as excinfo:
            make_feature(scope=FeatureScope.PROFILE, observation_id="obs-1")
        assert "must not also claim an observation_id" in str(excinfo.value)

    def test_an_unknown_field_is_refused(self):
        payload = {
            "feature_id": derive_feature_id("observation_count", FeatureScope.OBSERVATION, "obs-1"),
            "name": "observation_count",
            "category": FeatureCategory.ACTIVITY_VOLUME,
            "scope": FeatureScope.OBSERVATION,
            "value": CountValue(value=1),
            "observation_id": "obs-1",
        }
        with pytest.raises(ValidationError):
            BehavioralFeature(**payload, notes="something")


# --------------------------------------------------------------------------- #
# Missing required values
# --------------------------------------------------------------------------- #


class TestMissingRequiredValues:
    @pytest.mark.parametrize(
        "field", ["name", "category", "scope", "value", "feature_id"]
    )
    def test_a_required_field_cannot_be_omitted(self, field):
        payload = {
            "feature_id": derive_feature_id("observation_count", FeatureScope.OBSERVATION, "obs-1"),
            "name": "observation_count",
            "category": FeatureCategory.ACTIVITY_VOLUME,
            "scope": FeatureScope.OBSERVATION,
            "value": CountValue(value=1),
            "observation_id": "obs-1",
        }
        payload.pop(field)
        with pytest.raises(ValidationError):
            BehavioralFeature(**payload)

    def test_a_value_object_needs_its_measurement(self):
        with pytest.raises(ValidationError):
            CountValue()
        with pytest.raises(ValidationError):
            LabelSetValue(value=None)


# --------------------------------------------------------------------------- #
# Empty values, and the policy behind them
# --------------------------------------------------------------------------- #


class TestEmptyValues:
    def test_zero_counts_and_empty_sets_are_observed(self):
        """Nothing was found, which is an answer, not a missing measurement."""
        count = CountValue(value=0)
        values = LabelSetValue(value=())
        assert count.status is MeasurementStatus.OBSERVED
        assert values.status is MeasurementStatus.OBSERVED
        assert count.to_dict()["value"] == 0
        assert values.to_dict()["value"] == []

    def test_an_empty_set_cannot_be_declared_absent(self):
        with pytest.raises(ValidationError) as excinfo:
            LabelSetValue(value=(), status=MeasurementStatus.ABSENT)
        assert "an empty set is the measured answer" in str(excinfo.value)

    def test_a_scalar_with_no_data_is_absent_with_no_value(self):
        value = DecimalValue(status=MeasurementStatus.ABSENT, unit="BTC")
        assert value.value is None
        assert value.to_dict()["value"] is None

    def test_absent_and_present_cannot_disagree(self):
        with pytest.raises(ValidationError):
            DecimalValue(value=Decimal("1"), status=MeasurementStatus.ABSENT)
        with pytest.raises(ValidationError):
            CountValue(status=MeasurementStatus.ABSENT, value=3)

    def test_a_zero_amount_is_present_and_observed(self):
        zero = DecimalValue(value=Decimal("0"), unit="BTC", amount_kind=AmountKind.TOTAL_INPUT)
        absent = DecimalValue(status=MeasurementStatus.ABSENT, unit="BTC")
        assert zero.value == Decimal("0")
        assert zero.status is MeasurementStatus.OBSERVED
        assert absent.value is None
        assert zero.value != absent.value

    def test_partial_measurements_must_account_for_exclusions(self):
        with pytest.raises(ValidationError) as excinfo:
            CountValue(value=1, status=MeasurementStatus.PARTIAL)
        assert "how many records were excluded" in str(excinfo.value)
        value = CountValue(
            value=1, status=MeasurementStatus.PARTIAL, excluded_count=2
        )
        assert value.excluded_count == 2

    def test_counts_must_be_whole_and_non_negative(self):
        with pytest.raises(ValidationError):
            CountValue(value=-1)
        with pytest.raises(ValidationError):
            CountValue(value=True)
        with pytest.raises(ValidationError):
            CountValue(value="4")

    def test_a_decimal_measurement_must_be_exact(self):
        with pytest.raises(ValidationError):
            DecimalValue(value=1.5)
        with pytest.raises(ValidationError):
            DecimalValue(value=Decimal("NaN"))

    def test_a_money_value_must_say_which_statistic_it_is(self):
        with pytest.raises(ValidationError) as excinfo:
            DecimalValue(value=Decimal("1"), unit="seconds", amount_kind=AmountKind.MEAN)
        assert "monetary values only" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #


class TestTimestampValues:
    def test_an_aware_value_cannot_be_tagged_naive(self):
        with pytest.raises(ValidationError) as excinfo:
            TimestampValue(value=T0, timestamp_kind=TimestampKind.NAIVE)
        assert "contradicts the value's timezone" in str(excinfo.value)

    def test_a_naive_value_cannot_be_tagged_aware(self):
        naive = datetime(2024, 5, 1, 12, 0)
        with pytest.raises(ValidationError):
            TimestampValue(value=naive, timestamp_kind=TimestampKind.AWARE)
        tagged = TimestampValue(value=naive, timestamp_kind=TimestampKind.NAIVE)
        assert tagged.value == naive

    def test_an_absent_instant_keeps_no_offset(self):
        value = TimestampValue(status=MeasurementStatus.ABSENT)
        assert value.value is None
        assert value.to_dict()["value"] is None
        assert value.to_dict()["timestamp_kind"] == TimestampKind.AWARE.value

    def test_serialisation_never_invents_an_offset(self):
        naive = datetime(2024, 5, 1, 12, 0)
        rendered = TimestampValue(
            value=naive, timestamp_kind=TimestampKind.NAIVE
        ).to_dict()["value"]
        assert rendered == "2024-05-01T12:00:00"
        assert "Z" not in rendered
        assert "+00:00" not in rendered


# --------------------------------------------------------------------------- #
# Ordering, determinism and serialisation
# --------------------------------------------------------------------------- #


class TestOrderedCollections:
    def test_a_distinct_value_set_must_be_sorted_and_unique(self):
        with pytest.raises(ValidationError):
            LabelSetValue(value=("b", "a"))
        with pytest.raises(ValidationError):
            LabelSetValue(value=("a", "a"))
        with pytest.raises(ValidationError):
            LabelSetValue(value=("a", ""))

    def test_an_origin_must_be_sorted_and_unique(self):
        with pytest.raises(ValidationError):
            FeatureOrigin(observation_ids=("obs-2", "obs-1"))
        with pytest.raises(ValidationError):
            FeatureOrigin(ingestion_ids=("ing-1", "ing-1"))


class TestDeterministicSerialisation:
    def test_serialisation_is_repeatable_and_ordered(self):
        feature = make_feature(
            "unique_src_ip_count",
            category=FeatureCategory.NETWORK_BEHAVIOR,
            value=LabelSetValue(value=("10.0.0.1", "10.0.0.2")),
        )
        first = feature.to_dict()
        second = feature.to_dict()
        assert first == second
        assert list(first) == [
            "feature_id",
            "name",
            "category",
            "scope",
            "observation_id",
            "profile_id",
            "value",
            "origin",
        ]
        assert json.dumps(first, sort_keys=False) == json.dumps(second, sort_keys=False)

    def test_serialisation_is_json_safe(self):
        feature = make_feature(
            "median_amount",
            category=FeatureCategory.TRANSACTION_AMOUNTS,
            value=DecimalValue(
                value=Decimal("0.415"), unit="BTC", amount_kind=AmountKind.MEDIAN
            ),
        )
        assert json.loads(json.dumps(feature.to_dict())) == feature.to_dict()

    def test_money_is_never_serialised_as_a_float(self):
        """A float would undo the exactness the decimal type exists to provide."""
        value = DecimalValue(
            value=Decimal("0.1") + Decimal("0.2"), unit="BTC", amount_kind=AmountKind.MEAN
        )
        rendered = value.to_dict()["value"]
        assert isinstance(rendered, str)
        assert rendered == "0.3"
        assert "0.30000000000000004" not in json.dumps(value.to_dict())

    def test_every_value_kind_reports_the_same_key_order(self):
        for value in (
            CountValue(value=1),
            DecimalValue(value=Decimal("1"), unit="BTC"),
            TimestampValue(value=T0),
            LabelValue(value="x"),
            LabelSetValue(value=("x",)),
        ):
            keys = list(value.to_dict())
            assert keys[:3] == ["kind", "status", "value"]
            assert "excluded_count" in keys
            assert "unit" in keys

    def test_a_feature_renders_readably(self):
        feature = make_feature(
            "first_seen",
            category=FeatureCategory.TEMPORAL_ACTIVITY,
            value=TimestampValue(value=T0, timestamp_kind=TimestampKind.AWARE),
        )
        assert "first_seen" in str(feature)
        assert "AWARE" in str(feature)


class TestDerivedIdentifiers:
    def test_the_identifier_depends_only_on_what_is_described(self):
        first = derive_feature_id("observation_count", FeatureScope.OBSERVATION, "obs-1")
        second = derive_feature_id("observation_count", FeatureScope.OBSERVATION, "obs-1")
        assert first == second

    @pytest.mark.parametrize(
        "name, scope, observation_id, profile_id",
        [
            ("other_count", FeatureScope.OBSERVATION, "obs-1", None),
            ("observation_count", FeatureScope.PROFILE, None, "p"),
            ("observation_count", FeatureScope.OBSERVATION, "obs-2", None),
        ],
    )
    def test_a_different_subject_gives_a_different_identifier(
        self, name, scope, observation_id, profile_id
    ):
        base = derive_feature_id("observation_count", FeatureScope.OBSERVATION, "obs-1")
        assert (
            derive_feature_id(name, scope, observation_id, profile_id) != base
        )

    def test_the_identifier_is_a_uuid(self):
        identifier = derive_feature_id("observation_count", FeatureScope.OBSERVATION, "obs-1")
        assert re.match(r"^[0-9a-f-]{36}$", identifier)


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


class TestImmutableModels:
    @pytest.mark.parametrize(
        "attribute, value",
        [
            ("name", "other_count"),
            ("category", FeatureCategory.NETWORK_BEHAVIOR),
            ("observation_id", "obs-9"),
        ],
    )
    def test_a_feature_cannot_be_edited(self, attribute, value):
        feature = make_feature()
        with pytest.raises(ValidationError):
            setattr(feature, attribute, value)

    def test_a_value_cannot_be_edited(self):
        value = CountValue(value=1)
        with pytest.raises(ValidationError):
            value.value = 2

    def test_an_origin_cannot_be_edited(self):
        origin = FeatureOrigin(observation_ids=("obs-1",))
        with pytest.raises(ValidationError):
            origin.observation_ids = ("obs-2",)

    def test_a_frozen_model_survives_a_shallow_copy_attempt(self):
        feature = make_feature()
        with pytest.raises(ValidationError):
            feature.value.value = 99

    def test_the_model_defines_no_mutating_method(self):
        """Checked against what the module defines, not against pydantic's base.

        ``dir()`` on a pydantic model includes inherited machinery such as
        ``update_forward_refs``, which is a class-building tool and not a way
        to edit a constructed feature.  Scanning what this module actually
        declares keeps the assertion about the code under test.
        """
        defined = {
            name
            for name, value in vars(BehavioralFeature).items()
            if callable(value)
        }
        for name in defined:
            assert not identifier_words(name)[0] in {"set", "add", "update", "pop", "clear"}
            assert not name.startswith(("_set", "_add", "_update"))


# --------------------------------------------------------------------------- #
# Decimal arithmetic
# --------------------------------------------------------------------------- #


class TestDecimalStatistics:
    def test_the_context_is_pinned_not_inherited(self):
        """A caller who lowers the ambient precision must not change a mean."""
        original = decimal.getcontext().prec
        try:
            decimal.getcontext().prec = 3
            assert divide_exactly(Decimal(10), 3) == Decimal(
                "3.333333333333333333333333333333333"
            )
        finally:
            decimal.getcontext().prec = original

    def test_the_pinned_context_is_documented_and_constant(self):
        assert AMOUNT_STATISTICS_CONTEXT.prec == 34
        assert AMOUNT_STATISTICS_CONTEXT.rounding == decimal.ROUND_HALF_EVEN

    def test_division_refuses_a_non_positive_divisor(self):
        with pytest.raises(ValueError):
            divide_exactly(Decimal(1), 0)
        with pytest.raises(ValueError):
            divide_exactly(Decimal(1), -2)

    def test_a_mean_keeps_full_precision(self):
        assert divide_exactly(Decimal("0.1") + Decimal("0.2"), 2) == Decimal("0.15")


# --------------------------------------------------------------------------- #
# Empty-set policy, stated as a test
# --------------------------------------------------------------------------- #


class TestDocumentedEmptySetPolicy:
    def test_the_policy_is_written_down_where_it_is_implemented(self):
        source = pathlib.Path(behavior_models.__file__).read_text(encoding="utf-8")
        assert "ABSENT" in source
        assert "Zero and ``ABSENT`` are different answers" in source

    def test_count_and_amount_absent_means_different_underlying_facts(self):
        no_graph = CountValue(status=MeasurementStatus.ABSENT)
        empty_graph = CountValue(value=0)
        no_amounts = DecimalValue(status=MeasurementStatus.ABSENT, unit="BTC")
        zero_amount = DecimalValue(value=Decimal("0"), unit="BTC")
        assert no_graph.value is None and empty_graph.value == 0
        assert no_amounts.value is None and zero_amount.value == Decimal("0")


# --------------------------------------------------------------------------- #
# Anti-attribution safeguards
# --------------------------------------------------------------------------- #


class TestNoPartyFields:
    @pytest.mark.parametrize(
        "model",
        [
            BehavioralFeature,
            CountValue,
            DecimalValue,
            TimestampValue,
            LabelValue,
            LabelSetValue,
            FeatureOrigin,
        ],
    )
    def test_no_model_declares_a_banned_field(self, model):
        for field in model.model_fields:
            for word in identifier_words(field):
                assert word not in BANNED_FIELD_WORDS, "%s.%s" % (
                    model.__name__,
                    field,
                )

    def test_no_category_is_named_after_a_judgement(self):
        for category in FeatureCategory:
            for word in identifier_words(category.name):
                assert word not in BANNED_FIELD_WORDS
            for word in identifier_words(category.value):
                assert word not in BANNED_FIELD_WORDS

    def test_the_negative_assertion_is_available_and_false(self):
        assert make_feature().describes_attribution() is False

    def test_no_method_resolves_a_party(self):
        for name in dir(BehavioralFeature):
            for word in identifier_words(name):
                assert word not in {"owner", "actor", "identity", "attribution"} or (
                    name in ALLOWED_NEGATIVE_ASSERTIONS
                )


class TestForbiddenVocabularyScanner:
    """AST scan of this package and its tests.

    Scans identifiers and imports, not prose.  The docstrings in this package
    say outright that scoring, clustering and attribution are *not* implemented,
    and a scanner that read prose would fail on exactly the sentences that make
    the boundary clear.
    """

    def test_no_identifier_carries_a_banned_word(self):
        offenders = []
        for path in PACKAGE_FILES + STEP5_TEST_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for identifier in iter_identifiers(tree):
                if identifier in ALLOWED_NEGATIVE_ASSERTIONS:
                    continue
                for word in identifier_words(identifier):
                    if word in BANNED_FIELD_WORDS:
                        offenders.append(
                            "%s: %s (%s)" % (path.name, identifier, word)
                        )
        assert not offenders, "banned identifiers: " + "; ".join(sorted(offenders))

    def test_the_scanner_would_notice_a_banned_identifier(self):
        """A scanner that cannot fail is not a safeguard; prove it can."""
        tree = ast.parse("def attribute_actor(record):\n    return record\n")
        found = {
            word
            for identifier in iter_identifiers(tree)
            for word in identifier_words(identifier)
            if word in BANNED_FIELD_WORDS
        }
        assert found == {"actor"}

    def test_the_scanner_ignores_prose_on_purpose(self):
        tree = ast.parse('"""This module does not do attribution."""\n')
        assert list(iter_identifiers(tree)) == []

    def test_no_forbidden_module_is_imported_anywhere(self):
        offenders = []
        for path in ALL_SOURCE_FILES + STEP5_TEST_FILES:
            roots = imported_roots(ast.parse(path.read_text(encoding="utf-8")))
            shared = roots & FORBIDDEN_GLOBAL_MODULES
            if shared:
                offenders.append("%s: %s" % (path.name, sorted(shared)))
        assert not offenders, "forbidden imports: " + "; ".join(offenders)

    def test_this_package_imports_no_interface_or_store(self):
        offenders = []
        for path in PACKAGE_FILES + STEP5_TEST_FILES:
            roots = imported_roots(ast.parse(path.read_text(encoding="utf-8")))
            shared = roots & FORBIDDEN_HERE_MODULES
            if shared:
                offenders.append("%s: %s" % (path.name, sorted(shared)))
        assert not offenders, "forbidden imports: " + "; ".join(offenders)

    def test_the_package_declares_a_complete_public_surface(self):
        assert set(behavior_models.__all__) <= set(dir(behavior_models))
        for name in behavior_models.__all__:
            assert getattr(behavior_models, name) is not None

    def test_no_public_name_mentions_a_banned_concept(self):
        for name in behavior_models.__all__:
            for word in identifier_words(name):
                assert word not in BANNED_FIELD_WORDS


class TestPackageSurface:
    """The package's public surface is one short, complete, sorted list.

    Pinned so that a later step cannot widen what Step 5 offers by accident:
    every name a caller can reach is a name somebody chose on purpose, and the
    modules' own lists cannot drift away from the package's.
    """

    def test_every_exported_name_exists(self):
        for name in behavior_package.__all__:
            assert getattr(behavior_package, name) is not None, name

    def test_the_export_list_is_sorted(self):
        assert list(behavior_package.__all__) == sorted(behavior_package.__all__)

    def test_each_module_exports_only_names_the_package_exports(self):
        exported = set(behavior_package.__all__)
        for module in (behavior_models, behavior_features, behavior_profiles):
            assert set(module.__all__) <= exported, module.__name__
            assert list(module.__all__) == sorted(module.__all__), module.__name__

    def test_nothing_public_leaks_in_from_a_module_without_being_declared(self):
        declared = set(behavior_package.__all__)
        reachable = {
            name
            for name in dir(behavior_package)
            if not name.startswith("_")
            and name not in {"annotations", "features", "models", "profiles"}
        }
        assert reachable == declared

    def test_no_exported_name_mentions_a_banned_concept(self):
        for name in behavior_package.__all__:
            for word in identifier_words(name):
                assert word not in BANNED_FIELD_WORDS, name

    def test_the_package_documents_the_boundary_it_keeps(self):
        """The module docstring is part of the contract, so it is asserted.

        A package whose docstring described scoring would contradict everything
        the models refuse to hold, and nothing else in the suite would notice.
        """
        rendered = (behavior_package.__doc__ or "").lower()
        assert "it does not interpret" in rendered
        for concept in ("anomaly score", "clustering", "prediction"):
            assert concept in rendered, concept


class TestInputObjectsAreNotCopiedOrAltered:
    def test_the_models_hold_no_reference_to_a_source_record(self):
        """A feature describes; it does not keep the record it described.

        Holding a reference would be a quiet way for a later stage to mutate
        frozen evidence through this layer.
        """
        feature = make_feature()
        payload = feature.to_dict()
        assert "observation" not in payload
        assert "record" not in payload
        assert set(payload["origin"]) == {
            "observation_ids",
            "ingestion_ids",
            "source_files",
            "source_records",
            "relationship_ids",
        }

    def test_a_deep_copy_of_a_feature_stays_equal(self):
        feature = make_feature(
            value=LabelSetValue(value=("a", "b")),
        )
        assert copy.deepcopy(feature) == feature
