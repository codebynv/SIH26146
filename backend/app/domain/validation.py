"""Canonical field-level validation and normalisation types (SIH26146 - Step 3).

This module is the single source of truth for how raw input values become
trustworthy, canonical Python values.  It is deliberately free of any
knowledge about files, parsers, ingestion orchestration or cross-layer
correlation so that it can be shared by the ingestion layer and the domain
model without either importing the other.

Design principles
-----------------
* **Never silently corrupt.**  A value that cannot be parsed is rejected with
  a descriptive :class:`ValueError`; the caller decides what to do with the
  field.  Guessing is not an option.
* **Never invent precision or timezone.**  Numeric epoch timestamps are
  refused rather than guessed (seconds vs. milliseconds is undecidable).
  Timezone-naive timestamps are preserved as naive and are *never* stamped
  with the host's local timezone.
* **Money is :class:`~decimal.Decimal`.**  Binary floating point cannot
  represent Bitcoin amounts exactly, so amounts and fees are parsed as
  ``Decimal`` and serialised as floats only at the JSON boundary.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Mapping

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    IPvAnyAddress,
    PlainSerializer,
)

__all__ = [
    "MIN_PORT",
    "MAX_PORT",
    "UTC",
    "NULL_TOKENS",
    "CANONICAL_FIELDS",
    "LIST_FIELDS",
    "canonicalize_evidence",
    "parse_timestamp_detailed",
    "is_naive",
    "coerce_amount",
    "coerce_port",
    "coerce_asn",
    "coerce_text",
    "coerce_list",
    "coerce_text_list",
    "coerce_address_list",
    "coerce_amount_list",
    "canonicalize_key",
    "UtcTimestamp",
    "IpAddress",
    "PortNumber",
    "BtcAmount",
    "BtcAddress",
    "BtcAddressList",
    "BtcAmountList",
    "TextValue",
    "AsNumber",
]

MIN_PORT = 0
MAX_PORT = 65535
UTC = timezone.utc

#: Tokens that various exporters use to mean "no value".
NULL_TOKENS = frozenset(
    {
        "",
        "-",
        "--",
        "n/a",
        "na",
        "n.a.",
        "null",
        "none",
        "nil",
        "nan",
        "undefined",
        "?",
        "unknown",
    }
)

_LIST_SPLIT_PATTERN = re.compile(r"[;|]")
_INT_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_ASN_PREFIX_RE = re.compile(r"^AS", re.IGNORECASE)
_ASN_DIGITS_RE = re.compile(r"\d+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Timestamps that ISO-8601 alone does not cover but that are common in
# exported investigative data.  Tried in order after fromisoformat().
_FALLBACK_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


def _is_null_token(value: Any) -> bool:
    """Return ``True`` when *value* is one of the recognised "empty" markers."""
    return isinstance(value, str) and value.strip().lower() in NULL_TOKENS


def _blank_to_none(value: Any) -> Any:
    """Map common "empty" tokens onto ``None`` before type coercion."""
    if value is None or _is_null_token(value):
        return None
    if isinstance(value, str):
        return value.strip()
    return value


def is_naive(value: datetime) -> bool:
    """Return ``True`` when *value* carries no usable timezone information."""
    return value.tzinfo is None or value.tzinfo.utcoffset(value) is None


def parse_timestamp_detailed(value: Any) -> tuple[datetime, bool]:
    """Parse *value* into a datetime, also reporting whether it was naive.

    Returns ``(datetime, was_naive)``.  Aware inputs are converted to UTC.
    Naive inputs are returned **unchanged and still naive** -- the host's
    local timezone is never applied, because doing so would silently shift
    investigative data by an unknown offset.  Callers surface a warning so
    the ambiguity stays visible to the analyst.
    """
    parsed = _coerce_datetime(value)
    if is_naive(parsed):
        return parsed, True
    return parsed.astimezone(UTC), False


def _coerce_datetime(value: Any) -> datetime:
    """Coerce *value* into a :class:`datetime` without inventing a timezone."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, time):
        raise ValueError(
            "a time-of-day value ('%s') has no date and cannot be used as a "
            "timestamp" % value.isoformat()
        )
    if isinstance(value, bool):
        raise ValueError("booleans are not valid timestamps")
    if isinstance(value, (int, float, Decimal)):
        raise ValueError(
            "numeric timestamp %r is ambiguous (epoch seconds or "
            "milliseconds?) and is refused rather than guessed; supply an "
            "ISO-8601 string instead" % (value,)
        )
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty string is not a valid timestamp")
        candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
        for fmt in _FALLBACK_TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        raise ValueError(
            "could not parse %r as a timestamp; expected ISO-8601 such as "
            "'2024-05-01T12:30:00Z'" % text
        )
    raise ValueError("unsupported timestamp value of type %r" % type(value).__name__)


def _timestamp_after(value: datetime) -> datetime:
    """Normalise to UTC only when the datetime actually carries a timezone."""
    if is_naive(value):
        return value
    return value.astimezone(UTC)


def coerce_amount(value: Any) -> Decimal:
    """Coerce *value* into a non-negative :class:`Decimal` Bitcoin amount.

    Rejects booleans, non-numeric text and negative values.  Thousands
    separators and currency symbols are not accepted; ambiguous input is an
    error rather than a guess.
    """
    value = _blank_to_none(value)
    if value is None:
        raise ValueError("a required numeric amount was empty")
    if isinstance(value, bool):
        raise ValueError("booleans are not valid amounts")
    if isinstance(value, Decimal):
        candidate = value
    elif isinstance(value, int):
        candidate = Decimal(value)
    elif isinstance(value, float):
        candidate = Decimal(str(value))
    elif isinstance(value, str):
        text = value.strip().replace("_", "")
        if not _DECIMAL_RE.match(text):
            raise ValueError("%r is not a valid numeric amount" % (value,))
        try:
            candidate = Decimal(text)
        except InvalidOperation as exc:  # pragma: no cover - guarded by regex
            raise ValueError("%r is not a valid numeric amount" % (value,)) from exc
    else:
        raise ValueError("unsupported amount value of type %r" % type(value).__name__)
    if candidate < 0:
        raise ValueError("amount %s must be non-negative" % candidate)
    return candidate


def coerce_port(value: Any) -> int:
    """Coerce *value* into a port in the inclusive range 0-65535."""
    value = _blank_to_none(value)
    if value is None:
        raise ValueError("a required port was empty")
    if isinstance(value, bool):
        raise ValueError("booleans are not valid ports")
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, float):
        if not float(value).is_integer():
            raise ValueError("port %r must be a whole number" % (value,))
        candidate = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not _INT_RE.match(text):
            raise ValueError("port %r is not a valid integer" % (value,))
        candidate = int(text)
    else:
        raise ValueError("unsupported port value of type %r" % type(value).__name__)
    if not (MIN_PORT <= candidate <= MAX_PORT):
        raise ValueError(
            "port %d is outside the valid range %d-%d" % (candidate, MIN_PORT, MAX_PORT)
        )
    return candidate


def coerce_asn(value: Any) -> int:
    """Coerce *value* into a non-negative ASN integer.

    Accepts ``15169``, ``"15169"``, ``"AS15169"`` and ``"as15169"``.  A
    trailing descriptive label (``"AS15169 Google"``) is tolerated; the
    leading integer is used.
    """
    value = _blank_to_none(value)
    if value is None:
        raise ValueError("a required ASN was empty")
    if isinstance(value, bool):
        raise ValueError("booleans are not valid ASNs")
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str):
        text = _ASN_PREFIX_RE.sub("", value.strip(), count=1).strip()
        match = _ASN_DIGITS_RE.search(text)
        if not match:
            raise ValueError("ASN %r does not contain a number" % (value,))
        candidate = int(match.group())
    else:
        raise ValueError("unsupported ASN value of type %r" % type(value).__name__)
    if candidate < 0:
        raise ValueError("ASN %d must be non-negative" % candidate)
    return candidate


def coerce_text(value: Any) -> str:
    """Coerce *value* into a non-empty stripped string."""
    value = _blank_to_none(value)
    if value is None:
        raise ValueError("a required text value was empty")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty string is not a valid text value")
        return text
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    raise ValueError("unsupported text value of type %r" % type(value).__name__)


def coerce_list(value: Any) -> list[Any]:
    """Coerce *value* into a list without losing information.

    Handles real lists, JSON-encoded lists, delimited scalars and bare
    scalars.  An empty or null value yields an empty list.
    """
    value = _blank_to_none(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            # A leading '[' is a clear statement of intent: this is a list.
            # If it does not parse, that is malformed input to be reported,
            # not a scalar address that happens to start with a bracket.
            try:
                decoded = json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    "value starts with '[' so it was read as a JSON list, but it "
                    "is not valid JSON: %r" % value
                ) from exc
            if not isinstance(decoded, list):
                raise ValueError(
                    "expected a JSON list, got %r" % type(decoded).__name__
                )
            return [item for item in decoded if item is not None]
        if not text:
            return []
        parts = [part.strip() for part in _LIST_SPLIT_PATTERN.split(text)]
        parts = [part for part in parts if part]
        return parts or [text]
    return [value]


def coerce_text_list(value: Any) -> list[str]:
    """Coerce *value* into a list of non-empty strings, preserving order."""
    result: list[str] = []
    for index, item in enumerate(coerce_list(value)):
        if isinstance(item, (list, tuple, dict)):
            raise ValueError("nested collections are not valid list items: %r" % (item,))
        try:
            result.append(coerce_text(item))
        except ValueError as exc:
            raise ValueError("list item %d: %s" % (index, exc)) from exc
    return result


def coerce_address_list(value: Any) -> list[str]:
    """Coerce *value* into a list of Bitcoin address strings.

    Address *shape* is intentionally not enforced beyond non-emptiness:
    base58/bech32 validation would reject legitimate future encodings, and
    rejecting a real address is a worse failure than carrying it through.
    """
    return coerce_text_list(value)


def coerce_amount_list(value: Any) -> list[Decimal]:
    """Coerce *value* into a list of non-negative :class:`Decimal` amounts."""
    result: list[Decimal] = []
    for index, item in enumerate(coerce_list(value)):
        if isinstance(item, (list, tuple, dict)):
            raise ValueError("amount list item %d is a collection, not a number" % index)
        try:
            result.append(coerce_amount(item))
        except ValueError as exc:
            raise ValueError("amount list item %d: %s" % (index, exc)) from exc
    return result


def canonicalize_key(key: str) -> str:
    """Normalise a raw header/key into a comparable form.

    Lowercases, replaces runs of non-alphanumeric characters with a single
    underscore and strips the result, so ``"Src IP"``, ``"src-ip"`` and
    ``"SRC_IP"`` all collapse to ``"src_ip"``.
    """
    return _NON_ALNUM_RE.sub("_", str(key).strip().lower()).strip("_")


def canonicalize_evidence(value: Any) -> Any:
    """Render *value* as a deterministic, JSON-serialisable structure.

    Unmapped source fields are *evidence*, not decoration.  Two records that
    agree on every canonical SIH field but disagree on an unmapped field are
    two pieces of evidence, not one, so an unmapped field has to take part in
    any identity derived from a record -- both the Step 2 deduplication
    fingerprint and the Step 3 ``observation_id``.  Ignoring it in either
    place would let distinct evidence collapse into a single record.

    The rendering is:

    * **deterministic for JSON-like values** -- the same JSON-like value always
      yields the same structure, so re-ingesting the same file reproduces the
      same identity;
    * **order-insensitive for mappings** -- ``{"a": 1, "b": 2}`` and
      ``{"b": 2, "a": 1}`` are the same evidence;
    * **order-sensitive for sequences** -- the order of a list is itself
      evidence;
    * **type-aware** -- ``1``, ``"1"`` and ``True`` are three different claims
      in a forensic file, not one, so each carries its type tag.

    A value of a type with no justified, deterministic equality semantics is
    given a **unique token** and is therefore distinct from every other opaque
    value, including another instance of the very same object.  Using the type
    name alone would be wrong: it would make two different values of that type
    canonicalize identically and silently merge two pieces of evidence into
    one.  Over-separating costs an extra record; over-merging destroys
    evidence, and only one of those can be recovered.

    The consequence is deliberate and narrow: a record holding an opaque value
    never deduplicates against anything, and never shares an ``observation_id``
    with another record.  Its fingerprint is not reproducible across runs,
    which is acceptable precisely because such a record is never used to merge
    or identify anything.
    """
    if value is None:
        return ["none"]
    # bool is a subclass of int, so it must be tested first.
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", repr(value)]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (bytes, bytearray)):
        return ["bytes", bytes(value).hex()]
    if isinstance(value, Mapping):
        pairs = [
            [canonicalize_evidence(key), canonicalize_evidence(item)]
            for key, item in value.items()
        ]
        pairs.sort(key=repr)
        return ["map", pairs]
    if isinstance(value, (list, tuple)):
        return ["seq", [canonicalize_evidence(item) for item in value]]
    if isinstance(value, (set, frozenset)):
        # Sorting the rendered members also orders any unique opaque tokens, so
        # a set holding an opaque value is itself non-reproducible and can only
        # ever fail to merge.
        return ["set", sorted(repr(canonicalize_evidence(item)) for item in value)]
    return ["opaque", type(value).__name__, uuid.uuid4().hex]


def _optional_amount(value: Any) -> Any:
    return None if value is None or _is_null_token(value) else coerce_amount(value)


def _optional_port(value: Any) -> Any:
    return None if value is None or _is_null_token(value) else coerce_port(value)


def _optional_asn(value: Any) -> Any:
    return None if value is None or _is_null_token(value) else coerce_asn(value)


def _optional_text(value: Any) -> Any:
    return None if value is None or _is_null_token(value) else coerce_text(value)


def _optional_ip(value: Any) -> Any:
    return None if value is None or _is_null_token(value) else value


def _optional_list(value: Any) -> Any:
    """``None`` for a null token, otherwise the value unchanged.

    List-valued canonical fields are normalised by
    :func:`canonicalize_evidence` callers through their own ``BeforeValidator``
    (``_coerce_address_list_value`` / ``_coerce_amount_list_value``); this
    helper exists for optional list-typed fields that only need null-token
    handling.
    """
    return None if value is None or _is_null_token(value) else value


def _coerce_address_list_value(value: Any) -> Any:
    if value is None or _is_null_token(value):
        return []
    return coerce_address_list(value)


def _coerce_amount_list_value(value: Any) -> Any:
    if value is None or _is_null_token(value):
        return []
    return coerce_amount_list(value)


def _serialize_decimal(value: Decimal) -> float:
    return float(value)


#: A timestamp normalised to UTC when it carries a timezone, and otherwise
#: preserved verbatim as a naive datetime (never localised).
UtcTimestamp = Annotated[
    datetime,
    BeforeValidator(_coerce_datetime),
    AfterValidator(_timestamp_after),
]

#: An IPv4 or IPv6 address.  Hostnames are rejected: they cannot be compared
#: against a routing table or a geolocation database.
IpAddress = Annotated[IPvAnyAddress, BeforeValidator(_optional_ip)]

#: A port number in the inclusive range 0-65535.
PortNumber = Annotated[int, BeforeValidator(_optional_port)]

#: A non-negative Bitcoin amount held as ``Decimal`` internally.
BtcAmount = Annotated[
    Decimal,
    BeforeValidator(_optional_amount),
    Field(ge=0),
    PlainSerializer(_serialize_decimal, return_type=float, when_used="json"),
]

#: Free text preserved verbatim (stripped), e.g. a TXID, country or script type.
TextValue = Annotated[str, BeforeValidator(_optional_text)]

#: A non-negative Bitcoin address string.
BtcAddress = TextValue

#: An autonomous system number, accepting ``AS15169``-style input.
AsNumber = Annotated[int, BeforeValidator(_optional_asn)]

#: A free-text list of Bitcoin addresses, tolerant of JSON and delimited input.
BtcAddressList = Annotated[list[BtcAddress], BeforeValidator(_coerce_address_list_value)]

#: A list of non-negative Bitcoin amounts, tolerant of JSON and delimited input.
BtcAmountList = Annotated[list[BtcAmount], BeforeValidator(_coerce_amount_list_value)]

#: The canonical SIH26146 field set, in declaration order.
CANONICAL_FIELDS: tuple[str, ...] = (
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "txid",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
    "fee",
    "script_type",
    "geo_country",
    "asn",
)

#: Canonical fields whose values are lists.
LIST_FIELDS: frozenset[str] = frozenset(
    {"input_addresses", "output_addresses", "input_amounts", "output_amounts"}
)
