"""Canonical serialisation for Step 10 evaluation output.

Two runs of the same input must be *comparable*. Python's default JSON
encoding is not: dictionary insertion order leaks into the bytes, ``Decimal``
and ``datetime`` are not encodable, and a single run-scoped identifier would
make every run look different even when nothing changed.

This module produces one canonical form so a byte comparison is a real test:

* keys sorted, separators fixed, UTF-8, no whitespace drift;
* ``Decimal`` rendered as its exact string, never as a float;
* datetimes rendered in UTC with an explicit offset;
* sets and other unordered containers sorted before encoding.

Run-scoped values
-----------------
``ingestion_id`` and ``ingested_at`` are identifiers *of an ingestion run*, not
of the evidence. Two runs of the same file legitimately carry different ones,
so comparing them would report a difference that is not a defect.

They are not deleted. They are replaced with an explicit placeholder by
:func:`canonical`, and the harness separately asserts that each was present and
well-formed. That keeps the substitution visible and auditable rather than
silently dropping information.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "RUN_SCOPED_PLACEHOLDER",
    "RUN_SCOPED_KEYS",
    "RUN_SCOPED_KEY_PREFIXES",
    "RUN_SCOPED_VALUE_NAMES",
    "canonical",
    "canonical_bytes",
    "digest",
    "sha256_file",
    "find_run_scoped",
    "is_run_scoped",
    "is_run_scoped_value",
    "load_expected",
]

#: Substituted for a value that legitimately differs between two runs.
RUN_SCOPED_PLACEHOLDER = "<run-scoped>"

#: Keys whose values identify an ingestion *run* rather than the evidence.
#:
#: Step 2 assigns ``ingestion_id`` with :func:`new_ingestion_id` at the moment
#: an ingestion starts, and stamps ``ingested_at`` at the same moment. Two runs
#: of the same file therefore carry different values, by design and with no
#: injection point, and comparing them would report a difference that is not a
#: defect.
#:
#: Keys matched by prefix are included because the identifier surfaces under
#: derived names too: Step 5 exposes it as the behavioural feature
#: ``ingestion_id_values`` and as ``provenance_ingestion_ids``. Normalising only
#: the bare key would leave those fields to differ between runs.
RUN_SCOPED_KEYS = frozenset({"ingestion_id", "ingested_at", "library_version"})

#: Prefixes for keys that *contain* a run-scoped identifier under a derived
#: name. Deliberately narrow: ``unique_ingestion_count`` is a stable count and
#: must be left alone, which is why the prefix is anchored to the start of the
#: key rather than searching within it.
RUN_SCOPED_KEY_PREFIXES = ("ingestion_id", "provenance_ingestion")

#: Field *names* whose measured value is a run-scoped identifier.
#:
#: Step 5 emits the ingestion identifier as a behavioural feature, so the
#: run-scoped value sits under a generic ``"value"`` key with a sibling
#: ``"name"`` naming the measurement. A key rule cannot see that, so the
#: measurement name is matched instead and only its value is substituted.
RUN_SCOPED_VALUE_NAMES = frozenset({"ingestion_id_values"})


def is_run_scoped(key: str) -> bool:
    """True when *key* names a value that is expected to change per run."""
    return key in RUN_SCOPED_KEYS or key.startswith(RUN_SCOPED_KEY_PREFIXES)


def is_run_scoped_value(key: str, sibling_name: Any) -> bool:
    """True when ``key`` holds a value that is a run-scoped identifier.

    *sibling_name* is the ``"name"`` of the enclosing measurement, which is how
    a generic ``"value"`` key is identified as carrying an ingestion id.
    """
    if is_run_scoped(key):
        return True
    return key == "value" and sibling_name in RUN_SCOPED_VALUE_NAMES


def _coerce(value: Any) -> Any:
    """Reduce *value* to something with one canonical JSON representation."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # repr() round-trips, so the text is exact and platform independent.
        return repr(value)
    if isinstance(value, Decimal):
        # Exact decimal text. str(Decimal) keeps trailing zeros, which is the
        # value as supplied; a float would silently round it.
        return str(value)
    if isinstance(value, Enum):
        return _coerce(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            # A naive timestamp stays naive; stamping it with an offset here
            # would invent the very fact Step 2 refuses to invent.
            return value.isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(_coerce(item) for item in value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (PurePath, Path)):
        return value.as_posix()
    if isinstance(value, Mapping):
        return _coerce_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_coerce(item) for item in value]
    if hasattr(value, "model_dump"):
        return _coerce(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _coerce(value.to_dict())
    return str(value)


def _coerce_mapping(value: Mapping) -> dict:
    sibling_name = value.get("name")
    result: dict = {}
    for key, item in value.items():
        name = str(key)
        result[name] = (
            RUN_SCOPED_PLACEHOLDER
            if is_run_scoped_value(name, sibling_name)
            else _coerce(item)
        )
    return result


def canonical(value: Any) -> Any:
    """Return the canonical JSON-ready form of *value*."""
    return _coerce(value)


def canonical_bytes(value: Any) -> bytes:
    """Serialise *value* to canonical UTF-8 JSON bytes."""
    return json.dumps(
        canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """SHA-256 of the canonical encoding of *value*."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks so size is not a concern."""
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sorted_by(values: Iterable[Any], key) -> list:
    """Sort *values* by *key*, for containers whose order is not semantic."""
    return sorted(values, key=key)


def find_run_scoped(payload: Any, prefix: str = "") -> list:
    """Every path at which a run-scoped placeholder was substituted.

    Reported by the harness so a reader can see exactly which fields were
    compared with the substitution applied, rather than having to trust that
    nothing was quietly dropped.
    """
    found: list = []
    if isinstance(payload, Mapping):
        sibling_name = payload.get("name")
        for key, item in payload.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            if is_run_scoped_value(str(key), sibling_name):
                found.append(path)
            else:
                found.extend(find_run_scoped(item, path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            found.extend(find_run_scoped(item, "%s[%d]" % (prefix, index)))
    return found


def load_expected(path: Optional[Path]) -> Optional[Any]:
    """Load a stored expected-output fixture, or None when absent."""
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)