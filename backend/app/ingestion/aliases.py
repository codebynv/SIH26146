"""Input field aliases for the canonical SIH26146 schema (SIH26146 - Step 2).

Exports from Bitcoin node tooling, chain analytics vendors and network
monitors rarely agree on column names.  This module maps the reasonable
variants onto the fourteen canonical SIH fields so that downstream code only
ever sees one vocabulary.

Matching is done on *canonicalised* keys (see
:func:`~backend.app.domain.validation.canonicalize_key`), so ``"Src IP"``,
``"src-ip"`` and ``"SRC_IP"`` are all equivalent.  A raw key always takes
precedence over an alias, which means an alias can never shadow a genuine
canonical field.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from backend.app.domain.validation import CANONICAL_FIELDS, canonicalize_key

__all__ = [
    "FIELD_ALIASES",
    "REVERSE_ALIASES",
    "resolve_key",
    "normalize_keys",
]

#: Canonical SIH field -> accepted input aliases (already canonicalised).
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "ts",
        "time",
        "datetime",
        "date_time",
        "event_time",
        "eventtime",
        "timestamp_utc",
        "utc_timestamp",
        "observed_at",
        "first_seen",
        "block_time",
    ),
    "src_ip": (
        "source_ip",
        "srcip",
        "src_ip_address",
        "source_address",
        "client_ip",
        "origin_ip",
        "from_ip",
    ),
    "dst_ip": (
        "dest_ip",
        "destination_ip",
        "dst_ip_address",
        "destination_address",
        "server_ip",
        "target_ip",
        "to_ip",
    ),
    "src_port": (
        "source_port",
        "sport",
        "srcport",
        "source_port_number",
        "client_port",
        "local_port",
    ),
    "dst_port": (
        "dest_port",
        "destination_port",
        "dport",
        "dstport",
        "destination_port_number",
        "server_port",
        "remote_port",
    ),
    "txid": (
        "tx_id",
        "transaction_id",
        "tx_hash",
        "transaction_hash",
        "hash",
        "txid_hex",
        "tx",
    ),
    "input_addresses": (
        "input_address",
        "inputs",
        "input_addr",
        "input_address_list",
        "from_addresses",
        "from_address",
        "spent_outputs",
    ),
    "output_addresses": (
        "output_address",
        "outputs",
        "output_addr",
        "output_address_list",
        "to_addresses",
        "to_address",
        "received_outputs",
    ),
    "input_amounts": (
        "input_amount",
        "in_amounts",
        "input_value",
        "input_values",
        "from_amounts",
        "input_btc",
    ),
    "output_amounts": (
        "output_amount",
        "out_amounts",
        "output_value",
        "output_values",
        "to_amounts",
        "output_btc",
    ),
    "fee": (
        "fee_btc",
        "transaction_fee",
        "tx_fee",
        "txn_fee",
        "fee_satoshis",
        "fee_sat",
        "fees",
        "miner_fee",
    ),
    "script_type": (
        "script",
        "script_pubkey_type",
        "tx_type",
        "transaction_type",
        "lockup_type",
        "output_type",
    ),
    "geo_country": (
        "country",
        "country_code",
        "geo",
        "geolocation",
        "src_country",
        "dest_country",
    ),
    "asn": (
        "as_number",
        "asn_number",
        "autonomous_system",
        "autonomous_system_number",
        "origin_asn",
        "src_asn",
    ),
}


def _build_reverse() -> dict[str, str]:
    reverse: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            key = canonicalize_key(alias)
            # First writer wins so the mapping is deterministic and a
            # collision can never silently redirect a canonical field.
            reverse.setdefault(key, canonical)
    return reverse


#: Canonicalised alias -> canonical field.
REVERSE_ALIASES: dict[str, str] = _build_reverse()


def resolve_key(raw_key: str) -> str | None:
    """Return the canonical field for *raw_key*, or ``None`` if unmapped."""
    key = canonicalize_key(raw_key)
    if key in CANONICAL_FIELDS:
        return key
    return REVERSE_ALIASES.get(key)


def normalize_keys(raw: Mapping[object, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Split a raw record into canonical values and unmapped extras.

    Returns ``(canonical, extra)``.  Keys that map to no canonical SIH field
    are preserved in *extra* rather than dropped, so that useful unknown
    evidence survives the whole pipeline.
    """
    canonical: dict[str, object] = {}
    extra: dict[str, object] = {}
    for raw_key, value in raw.items():
        key = canonicalize_key(raw_key)
        field = resolve_key(key)
        if field is None:
            extra[str(raw_key)] = value
            continue
        # A canonical key always beats an alias for the same field, so the
        # most specific input is not overwritten by a looser alias.
        if field in canonical and key != field:
            extra[str(raw_key)] = value
            continue
        canonical[field] = value
    return canonical, extra


def aliases_for(field: str) -> Iterable[str]:
    """Return the aliases accepted for a canonical *field*."""
    return FIELD_ALIASES.get(field, ())


# Fail fast at import time if the alias table drifts from the canonical set.
_UNKNOWN_FIELDS = set(FIELD_ALIASES) - set(CANONICAL_FIELDS)
if _UNKNOWN_FIELDS:  # pragma: no cover - guards against future edits
    raise RuntimeError(
        "FIELD_ALIASES references unknown canonical fields: %s"
        % sorted(_UNKNOWN_FIELDS)
    )
