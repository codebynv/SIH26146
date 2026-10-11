"""XML parser (SIH26146 - Step 2)."""

from __future__ import annotations

import re
from typing import Any, Iterator, List
from xml.etree import ElementTree as ET

from backend.app.ingestion.parsers.base import IngestionError, RawRecord, decode_text

__all__ = ["parse_xml", "local_name"]

_DOCTYPE_RE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


def local_name(tag: str) -> str:
    """Strip any XML namespace from *tag*."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _element_value(element: ET.Element) -> Any:
    """Convert an element into a scalar, a list, or a mapping.

    An element with no children yields its stripped text.  An element whose
    children all share one tag yields a list of those children, which is how
    ``<input_addresses><address>a</address></input_addresses>`` becomes
    ``["a"]``.  Anything else becomes a mapping.
    """
    children = list(element)
    if not children:
        return (element.text or "").strip()

    child_tags = {local_name(child.tag) for child in children}
    if len(child_tags) == 1:
        tag = next(iter(child_tags))
        return [
            _element_value(child) for child in children if local_name(child.tag) == tag
        ]

    result: dict[str, Any] = {}
    for child in children:
        tag = local_name(child.tag)
        value = _element_value(child)
        if tag in result:
            existing = result[tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[tag] = [existing, value]
        else:
            result[tag] = value
    return result


def _unwrap_single_child(value: Any) -> Any:
    """Collapse a single-item wrapper so ``{'address': ['a']}`` reads as ``['a']``.

    XML has no native list type, so exporters wrap repeated values in an
    element.  This unwraps the wrapper when it is unambiguous.  A mapping with
    more than one key is left intact because it genuinely holds several fields.
    """
    if isinstance(value, dict) and len(value) == 1:
        (only,) = value.values()
        return only
    return value


def _merge_sibling_values(existing: Any, incoming: Any) -> Any:
    """Combine two occurrences of the same tag without losing either.

    This mirrors the convention :func:`_element_value` already applies to
    repeated *nested* tags: a repeated tag becomes a list.  Lists concatenate
    rather than nest, so two ``<input_addresses>`` wrapper elements yield
    ``["a", "b"]`` and not ``[["a"], ["b"]]``.

    Overwriting instead would be silent data loss, which is unacceptable in a
    forensic pipeline: ``<script_type>P2WPKH</script_type>`` followed by
    ``<script_type>P2TR</script_type>`` would otherwise appear to have said
    only ``P2TR``.  A repeated value that cannot map onto a canonical scalar
    field is reported as a validation error with both values present, rather
    than being quietly resolved to one of them.
    """
    if isinstance(existing, list):
        if isinstance(incoming, list):
            return [*existing, *incoming]
        return [*existing, incoming]
    if isinstance(incoming, list):
        return [existing, *incoming]
    return [existing, incoming]


def _record_fields(element: ET.Element) -> dict[str, Any]:
    """Reduce one record element to a mapping, merging repeated sibling tags."""
    data: dict[str, Any] = {}
    for child in element:
        tag = local_name(child.tag)
        value = _unwrap_single_child(_element_value(child))
        if tag in data:
            data[tag] = _merge_sibling_values(data[tag], value)
        else:
            data[tag] = value
    return data


def _find_record_elements(root: ET.Element) -> List[ET.Element]:
    """Locate the repeated elements that represent individual records."""
    children = list(root)
    if not children:
        return []

    if len({local_name(child.tag) for child in children}) == 1:
        return children

    # A wrapper such as <dataset><data>...</data><data>...</data></dataset>.
    for child in children:
        grandchildren = list(child)
        if len(grandchildren) > 1 and len(
            {local_name(grand.tag) for grand in grandchildren}
        ) == 1:
            return grandchildren

    # Otherwise the root itself is a single record.
    return [root]


def parse_xml(content: bytes | str) -> Iterator[RawRecord]:
    """Parse XML into raw records.

    Accepts both ``<record>...</record>`` repetitions and a single record at
    the document root.  Namespaces are stripped from element names.  Repeated
    sibling tags are merged into a list rather than overwritten, so no sibling
    value is ever silently dropped.  Documents declaring a DOCTYPE or ENTITY
    are refused outright, which blocks entity expansion ("billion laughs") and
    external-entity attacks without needing a third-party parser.
    """
    text = decode_text(content)
    if not text.strip():
        return

    if _DOCTYPE_RE.search(text):
        raise IngestionError(
            "XML documents declaring a DOCTYPE or ENTITY are refused to avoid "
            "entity-expansion and external-entity attacks",
            code="UNSAFE_XML",
        )

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise IngestionError(
            "malformed XML: %s" % exc, code="MALFORMED_XML"
        ) from exc

    for index, element in enumerate(_find_record_elements(root)):
        data = _record_fields(element)
        if not data:
            continue
        yield RawRecord(
            data=data,
            locator="<%s>[%d]" % (local_name(element.tag), index),
        )
