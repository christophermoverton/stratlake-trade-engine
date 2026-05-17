"""Deterministic load-source metadata for catalog read views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.catalog.canonicality import RESOLVER_HINT, portable_path, validate_portable_repository_path

LOAD_SOURCE_SCHEMA_VERSION = "load_source.v1"
LoadedFrom = Literal[
    "direct_scan",
    "derived_index",
    "lineage_export",
    "evidence_view",
    "workflow_view",
]
ResolvedMode = Literal["direct", "index"]


@dataclass(frozen=True)
class CatalogLoadResult:
    """Catalog records plus deterministic metadata about how they were loaded."""

    records: list[Any]
    load_source: dict[str, Any]


def build_load_source(
    *,
    loaded_from: LoadedFrom,
    requested_mode: str | None = None,
    resolved_mode: ResolvedMode | None = None,
    index_path: str | Path | None = None,
    index_validated: bool | None = None,
) -> dict[str, Any]:
    """Build portable deterministic source metadata for one read surface."""

    payload: dict[str, Any] = {
        "schema_version": LOAD_SOURCE_SCHEMA_VERSION,
        "loaded_from": loaded_from,
        "canonical_source": "artifacts",
        "non_authoritative": loaded_from != "direct_scan",
        "resolver_hint": RESOLVER_HINT,
    }
    if requested_mode is not None:
        payload["requested_mode"] = requested_mode
    if resolved_mode is not None:
        payload["resolved_mode"] = resolved_mode
    if index_path is not None:
        normalized_index_path = portable_path(index_path)
        validate_portable_repository_path(normalized_index_path)
        payload["index_path"] = normalized_index_path
    if index_validated is not None:
        payload["index_validated"] = index_validated
    return payload


def derive_view_load_source(
    source: dict[str, Any] | None,
    *,
    loaded_from: LoadedFrom,
) -> dict[str, Any]:
    """Convert catalog-load metadata into metadata for a derived view."""

    source = source or {}
    return build_load_source(
        loaded_from=loaded_from,
        requested_mode=source.get("requested_mode"),
        resolved_mode=source.get("resolved_mode"),
        index_path=source.get("index_path"),
        index_validated=source.get("index_validated"),
    )
