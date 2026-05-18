"""Deterministic canonicality envelopes for derived catalog surfaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Literal

from src.catalog.models import CatalogRecord

CANONICALITY_SCHEMA_VERSION = "canonicality.v1"
RESOLVER_HINT = "reopen canonical manifests/registries before decision-sensitive use"
DerivedClass = Literal["sqlite_read_model", "lineage_export", "evidence_view", "workflow_view"]


def build_canonicality_envelope(
    *,
    derived_class: DerivedClass,
    authority_root: str = "artifacts",
    authority_paths: Iterable[str | Path] = (),
    authority_fingerprint: str | None = None,
    fingerprint_payload: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Build one portable, deterministic non-authoritative derived envelope."""

    normalized_root = portable_path(authority_root)
    validate_portable_repository_path(normalized_root)
    normalized_paths: set[str] = set()
    for path in authority_paths:
        normalized_path = portable_path(path)
        if normalized_path:
            validate_portable_repository_path(normalized_path)
            normalized_paths.add(normalized_path)
    sorted_paths = sorted(normalized_paths)
    fingerprint = authority_fingerprint or _stable_fingerprint(
        {
            "authority_root": normalized_root,
            "authority_paths": sorted_paths,
            "payload": fingerprint_payload,
        }
    )
    return {
        "canonicality": {
            "schema_version": CANONICALITY_SCHEMA_VERSION,
            "authority_kind": "artifact_tree",
            "authority_root": normalized_root,
            "authority_paths": sorted_paths,
            "authority_fingerprint": fingerprint,
            "derived_class": derived_class,
            "rebuildable": True,
            "non_authoritative": True,
            "write_back_forbidden": True,
            "stale_if_source_changes": True,
            "resolver_hint": RESOLVER_HINT,
        }
    }


def canonical_authority_paths(records: Iterable[CatalogRecord]) -> list[str]:
    """Return sorted canonical source paths referenced by catalog records."""

    paths: set[str] = set()
    for record in records:
        for path in (
            record.source_registry_path,
            record.source_manifest_path,
            record.source_marker_path,
            *record.source_files,
        ):
            if path:
                normalized = portable_path(path)
                if normalized:
                    paths.add(normalized)
    return sorted(paths)


def canonicality_status(payload: dict[str, Any]) -> str:
    """Expose whether a derived payload carries the current envelope."""

    canonicality = payload.get("canonicality")
    if isinstance(canonicality, dict) and canonicality.get("schema_version") == CANONICALITY_SCHEMA_VERSION:
        return "canonicality_v1"
    return "legacy_no_envelope"


def portable_path(path: str | Path) -> str:
    """Normalize portable repository-relative text without machine-local separators."""

    text = Path(path).as_posix() if isinstance(path, Path) else str(path).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def validate_portable_repository_path(path: str) -> None:
    """Reject paths that are not portable repository-relative text."""

    parts = PurePosixPath(path).parts
    first_part = parts[0] if parts else ""
    invalid = (
        not path
        or "\\" in path
        or path.startswith("/")
        or "://" in path
        or any(part == ".." for part in parts)
        or (len(first_part) == 2 and first_part[1] == ":")
    )
    if invalid:
        raise ValueError(
            "Paths must be portable repository-relative paths."
        )


def _stable_fingerprint(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
