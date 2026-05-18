"""Static contract helpers for derived evidence review packs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.catalog.canonicality import build_canonicality_envelope, portable_path, validate_portable_repository_path
from src.catalog.load_source import build_load_source

REVIEW_PACK_SCHEMA_VERSION = "review_pack.v1"
DEFAULT_REVIEW_PACK_ROOT = "artifacts/_derived/evidence_review"
REQUIRED_REVIEW_PACK_FILES = (
    "manifest.json",
    "review_request.json",
    "review_summary.json",
    "catalog_health_diagnostics.json",
    "validation.json",
    "selected_record.json",
    "related_records.json",
    "resolver_resolution.json",
    "evidence_index.json",
    "artifact_inventory.csv",
    "report.md",
)
OPTIONAL_REVIEW_PACK_FILES = (
    "selected_lineage.openlineage.json",
    "selected_lineage.prov.json",
    "report.html",
)


def review_pack_root(review_id: str) -> str:
    """Return the deterministic repository-relative output root for one pack."""

    normalized_review_id = portable_path(review_id)
    validate_portable_repository_path(normalized_review_id)
    if "/" in normalized_review_id:
        raise ValueError("Review IDs must be single portable path segments.")
    return f"{DEFAULT_REVIEW_PACK_ROOT}/{normalized_review_id}"


def build_review_pack_metadata(
    *,
    authority_paths: Iterable[str | Path],
    fingerprint_payload: Any | None = None,
) -> dict[str, Any]:
    """Build shared non-authoritative metadata for review-pack payloads."""

    metadata = build_canonicality_envelope(
        derived_class="review_pack",
        authority_paths=authority_paths,
        fingerprint_payload=fingerprint_payload,
    )
    metadata["load_source"] = build_load_source(loaded_from="review_pack")
    return metadata
