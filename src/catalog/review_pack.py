"""Static contract helpers for derived evidence review packs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

from src.catalog.canonicality import (
    build_canonicality_envelope,
    canonical_authority_paths,
    portable_path,
    validate_portable_repository_path,
)
from src.catalog.derived_index import IndexMode, load_catalog_records_with_source
from src.catalog.lineage import build_lineage_edges
from src.catalog.lineage_export import export_lineage
from src.catalog.lineage_fingerprints import stable_json_fingerprint
from src.catalog.load_source import build_load_source, derive_view_load_source
from src.catalog.models import CatalogRecord
from src.catalog.query import records_to_dicts, related_records
from src.catalog.resolver import resolve_canonical_record, resolve_canonical_sources
from src.catalog.workflows import resolve_workflow_roots

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
EvidenceReviewLineageFormat = Literal["openlineage", "prov", "both"]


class EvidenceReviewError(ValueError):
    """Raised when a selected-run evidence review cannot be built deterministically."""


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


def build_evidence_review_for_workflow(
    artifacts_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    index_path: str | Path | None = None,
    index_mode: IndexMode = "direct",
    selected_run_id: str | None = None,
    selected_catalog_id: str | None = None,
    review_id: str | None = None,
    resolve_related: bool = False,
    lineage_format: EvidenceReviewLineageFormat = "both",
) -> dict[str, Any]:
    """Build one deterministic resolver-backed review model without writing files."""

    if lineage_format not in {"openlineage", "prov", "both"}:
        raise EvidenceReviewError(f"Unsupported lineage format: {lineage_format}")
    if selected_run_id is None and selected_catalog_id is None:
        raise EvidenceReviewError("selected_run_id or selected_catalog_id is required")

    resolved_artifacts, resolved_repo = resolve_workflow_roots(artifacts_root, repo_root=repo_root)
    load_result = load_catalog_records_with_source(
        resolved_artifacts,
        repo_root=resolved_repo,
        index_path=index_path,
        mode=index_mode,
    )
    selected = _select_subject(
        load_result.records,
        selected_run_id=selected_run_id,
        selected_catalog_id=selected_catalog_id,
    )
    related = related_records(selected, load_result.records, repo_root=resolved_repo)
    selected_resolution = resolve_canonical_record(
        selected,
        artifacts_root=resolved_artifacts,
        repo_root=resolved_repo,
    )
    related_resolutions = (
        resolve_canonical_sources(
            related,
            artifacts_root=resolved_artifacts,
            repo_root=resolved_repo,
        )
        if resolve_related
        else []
    )
    lineage_edges = build_lineage_edges(load_result.records, repo_root=resolved_repo)
    lineage_exports = _build_lineage_exports(
        load_result.records,
        lineage_edges,
        selected_run_id=selected.run_id,
        lineage_format=lineage_format,
        load_source=derive_view_load_source(load_result.load_source, loaded_from="lineage_export"),
    )
    resolved_review_id = review_id or _deterministic_review_id(
        selected_run_id=selected_run_id,
        selected_catalog_id=selected_catalog_id,
        index_mode=index_mode,
        index_path=load_result.load_source.get("index_path"),
        lineage_format=lineage_format,
        resolve_related=resolve_related,
    )
    metadata = build_canonicality_envelope(
        derived_class="review_pack",
        authority_root=_authority_root(resolved_artifacts, resolved_repo),
        authority_paths=canonical_authority_paths(load_result.records),
        fingerprint_payload={
            "review_id": resolved_review_id,
            "selected_catalog_id": selected.catalog_id,
            "selected_run_id": selected.run_id,
        },
    )
    review_load_source = derive_view_load_source(load_result.load_source, loaded_from="review_pack")
    warnings = sorted(
        {
            *selected_resolution.warnings,
            *(warning for resolution in related_resolutions for warning in resolution.warnings),
        }
    )
    source_fingerprints = {
        selected.catalog_id: selected_resolution.source_fingerprint,
        **{
            resolution.record.catalog_id: resolution.source_fingerprint
            for resolution in related_resolutions
        },
    }
    return _json_safe(
        {
            "schema_version": REVIEW_PACK_SCHEMA_VERSION,
            "review_id": resolved_review_id,
            "review_root": review_pack_root(resolved_review_id),
            "review_request": {
                "schema_version": "review_request.v1",
                "review_id": resolved_review_id,
                "selected_run_id": selected.run_id,
                "selected_catalog_id": selected_catalog_id,
                "index_mode": index_mode,
                "resolve_related": resolve_related,
                "lineage_format": lineage_format,
            },
            "selected_record": selected.to_dict(),
            "related_records": records_to_dicts(related),
            "resolver_resolution": selected_resolution.to_dict(),
            "related_resolver_resolutions": [resolution.to_dict() for resolution in related_resolutions],
            "canonical_sources": list(selected_resolution.source_paths),
            "source_fingerprints": source_fingerprints,
            "warning_summary": {
                "count": len(warnings),
                "warnings": warnings,
                "missing_source_count": len(selected_resolution.missing_sources),
            },
            "load_source_summary": {
                "requested_mode": load_result.load_source.get("requested_mode"),
                "resolved_mode": load_result.load_source.get("resolved_mode"),
                "loaded_from": load_result.load_source.get("loaded_from"),
                "index_path": load_result.load_source.get("index_path"),
                "index_validated": load_result.load_source.get("index_validated"),
            },
            "lineage_summary": _lineage_summary(lineage_exports),
            "selected_lineage": lineage_exports,
            **metadata,
            "load_source": review_load_source,
        }
    )


def _select_subject(
    records: list[CatalogRecord],
    *,
    selected_run_id: str | None,
    selected_catalog_id: str | None,
) -> CatalogRecord:
    matches = [
        record
        for record in records
        if (selected_run_id is None or record.run_id == selected_run_id)
        and (selected_catalog_id is None or record.catalog_id == selected_catalog_id)
    ]
    if not matches:
        raise EvidenceReviewError("Selected catalog record not found.")
    if len(matches) > 1:
        raise EvidenceReviewError("Selected catalog record is ambiguous.")
    return matches[0]


def _build_lineage_exports(
    records: list[CatalogRecord],
    edges: list[Any],
    *,
    selected_run_id: str | None,
    lineage_format: EvidenceReviewLineageFormat,
    load_source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if selected_run_id is None:
        return {}
    formats = ("openlineage", "prov") if lineage_format == "both" else (lineage_format,)
    return {
        export_format: export_lineage(
            records,
            edges,
            format=export_format,
            selected_run_id=selected_run_id,
            load_source=load_source,
        )
        for export_format in formats
    }


def _lineage_summary(lineage_exports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not lineage_exports:
        return {
            "formats": [],
            "selected_record_count": 0,
            "selected_edge_count": 0,
            "related_run_ids": [],
        }
    first = lineage_exports[sorted(lineage_exports)[0]]
    nodes = first.get("nodes") or first.get("entities") or []
    related_run_ids = sorted(
        {
            node_metadata.get("run_id")
            for node in nodes
            if isinstance((node_metadata := node.get("facets") or node.get("attributes")), dict)
            and node_metadata.get("run_id")
            and node_metadata.get("run_id") != first.get("selected_run_id")
        }
    )
    return {
        "formats": sorted(lineage_exports),
        "selected_record_count": first.get("record_count", 0),
        "selected_edge_count": first.get("edge_count", 0),
        "related_run_ids": related_run_ids,
    }


def _deterministic_review_id(
    *,
    selected_run_id: str | None,
    selected_catalog_id: str | None,
    index_mode: str,
    index_path: str | Path | None,
    lineage_format: str,
    resolve_related: bool,
) -> str:
    normalized_index_path = portable_path(index_path) if index_path is not None else None
    if normalized_index_path is not None:
        validate_portable_repository_path(normalized_index_path)
    digest = stable_json_fingerprint(
        {
            "selected_run_id": selected_run_id,
            "selected_catalog_id": selected_catalog_id,
            "index_mode": index_mode,
            "index_path": normalized_index_path,
            "lineage_format": lineage_format,
            "resolve_related": resolve_related,
        }
    )
    return f"review_{digest[:16]}"


def _authority_root(artifacts_root: Path, repo_root: Path) -> str:
    if artifacts_root.is_relative_to(repo_root):
        return artifacts_root.relative_to(repo_root).as_posix()
    return portable_path(artifacts_root.name)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
