"""Thin shared workflow helpers for catalog CLI, notebooks, and wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.catalog.canonicality import build_canonicality_envelope, canonical_authority_paths
from src.catalog.derived_index import IndexMode, load_catalog_records
from src.catalog.explorer import build_evidence_explorer_view
from src.catalog.lineage import build_lineage_edges
from src.catalog.lineage_export import LineageExportFormat, export_lineage
from src.catalog.models import CatalogRecord
from src.catalog.query import CatalogQuery


def resolve_workflow_roots(
    artifacts_root: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve repository and artifact roots consistently for workflow callers."""

    resolved_repo = Path(repo_root or ".")
    resolved_artifacts = Path(artifacts_root)
    if not resolved_artifacts.is_absolute():
        resolved_artifacts = resolved_repo / resolved_artifacts
    return resolved_artifacts, resolved_repo


def load_catalog_for_workflow(
    artifacts_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    index_path: str | Path | None = None,
    index_mode: IndexMode = "direct",
) -> list[CatalogRecord]:
    """Load records for shared direct/index/auto workflow use."""

    resolved_artifacts, resolved_repo = resolve_workflow_roots(artifacts_root, repo_root=repo_root)
    return load_catalog_records(
        resolved_artifacts,
        repo_root=resolved_repo,
        index_path=index_path,
        mode=index_mode,
    )


def build_lineage_export_for_workflow(
    artifacts_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    index_path: str | Path | None = None,
    index_mode: IndexMode = "direct",
    export_format: LineageExportFormat = "openlineage",
    selected_run_id: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic lineage export through shared catalog APIs."""

    resolved_artifacts, resolved_repo = resolve_workflow_roots(artifacts_root, repo_root=repo_root)
    records = load_catalog_for_workflow(
        resolved_artifacts,
        repo_root=resolved_repo,
        index_path=index_path,
        index_mode=index_mode,
    )
    return export_lineage(
        records,
        build_lineage_edges(records, repo_root=resolved_repo),
        format=export_format,
        selected_run_id=selected_run_id,
    )


def build_evidence_view_for_workflow(
    artifacts_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    index_path: str | Path | None = None,
    index_mode: IndexMode = "direct",
    query: CatalogQuery | None = None,
    selected_run_id: str | None = None,
    selected_catalog_id: str | None = None,
    include_lineage: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build the shared evidence explorer view from one catalog load path."""

    resolved_artifacts, resolved_repo = resolve_workflow_roots(artifacts_root, repo_root=repo_root)
    records = load_catalog_for_workflow(
        resolved_artifacts,
        repo_root=resolved_repo,
        index_path=index_path,
        index_mode=index_mode,
    )
    view = build_evidence_explorer_view(
        records,
        query=query,
        selected_run_id=selected_run_id,
        selected_catalog_id=selected_catalog_id,
        include_lineage=include_lineage,
        repo_root=resolved_repo,
        limit=limit,
    )
    view.update(
        build_canonicality_envelope(
            derived_class="evidence_view",
            authority_root=resolved_artifacts.relative_to(resolved_repo).as_posix()
            if resolved_artifacts.is_relative_to(resolved_repo)
            else resolved_artifacts.name,
            authority_paths=canonical_authority_paths(records),
            fingerprint_payload=[record.to_dict() for record in records],
        )
    )
    return view
