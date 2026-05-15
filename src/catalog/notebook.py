"""Notebook-friendly helpers for M35 evidence catalog exploration.

These functions are thin wrappers over the shared catalog query, lineage, and
explorer APIs. They do not execute workflows, mutate artifacts, or duplicate
CLI-specific logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.catalog.explorer import (
    build_evidence_explorer_view,
    render_evidence_json,
    render_evidence_markdown,
    render_evidence_table,
)
from src.catalog.lineage import build_lineage_edges
from src.catalog.models import CatalogRecord, LineageEdge
from src.catalog.query import CatalogQuery, query_catalog, records_to_rows

EVIDENCE_EDGE_TYPES: frozenset[str] = frozenset(
    {
        "run_to_robustness_evidence",
        "run_to_governance_evidence",
        "run_to_validation_bundle",
        "run_to_release_validation",
        "validation_bundle_to_release_validation",
        "campaign_to_evidence_bundle",
        "scenario_to_evidence_bundle",
    }
)


def find_robustness_evidence(
    records: Iterable[CatalogRecord],
    *,
    robustness_status: str | None = None,
    wfe_status: str | None = None,
) -> list[dict[str, object]]:
    """Return table-friendly robustness evidence rows."""

    return _record_rows(
        query_catalog(
            records,
            CatalogQuery(
                record_family="robustness_bundle",
                robustness_status=robustness_status,
                wfe_status=wfe_status,
            ),
        )
    )


def find_governance_evidence(
    records: Iterable[CatalogRecord],
    *,
    governance_status: str | None = None,
    promotion_review_status: str | None = None,
) -> list[dict[str, object]]:
    """Return table-friendly governance evidence rows."""

    return _record_rows(
        query_catalog(
            records,
            CatalogQuery(
                record_family="governance_bundle",
                governance_status=governance_status,
                promotion_review_status=promotion_review_status,
            ),
        )
    )


def find_validation_evidence(
    records: Iterable[CatalogRecord],
    *,
    validation_readiness_present: bool = True,
) -> list[dict[str, object]]:
    """Return table-friendly milestone validation evidence rows."""

    return _record_rows(
        query_catalog(
            records,
            CatalogQuery(
                record_family="milestone_validation_bundle",
                validation_readiness_present=validation_readiness_present,
            ),
        )
    )


def find_release_evidence(
    records: Iterable[CatalogRecord],
    *,
    release_validation_present: bool = True,
) -> list[dict[str, object]]:
    """Return table-friendly release-validation evidence rows."""

    return _record_rows(
        query_catalog(
            records,
            CatalogQuery(
                record_family="release_validation_artifact",
                release_validation_present=release_validation_present,
            ),
        )
    )


def evidence_lineage_rows(
    records: Iterable[CatalogRecord],
    *,
    run_id: str | None = None,
    catalog_id: str | None = None,
    repo_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Return deterministic evidence lineage rows for notebook display."""

    record_list = list(records)
    edges = [
        edge
        for edge in build_lineage_edges(record_list, repo_root=repo_root)
        if edge.edge_type in EVIDENCE_EDGE_TYPES
    ]
    if run_id is not None:
        edges = [edge for edge in edges if edge.source_run_id == run_id or edge.target_run_id == run_id]
    if catalog_id is not None:
        edges = [
            edge
            for edge in edges
            if edge.source_catalog_id == catalog_id or edge.target_catalog_id == catalog_id
        ]
    return [_edge_row(edge) for edge in edges]


def evidence_for_run(
    records: Iterable[CatalogRecord],
    run_id: str,
    *,
    repo_root: str | Path | None = None,
    include_lineage: bool = True,
) -> dict[str, object]:
    """Return a deterministic single-run evidence view."""

    record_list = list(records)
    view = build_evidence_explorer_view(
        record_list,
        selected_run_id=run_id,
        include_lineage=include_lineage,
        repo_root=repo_root,
    )
    return _jsonable(
        {
            "run_id": run_id,
            "records": view["catalog_records"],
            "evidence_status": view["evidence_status"],
            "evidence_summary": view["evidence_summary"],
            "lineage_edges": view["lineage_edges"],
        }
    )


def summarize_evidence_for_run(
    records: Iterable[CatalogRecord],
    run_id: str,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Return compact counts and evidence family names for a selected run."""

    view = evidence_for_run(records, run_id, repo_root=repo_root, include_lineage=True)
    evidence_rows = [
        row
        for row in view["records"]
        if isinstance(row, dict) and row.get("record_family")
    ]
    edge_rows = view["lineage_edges"] if isinstance(view["lineage_edges"], list) else []
    return _jsonable(
        {
            "run_id": run_id,
            "related_evidence_count": len(evidence_rows),
            "evidence_lineage_count": len(edge_rows),
            "record_families": sorted(
                {
                    str(row["record_family"])
                    for row in evidence_rows
                    if row.get("record_family")
                }
            ),
        }
    )


def build_notebook_evidence_view(
    records: Iterable[CatalogRecord],
    *,
    query: CatalogQuery | None = None,
    run_id: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Build the same explorer view used by the local CLI, as a plain dict."""

    return _jsonable(
        build_evidence_explorer_view(
            list(records),
            query=query,
            selected_run_id=run_id,
            include_lineage=True,
            repo_root=repo_root,
        )
    )


def render_notebook_markdown(
    records: Iterable[CatalogRecord],
    *,
    query: CatalogQuery | None = None,
    run_id: str | None = None,
    repo_root: str | Path | None = None,
) -> str:
    """Render notebook-friendly Markdown using the shared explorer renderer."""

    return render_evidence_markdown(
        build_evidence_explorer_view(
            list(records),
            query=query,
            selected_run_id=run_id,
            include_lineage=True,
            repo_root=repo_root,
        )
    )


def render_notebook_json(
    records: Iterable[CatalogRecord],
    *,
    query: CatalogQuery | None = None,
    run_id: str | None = None,
    repo_root: str | Path | None = None,
) -> str:
    """Render notebook-friendly JSON using the shared explorer renderer."""

    return render_evidence_json(
        build_evidence_explorer_view(
            list(records),
            query=query,
            selected_run_id=run_id,
            include_lineage=True,
            repo_root=repo_root,
        )
    )


def render_notebook_table(
    records: Iterable[CatalogRecord],
    *,
    query: CatalogQuery | None = None,
    run_id: str | None = None,
    repo_root: str | Path | None = None,
) -> str:
    """Render notebook-friendly table text using the shared explorer renderer."""

    return render_evidence_table(
        build_evidence_explorer_view(
            list(records),
            query=query,
            selected_run_id=run_id,
            include_lineage=True,
            repo_root=repo_root,
        )
    )


def _record_rows(records: Iterable[CatalogRecord]) -> list[dict[str, object]]:
    return _jsonable(records_to_rows(records))


def _edge_row(edge: LineageEdge) -> dict[str, object]:
    return _jsonable(
        {
            "edge_type": edge.edge_type,
            "source_run_id": edge.source_run_id,
            "target_run_id": edge.target_run_id,
            "relationship_source": edge.relationship_source,
            "relationship_path": edge.relationship_path,
            "metadata": edge.metadata,
        }
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value
