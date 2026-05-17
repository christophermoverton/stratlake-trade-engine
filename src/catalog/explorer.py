"""Local read-only evidence explorer for M35 catalog records.

The explorer renders deterministic review views from in-memory catalog records
and lineage edges. It does not write canonical artifacts, mutate source files,
start services, or create persistent storage.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.catalog.canonicality import build_canonicality_envelope, canonical_authority_paths
from src.catalog.lineage import build_lineage_edges
from src.catalog.models import CatalogRecord, LineageEdge
from src.catalog.query import CatalogQuery, query_catalog, records_to_rows

EXPLORER_SCHEMA_VERSION = 1

CATALOG_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_type",
    "status",
    "record_family",
    "artifact_root",
    "review_status",
    "promotion_status",
)

EVIDENCE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "robustness_status",
    "wfe_status",
    "sample_size_status",
    "trade_count_status",
    "sensitivity_status",
    "fragility_status",
    "multiple_testing_status",
    "temporal_validation_status",
    "governance_status",
    "promotion_review_status",
    "validation_readiness_present",
    "release_validation_present",
)

LINEAGE_COLUMNS: tuple[str, ...] = (
    "edge_type",
    "source_run_id",
    "target_run_id",
    "relationship_source",
    "relationship_path",
    "metadata",
)


def build_evidence_explorer_view(
    records: Iterable[CatalogRecord],
    *,
    query: CatalogQuery | None = None,
    selected_run_id: str | None = None,
    selected_catalog_id: str | None = None,
    include_lineage: bool = True,
    repo_root: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic local evidence explorer view.

    The input records are treated as the source of truth. The function returns
    a JSON-safe dictionary and never writes or mutates source artifacts.
    """

    all_records = list(records)
    selected_records = query_catalog(all_records, query)
    if selected_run_id or selected_catalog_id:
        selected = _find_selected_record(
            all_records,
            selected_run_id=selected_run_id,
            selected_catalog_id=selected_catalog_id,
        )
        selected_records = [selected] if selected is not None else []

    edges = build_lineage_edges(all_records, repo_root=repo_root) if include_lineage else []
    if selected_run_id or selected_catalog_id:
        selected_records = _expand_selected_records(selected_records, all_records, edges)

    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected_records = selected_records[:limit]

    record_ids = {record.catalog_id for record in selected_records}
    lineage_rows = [
        _edge_row(edge)
        for edge in edges
        if record_ids
        and (edge.source_catalog_id in record_ids or edge.target_catalog_id in record_ids)
    ]
    lineage_rows.sort(
        key=lambda row: (
            str(row["edge_type"]),
            str(row["source_run_id"] or ""),
            str(row["target_run_id"] or ""),
            str(row["relationship_source"]),
            str(row["relationship_path"] or ""),
        )
    )

    catalog_rows = [_select_columns(row, CATALOG_COLUMNS) for row in records_to_rows(selected_records)]
    evidence_rows = [_select_columns(row, EVIDENCE_COLUMNS) for row in records_to_rows(selected_records)]
    view = {
        "schema_version": EXPLORER_SCHEMA_VERSION,
        "title": "M35 Catalog Evidence Explorer",
        "total_matching_records": len(selected_records),
        "total_lineage_edges": len(lineage_rows),
        "catalog_records": catalog_rows,
        "evidence_status": evidence_rows,
        "evidence_summary": _evidence_summary(selected_records),
        "lineage_edges": lineage_rows,
    }
    view.update(
        build_canonicality_envelope(
            derived_class="evidence_view",
            authority_paths=canonical_authority_paths(all_records),
            fingerprint_payload=[record.to_dict() for record in all_records],
        )
    )
    return view


def render_evidence_json(view: dict[str, Any]) -> str:
    """Render a deterministic JSON explorer view."""

    return json.dumps(_jsonable(view), indent=2, sort_keys=True) + "\n"


def render_evidence_markdown(view: dict[str, Any]) -> str:
    """Render a deterministic Markdown explorer view."""

    lines = [
        "# M35 Catalog Evidence Explorer",
        "",
        f"- Schema Version: {view.get('schema_version')}",
        f"- Total Matching Records: {view.get('total_matching_records', 0)}",
        f"- Evidence Lineage Edges: {view.get('total_lineage_edges', 0)}",
        "",
        "## Catalog Records",
    ]
    catalog_rows = list(view.get("catalog_records", []))
    lines.extend(_markdown_table(catalog_rows, CATALOG_COLUMNS, empty_message="No matching records."))
    lines.extend(["", "## Evidence Status"])
    evidence_rows = list(view.get("evidence_status", []))
    lines.extend(_markdown_table(evidence_rows, EVIDENCE_COLUMNS, empty_message="No evidence records."))
    lines.extend(["", "## Evidence Summary"])
    lines.extend(_markdown_summary(view.get("evidence_summary", {})))
    lines.extend(["", "## Evidence Lineage"])
    lineage_rows = list(view.get("lineage_edges", []))
    lines.extend(_markdown_table(lineage_rows, LINEAGE_COLUMNS, empty_message="No evidence lineage found."))
    return "\n".join(lines).rstrip() + "\n"


def render_evidence_table(view: dict[str, Any]) -> str:
    """Render a stable text table explorer view."""

    rows: list[dict[str, Any]] = []
    for row in view.get("catalog_records", []):
        rows.append({"section": "catalog", **dict(row)})
    for row in view.get("evidence_status", []):
        rows.append({"section": "evidence", **dict(row)})
    for row in view.get("lineage_edges", []):
        rows.append({"section": "lineage", **dict(row)})
    columns = ("section", *CATALOG_COLUMNS, *EVIDENCE_COLUMNS[1:], *LINEAGE_COLUMNS)
    unique_columns = tuple(dict.fromkeys(columns))
    lines = ["\t".join(unique_columns)]
    if not rows:
        return lines[0] + "\n"
    for row in rows:
        lines.append("\t".join(_format_cell(row.get(column)) for column in unique_columns))
    return "\n".join(lines) + "\n"


def _find_selected_record(
    records: Iterable[CatalogRecord],
    *,
    selected_run_id: str | None,
    selected_catalog_id: str | None,
) -> CatalogRecord | None:
    for record in records:
        if selected_run_id is not None and record.run_id == selected_run_id:
            return record
        if selected_catalog_id is not None and record.catalog_id == selected_catalog_id:
            return record
    return None


def _expand_selected_records(
    selected_records: list[CatalogRecord],
    all_records: list[CatalogRecord],
    edges: list[LineageEdge],
) -> list[CatalogRecord]:
    if not selected_records:
        return []
    selected_ids = {record.catalog_id for record in selected_records}
    related_ids = set(selected_ids)
    for edge in edges:
        if edge.source_catalog_id in selected_ids and edge.target_catalog_id:
            related_ids.add(edge.target_catalog_id)
        if edge.target_catalog_id in selected_ids and edge.source_catalog_id:
            related_ids.add(edge.source_catalog_id)
    by_id = {record.catalog_id: record for record in all_records}
    return [by_id[catalog_id] for catalog_id in sorted(related_ids) if catalog_id in by_id]


def _edge_row(edge: LineageEdge) -> dict[str, Any]:
    return {
        "edge_type": edge.edge_type,
        "source_run_id": edge.source_run_id,
        "target_run_id": edge.target_run_id,
        "relationship_source": edge.relationship_source,
        "relationship_path": edge.relationship_path,
        "metadata": _jsonable(edge.metadata),
    }


def _evidence_summary(records: Iterable[CatalogRecord]) -> dict[str, Any]:
    record_list = list(records)
    return {
        "record_family_counts": _sorted_counter(Counter(record.record_family or "" for record in record_list)),
        "robustness_status_counts": _sorted_counter(Counter(record.robustness_status or "" for record in record_list)),
        "governance_status_counts": _sorted_counter(Counter(record.governance_status or "" for record in record_list)),
        "validation_readiness_present": sum(1 for record in record_list if record.validation_readiness_present),
        "release_validation_present": sum(1 for record in record_list if record.release_validation_present),
    }


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter) if key}


def _select_columns(row: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    return {column: row.get(column) for column in columns}


def _markdown_table(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    *,
    empty_message: str,
) -> list[str]:
    if not rows:
        return [empty_message]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown(_format_cell(row.get(column))) for column in columns) + " |")
    return lines


def _markdown_summary(summary: Any) -> list[str]:
    if not isinstance(summary, dict) or not summary:
        return ["No evidence summary."]
    lines: list[str] = []
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, dict):
            rendered = ", ".join(f"{item_key}: {value[item_key]}" for item_key in sorted(value)) or "none"
            lines.append(f"- {key}: {rendered}")
        else:
            lines.append(f"- {key}: {value}")
    return lines


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
