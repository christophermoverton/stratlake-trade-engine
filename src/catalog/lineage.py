"""Read-only lineage extraction over M29 catalog records.

The functions in this module derive in-memory LineageEdge objects from
CatalogRecord metadata and existing artifact files. They do not write, modify,
repair, register, or execute any workflow.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from src.catalog.indexer import build_artifact_records, load_json_file
from src.catalog.models import CatalogRecord, LineageEdge

_PORTFOLIO_FIELDS = ("component_run_ids", "components")
_COMPARISON_FIELDS = ("member_run_ids", "comparison_members", "run_ids", "inputs")
_BENCHMARK_FIELDS = ("member_run_ids", "child_run_ids", "scenario_run_ids", "run_ids")
_VALIDATION_FIELDS = ("referenced_run_ids", "validation_target_run_ids", "run_ids")
_PIPELINE_FIELDS = ("wrapped_run_id", "child_run_id", "stage_run_ids")
_EVIDENCE_RUN_FIELDS = (
    "source_run_ids",
    "source_run_id",
    "source_run_references",
    "referenced_run_ids",
    "validation_target_run_ids",
    "validated_run_ids",
    "run_ids",
)
_EVIDENCE_ARTIFACT_FIELDS = (
    "source_artifacts",
    "source_artifact_refs",
    "source_artifact_references",
    "upstream_artifacts",
)
_RELEASE_VALIDATION_BUNDLE_FIELDS = (
    "source_validation_bundle_run_ids",
    "source_validation_bundle_run_id",
    "validation_bundle_run_ids",
    "validation_bundle_run_id",
    "milestone_validation_bundle_run_ids",
    "milestone_validation_bundle_run_id",
)
_CAMPAIGN_EVIDENCE_FIELDS = ("campaign_id", "campaign_run_id", "source_campaign_id")
_SCENARIO_EVIDENCE_FIELDS = ("scenario_id", "scenario_run_id", "source_scenario_id")
_CAMPAIGN_PARENT_RUN_FIELDS = ("campaign_parent_run_id", "parent_run_id")
_CAMPAIGN_PARENT_CATALOG_FIELDS = ("campaign_parent_catalog_id", "parent_catalog_id")
_SCENARIO_PARENT_RUN_FIELDS = ("scenario_parent_run_id", "parent_scenario_run_id")
_SCENARIO_PARENT_CATALOG_FIELDS = ("scenario_parent_catalog_id", "parent_scenario_catalog_id")
_JSON_SOURCE_FILENAMES = {
    "checkpoint.json",
    "scenario_catalog.json",
    "manifest.json",
    "summary.json",
    "robustness_summary.json",
    "promotion_governance_summary.json",
    "consistency_validation.json",
    "release_validation.json",
    "release_validation_summary.json",
    "deterministic_rerun.json",
    "cross_layer_validation_report.json",
}
_CSV_SOURCE_FILENAMES = {
    "promotion_outcome_matrix.csv",
}


def build_lineage_edges(
    records: Iterable[CatalogRecord],
    *,
    repo_root: str | Path | None = None,
) -> list[LineageEdge]:
    """Return deterministic lineage edges derived from existing catalog records."""
    record_list = list(records)
    run_lookup = build_run_lookup(record_list)
    catalog_lookup = build_catalog_lookup(record_list)
    resolved_repo_root = Path(repo_root).resolve() if repo_root is not None else None

    edges: dict[str, LineageEdge] = {}
    for record in sorted(record_list, key=lambda r: (r.run_type, r.run_id or "", r.catalog_id)):
        for edge in lineage_from_record(
            record,
            run_lookup,
            catalog_lookup=catalog_lookup,
            repo_root=resolved_repo_root,
        ):
            edges.setdefault(edge.edge_id, edge)

    return sorted(
        edges.values(),
        key=lambda edge: (
            edge.edge_type,
            edge.source_run_id or "",
            edge.target_run_id or "",
            edge.relationship_source,
            edge.relationship_path or "",
        ),
    )


def build_run_lookup(records: Iterable[CatalogRecord]) -> dict[str, CatalogRecord]:
    """Build a deterministic run_id -> CatalogRecord lookup."""
    lookup: dict[str, CatalogRecord] = {}
    for record in sorted(records, key=lambda r: (r.run_id or "", r.catalog_id, r.artifact_root)):
        if record.run_id and record.run_id not in lookup:
            lookup[record.run_id] = record
    return lookup


def build_catalog_lookup(records: Iterable[CatalogRecord]) -> dict[str, CatalogRecord]:
    """Build a catalog_id -> CatalogRecord lookup."""
    return {record.catalog_id: record for record in records}


def lineage_from_record(
    record: CatalogRecord,
    run_lookup: dict[str, CatalogRecord],
    *,
    catalog_lookup: dict[str, CatalogRecord] | None = None,
    repo_root: str | Path | None = None,
) -> list[LineageEdge]:
    """Derive all resolvable lineage edges declared by one record."""
    catalog_lookup = catalog_lookup or {}
    resolved_repo_root = Path(repo_root).resolve() if repo_root is not None else None
    contexts = _relationship_contexts(record, repo_root=resolved_repo_root)
    edges: list[LineageEdge] = []

    if record.run_id:
        if _is_portfolio_record(record):
            edges.extend(_component_edges(record, run_lookup, contexts))
        if _is_comparison_record(record):
            edges.extend(_member_edges(record, run_lookup, contexts, "comparison_member", _COMPARISON_FIELDS))
        if _is_benchmark_record(record):
            edges.extend(_member_edges(record, run_lookup, contexts, "benchmark_member", _BENCHMARK_FIELDS))
        if _is_validation_record(record):
            edges.extend(_validation_edges(record, run_lookup, contexts))
        if _is_pipeline_record(record):
            edges.extend(_pipeline_edges(record, run_lookup, contexts))
        edges.extend(_campaign_child_edges(record, run_lookup, catalog_lookup, contexts))
        edges.extend(_scenario_child_edges(record, run_lookup, catalog_lookup, contexts))
        edges.extend(_evidence_edges(record, run_lookup, catalog_lookup, contexts))
        if resolved_repo_root is not None:
            edges.extend(_manifest_artifact_edges(record, repo_root=resolved_repo_root))

    deduped: dict[str, LineageEdge] = {}
    for edge in _dedupe_logical_edges(edges):
        deduped.setdefault(edge.edge_id, edge)
    return list(deduped.values())


def make_lineage_edge(
    *,
    edge_type: str,
    source: CatalogRecord | None,
    target: CatalogRecord | None,
    relationship_source: str,
    relationship_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LineageEdge:
    """Create a LineageEdge with the stable M29 edge id contract."""
    source_catalog_id = source.catalog_id if source is not None else None
    target_catalog_id = target.catalog_id if target is not None else None
    source_run_id = source.run_id if source is not None else None
    target_run_id = target.run_id if target is not None else None
    edge_id = _make_edge_id(
        edge_type=edge_type,
        source_catalog_id=source_catalog_id,
        target_catalog_id=target_catalog_id,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        relationship_source=relationship_source,
        relationship_path=relationship_path,
    )
    return LineageEdge(
        edge_id=edge_id,
        source_catalog_id=source_catalog_id,
        target_catalog_id=target_catalog_id,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        edge_type=edge_type,
        relationship_source=relationship_source,
        relationship_path=relationship_path,
        metadata=_json_like(metadata or {}),
    )


def _component_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    contexts: list[tuple[str, str | None, Mapping[str, Any]]],
) -> list[LineageEdge]:
    edges: list[LineageEdge] = []
    for source_name, source_path, payload in contexts:
        for field in _PORTFOLIO_FIELDS:
            for run_id in _extract_run_ids(payload.get(field)):
                source = run_lookup.get(run_id)
                if source is None:
                    continue
                edges.append(
                    make_lineage_edge(
                        edge_type="portfolio_component",
                        source=source,
                        target=record,
                        relationship_source=_field_source(source_name, field),
                        relationship_path=source_path,
                        metadata={"referenced_run_id": run_id},
                    )
                )
    return edges


def _member_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    contexts: list[tuple[str, str | None, Mapping[str, Any]]],
    edge_type: str,
    fields: tuple[str, ...],
) -> list[LineageEdge]:
    edges: list[LineageEdge] = []
    for source_name, source_path, payload in contexts:
        for field in fields:
            for run_id in _extract_run_ids(payload.get(field)):
                source = run_lookup.get(run_id)
                if source is None:
                    continue
                edges.append(
                    make_lineage_edge(
                        edge_type=edge_type,
                        source=source,
                        target=record,
                        relationship_source=_field_source(source_name, field),
                        relationship_path=source_path,
                        metadata={"referenced_run_id": run_id},
                    )
                )
    return edges


def _validation_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    contexts: list[tuple[str, str | None, Mapping[str, Any]]],
) -> list[LineageEdge]:
    edges: list[LineageEdge] = []
    for source_name, source_path, payload in contexts:
        for field in _VALIDATION_FIELDS:
            for run_id in _extract_run_ids(payload.get(field)):
                source = run_lookup.get(run_id)
                if source is None:
                    continue
                edges.append(
                    make_lineage_edge(
                        edge_type="validation_references_run",
                        source=source,
                        target=record,
                        relationship_source=_field_source(source_name, field),
                        relationship_path=source_path,
                        metadata={"referenced_run_id": run_id},
                    )
                )
    return edges


def _pipeline_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    contexts: list[tuple[str, str | None, Mapping[str, Any]]],
) -> list[LineageEdge]:
    edges: list[LineageEdge] = []
    for source_name, source_path, payload in contexts:
        for field in _PIPELINE_FIELDS:
            for run_id in _extract_run_ids(payload.get(field)):
                target = run_lookup.get(run_id)
                if target is None:
                    continue
                edges.append(
                    make_lineage_edge(
                        edge_type="pipeline_wraps_execution",
                        source=record,
                        target=target,
                        relationship_source=_field_source(source_name, field),
                        relationship_path=source_path,
                        metadata={"referenced_run_id": run_id},
                    )
                )
    return edges


def _campaign_child_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    catalog_lookup: Mapping[str, CatalogRecord],
    contexts: list[tuple[str, str | None, Mapping[str, Any]]],
) -> list[LineageEdge]:
    edges: list[LineageEdge] = []
    for source_name, source_path, payload in contexts:
        parent_records = _resolve_parent_records(
            payload,
            run_lookup=run_lookup,
            catalog_lookup=catalog_lookup,
            run_fields=_CAMPAIGN_PARENT_RUN_FIELDS,
            catalog_fields=_CAMPAIGN_PARENT_CATALOG_FIELDS,
        )
        campaign_id = _string_or_none(payload.get("campaign_id"))
        if (
            not parent_records
            and campaign_id
            and campaign_id in run_lookup
            and _is_campaign_parent_record(run_lookup[campaign_id])
        ):
            parent_records.append((run_lookup[campaign_id], "campaign_id", campaign_id))

        for parent, field, raw_value in parent_records:
            edges.append(
                make_lineage_edge(
                    edge_type="campaign_child",
                    source=parent,
                    target=record,
                    relationship_source=_field_source(source_name, field),
                    relationship_path=source_path,
                    metadata={"referenced_parent": raw_value},
                )
            )
    return edges


def _scenario_child_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    catalog_lookup: Mapping[str, CatalogRecord],
    contexts: list[tuple[str, str | None, Mapping[str, Any]]],
) -> list[LineageEdge]:
    edges: list[LineageEdge] = []
    for source_name, source_path, payload in contexts:
        parent_records = _resolve_parent_records(
            payload,
            run_lookup=run_lookup,
            catalog_lookup=catalog_lookup,
            run_fields=_SCENARIO_PARENT_RUN_FIELDS,
            catalog_fields=_SCENARIO_PARENT_CATALOG_FIELDS,
        )
        for parent, field, raw_value in parent_records:
            edges.append(
                make_lineage_edge(
                    edge_type="scenario_child",
                    source=parent,
                    target=record,
                    relationship_source=_field_source(source_name, field),
                    relationship_path=source_path,
                    metadata={"referenced_parent": raw_value},
                )
            )
    return edges


def _evidence_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    catalog_lookup: Mapping[str, CatalogRecord],
    contexts: list[tuple[str, str | None, Mapping[str, Any]]],
) -> list[LineageEdge]:
    edge_type = _evidence_edge_type(record)
    if edge_type is None:
        return []

    edges: list[LineageEdge] = []
    seen_references: set[tuple[str, str, str | None, str]] = set()
    for source_name, source_path, payload in contexts:
        for field in _EVIDENCE_RUN_FIELDS:
            for run_id in _extract_run_ids(payload.get(field)):
                source = run_lookup.get(run_id)
                if source is None or source.catalog_id == record.catalog_id:
                    continue
                key = (edge_type, source.catalog_id, source_path, field)
                if key in seen_references:
                    continue
                seen_references.add(key)
                edges.append(
                    make_lineage_edge(
                        edge_type=edge_type,
                        source=source,
                        target=record,
                        relationship_source=_field_source(source_name, field),
                        relationship_path=source_path,
                        metadata={"referenced_run_id": run_id},
                    )
                )

        for field in _EVIDENCE_ARTIFACT_FIELDS:
            for artifact_ref in _extract_artifact_references(payload.get(field)):
                source = _resolve_artifact_reference(artifact_ref, run_lookup)
                if source is None or source.catalog_id == record.catalog_id:
                    continue
                ref_text = _artifact_reference_text(artifact_ref)
                key = (edge_type, source.catalog_id, source_path, field)
                if key in seen_references:
                    continue
                seen_references.add(key)
                edges.append(
                    make_lineage_edge(
                        edge_type=edge_type,
                        source=source,
                        target=record,
                        relationship_source=_field_source(source_name, field),
                        relationship_path=source_path,
                        metadata={"artifact_reference": ref_text},
                    )
                )

        edges.extend(
            _release_validation_bundle_edges(
                record,
                run_lookup,
                source_name=source_name,
                source_path=source_path,
                payload=payload,
            )
        )
        edges.extend(
            _campaign_scenario_evidence_edges(
                record,
                run_lookup,
                catalog_lookup,
                source_name=source_name,
                source_path=source_path,
                payload=payload,
            )
        )
    return edges


def _release_validation_bundle_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    *,
    source_name: str,
    source_path: str | None,
    payload: Mapping[str, Any],
) -> list[LineageEdge]:
    if _record_family(record) != "release_validation_artifact":
        return []
    edges: list[LineageEdge] = []
    for field in _RELEASE_VALIDATION_BUNDLE_FIELDS:
        for run_id in _extract_run_ids(payload.get(field)):
            source = run_lookup.get(run_id)
            if source is None or source.catalog_id == record.catalog_id:
                continue
            edges.append(
                make_lineage_edge(
                    edge_type="validation_bundle_to_release_validation",
                    source=source,
                    target=record,
                    relationship_source=_field_source(source_name, field),
                    relationship_path=source_path,
                    metadata={"referenced_run_id": run_id},
                )
            )
    return edges


def _campaign_scenario_evidence_edges(
    record: CatalogRecord,
    run_lookup: Mapping[str, CatalogRecord],
    catalog_lookup: Mapping[str, CatalogRecord],
    *,
    source_name: str,
    source_path: str | None,
    payload: Mapping[str, Any],
) -> list[LineageEdge]:
    if _evidence_edge_type(record) is None:
        return []
    edges: list[LineageEdge] = []
    for field in _CAMPAIGN_EVIDENCE_FIELDS:
        for referenced in _extract_string_values(payload.get(field)):
            source = _resolve_campaign_or_scenario_reference(referenced, run_lookup, catalog_lookup)
            if source is None or source.catalog_id == record.catalog_id:
                continue
            edges.append(
                make_lineage_edge(
                    edge_type="campaign_to_evidence_bundle",
                    source=source,
                    target=record,
                    relationship_source=_field_source(source_name, field),
                    relationship_path=source_path,
                    metadata={"referenced_campaign": referenced},
                )
            )
    for field in _SCENARIO_EVIDENCE_FIELDS:
        for referenced in _extract_string_values(payload.get(field)):
            source = _resolve_campaign_or_scenario_reference(referenced, run_lookup, catalog_lookup)
            if source is None or source.catalog_id == record.catalog_id:
                continue
            edges.append(
                make_lineage_edge(
                    edge_type="scenario_to_evidence_bundle",
                    source=source,
                    target=record,
                    relationship_source=_field_source(source_name, field),
                    relationship_path=source_path,
                    metadata={"referenced_scenario": referenced},
                )
            )
    return edges


def _resolve_parent_records(
    payload: Mapping[str, Any],
    *,
    run_lookup: Mapping[str, CatalogRecord],
    catalog_lookup: Mapping[str, CatalogRecord],
    run_fields: tuple[str, ...],
    catalog_fields: tuple[str, ...],
) -> list[tuple[CatalogRecord, str, str]]:
    parent_records: list[tuple[CatalogRecord, str, str]] = []
    for field in run_fields:
        parent_run_id = _string_or_none(payload.get(field))
        if parent_run_id and parent_run_id in run_lookup:
            parent_records.append((run_lookup[parent_run_id], field, parent_run_id))
    for field in catalog_fields:
        parent_catalog_id = _string_or_none(payload.get(field))
        if parent_catalog_id and parent_catalog_id in catalog_lookup:
            parent_records.append((catalog_lookup[parent_catalog_id], field, parent_catalog_id))
    return parent_records


def _manifest_artifact_edges(record: CatalogRecord, *, repo_root: Path) -> list[LineageEdge]:
    edges: list[LineageEdge] = []
    for artifact in build_artifact_records(record, repo_root=repo_root):
        if not artifact.declared_in_manifest:
            continue
        relationship_path = _artifact_relationship_path(record.source_manifest_path, artifact.relative_path)
        edge_id = _make_edge_id(
            edge_type="manifest_declares_artifact",
            source_catalog_id=record.catalog_id,
            target_catalog_id=None,
            source_run_id=record.run_id,
            target_run_id=None,
            relationship_source="manifest.json",
            relationship_path=relationship_path,
        )
        edges.append(
            LineageEdge(
                edge_id=edge_id,
                source_catalog_id=record.catalog_id,
                target_catalog_id=None,
                source_run_id=record.run_id,
                target_run_id=None,
                edge_type="manifest_declares_artifact",
                relationship_source="manifest.json",
                relationship_path=relationship_path,
                metadata={
                    "artifact_id": artifact.artifact_id,
                    "artifact_path": artifact.path,
                    "artifact_type": artifact.artifact_type,
                    "relative_path": artifact.relative_path,
                },
            )
        )
    return edges


def _artifact_relationship_path(manifest_path: str | None, relative_path: str) -> str | None:
    if manifest_path is None:
        return relative_path
    return f"{manifest_path}#{relative_path}"


def _relationship_contexts(
    record: CatalogRecord,
    *,
    repo_root: Path | None,
) -> list[tuple[str, str | None, Mapping[str, Any]]]:
    contexts: list[tuple[str, str | None, Mapping[str, Any]]] = []
    contexts.append(("record", None, _record_base_payload(record)))
    if record.metadata:
        contexts.append(("metadata", None, record.metadata))
        for key, value in sorted(record.metadata.items(), key=lambda item: str(item[0])):
            if isinstance(value, Mapping):
                contexts.append((str(key), str(key), value))

    if repo_root is not None:
        candidate_paths = set(record.source_files)
        if record.source_manifest_path:
            candidate_paths.add(record.source_manifest_path)
        for rel_path in sorted(candidate_paths):
            path = repo_root / rel_path
            if path.name in _JSON_SOURCE_FILENAMES:
                payload = load_json_file(path)
                if payload is not None:
                    contexts.append((path.name, rel_path, payload))
            elif path.name in _CSV_SOURCE_FILENAMES:
                rows = _load_csv_rows(path)
                if rows:
                    contexts.append((path.name, rel_path, {"rows": rows, "run_ids": _row_run_ids(rows)}))
    return contexts


def _record_base_payload(record: CatalogRecord) -> dict[str, Any]:
    payload = {
        "run_id": record.run_id,
        "catalog_id": record.catalog_id,
        "campaign_id": record.campaign_id,
        "scenario_id": record.scenario_id,
    }
    if isinstance(record.metadata.get("metadata"), Mapping):
        payload.update(record.metadata["metadata"])
    return payload


def _extract_run_ids(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        values.append(value)
    elif isinstance(value, Mapping):
        for key in ("run_id", "child_run_id", "wrapped_run_id", "parent_run_id", "campaign_run_id"):
            run_id = _string_or_none(value.get(key))
            if run_id:
                values.append(run_id)
        for nested_key in ("run_ids", "member_run_ids", "child_run_ids", "scenario_run_ids", "inputs"):
            values.extend(_extract_run_ids(value.get(nested_key)))
    elif isinstance(value, Iterable) and not isinstance(value, bytes):
        for item in value:
            values.extend(_extract_run_ids(item))
    return sorted(set(run_id for run_id in (_string_or_none(v) for v in values) if run_id))


def _extract_string_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        values.append(value)
    elif isinstance(value, Mapping):
        for key in (
            "run_id",
            "catalog_id",
            "campaign_id",
            "scenario_id",
            "campaign_run_id",
            "scenario_run_id",
        ):
            item = _string_or_none(value.get(key))
            if item:
                values.append(item)
    elif isinstance(value, Iterable) and not isinstance(value, bytes):
        for item in value:
            values.extend(_extract_string_values(item))
    return sorted(set(item for item in (_string_or_none(value) for value in values) if item))


def _extract_artifact_references(value: Any) -> list[Any]:
    references: list[Any] = []
    if isinstance(value, str):
        references.append(value)
    elif isinstance(value, Mapping):
        references.append(value)
        for key in ("artifacts", "files", "source_artifacts", "upstream_artifacts"):
            references.extend(_extract_artifact_references(value.get(key)))
    elif isinstance(value, Iterable) and not isinstance(value, bytes):
        for item in value:
            references.extend(_extract_artifact_references(item))
    return references


def _resolve_artifact_reference(
    artifact_ref: Any,
    run_lookup: Mapping[str, CatalogRecord],
) -> CatalogRecord | None:
    for run_id in _extract_run_ids(artifact_ref):
        source = run_lookup.get(run_id)
        if source is not None:
            return source
    path = _artifact_reference_path(artifact_ref)
    if path is None:
        return None
    normalized_path = path.strip().replace("\\", "/")
    for source in run_lookup.values():
        root = source.artifact_root.rstrip("/")
        if normalized_path == root or normalized_path.startswith(f"{root}/"):
            return source
    return None


def _artifact_reference_path(artifact_ref: Any) -> str | None:
    if isinstance(artifact_ref, str):
        return artifact_ref
    if isinstance(artifact_ref, Mapping):
        for key in ("path", "relative_path", "artifact_path", "source_artifact_path"):
            value = _string_or_none(artifact_ref.get(key))
            if value:
                return value
    return None


def _artifact_reference_text(artifact_ref: Any) -> str:
    path = _artifact_reference_path(artifact_ref)
    if path:
        return path.replace("\\", "/")
    if isinstance(artifact_ref, Mapping):
        return json.dumps(_json_like(artifact_ref), sort_keys=True)
    return str(artifact_ref)


def _resolve_campaign_or_scenario_reference(
    reference: str,
    run_lookup: Mapping[str, CatalogRecord],
    catalog_lookup: Mapping[str, CatalogRecord],
) -> CatalogRecord | None:
    if reference in run_lookup:
        return run_lookup[reference]
    if reference in catalog_lookup:
        return catalog_lookup[reference]
    for record in run_lookup.values():
        if record.campaign_id == reference or record.scenario_id == reference:
            return record
    return None


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def _row_run_ids(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    run_ids: list[str] = []
    for row in rows:
        run_id = _string_or_none(row.get("run_id"))
        if run_id:
            run_ids.append(run_id)
    return sorted(set(run_ids))


def _record_family(record: CatalogRecord) -> str | None:
    return record.record_family or (
        record.run_type
        if record.run_type
        in {
            "robustness_bundle",
            "governance_bundle",
            "milestone_validation_bundle",
            "release_validation_artifact",
        }
        else None
    )


def _evidence_edge_type(record: CatalogRecord) -> str | None:
    return {
        "robustness_bundle": "run_to_robustness_evidence",
        "governance_bundle": "run_to_governance_evidence",
        "milestone_validation_bundle": "run_to_validation_bundle",
        "release_validation_artifact": "run_to_release_validation",
    }.get(_record_family(record) or "")


def _make_edge_id(
    *,
    edge_type: str,
    source_catalog_id: str | None,
    target_catalog_id: str | None,
    source_run_id: str | None,
    target_run_id: str | None,
    relationship_source: str,
    relationship_path: str | None,
) -> str:
    raw = "|".join(
        (
            edge_type,
            source_catalog_id or "",
            target_catalog_id or "",
            source_run_id or "",
            target_run_id or "",
            relationship_source,
            relationship_path or "",
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _field_source(source_name: str, field: str) -> str:
    if source_name in {"record", "metadata"}:
        return field
    return f"{source_name}:{field}"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_like(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str))


def _dedupe_logical_edges(edges: Iterable[LineageEdge]) -> list[LineageEdge]:
    deduped: dict[
        tuple[str, str | None, str | None, str | None, str | None, str | None],
        LineageEdge,
    ] = {}
    for edge in sorted(
        edges,
        key=lambda item: (item.relationship_source, item.relationship_path or "", item.edge_id),
    ):
        key = (
            edge.edge_type,
            edge.source_catalog_id,
            edge.target_catalog_id,
            edge.source_run_id,
            edge.target_run_id,
            edge.relationship_path if edge.edge_type == "manifest_declares_artifact" else None,
        )
        deduped.setdefault(key, edge)
    return list(deduped.values())


def _is_portfolio_record(record: CatalogRecord) -> bool:
    return record.run_type == "portfolio"


def _is_comparison_record(record: CatalogRecord) -> bool:
    return "comparison" in record.run_type


def _is_benchmark_record(record: CatalogRecord) -> bool:
    return "benchmark" in record.run_type


def _is_validation_record(record: CatalogRecord) -> bool:
    return record.run_type in {"qa", "validation", "milestone_validation"} or "validation" in record.run_type


def _is_pipeline_record(record: CatalogRecord) -> bool:
    return record.run_type == "pipeline"


def _is_campaign_parent_record(record: CatalogRecord) -> bool:
    return record.run_type == "campaign" or "campaign" in record.run_type
