"""Deterministic standards-oriented exports over explicit catalog lineage.

The ``prov`` format is PROV-style local JSON, not a formal W3C PROV
conformance implementation. It uses a conservative ``wasDerivedFrom``-style
relation while keeping StratLake's original ``edge_type`` as the authoritative
semantic label.

Selected-run exports intentionally emit only the selected catalog record and
its direct one-hop neighborhood. The validator rejects URI-like strings as well
as absolute local paths so today's local artifact exports cannot accidentally
leak machine-specific locations; external URL metadata is not supported yet.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import PurePosixPath
from typing import Any, Literal

from src.catalog.canonicality import (
    build_canonicality_envelope,
    canonical_authority_paths,
    canonicality_status,
)
from src.catalog.models import CatalogRecord, LineageEdge

LINEAGE_EXPORT_SCHEMA_VERSION = 1
LINEAGE_EXPORTER_VERSION = "m36_issue406_v1"
LineageExportFormat = Literal["openlineage", "prov"]


class LineageExportError(ValueError):
    """Raised when a lineage export request cannot be satisfied safely."""


def export_lineage(
    records: Iterable[CatalogRecord],
    edges: Iterable[LineageEdge],
    *,
    format: LineageExportFormat = "openlineage",
    selected_run_id: str | None = None,
) -> dict[str, Any]:
    """Export explicit catalog lineage using one supported JSON format."""

    record_list = list(records)
    edge_list = list(edges)
    graph = _build_export_graph(record_list, edge_list, selected_run_id=selected_run_id)
    if format == "openlineage":
        payload = _render_openlineage(graph)
    elif format == "prov":
        payload = _render_prov(graph)
    else:
        raise LineageExportError(f"Unsupported lineage export format: {format}")
    payload.update(
        build_canonicality_envelope(
            derived_class="lineage_export",
            authority_paths=canonical_authority_paths(record_list),
            fingerprint_payload=[record.to_dict() for record in record_list],
        )
    )
    validate_lineage_export(payload)
    return payload


def export_lineage_openlineage(
    records: Iterable[CatalogRecord],
    edges: Iterable[LineageEdge],
    *,
    selected_run_id: str | None = None,
) -> dict[str, Any]:
    """Export explicit lineage as local OpenLineage-style JSON."""

    return export_lineage(records, edges, format="openlineage", selected_run_id=selected_run_id)


def export_lineage_prov(
    records: Iterable[CatalogRecord],
    edges: Iterable[LineageEdge],
    *,
    selected_run_id: str | None = None,
) -> dict[str, Any]:
    """Export explicit lineage as local PROV-style JSON, not formal PROV."""

    return export_lineage(records, edges, format="prov", selected_run_id=selected_run_id)


def validate_lineage_export(payload: dict[str, Any]) -> None:
    """Validate the closed, portable structure emitted by this module.

    URI-like strings are rejected deliberately. Current exports are local
    artifact lineage payloads, so even future-looking ``https://`` metadata is
    outside the supported contract until a separately scoped allowlist exists.
    """

    required = {
        "schema_version",
        "format",
        "exporter_version",
        "generated_marker",
        "source",
        "selected_run_id",
        "record_count",
        "edge_count",
    }
    missing = required.difference(payload)
    if missing:
        raise LineageExportError(f"Lineage export is missing required keys: {sorted(missing)}")
    if payload["schema_version"] != LINEAGE_EXPORT_SCHEMA_VERSION:
        raise LineageExportError("Lineage export schema is incompatible.")
    if payload["format"] == "openlineage":
        node_key = "nodes"
        edge_key = "relationships"
    elif payload["format"] == "prov":
        node_key = "entities"
        edge_key = "relations"
    else:
        raise LineageExportError(f"Unsupported lineage export format: {payload['format']}")

    nodes = payload.get(node_key)
    edges = payload.get(edge_key)
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise LineageExportError("Lineage export nodes and edges must be lists.")

    node_ids = [node.get("id") for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise LineageExportError("Lineage export node ids must be unique.")
    node_id_set = set(node_ids)
    for edge in edges:
        if edge.get("source_id") not in node_id_set or edge.get("target_id") not in node_id_set:
            raise LineageExportError("Lineage export relation references an unknown node.")
    if payload["record_count"] != sum(1 for node in nodes if node.get("kind") == "catalog_record"):
        raise LineageExportError("Lineage export record count does not match emitted records.")
    if payload["edge_count"] != len(edges):
        raise LineageExportError("Lineage export edge count does not match emitted relations.")

    serialized = json.dumps(payload, sort_keys=True)
    if "file://" in serialized or "\\" in serialized:
        raise LineageExportError("Lineage export contains a non-portable path.")
    if _contains_absolute_path(payload):
        raise LineageExportError("Lineage export contains an absolute path.")
    canonicality = payload.get("canonicality")
    if canonicality is not None:
        if not isinstance(canonicality, dict):
            raise LineageExportError("Lineage export canonicality envelope must be an object.")
        if canonicality.get("non_authoritative") is not True:
            raise LineageExportError("Lineage export canonicality must remain non-authoritative.")
        if canonicality.get("write_back_forbidden") is not True:
            raise LineageExportError("Lineage export canonicality must forbid write-back.")
    payload["canonicality_status"] = canonicality_status(payload)


def _build_export_graph(
    records: Iterable[CatalogRecord],
    edges: Iterable[LineageEdge],
    *,
    selected_run_id: str | None,
) -> dict[str, Any]:
    record_list = sorted(records, key=_record_sort_key)
    edge_list = sorted(edges, key=_edge_sort_key)
    by_run_id = {record.run_id: record for record in record_list if record.run_id}

    if selected_run_id is not None:
        selected = by_run_id.get(selected_run_id)
        if selected is None:
            raise LineageExportError(f"Selected run not found: {selected_run_id}")
        edge_list = [
            edge
            for edge in edge_list
            if edge.source_catalog_id == selected.catalog_id or edge.target_catalog_id == selected.catalog_id
        ]
        selected_catalog_ids = {selected.catalog_id}
        for edge in edge_list:
            if edge.source_catalog_id:
                selected_catalog_ids.add(edge.source_catalog_id)
            if edge.target_catalog_id:
                selected_catalog_ids.add(edge.target_catalog_id)
        record_list = [record for record in record_list if record.catalog_id in selected_catalog_ids]
    nodes = [_record_node(record) for record in record_list]
    node_ids = {node["id"] for node in nodes}
    relationships: list[dict[str, Any]] = []
    for edge in edge_list:
        source_id = _catalog_node_id(edge.source_catalog_id)
        target_id = _catalog_node_id(edge.target_catalog_id)
        if edge.edge_type == "manifest_declares_artifact":
            artifact_id = _artifact_node_id(edge)
            if source_id is None or artifact_id is None or source_id not in node_ids:
                continue
            if artifact_id not in node_ids:
                nodes.append(_artifact_node(edge, artifact_id))
                node_ids.add(artifact_id)
            target_id = artifact_id
        if source_id is None or target_id is None:
            continue
        if source_id not in node_ids or target_id not in node_ids:
            continue
        relationships.append(_relationship(edge, source_id, target_id))

    nodes = sorted(nodes, key=lambda node: node["id"])
    relationships = sorted(relationships, key=lambda relation: relation["id"])
    return {
        "selected_run_id": selected_run_id,
        "nodes": nodes,
        "relationships": relationships,
    }


def _render_openlineage(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": LINEAGE_EXPORT_SCHEMA_VERSION,
        "format": "openlineage",
        "exporter_version": LINEAGE_EXPORTER_VERSION,
        "generated_marker": "deterministic",
        "source": "catalog_lineage",
        "selected_run_id": graph["selected_run_id"],
        "record_count": sum(1 for node in graph["nodes"] if node["kind"] == "catalog_record"),
        "edge_count": len(graph["relationships"]),
        "nodes": [
            {
                "id": node["id"],
                "kind": node["kind"],
                "namespace": "stratlake",
                "name": node["name"],
                "facets": node["metadata"],
            }
            for node in graph["nodes"]
        ],
        "relationships": [
            {
                "id": relation["id"],
                "source_id": relation["source_id"],
                "target_id": relation["target_id"],
                "relationship_type": relation["edge_type"],
                "stratlake_edge_type": relation["edge_type"],
                "facets": relation["metadata"],
            }
            for relation in graph["relationships"]
        ],
    }


def _render_prov(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": LINEAGE_EXPORT_SCHEMA_VERSION,
        "format": "prov",
        "exporter_version": LINEAGE_EXPORTER_VERSION,
        "generated_marker": "deterministic",
        "source": "catalog_lineage",
        "selected_run_id": graph["selected_run_id"],
        "record_count": sum(1 for node in graph["nodes"] if node["kind"] == "catalog_record"),
        "edge_count": len(graph["relationships"]),
        "entities": [
            {
                "id": node["id"],
                "kind": node["kind"],
                "prov_type": "entity",
                "label": node["name"],
                "attributes": node["metadata"],
            }
            for node in graph["nodes"]
        ],
        "relations": [
            {
                "id": relation["id"],
                "source_id": relation["source_id"],
                "target_id": relation["target_id"],
                "prov_relation": "wasDerivedFrom",
                "stratlake_edge_type": relation["edge_type"],
                "attributes": relation["metadata"],
            }
            for relation in graph["relationships"]
        ],
        "agents": [
            {
                "id": "agent:stratlake_lineage_exporter",
                "kind": "software_agent",
                "label": "StratLake lineage exporter",
            }
        ],
    }


def _record_node(record: CatalogRecord) -> dict[str, Any]:
    metadata = {
        "catalog_id": record.catalog_id,
        "run_id": record.run_id,
        "run_type": record.run_type,
        "record_family": record.record_family,
        "status": record.status,
        "artifact_root": _portable_text(record.artifact_root),
    }
    for key in ("dataset_lineage", "feature_lineage"):
        value = record.metadata.get(key)
        if isinstance(value, dict):
            metadata[key] = _portable_json(value)
    return {
        "id": _catalog_node_id(record.catalog_id),
        "kind": "catalog_record",
        "name": record.run_id or record.catalog_id,
        "metadata": metadata,
    }


def _artifact_node(edge: LineageEdge, artifact_id: str) -> dict[str, Any]:
    metadata = edge.metadata
    return {
        "id": artifact_id,
        "kind": "artifact",
        "name": str(metadata.get("relative_path") or metadata.get("artifact_id") or artifact_id),
        "metadata": {
            "artifact_id": metadata.get("artifact_id"),
            "artifact_type": metadata.get("artifact_type"),
            "artifact_path": _portable_text(metadata.get("artifact_path")),
            "relative_path": _portable_text(metadata.get("relative_path")),
        },
    }


def _relationship(edge: LineageEdge, source_id: str, target_id: str) -> dict[str, Any]:
    return {
        "id": f"edge:{edge.edge_id}",
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge.edge_type,
        "metadata": {
            "edge_id": edge.edge_id,
            "relationship_source": edge.relationship_source,
            "relationship_path": _portable_text(edge.relationship_path),
            "source_catalog_id": edge.source_catalog_id,
            "target_catalog_id": edge.target_catalog_id,
            "source_run_id": edge.source_run_id,
            "target_run_id": edge.target_run_id,
            "metadata": _portable_json(edge.metadata),
        },
    }


def _catalog_node_id(catalog_id: str | None) -> str | None:
    return f"catalog:{catalog_id}" if catalog_id else None


def _artifact_node_id(edge: LineageEdge) -> str | None:
    artifact_id = edge.metadata.get("artifact_id")
    if artifact_id is None:
        return None
    return f"artifact:{artifact_id}"


def _record_sort_key(record: CatalogRecord) -> tuple[str, str, str, str]:
    return (record.run_type, record.run_id or "", record.catalog_id, record.artifact_root)


def _edge_sort_key(edge: LineageEdge) -> tuple[str, str, str, str]:
    return (
        edge.edge_type,
        edge.source_run_id or "",
        edge.target_run_id or "",
        edge.edge_id,
    )


def _portable_json(value: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value, sort_keys=True, default=str))
    return _portable_value(normalized)


def _portable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _portable_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, str):
        return _portable_text(value)
    return value


def _portable_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace("\\", "/")


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        text = value.replace("\\", "/")
        if text.startswith("/") or "://" in text:
            return True
        parts = PurePosixPath(text).parts
        return bool(parts and len(parts[0]) == 2 and parts[0][1] == ":")
    return False
