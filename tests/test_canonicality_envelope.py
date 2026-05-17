from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from src.catalog import (
    build_canonicality_envelope,
    build_catalog,
    build_derived_index,
    build_evidence_view_for_workflow,
    build_lineage_edges,
    canonicality_status,
    export_lineage_openlineage,
    export_lineage_prov,
    load_catalog_records,
    validate_derived_index,
    validate_lineage_export,
)
from tests.catalog_scale_fixtures import build_catalog_scale_tree


def test_canonicality_envelope_is_deterministic_portable_and_sorted() -> None:
    first = build_canonicality_envelope(
        derived_class="lineage_export",
        authority_paths=["artifacts\\zeta\\manifest.json", "./artifacts/alpha/manifest.json"],
        fingerprint_payload={"b": 2, "a": 1},
    )
    second = build_canonicality_envelope(
        derived_class="lineage_export",
        authority_paths=["artifacts/alpha/manifest.json", "artifacts/zeta/manifest.json"],
        fingerprint_payload={"a": 1, "b": 2},
    )

    assert first == second
    envelope = first["canonicality"]
    assert envelope["schema_version"] == "canonicality.v1"
    assert envelope["authority_kind"] == "artifact_tree"
    assert envelope["authority_root"] == "artifacts"
    assert envelope["authority_paths"] == [
        "artifacts/alpha/manifest.json",
        "artifacts/zeta/manifest.json",
    ]
    assert envelope["derived_class"] == "lineage_export"
    assert envelope["rebuildable"] is True
    assert envelope["non_authoritative"] is True
    assert envelope["write_back_forbidden"] is True
    assert envelope["stale_if_source_changes"] is True
    assert "\\" not in json.dumps(first, sort_keys=True)


def test_derived_index_metadata_includes_envelope_and_legacy_index_remains_readable(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    metadata = build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    assert metadata["canonicality"]["derived_class"] == "sqlite_read_model"
    assert metadata["canonicality"]["non_authoritative"] is True
    assert metadata["canonicality"]["write_back_forbidden"] is True

    with sqlite3.connect(index_path) as connection:
        connection.execute("DELETE FROM metadata WHERE key = 'canonicality'")
        connection.commit()

    validation = validate_derived_index(index_path, artifacts_root=tmp_path, repo_root=tmp_path)
    assert validation.metadata["canonicality_status"] == "legacy_no_envelope"
    assert len(validation.records) == 48


def test_lineage_exports_include_envelope_and_legacy_payloads_validate(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)

    openlineage = export_lineage_openlineage(records, edges)
    prov = export_lineage_prov(records, edges)

    for payload in (openlineage, prov):
        assert payload["canonicality"]["derived_class"] == "lineage_export"
        assert payload["canonicality"]["non_authoritative"] is True
        assert payload["canonicality"]["write_back_forbidden"] is True
        assert payload["canonicality_status"] == "canonicality_v1"

    legacy = dict(openlineage)
    legacy.pop("canonicality")
    legacy.pop("canonicality_status")
    validate_lineage_export(legacy)
    assert legacy["canonicality_status"] == "legacy_no_envelope"


def test_workflow_evidence_view_is_annotated_and_direct_scan_is_unchanged(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    direct = build_catalog(tmp_path, repo_root=tmp_path)
    indexed = load_catalog_records(tmp_path, repo_root=tmp_path, mode="direct")
    view = build_evidence_view_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")

    assert [record.to_dict() for record in indexed] == [record.to_dict() for record in direct]
    assert view["canonicality"]["derived_class"] == "evidence_view"
    assert view["canonicality"]["non_authoritative"] is True
    assert canonicality_status(view) == "canonicality_v1"
