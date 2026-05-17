from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.catalog import (
    LineageExportError,
    build_catalog,
    build_derived_index,
    build_lineage_edges,
    export_lineage,
    export_lineage_openlineage,
    export_lineage_prov,
    load_catalog_records,
    validate_lineage_export,
)
from src.cli.export_catalog_lineage import run_cli
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


def test_full_exports_are_deterministic_and_preserve_edge_types(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)

    first = export_lineage_openlineage(records, edges)
    second = export_lineage_openlineage(records, edges)
    prov = export_lineage_prov(records, edges)

    assert first == second
    assert first["record_count"] == 48
    assert first["edge_count"] == len(edges)
    assert prov["record_count"] == 48
    assert prov["edge_count"] == len(edges)
    assert {row["stratlake_edge_type"] for row in first["relationships"]} == {
        edge.edge_type for edge in edges
    }
    assert {row["stratlake_edge_type"] for row in prov["relations"]} == {
        edge.edge_type for edge in edges
    }
    assert {row["prov_relation"] for row in prov["relations"]} <= {"wasDerivedFrom"}
    assert "w3c_prov_conformant" not in prov
    assert "prov_conformance" not in prov
    validate_lineage_export(first)
    validate_lineage_export(prov)
    _assert_portable(first)
    _assert_portable(prov)


def test_selected_run_export_includes_direct_neighborhood(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)

    payload = export_lineage(records, edges, format="prov", selected_run_id="strategy_000")
    entity_by_run_id = {
        entity["attributes"].get("run_id"): entity["id"]
        for entity in payload["entities"]
        if entity["kind"] == "catalog_record"
    }
    selected_id = entity_by_run_id["strategy_000"]

    assert payload["selected_run_id"] == "strategy_000"
    assert "portfolio_000" in entity_by_run_id
    assert "portfolio_001" not in entity_by_run_id
    assert any(entity["kind"] == "artifact" for entity in payload["entities"])
    assert all(
        relation["source_id"] == selected_id or relation["target_id"] == selected_id
        for relation in payload["relations"]
    )


def test_direct_and_index_loaded_exports_match(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    direct_records = build_catalog(tmp_path, repo_root=tmp_path)
    indexed_records = load_catalog_records(tmp_path, repo_root=tmp_path, index_path=index_path, mode="index")

    direct_payload = export_lineage_openlineage(
        direct_records,
        build_lineage_edges(direct_records, repo_root=tmp_path),
    )
    indexed_payload = export_lineage_openlineage(
        indexed_records,
        build_lineage_edges(indexed_records, repo_root=tmp_path),
    )

    assert indexed_payload == direct_payload


def test_empty_and_sparse_exports_are_safe(tmp_path: Path) -> None:
    payload = export_lineage_openlineage([], [])
    assert payload == {
        "schema_version": 1,
        "format": "openlineage",
        "exporter_version": "m36_issue406_v1",
        "generated_marker": "deterministic",
        "source": "catalog_lineage",
        "selected_run_id": None,
        "record_count": 0,
        "edge_count": 0,
        "nodes": [],
        "relationships": [],
        "canonicality": {
            "schema_version": "canonicality.v1",
            "authority_kind": "artifact_tree",
            "authority_root": "artifacts",
            "authority_paths": [],
            "authority_fingerprint": payload["canonicality"]["authority_fingerprint"],
            "derived_class": "lineage_export",
            "rebuildable": True,
            "non_authoritative": True,
            "write_back_forbidden": True,
            "stale_if_source_changes": True,
            "resolver_hint": "reopen canonical manifests/registries before decision-sensitive use",
        },
        "canonicality_status": "canonicality_v1",
        "load_source": {
            "schema_version": "load_source.v1",
            "loaded_from": "lineage_export",
            "canonical_source": "artifacts",
            "non_authoritative": True,
            "resolver_hint": "reopen canonical manifests/registries before decision-sensitive use",
        },
    }

    sparse_path = tmp_path / "robustness" / "orphan" / "robustness_summary.json"
    sparse_path.parent.mkdir(parents=True)
    sparse_path.write_text('{"report_id":"orphan","source_run_ids":["missing_run"]}', encoding="utf-8")
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)
    payload = export_lineage_prov(records, edges)

    assert payload["record_count"] == 1
    assert payload["edge_count"] == 0


def test_unsupported_format_and_missing_selected_run_fail_clearly(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)

    with pytest.raises(LineageExportError, match="Unsupported lineage export format"):
        export_lineage(records, edges, format="csv")  # type: ignore[arg-type]
    with pytest.raises(LineageExportError, match="Selected run not found"):
        export_lineage(records, edges, selected_run_id="missing")


def test_cli_writes_file_and_prints_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build_catalog_scale_tree(tmp_path)
    output_path = tmp_path / "exports" / "lineage.json"

    written = run_cli(
        [
            "--artifacts-root",
            str(tmp_path),
            "--repo-root",
            str(tmp_path),
            "--format",
            "openlineage",
            "--output",
            str(output_path),
        ]
    )
    printed = run_cli(
        [
            "--artifacts-root",
            str(tmp_path),
            "--repo-root",
            str(tmp_path),
            "--format",
            "prov",
            "--selected-run-id",
            "strategy_000",
        ]
    )
    stdout_payload = json.loads(capsys.readouterr().out)

    assert json.loads(output_path.read_text(encoding="utf-8")) == written
    assert stdout_payload == printed
    assert printed["format"] == "prov"


def test_export_does_not_mutate_source_artifacts(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)
    before = snapshot_tree(tmp_path)

    export_lineage_openlineage(records, edges)
    export_lineage_prov(records, edges)

    assert snapshot_tree(tmp_path) == before


@pytest.mark.parametrize(
    "artifact_root",
    [
        "file:///tmp/artifacts/run",
        "https://example.com/artifacts/run",
        "C:/Users/example/artifacts/run",
        "/tmp/artifacts/run",
    ],
)
def test_validation_rejects_uri_like_or_absolute_paths(artifact_root: str) -> None:
    payload = _portable_payload()
    payload["nodes"][0]["facets"]["artifact_root"] = artifact_root

    with pytest.raises(LineageExportError):
        validate_lineage_export(payload)


def test_validation_accepts_repository_relative_posix_paths() -> None:
    payload = _portable_payload()
    payload["nodes"][0]["facets"]["artifact_root"] = "artifacts/strategies/demo"

    validate_lineage_export(payload)


def _assert_portable(payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    assert "file://" not in serialized
    assert "\\" not in serialized
    assert str(Path.cwd()) not in serialized


def _portable_payload() -> dict:
    return {
        "schema_version": 1,
        "format": "openlineage",
        "exporter_version": "m36_issue406_v1",
        "generated_marker": "deterministic",
        "source": "catalog_lineage",
        "selected_run_id": None,
        "record_count": 1,
        "edge_count": 0,
        "nodes": [
            {
                "id": "catalog:demo",
                "kind": "catalog_record",
                "namespace": "stratlake",
                "name": "demo",
                "facets": {
                    "catalog_id": "demo",
                    "run_id": "demo",
                    "run_type": "strategy",
                    "record_family": None,
                    "status": "completed",
                    "artifact_root": "artifacts/strategies/demo",
                },
            }
        ],
        "relationships": [],
    }
