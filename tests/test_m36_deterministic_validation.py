from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from src.catalog import (
    DerivedIndexError,
    build_catalog,
    build_derived_index,
    build_evidence_view_for_workflow,
    build_lineage_edges,
    build_lineage_export_for_workflow,
    export_lineage_openlineage,
    export_lineage_prov,
    load_catalog_for_workflow,
    load_catalog_records,
    stable_json_fingerprint,
    validate_derived_index,
)
from src.cli.catalog_index import run_cli as run_catalog_index_cli
from src.cli.explore_catalog_evidence import run_cli as run_explore_cli
from src.cli.export_catalog_lineage import run_cli as run_export_cli
from src.cli.query_catalog import run_cli as run_query_cli
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = REPO_ROOT / "docs" / "m36_release_validation_checklist.md"
MILESTONE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "milestone_validation.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


def test_combined_m36_stack_is_equivalent_read_only_and_portable(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"

    direct = build_catalog(tmp_path, repo_root=tmp_path)
    metadata = build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    indexed = load_catalog_records(tmp_path, repo_root=tmp_path, index_path=index_path, mode="index")
    auto = load_catalog_records(tmp_path, repo_root=tmp_path, index_path=index_path, mode="auto")
    helper = load_catalog_for_workflow(".", repo_root=tmp_path, index_path=index_path, index_mode="auto")

    serialized_direct = [record.to_dict() for record in direct]
    assert [record.to_dict() for record in indexed] == serialized_direct
    assert [record.to_dict() for record in auto] == serialized_direct
    assert [record.to_dict() for record in helper] == serialized_direct
    assert [record.run_id for record in direct[:3]] == ["alpha_000", "alpha_001", "alpha_002"]

    view = build_evidence_view_for_workflow(
        ".",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    openlineage = build_lineage_export_for_workflow(
        ".",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    summary = _summary(direct, openlineage, repo_root=tmp_path)

    assert summary == {
        "schema_version": 1,
        "record_count": 48,
        "lineage_edge_count": 100,
        "openlineage_node_count": len(openlineage["nodes"]),
        "dataset_lineage_records": 4,
        "feature_lineage_records": 4,
        "index_record_count": 48,
    }
    assert metadata["is_derived"] is True
    assert metadata["canonical_source"] == "artifacts"
    assert metadata["record_count"] == 48
    assert metadata["record_family_counts"] == {
        "governance_bundle": 4,
        "milestone_validation_bundle": 3,
        "release_validation_artifact": 2,
        "robustness_bundle": 6,
    }
    assert before == {
        path: payload
        for path, payload in snapshot_tree(tmp_path).items()
        if path != "catalog_index.sqlite"
    }
    _assert_portable(metadata, serialized_direct, view, openlineage, tmp_path=tmp_path)


def test_index_rebuild_disposability_missing_stale_and_incompatible_failures(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    first = build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    index_path.unlink()
    second = build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    assert first == second
    assert snapshot_tree(tmp_path) == {**before, "catalog_index.sqlite": snapshot_tree(tmp_path)["catalog_index.sqlite"]}

    missing = tmp_path / "missing.sqlite"
    assert [record.to_dict() for record in load_catalog_records(tmp_path, repo_root=tmp_path, index_path=missing, mode="auto")] == [
        record.to_dict() for record in build_catalog(tmp_path, repo_root=tmp_path)
    ]
    with pytest.raises(DerivedIndexError, match="not found"):
        load_catalog_records(tmp_path, repo_root=tmp_path, index_path=missing, mode="index")

    (tmp_path / "strategies" / "strategy_new").mkdir(parents=True)
    (tmp_path / "strategies" / "strategy_new" / "_SUCCESS.json").write_text(
        '{"run_id":"strategy_new","status":"completed"}',
        encoding="utf-8",
    )
    with pytest.raises(DerivedIndexError, match="stale"):
        load_catalog_records(tmp_path, repo_root=tmp_path, index_path=index_path, mode="auto")

    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE metadata SET value_json = ? WHERE key = 'schema_version'",
            (json.dumps(999),),
        )
        connection.commit()
    with pytest.raises(DerivedIndexError, match="schema is incompatible"):
        validate_derived_index(index_path, artifacts_root=tmp_path, repo_root=tmp_path)


def test_lineage_exports_and_fingerprints_stay_deterministic_without_new_edges(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    direct = build_catalog(tmp_path, repo_root=tmp_path)
    indexed = load_catalog_records(tmp_path, repo_root=tmp_path, index_path=index_path, mode="index")
    direct_edges = build_lineage_edges(direct, repo_root=tmp_path)
    indexed_edges = build_lineage_edges(indexed, repo_root=tmp_path)

    direct_open = export_lineage_openlineage(direct, direct_edges)
    direct_prov = export_lineage_prov(direct, direct_edges)
    indexed_open = export_lineage_openlineage(indexed, indexed_edges)
    indexed_prov = export_lineage_prov(indexed, indexed_edges)
    selected = export_lineage_prov(direct, direct_edges, selected_run_id="strategy_000")

    assert direct_open == export_lineage_openlineage(direct, direct_edges)
    assert direct_prov == export_lineage_prov(direct, direct_edges)
    assert indexed_open == direct_open
    assert indexed_prov == direct_prov
    assert [edge.to_dict() for edge in indexed_edges] == [edge.to_dict() for edge in direct_edges]
    assert direct_open["edge_count"] == len(direct_open["relationships"]) == len(direct_edges)
    assert direct_prov["edge_count"] == len(direct_prov["relations"]) == len(direct_edges)
    assert {row["stratlake_edge_type"] for row in direct_open["relationships"]} == {
        edge.edge_type for edge in direct_edges
    }
    assert all(
        relation["source_id"] in {entity["id"] for entity in selected["entities"]}
        and relation["target_id"] in {entity["id"] for entity in selected["entities"]}
        for relation in selected["relations"]
    )
    assert all(
        relation["source_id"] == _entity_id(selected, "strategy_000")
        or relation["target_id"] == _entity_id(selected, "strategy_000")
        for relation in selected["relations"]
    )

    direct_by_run = {record.run_id: record for record in direct}
    indexed_by_run = {record.run_id: record for record in indexed}
    assert direct_by_run["strategy_000"].metadata["dataset_lineage"] == indexed_by_run["strategy_000"].metadata[
        "dataset_lineage"
    ]
    node = next(node for node in direct_open["nodes"] if node["kind"] == "catalog_record" and node["facets"]["run_id"] == "strategy_000")
    entity = next(
        entity
        for entity in direct_prov["entities"]
        if entity["kind"] == "catalog_record" and entity["attributes"]["run_id"] == "strategy_000"
    )
    assert node["facets"]["feature_lineage"] == entity["attributes"]["feature_lineage"]
    assert len(direct_edges) == 100
    assert direct_by_run["strategy_001"].metadata.get("dataset_lineage") is None
    assert stable_json_fingerprint({"b": 2, "a": 1}) == stable_json_fingerprint({"a": 1, "b": 2})
    _assert_portable(direct_open, direct_prov, selected, tmp_path=tmp_path)


def test_cli_api_parity_and_release_hardening_assumptions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    export_path = tmp_path / "exports" / "lineage.json"
    explorer_path = tmp_path / "exports" / "evidence.json"

    run_catalog_index_cli(["build", "--artifacts-root", ".", "--repo-root", str(tmp_path), "--output", str(index_path)])
    cli_query = run_query_cli(
        [
            "--artifacts-root",
            ".",
            "--repo-root",
            str(tmp_path),
            "--index",
            str(index_path),
            "--index-mode",
            "auto",
            "--record-family",
            "release_validation_artifact",
            "--format",
            "json",
        ]
    )
    cli_export = run_export_cli(
        [
            "--artifacts-root",
            ".",
            "--repo-root",
            str(tmp_path),
            "--index",
            str(index_path),
            "--index-mode",
            "auto",
            "--selected-run-id",
            "strategy_000",
            "--output",
            str(export_path),
        ]
    )
    cli_view = run_explore_cli(
        [
            "--artifacts-root",
            ".",
            "--repo-root",
            str(tmp_path),
            "--index",
            str(index_path),
            "--index-mode",
            "auto",
            "--run-id",
            "strategy_000",
            "--format",
            "json",
            "--output",
            str(explorer_path),
        ]
    )
    capsys.readouterr()

    api_export = build_lineage_export_for_workflow(
        ".",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    api_view = build_evidence_view_for_workflow(
        ".",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )

    assert [row["run_id"] for row in cli_query] == ["release_000", "release_001"]
    assert cli_export == api_export
    assert cli_view == api_view
    assert export_path.read_text(encoding="utf-8").endswith("\n")
    assert explorer_path.read_text(encoding="utf-8").endswith("\n")

    checklist = CHECKLIST.read_text(encoding="utf-8")
    milestone = MILESTONE_WORKFLOW.read_text(encoding="utf-8")
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert '- "feature/m*"' in milestone
    assert '- "v*"' in release
    for issue in ("405", "406", "407", "408", "409"):
        assert f"Issue #{issue}" in checklist


def _summary(records: list, openlineage: dict[str, object], *, repo_root: Path) -> dict[str, int]:
    return {
        "schema_version": 1,
        "record_count": len(records),
        "lineage_edge_count": len(build_lineage_edges(records, repo_root=repo_root)),
        "openlineage_node_count": len(openlineage["nodes"]),
        "dataset_lineage_records": sum(1 for record in records if "dataset_lineage" in record.metadata),
        "feature_lineage_records": sum(1 for record in records if "feature_lineage" in record.metadata),
        "index_record_count": len(records),
    }


def _entity_id(payload: dict[str, object], run_id: str) -> str:
    return next(
        entity["id"]
        for entity in payload["entities"]
        if entity["kind"] == "catalog_record" and entity["attributes"]["run_id"] == run_id
    )


def _assert_portable(*payloads: object, tmp_path: Path) -> None:
    serialized = "".join(json.dumps(payload, sort_keys=True) for payload in payloads)
    assert str(tmp_path) not in serialized
    assert "file://" not in serialized
    assert "\\" not in serialized
    assert "C:/" not in serialized
