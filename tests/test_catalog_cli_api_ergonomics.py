from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.catalog import (
    DerivedIndexError,
    build_derived_index,
    build_evidence_view_for_workflow,
    build_lineage_export_for_workflow,
    load_catalog_for_workflow,
)
from src.cli.catalog_index import run_cli as run_catalog_index_cli
from src.cli.explore_catalog_evidence import run_cli as run_explore_cli
from src.cli.export_catalog_lineage import run_cli as run_export_cli
from src.cli.query_catalog import run_cli as run_query_cli
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


def test_shared_helpers_keep_direct_index_and_auto_records_equivalent(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    direct = load_catalog_for_workflow(".", repo_root=tmp_path)
    indexed = load_catalog_for_workflow(".", repo_root=tmp_path, index_path=index_path, index_mode="index")
    auto = load_catalog_for_workflow(".", repo_root=tmp_path, index_path=index_path, index_mode="auto")

    assert [record.to_dict() for record in direct] == [record.to_dict() for record in indexed]
    assert [record.to_dict() for record in direct] == [record.to_dict() for record in auto]


def test_shared_lineage_and_evidence_helpers_surface_lineage_metadata(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)

    export = build_lineage_export_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    view = build_evidence_view_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    node = next(
        node
        for node in export["nodes"]
        if node["kind"] == "catalog_record" and node["facets"]["run_id"] == "strategy_000"
    )

    assert export["selected_run_id"] == "strategy_000"
    assert node["facets"]["dataset_lineage"]["logical_dataset_id"] == "features_daily"
    assert any(row["run_id"] == "strategy_000" for row in view["catalog_records"])


def test_cli_build_validate_query_export_and_explore_share_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    export_path = tmp_path / "exports" / "lineage.json"
    explorer_path = tmp_path / "exports" / "evidence.json"

    build_payload = run_catalog_index_cli(
        ["build", "--artifacts-root", ".", "--repo-root", str(tmp_path), "--output", str(index_path)]
    )
    validate_payload = run_catalog_index_cli(
        ["validate", "--index", str(index_path), "--artifacts-root", ".", "--repo-root", str(tmp_path)]
    )
    query_payload = run_query_cli(
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
    export_payload = run_export_cli(
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
    explorer_payload = run_explore_cli(
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

    assert build_payload["record_count"] == 48
    assert validate_payload["valid"] is True
    assert [row["run_id"] for row in query_payload["records"]] == ["release_000", "release_001"]
    assert query_payload["load_source"]["resolved_mode"] == "index"
    assert export_payload["selected_run_id"] == "strategy_000"
    assert explorer_payload["total_matching_records"] >= 1
    assert export_path.read_text(encoding="utf-8").endswith("\n")
    assert explorer_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(export_path.read_text(encoding="utf-8")) == export_payload
    assert json.loads(explorer_path.read_text(encoding="utf-8")) == explorer_payload
    assert before == {
        path: payload
        for path, payload in snapshot_tree(tmp_path).items()
        if path not in {"catalog_index.sqlite", "exports/lineage.json", "exports/evidence.json"}
    }


def test_missing_and_stale_index_errors_remain_explicit(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    missing = tmp_path / "missing.sqlite"

    with pytest.raises(DerivedIndexError, match="not found"):
        load_catalog_for_workflow(".", repo_root=tmp_path, index_path=missing, index_mode="index")

    index_path = tmp_path / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    (tmp_path / "strategies" / "strategy_new").mkdir(parents=True)
    (tmp_path / "strategies" / "strategy_new" / "_SUCCESS.json").write_text(
        '{"run_id":"strategy_new","status":"completed"}',
        encoding="utf-8",
    )
    with pytest.raises(DerivedIndexError, match="stale"):
        load_catalog_for_workflow(".", repo_root=tmp_path, index_path=index_path, index_mode="auto")
