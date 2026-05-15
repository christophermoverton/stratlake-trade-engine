from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from src.catalog import (
    CatalogQuery,
    DerivedIndexError,
    build_catalog,
    build_derived_index,
    load_catalog_records,
    query_catalog,
    validate_derived_index,
)
from src.cli.catalog_index import run_cli as run_catalog_index_cli
from src.cli.query_catalog import run_cli as run_query_catalog_cli
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


def test_build_validate_and_query_derived_index_equivalent_to_direct_scan(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)
    index_path = tmp_path / "catalog_index" / "catalog_index.sqlite"

    metadata = build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    validation = validate_derived_index(index_path, artifacts_root=tmp_path, repo_root=tmp_path)
    direct_records = build_catalog(tmp_path, repo_root=tmp_path)
    indexed_records = load_catalog_records(
        tmp_path,
        repo_root=tmp_path,
        index_path=index_path,
        mode="index",
    )

    assert metadata["is_derived"] is True
    assert metadata["canonical_source"] == "artifacts"
    assert metadata["source_artifact_root"] == "."
    assert metadata["repo_root"] == "."
    assert metadata["record_count"] == 48
    assert validation.metadata == metadata
    assert [record.to_dict() for record in indexed_records] == [record.to_dict() for record in direct_records]
    assert _query_ids(direct_records) == _query_ids(indexed_records)
    assert before == {
        path: payload
        for path, payload in snapshot_tree(tmp_path).items()
        if not path.startswith("catalog_index/")
    }


def test_index_can_be_deleted_and_rebuilt(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"

    first = build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    index_path.unlink()
    second = build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    assert first == second
    assert index_path.exists()


def test_missing_index_auto_falls_back_and_index_mode_fails(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    missing = tmp_path / "missing.sqlite"

    assert [record.to_dict() for record in load_catalog_records(tmp_path, repo_root=tmp_path, index_path=missing, mode="auto")] == [
        record.to_dict() for record in build_catalog(tmp_path, repo_root=tmp_path)
    ]
    with pytest.raises(DerivedIndexError, match="not found"):
        load_catalog_records(tmp_path, repo_root=tmp_path, index_path=missing, mode="index")


def test_stale_and_mismatched_indexes_fail_safely(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    (tmp_path / "strategies" / "strategy_new" / "_SUCCESS.json").parent.mkdir(parents=True)
    (tmp_path / "strategies" / "strategy_new" / "_SUCCESS.json").write_text(
        json.dumps({"run_id": "strategy_new", "status": "completed"}),
        encoding="utf-8",
    )
    with pytest.raises(DerivedIndexError, match="stale"):
        load_catalog_records(tmp_path, repo_root=tmp_path, index_path=index_path, mode="auto")

    other_root = tmp_path / "other"
    other_root.mkdir()
    with pytest.raises(DerivedIndexError, match="artifact root does not match"):
        validate_derived_index(
            index_path,
            artifacts_root=other_root,
            repo_root=tmp_path,
            check_source_fingerprint=False,
        )


def test_incompatible_schema_fails_with_rebuild_guidance(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE metadata SET value_json = ? WHERE key = 'schema_version'",
            (json.dumps(999),),
        )
        connection.commit()

    with pytest.raises(DerivedIndexError, match="schema is incompatible"):
        validate_derived_index(index_path, artifacts_root=tmp_path, repo_root=tmp_path)


def test_empty_and_sparse_roots_are_supported(tmp_path: Path) -> None:
    empty_index = tmp_path / "empty.sqlite"
    metadata = build_derived_index(tmp_path, empty_index, repo_root=tmp_path)
    assert metadata["record_count"] == 0
    assert validate_derived_index(empty_index, artifacts_root=tmp_path, repo_root=tmp_path).records == []

    sparse_root = tmp_path / "sparse"
    sparse_path = sparse_root / "robustness" / "sparse" / "robustness_summary.json"
    sparse_path.parent.mkdir(parents=True)
    sparse_path.write_text('{"report_id":"sparse"}', encoding="utf-8")
    sparse_index = tmp_path / "sparse.sqlite"
    build_derived_index(sparse_root, sparse_index, repo_root=sparse_root)
    records = load_catalog_records(sparse_root, repo_root=sparse_root, index_path=sparse_index, mode="index")
    assert [record.record_family for record in records] == ["robustness_bundle"]


def test_index_metadata_and_payload_paths_are_portable(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    metadata = build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    with sqlite3.connect(index_path) as connection:
        payloads = [
            row[0]
            for row in connection.execute("SELECT payload_json FROM catalog_records ORDER BY catalog_id")
        ]
    serialized = json.dumps(metadata, sort_keys=True) + "".join(payloads)

    assert str(tmp_path) not in serialized
    assert "file://" not in serialized
    assert "\\" not in serialized


def test_catalog_index_and_query_cli_support_index_workflow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"

    build_payload = run_catalog_index_cli(
        ["build", "--artifacts-root", str(tmp_path), "--repo-root", str(tmp_path), "--output", str(index_path)]
    )
    validate_payload = run_catalog_index_cli(
        ["validate", "--index", str(index_path), "--artifacts-root", str(tmp_path), "--repo-root", str(tmp_path)]
    )
    query_payload = run_query_catalog_cli(
        [
            "--artifacts-root",
            str(tmp_path),
            "--repo-root",
            str(tmp_path),
            "--index",
            str(index_path),
            "--index-mode",
            "index",
            "--record-family",
            "release_validation_artifact",
            "--format",
            "json",
        ]
    )
    capsys.readouterr()

    assert build_payload["record_count"] == 48
    assert validate_payload["valid"] is True
    assert [row["run_id"] for row in query_payload] == ["release_000", "release_001"]


def _query_ids(records: list) -> dict[str, list[str | None]]:
    return {
        "release": [
            record.run_id
            for record in query_catalog(records, CatalogQuery(release_validation_present=True))
        ],
        "robustness": [
            record.run_id
            for record in query_catalog(
                records,
                CatalogQuery(record_family="robustness_bundle", robustness_status="needs_review"),
            )
        ],
        "metric": [
            record.run_id
            for record in query_catalog(records, CatalogQuery(min_metric=("sharpe_ratio", 1.8)))
        ],
    }
