from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.catalog import (
    build_catalog,
    build_dataset_lineage,
    build_derived_index,
    build_feature_lineage,
    build_lineage_edges,
    dataset_schema_fingerprint,
    export_lineage_openlineage,
    feature_columns_fingerprint,
    load_catalog_records,
    portable_dataset_path,
    stable_json_fingerprint,
)
from src.data.feature_metadata import build_feature_metadata_summary
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


def test_fingerprint_helpers_are_stable_and_order_independent(tmp_path: Path) -> None:
    assert stable_json_fingerprint({"b": 2, "a": 1}) == stable_json_fingerprint({"a": 1, "b": 2})
    assert feature_columns_fingerprint(["feature_b", "feature_a"]) == feature_columns_fingerprint(
        ["feature_a", "feature_b"]
    )
    assert dataset_schema_fingerprint({"feature_b": "float64", "feature_a": "float64"}) == (
        dataset_schema_fingerprint([("feature_a", "float64"), ("feature_b", "float64")])
    )
    assert portable_dataset_path(tmp_path / "data" / "curated", repo_root=tmp_path) == "data/curated"


def test_lineage_builders_emit_portable_deterministic_metadata(tmp_path: Path) -> None:
    dataset_lineage = build_dataset_lineage(
        logical_dataset_id="features_daily",
        dataset_role="feature_dataset",
        dataset_path=tmp_path / "data" / "curated" / "features_daily",
        dataset_contract_version="feature_dataset_v1",
        schema={"symbol": "object", "feature_ret_1d": "float64"},
        row_count=10,
        symbol_count=2,
        timeframe="1D",
        start="2026-01-01",
        end="2026-01-10",
        source_payload={"source_dataset": "bars_daily"},
        repo_root=tmp_path,
    )
    feature_lineage = build_feature_lineage(
        feature_group_names=["trend", "returns"],
        feature_columns=["feature_ret_1d", "feature_sma_20"],
        schema={"feature_ret_1d": "float64", "feature_sma_20": "float64"},
        feature_contract_version="feature_contract_v1",
        build_config={"dataset": "features_daily", "window": 20},
    )
    serialized = json.dumps({"dataset_lineage": dataset_lineage, "feature_lineage": feature_lineage}, sort_keys=True)

    assert dataset_lineage["dataset_path"] == "data/curated/features_daily"
    assert feature_lineage["feature_group_names"] == ["returns", "trend"]
    assert "\\" not in serialized
    assert str(tmp_path) not in serialized


def test_feature_metadata_summary_includes_lineage_when_dataset_path_is_known() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "ts_utc": pd.Timestamp("2026-01-02T21:00:00Z"),
                "timeframe": "1D",
                "date": "2026-01-02",
                "feature_ret_1d": 0.01,
            }
        ]
    )
    summary = build_feature_metadata_summary(
        "features_daily",
        frame,
        source_dataset="bars_daily",
        feature_list=["feature_ret_1d"],
        dataset_path="data/curated/features_daily",
    )

    assert summary["dataset_lineage"]["logical_dataset_id"] == "features_daily"
    assert summary["dataset_lineage"]["source_fingerprint"]
    assert summary["feature_lineage"]["feature_column_count"] == 1


def test_catalog_preserves_explicit_lineage_and_old_records_remain_readable(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    by_run_id = {record.run_id: record for record in records}

    assert by_run_id["strategy_000"].metadata["dataset_lineage"]["logical_dataset_id"] == "features_daily"
    assert by_run_id["alpha_000"].metadata["feature_lineage"]["feature_contract_version"] == "feature_contract_v1"
    assert by_run_id["portfolio_000"].metadata["dataset_lineage"]["dataset_path"] == "data/curated/features_daily"
    assert by_run_id["campaign_000"].metadata["feature_lineage"]["feature_column_count"] == 2
    assert "dataset_lineage" not in by_run_id["strategy_001"].metadata


def test_index_loaded_records_preserve_lineage_equivalently(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    direct = build_catalog(tmp_path, repo_root=tmp_path)
    indexed = load_catalog_records(tmp_path, repo_root=tmp_path, index_path=index_path, mode="index")

    assert [record.to_dict() for record in indexed] == [record.to_dict() for record in direct]


def test_lineage_export_surfaces_metadata_without_inventing_edges(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)
    payload = export_lineage_openlineage(records, edges)
    node = next(
        node
        for node in payload["nodes"]
        if node["kind"] == "catalog_record" and node["facets"]["run_id"] == "strategy_000"
    )

    assert node["facets"]["dataset_lineage"]["logical_dataset_id"] == "features_daily"
    assert node["facets"]["feature_lineage"]["feature_contract_version"] == "feature_contract_v1"
    assert payload["edge_count"] == len(edges)


def test_empty_sparse_and_read_only_behavior(tmp_path: Path) -> None:
    assert build_catalog(tmp_path, repo_root=tmp_path) == []

    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    export_lineage_openlineage(records, build_lineage_edges(records, repo_root=tmp_path))

    assert snapshot_tree(tmp_path) == before
