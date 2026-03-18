from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.data.feature_metadata import export_feature_metadata


def _write_parquet(file_path: Path, df: pd.DataFrame) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("df_in", df)
        con.execute("CREATE TABLE t AS SELECT * FROM df_in")
        con.execute("COPY t TO ? (FORMAT PARQUET)", [str(file_path)])
    finally:
        con.close()


def _registry_yaml() -> str:
    return """
features_daily_v1:
  source_dataset: bars_daily
  features:
    - feature_ret_1d
    - feature_sma20
features_1m_v1:
  source_dataset: bars_1m
  features:
    - feature_ret_1m
    - feature_vol_30m
""".strip()


def test_export_feature_metadata_writes_deterministic_dataset_summaries(tmp_path: Path) -> None:
    registry_path = tmp_path / "configs" / "features.yml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(_registry_yaml(), encoding="utf-8")

    feature_root = tmp_path / "data" / "curated"
    artifact_path = tmp_path / "artifacts" / "features" / "feature_metadata.json"

    _write_parquet(
        feature_root / "features_1m" / "symbol=AAPL" / "date=2025-11-15" / "part-0.parquet",
        pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "ts_utc": pd.Timestamp("2025-11-15T14:30:00Z"),
                    "timeframe": "1Min",
                    "date": "2025-11-15",
                    "feature_ret_1m": 0.10,
                    "feature_vol_30m": None,
                },
                {
                    "symbol": "AAPL",
                    "ts_utc": pd.Timestamp("2025-11-15T14:31:00Z"),
                    "timeframe": "1Min",
                    "date": "2025-11-15",
                    "feature_ret_1m": float("inf"),
                    "feature_vol_30m": 15.0,
                },
            ]
        ),
    )
    _write_parquet(
        feature_root / "features_1m" / "symbol=MSFT" / "date=2025-11-16" / "part-0.parquet",
        pd.DataFrame(
            [
                {
                    "symbol": "MSFT",
                    "ts_utc": pd.Timestamp("2025-11-16T14:30:00Z"),
                    "timeframe": "1Min",
                    "date": "2025-11-16",
                    "feature_ret_1m": None,
                    "feature_vol_30m": 20.0,
                },
            ]
        ),
    )
    _write_parquet(
        feature_root / "features_daily" / "symbol=AAPL" / "year=2025" / "part-0.parquet",
        pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "ts_utc": pd.Timestamp("2025-11-14T21:00:00Z"),
                    "timeframe": "1D",
                    "date": "2025-11-14",
                    "feature_ret_1d": -0.02,
                    "feature_sma20": 150.0,
                },
            ]
        ),
    )
    _write_parquet(
        feature_root / "features_daily" / "symbol=MSFT" / "year=2025" / "part-0.parquet",
        pd.DataFrame(
            [
                {
                    "symbol": "MSFT",
                    "ts_utc": pd.Timestamp("2025-11-17T21:00:00Z"),
                    "timeframe": "1D",
                    "date": "2025-11-17",
                    "feature_ret_1d": 0.03,
                    "feature_sma20": 300.0,
                },
            ]
        ),
    )

    first_path = export_feature_metadata(
        artifact_path=artifact_path,
        registry_path=registry_path,
        feature_data_root=feature_root,
    )
    first_text = first_path.read_text(encoding="utf-8")

    second_path = export_feature_metadata(
        artifact_path=artifact_path,
        registry_path=registry_path,
        feature_data_root=feature_root,
    )
    second_text = second_path.read_text(encoding="utf-8")
    payload = json.loads(second_text)

    assert first_path == artifact_path
    assert second_path == artifact_path
    assert first_text == second_text

    datasets = {item["dataset_name"]: item for item in payload["datasets"]}
    minute = datasets["features_1m"]
    daily = datasets["features_daily"]

    assert [item["dataset_name"] for item in payload["datasets"]] == ["features_1m", "features_daily"]

    assert minute["source_dataset"] == "bars_1m"
    assert minute["feature_list"] == ["feature_ret_1m", "feature_vol_30m"]
    assert minute["feature_count"] == 2
    assert minute["schema"]["column_names"] == [
        "symbol",
        "ts_utc",
        "timeframe",
        "date",
        "feature_ret_1m",
        "feature_vol_30m",
    ]
    assert minute["schema"]["feature_columns"] == ["feature_ret_1m", "feature_vol_30m"]
    assert minute["metrics"]["row_count"] == 3
    assert minute["metrics"]["symbol_coverage"] == 2
    assert minute["metrics"]["date_range"] == {"start": "2025-11-15", "end": "2025-11-16"}
    assert minute["feature_statistics"]["feature_ret_1m"] == {
        "null_pct": 0.333333,
        "min": 0.1,
        "max": 0.1,
    }
    assert minute["feature_statistics"]["feature_vol_30m"] == {
        "null_pct": 0.333333,
        "min": 15.0,
        "max": 20.0,
    }

    assert daily["source_dataset"] == "bars_daily"
    assert daily["feature_list"] == ["feature_ret_1d", "feature_sma20"]
    assert daily["feature_count"] == 2
    assert daily["metrics"]["row_count"] == 2
    assert daily["metrics"]["symbol_coverage"] == 2
    assert daily["metrics"]["date_range"] == {"start": "2025-11-14", "end": "2025-11-17"}
    assert daily["feature_statistics"]["feature_ret_1d"] == {
        "null_pct": 0.0,
        "min": -0.02,
        "max": 0.03,
    }
