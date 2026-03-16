from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import pandas as pd

from src.data.catalog import build_where_clause, quote_sql_path
from src.data.loaders import _ensure_duckdb_con

SUPPORTED_FEATURE_DATASETS = {"features_daily", "features_1m"}
FEATURE_CORE_COLUMNS = ["symbol", "ts_utc", "timeframe", "date"]


@dataclass(frozen=True)
class FeaturePaths:
    root: Path = field(default_factory=lambda: Path(os.getenv("FEATURES_ROOT", "data")))

    def dataset_root(self, dataset: str) -> Path:
        if dataset not in SUPPORTED_FEATURE_DATASETS:
            expected = ", ".join(sorted(SUPPORTED_FEATURE_DATASETS))
            raise ValueError(f"Unknown feature dataset: {dataset!r}. Expected one of: {expected}.")

        nested_root = self.root / "features" / dataset
        direct_root = self.root / dataset

        if nested_root.exists():
            return nested_root
        if direct_root.exists():
            return direct_root
        if self.root.name == "features":
            return direct_root
        return nested_root

    def dataset_glob(self, dataset: str) -> str:
        return (self.dataset_root(dataset) / "**" / "*.parquet").as_posix()


def load_features(
    dataset: str,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
    *,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    paths: Optional[FeaturePaths] = None,
) -> pd.DataFrame:
    """
    Load an engineered feature dataset from partitioned parquet storage.

    Supported layouts:
      - data/features/<dataset>/symbol=<SYMBOL>/date=<YYYY-MM-DD>/*.parquet
      - <features_root>/<dataset>/symbol=<SYMBOL>/.../*.parquet

    Filtering uses the dataset's `symbol` and `date` columns or Hive partitions.
    `end` is exclusive to match the existing data loader pattern.
    """
    resolved_paths = paths or FeaturePaths()
    dataset_glob = resolved_paths.dataset_glob(dataset)
    parquet_files = sorted(resolved_paths.dataset_root(dataset).glob("**/*.parquet"))

    if not parquet_files:
        return pd.DataFrame(columns=FEATURE_CORE_COLUMNS)

    where_sql, params = build_where_clause(symbols=symbols, start_date=start, end_date=end)
    sql = f"""
        SELECT *
        FROM read_parquet('{quote_sql_path(dataset_glob)}', hive_partitioning = true)
        {where_sql}
        ORDER BY symbol, ts_utc
    """
    df = _ensure_duckdb_con(con).execute(sql, params).df()
    return _postprocess(df)


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)

    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")

    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    for column in ("symbol", "timeframe", "date"):
        if column in df.columns:
            df[column] = df[column].astype("string")

    return df.reset_index(drop=True)
