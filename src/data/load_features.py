from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.data.catalog import build_where_clause, parquet_scan_sql
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
        curated_root = self.root / "curated" / dataset
        direct_root = self.root / dataset
        candidates = (nested_root, curated_root, direct_root)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        if self.root.name == "features":
            return direct_root
        if self.root.name == "curated":
            return direct_root
        return curated_root

    def dataset_glob(self, dataset: str) -> str:
        return (self.dataset_root(dataset) / "**" / "*.parquet").as_posix()


@dataclass(frozen=True)
class FeatureViewConfig:
    view_daily: str = "features_daily"
    view_1m: str = "features_1m"

    def view_name(self, dataset: str) -> str:
        if dataset == "features_daily":
            return self.view_daily
        if dataset == "features_1m":
            return self.view_1m
        expected = ", ".join(sorted(SUPPORTED_FEATURE_DATASETS))
        raise ValueError(f"Unknown feature dataset: {dataset!r}. Expected one of: {expected}.")


def _create_empty_feature_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
            CAST(NULL AS VARCHAR) AS symbol,
            CAST(NULL AS TIMESTAMP) AS ts_utc,
            CAST(NULL AS VARCHAR) AS timeframe,
            CAST(NULL AS VARCHAR) AS date
        WHERE FALSE
        """
    )


def create_feature_views(
    con: duckdb.DuckDBPyConnection,
    paths: Optional[FeaturePaths] = None,
    *,
    cfg: Optional[FeatureViewConfig] = None,
) -> None:
    """
    Register DuckDB views over partitioned feature parquet datasets.

    Views use recursive parquet scanning with Hive partition discovery so
    analytics queries can hit engineered features directly with SQL.
    """
    resolved_paths = paths or FeaturePaths()
    resolved_cfg = cfg or FeatureViewConfig()

    for dataset in sorted(SUPPORTED_FEATURE_DATASETS):
        parquet_files = sorted(resolved_paths.dataset_root(dataset).glob("**/*.parquet"))
        view_name = resolved_cfg.view_name(dataset)

        if parquet_files:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM {parquet_scan_sql(resolved_paths.dataset_glob(dataset))}
                """
            )
        else:
            _create_empty_feature_view(con, view_name)


def load_features(
    dataset: str,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
    *,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    paths: Optional[FeaturePaths] = None,
    cfg: Optional[FeatureViewConfig] = None,
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
    resolved_cfg = cfg or FeatureViewConfig()
    parquet_files = sorted(resolved_paths.dataset_root(dataset).glob("**/*.parquet"))

    if not parquet_files:
        return pd.DataFrame(columns=FEATURE_CORE_COLUMNS)

    connection = _ensure_duckdb_con(con)
    create_feature_views(connection, resolved_paths, cfg=resolved_cfg)

    where_sql, params = build_where_clause(symbols=symbols, start_date=start, end_date=end)
    view_name = resolved_cfg.view_name(dataset)
    sql = f"""
        SELECT *
        FROM {view_name}
        {where_sql}
        ORDER BY symbol, ts_utc
    """
    df = connection.execute(sql, params).df()
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
