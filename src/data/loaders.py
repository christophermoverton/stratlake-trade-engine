from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import pandas as pd

from src.data.catalog import (
    CANONICAL_COLS,
    CuratedPaths,
    build_where_clause,
    count_parquet_partitions,
    create_curated_views,
    parquet_scan_sql,
    quote_sql_path,
)
from src.data.contract_validation import BarsContract

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadConfig:
    """
    Controls view names and ensures the canonical schema is returned.
    """
    view_daily: str = "bars_daily"
    view_1m : str = "bars_1m"


def _default_curated_paths() -> CuratedPaths:
    marketlake_root = os.getenv("MARKETLAKE_ROOT")
    if marketlake_root:
        return CuratedPaths(root=Path(marketlake_root))
    return CuratedPaths()
    
def _ensure_duckdb_con(con: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyConnection:
    if con is not None:
        return con
    return duckdb.connect(database=os.getenv("DUCKDB_PATH", ":memory:"))

def _select_canonical_sql(view_name: str, where_sql: str) -> str:
    #Select explicit columns to guarantee scchema order and avoid surprises.
    cols = ", ".join(CANONICAL_COLS)
    return f"SELECT {cols} FROM {view_name} {where_sql} ORDER BY symbol, ts_utc"


def debug_count_parquet(con: duckdb.DuckDBPyConnection, parquet_path: str) -> int:
    return con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{quote_sql_path(parquet_path)}', hive_partitioning = true)"
    ).fetchone()[0]


def _log_loader_debug(
    *,
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    parquet_path: str,
    where_sql: str,
    params: dict,
) -> None:
    partitions_discovered = count_parquet_partitions(parquet_path)
    if partitions_discovered:
        rows_before_filter = debug_count_parquet(con, parquet_path)
    else:
        rows_before_filter = 0
    rows_after_filter = con.execute(
        f"SELECT COUNT(*) FROM {view_name} {where_sql}",
        params,
    ).fetchone()[0]
    LOGGER.debug(
        "Parquet load debug view=%s parquet_scan_path=%s parquet_scan_sql=%s partitions_discovered=%s rows_before_filter=%s rows_after_filter=%s",
        view_name,
        parquet_path,
        parquet_scan_sql(parquet_path),
        partitions_discovered,
        rows_before_filter,
        rows_after_filter,
    )

def load_bars_daily(
    symbols: Optional[Sequence[str]] = None,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    paths: Optional[CuratedPaths] = None,
    cfg: Optional[LoadConfig] = None,
    validate_contract: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Load curated daily bars for symbols + date range.
    Returns canonical schema DataFrame.
    """
    cfg = cfg or LoadConfig()
    con = _ensure_duckdb_con(con)
    paths = paths or _default_curated_paths()
    create_curated_views(con, paths, view_daily=cfg.view_daily, view_1m=cfg.view_1m)

    parquet_path = paths.dataset_glob("bars_daily")
    where_sql, params = build_where_clause(symbols=symbols, start_date=start_date, end_date=end_date)
    sql = _select_canonical_sql(cfg.view_daily, where_sql)

    _log_loader_debug(
        con=con,
        view_name=cfg.view_daily,
        parquet_path=parquet_path,
        where_sql=where_sql,
        params=params,
    )
    df = con.execute(sql, params).df()
    df = _postprocess(df)

    if validate_contract:
        BarsContract().validate(df, strict=strict, normalize_ts_utc=True)

    return df

def load_bars_1m(
    symbols: Optional[Sequence[str]] = None,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    paths: Optional[CuratedPaths] = None,
    cfg: Optional[LoadConfig] = None,
    validate_contract: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Load curated 1-minute bars for symbols + date range.
    Returns canonical schema DataFrame.
    """
    cfg = cfg or LoadConfig()
    con = _ensure_duckdb_con(con)
    paths = paths or _default_curated_paths()
    create_curated_views(con, paths, view_daily=cfg.view_daily, view_1m=cfg.view_1m)

    parquet_path = paths.dataset_glob("bars_1m")
    where_sql, params = build_where_clause(symbols=symbols, start_date=start_date, end_date=end_date)
    sql = _select_canonical_sql(cfg.view_1m, where_sql)

    _log_loader_debug(
        con=con,
        view_name=cfg.view_1m,
        parquet_path=parquet_path,
        where_sql=where_sql,
        params=params,
    )
    df = con.execute(sql, params).df()
    df = _postprocess(df)

    if validate_contract:
        BarsContract().validate(df, strict=strict, normalize_ts_utc=True)

    return df

def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight normalization/typing guards.
    Assumes ts_utc stored as timestamp; DuckDB usually returns pandas datetime64[ns].
    """
    if df.empty:
        # Ensure expected columns exist even when empty
        return pd.DataFrame(columns=CANONICAL_COLS)

    # Enforce canonical column order (DuckDB already does, but keep it deterministic)
    df = df.loc[:, CANONICAL_COLS]

    # ts_utc should be datetime
    if "ts_utc" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")

    # date should be string YYYY-MM-DD (your partition helper)
    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # symbol/timeframe/source strings
    for c in ("symbol", "source", "timeframe", "date"):
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df
