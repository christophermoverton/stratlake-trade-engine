from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import duckdb
import pandas as pd

from src.data.catalog import CANONICAL_COLS, CuratedPaths, build_where_clause, create_curated_views

@dataclass(frozen=True)
class LoadConfig:
    """
    Controls view names and ensures the canonical schema is returned.
    """
    view_daily: str = "bars_daily"
    view_1m : str = "bars_1m"
    
def _ensure_duckdb_con(con: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyConnection:
    return con if con is not None else duckdb.connect(database=":memory:")

def _select_canonical_sql(view_name: str, where_sql: str) -> str:
    #Select explicit columns to guarantee scchema order and avoid surprises.
    cols = ", ".join(CANONICAL_COLS)
    return f"SELECT {cols} FROM {view_name} {where_sql} ORDER BY symbol, ts_utc"

def load_bars_daily(
    symbols: Optional[Sequence[str]] = None,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    paths: Optional[CuratedPaths] = None,
    cfg: Optional[LoadConfig] = None,
) -> pd.DataFrame:
    """
    Load curated daily bars for symbols + date range.
    Returns canonical schema DataFrame.
    """
    cfg = cfg or LoadConfig()
    con = _ensure_duckdb_con(con)
    create_curated_views(con, paths, view_daily=cfg.view_daily, view_1m=cfg.view_1m)
    
    where_sql, params = build_where_clause(symbols=symbols, start_date=start_date, end_date=end_date)
    sql = _select_canonical_sql(cfg.view_daily, where_sql)
    
    df = con.execute(sql, params).df()
    return _postprocess(df)

def load_bars_1m(
    symbols: Optional[Sequence[str]] = None,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    paths: Optional[CuratedPaths] = None,
    cfg: Optional[LoadConfig] = None,
) -> pd.DataFrame:
    """
    Load curated 1-minute bars for symbols + date range.
    Returns canonical schema DataFrame.
    """
    cfg = cfg or LoadConfig()
    con = _ensure_duckdb_con(con)
    create_curated_views(con, paths, view_daily=cfg.view_daily, view_1m=cfg.view_1m)

    where_sql, params = build_where_clause(symbols=symbols, start_date=start_date, end_date=end_date)
    sql = _select_canonical_sql(cfg.view_1m, where_sql)

    df = con.execute(sql, params).df()
    return _postprocess(df)

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