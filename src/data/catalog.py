from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Iterable, Optional, Sequence

import duckdb

CANONICAL_COLS = [
    "symbol",
    "ts_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "timeframe",
    "date"
]

@dataclass(frozen=True)
class CuratedPaths:
    """
    Catalog of curated datset locations.
    Default layout (matches your ingestion conventions):
      data/curated/bars_daily/symbol=XYZ/year=YYYY/*.parquet
      data/curated/bars_1m/symbol=XYZ/date=YYYY-MM-DD/*.parquet
    """
    root: Path = Path("data/curated")

    @property
    def bars_daily_glob(self) -> str:
        return (self.root / "bars_daily" / "**" / "*.parquet").as_posix()

    @property
    def bars_1m_glob(self) -> str:
        return (self.root / "bars_1m" / "**" / "*.parquet").as_posix()

    def dataset_root(self, dataset: str) -> Path:
        if dataset == "bars_daily":
            return self.root / "bars_daily"
        if dataset == "bars_1m":
            return self.root / "bars_1m"
        raise ValueError(f"Unknown dataset: {dataset!r}.  Expected 'bars_daily' or 'bars_1m'.")

    def dataset_glob(self, dataset: str) -> str:
        return (self.dataset_root(dataset) / "**" / "*.parquet").as_posix()


def _normalize_symbols(symbols: Optional[Sequence[str]]) -> Optional[list[str]]:
    if symbols is None:
        return None
    out = [s.strip().upper() for s in symbols if s and s.strip()]
    return out or None

def _validate_date_str(d: Optional[str], label: str) -> None:
    if d is None:
        return
    #Basic guard; Loaders will ttreat this as 'YYY-MM-DD' for filtering
    if len(d) != 10 or d[4] != "-" or d[7] != "-":
        raise ValueError(f"{label} must be 'YYYY-MM-DD' (got {dir}).")


def _glob_has_files(glob_pattern: str) -> bool:
    """
    Return True if at least one parquet file matches the glob.
    Works with absolute Windows paths.
    """
    return len(glob.glob(glob_pattern, recursive=True)) > 0


def count_parquet_partitions(glob_pattern: str) -> int:
    parquet_files = glob.glob(glob_pattern, recursive=True)
    return len({str(Path(file_path).parent) for file_path in parquet_files})


def quote_sql_path(path: str) -> str:
    return path.replace("'", "''")


def parquet_scan_sql(glob_pattern: str) -> str:
    return f"read_parquet('{quote_sql_path(glob_pattern)}', hive_partitioning = true)"


def _create_empty_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
) -> None:
    """
    Create an empty view with canonical schema.
    """
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
            CAST(NULL AS VARCHAR) AS symbol,
            CAST(NULL AS TIMESTAMP) AS ts_utc,
            CAST(NULL AS DOUBLE) AS open,
            CAST(NULL AS DOUBLE) AS high,
            CAST(NULL AS DOUBLE) AS low,
            CAST(NULL AS DOUBLE) AS close,
            CAST(NULL AS BIGINT) AS volume,
            CAST(NULL AS VARCHAR) AS source,
            CAST(NULL AS VARCHAR) AS timeframe,
            CAST(NULL AS VARCHAR) AS date
        WHERE FALSE
        """
    )


def create_curated_views(
    con: duckdb.DuckDBPyConnection,
    paths: Optional[CuratedPaths] = None,
    *,
    view_daily: str = "bars_daily",
    view_1m: str = "bars_1m",
) -> None:
    """
    Create DuckDB views backed by parquet globs.
    Handles empty datasets safely.
    """
    paths = paths or CuratedPaths()

    daily_glob = paths.bars_daily_glob
    one_min_glob = paths.bars_1m_glob

    # Daily
    if _glob_has_files(daily_glob):
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view_daily} AS
            SELECT * FROM {parquet_scan_sql(daily_glob)}
            """
        )
    else:
        _create_empty_view(con, view_daily)

    # 1-Minute
    if _glob_has_files(one_min_glob):
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view_1m} AS
            SELECT * FROM {parquet_scan_sql(one_min_glob)}
            """
        )
    else:
        _create_empty_view(con, view_1m)
        
def build_where_clause(
    symbols: Optional[Sequence[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Build a safe filter clause + parameter dict for DuckDB.
    Filters:
      - symbols IN (...)
      - date between [start_date, end_date) by default (end exclusive)
    """
    syms = _normalize_symbols(symbols)
    _validate_date_str(start_date, "start_date")
    _validate_date_str(end_date, "end_date")

    clauses: list[str] = []
    params: dict = {}

    if syms:
        # DuckDB supports list parameter for IN via ANY(?)
        clauses.append("symbol = ANY($symbols)")
        params["symbols"] = syms

    if start_date:
        clauses.append("date >= $start_date")
        params["start_date"] = start_date

    if end_date:
        clauses.append("date < $end_date")
        params["end_date"] = end_date

    where_sql = ""
    if clauses:
        where_sql = "WHERE " + " AND ".join(clauses)

    return where_sql, params
