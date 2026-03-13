from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.data.catalog import CuratedPaths
from src.data.loaders import load_bars_1m


def _write_parquet(file_path: Path, df: pd.DataFrame) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("df_in", df)
        con.execute("CREATE TABLE t AS SELECT * FROM df_in")
        con.execute("COPY t TO ? (FORMAT PARQUET)", [str(file_path)])
    finally:
        con.close()


def test_load_bars_1m_reads_hive_partitioned_parquet(tmp_path: Path) -> None:
    curated_root = tmp_path / "marketlake"
    paths = CuratedPaths(root=curated_root)
    parquet_path = curated_root / "bars_1m" / "symbol=AAPL" / "date=2025-11-15" / "part-000.parquet"

    # Intentionally omit symbol/date so DuckDB must recover them from the Hive partitions.
    df_in = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(
                ["2025-11-15T14:30:00Z", "2025-11-15T14:31:00Z"],
                utc=True,
            ),
            "open": [150.0, 150.5],
            "high": [151.0, 151.5],
            "low": [149.5, 150.25],
            "close": [150.75, 151.25],
            "volume": [100, 120],
            "source": pd.Series(["alpaca_iex", "alpaca_iex"], dtype="string"),
            "timeframe": pd.Series(["1Min", "1Min"], dtype="string"),
        }
    )
    _write_parquet(parquet_path, df_in)

    result = load_bars_1m(
        ["AAPL"],
        start_date="2025-11-15",
        end_date="2025-11-16",
        paths=paths,
    )

    assert len(result) == 2
    assert result["symbol"].tolist() == ["AAPL", "AAPL"]
    assert result["date"].tolist() == ["2025-11-15", "2025-11-15"]
    assert result["timeframe"].tolist() == ["1Min", "1Min"]
