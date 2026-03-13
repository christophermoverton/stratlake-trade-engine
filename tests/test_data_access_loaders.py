from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.data.catalog import CANONICAL_COLS, CuratedPaths
from src.data.loaders import _ensure_duckdb_con, load_bars_daily, load_bars_1m


def _write_parquet(file_path: Path, df: pd.DataFrame) -> None:
    """
    Write parquet without pyarrow/fastparquet by using DuckDB COPY.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        # Create a DuckDB table from the pandas dataframe and export to parquet
        con.register("df_in", df)
        con.execute("CREATE TABLE t AS SELECT * FROM df_in")
        con.execute(
            "COPY t TO ? (FORMAT PARQUET)",
            [str(file_path)],
        )
    finally:
        con.close()


def _make_row(
    *,
    symbol: str,
    ts_utc: str,
    o: float,
    h: float,
    l: float,
    c: float,
    v: int,
    source: str,
    timeframe: str,
    date: str,
) -> dict:
    return {
        "symbol": symbol,
        "ts_utc": pd.Timestamp(ts_utc, tz="UTC"),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "source": source,
        "timeframe": timeframe,
        "date": date,  # keep as string YYYY-MM-DD for partition helper
    }


def test_loaders_empty_returns_canonical_cols(tmp_path: Path) -> None:
    curated_root = tmp_path / "data" / "curated"
    paths = CuratedPaths(root=curated_root)

    df = load_bars_daily(
        ["AAPL"],
        start_date="2025-11-01",
        end_date="2025-11-02",
        paths=paths,
    )

    assert list(df.columns) == CANONICAL_COLS
    assert df.empty


def test_load_bars_daily_filters_symbol_and_date(tmp_path: Path) -> None:
    curated_root = tmp_path / "data" / "curated"
    paths = CuratedPaths(root=curated_root)

    # Write daily parquet files for AAPL (two dates) and MSFT (one date)
    aapl_1101 = pd.DataFrame(
        [
            _make_row(
                symbol="AAPL",
                ts_utc="2025-11-01T14:30:00Z",
                o=100,
                h=110,
                l=95,
                c=105,
                v=1000,
                source="alpaca_iex",
                timeframe="1D",
                date="2025-11-01",
            ),
            _make_row(
                symbol="AAPL",
                ts_utc="2025-11-01T20:59:00Z",
                o=105,
                h=112,
                l=104,
                c=110,
                v=1200,
                source="alpaca_iex",
                timeframe="1D",
                date="2025-11-01",
            ),
        ]
    )
    _write_parquet(
        curated_root / "bars_daily" / "symbol=AAPL" / "date=2025-11-01" / "data.parquet",
        aapl_1101,
    )

    aapl_1102 = pd.DataFrame(
        [
            _make_row(
                symbol="AAPL",
                ts_utc="2025-11-02T20:59:00Z",
                o=110,
                h=115,
                l=109,
                c=114,
                v=900,
                source="alpaca_iex",
                timeframe="1D",
                date="2025-11-02",
            )
        ]
    )
    _write_parquet(
        curated_root / "bars_daily" / "symbol=AAPL" / "date=2025-11-02" / "data.parquet",
        aapl_1102,
    )

    msft_1101 = pd.DataFrame(
        [
            _make_row(
                symbol="MSFT",
                ts_utc="2025-11-01T20:59:00Z",
                o=200,
                h=205,
                l=198,
                c=203,
                v=800,
                source="alpaca_iex",
                timeframe="1D",
                date="2025-11-01",
            )
        ]
    )
    _write_parquet(
        curated_root / "bars_daily" / "symbol=MSFT" / "date=2025-11-01" / "data.parquet",
        msft_1101,
    )

    # Query: AAPL only, date range [2025-11-01, 2025-11-02) -> ONLY 2025-11-01 rows
    df = load_bars_daily(
        ["AAPL"],
        start_date="2025-11-01",
        end_date="2025-11-02",
        paths=paths,
    )

    assert list(df.columns) == CANONICAL_COLS
    assert not df.empty
    assert df["symbol"].unique().tolist() == ["AAPL"]
    assert df["date"].unique().tolist() == ["2025-11-01"]
    assert len(df) == 2  # only the two AAPL rows from 11-01


def test_load_bars_1m_filters_symbol_and_date(tmp_path: Path) -> None:
    curated_root = tmp_path / "data" / "curated"
    paths = CuratedPaths(root=curated_root)

    # Write 1m parquet for AAPL on 2025-11-15 and 2025-11-16
    aapl_1115 = pd.DataFrame(
        [
            _make_row(
                symbol="AAPL",
                ts_utc="2025-11-15T14:30:00Z",
                o=150,
                h=151,
                l=149.5,
                c=150.5,
                v=100,
                source="alpaca_iex",
                timeframe="1Min",
                date="2025-11-15",
            ),
            _make_row(
                symbol="AAPL",
                ts_utc="2025-11-15T14:31:00Z",
                o=150.5,
                h=152,
                l=150.25,
                c=151.75,
                v=120,
                source="alpaca_iex",
                timeframe="1Min",
                date="2025-11-15",
            ),
        ]
    )
    _write_parquet(
        curated_root / "bars_1m" / "symbol=AAPL" / "date=2025-11-15" / "chunk.parquet",
        aapl_1115,
    )

    aapl_1116 = pd.DataFrame(
        [
            _make_row(
                symbol="AAPL",
                ts_utc="2025-11-16T14:30:00Z",
                o=152,
                h=153,
                l=151.5,
                c=152.25,
                v=90,
                source="alpaca_iex",
                timeframe="1Min",
                date="2025-11-16",
            )
        ]
    )
    _write_parquet(
        curated_root / "bars_1m" / "symbol=AAPL" / "date=2025-11-16" / "chunk.parquet",
        aapl_1116,
    )

    # Query: [2025-11-15, 2025-11-16) -> ONLY 2025-11-15 rows
    df = load_bars_1m(
        ["AAPL"],
        start_date="2025-11-15",
        end_date="2025-11-16",
        paths=paths,
    )

    assert list(df.columns) == CANONICAL_COLS
    assert not df.empty
    assert df["symbol"].unique().tolist() == ["AAPL"]
    assert df["date"].unique().tolist() == ["2025-11-15"]
    assert df["timeframe"].unique().tolist() == ["1Min"]
    assert len(df) == 2


def test_loaders_default_to_marketlake_root_env(tmp_path: Path, monkeypatch) -> None:
    curated_root = tmp_path / "marketlake_curated"
    monkeypatch.setenv("MARKETLAKE_ROOT", str(curated_root))

    data = pd.DataFrame(
        [
            _make_row(
                symbol="AAPL",
                ts_utc="2025-11-01T20:59:00Z",
                o=100,
                h=110,
                l=95,
                c=105,
                v=1000,
                source="alpaca_iex",
                timeframe="1D",
                date="2025-11-01",
            )
        ]
    )
    _write_parquet(
        curated_root / "bars_daily" / "symbol=AAPL" / "date=2025-11-01" / "data.parquet",
        data,
    )

    df = load_bars_daily(["AAPL"], start_date="2025-11-01", end_date="2025-11-02")

    assert len(df) == 1
    assert df["symbol"].tolist() == ["AAPL"]


def test_ensure_duckdb_con_defaults_to_env_path(monkeypatch) -> None:
    calls: list[str] = []

    class StubConnection:
        def close(self) -> None:
            return None

    def fake_connect(*, database: str):
        calls.append(database)
        return StubConnection()

    monkeypatch.setenv("DUCKDB_PATH", "custom.duckdb")
    monkeypatch.setattr("src.data.loaders.duckdb.connect", fake_connect)

    con = _ensure_duckdb_con()

    assert isinstance(con, StubConnection)
    assert calls == ["custom.duckdb"]
