from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.data.load_features import (
    FEATURE_CORE_COLUMNS,
    FeaturePaths,
    create_feature_views,
    load_features,
)


def _write_parquet(file_path: Path, df: pd.DataFrame) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("df_in", df)
        con.execute("CREATE TABLE t AS SELECT * FROM df_in")
        con.execute("COPY t TO ? (FORMAT PARQUET)", [str(file_path)])
    finally:
        con.close()


def _make_feature_row(
    *,
    symbol: str,
    ts_utc: str,
    timeframe: str,
    date: str,
    feature_alpha: float,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "ts_utc": pd.Timestamp(ts_utc, tz="UTC"),
        "timeframe": timeframe,
        "date": date,
        "feature_alpha": feature_alpha,
    }


def test_load_features_empty_returns_core_columns(tmp_path: Path) -> None:
    paths = FeaturePaths(root=tmp_path / "data")

    df = load_features("features_daily", start="2025-11-01", end="2025-11-02", paths=paths)

    assert list(df.columns) == FEATURE_CORE_COLUMNS
    assert df.empty


def test_load_features_daily_filters_and_orders_rows(tmp_path: Path) -> None:
    paths = FeaturePaths(root=tmp_path / "data")
    dataset_root = paths.dataset_root("features_daily")

    _write_parquet(
        dataset_root / "symbol=AAPL" / "date=2025-11-02" / "part-0.parquet",
        pd.DataFrame(
            [
                _make_feature_row(
                    symbol="AAPL",
                    ts_utc="2025-11-02T20:59:00Z",
                    timeframe="1D",
                    date="2025-11-02",
                    feature_alpha=2.0,
                )
            ]
        ),
    )
    _write_parquet(
        dataset_root / "symbol=AAPL" / "date=2025-11-01" / "part-0.parquet",
        pd.DataFrame(
            [
                _make_feature_row(
                    symbol="AAPL",
                    ts_utc="2025-11-01T20:59:00Z",
                    timeframe="1D",
                    date="2025-11-01",
                    feature_alpha=1.0,
                )
            ]
        ),
    )
    _write_parquet(
        dataset_root / "symbol=MSFT" / "date=2025-11-01" / "part-0.parquet",
        pd.DataFrame(
            [
                _make_feature_row(
                    symbol="MSFT",
                    ts_utc="2025-11-01T20:59:00Z",
                    timeframe="1D",
                    date="2025-11-01",
                    feature_alpha=3.0,
                )
            ]
        ),
    )

    df = load_features(
        "features_daily",
        start="2025-11-01",
        end="2025-11-03",
        symbols=["aapl", "msft"],
        paths=paths,
    )

    assert df["symbol"].tolist() == ["AAPL", "AAPL", "MSFT"]
    assert df["ts_utc"].dt.strftime("%Y-%m-%d").tolist() == ["2025-11-01", "2025-11-02", "2025-11-01"]
    assert df["feature_alpha"].tolist() == [1.0, 2.0, 3.0]


def test_load_features_1m_filters_by_symbol_and_date(tmp_path: Path) -> None:
    paths = FeaturePaths(root=tmp_path / "data")
    dataset_root = paths.dataset_root("features_1m")

    _write_parquet(
        dataset_root / "symbol=AAPL" / "date=2025-11-15" / "part-0.parquet",
        pd.DataFrame(
            [
                _make_feature_row(
                    symbol="AAPL",
                    ts_utc="2025-11-15T14:31:00Z",
                    timeframe="1Min",
                    date="2025-11-15",
                    feature_alpha=11.0,
                ),
                _make_feature_row(
                    symbol="AAPL",
                    ts_utc="2025-11-15T14:30:00Z",
                    timeframe="1Min",
                    date="2025-11-15",
                    feature_alpha=10.0,
                ),
            ]
        ),
    )
    _write_parquet(
        dataset_root / "symbol=AAPL" / "date=2025-11-16" / "part-0.parquet",
        pd.DataFrame(
            [
                _make_feature_row(
                    symbol="AAPL",
                    ts_utc="2025-11-16T14:30:00Z",
                    timeframe="1Min",
                    date="2025-11-16",
                    feature_alpha=12.0,
                )
            ]
        ),
    )
    _write_parquet(
        dataset_root / "symbol=MSFT" / "date=2025-11-15" / "part-0.parquet",
        pd.DataFrame(
            [
                _make_feature_row(
                    symbol="MSFT",
                    ts_utc="2025-11-15T14:30:00Z",
                    timeframe="1Min",
                    date="2025-11-15",
                    feature_alpha=20.0,
                )
            ]
        ),
    )

    df = load_features(
        "features_1m",
        start="2025-11-15",
        end="2025-11-16",
        symbols=["AAPL"],
        paths=paths,
    )

    assert df["symbol"].tolist() == ["AAPL", "AAPL"]
    assert df["date"].tolist() == ["2025-11-15", "2025-11-15"]
    assert df["ts_utc"].dt.strftime("%H:%M:%S").tolist() == ["14:30:00", "14:31:00"]
    assert df["feature_alpha"].tolist() == [10.0, 11.0]


def test_load_features_supports_direct_root_and_year_partitions(tmp_path: Path) -> None:
    paths = FeaturePaths(root=tmp_path / "features")
    dataset_root = paths.dataset_root("features_daily")

    _write_parquet(
        dataset_root / "symbol=AAPL" / "year=2025" / "part-0.parquet",
        pd.DataFrame(
            [
                _make_feature_row(
                    symbol="AAPL",
                    ts_utc="2025-03-15T00:00:00Z",
                    timeframe="1D",
                    date="2025-03-15",
                    feature_alpha=5.0,
                )
            ]
        ),
    )

    df = load_features("features_daily", start="2025-01-01", end="2026-01-01", paths=paths)

    assert len(df) == 1
    assert df["symbol"].tolist() == ["AAPL"]
    assert df["feature_alpha"].tolist() == [5.0]


def test_create_feature_views_supports_sql_analytics_and_partition_discovery(tmp_path: Path) -> None:
    paths = FeaturePaths(root=tmp_path / "data")
    dataset_root = paths.dataset_root("features_1m")

    # Omit symbol/date in parquet payload so Hive partition discovery has to recover them.
    _write_parquet(
        dataset_root / "symbol=AAPL" / "date=2025-11-15" / "part-0.parquet",
        pd.DataFrame(
            [
                {
                    "ts_utc": pd.Timestamp("2025-11-15T14:30:00Z"),
                    "timeframe": "1Min",
                    "feature_ret_1m": 0.10,
                },
                {
                    "ts_utc": pd.Timestamp("2025-11-15T14:31:00Z"),
                    "timeframe": "1Min",
                    "feature_ret_1m": 0.30,
                },
            ]
        ),
    )
    _write_parquet(
        dataset_root / "symbol=MSFT" / "date=2025-11-15" / "part-0.parquet",
        pd.DataFrame(
            [
                {
                    "ts_utc": pd.Timestamp("2025-11-15T14:30:00Z"),
                    "timeframe": "1Min",
                    "feature_ret_1m": 0.20,
                }
            ]
        ),
    )

    con = duckdb.connect(database=":memory:")
    try:
        create_feature_views(con, paths)

        rows = con.execute(
            """
            SELECT symbol, CAST(date AS VARCHAR) AS date, feature_ret_1m
            FROM features_1m
            ORDER BY symbol, ts_utc
            """
        ).fetchall()
        aggregates = con.execute(
            """
            SELECT symbol, AVG(feature_ret_1m) AS avg_feature_ret_1m
            FROM features_1m
            GROUP BY symbol
            ORDER BY symbol
            """
        ).fetchall()

        assert rows == [
            ("AAPL", "2025-11-15", 0.10),
            ("AAPL", "2025-11-15", 0.30),
            ("MSFT", "2025-11-15", 0.20),
        ]
        assert aggregates == [("AAPL", 0.20), ("MSFT", 0.20)]
    finally:
        con.close()
