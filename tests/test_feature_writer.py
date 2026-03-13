import pandas as pd

from src.data import feature_writer


def _make_features(
    *,
    symbol: str,
    timeframe: str,
    ts_values: list[str],
    feature_values: list[float],
) -> pd.DataFrame:
    ts = pd.to_datetime(ts_values, utc=True)
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(ts),
            "ts_utc": ts,
            "timeframe": [timeframe] * len(ts),
            "date": ts.strftime("%Y-%m-%d"),
            "feature_value": feature_values,
        }
    )


def test_daily_partition_writes(tmp_path, monkeypatch):
    monkeypatch.setitem(feature_writer.FEATURE_PATHS, "1D", tmp_path / "features_daily")

    df = _make_features(
        symbol="AAPL",
        timeframe="1D",
        ts_values=["2025-01-02T00:00:00Z", "2025-03-15T00:00:00Z"],
        feature_values=[1.0, 2.0],
    )

    feature_writer.write_features(df, "1D")

    partition = tmp_path / "features_daily" / "symbol=AAPL" / "year=2025"
    files = sorted(partition.glob("part-*.parquet"))

    assert len(files) == 1
    written = pd.read_parquet(files[0]).sort_values("ts_utc").reset_index(drop=True)
    assert written["ts_utc"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-03-15"]


def test_1m_partition_writes(tmp_path, monkeypatch):
    monkeypatch.setitem(feature_writer.FEATURE_PATHS, "1Min", tmp_path / "features_1m")

    df = _make_features(
        symbol="AAPL",
        timeframe="1Min",
        ts_values=["2025-01-02T14:30:00Z", "2025-01-02T14:31:00Z"],
        feature_values=[1.0, 2.0],
    )

    feature_writer.write_features(df, "1Min")

    partition = tmp_path / "features_1m" / "symbol=AAPL" / "date=2025-01-02"
    files = sorted(partition.glob("part-*.parquet"))

    assert len(files) == 1
    written = pd.read_parquet(files[0]).sort_values("ts_utc").reset_index(drop=True)
    assert written["feature_value"].tolist() == [1.0, 2.0]


def test_rerun_idempotency(tmp_path, monkeypatch):
    monkeypatch.setitem(feature_writer.FEATURE_PATHS, "1Min", tmp_path / "features_1m")

    df = _make_features(
        symbol="AAPL",
        timeframe="1Min",
        ts_values=["2025-01-02T14:30:00Z", "2025-01-02T14:31:00Z"],
        feature_values=[1.0, 2.0],
    )

    feature_writer.write_features(df, "1Min")
    feature_writer.write_features(df, "1Min")

    partition = tmp_path / "features_1m" / "symbol=AAPL" / "date=2025-01-02"
    files = sorted(partition.glob("part-*.parquet"))

    assert len(files) == 1
    written = pd.read_parquet(files[0]).sort_values("ts_utc").reset_index(drop=True)
    assert len(written) == 2
    assert written["feature_value"].tolist() == [1.0, 2.0]


def test_dedupe_behavior_with_existing_partition_data(tmp_path, monkeypatch):
    monkeypatch.setitem(feature_writer.FEATURE_PATHS, "1Min", tmp_path / "features_1m")

    existing = _make_features(
        symbol="AAPL",
        timeframe="1Min",
        ts_values=["2025-01-02T14:30:00Z", "2025-01-02T14:31:00Z"],
        feature_values=[1.0, 2.0],
    )
    incoming = _make_features(
        symbol="AAPL",
        timeframe="1Min",
        ts_values=["2025-01-02T14:31:00Z", "2025-01-02T14:32:00Z"],
        feature_values=[22.0, 3.0],
    )

    feature_writer.write_features(existing, "1Min")
    feature_writer.write_features(incoming, "1Min")

    partition = tmp_path / "features_1m" / "symbol=AAPL" / "date=2025-01-02"
    written = pd.read_parquet(next(partition.glob("part-*.parquet")))
    written = written.sort_values("ts_utc").reset_index(drop=True)

    assert written["feature_value"].tolist() == [1.0, 22.0, 3.0]
    assert written[["symbol", "ts_utc", "timeframe"]].duplicated().sum() == 0
