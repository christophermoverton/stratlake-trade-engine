from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.signal_engine import generate_signals
from src.research.strategy_base import BaseStrategy


class DummyStrategy(BaseStrategy):
    name = "dummy_strategy"
    dataset = "features_daily"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([1, 0, -1], index=df.index, dtype="int64")


class WrongTypeStrategy(BaseStrategy):
    name = "wrong_type_strategy"
    dataset = "features_daily"

    def generate_signals(self, df: pd.DataFrame) -> list[int]:
        return [1 for _ in range(len(df))]


class MisalignedStrategy(BaseStrategy):
    name = "misaligned_strategy"
    dataset = "features_daily"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([1, 0, -1], index=pd.RangeIndex(start=0, stop=len(df)), dtype="int64")


def _feature_frame() -> pd.DataFrame:
    index = pd.Index(["row_a", "row_b", "row_c"], name="row_id")
    ts_utc = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL", "AAPL"], index=index, dtype="string"),
            "ts_utc": pd.Series(ts_utc, index=index),
            "timeframe": pd.Series(["1D", "1D", "1D"], index=index, dtype="string"),
            "feature_alpha": [0.1, -0.2, 0.3],
        },
        index=index,
    )


def test_generate_signals_adds_signal_column() -> None:
    df = _feature_frame()

    result = generate_signals(df, DummyStrategy())

    assert "signal" in result.columns
    assert result.index.equals(df.index)
    assert result["signal"].tolist() == [1, 0, -1]


def test_generate_signals_preserves_input_columns() -> None:
    df = _feature_frame()

    result = generate_signals(df, DummyStrategy())

    assert result["feature_alpha"].tolist() == df["feature_alpha"].tolist()


def test_generate_signals_attaches_signal_diagnostics() -> None:
    df = _feature_frame()

    result = generate_signals(df, DummyStrategy())

    assert result.attrs["signal_diagnostics"] == {
        "total_rows": 3,
        "pct_long": pytest.approx(1 / 3),
        "pct_short": pytest.approx(1 / 3),
        "pct_flat": pytest.approx(1 / 3),
        "total_trades": 2,
        "turnover": pytest.approx(2 / 3),
        "avg_holding_period": 1.0,
        "exposure_pct": pytest.approx(2 / 3),
        "flags": {
            "always_flat": False,
            "always_long": False,
            "always_short": False,
            "no_trades": False,
            "high_turnover": True,
        },
    }


def test_generate_signals_requires_series_output() -> None:
    df = _feature_frame()

    with pytest.raises(TypeError, match="must return a pandas Series"):
        generate_signals(df, WrongTypeStrategy())


def test_generate_signals_requires_matching_index() -> None:
    df = _feature_frame()

    with pytest.raises(ValueError, match="aligned exactly with the input DataFrame index"):
        generate_signals(df, MisalignedStrategy())


def test_generate_signals_fails_for_missing_timeframe_column() -> None:
    df = _feature_frame().drop(columns=["timeframe"])

    with pytest.raises(ValueError, match="missing required strategy input columns"):
        generate_signals(df, DummyStrategy())


def test_generate_signals_rejects_future_feature_timestamps() -> None:
    df = _feature_frame()
    df["feature_source_ts_utc"] = pd.Series(
        [
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-04T00:00:00Z"),
            pd.Timestamp("2025-01-03T00:00:00Z"),
        ],
        index=df.index,
    )

    with pytest.raises(ValueError, match="future_feature_timestamp"):
        generate_signals(df, DummyStrategy())
