from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.backtest_runner import run_backtest


def _backtest_frame() -> pd.DataFrame:
    index = pd.Index(["row_a", "row_b", "row_c", "row_d"], name="row_id")
    ts_utc = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL", "AAPL", "AAPL"], index=index, dtype="string"),
            "ts_utc": pd.Series(ts_utc, index=index),
            "signal": [1, 1, -1, 0],
            "feature_ret_1d": [0.01, -0.02, 0.03, -0.01],
            "feature_alpha": [0.5, 0.6, 0.4, 0.7],
        },
        index=index,
    )


def test_run_backtest_computes_strategy_returns_from_shifted_signals() -> None:
    df = _backtest_frame()

    result = run_backtest(df)

    assert result["strategy_return"].tolist() == [0.0, -0.02, 0.03, 0.01]


def test_run_backtest_computes_deterministic_equity_curve() -> None:
    df = _backtest_frame()

    result = run_backtest(df)

    expected = [1.0, 0.98, 1.0094, 1.019494]
    assert result["equity_curve"].tolist() == pytest.approx(expected)


def test_run_backtest_preserves_input_columns() -> None:
    df = _backtest_frame()

    result = run_backtest(df)

    assert result.index.equals(df.index)
    assert result["feature_alpha"].tolist() == df["feature_alpha"].tolist()


def test_run_backtest_raises_when_signal_column_is_missing() -> None:
    df = _backtest_frame().drop(columns=["signal"])

    with pytest.raises(ValueError, match="must include a 'signal' column"):
        run_backtest(df)


def test_run_backtest_raises_when_supported_return_column_is_missing() -> None:
    df = _backtest_frame().drop(columns=["feature_ret_1d"])

    with pytest.raises(ValueError, match="Expected one of"):
        run_backtest(df)
