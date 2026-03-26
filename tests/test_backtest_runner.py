from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.execution import ExecutionConfig
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

    assert result["executed_signal"].tolist() == [0.0, 1.0, 1.0, -1.0]
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


def test_run_backtest_applies_execution_costs_and_slippage_on_position_changes() -> None:
    df = _backtest_frame()
    config = ExecutionConfig(
        enabled=True,
        execution_delay=1,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    result = run_backtest(df, config)

    assert result["delta_position"].tolist() == [0.0, 1.0, 0.0, -2.0]
    assert result["abs_delta_position"].tolist() == [0.0, 1.0, 0.0, 2.0]
    assert result["turnover"].tolist() == [0.0, 1.0, 0.0, 2.0]
    assert result["trade_event"].tolist() == [False, True, False, True]
    assert result["transaction_cost"].tolist() == pytest.approx([0.0, 0.001, 0.0, 0.002])
    assert result["slippage_cost"].tolist() == pytest.approx([0.0, 0.0005, 0.0, 0.001])
    assert result["execution_friction"].tolist() == pytest.approx([0.0, 0.0015, 0.0, 0.003])
    assert result["gross_strategy_return"].tolist() == pytest.approx([0.0, -0.02, 0.03, 0.01])
    assert result["net_strategy_return"].tolist() == pytest.approx([0.0, -0.0215, 0.03, 0.007])
    assert result["strategy_return"].tolist() == pytest.approx(result["net_strategy_return"].tolist())


def test_run_backtest_turnover_tracks_entries_exits_and_flips_from_executed_positions() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "ts_utc": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
            "signal": [1, -1, -1, 0, 1],
            "feature_ret_1d": [0.0, 0.01, -0.02, 0.03, 0.04],
        }
    )

    result = run_backtest(df, ExecutionConfig(enabled=False, execution_delay=1, transaction_cost_bps=0.0, slippage_bps=0.0))

    assert result["executed_signal"].tolist() == [0.0, 1.0, -1.0, -1.0, 0.0]
    assert result["delta_position"].tolist() == [0.0, 1.0, -2.0, 0.0, 1.0]
    assert result["turnover"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]
    assert result["trade_event"].tolist() == [False, True, True, False, True]


def test_run_backtest_supports_longer_execution_delay_deterministically() -> None:
    df = _backtest_frame()
    config = ExecutionConfig(
        enabled=False,
        execution_delay=2,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    result = run_backtest(df, config)

    assert result["executed_signal"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert result["strategy_return"].tolist() == pytest.approx([0.0, 0.0, 0.03, -0.01])


def test_run_backtest_raises_when_signal_column_is_missing() -> None:
    df = _backtest_frame().drop(columns=["signal"])

    with pytest.raises(ValueError, match="must include a 'signal' column"):
        run_backtest(df)


def test_run_backtest_raises_when_supported_return_column_is_missing() -> None:
    df = _backtest_frame().drop(columns=["feature_ret_1d"])

    with pytest.raises(ValueError, match="Run failed: missing returns"):
        run_backtest(df)


def test_run_backtest_raises_when_return_column_has_no_usable_values() -> None:
    df = _backtest_frame()
    df["feature_ret_1d"] = pd.Series([pd.NA] * len(df), dtype="Float64")

    with pytest.raises(ValueError, match="contains no usable values"):
        run_backtest(df)


def test_run_backtest_rejects_same_bar_execution_if_positions_are_not_shifted() -> None:
    df = _backtest_frame()

    from src.research.integrity import validate_research_integrity

    with pytest.raises(ValueError, match="same_bar_execution"):
        validate_research_integrity(df, df["signal"], positions=df["signal"].astype("float64"))
