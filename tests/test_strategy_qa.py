from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.strategy_qa import generate_strategy_qa_summary


def _frame(
    signals: list[int],
    *,
    rows: int | None = None,
    include_returns: bool = True,
    include_equity: bool = True,
    symbol: str = "AAPL",
) -> pd.DataFrame:
    total_rows = rows if rows is not None else len(signals)
    signal_values = signals if rows is None else (signals + [signals[-1]] * max(0, rows - len(signals)))
    ts = pd.date_range("2025-01-01", periods=total_rows, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * total_rows, dtype="string"),
            "ts_utc": ts,
            "timeframe": pd.Series(["1D"] * total_rows, dtype="string"),
            "signal": pd.Series(signal_values[:total_rows], dtype="int64"),
        }
    )
    df.attrs["dataset"] = "features_daily"

    if include_returns:
        df["strategy_return"] = pd.Series([0.0] * total_rows, dtype="float64")
    if include_equity:
        df["equity_curve"] = pd.Series([1.0] * total_rows, dtype="float64")
    return df


def test_generate_strategy_qa_summary_passes_for_valid_run() -> None:
    df = _frame([0, 1, 1, 0] * 30, rows=120)
    df["strategy_return"] = pd.Series([0.0, 0.01, 0.01, -0.01] * 30, dtype="float64")
    df["equity_curve"] = (1.0 + df["strategy_return"]).cumprod()
    diagnostics = compute_signal_diagnostics(df["signal"], df)

    result = generate_strategy_qa_summary(
        df,
        df["signal"],
        diagnostics,
        {"total_return": 0.25, "sharpe_ratio": 1.2, "max_drawdown": 0.15},
        strategy_name="momentum_v1",
        run_id="abc123",
    )

    assert result == {
        "run_id": "abc123",
        "strategy_name": "momentum_v1",
        "dataset": "features_daily",
        "timeframe": "1D",
        "row_count": 120,
        "symbols_present": 1,
        "date_range": ["2025-01-01T00:00:00Z", "2025-04-30T00:00:00Z"],
        "input_validation": {},
        "signal": {
            "pct_long": 0.5,
            "pct_short": 0.0,
            "pct_flat": 0.5,
            "turnover": 0.5,
            "total_trades": 60,
        },
        "execution": {
            "valid_returns": True,
            "equity_curve_present": True,
        },
        "metrics": {
            "total_return": 0.25,
            "sharpe": 1.2,
            "max_drawdown": 0.15,
        },
        "relative": {
            "benchmark_return": None,
            "excess_return": None,
            "correlation": None,
            "relative_drawdown": None,
        },
        "flags": {
            "no_data": False,
            "degenerate_signal": False,
            "no_trades": False,
            "high_turnover": False,
            "low_data": False,
            "high_benchmark_correlation": False,
            "low_excess_return": False,
            "high_turnover_low_edge": False,
            "beta_dominated_strategy": False,
        },
        "overall_status": "pass",
    }


def test_generate_strategy_qa_summary_warns_for_degenerate_signal() -> None:
    df = _frame([1] * 120)
    diagnostics = compute_signal_diagnostics(df["signal"], df)

    result = generate_strategy_qa_summary(
        df,
        df["signal"],
        diagnostics,
        {},
        strategy_name="buy_and_hold_v1",
        run_id="run-1",
    )

    assert result["flags"]["degenerate_signal"] is True
    assert result["overall_status"] == "warn"


def test_generate_strategy_qa_summary_warns_for_no_trades() -> None:
    df = _frame([0] * 120)
    diagnostics = compute_signal_diagnostics(df["signal"], df)

    result = generate_strategy_qa_summary(
        df,
        df["signal"],
        diagnostics,
        {},
        strategy_name="flat_v1",
        run_id="run-2",
    )

    assert result["flags"]["no_trades"] is True
    assert result["overall_status"] == "warn"


def test_generate_strategy_qa_summary_warns_for_low_data() -> None:
    df = _frame([0, 1, 0, -1], rows=20)
    diagnostics = compute_signal_diagnostics(df["signal"], df)

    result = generate_strategy_qa_summary(
        df,
        df["signal"],
        diagnostics,
        {},
        strategy_name="small_sample_v1",
        run_id="run-3",
    )

    assert result["flags"]["low_data"] is True
    assert result["overall_status"] == "warn"


def test_generate_strategy_qa_summary_fails_for_missing_returns() -> None:
    df = _frame([0, 1, 0, -1] * 30, include_returns=False)
    diagnostics = compute_signal_diagnostics(df["signal"], df)

    result = generate_strategy_qa_summary(
        df,
        df["signal"],
        diagnostics,
        {},
        strategy_name="broken_v1",
        run_id="run-4",
    )

    assert result["execution"]["valid_returns"] is False
    assert result["overall_status"] == "fail"


def test_generate_strategy_qa_summary_aggregates_diagnostics_and_metrics() -> None:
    df = _frame([0, 1, -1, 0] * 30, rows=120)

    result = generate_strategy_qa_summary(
        df,
        df["signal"],
        {
            "pct_long": 0.25,
            "pct_short": 0.25,
            "pct_flat": 0.5,
            "turnover": 0.75,
            "total_trades": 90,
            "flags": {
                "always_flat": False,
                "always_long": False,
                "always_short": False,
                "no_trades": False,
                "high_turnover": True,
            },
        },
        {"total_return": 0.12, "sharpe_ratio": 0.8, "max_drawdown": 0.2},
        strategy_name="diagnostic_v1",
        run_id="run-5",
    )

    assert result["signal"] == {
        "pct_long": 0.25,
        "pct_short": 0.25,
        "pct_flat": 0.5,
        "turnover": 0.75,
        "total_trades": 90,
    }
    assert result["metrics"] == {
        "total_return": 0.12,
        "sharpe": 0.8,
        "max_drawdown": 0.2,
    }
    assert result["relative"] == {
        "benchmark_return": None,
        "excess_return": None,
        "correlation": None,
        "relative_drawdown": None,
    }
    assert result["flags"]["high_turnover"] is True
    assert result["overall_status"] == "warn"
