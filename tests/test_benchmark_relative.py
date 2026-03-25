from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.metrics import compute_benchmark_relative_metrics
from src.research.strategy_qa import generate_strategy_qa_summary


def _results_frame(
    strategy_returns: list[float],
    *,
    signal: list[int] | None = None,
    symbol: str = "AAPL",
) -> pd.DataFrame:
    periods = len(strategy_returns)
    signals = signal if signal is not None else [1] * periods
    ts = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * periods, dtype="string"),
            "ts_utc": ts,
            "timeframe": pd.Series(["1D"] * periods, dtype="string"),
            "signal": pd.Series(signals, dtype="int64"),
            "strategy_return": pd.Series(strategy_returns, dtype="float64"),
        }
    )
    frame["equity_curve"] = (1.0 + frame["strategy_return"]).cumprod()
    return frame


def test_compute_benchmark_relative_metrics_calculates_excess_return_and_correlation() -> None:
    strategy_returns = [0.0, 0.02, 0.01, -0.01]
    benchmark_returns = [0.0, 0.01, 0.005, -0.005]
    strategy_results = _results_frame(strategy_returns, signal=[1, 1, 1, 0])
    benchmark_results = _results_frame(benchmark_returns, signal=[1, 1, 1, 1])

    metrics = compute_benchmark_relative_metrics(strategy_results, benchmark_results)
    expected_benchmark_total_return = (1.0 + pd.Series(benchmark_returns, dtype="float64")).prod() - 1.0
    expected_strategy_total_return = (1.0 + pd.Series(strategy_returns, dtype="float64")).prod() - 1.0

    assert metrics["benchmark_total_return"] == pytest.approx(expected_benchmark_total_return)
    assert metrics["excess_return"] == pytest.approx(expected_strategy_total_return - expected_benchmark_total_return)
    assert metrics["benchmark_correlation"] == pytest.approx(1.0)
    assert metrics["relative_drawdown"] == pytest.approx(0.0050000000000000044)


def test_compute_benchmark_relative_metrics_flags_high_correlation_and_low_excess() -> None:
    strategy_results = _results_frame([0.0, 0.01, -0.01, 0.01], signal=[1, 1, 1, 1])
    benchmark_results = _results_frame([0.0, 0.01, -0.01, 0.01], signal=[1, 1, 1, 1])

    metrics = compute_benchmark_relative_metrics(strategy_results, benchmark_results)

    assert metrics["plausibility_flags"] == {
        "high_benchmark_correlation": True,
        "low_excess_return": True,
        "high_turnover_low_edge": False,
        "beta_dominated_strategy": False,
    }


def test_compute_benchmark_relative_metrics_flags_high_turnover_low_edge() -> None:
    strategy_results = _results_frame([0.0, 0.01, -0.01, 0.01, -0.01], signal=[1, -1, 1, -1, 1])
    benchmark_results = _results_frame([0.0, 0.009, -0.009, 0.009, -0.009], signal=[1, 1, 1, 1, 1])

    metrics = compute_benchmark_relative_metrics(strategy_results, benchmark_results)

    assert metrics["plausibility_flags"]["high_turnover_low_edge"] is True


def test_compute_benchmark_relative_metrics_flags_beta_dominated_strategy() -> None:
    strategy_results = _results_frame([0.0, 0.10, 0.10, 0.05], signal=[1, 1, 1, 1])
    benchmark_results = _results_frame([0.0, 0.09, 0.09, 0.045], signal=[1, 1, 1, 1])

    metrics = compute_benchmark_relative_metrics(strategy_results, benchmark_results)

    assert metrics["plausibility_flags"]["high_benchmark_correlation"] is True
    assert metrics["plausibility_flags"]["beta_dominated_strategy"] is True


def test_generate_strategy_qa_summary_includes_relative_metrics_and_flags() -> None:
    df = _results_frame([0.0, 0.10, 0.10, 0.05], signal=[1, 1, 1, 1])
    metrics = {
        "total_return": 0.2705,
        "sharpe_ratio": 1.4,
        "max_drawdown": 0.0,
        "benchmark_total_return": 0.233605,
        "excess_return": 0.036895,
        "benchmark_correlation": 0.99,
        "relative_drawdown": 0.0,
        "plausibility_flags": {
            "high_benchmark_correlation": True,
            "low_excess_return": False,
            "high_turnover_low_edge": False,
            "beta_dominated_strategy": True,
        },
    }
    diagnostics = {
        "pct_long": 1.0,
        "pct_short": 0.0,
        "pct_flat": 0.0,
        "turnover": 0.0,
        "total_trades": 1,
        "flags": {
            "always_flat": False,
            "always_long": True,
            "always_short": False,
            "no_trades": False,
            "high_turnover": False,
        },
    }

    summary = generate_strategy_qa_summary(
        df,
        df["signal"],
        diagnostics,
        metrics,
        strategy_name="momentum_v1",
        run_id="relative-qa",
    )

    assert summary["relative"] == {
        "benchmark_return": pytest.approx(0.233605),
        "excess_return": pytest.approx(0.036895),
        "correlation": pytest.approx(0.99),
        "relative_drawdown": pytest.approx(0.0),
    }
    assert summary["flags"]["high_benchmark_correlation"] is True
    assert summary["flags"]["beta_dominated_strategy"] is True
