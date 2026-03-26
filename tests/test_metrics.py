from __future__ import annotations

import math

import pandas as pd
import pytest

from src.research.metrics import (
    MINUTE_PERIODS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    compute_performance_metrics,
    cumulative_return,
    exposure_pct,
    hit_rate,
    infer_periods_per_year,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    turnover,
    volatility,
    win_rate,
)


def _strategy_returns() -> pd.Series:
    return pd.Series(
        [0.10, -0.05, 0.02, -0.03, 0.04],
        index=pd.Index(["row_a", "row_b", "row_c", "row_d", "row_e"], name="row_id"),
        name="strategy_return",
        dtype="float64",
    )


def _trade_metric_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timeframe": ["1d"] * 8,
            "signal": [1, 1, 0, -1, -1, 1, 0, 0],
            "strategy_return": [0.0, 0.10, -0.05, 0.0, 0.04, -0.02, -0.03, 0.0],
        }
    )


def test_total_return_alias_matches_cumulative_return() -> None:
    strategy_return = _strategy_returns()

    assert total_return(strategy_return) == pytest.approx(cumulative_return(strategy_return))


def test_cumulative_return_computes_total_compounded_return() -> None:
    strategy_return = _strategy_returns()

    result = cumulative_return(strategy_return)

    assert result == pytest.approx(0.07527992)


def test_annualized_return_uses_observation_count_and_timeframe_factor() -> None:
    strategy_return = _strategy_returns()

    result = annualized_return(strategy_return, periods_per_year=TRADING_DAYS_PER_YEAR)
    expected = (1.0 + 0.07527992) ** (TRADING_DAYS_PER_YEAR / 5.0) - 1.0

    assert result == pytest.approx(expected)


def test_volatility_returns_sample_standard_deviation() -> None:
    strategy_return = _strategy_returns()

    result = volatility(strategy_return)

    assert result == pytest.approx(0.0594138031100518)


def test_annualized_volatility_scales_period_volatility() -> None:
    strategy_return = _strategy_returns()

    result = annualized_volatility(strategy_return, periods_per_year=TRADING_DAYS_PER_YEAR)
    expected = 0.0594138031100518 * math.sqrt(TRADING_DAYS_PER_YEAR)

    assert result == pytest.approx(expected)


def test_sharpe_ratio_uses_zero_risk_free_rate_and_annualized_mean_return() -> None:
    strategy_return = _strategy_returns()

    result = sharpe_ratio(strategy_return, periods_per_year=TRADING_DAYS_PER_YEAR)
    expected = strategy_return.mean() * TRADING_DAYS_PER_YEAR / (
        strategy_return.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    )

    assert result == pytest.approx(expected)


def test_sharpe_ratio_returns_zero_for_zero_volatility() -> None:
    strategy_return = pd.Series([0.01, 0.01, 0.01], dtype="float64")

    assert sharpe_ratio(strategy_return) == 0.0
    assert annualized_volatility(strategy_return) == 0.0


def test_max_drawdown_computes_largest_peak_to_trough_decline() -> None:
    strategy_return = _strategy_returns()

    result = max_drawdown(strategy_return)

    assert result == pytest.approx(0.06007)


def test_win_rate_counts_positive_return_periods() -> None:
    strategy_return = _strategy_returns()

    result = win_rate(strategy_return)

    assert result == pytest.approx(0.6)


def test_hit_rate_and_profit_factor_use_closed_trade_returns() -> None:
    trade_returns = pd.Series([0.045, 0.0192, -0.03], dtype="float64")

    assert hit_rate(trade_returns) == pytest.approx(2.0 / 3.0)
    assert profit_factor(trade_returns) == pytest.approx((0.045 + 0.0192) / 0.03)


def test_profit_factor_returns_none_when_closed_trades_have_no_losses() -> None:
    trade_returns = pd.Series([0.02, 0.03], dtype="float64")

    assert profit_factor(trade_returns) is None


def test_turnover_and_exposure_pct_use_executed_positions() -> None:
    positions = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0], dtype="float64")

    assert turnover(positions) == pytest.approx(0.75)
    assert exposure_pct(positions) == pytest.approx(62.5)


def test_compute_performance_metrics_includes_expanded_fields_with_known_trade_values() -> None:
    results_df = _trade_metric_results()

    metrics = compute_performance_metrics(results_df)

    assert metrics["total_return"] == pytest.approx(metrics["cumulative_return"])
    assert metrics["hit_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["profit_factor"] == pytest.approx((0.045 + 0.0192) / 0.03)
    assert metrics["turnover"] == pytest.approx(0.75)
    assert metrics["total_turnover"] == pytest.approx(6.0)
    assert metrics["average_turnover"] == pytest.approx(0.75)
    assert metrics["trade_count"] == pytest.approx(5.0)
    assert metrics["rebalance_count"] == pytest.approx(5.0)
    assert metrics["percent_periods_traded"] == pytest.approx(62.5)
    assert metrics["average_trade_size"] == pytest.approx(1.2)
    assert metrics["exposure_pct"] == pytest.approx(62.5)


def test_compute_performance_metrics_aggregates_execution_cost_attribution() -> None:
    results_df = pd.DataFrame(
        {
            "timeframe": ["1d"] * 4,
            "executed_signal": [0.0, 1.0, 1.0, -1.0],
            "strategy_return": [0.0, 0.01, 0.02, -0.03],
            "transaction_cost": [0.0, 0.001, 0.0, 0.002],
            "slippage_cost": [0.0, 0.0005, 0.0, 0.001],
            "execution_friction": [0.0, 0.0015, 0.0, 0.003],
        }
    )

    metrics = compute_performance_metrics(results_df)

    assert metrics["total_transaction_cost"] == pytest.approx(0.003)
    assert metrics["total_slippage_cost"] == pytest.approx(0.0015)
    assert metrics["total_execution_friction"] == pytest.approx(0.0045)
    assert metrics["average_execution_friction_per_trade"] == pytest.approx(0.00225)


def test_compute_performance_metrics_handles_empty_and_flat_inputs() -> None:
    empty_results = pd.DataFrame({"signal": pd.Series(dtype="float64"), "strategy_return": pd.Series(dtype="float64")})
    flat_results = pd.DataFrame(
        {
            "timeframe": ["1d"],
            "signal": [0.0],
            "strategy_return": [0.0],
        }
    )

    empty_metrics = compute_performance_metrics(empty_results)
    flat_metrics = compute_performance_metrics(flat_results)

    assert empty_metrics["total_return"] == 0.0
    assert empty_metrics["profit_factor"] == 0.0
    assert empty_metrics["exposure_pct"] == 0.0
    assert flat_metrics["annualized_volatility"] == 0.0
    assert flat_metrics["sharpe_ratio"] == 0.0
    assert flat_metrics["hit_rate"] == 0.0


def test_compute_performance_metrics_excludes_open_terminal_trade_from_trade_stats() -> None:
    results_df = pd.DataFrame(
        {
            "timeframe": ["1d"] * 4,
            "signal": [1, 1, 1, 1],
            "strategy_return": [0.0, 0.02, -0.01, 0.03],
        }
    )

    metrics = compute_performance_metrics(results_df)

    assert metrics["hit_rate"] == 0.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["exposure_pct"] == pytest.approx(75.0)


def test_infer_periods_per_year_supports_minute_timeframes() -> None:
    minute_results = pd.DataFrame(
        {
            "timeframe": ["1m", "1m"],
            "strategy_return": [0.0, 0.001],
        }
    )

    assert infer_periods_per_year(minute_results) == MINUTE_PERIODS_PER_YEAR
