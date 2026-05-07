from __future__ import annotations

import json
import math

import pandas as pd
import pytest
from scipy import stats

from src.research.experiment_tracker import save_experiment
from src.research.metrics import (
    MetricsAggregationError,
    MINUTE_PERIODS_PER_YEAR,
    RETURN_INFERENCE_STD_ATOL,
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    compute_autocorr_lag1,
    compute_benchmark_relative_metrics,
    compute_confidence_interval,
    compute_effective_sample_size,
    compute_hit_rate_p_value,
    compute_performance_metrics,
    compute_p_value,
    compute_split_mean_diff,
    compute_split_mean_diff_p_value,
    compute_split_period_diagnostics,
    compute_t_statistic,
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


def test_return_inference_metrics_match_scipy_for_positive_returns() -> None:
    strategy_return = pd.Series([0.01, 0.02, -0.005, 0.015, 0.03], dtype="float64")
    sample_size = len(strategy_return)
    sample_std = strategy_return.std(ddof=1)
    expected_t = strategy_return.mean() / (sample_std / math.sqrt(sample_size))
    expected_p = 2.0 * (1.0 - stats.t.cdf(abs(expected_t), df=sample_size - 1))
    t_crit = stats.t.ppf(0.975, df=sample_size - 1)
    margin = t_crit * sample_std / math.sqrt(sample_size)

    lower, upper = compute_confidence_interval(strategy_return)

    assert compute_t_statistic(strategy_return) == pytest.approx(expected_t)
    assert compute_p_value(strategy_return) == pytest.approx(expected_p)
    assert 0.0 <= compute_p_value(strategy_return) <= 1.0
    assert lower == pytest.approx(strategy_return.mean() - margin)
    assert upper == pytest.approx(strategy_return.mean() + margin)
    assert lower <= upper


def test_return_inference_metrics_match_scipy_for_negative_returns() -> None:
    strategy_return = pd.Series([-0.01, -0.02, 0.005, -0.015, -0.03], dtype="float64")
    sample_size = len(strategy_return)
    sample_std = strategy_return.std(ddof=1)
    expected_t = strategy_return.mean() / (sample_std / math.sqrt(sample_size))
    expected_p = 2.0 * (1.0 - stats.t.cdf(abs(expected_t), df=sample_size - 1))

    lower, upper = compute_confidence_interval(strategy_return)

    assert compute_t_statistic(strategy_return) == pytest.approx(expected_t)
    assert compute_t_statistic(strategy_return) < 0.0
    assert compute_p_value(strategy_return) == pytest.approx(expected_p)
    assert 0.0 <= compute_p_value(strategy_return) <= 1.0
    assert lower <= upper


def test_return_inference_metrics_handle_zero_single_and_constant_streams() -> None:
    zero_returns = pd.Series([0.0, 0.0, 0.0], dtype="float64")
    single_return = pd.Series([0.01], dtype="float64")
    constant_positive = pd.Series([0.01, 0.01, 0.01], dtype="float64")
    constant_negative = pd.Series([-0.01, -0.01, -0.01], dtype="float64")

    assert compute_t_statistic(zero_returns) is None
    assert compute_p_value(zero_returns) is None
    assert compute_confidence_interval(zero_returns) == (0.0, 0.0)
    assert compute_t_statistic(single_return) is None
    assert compute_p_value(single_return) is None
    assert compute_confidence_interval(single_return) == (None, None)
    assert compute_t_statistic(constant_positive) is None
    assert compute_p_value(constant_positive) is None
    assert compute_confidence_interval(constant_positive) == pytest.approx((0.01, 0.01))
    assert compute_t_statistic(constant_negative) is None
    assert compute_p_value(constant_negative) is None
    assert compute_confidence_interval(constant_negative) == pytest.approx((-0.01, -0.01))


def test_return_inference_metrics_treat_near_constant_positive_stream_as_degenerate() -> None:
    strategy_return = pd.Series([0.01, 0.0100000000005, 0.01, 0.01], dtype="float64")

    assert 0.0 < strategy_return.std(ddof=1) < RETURN_INFERENCE_STD_ATOL
    assert compute_t_statistic(strategy_return) is None
    assert compute_p_value(strategy_return) is None
    assert compute_confidence_interval(strategy_return) == pytest.approx((0.01, 0.01))


def test_return_inference_metrics_treat_near_constant_negative_stream_as_degenerate() -> None:
    strategy_return = pd.Series([-0.01, -0.0100000000005, -0.01, -0.01], dtype="float64")
    expected_mean = float(strategy_return.mean())

    assert 0.0 < strategy_return.std(ddof=1) < RETURN_INFERENCE_STD_ATOL
    assert compute_t_statistic(strategy_return) is None
    assert compute_p_value(strategy_return) is None
    assert compute_confidence_interval(strategy_return) == pytest.approx(
        (expected_mean, expected_mean)
    )


def test_return_inference_metrics_drop_nan_contaminated_returns() -> None:
    contaminated = pd.Series([0.01, float("nan"), 0.02, -0.005, float("nan"), 0.015], dtype="float64")
    clean = contaminated.dropna()

    assert compute_t_statistic(contaminated) == pytest.approx(compute_t_statistic(clean))
    assert compute_p_value(contaminated) == pytest.approx(compute_p_value(clean))
    assert compute_confidence_interval(contaminated) == pytest.approx(compute_confidence_interval(clean))


def test_compute_autocorr_lag1_returns_none_for_empty_single_or_invalid_streams() -> None:
    assert compute_autocorr_lag1(pd.Series(dtype="float64")) is None
    assert compute_autocorr_lag1(pd.Series([0.01], dtype="float64")) is None
    assert compute_autocorr_lag1(
        pd.Series([float("nan"), float("inf"), -float("inf")])
    ) is None


def test_compute_autocorr_lag1_returns_none_for_constant_or_near_constant_streams() -> None:
    assert compute_autocorr_lag1(pd.Series([0.01, 0.01, 0.01], dtype="float64")) is None
    assert compute_autocorr_lag1(
        pd.Series([0.01, 0.0100000000005, 0.01], dtype="float64")
    ) is None


def test_compute_autocorr_lag1_handles_positive_negative_and_mixed_streams() -> None:
    positive = compute_autocorr_lag1(
        pd.Series([0.0, 0.01, 0.02, 0.03, 0.04], dtype="float64")
    )
    negative = compute_autocorr_lag1(
        pd.Series([0.01, -0.01, 0.01, -0.01, 0.01], dtype="float64")
    )
    mixed = compute_autocorr_lag1(pd.Series([0.01, 0.0, -0.01, 0.0], dtype="float64"))

    assert positive is not None and positive > 0.0
    assert negative is not None and negative < 0.0
    assert mixed is not None
    for value in (positive, negative, mixed):
        assert value is not None
        assert math.isfinite(value)
        assert -1.0 <= value <= 1.0
        json.dumps({"autocorr_lag1": value}, allow_nan=False)


def test_compute_effective_sample_size_handles_undefined_zero_positive_and_negative_autocorrelation() -> None:
    assert compute_effective_sample_size(pd.Series([0.01], dtype="float64")) is None

    positive = compute_effective_sample_size(pd.Series([0.0, 0.01, 0.02, 0.03, 0.04], dtype="float64"))
    zero = compute_effective_sample_size(pd.Series([0.01, 0.0, -0.01, 0.0], dtype="float64"))
    negative = compute_effective_sample_size(pd.Series([0.01, -0.01, 0.01, -0.01, 0.01], dtype="float64"))

    assert positive is not None and 0.0 <= positive < 5.0
    assert zero == pytest.approx(4.0)
    assert negative == pytest.approx(5.0)
    for value in (positive, zero, negative):
        assert value is not None
        assert math.isfinite(value)
        json.dumps({"effective_n": value}, allow_nan=False)


def test_compute_effective_sample_size_handles_high_positive_autocorrelation_safely() -> None:
    high_positive = compute_effective_sample_size(pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64"))

    assert high_positive is not None
    assert 0.0 <= high_positive <= 5.0


def test_compute_split_period_diagnostics_return_none_for_empty_single_or_invalid_streams() -> None:
    assert compute_split_period_diagnostics(pd.Series(dtype="float64")) == {
        "split_mean_diff": None,
        "split_mean_diff_p": None,
    }
    assert compute_split_period_diagnostics(pd.Series([0.01], dtype="float64")) == {
        "split_mean_diff": None,
        "split_mean_diff_p": None,
    }
    assert compute_split_period_diagnostics(pd.Series([float("nan"), float("inf"), -float("inf")])) == {
        "split_mean_diff": None,
        "split_mean_diff_p": None,
    }


def test_compute_split_period_diagnostics_require_two_observations_per_half() -> None:
    diagnostics = compute_split_period_diagnostics(pd.Series([0.01, 0.02, -0.01], dtype="float64"))

    assert diagnostics["split_mean_diff"] is None
    assert diagnostics["split_mean_diff_p"] is None


def test_compute_split_period_diagnostics_stable_series_is_json_safe() -> None:
    returns = pd.Series([0.010, 0.012, 0.009, 0.011, 0.010, 0.012], dtype="float64")
    diagnostics = compute_split_period_diagnostics(returns)

    assert diagnostics["split_mean_diff"] == pytest.approx(0.0, abs=0.002)
    assert diagnostics["split_mean_diff_p"] is not None
    assert 0.0 <= diagnostics["split_mean_diff_p"] <= 1.0
    json.dumps(diagnostics, allow_nan=False, sort_keys=True)


def test_compute_split_period_diagnostics_drifted_series_has_signed_difference() -> None:
    returns = pd.Series([0.03, 0.02, 0.04, -0.02, -0.03, -0.01], dtype="float64")
    diagnostics = compute_split_period_diagnostics(returns)

    assert diagnostics["split_mean_diff"] is not None
    assert diagnostics["split_mean_diff"] > 0.0
    assert diagnostics["split_mean_diff_p"] is not None
    assert 0.0 <= diagnostics["split_mean_diff_p"] <= 1.0


def test_compute_split_period_diagnostics_keeps_finite_diff_when_welch_test_is_degenerate() -> None:
    diagnostics = compute_split_period_diagnostics(pd.Series([0.01, 0.01, 0.02, 0.02], dtype="float64"))

    assert diagnostics["split_mean_diff"] == pytest.approx(-0.01)
    assert diagnostics["split_mean_diff_p"] is None
    json.dumps(diagnostics, allow_nan=False, sort_keys=True)


def test_compute_split_period_diagnostics_matches_scipy_welch_test_for_unequal_variance() -> None:
    returns = pd.Series([0.01, 0.03, -0.02, 0.02, -0.04, 0.05, 0.00, 0.01], dtype="float64")
    first_half = returns.iloc[:4]
    second_half = returns.iloc[4:]
    expected = stats.ttest_ind(first_half, second_half, equal_var=False, nan_policy="omit")

    diagnostics = compute_split_period_diagnostics(returns)

    assert diagnostics["split_mean_diff"] == pytest.approx(first_half.mean() - second_half.mean())
    assert diagnostics["split_mean_diff_p"] == pytest.approx(expected.pvalue)


def test_compute_split_period_diagnostics_filters_non_finite_values_before_splitting() -> None:
    contaminated = pd.Series(
        [0.01, float("nan"), 0.02, float("inf"), -0.01, -float("inf"), -0.02],
        dtype="float64",
    )
    clean = pd.Series([0.01, 0.02, -0.01, -0.02], dtype="float64")

    assert compute_split_period_diagnostics(contaminated) == pytest.approx(
        compute_split_period_diagnostics(clean)
    )
    assert compute_split_mean_diff(contaminated) == pytest.approx(0.03)
    assert compute_split_mean_diff_p_value(contaminated) == pytest.approx(
        compute_split_period_diagnostics(clean)["split_mean_diff_p"]
    )


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


def test_compute_hit_rate_p_value_returns_none_for_zero_valid_trades() -> None:
    assert compute_hit_rate_p_value(pd.Series(dtype="float64")) is None
    assert compute_hit_rate_p_value(pd.Series([float("nan"), float("inf"), -float("inf")])) is None


@pytest.mark.parametrize(
    ("trade_returns", "wins", "total"),
    [
        ([0.01, 0.02, 0.03, 0.04, 0.05, -0.01, -0.02, -0.03, -0.04, -0.05], 5, 10),
        ([0.01] * 13 + [-0.01] * 7, 13, 20),
        ([0.01, 0.02] + [-0.01] * 8, 2, 10),
    ],
)
def test_compute_hit_rate_p_value_matches_scipy_binomtest(
    trade_returns: list[float],
    wins: int,
    total: int,
) -> None:
    expected = stats.binomtest(wins, total, p=0.5, alternative="greater").pvalue

    result = compute_hit_rate_p_value(pd.Series(trade_returns, dtype="float64"))

    assert result == pytest.approx(expected)
    assert result is not None
    assert 0.0 <= result <= 1.0
    json.dumps({"hit_rate_p_value": result}, allow_nan=False)


def test_compute_hit_rate_p_value_counts_zero_returns_as_non_winning_trades() -> None:
    trade_returns = pd.Series([0.01, 0.0, -0.01, 0.0], dtype="float64")
    expected = stats.binomtest(1, 4, p=0.5, alternative="greater").pvalue

    assert compute_hit_rate_p_value(trade_returns) == pytest.approx(expected)


def test_compute_hit_rate_p_value_is_deterministic() -> None:
    trade_returns = pd.Series([0.02, -0.01, 0.03, 0.0, 0.04], dtype="float64")

    assert compute_hit_rate_p_value(trade_returns) == compute_hit_rate_p_value(trade_returns)


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
    assert {
        "t_stat",
        "p_value",
        "conf_int_lower",
        "conf_int_upper",
        "autocorr_lag1",
        "effective_n",
        "split_mean_diff",
        "split_mean_diff_p",
    }.issubset(metrics)
    assert metrics["conf_int_lower"] <= metrics["conf_int_upper"]
    assert metrics["hit_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["hit_rate_p_value"] == pytest.approx(
        stats.binomtest(2, 3, p=0.5, alternative="greater").pvalue
    )
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
    assert empty_metrics["t_stat"] is None
    assert empty_metrics["p_value"] is None
    assert empty_metrics["conf_int_lower"] is None
    assert empty_metrics["conf_int_upper"] is None
    assert empty_metrics["autocorr_lag1"] is None
    assert empty_metrics["effective_n"] is None
    assert empty_metrics["split_mean_diff"] is None
    assert empty_metrics["split_mean_diff_p"] is None
    assert empty_metrics["hit_rate_p_value"] is None
    assert empty_metrics["profit_factor"] == 0.0
    assert empty_metrics["exposure_pct"] == 0.0
    assert flat_metrics["annualized_volatility"] == 0.0
    assert flat_metrics["sharpe_ratio"] == 0.0
    assert flat_metrics["t_stat"] is None
    assert flat_metrics["p_value"] is None
    assert flat_metrics["autocorr_lag1"] is None
    assert flat_metrics["effective_n"] is None
    assert flat_metrics["split_mean_diff"] is None
    assert flat_metrics["split_mean_diff_p"] is None
    assert flat_metrics["hit_rate"] == 0.0
    assert flat_metrics["hit_rate_p_value"] is None


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
    assert metrics["hit_rate_p_value"] is None
    assert metrics["profit_factor"] == 0.0
    assert metrics["exposure_pct"] == pytest.approx(75.0)


def test_compute_performance_metrics_aggregates_multi_symbol_returns_by_timestamp_mean() -> None:
    multi_symbol = pd.DataFrame(
        {
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "timeframe": ["1d"] * 4,
            "signal": [1.0, 1.0, 1.0, 1.0],
            "strategy_return": [0.10, 0.10, -0.05, -0.05],
        }
    )
    single_stream = pd.DataFrame(
        {
            "ts_utc": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
            "timeframe": ["1d", "1d"],
            "signal": [1.0, 1.0],
            "strategy_return": [0.10, -0.05],
        }
    )

    multi_metrics = compute_performance_metrics(multi_symbol)
    single_metrics = compute_performance_metrics(single_stream)

    assert multi_metrics["total_return"] == pytest.approx(single_metrics["total_return"])
    assert multi_metrics["max_drawdown"] == pytest.approx(single_metrics["max_drawdown"])
    assert multi_metrics["win_rate"] == pytest.approx(single_metrics["win_rate"])
    assert multi_metrics["total_return"] == pytest.approx(0.045)


def test_compute_performance_metrics_does_not_scale_with_symbol_count() -> None:
    two_symbols = pd.DataFrame(
        {
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "timeframe": ["1d"] * 4,
            "signal": [1.0, 1.0, 1.0, 1.0],
            "strategy_return": [0.10, 0.10, -0.10, -0.10],
        }
    )
    summed_semantics_total = cumulative_return(pd.Series([0.20, -0.20], dtype="float64"))

    metrics = compute_performance_metrics(two_symbols)

    assert metrics["total_return"] == pytest.approx(-0.01)
    assert metrics["total_return"] != pytest.approx(summed_semantics_total)


def test_compute_benchmark_relative_metrics_aggregates_multi_symbol_strategy_and_benchmark() -> None:
    strategy = pd.DataFrame(
        {
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "strategy_return": [0.04, 0.02, 0.00, 0.02],
        }
    )
    benchmark = pd.DataFrame(
        {
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "strategy_return": [0.02, 0.02, 0.01, 0.01],
        }
    )

    relative = compute_benchmark_relative_metrics(strategy, benchmark)

    assert relative["benchmark_total_return"] == pytest.approx((1.02 * 1.01) - 1.0)
    assert relative["excess_return"] == pytest.approx(((1.03 * 1.01) - 1.0) - ((1.02 * 1.01) - 1.0))
    assert relative["benchmark_correlation"] == pytest.approx(1.0)


def test_compute_benchmark_relative_metrics_rejects_single_symbol_benchmark_for_multi_symbol_strategy() -> None:
    strategy = pd.DataFrame(
        {
            "ts_utc": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "symbol": ["AAPL", "MSFT"],
            "strategy_return": [0.01, 0.02],
        }
    )
    benchmark = pd.DataFrame(
        {
            "ts_utc": ["2025-01-01T00:00:00Z"],
            "symbol": ["SPY"],
            "strategy_return": [0.01],
        }
    )

    with pytest.raises(MetricsAggregationError, match="same symbol universe"):
        compute_benchmark_relative_metrics(strategy, benchmark)


def test_compute_performance_metrics_is_deterministic_for_multi_symbol_inputs() -> None:
    multi_symbol = pd.DataFrame(
        {
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "timeframe": ["1d"] * 4,
            "signal": [1.0, 1.0, 1.0, 1.0],
            "strategy_return": [0.03, 0.01, -0.02, 0.00],
        }
    )

    first = compute_performance_metrics(multi_symbol)
    second = compute_performance_metrics(multi_symbol)

    assert first == second


def test_metrics_json_artifact_includes_inference_fields_and_json_safe_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    results_df = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
            "timeframe": ["1d"] * 5,
            "signal": [0.0, 1.0, 1.0, -1.0, 0.0],
            "strategy_return": [0.0, 0.01, 0.02, -0.005, 0.015],
            "equity_curve": [1.0, 1.01, 1.0302, 1.025049, 1.040424735],
        }
    )
    metrics = compute_performance_metrics(results_df)

    experiment_dir = save_experiment(
        "inference_metrics",
        results_df,
        metrics,
        {"strategy_name": "inference_metrics", "dataset": "unit"},
    )
    metrics_payload = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    serialized = json.dumps(metrics_payload, allow_nan=False, sort_keys=True)

    assert serialized
    assert {
        "hit_rate",
        "hit_rate_p_value",
        "t_stat",
        "p_value",
        "conf_int_lower",
        "conf_int_upper",
        "autocorr_lag1",
        "effective_n",
        "split_mean_diff",
        "split_mean_diff_p",
    }.issubset(metrics_payload)
    assert {"cumulative_return", "sharpe_ratio", "max_drawdown", "win_rate"}.issubset(metrics_payload)
    assert metrics_payload["p_value"] == pytest.approx(compute_p_value(results_df["strategy_return"]))
    assert metrics_payload["conf_int_lower"] <= metrics_payload["conf_int_upper"]


def test_infer_periods_per_year_supports_minute_timeframes() -> None:
    minute_results = pd.DataFrame(
        {
            "timeframe": ["1m", "1m"],
            "strategy_return": [0.0, 0.001],
        }
    )

    assert infer_periods_per_year(minute_results) == MINUTE_PERIODS_PER_YEAR
