from __future__ import annotations

from collections.abc import Iterable
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from src.research.turnover import compute_position_change_frame

TRADING_DAYS_PER_YEAR = 252
TRADING_MINUTES_PER_DAY = 390
MINUTE_PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY
DEFAULT_PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR
HIGH_BENCHMARK_CORRELATION_THRESHOLD = 0.9
LOW_EXCESS_RETURN_THRESHOLD = 0.02
HIGH_TURNOVER_THRESHOLD = 0.5
BETA_DOMINATED_RETURN_THRESHOLD = 0.2
RETURN_INFERENCE_STD_ATOL = 1e-12
AUTOCORR_DENOMINATOR_ATOL = 1e-12
SPLIT_MIN_HALF_OBSERVATIONS = 2
ROLLING_SHARPE_MIN_OBSERVATIONS = 12
ROLLING_SHARPE_SD_ATOL = 1e-12
METRICS_READINESS_FILENAME = "metrics_readiness.json"
DEFAULT_MINIMUM_EFFECTIVE_N = 30.0


class MetricsAggregationError(ValueError):
    """Raised when strategy returns cannot be aggregated into one series safely."""


def build_metrics_readiness_manifest(
    metrics: dict[str, Any],
    *,
    run_id: str | None = None,
    source_metrics_artifact: str = "metrics.json",
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic research-readiness manifest from metric outputs."""

    normalized_metrics = _json_safe_mapping(metrics)
    minimum_effective_n = _minimum_effective_n_threshold(thresholds)

    diagnostics = {
        "return_inference": {
            "t_stat": normalized_metrics.get("t_stat"),
            "p_value": normalized_metrics.get("p_value"),
            "conf_int_lower": normalized_metrics.get("conf_int_lower"),
            "conf_int_upper": normalized_metrics.get("conf_int_upper"),
        },
        "hit_rate": {
            "hit_rate": normalized_metrics.get("hit_rate"),
            "hit_rate_p_value": normalized_metrics.get("hit_rate_p_value"),
        },
        "serial_dependence": {
            "autocorr_lag1": normalized_metrics.get("autocorr_lag1"),
            "effective_n": normalized_metrics.get("effective_n"),
        },
        "split_period": {
            "split_mean_diff": normalized_metrics.get("split_mean_diff"),
            "split_mean_diff_p": normalized_metrics.get("split_mean_diff_p"),
        },
        "rolling_stability": {
            "rolling_sharpe_mean": normalized_metrics.get("rolling_sharpe_mean"),
            "rolling_sharpe_sd": normalized_metrics.get("rolling_sharpe_sd"),
            "sharpe_stability_ratio": normalized_metrics.get("sharpe_stability_ratio"),
        },
    }

    effective_n = _numeric_or_none(normalized_metrics.get("effective_n"))
    observed_count = _observed_period_count(normalized_metrics)
    sample_size_value = effective_n if effective_n is not None else observed_count
    checks = [
        _minimum_observations_check(sample_size_value, threshold=minimum_effective_n),
        _availability_check(
            "return_p_value_available",
            normalized_metrics.get("p_value") is not None,
            value=normalized_metrics.get("p_value"),
            message_pass="Return p-value is available.",
            message_warn="Return p-value is unavailable.",
        ),
        _availability_check(
            "hit_rate_p_value_available",
            normalized_metrics.get("hit_rate_p_value") is not None,
            value=normalized_metrics.get("hit_rate_p_value"),
            message_pass="Trade-level hit-rate p-value is available.",
            message_warn="Trade-level hit-rate p-value is unavailable.",
        ),
        _availability_check(
            "serial_dependence_available",
            normalized_metrics.get("autocorr_lag1") is not None and normalized_metrics.get("effective_n") is not None,
            value=normalized_metrics.get("effective_n"),
            message_pass="Serial-dependence diagnostics are available.",
            message_warn="Serial-dependence diagnostics are incomplete.",
        ),
        _availability_check(
            "split_period_available",
            normalized_metrics.get("split_mean_diff_p") is not None,
            value=normalized_metrics.get("split_mean_diff_p"),
            message_pass="Split-period p-value is available.",
            message_warn="Split-period p-value is unavailable.",
        ),
        _availability_check(
            "rolling_stability_available",
            normalized_metrics.get("rolling_sharpe_mean") is not None
            and normalized_metrics.get("rolling_sharpe_sd") is not None,
            value=normalized_metrics.get("rolling_sharpe_sd"),
            message_pass="Rolling Sharpe stability diagnostics are available.",
            message_warn="Rolling Sharpe stability diagnostics are incomplete.",
        ),
    ]
    summary = _readiness_summary(checks)

    return {
        "schema_version": 1,
        "status": _overall_readiness_status(checks),
        "run_id": run_id,
        "source_metrics_artifact": source_metrics_artifact,
        "diagnostics": diagnostics,
        "checks": checks,
        "summary": summary,
    }


def write_metrics_readiness_manifest(
    run_dir: Path,
    metrics: dict[str, Any],
    *,
    run_id: str | None = None,
    source_metrics_artifact: str = "metrics.json",
    thresholds: dict[str, Any] | None = None,
) -> Path:
    """Write ``metrics_readiness.json`` beside a metrics artifact."""

    manifest = build_metrics_readiness_manifest(
        metrics,
        run_id=run_id,
        source_metrics_artifact=source_metrics_artifact,
        thresholds=thresholds,
    )
    path = Path(run_dir) / METRICS_READINESS_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _normalized_returns(strategy_return: pd.Series) -> pd.Series:
    """Return a float series with missing values removed for stable metric calculations."""

    return strategy_return.dropna().astype("float64")


def cumulative_return(strategy_return: pd.Series) -> float:
    """
    Compute the total compounded return across the full strategy return series.

    Args:
        strategy_return: Period-by-period strategy returns.

    Returns:
        The compounded return over the full series.
    """

    returns = _normalized_returns(strategy_return)
    if returns.empty:
        return 0.0

    return float((1.0 + returns).prod() - 1.0)


def total_return(strategy_return: pd.Series) -> float:
    """
    Compute the cumulative return across the evaluation window.

    This is an alias for ``cumulative_return()`` so existing callers and new
    metric payloads stay numerically aligned.
    """

    return cumulative_return(strategy_return)


def volatility(strategy_return: pd.Series) -> float:
    """
    Compute the sample standard deviation of the strategy return series.

    Args:
        strategy_return: Period-by-period strategy returns.

    Returns:
        The sample standard deviation of the returns.
    """

    returns = _normalized_returns(strategy_return)
    if len(returns) < 2:
        return 0.0

    return float(returns.std())


def compute_t_statistic(strategy_return: pd.Series) -> float | None:
    """
    Compute the one-sample t-statistic for mean period return versus zero.

    Returns ``None`` when the statistic is undefined, including fewer than two
    finite observations or zero sample volatility.
    """

    inference = _return_inference_inputs(strategy_return)
    if inference is None:
        return None

    t_stat = inference["mean"] / inference["standard_error"]
    return _json_safe_float(t_stat)


def compute_p_value(strategy_return: pd.Series) -> float | None:
    """
    Compute the two-sided Student-t p-value for mean period return versus zero.

    Missing and non-finite returns are excluded before inference. Undefined
    cases return ``None`` so JSON artifacts never serialize NaN or infinity.
    """

    inference = _return_inference_inputs(strategy_return)
    if inference is None:
        return None

    t_stat = inference["mean"] / inference["standard_error"]
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=inference["degrees_of_freedom"]))
    bounded = min(max(p_value, 0.0), 1.0)
    return _json_safe_float(bounded)


def compute_confidence_interval(
    strategy_return: pd.Series,
    *,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    """
    Compute a Student-t confidence interval for mean period return.

    For zero-variance streams with at least two finite observations, the
    interval is the deterministic degenerate interval ``(mean, mean)``. For
    fewer than two finite observations, both bounds are ``None``.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    returns = _finite_returns(strategy_return)
    if len(returns) < 2:
        return (None, None)

    mean_return = float(returns.mean())
    sample_std = float(returns.std(ddof=1))
    if math.isclose(sample_std, 0.0, abs_tol=RETURN_INFERENCE_STD_ATOL):
        safe_mean = _json_safe_float(mean_return)
        return (safe_mean, safe_mean)

    standard_error = sample_std / math.sqrt(len(returns))
    alpha = 1.0 - confidence_level
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=len(returns) - 1)
    if not math.isfinite(t_crit):
        return (None, None)

    margin = float(t_crit * standard_error)
    lower = _json_safe_float(mean_return - margin)
    upper = _json_safe_float(mean_return + margin)
    return (lower, upper)


def compute_autocorr_lag1(strategy_return: pd.Series) -> float | None:
    """
    Compute lag-1 autocorrelation for finite period returns.

    Undefined streams return ``None`` so JSON artifacts never serialize NaN or
    infinity. Near-constant lagged vectors are treated as undefined.
    """

    returns = _finite_returns(strategy_return)
    if len(returns) < 2:
        return None

    previous_returns = returns.iloc[:-1].reset_index(drop=True)
    next_returns = returns.iloc[1:].reset_index(drop=True)
    previous_std = float(previous_returns.std(ddof=1))
    next_std = float(next_returns.std(ddof=1))
    if (
        not math.isfinite(previous_std)
        or not math.isfinite(next_std)
        or math.isclose(previous_std, 0.0, abs_tol=RETURN_INFERENCE_STD_ATOL)
        or math.isclose(next_std, 0.0, abs_tol=RETURN_INFERENCE_STD_ATOL)
    ):
        return None

    autocorr = previous_returns.corr(next_returns)
    safe_autocorr = _json_safe_float(float(autocorr))
    if safe_autocorr is None:
        return None
    if safe_autocorr > 1.0 and math.isclose(safe_autocorr, 1.0, abs_tol=AUTOCORR_DENOMINATOR_ATOL):
        return 1.0
    if safe_autocorr < -1.0 and math.isclose(safe_autocorr, -1.0, abs_tol=AUTOCORR_DENOMINATOR_ATOL):
        return -1.0
    return safe_autocorr


def compute_effective_sample_size(strategy_return: pd.Series) -> float | None:
    """
    Estimate conservative AR(1)-style effective sample size.

    Positive lag-1 autocorrelation reduces the estimate. Negative
    autocorrelation is capped at the observed finite sample size for
    conservative reporting.
    """

    returns = _finite_returns(strategy_return)
    sample_size = len(returns)
    rho = compute_autocorr_lag1(returns)
    if rho is None:
        return None
    if rho < 0.0:
        return float(sample_size)

    denominator = 1.0 + rho
    if denominator <= AUTOCORR_DENOMINATOR_ATOL:
        return None

    effective_n = sample_size * (1.0 - rho) / denominator
    bounded_effective_n = max(0.0, min(float(sample_size), effective_n))
    return _json_safe_float(bounded_effective_n)


def compute_split_period_diagnostics(strategy_return: pd.Series) -> dict[str, float | None]:
    """
    Compare first-half and second-half mean returns with a deterministic Welch test.

    Finite period returns are split by observation count after filtering invalid
    values. ``split_mean_diff`` is reported only when both halves contain at
    least two observations, matching the minimum sample convention for the
    two-sided Welch t-test p-value.
    """

    returns = _finite_returns(strategy_return).reset_index(drop=True)
    split_index = len(returns) // 2
    first_half = returns.iloc[:split_index]
    second_half = returns.iloc[split_index:]
    if len(first_half) < SPLIT_MIN_HALF_OBSERVATIONS or len(second_half) < SPLIT_MIN_HALF_OBSERVATIONS:
        return {
            "split_mean_diff": None,
            "split_mean_diff_p": None,
        }

    split_mean_diff = _json_safe_float(float(first_half.mean() - second_half.mean()))
    first_std = float(first_half.std(ddof=1))
    second_std = float(second_half.std(ddof=1))
    if (
        not math.isfinite(first_std)
        or not math.isfinite(second_std)
        or math.isclose(first_std, 0.0, abs_tol=RETURN_INFERENCE_STD_ATOL)
        or math.isclose(second_std, 0.0, abs_tol=RETURN_INFERENCE_STD_ATOL)
    ):
        return {
            "split_mean_diff": split_mean_diff,
            "split_mean_diff_p": None,
        }

    test_result = stats.ttest_ind(
        first_half,
        second_half,
        equal_var=False,
        nan_policy="omit",
    )
    split_mean_diff_p = _json_safe_float(float(test_result.pvalue))
    if split_mean_diff_p is not None:
        split_mean_diff_p = min(max(split_mean_diff_p, 0.0), 1.0)

    return {
        "split_mean_diff": split_mean_diff,
        "split_mean_diff_p": split_mean_diff_p,
    }


def compute_split_mean_diff(strategy_return: pd.Series) -> float | None:
    """Return first-half mean period return minus second-half mean period return."""

    return compute_split_period_diagnostics(strategy_return)["split_mean_diff"]


def compute_split_mean_diff_p_value(strategy_return: pd.Series) -> float | None:
    """Return the two-sided Welch t-test p-value for split-half mean stability."""

    return compute_split_period_diagnostics(strategy_return)["split_mean_diff_p"]


def compute_rolling_sharpe_values(
    strategy_return: pd.Series,
    *,
    window_size: int | None = None,
    min_periods: int | None = None,
    step_size: int | None = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> list[float]:
    """
    Compute deterministic sequential window Sharpe ratios for finite returns.

    Missing and non-finite returns are filtered before windowing. By default,
    windows are non-overlapping and full-sized, preserving return order while
    avoiding false precision from highly overlapping samples.
    """

    returns = _finite_returns(strategy_return).reset_index(drop=True)
    sample_size = len(returns)
    if window_size is None:
        window_size = _default_rolling_sharpe_window_size(sample_size)
    if window_size is None:
        return []
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if step_size is None:
        step_size = window_size
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    if min_periods is None:
        min_periods = window_size
    if min_periods <= 0:
        raise ValueError("min_periods must be positive.")
    if min_periods > window_size:
        raise ValueError("min_periods cannot exceed window_size.")

    window_sharpes: list[float] = []
    for start in range(0, sample_size, step_size):
        window = returns.iloc[start : start + window_size]
        if len(window) < min_periods:
            continue
        safe_sharpe = _json_safe_float(sharpe_ratio(window, periods_per_year=periods_per_year))
        if safe_sharpe is not None:
            window_sharpes.append(safe_sharpe)
    return window_sharpes


def compute_rolling_sharpe_diagnostics(
    strategy_return: pd.Series,
    *,
    window_size: int | None = None,
    min_periods: int | None = None,
    step_size: int | None = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, float | None]:
    """
    Summarize stability of Sharpe-like performance across sequential windows.

    Returns ``None`` diagnostics when fewer than two valid window Sharpe ratios
    are available. The stability ratio is mean window Sharpe divided by sample
    standard deviation and is undefined when the denominator is near zero.
    """

    empty = {
        "rolling_sharpe_mean": None,
        "rolling_sharpe_sd": None,
        "sharpe_stability_ratio": None,
    }
    window_sharpes = compute_rolling_sharpe_values(
        strategy_return,
        window_size=window_size,
        min_periods=min_periods,
        step_size=step_size,
        periods_per_year=periods_per_year,
    )
    if len(window_sharpes) < 2:
        return empty

    sharpe_series = pd.Series(window_sharpes, dtype="float64")
    rolling_sharpe_mean = _json_safe_float(float(sharpe_series.mean()))
    rolling_sharpe_sd = _json_safe_float(float(sharpe_series.std(ddof=1)))
    if rolling_sharpe_mean is None or rolling_sharpe_sd is None:
        return empty

    sharpe_stability_ratio = None
    if not math.isclose(rolling_sharpe_sd, 0.0, abs_tol=ROLLING_SHARPE_SD_ATOL):
        sharpe_stability_ratio = _json_safe_float(rolling_sharpe_mean / rolling_sharpe_sd)

    return {
        "rolling_sharpe_mean": rolling_sharpe_mean,
        "rolling_sharpe_sd": rolling_sharpe_sd,
        "sharpe_stability_ratio": sharpe_stability_ratio,
    }


def annualized_return(strategy_return: pd.Series, *, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    """
    Annualize cumulative return using the observed count of return observations.

    Args:
        strategy_return: Period-by-period strategy returns.
        periods_per_year: Deterministic annualization factor for the strategy timeframe.

    Returns:
        Compounded annualized return. Empty inputs return ``0.0``.
    """

    returns = _normalized_returns(strategy_return)
    if returns.empty:
        return 0.0

    total = cumulative_return(returns)
    growth = 1.0 + total
    if growth <= 0.0:
        return -1.0

    return float(growth ** (periods_per_year / len(returns)) - 1.0)


def annualized_volatility(
    strategy_return: pd.Series,
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """
    Compute annualized return volatility from sample period volatility.

    Args:
        strategy_return: Period-by-period strategy returns.
        periods_per_year: Deterministic annualization factor for the strategy timeframe.

    Returns:
        Annualized sample standard deviation of returns.
    """

    period_volatility = volatility(strategy_return)
    if period_volatility == 0.0:
        return 0.0

    return float(period_volatility * math.sqrt(periods_per_year))


def sharpe_ratio(strategy_return: pd.Series, *, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    """
    Compute an annualized Sharpe ratio assuming a zero risk-free rate.

    Args:
        strategy_return: Period-by-period strategy returns.
        periods_per_year: Deterministic annualization factor for the strategy timeframe.

    Returns:
        Annualized mean excess return divided by annualized volatility. Returns
        ``0.0`` when the ratio is undefined because the series is empty or has
        zero volatility.
    """

    returns = _normalized_returns(strategy_return)
    if returns.empty:
        return 0.0

    return_volatility = annualized_volatility(returns, periods_per_year=periods_per_year)
    if return_volatility == 0.0:
        return 0.0

    annualized_mean_excess_return = float(returns.mean() * periods_per_year)
    return float(annualized_mean_excess_return / return_volatility)


def max_drawdown(strategy_return: pd.Series) -> float:
    """
    Compute the maximum drawdown implied by the strategy return series.

    The function derives an equity curve by compounding returns from an initial
    value of ``1.0`` and reports the largest peak-to-trough decline as a
    positive fraction.

    Args:
        strategy_return: Period-by-period strategy returns.

    Returns:
        The maximum drawdown as a positive decimal fraction.
    """

    returns = _normalized_returns(strategy_return)
    if returns.empty:
        return 0.0

    equity_curve = (1.0 + returns).cumprod()
    drawdown = 1.0 - (equity_curve / equity_curve.cummax())
    return float(drawdown.max())


def win_rate(strategy_return: pd.Series) -> float:
    """
    Compute the fraction of periods with strictly positive strategy returns.

    Args:
        strategy_return: Period-by-period strategy returns.

    Returns:
        The proportion of observations greater than zero.
    """

    returns = _normalized_returns(strategy_return)
    if returns.empty:
        return 0.0

    return float((returns > 0.0).mean())


def hit_rate(trade_returns: pd.Series) -> float:
    """
    Compute the share of closed trades with positive compounded trade return.

    Args:
        trade_returns: One compounded return per closed trade.

    Returns:
        The proportion of trades with return greater than zero. Empty inputs return ``0.0``.
    """

    trades = _normalized_returns(trade_returns)
    if trades.empty:
        return 0.0

    return float((trades > 0.0).mean())


def compute_hit_rate_p_value(
    trade_returns: pd.Series,
    *,
    null_probability: float = 0.5,
    alternative: str = "greater",
) -> float | None:
    """
    Compute a one-sample binomial p-value for trade-level hit rate.

    Finite closed trade returns are counted as trials. Strictly positive returns
    are wins; zero-return trades are valid closed trades but not wins. Undefined
    empty samples return ``None`` so JSON artifacts never serialize non-finite
    values.
    """

    trades = pd.to_numeric(trade_returns, errors="coerce").dropna().astype("float64")
    finite_trades = trades.loc[trades.map(math.isfinite)]
    total = len(finite_trades)
    if total == 0:
        return None

    wins = int((finite_trades > 0.0).sum())
    p_value = stats.binomtest(
        wins,
        total,
        p=null_probability,
        alternative=alternative,
    ).pvalue
    safe_p_value = _json_safe_float(min(max(float(p_value), 0.0), 1.0))
    return safe_p_value


def profit_factor(trade_returns: pd.Series) -> float | None:
    """
    Compute gross profits divided by gross losses across closed trades.

    Args:
        trade_returns: One compounded return per closed trade.

    Returns:
        ``None`` when no losing trades exist, otherwise the ratio of summed gains
        to absolute summed losses. Empty inputs return ``0.0``.
    """

    trades = _normalized_returns(trade_returns)
    if trades.empty:
        return 0.0

    gross_profit = float(trades.loc[trades > 0.0].sum())
    gross_loss = float((-trades.loc[trades < 0.0]).sum())

    if gross_loss == 0.0:
        return None if gross_profit > 0.0 else 0.0

    return float(gross_profit / gross_loss)


def turnover(position: pd.Series, *, group_keys: pd.Series | None = None) -> float:
    """
    Compute average absolute position change per observation.

    Args:
        position: Executed position series applied to returns.

    Returns:
        Mean absolute change in position, including entries, exits, and flips.
    """

    positions = position.fillna(0.0).astype("float64")
    if positions.empty:
        return 0.0

    return float(compute_position_change_frame(positions, group_keys=group_keys)["turnover"].mean())


def exposure_pct(position: pd.Series) -> float:
    """
    Compute the percentage of observations spent with non-zero market exposure.

    Args:
        position: Executed position series applied to returns.

    Returns:
        Percentage of rows where absolute position is strictly greater than zero.
    """

    positions = position.fillna(0.0).astype("float64")
    if positions.empty:
        return 0.0

    return float((positions.ne(0.0).mean()) * 100.0)


def compute_performance_metrics(results_df: pd.DataFrame) -> dict[str, float | None]:
    """
    Build the standard serializable metric payload for a backtest result frame.

    The summary keeps legacy metric names for compatibility and adds expanded
    risk-adjusted, trade-level, and activity metrics. Annualization uses
    deterministic defaults of ``252`` periods per year for daily data and
    ``252 * 390`` for one-minute data. When timeframe cannot be inferred, the
    daily assumption is used.

    Args:
        results_df: Backtest results that include at least ``strategy_return`` and
            usually the original ``signal`` and ``timeframe`` columns.

    Returns:
        A JSON-serializable dictionary of metric values.
    """

    aggregated_returns_frame = aggregate_strategy_returns(results_df)
    strategy_return = aggregated_returns_frame["strategy_return"]
    periods_per_year = infer_periods_per_year(aggregated_returns_frame if not aggregated_returns_frame.empty else results_df)
    position = infer_position_series(results_df)
    group_keys = results_df["symbol"] if "symbol" in results_df.columns else None
    position_change = compute_position_change_frame(position, group_keys=group_keys)
    closed_trade_returns = extract_closed_trade_returns(results_df)
    transaction_cost = _optional_numeric_series(results_df, "transaction_cost")
    slippage_cost = _optional_numeric_series(results_df, "slippage_cost")
    execution_friction = _optional_numeric_series(results_df, "execution_friction")

    total = total_return(strategy_return)
    annual_return = annualized_return(strategy_return, periods_per_year=periods_per_year)
    annual_vol = annualized_volatility(strategy_return, periods_per_year=periods_per_year)
    period_vol = volatility(strategy_return)
    period_win_rate = win_rate(strategy_return)
    trade_count = int(position_change["trade_event"].sum())
    total_turnover = float(position_change["turnover"].sum())
    average_turnover = float(position_change["turnover"].mean()) if not position_change.empty else 0.0
    conf_int_lower, conf_int_upper = compute_confidence_interval(strategy_return)
    split_diagnostics = compute_split_period_diagnostics(strategy_return)
    rolling_sharpe_diagnostics = compute_rolling_sharpe_diagnostics(
        strategy_return,
        periods_per_year=periods_per_year,
    )

    return {
        "cumulative_return": total,
        "total_return": total,
        "volatility": period_vol,
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": sharpe_ratio(strategy_return, periods_per_year=periods_per_year),
        "t_stat": compute_t_statistic(strategy_return),
        "p_value": compute_p_value(strategy_return),
        "conf_int_lower": conf_int_lower,
        "conf_int_upper": conf_int_upper,
        "autocorr_lag1": compute_autocorr_lag1(strategy_return),
        "effective_n": compute_effective_sample_size(strategy_return),
        "split_mean_diff": split_diagnostics["split_mean_diff"],
        "split_mean_diff_p": split_diagnostics["split_mean_diff_p"],
        "rolling_sharpe_mean": rolling_sharpe_diagnostics["rolling_sharpe_mean"],
        "rolling_sharpe_sd": rolling_sharpe_diagnostics["rolling_sharpe_sd"],
        "sharpe_stability_ratio": rolling_sharpe_diagnostics["sharpe_stability_ratio"],
        "max_drawdown": max_drawdown(strategy_return),
        "win_rate": period_win_rate,
        "hit_rate": hit_rate(closed_trade_returns),
        "hit_rate_p_value": compute_hit_rate_p_value(closed_trade_returns),
        "profit_factor": profit_factor(closed_trade_returns),
        "turnover": average_turnover,
        "total_turnover": total_turnover,
        "average_turnover": average_turnover,
        "trade_count": float(trade_count),
        "rebalance_count": float(trade_count),
        "percent_periods_traded": float(position_change["trade_event"].mean() * 100.0) if not position_change.empty else 0.0,
        "average_trade_size": (total_turnover / trade_count) if trade_count else 0.0,
        "total_transaction_cost": float(transaction_cost.sum()),
        "total_slippage_cost": float(slippage_cost.sum()),
        "total_execution_friction": float(execution_friction.sum()),
        "average_execution_friction_per_trade": float(execution_friction.sum() / trade_count) if trade_count else 0.0,
        "exposure_pct": exposure_pct(position),
    }


def compute_benchmark_relative_metrics(
    results_df: pd.DataFrame,
    benchmark_results_df: pd.DataFrame,
) -> dict[str, float | dict[str, bool]]:
    """
    Compute lightweight benchmark-relative diagnostics for a strategy result frame.

    The strategy and benchmark are aligned on shared symbol/time columns when
    available so the correlation is based on comparable observations.
    """

    _validate_benchmark_comparability(results_df, benchmark_results_df)
    strategy_frame = aggregate_strategy_returns(results_df)
    benchmark_frame = aggregate_strategy_returns(benchmark_results_df)
    strategy_returns = strategy_frame["strategy_return"]
    benchmark_returns = benchmark_frame["strategy_return"]
    aligned = _align_return_frames(strategy_frame, benchmark_frame)
    strategy_total = total_return(strategy_returns)
    benchmark_total = total_return(benchmark_returns)
    excess = float(strategy_total - benchmark_total)
    correlation = benchmark_correlation(aligned["strategy_return"], aligned["benchmark_return"])
    relative_dd = float(max_drawdown(strategy_returns) - max_drawdown(benchmark_returns))
    plausibility_flags = evaluate_strategy_plausibility(
        total_return_value=strategy_total,
        excess_return=excess,
        benchmark_correlation_value=correlation,
        turnover_value=turnover(
            infer_position_series(results_df),
            group_keys=results_df["symbol"] if "symbol" in results_df.columns else None,
        ),
    )

    return {
        "benchmark_total_return": benchmark_total,
        "excess_return": excess,
        "benchmark_correlation": correlation,
        "relative_drawdown": relative_dd,
        "plausibility_flags": plausibility_flags,
    }


def benchmark_correlation(strategy_return: pd.Series, benchmark_return: pd.Series) -> float:
    """Compute a deterministic correlation between aligned strategy and benchmark returns."""

    strategy_returns = _normalized_returns(strategy_return)
    benchmark_returns = _normalized_returns(benchmark_return)
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0

    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0

    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


def evaluate_strategy_plausibility(
    *,
    total_return_value: float,
    excess_return: float,
    benchmark_correlation_value: float,
    turnover_value: float,
) -> dict[str, bool]:
    """Return deterministic warning-only plausibility flags for relative performance interpretation."""

    low_excess = abs(excess_return) <= LOW_EXCESS_RETURN_THRESHOLD
    high_correlation = benchmark_correlation_value > HIGH_BENCHMARK_CORRELATION_THRESHOLD
    return {
        "high_benchmark_correlation": high_correlation,
        "low_excess_return": low_excess,
        "high_turnover_low_edge": turnover_value >= HIGH_TURNOVER_THRESHOLD and low_excess,
        "beta_dominated_strategy": total_return_value >= BETA_DOMINATED_RETURN_THRESHOLD and high_correlation,
    }


def infer_periods_per_year(results_df: pd.DataFrame) -> int:
    """
    Infer the annualization factor from the backtest result frame.

    Daily data uses ``252`` periods per year. Minute data uses ``98,280``
    periods per year, assuming 252 trading days and 390 regular-session minutes
    per trading day. Unknown inputs fall back to the daily assumption.
    """

    timeframe_value = _first_non_empty(results_df.get("timeframe"))
    if timeframe_value is not None:
        normalized = timeframe_value.strip().lower()
        if normalized in {"1m", "1min", "1minute", "minute", "minutes"}:
            return MINUTE_PERIODS_PER_YEAR
        if normalized in {"1d", "1day", "day", "daily"}:
            return TRADING_DAYS_PER_YEAR

    inferred_from_columns = _infer_periods_per_year_from_columns(results_df.columns)
    if inferred_from_columns is not None:
        return inferred_from_columns

    if "ts_utc" in results_df.columns:
        timestamps = pd.to_datetime(results_df["ts_utc"], utc=True, errors="coerce").dropna()
        if len(timestamps) >= 2:
            median_delta = timestamps.sort_values().diff().dropna().median()
            if pd.notna(median_delta) and median_delta <= pd.Timedelta(minutes=5):
                return MINUTE_PERIODS_PER_YEAR

    return DEFAULT_PERIODS_PER_YEAR


def infer_position_series(results_df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct the executed position series used by the backtest.

    The research backtest applies the previous row's signal to the current row's
    return, so the executed position is ``signal.shift(1).fillna(0.0)``.
    """

    if "position" in results_df.columns:
        position = pd.to_numeric(results_df["position"], errors="coerce").fillna(0.0).astype("float64")
        position.name = "position"
        return position

    if "executed_signal" in results_df.columns:
        position = pd.to_numeric(results_df["executed_signal"], errors="coerce").fillna(0.0).astype("float64")
        position.name = "position"
        return position

    if "signal" not in results_df.columns:
        return pd.Series(0.0, index=results_df.index, dtype="float64", name="position")

    signal = pd.to_numeric(results_df["signal"], errors="coerce").fillna(0.0).astype("float64")
    if "symbol" in results_df.columns:
        position = (
            signal.groupby(results_df["symbol"].astype("string"), sort=False, dropna=False)
            .shift(1)
            .fillna(0.0)
            .astype("float64")
        )
    else:
        position = signal.shift(1).fillna(0.0).astype("float64")
    position.name = "position"
    return position


def _optional_numeric_series(results_df: pd.DataFrame, column: str) -> pd.Series:
    if column not in results_df.columns:
        return pd.Series(0.0, index=results_df.index, dtype="float64")
    return pd.to_numeric(results_df[column], errors="coerce").fillna(0.0).astype("float64")


def _finite_returns(strategy_return: pd.Series) -> pd.Series:
    returns = _normalized_returns(strategy_return)
    return returns.loc[returns.map(math.isfinite)]


def _return_inference_inputs(strategy_return: pd.Series) -> dict[str, float] | None:
    returns = _finite_returns(strategy_return)
    sample_size = len(returns)
    if sample_size < 2:
        return None

    sample_std = float(returns.std(ddof=1))
    if not math.isfinite(sample_std) or math.isclose(sample_std, 0.0, abs_tol=RETURN_INFERENCE_STD_ATOL):
        return None

    standard_error = sample_std / math.sqrt(sample_size)
    if standard_error == 0.0 or not math.isfinite(standard_error):
        return None

    return {
        "mean": float(returns.mean()),
        "standard_error": standard_error,
        "degrees_of_freedom": float(sample_size - 1),
    }


def _default_rolling_sharpe_window_size(sample_size: int) -> int | None:
    if sample_size >= TRADING_DAYS_PER_YEAR:
        return TRADING_DAYS_PER_YEAR
    if sample_size >= ROLLING_SHARPE_MIN_OBSERVATIONS:
        return max(4, sample_size // 3)
    return None


def _json_safe_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        return _json_safe_float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list | tuple):
        return [_json_safe_value(item) for item in value]
    return value


def _json_safe_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value[key]) for key in sorted(value, key=str)}


def _minimum_effective_n_threshold(thresholds: dict[str, Any] | None) -> float:
    raw_threshold = (thresholds or {}).get("minimum_effective_n", DEFAULT_MINIMUM_EFFECTIVE_N)
    if isinstance(raw_threshold, bool):
        return DEFAULT_MINIMUM_EFFECTIVE_N
    if isinstance(raw_threshold, int | float):
        safe_threshold = _json_safe_float(float(raw_threshold))
        if safe_threshold is not None and safe_threshold > 0.0:
            return safe_threshold
    return DEFAULT_MINIMUM_EFFECTIVE_N


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _json_safe_float(float(value))
    return None


def _observed_period_count(metrics: dict[str, Any]) -> float | None:
    for key in ("observed_period_count", "period_count", "row_count", "test_rows", "split_rows"):
        value = _numeric_or_none(metrics.get(key))
        if value is not None:
            return value
    return None


def _minimum_observations_check(value: float | None, *, threshold: float) -> dict[str, Any]:
    if value is None:
        return {
            "name": "minimum_observations",
            "status": "WARN",
            "value": None,
            "threshold": threshold,
            "message": "Effective sample size or observed period count is unavailable.",
        }
    if value <= 0.0:
        return {
            "name": "minimum_observations",
            "status": "FAIL",
            "value": value,
            "threshold": threshold,
            "message": "No usable return observations are available.",
        }
    if value < threshold:
        return {
            "name": "minimum_observations",
            "status": "WARN",
            "value": value,
            "threshold": threshold,
            "message": f"Usable observation count is below the advisory threshold of {threshold:g}.",
        }
    return {
        "name": "minimum_observations",
        "status": "PASS",
        "value": value,
        "threshold": threshold,
        "message": f"Usable observation count meets the advisory threshold of {threshold:g}.",
    }


def _availability_check(
    name: str,
    available: bool,
    *,
    value: Any,
    message_pass: str,
    message_warn: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if available else "WARN",
        "value": value,
        "threshold": None,
        "message": message_pass if available else message_warn,
    }


def _overall_readiness_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _readiness_summary(checks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_checks": len(checks),
        "passed_checks": sum(1 for check in checks if check.get("status") == "PASS"),
        "warn_checks": sum(1 for check in checks if check.get("status") == "WARN"),
        "failed_checks": sum(1 for check in checks if check.get("status") == "FAIL"),
    }


def extract_closed_trade_returns(results_df: pd.DataFrame) -> pd.Series:
    """
    Extract compounded returns for closed trades from a backtest result frame.

    A trade is defined as one contiguous non-zero executed-position segment.
    Only trades that are closed before the dataset ends are included, so an open
    terminal position does not affect hit rate or profit factor.
    """

    if results_df.empty or "strategy_return" not in results_df.columns:
        return pd.Series(dtype="float64", name="trade_return")

    returns = results_df["strategy_return"].fillna(0.0).astype("float64")
    position = infer_position_series(results_df)
    if position.empty:
        return pd.Series(dtype="float64", name="trade_return")

    closed_returns: list[float] = []
    current_trade_returns: list[float] = []
    in_trade = False

    for idx, current_position in enumerate(position.tolist()):
        period_return = float(returns.iloc[idx])
        has_next = idx + 1 < len(position)
        next_position = float(position.iloc[idx + 1]) if has_next else 0.0

        if current_position != 0.0:
            current_trade_returns.append(period_return)
            in_trade = True

        trade_closes = in_trade and current_position != 0.0 and has_next and next_position != current_position
        if trade_closes:
            closed_returns.append(float((pd.Series(current_trade_returns, dtype="float64") + 1.0).prod() - 1.0))
            current_trade_returns = []
            in_trade = False

    return pd.Series(closed_returns, dtype="float64", name="trade_return")


def _first_non_empty(values: Iterable[Any] | pd.Series | None) -> str | None:
    if values is None:
        return None

    for value in values:
        if pd.notna(value):
            text = str(value)
            if text.strip():
                return text
    return None


def _infer_periods_per_year_from_columns(columns: Iterable[str]) -> int | None:
    normalized_columns = {column.lower() for column in columns}
    if any("1m" in column for column in normalized_columns):
        return MINUTE_PERIODS_PER_YEAR
    if any("1d" in column for column in normalized_columns):
        return TRADING_DAYS_PER_YEAR
    return None


def _align_return_frames(results_df: pd.DataFrame, benchmark_results_df: pd.DataFrame) -> pd.DataFrame:
    strategy_frame = _return_alignment_frame(results_df, "strategy_return")
    benchmark_frame = _return_alignment_frame(benchmark_results_df, "benchmark_return")
    join_keys = [column for column in ("ts_utc", "date") if column in strategy_frame.columns and column in benchmark_frame.columns]

    if join_keys:
        return strategy_frame.merge(benchmark_frame, on=join_keys, how="inner", sort=False)

    return pd.concat(
        [
            strategy_frame["strategy_return"].reset_index(drop=True),
            benchmark_frame["benchmark_return"].reset_index(drop=True),
        ],
        axis=1,
    ).dropna()


def _return_alignment_frame(results_df: pd.DataFrame, return_column_name: str) -> pd.DataFrame:
    if "strategy_return" not in results_df.columns:
        return pd.DataFrame(columns=["strategy_return" if return_column_name == "strategy_return" else return_column_name])

    frame = pd.DataFrame(index=results_df.index)
    for column in ("symbol", "ts_utc", "date"):
        if column in results_df.columns:
            frame[column] = results_df[column]

    if "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

    frame[return_column_name] = pd.to_numeric(results_df["strategy_return"], errors="coerce")
    return frame.dropna(subset=[return_column_name]).reset_index(drop=True)


def aggregate_strategy_returns(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse one results frame into one strategy return per timestamp.

    Multi-symbol strategy runs must be aggregated cross-sectionally before
    compounding into an equity curve or computing benchmark-relative metrics.
    The current default is explicit equal-weight aggregation via arithmetic mean.
    """

    if "strategy_return" not in results_df.columns:
        frame = _empty_aggregated_return_frame(results_df)
        frame.attrs["aggregation"] = _aggregation_metadata(method="none", aggregated=False, symbol_counts=pd.Series(dtype="int64"))
        return frame

    time_column = _strategy_time_column(results_df)
    if time_column is None:
        if _symbol_count(results_df) > 1:
            raise MetricsAggregationError(
                "Multi-symbol returns must include 'ts_utc' or 'date' so metrics can aggregate them safely."
            )
        frame = _empty_aggregated_return_frame(results_df)
        frame["strategy_return"] = pd.to_numeric(results_df["strategy_return"], errors="coerce").astype("float64")
        frame.attrs["aggregation"] = _aggregation_metadata(method="none", aggregated=False, symbol_counts=pd.Series(dtype="int64"))
        return frame

    normalized = pd.DataFrame(index=results_df.index)
    if time_column == "ts_utc":
        normalized["ts_utc"] = pd.to_datetime(results_df["ts_utc"], utc=True, errors="coerce")
    else:
        normalized["date"] = pd.to_datetime(results_df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    if "timeframe" in results_df.columns:
        normalized["timeframe"] = results_df["timeframe"]
    normalized["strategy_return"] = pd.to_numeric(results_df["strategy_return"], errors="coerce").astype("float64")
    if "symbol" in results_df.columns:
        normalized["symbol"] = results_df["symbol"].astype("string")
    normalized["_row_order"] = range(len(normalized))
    normalized = normalized.dropna(subset=[time_column]).sort_values([time_column, "_row_order"], kind="stable")

    if normalized.empty:
        frame = normalized.drop(columns="_row_order", errors="ignore").reset_index(drop=True)
        frame.attrs["aggregation"] = _aggregation_metadata(method="mean", aggregated=False, symbol_counts=pd.Series(dtype="int64"))
        return frame

    symbol_counts = _symbol_count_by_timestamp(normalized, time_column=time_column)
    duplicate_timestamps = normalized.duplicated(subset=[time_column], keep=False)
    requires_aggregation = bool(duplicate_timestamps.any() or symbol_counts.gt(1).any())

    if not requires_aggregation:
        frame = normalized.drop(columns=["_row_order", "symbol"], errors="ignore").reset_index(drop=True)
        frame.attrs["aggregation"] = _aggregation_metadata(method="none", aggregated=False, symbol_counts=symbol_counts)
        return frame

    aggregated = (
        normalized.groupby(time_column, sort=False, as_index=False)
        .agg(strategy_return=("strategy_return", "mean"))
    )
    if "timeframe" in normalized.columns:
        aggregated["timeframe"] = (
            normalized.groupby(time_column, sort=False)["timeframe"].first().reset_index(drop=True)
        )
    aggregated.attrs["aggregation"] = _aggregation_metadata(method="mean", aggregated=True, symbol_counts=symbol_counts)
    return aggregated.reset_index(drop=True)


def broadcast_strategy_equity_curve(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one strategy-level return/equity stream, broadcast onto the input rows.

    This keeps artifact row identity intact while ensuring each timestamp reflects
    the explicitly aggregated strategy return instead of raw per-symbol returns.
    """

    if results_df.empty:
        return pd.DataFrame(index=results_df.index, columns=["strategy_return", "equity"], dtype="float64")

    aggregated = aggregate_strategy_returns(results_df)
    if aggregated.empty:
        return pd.DataFrame(
            {
                "strategy_return": pd.Series([0.0] * len(results_df), index=results_df.index, dtype="float64"),
                "equity": pd.Series([1.0] * len(results_df), index=results_df.index, dtype="float64"),
            }
        )

    return_column = pd.to_numeric(aggregated["strategy_return"], errors="coerce").fillna(0.0).astype("float64")
    equity = (1.0 + return_column).cumprod().astype("float64")
    time_column = _strategy_time_column(results_df)
    if time_column is None:
        return pd.DataFrame({"strategy_return": return_column.reset_index(drop=True), "equity": equity.reset_index(drop=True)})

    if time_column == "ts_utc":
        left_key = pd.to_datetime(results_df["ts_utc"], utc=True, errors="coerce")
        right_key = pd.to_datetime(aggregated["ts_utc"], utc=True, errors="coerce")
    else:
        left_key = pd.to_datetime(results_df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        right_key = pd.to_datetime(aggregated["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

    lookup = pd.DataFrame(
        {
            "_metric_time_key": right_key,
            "strategy_return": return_column.to_numpy(copy=True),
            "equity": equity.to_numpy(copy=True),
        }
    )
    joined = pd.DataFrame({"_metric_time_key": left_key}, index=results_df.index).merge(
        lookup,
        on="_metric_time_key",
        how="left",
        sort=False,
    )
    return joined.loc[:, ["strategy_return", "equity"]].astype("float64")


def _empty_aggregated_return_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=results_df.index)
    for column in ("ts_utc", "date", "timeframe"):
        if column in results_df.columns:
            frame[column] = results_df[column]
    frame["strategy_return"] = pd.Series(dtype="float64")
    return frame.reset_index(drop=True)


def _strategy_time_column(results_df: pd.DataFrame) -> str | None:
    if "ts_utc" in results_df.columns:
        return "ts_utc"
    if "date" in results_df.columns:
        return "date"
    return None


def _symbol_count(results_df: pd.DataFrame) -> int:
    if "symbol" not in results_df.columns:
        return 0
    return int(results_df["symbol"].dropna().astype("string").nunique())


def _symbol_count_by_timestamp(results_df: pd.DataFrame, *, time_column: str) -> pd.Series:
    if "symbol" not in results_df.columns:
        counts = results_df.groupby(time_column, sort=False).size().astype("int64")
        counts.name = "symbol_count"
        return counts
    counts = (
        results_df.assign(symbol=results_df["symbol"].astype("string"))
        .groupby(time_column, sort=False)["symbol"]
        .nunique(dropna=True)
        .astype("int64")
    )
    counts.name = "symbol_count"
    return counts


def _aggregation_metadata(*, method: str, aggregated: bool, symbol_counts: pd.Series) -> dict[str, Any]:
    max_symbol_count = int(symbol_counts.max()) if not symbol_counts.empty else 0
    return {
        "method": method,
        "aggregated": aggregated,
        "max_symbol_count": max_symbol_count,
        "symbol_count_by_timestamp": {str(index): int(value) for index, value in symbol_counts.items()},
    }


def _validate_benchmark_comparability(results_df: pd.DataFrame, benchmark_results_df: pd.DataFrame) -> None:
    strategy_symbols = _symbol_count(results_df)
    benchmark_symbols = _symbol_count(benchmark_results_df)
    if strategy_symbols > 1 and benchmark_symbols == 1:
        raise MetricsAggregationError(
            "Benchmark-relative metrics require a benchmark aggregated over the same symbol universe as the strategy."
        )
