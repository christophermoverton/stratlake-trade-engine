from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_MINUTES_PER_DAY = 390
MINUTE_PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY
DEFAULT_PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR


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


def turnover(position: pd.Series) -> float:
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

    position_change = positions.diff().fillna(positions)
    return float(position_change.abs().mean())


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

    strategy_return = results_df["strategy_return"] if "strategy_return" in results_df.columns else pd.Series(dtype="float64")
    periods_per_year = infer_periods_per_year(results_df)
    position = infer_position_series(results_df)
    closed_trade_returns = extract_closed_trade_returns(results_df)

    total = total_return(strategy_return)
    annual_return = annualized_return(strategy_return, periods_per_year=periods_per_year)
    annual_vol = annualized_volatility(strategy_return, periods_per_year=periods_per_year)
    period_vol = volatility(strategy_return)
    period_win_rate = win_rate(strategy_return)

    return {
        "cumulative_return": total,
        "total_return": total,
        "volatility": period_vol,
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": sharpe_ratio(strategy_return, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(strategy_return),
        "win_rate": period_win_rate,
        "hit_rate": hit_rate(closed_trade_returns),
        "profit_factor": profit_factor(closed_trade_returns),
        "turnover": turnover(position),
        "exposure_pct": exposure_pct(position),
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

    if "signal" not in results_df.columns:
        return pd.Series(0.0, index=results_df.index, dtype="float64", name="position")

    position = results_df["signal"].shift(1).fillna(0.0).astype("float64")
    position.name = "position"
    return position


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
