from __future__ import annotations

import pandas as pd


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


def sharpe_ratio(strategy_return: pd.Series) -> float:
    """
    Compute the Sharpe ratio of the strategy return series.

    This initial implementation assumes a zero risk-free rate and does not
    annualize the result.

    Args:
        strategy_return: Period-by-period strategy returns.

    Returns:
        The mean return divided by return volatility. Returns ``0.0`` when the
        ratio is undefined because the series is empty or has zero volatility.
    """

    returns = _normalized_returns(strategy_return)
    if returns.empty:
        return 0.0

    return_volatility = volatility(returns)
    if return_volatility == 0.0:
        return 0.0

    return float(returns.mean() / return_volatility)


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
