from __future__ import annotations

import pandas as pd

from src.portfolio.contracts import PortfolioContractError, validate_portfolio_output
from src.research.metrics import (
    MINUTE_PERIODS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    volatility,
    win_rate,
)

_DAILY_TIMEFRAMES = frozenset({"1d", "1day", "day", "daily"})
_MINUTE_TIMEFRAMES = frozenset({"1m", "1min", "1minute", "minute", "minutes"})


def compute_portfolio_metrics(
    portfolio_output: pd.DataFrame,
    timeframe: str,
) -> dict[str, float | None]:
    """
    Compute deterministic standardized metrics for validated portfolio output.

    Return-based metrics reuse the strategy research metric primitives on the
    portfolio return stream directly. Portfolio ``hit_rate`` and
    ``profit_factor`` are interpreted as period-level statistics because the
    portfolio layer currently operates on return streams rather than discrete
    portfolio trade records.

    Activity metrics require traceable ``weight__<strategy>`` columns:
      - ``turnover`` is the mean absolute component-weight change between
        consecutive timestamps, excluding the initial allocation row.
      - ``exposure_pct`` is the percentage of timestamps with non-zero gross
        portfolio exposure inferred from component weights.

    When traceable weights are unavailable, ``turnover`` and ``exposure_pct``
    are returned as ``None`` explicitly rather than inferred heuristically.
    """

    normalized = _validate_portfolio_metrics_input(portfolio_output)
    periods_per_year = _resolve_periods_per_year(timeframe)
    portfolio_returns = normalized["portfolio_return"]

    total = total_return(portfolio_returns)
    weight_frame = _extract_weight_frame(normalized)

    return {
        "cumulative_return": total,
        "total_return": total,
        "volatility": volatility(portfolio_returns),
        "annualized_return": annualized_return(portfolio_returns, periods_per_year=periods_per_year),
        "annualized_volatility": annualized_volatility(
            portfolio_returns,
            periods_per_year=periods_per_year,
        ),
        "sharpe_ratio": sharpe_ratio(portfolio_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(portfolio_returns),
        "win_rate": win_rate(portfolio_returns),
        "hit_rate": hit_rate(portfolio_returns),
        "profit_factor": profit_factor(portfolio_returns),
        "turnover": _portfolio_turnover(weight_frame),
        "exposure_pct": _portfolio_exposure_pct(weight_frame),
    }


def _validate_portfolio_metrics_input(portfolio_output: pd.DataFrame) -> pd.DataFrame:
    try:
        normalized = validate_portfolio_output(portfolio_output)
    except PortfolioContractError as exc:
        raise ValueError(f"portfolio_output must be valid portfolio output: {exc}") from exc

    if normalized.empty:
        raise ValueError("portfolio_output is empty; cannot compute portfolio metrics.")
    if "portfolio_return" not in normalized.columns:
        raise ValueError("portfolio_output must contain required column 'portfolio_return'.")

    return normalized


def _resolve_periods_per_year(timeframe: str) -> int:
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError("timeframe must be a non-empty string.")

    normalized = timeframe.strip().lower()
    if normalized in _DAILY_TIMEFRAMES:
        return TRADING_DAYS_PER_YEAR
    if normalized in _MINUTE_TIMEFRAMES:
        return MINUTE_PERIODS_PER_YEAR

    raise ValueError(f"Unsupported portfolio metrics timeframe: {timeframe!r}.")


def _extract_weight_frame(portfolio_output: pd.DataFrame) -> pd.DataFrame | None:
    weight_columns = [column for column in portfolio_output.columns if column.startswith("weight__")]
    if not weight_columns:
        return None

    weights = portfolio_output.loc[:, weight_columns].copy()
    weights.columns = [column.removeprefix("weight__") for column in weight_columns]
    return weights.astype("float64")


def _portfolio_turnover(weight_frame: pd.DataFrame | None) -> float | None:
    if weight_frame is None:
        return None
    if weight_frame.empty:
        return 0.0

    weight_changes = weight_frame.diff().fillna(0.0)
    return float(weight_changes.abs().sum(axis=1).mean())


def _portfolio_exposure_pct(weight_frame: pd.DataFrame | None) -> float | None:
    if weight_frame is None:
        return None
    if weight_frame.empty:
        return 0.0

    gross_exposure = weight_frame.abs().sum(axis=1)
    return float((gross_exposure.gt(0.0).mean()) * 100.0)


__all__ = ["compute_portfolio_metrics"]
