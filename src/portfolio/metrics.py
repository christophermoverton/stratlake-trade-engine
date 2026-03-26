from __future__ import annotations

import pandas as pd

from src.portfolio.contracts import (
    PortfolioContractError,
    PortfolioValidationConfig,
    resolve_portfolio_validation_config,
    validate_portfolio_output,
)
from src.portfolio.validation import summarize_weight_diagnostics, validate_portfolio_output_constraints
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
from src.research.turnover import compute_weight_change_frame

_DAILY_TIMEFRAMES = frozenset({"1d", "1day", "day", "daily"})
_MINUTE_TIMEFRAMES = frozenset({"1m", "1min", "1minute", "minute", "minutes"})


def compute_portfolio_metrics(
    portfolio_output: pd.DataFrame,
    timeframe: str,
    validation_config: PortfolioValidationConfig | dict[str, object] | None = None,
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

    normalized = _validate_portfolio_metrics_input(portfolio_output, validation_config=validation_config)
    periods_per_year = _resolve_periods_per_year(timeframe)
    portfolio_returns = normalized["portfolio_return"]

    total = total_return(portfolio_returns)
    weight_frame = _extract_weight_frame(normalized)
    weight_change = None if weight_frame is None else compute_weight_change_frame(weight_frame)
    trade_count = 0 if weight_change is None else int(weight_change["portfolio_rebalance_event"].sum())
    transaction_cost = _optional_numeric_series(normalized, "portfolio_transaction_cost")
    slippage_cost = _optional_numeric_series(normalized, "portfolio_slippage_cost")
    execution_friction = _optional_numeric_series(normalized, "portfolio_execution_friction")
    weight_diagnostics = (
        summarize_weight_diagnostics(weight_frame)
        if weight_frame is not None
        else summarize_weight_diagnostics(pd.DataFrame(dtype="float64"))
    )
    sanity_issue_count = float(
        len(normalized.attrs.get("portfolio_validation", {}).get("sanity_issues", []))
    )
    research_sanity = normalized.attrs.get("sanity_check", {})
    research_sanity_issue_count = float(research_sanity.get("issue_count", 0.0)) if isinstance(research_sanity, dict) else 0.0
    research_sanity_warning_count = float(research_sanity.get("warning_count", 0.0)) if isinstance(research_sanity, dict) else 0.0
    research_sanity_status = str(research_sanity.get("status", "pass")) if isinstance(research_sanity, dict) else "pass"
    research_sanity_strict_mode = bool(research_sanity.get("strict_sanity_checks", False)) if isinstance(research_sanity, dict) else False

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
        "total_turnover": None if weight_change is None else float(weight_change["portfolio_turnover"].sum()),
        "average_turnover": _portfolio_turnover(weight_frame),
        "trade_count": None if weight_change is None else float(trade_count),
        "rebalance_count": None if weight_change is None else float(trade_count),
        "percent_periods_traded": None if weight_change is None else float(weight_change["portfolio_rebalance_event"].mean() * 100.0),
        "average_trade_size": None if weight_change is None else (float(weight_change["portfolio_turnover"].sum()) / trade_count if trade_count else 0.0),
        "total_transaction_cost": float(transaction_cost.sum()),
        "total_slippage_cost": float(slippage_cost.sum()),
        "total_execution_friction": float(execution_friction.sum()),
        "average_execution_friction_per_trade": float(execution_friction.sum() / trade_count) if trade_count else 0.0,
        "exposure_pct": _portfolio_exposure_pct(weight_frame),
        "average_gross_exposure": float(weight_diagnostics["average_gross_exposure"]),
        "max_gross_exposure": float(weight_diagnostics["max_gross_exposure"]),
        "average_net_exposure": float(weight_diagnostics["average_net_exposure"]),
        "min_net_exposure": float(weight_diagnostics["min_net_exposure"]),
        "max_net_exposure": float(weight_diagnostics["max_net_exposure"]),
        "average_leverage": float(weight_diagnostics["average_leverage"]),
        "max_leverage": float(weight_diagnostics["max_leverage"]),
        "max_single_weight": float(weight_diagnostics["max_single_weight"]),
        "max_weight_sum_deviation": float(weight_diagnostics["max_weight_sum_deviation"]),
        "validation_issue_count": sanity_issue_count,
        "sanity_issue_count": research_sanity_issue_count,
        "sanity_warning_count": research_sanity_warning_count,
        "sanity_status": research_sanity_status,
        "sanity_strict_mode": research_sanity_strict_mode,
    }


def _validate_portfolio_metrics_input(
    portfolio_output: pd.DataFrame,
    *,
    validation_config: PortfolioValidationConfig | dict[str, object] | None,
) -> pd.DataFrame:
    resolved_validation = resolve_portfolio_validation_config(validation_config)
    try:
        normalized = validate_portfolio_output_constraints(
            portfolio_output,
            validation_config=resolved_validation,
            require_traceability=False,
            strict_sanity_checks=False,
        )
    except (PortfolioContractError, ValueError) as exc:
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

    return float(compute_weight_change_frame(weight_frame)["portfolio_turnover"].mean())


def _portfolio_exposure_pct(weight_frame: pd.DataFrame | None) -> float | None:
    if weight_frame is None:
        return None
    if weight_frame.empty:
        return 0.0

    gross_exposure = weight_frame.abs().sum(axis=1)
    return float((gross_exposure.gt(0.0).mean()) * 100.0)


def _optional_numeric_series(portfolio_output: pd.DataFrame, column: str) -> pd.Series:
    if column not in portfolio_output.columns:
        return pd.Series(0.0, index=portfolio_output.index, dtype="float64")
    return pd.to_numeric(portfolio_output[column], errors="coerce").fillna(0.0).astype("float64")


__all__ = ["compute_portfolio_metrics"]
