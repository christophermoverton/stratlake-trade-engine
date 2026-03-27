from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.execution import (
    ExecutionConfig,
    SUPPORTED_FIXED_FEE_MODELS,
    SUPPORTED_SLIPPAGE_MODELS,
)
from src.research.turnover import compute_weight_change_frame, validate_weight_change_frame


class PortfolioExecutionError(ValueError):
    """Raised when deterministic portfolio execution-friction accounting fails."""


@dataclass(frozen=True)
class PortfolioExecutionResult:
    frame: pd.DataFrame
    summary: dict[str, Any]


def apply_portfolio_execution_model(
    returns_wide: pd.DataFrame,
    weights_wide: pd.DataFrame,
    execution_config: ExecutionConfig,
) -> PortfolioExecutionResult:
    """Compute deterministic portfolio execution frictions from aligned returns and weights."""

    validate_portfolio_execution_inputs(
        returns_wide=returns_wide,
        weights_wide=weights_wide,
        execution_config=execution_config,
    )
    weight_change = compute_weight_change_frame(weights_wide)
    validate_weight_change_frame(weight_change)

    changed_sleeve_count = weights_wide.diff().fillna(weights_wide).abs().gt(0.0).sum(axis=1).astype("int64")
    turnover = weight_change["portfolio_turnover"].astype("float64")
    rebalance_event = weight_change["portfolio_rebalance_event"].astype("int64")
    volatility_proxy = _volatility_proxy(returns_wide, weights_wide)

    proportional_cost = _bps_cost(turnover, execution_config.transaction_cost_bps, enabled=execution_config.enabled)
    fixed_fee = _fixed_fee_cost(
        rebalance_event=rebalance_event,
        fixed_fee=execution_config.fixed_fee,
        fixed_fee_model=execution_config.fixed_fee_model,
        enabled=execution_config.enabled,
    )
    slippage_cost = _slippage_cost(
        turnover=turnover,
        volatility_proxy=volatility_proxy,
        config=execution_config,
    )
    execution_friction = (proportional_cost + fixed_fee + slippage_cost).astype("float64")

    frame = pd.DataFrame(
        {
            "portfolio_weight_change": weight_change["portfolio_weight_change"].astype("float64"),
            "portfolio_abs_weight_change": weight_change["portfolio_abs_weight_change"].astype("float64"),
            "portfolio_turnover": turnover,
            "portfolio_rebalance_event": rebalance_event,
            "portfolio_changed_sleeve_count": changed_sleeve_count,
            "portfolio_transaction_cost": proportional_cost,
            "portfolio_fixed_fee": fixed_fee,
            "portfolio_slippage_proxy": volatility_proxy,
            "portfolio_slippage_cost": slippage_cost,
            "portfolio_execution_friction": execution_friction,
        },
        index=returns_wide.index,
    )
    if not pd.notna(frame.to_numpy(dtype="float64")).all():
        raise PortfolioExecutionError("Portfolio execution accounting produced non-finite outputs.")

    summary = {
        "enabled": bool(execution_config.enabled),
        "transaction_cost_model": {
            "basis": "portfolio_turnover",
            "rate_bps": float(execution_config.transaction_cost_bps),
        },
        "fixed_fee_model": {
            "amount": float(execution_config.fixed_fee),
            "charge_unit": str(execution_config.fixed_fee_model),
            "charged_rebalance_count": int(rebalance_event.sum()) if execution_config.fixed_fee > 0.0 else 0,
        },
        "slippage_model": {
            "method": str(execution_config.slippage_model),
            "rate_bps": float(execution_config.slippage_bps),
            "turnover_scale": float(execution_config.slippage_turnover_scale),
            "volatility_scale": float(execution_config.slippage_volatility_scale),
            "proxy": _slippage_proxy_name(execution_config.slippage_model),
        },
        "totals": {
            "total_turnover": float(turnover.sum()),
            "rebalance_count": int(rebalance_event.sum()),
            "changed_sleeve_events": int(changed_sleeve_count.sum()),
            "total_transaction_cost": float(proportional_cost.sum()),
            "total_fixed_fee": float(fixed_fee.sum()),
            "total_slippage_cost": float(slippage_cost.sum()),
            "total_execution_friction": float(execution_friction.sum()),
        },
    }
    return PortfolioExecutionResult(frame=frame, summary=summary)


def validate_portfolio_execution_inputs(
    *,
    returns_wide: pd.DataFrame,
    weights_wide: pd.DataFrame,
    execution_config: ExecutionConfig,
) -> None:
    if not returns_wide.index.equals(weights_wide.index):
        raise PortfolioExecutionError("Portfolio execution inputs must share exactly matching indices.")
    if not returns_wide.columns.equals(weights_wide.columns):
        raise PortfolioExecutionError("Portfolio execution inputs must share exactly matching columns.")
    if execution_config.fixed_fee_model not in SUPPORTED_FIXED_FEE_MODELS:
        raise PortfolioExecutionError(
            f"Unsupported fixed fee model {execution_config.fixed_fee_model!r}; "
            f"supported values are {sorted(SUPPORTED_FIXED_FEE_MODELS)}."
        )
    if execution_config.slippage_model not in SUPPORTED_SLIPPAGE_MODELS:
        raise PortfolioExecutionError(
            f"Unsupported slippage model {execution_config.slippage_model!r}; "
            f"supported values are {sorted(SUPPORTED_SLIPPAGE_MODELS)}."
        )
    if not pd.notna(returns_wide.to_numpy(dtype="float64")).all():
        raise PortfolioExecutionError("Portfolio execution inputs must contain finite aligned returns.")
    if not pd.notna(weights_wide.to_numpy(dtype="float64")).all():
        raise PortfolioExecutionError("Portfolio execution inputs must contain finite weights.")


def _bps_cost(turnover: pd.Series, bps: float, *, enabled: bool) -> pd.Series:
    if not enabled or bps == 0.0:
        return pd.Series(0.0, index=turnover.index, dtype="float64")
    return (turnover * (float(bps) / 10_000.0)).astype("float64")


def _fixed_fee_cost(
    *,
    rebalance_event: pd.Series,
    fixed_fee: float,
    fixed_fee_model: str,
    enabled: bool,
) -> pd.Series:
    if not enabled or fixed_fee == 0.0:
        return pd.Series(0.0, index=rebalance_event.index, dtype="float64")
    if fixed_fee_model != "per_rebalance":
        raise PortfolioExecutionError(
            f"Unsupported fixed fee model {fixed_fee_model!r}; supported values are ['per_rebalance']."
        )
    return (rebalance_event.astype("float64") * float(fixed_fee)).astype("float64")


def _slippage_cost(
    *,
    turnover: pd.Series,
    volatility_proxy: pd.Series,
    config: ExecutionConfig,
) -> pd.Series:
    if not config.enabled or config.slippage_bps == 0.0:
        return pd.Series(0.0, index=turnover.index, dtype="float64")

    base_cost = turnover * (float(config.slippage_bps) / 10_000.0)
    if config.slippage_model == "constant":
        return base_cost.astype("float64")
    if config.slippage_model == "turnover_scaled":
        return (base_cost * turnover * float(config.slippage_turnover_scale)).astype("float64")
    if config.slippage_model == "volatility_scaled":
        return (base_cost * volatility_proxy * float(config.slippage_volatility_scale)).astype("float64")
    raise PortfolioExecutionError(
        f"Unsupported slippage model {config.slippage_model!r}; supported values are {sorted(SUPPORTED_SLIPPAGE_MODELS)}."
    )


def _volatility_proxy(returns_wide: pd.DataFrame, weights_wide: pd.DataFrame) -> pd.Series:
    proxy = (weights_wide.abs() * returns_wide.abs()).sum(axis=1).astype("float64")
    if not pd.notna(proxy).all():
        raise PortfolioExecutionError("Portfolio slippage volatility proxy must be finite.")
    return proxy


def _slippage_proxy_name(slippage_model: str) -> str | None:
    if slippage_model == "volatility_scaled":
        return "sum(abs(weight) * abs(strategy_return))"
    if slippage_model == "turnover_scaled":
        return "portfolio_turnover"
    return None


__all__ = [
    "PortfolioExecutionError",
    "PortfolioExecutionResult",
    "apply_portfolio_execution_model",
    "validate_portfolio_execution_inputs",
]
