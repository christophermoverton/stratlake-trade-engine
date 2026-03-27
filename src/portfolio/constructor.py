from __future__ import annotations

import pandas as pd

from src.config.execution import ExecutionConfig, resolve_execution_config
from src.research.consistency import validate_portfolio_sleeve_aggregation_consistency
from src.research.turnover import compute_weight_change_frame, validate_weight_change_frame
from .allocators import BaseAllocator
from .contracts import (
    PortfolioContractError,
    PortfolioValidationConfig,
    resolve_portfolio_validation_config,
    validate_aligned_returns,
    validate_portfolio_output,
)
from .validation import validate_portfolio_output_constraints, validate_portfolio_weights


def compute_portfolio_returns(
    returns_wide: pd.DataFrame,
    weights_wide: pd.DataFrame,
    execution_config: ExecutionConfig | None = None,
    validation_config: PortfolioValidationConfig | dict[str, object] | None = None,
) -> pd.DataFrame:
    """Compute a deterministic portfolio return stream from aligned returns and weights."""

    try:
        normalized_returns = validate_aligned_returns(returns_wide)
    except PortfolioContractError as exc:
        raise ValueError(f"returns_wide must be a valid aligned return matrix: {exc}") from exc

    config_validation = resolve_portfolio_validation_config(validation_config)
    try:
        normalized_weights = validate_portfolio_weights(weights_wide, config_validation)
    except (PortfolioContractError, ValueError) as exc:
        raise ValueError(f"weights_wide must be a valid weights matrix: {exc}") from exc

    if normalized_returns.empty:
        raise ValueError("Aligned return matrix is empty; cannot compute portfolio returns.")
    if normalized_weights.empty:
        raise ValueError("Weights matrix is empty; cannot compute portfolio returns.")
    if not normalized_returns.index.equals(normalized_weights.index):
        raise ValueError("returns_wide and weights_wide must have exactly matching indices.")
    if not normalized_returns.columns.equals(normalized_weights.columns):
        raise ValueError("returns_wide and weights_wide must have exactly matching columns.")

    config = execution_config or resolve_execution_config()

    output = pd.DataFrame({"ts_utc": normalized_returns.index}, copy=False)
    for strategy_name in normalized_returns.columns:
        output[f"strategy_return__{strategy_name}"] = normalized_returns[strategy_name].to_numpy(copy=True)
    for strategy_name in normalized_weights.columns:
        output[f"weight__{strategy_name}"] = normalized_weights[strategy_name].to_numpy(copy=True)

    gross_returns = (normalized_returns * normalized_weights).sum(axis=1).astype("float64")
    weight_change = compute_weight_change_frame(normalized_weights)
    validate_weight_change_frame(weight_change)
    transaction_cost = _weight_execution_cost(weight_change["portfolio_turnover"], config.transaction_cost_bps, enabled=config.enabled)
    slippage_cost = _weight_execution_cost(weight_change["portfolio_turnover"], config.slippage_bps, enabled=config.enabled)
    output["gross_portfolio_return"] = gross_returns.to_numpy(copy=True)
    output["portfolio_weight_change"] = weight_change["portfolio_weight_change"].to_numpy(copy=True)
    output["portfolio_abs_weight_change"] = weight_change["portfolio_abs_weight_change"].to_numpy(copy=True)
    output["portfolio_turnover"] = weight_change["portfolio_turnover"].to_numpy(copy=True)
    output["portfolio_rebalance_event"] = weight_change["portfolio_rebalance_event"].astype("int64").to_numpy(copy=True)
    output["portfolio_transaction_cost"] = transaction_cost.to_numpy(copy=True)
    output["portfolio_slippage_cost"] = slippage_cost.to_numpy(copy=True)
    output["portfolio_execution_friction"] = (transaction_cost + slippage_cost).to_numpy(copy=True)
    output["net_portfolio_return"] = (
        gross_returns - transaction_cost - slippage_cost
    ).to_numpy(copy=True)
    output["portfolio_return"] = output["net_portfolio_return"]

    try:
        validated = validate_portfolio_output_constraints(
            output,
            validation_config=config_validation,
            require_traceability=True,
        )
    except (PortfolioContractError, ValueError) as exc:
        raise ValueError(f"Portfolio return aggregation produced invalid output: {exc}") from exc

    validated.attrs["portfolio_constructor"] = {
        "stage": "returns",
        "strategy_count": len(normalized_returns.columns),
        "timestamp_count": len(normalized_returns.index),
        "execution": config.to_dict(),
        "validation": config_validation.to_dict(),
    }
    validate_portfolio_sleeve_aggregation_consistency(
        normalized_returns,
        normalized_weights,
        validated,
    )
    return validated


def compute_portfolio_equity_curve(
    portfolio_returns: pd.DataFrame,
    initial_capital: float = 1.0,
    validation_config: PortfolioValidationConfig | dict[str, object] | None = None,
) -> pd.DataFrame:
    """Append a compounded portfolio equity curve to validated portfolio return output."""

    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")

    config_validation = resolve_portfolio_validation_config(validation_config)
    try:
        normalized_portfolio = validate_portfolio_output(portfolio_returns)
    except PortfolioContractError as exc:
        raise ValueError(f"portfolio_returns must be valid portfolio output: {exc}") from exc

    if normalized_portfolio.empty:
        raise ValueError("portfolio_returns is empty; cannot compute portfolio equity curve.")
    if "portfolio_return" not in normalized_portfolio.columns:
        raise ValueError("portfolio_returns must contain required column 'portfolio_return'.")

    output = normalized_portfolio.copy()
    output["portfolio_equity_curve"] = (
        float(initial_capital) * (1.0 + output["portfolio_return"]).cumprod()
    ).astype("float64")

    try:
        validated = validate_portfolio_output_constraints(
            output,
            validation_config=config_validation,
            initial_capital=float(initial_capital),
            require_traceability=True,
        )
    except (PortfolioContractError, ValueError) as exc:
        raise ValueError(f"Portfolio equity curve construction produced invalid output: {exc}") from exc

    validated.attrs["portfolio_constructor"] = {
        "stage": "equity_curve",
        "initial_capital": float(initial_capital),
        "timestamp_count": len(validated),
        "validation": config_validation.to_dict(),
    }
    return validated


def construct_portfolio(
    returns_wide: pd.DataFrame,
    allocator: BaseAllocator,
    initial_capital: float = 1.0,
    execution_config: ExecutionConfig | None = None,
    validation_config: PortfolioValidationConfig | dict[str, object] | None = None,
    optimization_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct a complete in-memory portfolio output from aligned strategy returns."""

    if not isinstance(allocator, BaseAllocator):
        raise ValueError("allocator must be an instance of BaseAllocator.")

    try:
        normalized_returns = validate_aligned_returns(returns_wide)
    except PortfolioContractError as exc:
        raise ValueError(f"returns_wide must be a valid aligned return matrix: {exc}") from exc
    if optimization_returns is None:
        normalized_optimization_returns = normalized_returns
    else:
        try:
            normalized_optimization_returns = validate_aligned_returns(optimization_returns)
        except PortfolioContractError as exc:
            raise ValueError(f"optimization_returns must be a valid aligned return matrix: {exc}") from exc
        if normalized_optimization_returns.columns.tolist() != normalized_returns.columns.tolist():
            raise ValueError(
                "optimization_returns and returns_wide must contain exactly matching strategy columns."
            )

    config_validation = resolve_portfolio_validation_config(validation_config)
    weights = allocator.allocate_for_application(
        normalized_optimization_returns,
        normalized_returns,
    )
    optimizer_payload = weights.attrs.get("portfolio_optimizer")
    portfolio_returns = compute_portfolio_returns(
        normalized_returns,
        weights,
        execution_config=execution_config,
        validation_config=config_validation,
    )
    portfolio_output = compute_portfolio_equity_curve(
        portfolio_returns,
        initial_capital=initial_capital,
        validation_config=config_validation,
    )
    portfolio_output.attrs["portfolio_constructor"] = {
        "stage": "complete",
        "allocator": allocator.name,
        "optimizer": optimizer_payload,
        "strategy_count": len(normalized_returns.columns),
        "timestamp_count": len(normalized_returns.index),
        "initial_capital": float(initial_capital),
        "execution": (execution_config or resolve_execution_config()).to_dict(),
        "validation": config_validation.to_dict(),
    }
    return portfolio_output


def _weight_execution_cost(weight_turnover: pd.Series, bps: float, *, enabled: bool) -> pd.Series:
    if not enabled or bps == 0.0:
        return pd.Series(0.0, index=weight_turnover.index, dtype="float64")
    return (weight_turnover * (float(bps) / 10_000.0)).astype("float64")


__all__ = [
    "compute_portfolio_equity_curve",
    "compute_portfolio_returns",
    "construct_portfolio",
]
