from __future__ import annotations

import pandas as pd

from src.config.execution import ExecutionConfig, resolve_execution_config
from src.research.consistency import validate_portfolio_sleeve_aggregation_consistency
from .allocators import BaseAllocator
from .contracts import (
    PortfolioContractError,
    PortfolioValidationConfig,
    resolve_portfolio_validation_config,
    validate_aligned_returns,
    validate_portfolio_output,
)
from .execution import PortfolioExecutionError, apply_portfolio_execution_model
from .risk import (
    PortfolioRiskConfig,
    PortfolioVolatilityTargetingConfig,
    apply_volatility_targeting,
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
    try:
        execution_result = apply_portfolio_execution_model(
            normalized_returns,
            normalized_weights,
            config,
        )
    except PortfolioExecutionError as exc:
        raise ValueError(f"Portfolio execution accounting failed: {exc}") from exc

    output["gross_portfolio_return"] = gross_returns.to_numpy(copy=True)
    for column in execution_result.frame.columns:
        output[column] = execution_result.frame[column].to_numpy(copy=True)
    output["net_portfolio_return"] = (
        gross_returns - execution_result.frame["portfolio_execution_friction"]
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
        "execution_summary": execution_result.summary,
        "validation": config_validation.to_dict(),
    }
    validated.attrs["portfolio_execution"] = execution_result.summary
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
    risk_config: PortfolioRiskConfig | dict[str, object] | None = None,
    volatility_targeting_config: PortfolioVolatilityTargetingConfig | dict[str, object] | None = None,
    periods_per_year: int = 252,
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
    base_weights = allocator.allocate_for_application(
        normalized_optimization_returns,
        normalized_returns,
    )
    optimizer_payload = base_weights.attrs.get("portfolio_optimizer")
    targeted_weights, targeting_metadata = apply_volatility_targeting(
        base_weights,
        normalized_returns,
        config=volatility_targeting_config,
        periods_per_year=periods_per_year,
    )
    effective_validation = _effective_validation_config_for_targeting(
        config_validation,
        targeting_metadata=targeting_metadata,
    )
    portfolio_returns = compute_portfolio_returns(
        normalized_returns,
        targeted_weights,
        execution_config=execution_config,
        validation_config=effective_validation,
    )
    portfolio_output = compute_portfolio_equity_curve(
        portfolio_returns,
        initial_capital=initial_capital,
        validation_config=effective_validation,
    )
    portfolio_output.attrs["portfolio_constructor"] = {
        "stage": "complete",
        "allocator": allocator.name,
        "optimizer": optimizer_payload,
        "risk": None if risk_config is None else (
            risk_config.to_dict() if hasattr(risk_config, "to_dict") else dict(risk_config)
        ),
        "volatility_targeting": {
            key: value
            for key, value in targeting_metadata.items()
            if key not in {"base_weights", "targeted_weights"}
        },
        "strategy_count": len(normalized_returns.columns),
        "timestamp_count": len(normalized_returns.index),
        "initial_capital": float(initial_capital),
        "execution": (execution_config or resolve_execution_config()).to_dict(),
        "execution_summary": portfolio_returns.attrs.get("portfolio_execution", {}),
        "validation": config_validation.to_dict(),
        "effective_validation": effective_validation.to_dict(),
    }
    portfolio_output.attrs["portfolio_volatility_targeting"] = targeting_metadata
    return portfolio_output


__all__ = [
    "compute_portfolio_equity_curve",
    "compute_portfolio_returns",
    "construct_portfolio",
]


def _effective_validation_config_for_targeting(
    validation_config: PortfolioValidationConfig,
    *,
    targeting_metadata: dict[str, object],
) -> PortfolioValidationConfig:
    scaling_factor = targeting_metadata.get("volatility_scaling_factor")
    if scaling_factor is None:
        return validation_config

    scale = float(scaling_factor)
    return PortfolioValidationConfig(
        long_only=validation_config.long_only,
        target_weight_sum=float(validation_config.target_weight_sum * scale),
        weight_sum_tolerance=validation_config.weight_sum_tolerance,
        target_net_exposure=float(validation_config.target_net_exposure * scale),
        net_exposure_tolerance=validation_config.net_exposure_tolerance,
        max_gross_exposure=float(validation_config.max_gross_exposure * scale),
        max_leverage=float(validation_config.max_leverage * scale),
        max_single_sleeve_weight=(
            None
            if validation_config.max_single_sleeve_weight is None
            else float(validation_config.max_single_sleeve_weight * scale)
        ),
        min_single_sleeve_weight=(
            None
            if validation_config.min_single_sleeve_weight is None
            else float(validation_config.min_single_sleeve_weight * scale)
        ),
        max_abs_period_return=validation_config.max_abs_period_return,
        max_equity_multiple=validation_config.max_equity_multiple,
        strict_sanity_checks=validation_config.strict_sanity_checks,
    )
