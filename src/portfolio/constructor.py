from __future__ import annotations

import pandas as pd

from .allocators import BaseAllocator
from .contracts import (
    PortfolioContractError,
    validate_aligned_returns,
    validate_portfolio_output,
    validate_weights,
)


def compute_portfolio_returns(
    returns_wide: pd.DataFrame,
    weights_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Compute a deterministic portfolio return stream from aligned returns and weights."""

    try:
        normalized_returns = validate_aligned_returns(returns_wide)
    except PortfolioContractError as exc:
        raise ValueError(f"returns_wide must be a valid aligned return matrix: {exc}") from exc

    try:
        normalized_weights = validate_weights(weights_wide)
    except PortfolioContractError as exc:
        raise ValueError(f"weights_wide must be a valid weights matrix: {exc}") from exc

    if normalized_returns.empty:
        raise ValueError("Aligned return matrix is empty; cannot compute portfolio returns.")
    if normalized_weights.empty:
        raise ValueError("Weights matrix is empty; cannot compute portfolio returns.")
    if not normalized_returns.index.equals(normalized_weights.index):
        raise ValueError("returns_wide and weights_wide must have exactly matching indices.")
    if not normalized_returns.columns.equals(normalized_weights.columns):
        raise ValueError("returns_wide and weights_wide must have exactly matching columns.")

    output = pd.DataFrame({"ts_utc": normalized_returns.index}, copy=False)
    for strategy_name in normalized_returns.columns:
        output[f"strategy_return__{strategy_name}"] = normalized_returns[strategy_name].to_numpy(copy=True)
    for strategy_name in normalized_weights.columns:
        output[f"weight__{strategy_name}"] = normalized_weights[strategy_name].to_numpy(copy=True)
    output["portfolio_return"] = (normalized_returns * normalized_weights).sum(axis=1).to_numpy(copy=True)

    try:
        validated = validate_portfolio_output(output)
    except PortfolioContractError as exc:
        raise ValueError(f"Portfolio return aggregation produced invalid output: {exc}") from exc

    validated.attrs["portfolio_constructor"] = {
        "stage": "returns",
        "strategy_count": len(normalized_returns.columns),
        "timestamp_count": len(normalized_returns.index),
    }
    return validated


def compute_portfolio_equity_curve(
    portfolio_returns: pd.DataFrame,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Append a compounded portfolio equity curve to validated portfolio return output."""

    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")

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
        validated = validate_portfolio_output(output)
    except PortfolioContractError as exc:
        raise ValueError(f"Portfolio equity curve construction produced invalid output: {exc}") from exc

    validated.attrs["portfolio_constructor"] = {
        "stage": "equity_curve",
        "initial_capital": float(initial_capital),
        "timestamp_count": len(validated),
    }
    return validated


def construct_portfolio(
    returns_wide: pd.DataFrame,
    allocator: BaseAllocator,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Construct a complete in-memory portfolio output from aligned strategy returns."""

    if not isinstance(allocator, BaseAllocator):
        raise ValueError("allocator must be an instance of BaseAllocator.")

    try:
        normalized_returns = validate_aligned_returns(returns_wide)
    except PortfolioContractError as exc:
        raise ValueError(f"returns_wide must be a valid aligned return matrix: {exc}") from exc

    weights = allocator.allocate(normalized_returns)
    portfolio_returns = compute_portfolio_returns(normalized_returns, weights)
    portfolio_output = compute_portfolio_equity_curve(
        portfolio_returns,
        initial_capital=initial_capital,
    )
    portfolio_output.attrs["portfolio_constructor"] = {
        "stage": "complete",
        "allocator": allocator.name,
        "strategy_count": len(normalized_returns.columns),
        "timestamp_count": len(normalized_returns.index),
        "initial_capital": float(initial_capital),
    }
    return portfolio_output


__all__ = [
    "compute_portfolio_equity_curve",
    "compute_portfolio_returns",
    "construct_portfolio",
]
