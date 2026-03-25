from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .contracts import PortfolioContractError, validate_aligned_returns, validate_weights


class BaseAllocator(ABC):
    """Abstract base class for portfolio weight allocators."""

    name: str

    @abstractmethod
    def allocate(self, returns_wide: pd.DataFrame) -> pd.DataFrame:
        """Allocate weights for a validated aligned return matrix."""


class EqualWeightAllocator(BaseAllocator):
    """Assign equal weight to every strategy on every timestamp."""

    name = "equal_weight"

    def allocate(self, returns_wide: pd.DataFrame) -> pd.DataFrame:
        try:
            normalized_returns = validate_aligned_returns(returns_wide)
        except PortfolioContractError as exc:
            raise ValueError(f"returns_wide must be a valid aligned return matrix: {exc}") from exc

        if normalized_returns.empty:
            raise ValueError("Aligned return matrix is empty; cannot allocate portfolio weights.")
        if len(normalized_returns.columns) == 0:
            raise ValueError("Aligned return matrix must contain at least one strategy column.")

        strategy_count = len(normalized_returns.columns)
        weight_value = 1.0 / strategy_count
        weights = pd.DataFrame(
            weight_value,
            index=normalized_returns.index.copy(),
            columns=normalized_returns.columns.copy(),
            dtype="float64",
        )
        weights.index.name = normalized_returns.index.name
        weights.columns.name = normalized_returns.columns.name

        try:
            validated = validate_weights(weights)
        except PortfolioContractError as exc:
            raise ValueError(f"Equal-weight allocation produced invalid weights: {exc}") from exc

        validated.attrs["portfolio_allocator"] = {
            "allocator": self.name,
            "strategy_count": strategy_count,
        }
        return validated


__all__ = ["BaseAllocator", "EqualWeightAllocator"]
