from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .contracts import PortfolioContractError, validate_aligned_returns, validate_weights
from .optimizer import (
    PortfolioOptimizationError,
    PortfolioOptimizerConfig,
    optimize_portfolio,
    resolve_portfolio_optimizer_config,
    static_weight_frame,
)


class BaseAllocator(ABC):
    """Abstract base class for portfolio weight allocators."""

    name: str

    @abstractmethod
    def allocate(self, returns_wide: pd.DataFrame) -> pd.DataFrame:
        """Allocate weights for a validated aligned return matrix."""

    def allocate_for_application(
        self,
        estimation_returns: pd.DataFrame,
        application_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Allocate weights using one return sample and apply them to another."""

        del estimation_returns
        return self.allocate(application_returns)


class OptimizerAllocator(BaseAllocator):
    """Allocate deterministic static portfolio weights via the centralized optimizer."""

    def __init__(
        self,
        optimizer_config: PortfolioOptimizerConfig | dict[str, Any] | None = None,
        *,
        fallback_method: str = "equal_weight",
    ) -> None:
        self.optimizer_config = resolve_portfolio_optimizer_config(
            optimizer_config,
            fallback_method=fallback_method,
        )
        self.name = self.optimizer_config.method

    def allocate(self, returns_wide: pd.DataFrame) -> pd.DataFrame:
        return self.allocate_for_application(returns_wide, returns_wide)

    def allocate_for_application(
        self,
        estimation_returns: pd.DataFrame,
        application_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            normalized_estimation = validate_aligned_returns(estimation_returns)
        except PortfolioContractError as exc:
            raise ValueError(
                f"estimation_returns must be a valid aligned return matrix: {exc}"
            ) from exc
        try:
            normalized_application = validate_aligned_returns(application_returns)
        except PortfolioContractError as exc:
            raise ValueError(
                f"application_returns must be a valid aligned return matrix: {exc}"
            ) from exc

        if normalized_estimation.columns.tolist() != normalized_application.columns.tolist():
            raise ValueError(
                "estimation_returns and application_returns must contain exactly matching strategy columns."
            )
        if normalized_estimation.empty:
            raise ValueError("Aligned return matrix is empty; cannot allocate portfolio weights.")
        if normalized_application.empty:
            raise ValueError("Application return matrix is empty; cannot allocate portfolio weights.")

        try:
            optimization_result = optimize_portfolio(
                normalized_estimation,
                optimizer_config=self.optimizer_config,
            )
            weights = static_weight_frame(normalized_application, optimization_result)
            validated = validate_weights(
                weights,
                validation_config={
                    "target_weight_sum": self.optimizer_config.target_weight_sum,
                },
            )
        except (PortfolioContractError, PortfolioOptimizationError) as exc:
            raise ValueError(f"{self.name} allocation produced invalid weights: {exc}") from exc

        validated.attrs.update(weights.attrs)
        validated.attrs["portfolio_allocator"] = {
            "allocator": self.name,
            "strategy_count": len(validated.columns),
            "optimizer": self.optimizer_config.to_dict(),
        }
        return validated


class EqualWeightAllocator(BaseAllocator):
    """Assign equal weight to every strategy on every timestamp."""

    name = "equal_weight"

    def __init__(self) -> None:
        self._optimizer_allocator = OptimizerAllocator({"method": self.name})

    def allocate(self, returns_wide: pd.DataFrame) -> pd.DataFrame:
        return self._optimizer_allocator.allocate(returns_wide)

    def allocate_for_application(
        self,
        estimation_returns: pd.DataFrame,
        application_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        return self._optimizer_allocator.allocate_for_application(
            estimation_returns,
            application_returns,
        )


__all__ = ["BaseAllocator", "EqualWeightAllocator", "OptimizerAllocator"]
