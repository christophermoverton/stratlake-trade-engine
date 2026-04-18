"""Short availability and hard-to-borrow constraint modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class ShortAvailabilityError(ValueError):
    """Raised when short availability constraint or operation is invalid."""


@dataclass(frozen=True)
class ShortAvailabilityConstraint:
    """Deterministic short availability and hard-to-borrow modeling."""

    short_available: dict[str, bool] | None = None  # Symbol -> available flag
    hard_to_borrow: dict[str, bool] | None = None  # Symbol -> HTB flag
    policy: str = "exclude"  # {exclude, cap, penalty}
    hard_to_borrow_penalty_bps: float = 0.0  # Additional cost for HTB
    max_short_positions_with_constraints: int | None = None  # Cap on constrained shorts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "short_available": self.short_available,
            "hard_to_borrow": self.hard_to_borrow,
            "policy": self.policy,
            "hard_to_borrow_penalty_bps": self.hard_to_borrow_penalty_bps,
            "max_short_positions_with_constraints": self.max_short_positions_with_constraints,
        }

    @classmethod
    def default(cls) -> "ShortAvailabilityConstraint":
        """Return unconstrained default (all symbols available)."""
        return cls()

    def has_constraints(self) -> bool:
        """True if any availability constraints are set."""
        return bool(
            self.short_available is not None
            or self.hard_to_borrow is not None
            or self.hard_to_borrow_penalty_bps > 0.0
            or self.max_short_positions_with_constraints is not None
        )

    def is_shortable(self, symbol: str) -> bool:
        """Check if a symbol can be shorted."""
        if self.short_available is not None:
            return self.short_available.get(symbol, True)  # Default to available
        return True

    def is_hard_to_borrow(self, symbol: str) -> bool:
        """Check if a symbol is hard-to-borrow."""
        if self.hard_to_borrow is not None:
            return self.hard_to_borrow.get(symbol, False)  # Default to not HTB
        return False


def apply_short_availability_constraints(
    positions: pd.Series,
    symbols: pd.Series,
    constraint: ShortAvailabilityConstraint,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Apply short availability constraints to positions.
    
    Args:
        positions: Series of positions (negative = short)
        symbols: Series of symbol names
        constraint: ShortAvailabilityConstraint configuration
    
    Returns:
        (constrained_positions, diagnostics) tuple
    """
    diagnostics: dict[str, Any] = {
        "original_short_positions": 0,
        "constrained_short_positions": 0,
        "excluded_symbols": [],
        "hard_to_borrow_symbols": [],
        "penalty_applied_bps": 0.0,
    }

    if not constraint.has_constraints():
        diagnostics["original_short_positions"] = int((positions < 0).sum())
        return positions.copy(), diagnostics

    constrained = positions.copy()

    # Find short positions
    short_mask = constrained < 0.0
    diagnostics["original_short_positions"] = int(short_mask.sum())

    # Apply policy
    if constraint.policy == "exclude":
        # Exclude non-shortable symbols
        for i in range(len(constrained)):
            if short_mask.iloc[i] and not constraint.is_shortable(symbols.iloc[i]):
                constrained.iloc[i] = 0.0
                diagnostics["excluded_symbols"].append(symbols.iloc[i])

    elif constraint.policy == "cap":
        # Cap total constrained short positions
        if constraint.max_short_positions_with_constraints is not None:
            current_short_count = int((constrained < 0).sum())
            if current_short_count > constraint.max_short_positions_with_constraints:
                # Keep only the most negative positions (strongest shorts)
                short_positions_idx = constrained[constrained < 0].abs().nlargest(
                    constraint.max_short_positions_with_constraints
                ).index
                for i in range(len(constrained)):
                    if constrained.iloc[i] < 0.0 and i not in short_positions_idx:
                        constrained.iloc[i] = 0.0

    elif constraint.policy == "penalty":
        # Apply penalty cost for constrained shorts (handled at cost layer, not here)
        for i in range(len(constrained)):
            if short_mask.iloc[i] and constraint.is_hard_to_borrow(symbols.iloc[i]):
                diagnostics["hard_to_borrow_symbols"].append(symbols.iloc[i])
                if constraint.hard_to_borrow_penalty_bps > 0.0:
                    diagnostics["penalty_applied_bps"] = constraint.hard_to_borrow_penalty_bps

    diagnostics["constrained_short_positions"] = int((constrained < 0).sum())
    return constrained, diagnostics


def compute_short_availability_costs(
    positions: pd.Series,
    symbols: pd.Series,
    constraint: ShortAvailabilityConstraint,
) -> pd.Series:
    """
    Compute additional costs for short availability constraints.
    
    Args:
        positions: Series of positions (negative = short)
        symbols: Series of symbol names
        constraint: ShortAvailabilityConstraint configuration
    
    Returns:
        Series of additional costs for short positions
    """
    if not constraint.has_constraints() or constraint.hard_to_borrow_penalty_bps == 0.0:
        return pd.Series(0.0, index=positions.index, dtype="float64")

    costs = pd.Series(0.0, index=positions.index, dtype="float64")

    # Apply HTB penalty to qualifying positions
    for i in range(len(positions)):
        if positions.iloc[i] < 0.0 and constraint.is_hard_to_borrow(symbols.iloc[i]):
            # Cost applied to position size
            costs.iloc[i] = abs(positions.iloc[i]) * (constraint.hard_to_borrow_penalty_bps / 10_000.0)

    return costs.astype("float64")


__all__ = [
    "ShortAvailabilityConstraint",
    "ShortAvailabilityError",
    "apply_short_availability_constraints",
    "compute_short_availability_costs",
]
