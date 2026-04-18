from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .contracts import PortfolioContractError, validate_aligned_returns

SUPPORTED_PORTFOLIO_OPTIMIZERS: tuple[str, ...] = (
    "equal_weight",
    "max_sharpe",
    "risk_parity",
)
_DEFAULT_OPTIMIZER_METHOD = "equal_weight"
_DEFAULT_TARGET_WEIGHT_SUM = 1.0
_DEFAULT_COVARIANCE_RIDGE = 1e-8
_DEFAULT_MAX_ITERATIONS = 500
_DEFAULT_TOLERANCE = 1e-8
_DEFAULT_RISK_FREE_RATE = 0.0
_DEFAULT_GRADIENT_STEP = 0.25
_FINITE_DIFFERENCE_EPSILON = 1e-6


class PortfolioOptimizationError(ValueError):
    """Raised when optimizer inputs, configuration, or outputs are invalid."""


@dataclass(frozen=True)
class PortfolioOptimizerConfig:
    """Deterministic portfolio optimization settings."""

    method: str = _DEFAULT_OPTIMIZER_METHOD
    long_only: bool = True
    target_weight_sum: float = _DEFAULT_TARGET_WEIGHT_SUM
    min_weight: float | None = 0.0
    max_weight: float | None = None
    leverage_ceiling: float = _DEFAULT_TARGET_WEIGHT_SUM
    full_investment: bool = True
    max_single_weight: float | None = None
    max_turnover: float | None = None
    risk_free_rate: float = _DEFAULT_RISK_FREE_RATE
    covariance_ridge: float = _DEFAULT_COVARIANCE_RIDGE
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    tolerance: float = _DEFAULT_TOLERANCE
    # Directional long/short constraints (optional)
    directional_constraints: "DirectionalPortfolioConstraints | None" = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "method": self.method,
            "long_only": self.long_only,
            "target_weight_sum": self.target_weight_sum,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "leverage_ceiling": self.leverage_ceiling,
            "full_investment": self.full_investment,
            "max_single_weight": self.max_single_weight,
            "max_turnover": self.max_turnover,
            "risk_free_rate": self.risk_free_rate,
            "covariance_ridge": self.covariance_ridge,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
        }
        if self.directional_constraints is not None:
            payload["directional_constraints"] = self.directional_constraints.to_dict()
        return payload


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    """One deterministic portfolio optimization result."""

    weights: pd.Series
    diagnostics: dict[str, Any]
    config: PortfolioOptimizerConfig


@dataclass(frozen=True)
class DirectionalPortfolioConstraints:
    """Deterministic long/short exposure and position constraints."""

    # Long-side constraints
    max_long_weight_sum: float | None = None  # max total long exposure (e.g., 0.60)
    min_long_positions: int | None = None  # min required long positions
    max_long_positions: int | None = None  # max allowed long positions
    max_long_position_size: float | None = None  # max weight per long position

    # Short-side constraints
    max_short_weight_sum: float | None = None  # max total short exposure (e.g., 0.40)
    min_short_positions: int | None = None  # min required short positions
    max_short_positions: int | None = None  # max allowed short positions
    max_short_position_size: float | None = None  # max weight per short position

    # Aggregate constraints
    max_gross_exposure: float | None = None  # total long + abs(short) (e.g., 1.20)
    min_net_exposure: float | None = None  # min allowed long - abs(short) (e.g., 0.20)
    max_net_exposure: float | None = None  # max allowed long - abs(short) (e.g., 1.00)

    # Short availability
    hard_to_borrow_penalty_bps: float = 0.0  # additional cost for hard-to-borrow shorts
    short_availability_policy: str = "exclude"  # {exclude, cap, penalty}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_long_weight_sum": self.max_long_weight_sum,
            "min_long_positions": self.min_long_positions,
            "max_long_positions": self.max_long_positions,
            "max_long_position_size": self.max_long_position_size,
            "max_short_weight_sum": self.max_short_weight_sum,
            "min_short_positions": self.min_short_positions,
            "max_short_positions": self.max_short_positions,
            "max_short_position_size": self.max_short_position_size,
            "max_gross_exposure": self.max_gross_exposure,
            "min_net_exposure": self.min_net_exposure,
            "max_net_exposure": self.max_net_exposure,
            "hard_to_borrow_penalty_bps": self.hard_to_borrow_penalty_bps,
            "short_availability_policy": self.short_availability_policy,
        }

    @classmethod
    def default(cls) -> "DirectionalPortfolioConstraints":
        """Return unconstrained default."""
        return cls()

    def has_constraints(self) -> bool:
        """True if any directional constraints are set."""
        return any([
            self.max_long_weight_sum is not None,
            self.min_long_positions is not None,
            self.max_long_positions is not None,
            self.max_long_position_size is not None,
            self.max_short_weight_sum is not None,
            self.min_short_positions is not None,
            self.max_short_positions is not None,
            self.max_short_position_size is not None,
            self.max_gross_exposure is not None,
            self.min_net_exposure is not None,
            self.max_net_exposure is not None,
            self.hard_to_borrow_penalty_bps > 0.0,
            self.short_availability_policy != "exclude",
        ])


def validate_directional_constraints(
    weights: pd.Series,
    constraints: DirectionalPortfolioConstraints,
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """
    Validate portfolio weights against directional constraints.
    
    Returns:
        Dictionary with validation results and metrics
    """
    results: dict[str, Any] = {
        "valid": True,
        "violations": [],
        "metrics": {},
    }

    if not constraints.has_constraints():
        results["metrics"]["has_directional_constraints"] = False
        return results

    results["metrics"]["has_directional_constraints"] = True

    # Separate long and short weights
    long_weights = weights.clip(lower=0.0)
    short_weights = weights.clip(upper=0.0).abs()

    long_weight_sum = float(long_weights.sum())
    short_weight_sum = float(short_weights.sum())
    gross_exposure = long_weight_sum + short_weight_sum
    net_exposure = long_weight_sum - short_weight_sum

    # Count positions
    long_positions = int((long_weights > tolerance).sum())
    short_positions = int((short_weights > tolerance).sum())

    # Store metrics
    results["metrics"]["long_weight_sum"] = long_weight_sum
    results["metrics"]["short_weight_sum"] = short_weight_sum
    results["metrics"]["gross_exposure"] = gross_exposure
    results["metrics"]["net_exposure"] = net_exposure
    results["metrics"]["long_positions"] = long_positions
    results["metrics"]["short_positions"] = short_positions

    # Check long constraints
    if constraints.max_long_weight_sum is not None and long_weight_sum > constraints.max_long_weight_sum + tolerance:
        results["violations"].append(
            f"Long weight sum {long_weight_sum:.4f} exceeds max {constraints.max_long_weight_sum:.4f}"
        )
        results["valid"] = False

    if constraints.min_long_positions is not None and long_positions < constraints.min_long_positions:
        results["violations"].append(
            f"Long positions {long_positions} less than minimum {constraints.min_long_positions}"
        )
        results["valid"] = False

    if constraints.max_long_positions is not None and long_positions > constraints.max_long_positions:
        results["violations"].append(
            f"Long positions {long_positions} exceeds maximum {constraints.max_long_positions}"
        )
        results["valid"] = False

    if constraints.max_long_position_size is not None:
        max_long_weight = float(long_weights.max())
        if max_long_weight > constraints.max_long_position_size + tolerance:
            results["violations"].append(
                f"Max long position size {max_long_weight:.4f} exceeds limit {constraints.max_long_position_size:.4f}"
            )
            results["valid"] = False

    # Check short constraints
    if constraints.max_short_weight_sum is not None and short_weight_sum > constraints.max_short_weight_sum + tolerance:
        results["violations"].append(
            f"Short weight sum {short_weight_sum:.4f} exceeds max {constraints.max_short_weight_sum:.4f}"
        )
        results["valid"] = False

    if constraints.min_short_positions is not None and short_positions < constraints.min_short_positions:
        results["violations"].append(
            f"Short positions {short_positions} less than minimum {constraints.min_short_positions}"
        )
        results["valid"] = False

    if constraints.max_short_positions is not None and short_positions > constraints.max_short_positions:
        results["violations"].append(
            f"Short positions {short_positions} exceeds maximum {constraints.max_short_positions}"
        )
        results["valid"] = False

    if constraints.max_short_position_size is not None:
        max_short_weight = float(short_weights.max())
        if max_short_weight > constraints.max_short_position_size + tolerance:
            results["violations"].append(
                f"Max short position size {max_short_weight:.4f} exceeds limit {constraints.max_short_position_size:.4f}"
            )
            results["valid"] = False

    # Check aggregate constraints
    if constraints.max_gross_exposure is not None and gross_exposure > constraints.max_gross_exposure + tolerance:
        results["violations"].append(
            f"Gross exposure {gross_exposure:.4f} exceeds maximum {constraints.max_gross_exposure:.4f}"
        )
        results["valid"] = False

    if constraints.min_net_exposure is not None and net_exposure < constraints.min_net_exposure - tolerance:
        results["violations"].append(
            f"Net exposure {net_exposure:.4f} below minimum {constraints.min_net_exposure:.4f}"
        )
        results["valid"] = False

    if constraints.max_net_exposure is not None and net_exposure > constraints.max_net_exposure + tolerance:
        results["violations"].append(
            f"Net exposure {net_exposure:.4f} exceeds maximum {constraints.max_net_exposure:.4f}"
        )
        results["valid"] = False

    if not results["valid"]:
        results["error_message"] = "; ".join(results["violations"])

    return results


def resolve_directional_constraints_config(
    payload: dict[str, Any] | DirectionalPortfolioConstraints | None,
) -> DirectionalPortfolioConstraints | None:
    if payload is None:
        return None
    if isinstance(payload, DirectionalPortfolioConstraints):
        return payload
    if not isinstance(payload, dict):
        raise PortfolioOptimizationError("optimizer.directional_constraints must be a dictionary when provided.")

    policy = payload.get("short_availability_policy", "exclude")
    if policy not in {"exclude", "cap", "penalty"}:
        raise PortfolioOptimizationError(
            "optimizer.directional_constraints.short_availability_policy must be one of ['exclude', 'cap', 'penalty']."
        )

    return DirectionalPortfolioConstraints(
        max_long_weight_sum=_coerce_optional_non_negative_float(
            payload.get("max_long_weight_sum"),
            field_name="optimizer.directional_constraints.max_long_weight_sum",
        ),
        min_long_positions=_coerce_optional_non_negative_int(
            payload.get("min_long_positions"),
            field_name="optimizer.directional_constraints.min_long_positions",
        ),
        max_long_positions=_coerce_optional_non_negative_int(
            payload.get("max_long_positions"),
            field_name="optimizer.directional_constraints.max_long_positions",
        ),
        max_long_position_size=_coerce_optional_non_negative_float(
            payload.get("max_long_position_size"),
            field_name="optimizer.directional_constraints.max_long_position_size",
        ),
        max_short_weight_sum=_coerce_optional_non_negative_float(
            payload.get("max_short_weight_sum"),
            field_name="optimizer.directional_constraints.max_short_weight_sum",
        ),
        min_short_positions=_coerce_optional_non_negative_int(
            payload.get("min_short_positions"),
            field_name="optimizer.directional_constraints.min_short_positions",
        ),
        max_short_positions=_coerce_optional_non_negative_int(
            payload.get("max_short_positions"),
            field_name="optimizer.directional_constraints.max_short_positions",
        ),
        max_short_position_size=_coerce_optional_non_negative_float(
            payload.get("max_short_position_size"),
            field_name="optimizer.directional_constraints.max_short_position_size",
        ),
        max_gross_exposure=_coerce_optional_non_negative_float(
            payload.get("max_gross_exposure"),
            field_name="optimizer.directional_constraints.max_gross_exposure",
        ),
        min_net_exposure=_coerce_optional_float(
            payload.get("min_net_exposure"),
            field_name="optimizer.directional_constraints.min_net_exposure",
        ),
        max_net_exposure=_coerce_optional_float(
            payload.get("max_net_exposure"),
            field_name="optimizer.directional_constraints.max_net_exposure",
        ),
        hard_to_borrow_penalty_bps=_coerce_non_negative_float(
            payload.get("hard_to_borrow_penalty_bps", 0.0),
            field_name="optimizer.directional_constraints.hard_to_borrow_penalty_bps",
        ),
        short_availability_policy=str(policy),
    )

def resolve_portfolio_optimizer_config(
    payload: PortfolioOptimizerConfig | dict[str, Any] | None,
    *,
    fallback_method: str = _DEFAULT_OPTIMIZER_METHOD,
) -> PortfolioOptimizerConfig:
    if payload is None:
        return PortfolioOptimizerConfig(method=_normalize_method(fallback_method))
    if isinstance(payload, PortfolioOptimizerConfig):
        return payload
    if not isinstance(payload, dict):
        raise PortfolioOptimizationError("portfolio optimizer config must be a dictionary when provided.")

    method = _normalize_method(payload.get("method", fallback_method))
    long_only = payload.get("long_only", True)
    if not isinstance(long_only, bool):
        raise PortfolioOptimizationError("portfolio optimizer field 'long_only' must be a boolean.")
    full_investment = payload.get("full_investment", True)
    if not isinstance(full_investment, bool):
        raise PortfolioOptimizationError("portfolio optimizer field 'full_investment' must be a boolean.")

    target_weight_sum = _coerce_finite_float(
        payload.get("target_weight_sum", _DEFAULT_TARGET_WEIGHT_SUM),
        field_name="optimizer.target_weight_sum",
    )
    min_weight = _coerce_optional_float(payload.get("min_weight", 0.0 if long_only else None), field_name="optimizer.min_weight")
    max_weight = _coerce_optional_float(payload.get("max_weight"), field_name="optimizer.max_weight")
    leverage_ceiling = _coerce_non_negative_float(
        payload.get("leverage_ceiling", abs(target_weight_sum)),
        field_name="optimizer.leverage_ceiling",
    )
    max_single_weight = _coerce_optional_float(
        payload.get("max_single_weight"),
        field_name="optimizer.max_single_weight",
    )
    max_turnover = _coerce_optional_non_negative_float(
        payload.get("max_turnover"),
        field_name="optimizer.max_turnover",
    )
    risk_free_rate = _coerce_finite_float(
        payload.get("risk_free_rate", _DEFAULT_RISK_FREE_RATE),
        field_name="optimizer.risk_free_rate",
    )
    covariance_ridge = _coerce_non_negative_float(
        payload.get("covariance_ridge", _DEFAULT_COVARIANCE_RIDGE),
        field_name="optimizer.covariance_ridge",
    )
    max_iterations = payload.get("max_iterations", _DEFAULT_MAX_ITERATIONS)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise PortfolioOptimizationError("portfolio optimizer field 'max_iterations' must be a positive integer.")
    tolerance = _coerce_non_negative_float(
        payload.get("tolerance", _DEFAULT_TOLERANCE),
        field_name="optimizer.tolerance",
    )
    directional_constraints = resolve_directional_constraints_config(payload.get("directional_constraints"))

    if target_weight_sum <= 0.0:
        raise PortfolioOptimizationError("portfolio optimizer field 'target_weight_sum' must be positive.")
    
    # Validate constraints based on long_only mode
    if long_only:
        if leverage_ceiling + tolerance < target_weight_sum:
            raise PortfolioOptimizationError(
                "portfolio optimizer field 'leverage_ceiling' must be >= 'target_weight_sum' for long-only allocations."
            )
    else:
        # For long/short: leverage_ceiling applies to gross exposure (long + abs(short))
        if leverage_ceiling <= 0.0:
            raise PortfolioOptimizationError(
                "portfolio optimizer field 'leverage_ceiling' must be positive for long/short allocations."
            )
    
    if min_weight is not None and min_weight < 0.0:
        raise PortfolioOptimizationError(
            "portfolio optimizer field 'min_weight' must be >= 0 for long-only allocations."
        )
    if max_weight is not None and max_weight <= 0.0:
        raise PortfolioOptimizationError("portfolio optimizer field 'max_weight' must be positive when provided.")
    if max_single_weight is not None and max_single_weight <= 0.0:
        raise PortfolioOptimizationError(
            "portfolio optimizer field 'max_single_weight' must be positive when provided."
        )
    effective_max_weight = _effective_max_weight(
        max_weight=max_weight,
        max_single_weight=max_single_weight,
        target_weight_sum=target_weight_sum,
    )
    effective_min_weight = 0.0 if min_weight is None else min_weight
    if effective_max_weight < effective_min_weight:
        raise PortfolioOptimizationError(
            "portfolio optimizer requires min_weight <= effective maximum weight bound."
        )

    return PortfolioOptimizerConfig(
        method=method,
        long_only=long_only,
        target_weight_sum=target_weight_sum,
        min_weight=min_weight,
        max_weight=max_weight,
        leverage_ceiling=leverage_ceiling,
        full_investment=full_investment,
        max_single_weight=max_single_weight,
        max_turnover=max_turnover,
        risk_free_rate=risk_free_rate,
        covariance_ridge=covariance_ridge,
        max_iterations=max_iterations,
        tolerance=tolerance,
        directional_constraints=directional_constraints,
    )


def optimizer_validation_overrides(config: PortfolioOptimizerConfig) -> dict[str, Any]:
    min_weight = 0.0 if config.min_weight is None else float(config.min_weight)
    effective_max_weight = _effective_max_weight(
        max_weight=config.max_weight,
        max_single_weight=config.max_single_weight,
        target_weight_sum=config.target_weight_sum,
    )
    return {
        "long_only": config.long_only,
        "target_weight_sum": float(config.target_weight_sum),
        "target_net_exposure": float(config.target_weight_sum),
        "max_gross_exposure": float(config.leverage_ceiling),
        "max_leverage": float(config.leverage_ceiling),
        "min_single_sleeve_weight": min_weight,
        "max_single_sleeve_weight": effective_max_weight,
        "directional_constraints": None if config.directional_constraints is None else config.directional_constraints.to_dict(),
    }


def optimize_portfolio(
    returns_wide: pd.DataFrame,
    optimizer_config: PortfolioOptimizerConfig | dict[str, Any] | None = None,
    *,
    previous_weights: pd.Series | None = None,
) -> PortfolioOptimizationResult:
    config = resolve_portfolio_optimizer_config(optimizer_config)
    try:
        normalized_returns = validate_aligned_returns(returns_wide)
    except PortfolioContractError as exc:
        raise PortfolioOptimizationError(
            f"optimizer returns input must be a valid aligned return matrix: {exc}"
        ) from exc

    if normalized_returns.empty:
        raise PortfolioOptimizationError("optimizer returns input is empty; cannot compute portfolio weights.")
    if len(normalized_returns.columns) == 0:
        raise PortfolioOptimizationError("optimizer returns input must contain at least one strategy column.")
    if normalized_returns.isna().any().any():
        bad_ts, bad_strategy = normalized_returns.isna().stack()[lambda values: values].index[0]
        raise PortfolioOptimizationError(
            "optimizer returns input must not contain NaN values. "
            f"First NaN at ts_utc={bad_ts}, strategy={bad_strategy!r}."
        )

    expected_returns = _historical_expected_returns(normalized_returns, risk_free_rate=config.risk_free_rate)
    covariance = _historical_covariance_matrix(
        normalized_returns,
        ridge=config.covariance_ridge,
        method=config.method,
    )
    lower_bounds, upper_bounds = _bound_vectors(config, strategy_count=len(normalized_returns.columns))

    if config.method == "equal_weight":
        weights, iterations, converged = _equal_weight_solution(
            expected_returns.index,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_weight_sum=config.target_weight_sum,
        )
    elif config.method == "max_sharpe":
        weights, iterations, converged = _max_sharpe_solution(
            expected_returns,
            covariance,
            config=config,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
    elif config.method == "risk_parity":
        weights, iterations, converged = _risk_parity_solution(
            covariance,
            config=config,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
    else:  # pragma: no cover - protected by config validation
        raise PortfolioOptimizationError(
            f"Unsupported portfolio optimizer method {config.method!r}."
        )

    if previous_weights is not None:
        weights = _apply_turnover_constraint(weights, previous_weights, config=config)

    _validate_weight_vector(weights, config=config, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    directional_validation = None
    if config.directional_constraints is not None:
        directional_validation = validate_directional_constraints(weights, config.directional_constraints)
        if not directional_validation["valid"]:
            raise PortfolioOptimizationError(
                "Optimized weights violate directional constraints: "
                f"{directional_validation['error_message']}"
            )

    realized_return = float(np.dot(expected_returns.to_numpy(dtype="float64"), weights.to_numpy(dtype="float64")))
    variance = float(weights.to_numpy(dtype="float64") @ covariance.to_numpy(dtype="float64") @ weights.to_numpy(dtype="float64"))
    volatility = math.sqrt(max(variance, 0.0))
    sharpe = 0.0 if volatility <= config.tolerance else realized_return / volatility
    gross_exposure = float(weights.abs().sum())
    diagnostics = {
        "method": config.method,
        "strategy_count": int(len(weights)),
        "observation_count": int(len(normalized_returns)),
        "expected_return_estimator": "historical_mean_excess_return_per_period",
        "covariance_estimator": "sample_covariance_ddof_0_plus_diagonal_ridge",
        "risk_free_rate": float(config.risk_free_rate),
        "covariance_ridge": float(config.covariance_ridge),
        "iterations": int(iterations),
        "converged": bool(converged),
        "objective_expected_return": realized_return,
        "objective_volatility": volatility,
        "objective_sharpe_ratio": sharpe,
        "weight_sum": float(weights.sum()),
        "gross_exposure": gross_exposure,
        "net_exposure": float(weights.sum()),
        "max_single_weight": float(weights.max()),
        "min_single_weight": float(weights.min()),
        "turnover_vs_previous": None if previous_weights is None else float((weights - _normalize_previous_weights(previous_weights, weights.index)).abs().sum()),
        "strategy_order": weights.index.tolist(),
        "directional_constraints": directional_validation,
    }
    return PortfolioOptimizationResult(weights=weights, diagnostics=diagnostics, config=config)


def static_weight_frame(
    application_returns: pd.DataFrame,
    optimization_result: PortfolioOptimizationResult,
) -> pd.DataFrame:
    try:
        normalized_returns = validate_aligned_returns(application_returns)
    except PortfolioContractError as exc:
        raise PortfolioOptimizationError(
            f"application returns must be a valid aligned return matrix: {exc}"
        ) from exc

    if normalized_returns.columns.tolist() != optimization_result.weights.index.tolist():
        raise PortfolioOptimizationError(
            "optimizer output strategy order must match the application return matrix exactly."
        )

    repeated = np.repeat(
        optimization_result.weights.to_numpy(dtype="float64")[np.newaxis, :],
        len(normalized_returns),
        axis=0,
    )
    weights = pd.DataFrame(
        repeated,
        index=normalized_returns.index.copy(),
        columns=normalized_returns.columns.copy(),
        dtype="float64",
    )
    weights.attrs["portfolio_optimizer"] = {
        "config": optimization_result.config.to_dict(),
        "diagnostics": optimization_result.diagnostics,
        "weights": {
            strategy_name: float(weight_value)
            for strategy_name, weight_value in optimization_result.weights.items()
        },
    }
    return weights


def _normalize_method(method: Any) -> str:
    if not isinstance(method, str):
        raise PortfolioOptimizationError("portfolio optimizer field 'method' must be a non-empty string.")
    normalized = method.strip().lower()
    if normalized not in SUPPORTED_PORTFOLIO_OPTIMIZERS:
        formatted = ", ".join(SUPPORTED_PORTFOLIO_OPTIMIZERS)
        raise PortfolioOptimizationError(
            f"Unsupported portfolio optimizer method {method!r}. Supported methods: {formatted}."
        )
    return normalized


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PortfolioOptimizationError(f"{field_name} must be a finite float.") from exc
    if not math.isfinite(numeric):
        raise PortfolioOptimizationError(f"{field_name} must be a finite float.")
    return numeric


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    numeric = _coerce_finite_float(value, field_name=field_name)
    if numeric < 0.0:
        raise PortfolioOptimizationError(f"{field_name} must be non-negative.")
    return numeric


def _coerce_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _coerce_finite_float(value, field_name=field_name)


def _coerce_optional_non_negative_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _coerce_non_negative_float(value, field_name=field_name)


def _coerce_optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PortfolioOptimizationError(f"{field_name} must be a non-negative integer when provided.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise PortfolioOptimizationError(f"{field_name} must be a non-negative integer when provided.") from exc
    if normalized < 0:
        raise PortfolioOptimizationError(f"{field_name} must be a non-negative integer when provided.")
    if normalized != value and not (isinstance(value, float) and value.is_integer()):
        raise PortfolioOptimizationError(f"{field_name} must be a non-negative integer when provided.")
    return normalized


def _effective_max_weight(*, max_weight: float | None, max_single_weight: float | None, target_weight_sum: float) -> float:
    candidates = [target_weight_sum]
    if max_weight is not None:
        candidates.append(max_weight)
    if max_single_weight is not None:
        candidates.append(max_single_weight)
    return float(min(candidates))


def _historical_expected_returns(returns_wide: pd.DataFrame, *, risk_free_rate: float) -> pd.Series:
    expected = returns_wide.mean(axis=0).astype("float64") - float(risk_free_rate)
    if not np.isfinite(expected.to_numpy(dtype="float64")).all():
        raise PortfolioOptimizationError("expected returns contain non-finite values.")
    return expected


def _historical_covariance_matrix(
    returns_wide: pd.DataFrame,
    *,
    ridge: float,
    method: str,
) -> pd.DataFrame:
    if method == "equal_weight":
        covariance = returns_wide.cov(ddof=0).astype("float64")
    else:
        if len(returns_wide) < 2:
            raise PortfolioOptimizationError(
                f"portfolio optimizer method '{method}' requires at least two return observations."
            )
        covariance = returns_wide.cov(ddof=0).astype("float64")
    if covariance.isna().any().any():
        raise PortfolioOptimizationError("covariance estimate contains NaN values.")
    covariance = covariance.loc[returns_wide.columns, returns_wide.columns]
    covariance_values = covariance.to_numpy(dtype="float64")
    covariance_values = 0.5 * (covariance_values + covariance_values.T)
    covariance_values = covariance_values + (float(ridge) * np.eye(covariance_values.shape[0], dtype="float64"))
    diagonal = np.diag(covariance_values)
    if np.any(diagonal <= 0.0):
        raise PortfolioOptimizationError(
            "covariance estimate must have strictly positive diagonal entries after regularization."
        )
    return pd.DataFrame(covariance_values, index=covariance.index.copy(), columns=covariance.columns.copy())


def _bound_vectors(
    config: PortfolioOptimizerConfig,
    *,
    strategy_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    # For long_only: lower bound is 0.0, for long/short: lower bound is negative (unbounded)
    if config.long_only:
        lower_value = 0.0 if config.min_weight is None else float(config.min_weight)
    else:
        # For long/short, allow negative weights
        lower_value = -10.0 if config.min_weight is None else float(config.min_weight)  # -10.0 as practical unbounded
    
    upper_value = _effective_max_weight(
        max_weight=config.max_weight,
        max_single_weight=config.max_single_weight,
        target_weight_sum=config.target_weight_sum,
    )
    lower_bounds = np.full(strategy_count, lower_value, dtype="float64")
    upper_bounds = np.full(strategy_count, upper_value, dtype="float64")
    if lower_bounds.sum() - config.tolerance > config.target_weight_sum:
        raise PortfolioOptimizationError(
            "optimizer minimum weight bounds are infeasible for the configured target_weight_sum."
        )
    if upper_bounds.sum() + config.tolerance < config.target_weight_sum:
        raise PortfolioOptimizationError(
            "optimizer maximum weight bounds are infeasible for the configured target_weight_sum."
        )
    return lower_bounds, upper_bounds


def _equal_weight_solution(
    strategy_index: pd.Index,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    target_weight_sum: float,
) -> tuple[pd.Series, int, bool]:
    raw = np.full(len(strategy_index), target_weight_sum / len(strategy_index), dtype="float64")
    projected = _project_bounded_simplex(raw, lower_bounds=lower_bounds, upper_bounds=upper_bounds, target_sum=target_weight_sum)
    return pd.Series(projected, index=strategy_index.copy(), dtype="float64"), 1, True


def _max_sharpe_solution(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    config: PortfolioOptimizerConfig,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[pd.Series, int, bool]:
    weights = _project_bounded_simplex(
        np.full(len(expected_returns), config.target_weight_sum / len(expected_returns), dtype="float64"),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        target_sum=config.target_weight_sum,
    )
    covariance_values = covariance.to_numpy(dtype="float64")
    expected_values = expected_returns.to_numpy(dtype="float64")
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        port_return = float(expected_values @ weights)
        variance = float(weights @ covariance_values @ weights)
        if variance <= config.tolerance:
            raise PortfolioOptimizationError(
                "maximum Sharpe optimization is ill-posed because portfolio variance is non-positive."
            )
        volatility = math.sqrt(variance)
        gradient = (expected_values / volatility) - ((port_return / (variance * volatility)) * (covariance_values @ weights))
        candidate = weights + (_DEFAULT_GRADIENT_STEP / math.sqrt(iteration)) * gradient
        projected = _project_bounded_simplex(
            candidate,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_sum=config.target_weight_sum,
        )
        if np.max(np.abs(projected - weights)) <= config.tolerance:
            weights = projected
            converged = True
            return pd.Series(weights, index=expected_returns.index.copy(), dtype="float64"), iteration, converged
        weights = projected

    return pd.Series(weights, index=expected_returns.index.copy(), dtype="float64"), config.max_iterations, converged


def _risk_parity_solution(
    covariance: pd.DataFrame,
    *,
    config: PortfolioOptimizerConfig,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[pd.Series, int, bool]:
    weights = _project_bounded_simplex(
        np.full(len(covariance.columns), config.target_weight_sum / len(covariance.columns), dtype="float64"),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        target_sum=config.target_weight_sum,
    )
    covariance_values = covariance.to_numpy(dtype="float64")
    target_share = np.full(len(weights), 1.0 / len(weights), dtype="float64")
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        portfolio_variance = float(weights @ covariance_values @ weights)
        if portfolio_variance <= config.tolerance:
            raise PortfolioOptimizationError(
                "risk parity optimization is ill-posed because portfolio variance is non-positive."
            )
        marginal_risk = covariance_values @ weights
        risk_contributions = weights * marginal_risk
        if np.any(risk_contributions <= 0.0):
            gradient = _finite_difference_gradient(
                weights,
                objective=lambda candidate: _risk_parity_objective(candidate, covariance_values, target_share, config.tolerance),
            )
            candidate = weights - (_DEFAULT_GRADIENT_STEP / math.sqrt(iteration)) * gradient
        else:
            risk_share = risk_contributions / portfolio_variance
            candidate = weights * (target_share / risk_share)
        projected = _project_bounded_simplex(
            candidate,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_sum=config.target_weight_sum,
        )
        if _risk_parity_objective(projected, covariance_values, target_share, config.tolerance) <= config.tolerance:
            weights = projected
            converged = True
            return pd.Series(weights, index=covariance.index.copy(), dtype="float64"), iteration, converged
        if np.max(np.abs(projected - weights)) <= config.tolerance:
            weights = projected
            converged = True
            return pd.Series(weights, index=covariance.index.copy(), dtype="float64"), iteration, converged
        weights = projected

    return pd.Series(weights, index=covariance.index.copy(), dtype="float64"), config.max_iterations, converged


def _finite_difference_gradient(
    weights: np.ndarray,
    *,
    objective,
) -> np.ndarray:
    gradient = np.zeros_like(weights)
    for idx in range(len(weights)):
        step = np.zeros_like(weights)
        step[idx] = _FINITE_DIFFERENCE_EPSILON
        gradient[idx] = (
            objective(weights + step) - objective(weights - step)
        ) / (2.0 * _FINITE_DIFFERENCE_EPSILON)
    return gradient


def _risk_parity_objective(
    weights: np.ndarray,
    covariance_values: np.ndarray,
    target_share: np.ndarray,
    tolerance: float,
) -> float:
    portfolio_variance = float(weights @ covariance_values @ weights)
    if portfolio_variance <= tolerance:
        return float("inf")
    marginal_risk = covariance_values @ weights
    risk_share = (weights * marginal_risk) / portfolio_variance
    return float(np.square(risk_share - target_share).sum())


def _project_bounded_simplex(
    values: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    lower_sum = float(lower_bounds.sum())
    upper_sum = float(upper_bounds.sum())
    if target_sum < lower_sum - 1e-12 or target_sum > upper_sum + 1e-12:
        raise PortfolioOptimizationError(
            "optimizer bounds are infeasible for the requested target_weight_sum."
        )

    lower = float(np.min(values - upper_bounds))
    upper = float(np.max(values - lower_bounds))
    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        projected = np.clip(values - midpoint, lower_bounds, upper_bounds)
        projected_sum = float(projected.sum())
        if abs(projected_sum - target_sum) <= 1e-12:
            return projected
        if projected_sum > target_sum:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(values - ((lower + upper) / 2.0), lower_bounds, upper_bounds)
    adjustment = target_sum - float(projected.sum())
    if abs(adjustment) > 1e-8:
        raise PortfolioOptimizationError(
            "optimizer projection failed to satisfy the configured target_weight_sum."
        )
    return projected


def _normalize_previous_weights(previous_weights: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(previous_weights, pd.Series):
        raise PortfolioOptimizationError("previous_weights must be a pandas Series when provided.")
    normalized = previous_weights.copy()
    normalized.index = pd.Index([str(value).strip() for value in normalized.index])
    normalized = normalized.astype("float64")
    normalized = normalized.reindex(index)
    if normalized.isna().any():
        missing = normalized.index[normalized.isna()].tolist()
        raise PortfolioOptimizationError(
            f"previous_weights is missing required strategy identifiers: {missing}."
        )
    return normalized


def _apply_turnover_constraint(
    weights: pd.Series,
    previous_weights: pd.Series,
    *,
    config: PortfolioOptimizerConfig,
) -> pd.Series:
    if config.max_turnover is None:
        return weights
    reference = _normalize_previous_weights(previous_weights, weights.index)
    turnover = float((weights - reference).abs().sum())
    if turnover <= config.max_turnover + config.tolerance:
        return weights
    if turnover <= config.tolerance:
        return weights
    blend = float(config.max_turnover / turnover)
    constrained = reference + (weights - reference) * blend
    return constrained.astype("float64")


def _validate_weight_vector(
    weights: pd.Series,
    *,
    config: PortfolioOptimizerConfig,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> None:
    values = weights.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise PortfolioOptimizationError("optimizer produced non-finite weights.")
    if abs(float(values.sum()) - config.target_weight_sum) > config.tolerance:
        raise PortfolioOptimizationError(
            "optimizer produced weights whose sum does not match target_weight_sum."
        )
    if config.long_only and np.any(values < -config.tolerance):
        raise PortfolioOptimizationError("optimizer produced negative weights under long_only=True.")
    if np.any(values < (lower_bounds - config.tolerance)):
        raise PortfolioOptimizationError("optimizer produced weights below the configured minimum bound.")
    if np.any(values > (upper_bounds + config.tolerance)):
        raise PortfolioOptimizationError("optimizer produced weights above the configured maximum bound.")
    gross_exposure = float(np.abs(values).sum())
    if gross_exposure > config.leverage_ceiling + config.tolerance:
        raise PortfolioOptimizationError(
            "optimizer produced weights that exceed the configured leverage_ceiling."
        )


__all__ = [
    "PortfolioOptimizationError",
    "PortfolioOptimizationResult",
    "PortfolioOptimizerConfig",
    "SUPPORTED_PORTFOLIO_OPTIMIZERS",
    "optimize_portfolio",
    "optimizer_validation_overrides",
    "resolve_portfolio_optimizer_config",
    "static_weight_frame",
]
