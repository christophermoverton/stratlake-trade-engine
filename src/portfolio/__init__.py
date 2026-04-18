from .allocators import BaseAllocator, EqualWeightAllocator, OptimizerAllocator
from .artifacts import (
    build_portfolio_registry_metadata,
    register_validated_portfolio_run,
    write_portfolio_artifacts,
)
from .contracts import (
    PortfolioContractError,
    PortfolioValidationConfig,
    resolve_portfolio_validation_config,
    validate_aligned_returns,
    validate_portfolio_config,
    validate_portfolio_output,
    validate_strategy_returns,
    validate_weights,
)
from .constructor import (
    compute_portfolio_equity_curve,
    compute_portfolio_returns,
    construct_portfolio,
)
from .execution import (
    PortfolioExecutionError,
    apply_portfolio_execution_model,
)
from .loaders import (
    SUPPORTED_PORTFOLIO_COMPONENT_ARTIFACT_TYPES,
    build_aligned_return_matrix,
    load_portfolio_component_returns,
    load_portfolio_component_runs_returns,
    load_strategy_run_returns,
    load_strategy_runs_returns,
)
from .metrics import compute_portfolio_metrics, compute_long_short_directional_metrics
from .optimizer import (
    DirectionalPortfolioConstraints,
    PortfolioOptimizationError,
    PortfolioOptimizerConfig,
    SUPPORTED_PORTFOLIO_OPTIMIZERS,
    optimize_portfolio,
    resolve_portfolio_optimizer_config,
    validate_directional_constraints,
)
from .risk import (
    PortfolioRiskConfig,
    PortfolioRiskError,
    PortfolioVolatilityTargetingConfig,
    apply_volatility_targeting,
    drawdown_series_from_equity,
    drawdown_series_from_returns,
    historical_cvar,
    historical_var,
    resolve_portfolio_risk_config,
    resolve_portfolio_volatility_targeting_config,
    rolling_volatility,
    summarize_drawdown,
    summarize_portfolio_risk,
    validate_equity_curve_series,
    validate_return_series,
    volatility_target_diagnostics,
)
from .qa import (
    PortfolioQAError,
    generate_portfolio_qa_summary,
    run_portfolio_qa,
    validate_equity_curve,
    validate_portfolio_artifact_consistency,
    validate_portfolio_return_consistency,
    validate_weights_behavior,
)
from .validation import (
    PortfolioValidationError,
    summarize_weight_diagnostics,
    validate_portfolio_output_constraints,
    validate_portfolio_weights,
)

__all__ = [
    "PortfolioContractError",
    "PortfolioValidationConfig",
    "PortfolioValidationError",
    "BaseAllocator",
    "EqualWeightAllocator",
    "OptimizerAllocator",
    "DirectionalPortfolioConstraints",
    "PortfolioOptimizationError",
    "PortfolioOptimizerConfig",
    "PortfolioRiskConfig",
    "PortfolioRiskError",
    "PortfolioVolatilityTargetingConfig",
    "build_portfolio_registry_metadata",
    "register_validated_portfolio_run",
    "write_portfolio_artifacts",
    "SUPPORTED_PORTFOLIO_COMPONENT_ARTIFACT_TYPES",
    "load_strategy_run_returns",
    "load_strategy_runs_returns",
    "load_portfolio_component_returns",
    "load_portfolio_component_runs_returns",
    "build_aligned_return_matrix",
    "compute_portfolio_metrics",
    "compute_long_short_directional_metrics",
    "optimize_portfolio",
    "compute_portfolio_returns",
    "compute_portfolio_equity_curve",
    "construct_portfolio",
    "PortfolioExecutionError",
    "apply_portfolio_execution_model",
    "PortfolioQAError",
    "generate_portfolio_qa_summary",
    "run_portfolio_qa",
    "PORTFOLIO_WALK_FORWARD_METRIC_KEYS",
    "PortfolioWalkForwardError",
    "run_portfolio_walk_forward",
    "validate_strategy_returns",
    "validate_aligned_returns",
    "validate_weights",
    "validate_portfolio_output",
    "validate_portfolio_config",
    "resolve_portfolio_validation_config",
    "resolve_portfolio_optimizer_config",
    "SUPPORTED_PORTFOLIO_OPTIMIZERS",
    "validate_directional_constraints",
    "resolve_portfolio_risk_config",
    "resolve_portfolio_volatility_targeting_config",
    "apply_volatility_targeting",
    "validate_portfolio_return_consistency",
    "validate_equity_curve",
    "validate_weights_behavior",
    "validate_portfolio_artifact_consistency",
    "validate_portfolio_weights",
    "validate_portfolio_output_constraints",
    "summarize_weight_diagnostics",
    "validate_return_series",
    "validate_equity_curve_series",
    "rolling_volatility",
    "drawdown_series_from_equity",
    "drawdown_series_from_returns",
    "summarize_drawdown",
    "historical_var",
    "historical_cvar",
    "volatility_target_diagnostics",
    "summarize_portfolio_risk",
]


def __getattr__(name: str):
    if name in {
        "PORTFOLIO_WALK_FORWARD_METRIC_KEYS",
        "PortfolioWalkForwardError",
        "run_portfolio_walk_forward",
    }:
        from .walk_forward import (
            PORTFOLIO_WALK_FORWARD_METRIC_KEYS,
            PortfolioWalkForwardError,
            run_portfolio_walk_forward,
        )

        exports = {
            "PORTFOLIO_WALK_FORWARD_METRIC_KEYS": PORTFOLIO_WALK_FORWARD_METRIC_KEYS,
            "PortfolioWalkForwardError": PortfolioWalkForwardError,
            "run_portfolio_walk_forward": run_portfolio_walk_forward,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
