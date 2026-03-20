"""Research layer interfaces and utilities."""

from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import save_experiment
from src.research.metrics import (
    annualized_return,
    annualized_volatility,
    compute_performance_metrics,
    cumulative_return,
    exposure_pct,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    turnover,
    volatility,
    win_rate,
)
from src.research.registry import (
    filter_by_metric_threshold,
    filter_by_strategy_name,
    load_registry,
)
from src.research.reporting import load_run_artifacts, print_quick_report, summarize_run
from src.research.signal_engine import generate_signals
from src.research.splits import (
    EvaluationSplit,
    EvaluationSplitConfigError,
    generate_evaluation_splits,
    load_and_generate_evaluation_splits,
)
from src.research.strategy_base import BaseStrategy
from src.research.strategies import build_strategy

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "BaseStrategy",
    "EvaluationSplit",
    "EvaluationSplitConfigError",
    "build_strategy",
    "compute_performance_metrics",
    "cumulative_return",
    "exposure_pct",
    "filter_by_metric_threshold",
    "filter_by_strategy_name",
    "generate_evaluation_splits",
    "generate_signals",
    "hit_rate",
    "load_registry",
    "load_and_generate_evaluation_splits",
    "load_run_artifacts",
    "max_drawdown",
    "print_quick_report",
    "profit_factor",
    "run_backtest",
    "save_experiment",
    "sharpe_ratio",
    "summarize_run",
    "total_return",
    "turnover",
    "volatility",
    "win_rate",
]
