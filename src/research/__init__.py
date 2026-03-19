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
    "generate_evaluation_splits",
    "generate_signals",
    "hit_rate",
    "load_and_generate_evaluation_splits",
    "max_drawdown",
    "profit_factor",
    "run_backtest",
    "save_experiment",
    "sharpe_ratio",
    "total_return",
    "turnover",
    "volatility",
    "win_rate",
]
