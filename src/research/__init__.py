"""Research layer interfaces and utilities."""

from src.research.backtest_runner import run_backtest
from src.research.metrics import (
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    volatility,
    win_rate,
)
from src.research.signal_engine import generate_signals
from src.research.strategy_base import BaseStrategy

__all__ = [
    "BaseStrategy",
    "cumulative_return",
    "generate_signals",
    "max_drawdown",
    "run_backtest",
    "sharpe_ratio",
    "volatility",
    "win_rate",
]
