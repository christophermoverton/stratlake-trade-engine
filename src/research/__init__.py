"""Research layer interfaces and utilities."""

from src.research.backtest_runner import run_backtest
from src.research.signal_engine import generate_signals
from src.research.strategy_base import BaseStrategy

__all__ = ["BaseStrategy", "generate_signals", "run_backtest"]
