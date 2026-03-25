from __future__ import annotations

import pandas as pd

from src.research.integrity import validate_research_integrity
from src.research.strategy_base import BaseStrategy


def generate_signals(df: pd.DataFrame, strategy: BaseStrategy) -> pd.DataFrame:
    """
    Apply a research strategy to a feature dataset and attach standardized signals.

    Args:
        df: Feature dataset consumed by the strategy.
        strategy: Strategy implementation that generates trading signals.

    Returns:
        A copy of ``df`` with an added ``signal`` column aligned to ``df.index``.

    Raises:
        TypeError: If the strategy output is not a pandas Series.
        ValueError: If the strategy output index does not match ``df.index`` or
            if the signal values are outside the standard ``{-1, 0, 1}`` set.
    """

    signals = strategy.generate_signals(df)

    if not isinstance(signals, pd.Series):
        raise TypeError(
            f"Strategy {strategy.__class__.__name__}.generate_signals() must return a pandas Series."
        )

    validate_research_integrity(df, signals)

    result = df.copy()
    result["signal"] = signals.rename("signal")
    return result
