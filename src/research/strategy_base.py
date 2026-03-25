from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base contract for research strategies.

    Strategies consume an analytics-ready feature dataset and return a signal
    series aligned to the input frame's index, using the convention:
    1 for long, 0 for flat, and -1 for short.
    """

    name: str
    dataset: str
    required_input_columns: tuple[str, ...] = ()
    requires_return_column: bool = False

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate deterministic trading signals from a feature dataset.

        Args:
            df: Feature dataset required by the strategy.

        Returns:
            A pandas Series of signals aligned to ``df.index`` where values use
            the standard trading convention: 1 for long, 0 for flat, and -1 for
            short.
        """

