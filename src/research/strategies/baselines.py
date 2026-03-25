from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.strategy_base import BaseStrategy

from src.research.strategies._helpers import resolve_return_column, valid_return_mask


class BuyAndHoldStrategy(BaseStrategy):
    """Stay long from the first row with a valid asset return through the end of the dataset."""

    name = "buy_and_hold_v1"
    dataset = "features_daily"
    required_input_columns = ()
    requires_return_column = True

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype="int64")
        if df.empty:
            return signals.rename("signal")

        tradable_rows = valid_return_mask(df)
        if not tradable_rows.any():
            return signals.rename("signal")

        first_valid_index = tradable_rows[tradable_rows].index[0]
        signals.loc[first_valid_index:] = 1
        return signals.rename("signal")


class SMACrossoverStrategy(BaseStrategy):
    """Long when a fast SMA exceeds a slow SMA on a synthetic price series built from returns."""

    name = "sma_crossover_v1"
    dataset = "features_daily"
    required_input_columns = ()
    requires_return_column = True

    def __init__(self, *, fast_window: int, slow_window: int) -> None:
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("SMA crossover windows must be positive integers.")
        if fast_window >= slow_window:
            raise ValueError("SMA crossover requires fast_window to be smaller than slow_window.")

        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype="int64")
        if df.empty:
            return signals.rename("signal")

        return_column = resolve_return_column(df)
        returns = df[return_column].astype("float64")
        synthetic_price = (1.0 + returns.fillna(0.0)).cumprod()
        synthetic_price = synthetic_price.where(returns.notna())

        fast_sma = synthetic_price.rolling(window=self.fast_window, min_periods=self.fast_window).mean()
        slow_sma = synthetic_price.rolling(window=self.slow_window, min_periods=self.slow_window).mean()

        long_mask = (fast_sma > slow_sma) & fast_sma.notna() & slow_sma.notna()
        signals.loc[long_mask] = 1
        return signals.rename("signal")


class SeededRandomStrategy(BaseStrategy):
    """Deterministic random long-or-flat benchmark using a local seeded RNG."""

    name = "seeded_random_v1"
    dataset = "features_daily"
    required_input_columns = ()
    requires_return_column = True

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype="int64")
        if df.empty:
            return signals.rename("signal")

        tradable_rows = valid_return_mask(df)
        if not tradable_rows.any():
            return signals.rename("signal")

        rng = np.random.default_rng(self.seed)
        signals.loc[tradable_rows] = rng.integers(0, 2, size=int(tradable_rows.sum()), dtype=np.int64)
        return signals.rename("signal")
