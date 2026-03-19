from __future__ import annotations

import pandas as pd

from src.research.strategy_base import BaseStrategy

from src.research.strategies._helpers import resolve_return_column


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy driven by rolling average returns."""

    name = "momentum_v1"
    dataset = "features_daily"

    def __init__(self, *, lookback_short: int, lookback_long: int) -> None:
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return_column = resolve_return_column(df)
        short_trend = df[return_column].rolling(window=self.lookback_short, min_periods=1).mean()
        long_trend = df[return_column].rolling(window=self.lookback_long, min_periods=1).mean()
        return ((short_trend > long_trend).astype("int64") - (short_trend < long_trend).astype("int64")).rename(
            "signal"
        )


class MeanReversionStrategy(BaseStrategy):
    """Simple mean-reversion strategy based on a rolling return z-score."""

    name = "mean_reversion_v1"
    dataset = "features_daily"

    def __init__(self, *, zscore_window: int, entry_threshold: float) -> None:
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return_column = resolve_return_column(df)
        rolling_mean = df[return_column].rolling(window=self.zscore_window, min_periods=1).mean()
        rolling_std = (
            df[return_column]
            .rolling(window=self.zscore_window, min_periods=1)
            .std()
            .replace(0.0, pd.NA)
        )
        zscore = ((df[return_column] - rolling_mean) / rolling_std).fillna(0.0)
        threshold = abs(float(self.entry_threshold))

        signals = pd.Series(0, index=df.index, dtype="int64")
        signals.loc[zscore <= -threshold] = 1
        signals.loc[zscore >= threshold] = -1
        return signals.rename("signal")
