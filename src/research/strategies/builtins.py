from __future__ import annotations

import pandas as pd

from src.research.strategy_base import BaseStrategy

from src.research.strategies._helpers import resolve_return_column


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy driven by rolling average returns."""

    name = "momentum_v1"
    dataset = "features_daily"
    required_input_columns = ()
    requires_return_column = True

    def __init__(self, *, lookback_short: int, lookback_long: int) -> None:
        if lookback_short <= 0 or lookback_long <= 0:
            raise ValueError("MomentumStrategy lookback windows must be positive integers.")
        if lookback_short >= lookback_long:
            raise ValueError("MomentumStrategy lookback_short must be smaller than lookback_long.")

        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return_column = resolve_return_column(df)
        returns = df[return_column]
        if "symbol" in df.columns:
            grouped_returns = returns.groupby(df["symbol"], sort=False)
            short_trend = grouped_returns.transform(
                lambda series: series.rolling(window=self.lookback_short, min_periods=self.lookback_short).mean()
            )
            long_trend = grouped_returns.transform(
                lambda series: series.rolling(window=self.lookback_long, min_periods=self.lookback_long).mean()
            )
        else:
            short_trend = returns.rolling(window=self.lookback_short, min_periods=self.lookback_short).mean()
            long_trend = returns.rolling(window=self.lookback_long, min_periods=self.lookback_long).mean()

        signal = (short_trend > long_trend).astype("int64") - (short_trend < long_trend).astype("int64")
        return signal.rename("signal")


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion strategy based on rolling close-price z-scores."""

    name = "mean_reversion_v1"
    dataset = "features_daily"
    required_input_columns = ("close",)

    def __init__(self, *, lookback: int, threshold: float) -> None:
        if lookback <= 1:
            raise ValueError("MeanReversionStrategy lookback must be greater than 1.")
        if threshold <= 0:
            raise ValueError("MeanReversionStrategy threshold must be greater than 0.")

        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("MeanReversionStrategy requires a 'close' column in the feature dataset.")

        closes = df["close"].astype("float64")
        if "symbol" in df.columns:
            grouped_closes = closes.groupby(df["symbol"], sort=False)
            rolling_mean = grouped_closes.transform(
                lambda series: series.rolling(window=self.lookback, min_periods=self.lookback).mean()
            )
            rolling_std = grouped_closes.transform(
                lambda series: series.rolling(window=self.lookback, min_periods=self.lookback).std()
            )
        else:
            rolling_mean = closes.rolling(window=self.lookback, min_periods=self.lookback).mean()
            rolling_std = closes.rolling(window=self.lookback, min_periods=self.lookback).std()

        zscore = ((closes - rolling_mean) / rolling_std.replace(0.0, pd.NA)).fillna(0.0)

        signals = pd.Series(0, index=df.index, dtype="int64")
        signals.loc[zscore < -self.threshold] = 1
        signals.loc[zscore > self.threshold] = -1
        return signals.rename("signal")
