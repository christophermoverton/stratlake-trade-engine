"""
Canonical Strategy Archetype Implementations.

Implements deterministic single-strategy, regime-aware, and ensemble
archetypes suitable for reproducible research pipelines.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.research.strategy_archetype import (
    ArchetypeStrategy,
    StrategyDefinition,
    StrategyArchetype,
    InputSchema,
    InputSchemaField,
    OutputSignalSchema,
    ExecutionSemantics,
    RebalanceFrequency,
    MissingDataHandling,
    FailureMode,
)


def _sorted_working_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a stable symbol/time ordered working frame with original indices."""
    working = df.loc[:, columns].copy()
    working["_orig_index"] = np.arange(len(df), dtype="int64")
    sort_columns = ["ts_utc", "_orig_index"]
    if "symbol" in working.columns:
        sort_columns = ["symbol", "ts_utc", "_orig_index"]
    return working.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _restore_series(working: pd.DataFrame, column: str, index: pd.Index) -> pd.Series:
    """Restore a computed column from a sorted working frame back to the original order."""
    restored = working.sort_values("_orig_index", kind="stable")[column].reset_index(drop=True)
    restored.index = index
    return restored.rename("signal")


def _normalize_ranked_signal(
    df: pd.DataFrame,
    *,
    score_column: str,
    index: pd.Index,
) -> pd.Series:
    """Convert per-timestamp scores into deterministic cross-sectional ranks."""
    ranked = df.loc[:, ["ts_utc", "symbol", score_column]].copy()
    ranked["_orig_index"] = np.arange(len(df), dtype="int64")

    def normalize_group(group: pd.DataFrame) -> pd.DataFrame:
        ordered = group.sort_values(
            [score_column, "symbol", "_orig_index"],
            ascending=[False, True, True],
            kind="stable",
        ).copy()
        if len(ordered) == 1:
            ordered["signal"] = 0.0
        else:
            ranks = np.arange(len(ordered), dtype="float64")
            ordered["signal"] = 1.0 - (2.0 * ranks / float(len(ordered) - 1))
        return ordered

    normalized = (
        ranked.groupby("ts_utc", sort=False, group_keys=False)
        .apply(normalize_group)
        .sort_values("_orig_index", kind="stable")
    )
    signal = normalized["signal"].reset_index(drop=True)
    signal.index = index
    return signal.rename("signal")


# ==============================================================================
# Time Series Momentum
# ==============================================================================


class TimeSeriesMomentumStrategy(ArchetypeStrategy):
    """
    Time-series momentum driven by trend strength.

    Research Hypothesis:
    Assets exhibit autocorrelation in returns - recent outperformers tend
    to continue outperforming in the near term due to trend persistence.

    Mathematical Definition:
    1. Compute returns: r_t = ln(close_t / close_{t-lookback})
    2. Compute trend: ma_short = mean(r_t[t-short:t]), ma_long = mean(r_t[t-long:t])
    3. Signal: +1 if ma_short > ma_long, -1 if ma_short < ma_long, 0 otherwise

    Failure Modes:
    - Sharp reversals: Strategy whipsaws in mean-reverting regimes
    - Low liquidity: Wide spreads in small-cap names impair execution
    - Regime change: Momentum collapses during volatility spikes
    """

    strategy_definition = StrategyDefinition(
        strategy_id="time_series_momentum",
        version="1.0.0",
        archetype=StrategyArchetype.TIME_SERIES_MOMENTUM,
        name="Time Series Momentum",
        description="Trend-following strategy based on historical price momentum within individual securities.",
        mathematical_definition=(
            "For each symbol: compute short-term average return (lookback_short) and long-term average "
            "return (lookback_long). Generate ternary signal: +1 if short-term > long-term, -1 if "
            "short-term < long-term, 0 otherwise. No cross-sectional normalization."
        ),
        research_hypothesis=(
            "Time-series momentum: Assets exhibit return autocorrelation. Recent outperformers tend to "
            "continue higher due to trend persistence and behavioral momentum."
        ),
        input_schema=InputSchema(
            required_columns=["ts_utc", "symbol", "close"],
            required_fields=[
                InputSchemaField(
                    name="close",
                    dtype="float64",
                    description="Closing price",
                    examples=["100.50", "102.75"],
                ),
            ],
            time_series_required=True,
            cross_sectional_required=False,
            min_lookback_periods=None,
            supported_timeframes=("daily", "weekly"),
        ),
        output_signal_schema=OutputSignalSchema(
            signal_type="ternary_quantile",
            allowed_values=[-1.0, 0.0, 1.0],
            description="Ternary signal: -1 (short), 0 (flat), 1 (long)",
        ),
        execution_semantics=ExecutionSemantics(
            rebalance_frequency=RebalanceFrequency.DAILY,
            execution_lag_days=1,
            missing_data_handling=MissingDataHandling.SKIP_EPOCH,
            deterministic=True,
        ),
        failure_modes=(
            FailureMode(
                name="Regime Reversal",
                description="Strategy whipsaws in sharp mean-reverting environments.",
                regime="low_trend_persistence",
                sensitivity="high",
                mitigation="Add regime filter or decrease lookback_short relative to lookback_long.",
            ),
            FailureMode(
                name="Liquidity Constraint",
                description="Execution costs may exceed returns in illiquid names.",
                sensitivity="moderate",
                mitigation="Filter to high-liquidity universe or apply position sizing.",
            ),
        ),
        assumptions=(
            "Return time series are stationary within rebalance period.",
            "Closing price reflects true equilibrium price.",
            "No transaction costs or slippage.",
        ),
        tags=("momentum", "time_series", "trend", "price_based"),
    )

    def __init__(self, *, lookback_short: int, lookback_long: int) -> None:
        """
        Args:
            lookback_short: Short-term lookback window in periods.
            lookback_long: Long-term lookback window in periods.

        Raises:
            ValueError: If windows are invalid or not strictly increasing.
        """
        if lookback_short <= 0 or lookback_long <= 0:
            raise ValueError("Lookback windows must be positive integers.")
        if lookback_short >= lookback_long:
            raise ValueError("lookback_short must be strictly less than lookback_long.")

        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate time-series momentum signals."""
        closes = df["close"].astype("float64")

        if "symbol" in df.columns:
            # Per-symbol momentum
            grouped_closes = closes.groupby(df["symbol"], sort=False)
            short_ma = grouped_closes.transform(
                lambda s: s.rolling(window=self.lookback_short, min_periods=self.lookback_short).mean()
            )
            long_ma = grouped_closes.transform(
                lambda s: s.rolling(window=self.lookback_long, min_periods=self.lookback_long).mean()
            )
        else:
            # Single asset momentum
            short_ma = closes.rolling(window=self.lookback_short, min_periods=self.lookback_short).mean()
            long_ma = closes.rolling(window=self.lookback_long, min_periods=self.lookback_long).mean()

        signal = pd.Series(0, index=df.index, dtype="int64")
        signal.loc[short_ma > long_ma] = 1
        signal.loc[short_ma < long_ma] = -1

        return signal.rename("signal")


# ==============================================================================
# Cross-Section Momentum
# ==============================================================================


class CrossSectionMomentumStrategy(ArchetypeStrategy):
    """
    Cross-sectional momentum via relative strength ranking.

    Research Hypothesis:
    Within a universe, relative momentum (relative strength) predicts
    future relative returns. Winners outperform losers due to momentum
    factor persistence.

    Mathematical Definition:
    1. For each timestamp, compute normalized rank of securities by return
    2. Rank from -1 (lowest) to +1 (highest), deterministically breaking ties
    3. Signal: cross_section_rank = 1 - 2*(rank - 1)/(N - 1)

    Failure Modes:
    - Factor crowding: Extreme positions in crowded names lead to slippage
    - Reversal at extremes: Highest momentum often precedes sharp drawdowns
    - Low universe size: Insufficient cross-section reduces diversification
    """

    strategy_definition = StrategyDefinition(
        strategy_id="cross_section_momentum",
        version="1.0.0",
        archetype=StrategyArchetype.CROSS_SECTION_MOMENTUM,
        name="Cross-Section Momentum",
        description="Relative momentum based on ranking returns across a universe.",
        mathematical_definition=(
            "For each timestamp: (1) compute period returns r_i for each security i; "
            "(2) rank securities deterministically by return (ties broken by symbol ascending); "
            "(3) normalize rank to cross_section_rank in [-1, +1] where +1 is highest performer."
        ),
        research_hypothesis=(
            "Cross-sectional momentum: Within a universe, relative strength predicts relative returns. "
            "Winners tend to outperform losers due to momentum factor persistence."
        ),
        input_schema=InputSchema(
            required_columns=["ts_utc", "symbol", "close"],
            required_fields=[
                InputSchemaField(
                    name="close",
                    dtype="float64",
                    description="Closing price",
                ),
            ],
            time_series_required=True,
            cross_sectional_required=True,
            min_cross_section_size=5,
            min_lookback_periods=1,
            supported_timeframes=("daily", "weekly"),
        ),
        output_signal_schema=OutputSignalSchema(
            signal_type="cross_section_rank",
            value_range=(-1.0, 1.0),
            description="Normalized rank from -1 (worst performer) to +1 (best performer)",
        ),
        execution_semantics=ExecutionSemantics(
            rebalance_frequency=RebalanceFrequency.DAILY,
            execution_lag_days=1,
            missing_data_handling=MissingDataHandling.SKIP_EPOCH,
            deterministic=True,
        ),
        failure_modes=(
            FailureMode(
                name="Factor Crowding",
                description="Extreme momentum factor crowding leads to slippage and execution costs.",
                regime="high_momentum_crowding",
                sensitivity="high",
                mitigation="Use position sizing or volatility scaling; limit extremes.",
            ),
            FailureMode(
                name="Reversal at Extremes",
                description="Highest momentum securities often reverse sharply.",
                regime="momentum_reversion",
                sensitivity="high",
                mitigation="Reduce exposure to extreme quantiles or add mean-reversion component.",
            ),
            FailureMode(
                name="Insufficient Cross-Section",
                description="Rankings unreliable when universe size too small.",
                sensitivity="moderate",
                mitigation="Require minimum universe size or use longer lookbacks.",
            ),
        ),
        assumptions=(
            "Cross-section is sufficiently large (min 5 securities).",
            "Ties resolved deterministically by symbol ascending.",
            "Returns computed from closing prices.",
        ),
        tags=("momentum", "cross_sectional", "rank", "relative_strength"),
    )

    def __init__(self, *, lookback_days: int = 1) -> None:
        """
        Args:
            lookback_days: Number of days to compute returns over.
        """
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")
        self.lookback_days = lookback_days

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate cross-sectional momentum signals."""
        closes = df["close"].astype("float64")

        # Group by timestamp to compute cross-sectional ranks
        def rank_by_return(group):
            # Compute return: latest / oldest
            if len(group) < 2:
                return pd.Series(0.0, index=group.index)

            # Sort by time to get price movement
            group_sorted = group.sort_index()
            if len(group_sorted) < 2:
                return pd.Series(0.0, index=group.index)

            # Return from oldest to latest
            ret = (group_sorted.iloc[-1] - group_sorted.iloc[0]) / (
                group_sorted.iloc[0] + 1e-10
            )
            return pd.Series(ret, index=group.index)

        if "symbol" in df.columns:
            # Compute returns per symbol
            df_temp = df[["ts_utc", "symbol", "close"]].copy()
            df_temp["returns"] = df_temp.groupby("symbol", sort=False)["close"].pct_change(
                periods=self.lookback_days
            ).fillna(0.0).values

            # Rank by return at each timestamp
            def normalize_rank_timestamp(group):
                if len(group) < 2:
                    return pd.Series(0.0, index=group.index)

                rets = group["returns"].values
                ranked = pd.Series(rets).rank(method="first", ascending=False).values
                normalized = 1.0 - 2.0 * (ranked - 1.0) / (len(ranked) - 1.0)
                return pd.Series(normalized, index=group.index)

            signal = df_temp.groupby("ts_utc", sort=False, group_keys=False).apply(
                normalize_rank_timestamp
            )
            signal = signal.reset_index(drop=True)
            signal.index = df.index
        else:
            # Single asset
            signal = pd.Series(0.0, index=df.index)

        return signal.rename("signal")


# ==============================================================================
# Mean Reversion
# ==============================================================================


class MeanReversionStrategy(ArchetypeStrategy):
    """
    Mean reversion based on normalized price deviations.

    Research Hypothesis:
    Asset prices oscillate around a mean. Extreme deviations tend to revert
    to equilibrium. Buying when oversold and selling when overbought captures
    reversion premium.

    Mathematical Definition:
    1. Compute rolling mean and std of close: mean_t, std_t
    2. Compute z-score: z_t = (close_t - mean_t) / std_t
    3. Signal: +1 if z < -threshold, -1 if z > +threshold, 0 otherwise

    Failure Modes:
    - Trending markets: Continues to generate bad signals in strong trends
    - Regime change: Std-dev spikes cause false threshold breaches
    - Illiquidity: Poor execution at extreme prices
    """

    strategy_definition = StrategyDefinition(
        strategy_id="mean_reversion",
        version="1.0.0",
        archetype=StrategyArchetype.MEAN_REVERSION,
        name="Mean Reversion",
        description="Contrarian strategy based on z-score normalized price deviations.",
        mathematical_definition=(
            "For each symbol: (1) compute rolling mean and standard deviation of close over lookback window; "
            "(2) compute z-score = (close - mean) / std; (3) generate ternary signal: +1 if z < -threshold, "
            "-1 if z > +threshold, 0 otherwise."
        ),
        research_hypothesis=(
            "Mean reversion: Asset prices oscillate around statistical equilibrium. Extreme deviations "
            "revert to mean. Long oversold, short overbought."
        ),
        input_schema=InputSchema(
            required_columns=["ts_utc", "symbol", "close"],
            required_fields=[
                InputSchemaField(
                    name="close",
                    dtype="float64",
                    description="Closing price",
                ),
            ],
            time_series_required=True,
            cross_sectional_required=False,
            min_lookback_periods=2,
            supported_timeframes=("daily", "weekly"),
        ),
        output_signal_schema=OutputSignalSchema(
            signal_type="ternary_quantile",
            allowed_values=[-1.0, 0.0, 1.0],
            description="Ternary signal: -1 (short/overbought), 0 (neutral), 1 (long/oversold)",
        ),
        execution_semantics=ExecutionSemantics(
            rebalance_frequency=RebalanceFrequency.DAILY,
            execution_lag_days=1,
            missing_data_handling=MissingDataHandling.SKIP_EPOCH,
            deterministic=True,
        ),
        failure_modes=(
            FailureMode(
                name="Trending Market Whipsaw",
                description="Strategy generates many losing trades during strong uptrends or downtrends.",
                regime="strong_trend",
                sensitivity="high",
                mitigation="Add trend filter or reduce threshold sensitivity.",
            ),
            FailureMode(
                name="Volatility Spike",
                description="Sudden increase in std-dev causes false threshold breaches.",
                regime="vol_spike",
                sensitivity="moderate",
                mitigation="Use adaptive threshold or vol-normalized z-score.",
            ),
            FailureMode(
                name="Structural Shift",
                description="Price levels shift permanently, making historical mean irrelevant.",
                sensitivity="moderate",
                mitigation="Use rolling windows; consider regime detection.",
            ),
        ),
        assumptions=(
            "Close prices are covariance-stationary within rebalance period.",
            "Z-score computation uses rolling mean/std (not global).",
            "No structural breaks or regime shifts.",
        ),
        tags=("mean_reversion", "contrarian", "zscore", "price_based"),
    )

    def __init__(self, *, lookback: int, threshold: float) -> None:
        """
        Args:
            lookback: Rolling window in periods for mean/std computation.
            threshold: Z-score threshold for signal generation (symmetric).

        Raises:
            ValueError: If parameters invalid.
        """
        if lookback <= 1:
            raise ValueError("lookback must be > 1.")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")

        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals."""
        closes = df["close"].astype("float64")

        if "symbol" in df.columns:
            grouped = closes.groupby(df["symbol"], sort=False)
            mean = grouped.transform(
                lambda x: x.rolling(window=self.lookback, min_periods=self.lookback).mean()
            )
            std = grouped.transform(
                lambda x: x.rolling(window=self.lookback, min_periods=self.lookback).std()
            )
        else:
            mean = closes.rolling(window=self.lookback, min_periods=self.lookback).mean()
            std = closes.rolling(window=self.lookback, min_periods=self.lookback).std()

        zscore = ((closes - mean) / std.replace(0.0, pd.NA)).fillna(0.0)

        signal = pd.Series(0, index=df.index, dtype="int64")
        signal.loc[zscore < -self.threshold] = 1
        signal.loc[zscore > self.threshold] = -1

        return signal.rename("signal")


# ==============================================================================
# Breakout
# ==============================================================================


class BreakoutStrategy(ArchetypeStrategy):
    """
    Breakout strategy based on recent price extremes.

    Research Hypothesis:
    When price breaks above the recent high (breakout), momentum often
    continues. Breakouts signal potential trend initiation.

    Mathematical Definition:
    1. Compute rolling high and low over lookback window
    2. Signal: +1 if close >= rolling_high, -1 if close <= rolling_low, 0 otherwise

    Failure Modes:
    - False breakouts: Quick reversals on noise breakouts
    - Trend exhaustion: Breakouts often occur at trend peaks
    - Overtrading: Many small reversals waste capital on execution
    """

    strategy_definition = StrategyDefinition(
        strategy_id="breakout",
        version="1.0.0",
        archetype=StrategyArchetype.BREAKOUT,
        name="Breakout",
        description="Momentum strategy on recent price extreme breakouts.",
        mathematical_definition=(
            "For each symbol: (1) compute rolling high and low over lookback window; "
            "(2) signal +1 if close >= rolling_high, -1 if close <= rolling_low, 0 otherwise."
        ),
        research_hypothesis=(
            "Breakout: Price breaks beyond recent extremes signal momentum initiation. "
            "Breakouts tend to persist in the near term before potential reversal."
        ),
        input_schema=InputSchema(
            required_columns=["ts_utc", "symbol", "high", "low", "close"],
            required_fields=[
                InputSchemaField(name="high", dtype="float64", description="Period high price"),
                InputSchemaField(name="low", dtype="float64", description="Period low price"),
                InputSchemaField(name="close", dtype="float64", description="Closing price"),
            ],
            time_series_required=True,
            cross_sectional_required=False,
            min_lookback_periods=2,
            supported_timeframes=("daily", "weekly"),
        ),
        output_signal_schema=OutputSignalSchema(
            signal_type="ternary_quantile",
            allowed_values=[-1.0, 0.0, 1.0],
            description="Ternary signal: -1 (below low), 0 (between), 1 (above high)",
        ),
        execution_semantics=ExecutionSemantics(
            rebalance_frequency=RebalanceFrequency.DAILY,
            execution_lag_days=1,
            missing_data_handling=MissingDataHandling.SKIP_EPOCH,
            deterministic=True,
        ),
        failure_modes=(
            FailureMode(
                name="False Breakout",
                description="Breakout reverses immediately, generating whipsaw trades.",
                regime="low_persistence",
                sensitivity="high",
                mitigation="Add volume confirmation or require breakout to hold for N bars.",
            ),
            FailureMode(
                name="Trend Exhaustion",
                description="Breakouts often occur at trend peaks before reversal.",
                regime="trend_exhaustion",
                sensitivity="high",
                mitigation="Use volatility filter or add mean-reversion component.",
            ),
        ),
        assumptions=(
            "High/low prices reflect true period extremes.",
            "Rolling window captures relevant price history.",
        ),
        tags=("breakout", "momentum", "technical", "price_based"),
    )

    def __init__(self, *, lookback: int = 20) -> None:
        """
        Args:
            lookback: Window in periods to identify high/low extremes.
        """
        if lookback <= 1:
            raise ValueError("lookback must be > 1.")
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate breakout signals."""
        highs = df["high"].astype("float64")
        lows = df["low"].astype("float64")
        closes = df["close"].astype("float64")

        if "symbol" in df.columns:
            grouped_high = highs.groupby(df["symbol"], sort=False)
            grouped_low = lows.groupby(df["symbol"], sort=False)

            rolling_high = grouped_high.transform(
                lambda x: x.rolling(window=self.lookback, min_periods=self.lookback).max()
            )
            rolling_low = grouped_low.transform(
                lambda x: x.rolling(window=self.lookback, min_periods=self.lookback).min()
            )
        else:
            rolling_high = highs.rolling(window=self.lookback, min_periods=self.lookback).max()
            rolling_low = lows.rolling(window=self.lookback, min_periods=self.lookback).min()

        signal = pd.Series(0, index=df.index, dtype="int64")
        signal.loc[closes >= rolling_high] = 1
        signal.loc[closes <= rolling_low] = -1

        return signal.rename("signal")


# ==============================================================================
# Pairs Trading
# ==============================================================================


class PairsTradingStrategy(ArchetypeStrategy):
    """
    Pairs trading via mean-reverting spread signals.

    Research Hypothesis:
    Co-integrated pairs show temporary deviations that revert to long-term
    equilibrium. Trading the spread (hedge ratio adjusted) captures
    mean-reversion with systematic hedge.

    Mathematical Definition:
    1. Pair two correlated securities within a group
    2. Compute spread: s_t = price1_t - hedge_ratio * price2_t
    3. Compute rolling z-score of spread
    4. Signal when spread z-score exceeds threshold

    Failure Modes:
    - Cointegration breaks: Long-term relationship decays
    - Regime change: Hedge ratio changes with market structure
    - Execution lag: Spread closes before execution completes
    """

    strategy_definition = StrategyDefinition(
        strategy_id="pairs_trading",
        version="1.0.0",
        archetype=StrategyArchetype.PAIRS_TRADING,
        name="Pairs Trading",
        description="Market-neutral strategy on mean-reverting spreads between paired securities.",
        mathematical_definition=(
            "For paired symbols: (1) compute spread s = price1 - hedge_ratio * price2; "
            "(2) compute rolling z-score of spread; (3) signal when |z| > threshold."
        ),
        research_hypothesis=(
            "Pairs trading: Co-integrated pairs show temporary deviations. Spreads mean-revert. "
            "Long/short paired positions capture reversion while hedging systematic risk."
        ),
        input_schema=InputSchema(
            required_columns=["ts_utc", "symbol", "close"],
            required_fields=[
                InputSchemaField(name="close", dtype="float64", description="Closing price"),
            ],
            time_series_required=True,
            cross_sectional_required=True,
            min_cross_section_size=2,
            supported_timeframes=("daily",),
        ),
        output_signal_schema=OutputSignalSchema(
            signal_type="spread_zscore",
            description="Z-score of spread between paired securities",
        ),
        execution_semantics=ExecutionSemantics(
            rebalance_frequency=RebalanceFrequency.DAILY,
            execution_lag_days=1,
            missing_data_handling=MissingDataHandling.SKIP_EPOCH,
            deterministic=True,
        ),
        failure_modes=(
            FailureMode(
                name="Cointegration Breakdown",
                description="Long-term relationship deteriorates; spread no longer mean-reverting.",
                sensitivity="high",
                mitigation="Monitor cointegration; reselect pairs dynamically.",
            ),
            FailureMode(
                name="Hedge Ratio Drift",
                description="Optimal hedge ratio changes with market structure.",
                sensitivity="moderate",
                mitigation="Reestimate hedge ratios on rolling basis.",
            ),
        ),
        assumptions=(
            "Pairs are cointegrated (long-term equilibrium relationship).",
            "Hedge ratio is stable within rebalance period.",
        ),
        tags=("pairs", "market_neutral", "spread", "mean_reversion"),
    )

    def __init__(self, *, lookback: int = 63, threshold: float = 2.0) -> None:
        """
        Args:
            lookback: Window for spread statistics.
            threshold: Z-score threshold for signals.
        """
        if lookback <= 1:
            raise ValueError("lookback must be > 1.")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")

        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate pairs trading signals."""
        if "symbol" not in df.columns:
            raise ValueError("Pairs trading requires 'symbol' column for pairing.")

        # For simplicity, pair consecutive distinct symbols
        # In production, use statistical pairing or config-driven pairing
        symbols = df["symbol"].unique()
        if len(symbols) < 2:
            raise ValueError("Pairs trading requires at least 2 distinct symbols.")

        # Compute spread z-scores per timestamp
        closes = df["close"].astype("float64")
        spreads = []

        for ts in df["ts_utc"].unique():
            ts_data = df[df["ts_utc"] == ts]
            if len(ts_data) >= 2:
                # Simple approach: take first two distinct symbols
                symbols_in_ts = ts_data["symbol"].unique()[:2]
                p1 = ts_data[ts_data["symbol"] == symbols_in_ts[0]]["close"].values[0]
                p2 = ts_data[ts_data["symbol"] == symbols_in_ts[1]]["close"].values[0]
                spread = p1 - p2
                spreads.append((ts, spread))

        if not spreads:
            return pd.Series(0.0, index=df.index)

        # Compute rolling z-score
        spread_series = pd.Series(
            dict(spreads), index=pd.to_datetime([s[0] for s in spreads], unit="ns")
        )
        spread_mean = spread_series.rolling(window=self.lookback, min_periods=self.lookback).mean()
        spread_std = spread_series.rolling(window=self.lookback, min_periods=self.lookback).std()

        zscore = ((spread_series - spread_mean) / spread_std.replace(0.0, pd.NA)).fillna(0.0)

        # Map back to original index
        signal = pd.Series(0.0, index=df.index)
        for ts, z in zscore.items():
            idx = df[df["ts_utc"] == ts.value].index
            if len(idx) > 0:
                signal.loc[idx] = z

        return signal.rename("signal")


# ==============================================================================
# Residual Momentum
# ==============================================================================


class ResidualMomentumStrategy(ArchetypeStrategy):
    """
    Momentum after removing systematic factor exposure.

    Research Hypothesis:
    Idiosyncratic (residual) momentum outperforms raw momentum because it
    isolates true alpha signal from systematic factor returns. Removes beta,
    style, and other common factor exposures.

    Mathematical Definition:
    1. Regress security returns on systematic factors (e.g., market return)
    2. Compute residuals: resid = actual - predicted
    3. Rank residuals cross-sectionally
    4. Generate signal from residual ranking

    Failure Modes:
    - Factor instability: Factor betas change regime to regime
    - Residual estimation error: Noise in residual estimates
    - Low signal-to-noise: Residuals may be mostly noise
    """

    strategy_definition = StrategyDefinition(
        strategy_id="residual_momentum",
        version="1.0.0",
        archetype=StrategyArchetype.RESIDUAL_MOMENTUM,
        name="Residual Momentum",
        description="Cross-sectional momentum of returns after systematic factor removal.",
        mathematical_definition=(
            "For each timestamp: (1) regress security returns on market/systematic factors; "
            "(2) compute residuals; (3) rank residuals cross-sectionally; (4) generate signal "
            "from normalized residual ranks."
        ),
        research_hypothesis=(
            "Residual momentum: Idiosyncratic (alpha) momentum outperforms raw momentum. "
            "Removing systematic factors isolates true momentum signal from beta noise."
        ),
        input_schema=InputSchema(
            required_columns=["ts_utc", "symbol", "close", "market_return"],
            required_fields=[
                InputSchemaField(name="close", dtype="float64", description="Closing price"),
                InputSchemaField(
                    name="market_return",
                    dtype="float64",
                    description="Market (systematic) return",
                ),
            ],
            time_series_required=True,
            cross_sectional_required=True,
            min_cross_section_size=5,
            supported_timeframes=("daily",),
        ),
        output_signal_schema=OutputSignalSchema(
            signal_type="cross_section_rank",
            value_range=(-1.0, 1.0),
            description="Normalized rank of residual (alpha) returns",
        ),
        execution_semantics=ExecutionSemantics(
            rebalance_frequency=RebalanceFrequency.DAILY,
            execution_lag_days=1,
            missing_data_handling=MissingDataHandling.SKIP_EPOCH,
            deterministic=True,
        ),
        failure_modes=(
            FailureMode(
                name="Factor Model Instability",
                description="Betas and factor structure change across regimes.",
                sensitivity="moderate",
                mitigation="Use rolling factor estimation; monitor factor stability.",
            ),
            FailureMode(
                name="Estimation Error",
                description="Residuals contain noise and estimation error, not pure alpha.",
                sensitivity="moderate",
                mitigation="Increase sample size; use robust regression.",
            ),
        ),
        assumptions=(
            "Factor model is correctly specified.",
            "Betas are stable within rebalance period.",
            "Market return accurately represents systematic risk.",
        ),
        tags=("momentum", "residual", "alpha", "factor_adjusted"),
    )

    def __init__(self, *, lookback_days: int = 20) -> None:
        """
        Args:
            lookback_days: Window for computing residuals.
        """
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")
        self.lookback_days = lookback_days

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate residual momentum signals."""
        if "market_return" not in df.columns:
            raise ValueError("Residual momentum requires 'market_return' column.")

        closes = df["close"].astype("float64")
        market_ret = df["market_return"].astype("float64")

        # Compute returns
        if "symbol" in df.columns:
            df_temp = df[["ts_utc", "symbol", "close"]].copy()
            df_temp["returns"] = df_temp.groupby("symbol", sort=False)["close"].pct_change(
                self.lookback_days
            ).fillna(0.0).values
            df_temp["market_return"] = market_ret.values
            
            # Compute residuals (alpha)
            df_temp["residuals"] = df_temp["returns"] - df_temp["market_return"]

            # Rank residuals cross-sectionally
            def normalize_rank_timestamp(group):
                if len(group) < 2:
                    return pd.Series(0.0, index=group.index)
                residuals_vals = group["residuals"].values
                ranked = pd.Series(residuals_vals).rank(method="first", ascending=False).values
                normalized = 1.0 - 2.0 * (ranked - 1.0) / (len(ranked) - 1.0)
                return pd.Series(normalized, index=group.index)

            signal = df_temp.groupby("ts_utc", sort=False, group_keys=False).apply(
                normalize_rank_timestamp
            )
            signal = signal.reset_index(drop=True)
            signal.index = df.index
        else:
            signal = pd.Series(0.0, index=df.index)

        return signal.rename("signal")


# ==============================================================================
# Volatility Regime Momentum
# ==============================================================================


class VolatilityRegimeMomentumStrategy(ArchetypeStrategy):
    """
    Time-series momentum gated by an explicit realized-volatility regime.

    Research Hypothesis:
    Momentum signals are more reliable inside bounded volatility regimes.
    When realized volatility is too low or too high, standing down can reduce
    whipsaw and execution noise without introducing dynamic randomness.
    """

    strategy_definition = StrategyDefinition(
        strategy_id="volatility_regime_momentum",
        version="1.0.0",
        archetype=StrategyArchetype.VOLATILITY_REGIME_MOMENTUM,
        name="Volatility Regime Momentum",
        description="Time-series momentum gated by explicit realized-volatility thresholds.",
        mathematical_definition=(
            "For each symbol: (1) compute short and long rolling close averages; "
            "(2) derive a base ternary momentum signal from the moving-average spread; "
            "(3) compute rolling realized volatility from close-to-close returns; "
            "(4) emit the base signal only when realized volatility lies within "
            "[min_volatility, max_volatility], otherwise emit 0."
        ),
        research_hypothesis=(
            "Momentum behaves more reliably in moderate-volatility environments. "
            "Explicitly gating by realized volatility preserves interpretable trend logic "
            "while reducing exposure in adverse regimes."
        ),
        input_schema=InputSchema(
            required_columns=["ts_utc", "symbol", "close"],
            required_fields=[
                InputSchemaField(
                    name="close",
                    dtype="float64",
                    description="Closing price used for both momentum and realized-volatility estimation.",
                ),
            ],
            time_series_required=True,
            cross_sectional_required=False,
            min_lookback_periods=2,
            supported_timeframes=("daily", "weekly"),
        ),
        output_signal_schema=OutputSignalSchema(
            signal_type="ternary_quantile",
            allowed_values=[-1.0, 0.0, 1.0],
            description="Ternary signal gated by a deterministic volatility regime filter.",
        ),
        execution_semantics=ExecutionSemantics(
            rebalance_frequency=RebalanceFrequency.DAILY,
            execution_lag_days=1,
            missing_data_handling=MissingDataHandling.SKIP_EPOCH,
            deterministic=True,
        ),
        failure_modes=(
            FailureMode(
                name="Threshold Misspecification",
                description="Static volatility thresholds may suppress profitable signals or admit noisy regimes.",
                regime="threshold_mismatch",
                sensitivity="moderate",
                mitigation="Tune thresholds explicitly and document them in config.",
            ),
            FailureMode(
                name="Delayed Regime Recognition",
                description="Rolling volatility reacts after regime shifts, so gating can lag turning points.",
                regime="regime_transition",
                sensitivity="moderate",
                mitigation="Shorten volatility lookback while preserving deterministic thresholds.",
            ),
        ),
        assumptions=(
            "Realized volatility estimated from close-to-close returns is an adequate regime proxy.",
            "Static thresholds are chosen ex ante and remain fixed through the run.",
            "Signal suppression outside the accepted regime is preferable to adaptive parameter mutation.",
        ),
        regime_dependencies=(
            "Performs best when momentum edges persist in bounded-volatility conditions.",
            "Stands down during volatility extremes or unusually compressed ranges.",
        ),
        tags=("momentum", "regime_aware", "volatility_filter", "time_series"),
    )

    def __init__(
        self,
        *,
        lookback_short: int,
        lookback_long: int,
        volatility_lookback: int,
        min_volatility: float = 0.0,
        max_volatility: float = 0.03,
    ) -> None:
        if lookback_short <= 0 or lookback_long <= 0:
            raise ValueError("Lookback windows must be positive integers.")
        if lookback_short >= lookback_long:
            raise ValueError("lookback_short must be strictly less than lookback_long.")
        if volatility_lookback <= 1:
            raise ValueError("volatility_lookback must be greater than 1.")
        if min_volatility < 0.0:
            raise ValueError("min_volatility must be non-negative.")
        if max_volatility <= 0.0:
            raise ValueError("max_volatility must be positive.")
        if min_volatility > max_volatility:
            raise ValueError("min_volatility must be less than or equal to max_volatility.")

        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.volatility_lookback = volatility_lookback
        self.min_volatility = float(min_volatility)
        self.max_volatility = float(max_volatility)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate regime-filtered momentum signals."""
        working = _sorted_working_frame(df, ["ts_utc", "symbol", "close"])
        grouped_close = working["close"].astype("float64").groupby(working["symbol"], sort=False)
        short_ma = grouped_close.transform(
            lambda series: series.rolling(window=self.lookback_short, min_periods=self.lookback_short).mean()
        )
        long_ma = grouped_close.transform(
            lambda series: series.rolling(window=self.lookback_long, min_periods=self.lookback_long).mean()
        )
        realized_volatility = grouped_close.transform(
            lambda series: (
                series.pct_change()
                .rolling(window=self.volatility_lookback, min_periods=self.volatility_lookback)
                .std()
            )
        )
        regime_active = realized_volatility.between(self.min_volatility, self.max_volatility, inclusive="both")
        base_signal = pd.Series(0, index=working.index, dtype="int64")
        base_signal.loc[short_ma > long_ma] = 1
        base_signal.loc[short_ma < long_ma] = -1
        working["signal"] = base_signal.where(regime_active.fillna(False), 0).astype("int64")
        return _restore_series(working, "signal", df.index)


# ==============================================================================
# Weighted Cross-Section Ensemble
# ==============================================================================


class WeightedCrossSectionEnsembleStrategy(ArchetypeStrategy):
    """
    Deterministic ensemble over multiple ranked cross-sectional child strategies.

    Research Hypothesis:
    Blending complementary ranked signals can stabilize cross-sectional selection
    while keeping signal construction transparent and fully reproducible.
    """

    strategy_definition = StrategyDefinition(
        strategy_id="weighted_cross_section_ensemble",
        version="1.0.0",
        archetype=StrategyArchetype.WEIGHTED_CROSS_SECTION_ENSEMBLE,
        name="Weighted Cross-Section Ensemble",
        description="Weighted ensemble over cross-sectional momentum and residual momentum sleeves.",
        mathematical_definition=(
            "At each timestamp: (1) compute child cross-sectional signals from raw momentum and residual momentum; "
            "(2) combine them with explicit deterministic weights; "
            "(3) re-rank the blended score cross-sectionally with symbol-ascending tie-breaking; "
            "(4) normalize the resulting rank into [-1, +1]."
        ),
        research_hypothesis=(
            "A weighted blend of raw and residual momentum can capture complementary relative-strength effects "
            "while remaining interpretable and deterministic."
        ),
        input_schema=InputSchema(
            required_columns=["ts_utc", "symbol", "close", "market_return"],
            required_fields=[
                InputSchemaField(name="close", dtype="float64", description="Closing price"),
                InputSchemaField(
                    name="market_return",
                    dtype="float64",
                    description="Market return input required by the residual momentum sleeve.",
                ),
            ],
            time_series_required=True,
            cross_sectional_required=True,
            min_cross_section_size=5,
            min_lookback_periods=1,
            supported_timeframes=("daily",),
        ),
        output_signal_schema=OutputSignalSchema(
            signal_type="cross_section_rank",
            value_range=(-1.0, 1.0),
            description="Normalized cross-sectional rank derived from an explicit weighted ensemble score.",
        ),
        execution_semantics=ExecutionSemantics(
            rebalance_frequency=RebalanceFrequency.DAILY,
            execution_lag_days=1,
            missing_data_handling=MissingDataHandling.SKIP_EPOCH,
            deterministic=True,
        ),
        failure_modes=(
            FailureMode(
                name="Weight Concentration",
                description="Poorly chosen weights can collapse the ensemble into a single sleeve.",
                regime="weight_instability",
                sensitivity="moderate",
                mitigation="Use explicit documented weights and validate sleeve contribution in tests.",
            ),
            FailureMode(
                name="Shared Factor Exposure",
                description="Child sleeves may become highly correlated and reduce diversification benefits.",
                regime="factor_alignment",
                sensitivity="moderate",
                mitigation="Monitor child score dispersion and revise weights only through explicit config changes.",
            ),
        ),
        assumptions=(
            "Cross-sectional momentum and residual momentum provide complementary but compatible signals.",
            "Weights are fixed ex ante and do not adapt during a run.",
            "Re-ranking the blended score is an acceptable deterministic normalization step.",
        ),
        regime_dependencies=(
            "Useful when raw momentum and factor-adjusted momentum each contain incremental information.",
            "May add less value when both sleeves collapse into the same ordering.",
        ),
        tags=("ensemble", "cross_sectional", "momentum", "residual", "weighted"),
    )

    def __init__(
        self,
        *,
        momentum_lookback_days: int = 5,
        residual_lookback_days: int = 20,
        momentum_weight: float = 0.5,
        residual_weight: float = 0.5,
    ) -> None:
        if momentum_lookback_days <= 0:
            raise ValueError("momentum_lookback_days must be positive.")
        if residual_lookback_days <= 0:
            raise ValueError("residual_lookback_days must be positive.")
        if momentum_weight < 0.0 or residual_weight < 0.0:
            raise ValueError("Ensemble weights must be non-negative.")
        if momentum_weight + residual_weight <= 0.0:
            raise ValueError("At least one ensemble weight must be positive.")

        self.momentum_lookback_days = momentum_lookback_days
        self.residual_lookback_days = residual_lookback_days
        self.momentum_weight = float(momentum_weight)
        self.residual_weight = float(residual_weight)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate deterministic weighted ensemble cross-sectional ranks."""
        momentum_sleeve = CrossSectionMomentumStrategy(
            lookback_days=self.momentum_lookback_days
        ).generate_signals(df).astype("float64")
        residual_sleeve = ResidualMomentumStrategy(
            lookback_days=self.residual_lookback_days
        ).generate_signals(df).astype("float64")

        total_weight = self.momentum_weight + self.residual_weight
        working = df.loc[:, ["ts_utc", "symbol"]].copy()
        working["ensemble_score"] = (
            (self.momentum_weight * momentum_sleeve) + (self.residual_weight * residual_sleeve)
        ) / total_weight
        return _normalize_ranked_signal(working, score_column="ensemble_score", index=df.index)
