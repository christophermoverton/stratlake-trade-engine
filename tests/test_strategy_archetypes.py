"""
Comprehensive tests for Strategy Archetype Library.

Tests:
- Determinism verification
- Schema validation
- Signal validation
- Integration with registry
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.research.strategies.archetypes import (
    TimeSeriesMomentumStrategy,
    CrossSectionMomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    PairsTradingStrategy,
    ResidualMomentumStrategy,
)
from src.research.strategies.registry import build_strategy
from src.research.strategies.validation import (
    load_strategies_registry,
    validate_strategy_config,
    validate_signal_output,
    verify_determinism,
    list_available_strategies,
    get_strategy_by_archetype,
)


@pytest.fixture
def sample_daily_df():
    """Create a sample daily feature DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                "ts_utc": date.value,  # nanosecond timestamp
                "symbol": symbol,
                "open": 100.0 + np.random.randn() * 5,
                "high": 105.0 + np.random.randn() * 5,
                "low": 95.0 + np.random.randn() * 5,
                "close": 100.0 + np.random.randn() * 5,
                "volume": 1000000,
            })

    df = pd.DataFrame(data)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


@pytest.fixture
def sample_single_symbol_df():
    """Create a sample DataFrame for a single symbol."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    # Create realistic price movements
    prices = [100.0]
    for _ in range(99):
        change = np.random.randn() * 0.02
        prices.append(prices[-1] * (1 + change))

    data = {
        "ts_utc": pd.to_datetime(dates, utc=True),
        "symbol": "TEST",
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": 1000000,
    }

    return pd.DataFrame(data)


class TestTimeSeriesMomentum:
    """Tests for TimeSeriesMomentumStrategy."""

    def test_instantiation(self):
        """Test strategy instantiation."""
        strategy = TimeSeriesMomentumStrategy(lookback_short=20, lookback_long=200)
        assert strategy.strategy_id == "time_series_momentum"
        assert strategy.version == "1.0.0"

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            TimeSeriesMomentumStrategy(lookback_short=-1, lookback_long=200)

        with pytest.raises(ValueError):
            TimeSeriesMomentumStrategy(lookback_short=200, lookback_long=20)

    def test_generate_signals_single_symbol(self, sample_single_symbol_df):
        """Test signal generation for single symbol."""
        strategy = TimeSeriesMomentumStrategy(lookback_short=10, lookback_long=30)
        signals = strategy.generate_signals(sample_single_symbol_df)

        assert len(signals) == len(sample_single_symbol_df)
        assert signals.dtype in ("int64", "int32")
        assert set(signals.unique()) <= {-1, 0, 1}

    def test_generate_signals_multiple_symbols(self, sample_daily_df):
        """Test signal generation for multiple symbols."""
        strategy = TimeSeriesMomentumStrategy(lookback_short=10, lookback_long=30)
        signals = strategy.generate_signals(sample_daily_df)

        assert len(signals) == len(sample_daily_df)
        assert set(signals.unique()) <= {-1, 0, 1}

    def test_determinism(self, sample_single_symbol_df):
        """Test that strategy produces deterministic outputs."""
        strategy = TimeSeriesMomentumStrategy(lookback_short=10, lookback_long=30)

        outputs = []
        for _ in range(5):
            signal = strategy.generate_signals(sample_single_symbol_df)
            outputs.append(signal)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert outputs[i].equals(outputs[0]), f"Run 0 != Run {i}"

    def test_output_schema_validation(self, sample_single_symbol_df):
        """Test that output conforms to ternary_quantile schema."""
        strategy = TimeSeriesMomentumStrategy(lookback_short=10, lookback_long=30)
        signals = strategy.generate_signals(sample_single_symbol_df)

        validate_signal_output("time_series_momentum", signals)


class TestCrossSectionMomentum:
    """Tests for CrossSectionMomentumStrategy."""

    def test_instantiation(self):
        """Test strategy instantiation."""
        strategy = CrossSectionMomentumStrategy(lookback_days=1)
        assert strategy.strategy_id == "cross_section_momentum"

    def test_generate_signals(self, sample_daily_df):
        """Test signal generation for cross-section."""
        strategy = CrossSectionMomentumStrategy(lookback_days=1)
        signals = strategy.generate_signals(sample_daily_df)

        assert len(signals) == len(sample_daily_df)
        assert signals.dtype == "float64"
        # Signals should be in [-1, 1] range
        assert signals.min() >= -1.0 and signals.max() <= 1.0

    def test_determinism(self, sample_daily_df):
        """Test deterministic output."""
        strategy = CrossSectionMomentumStrategy(lookback_days=1)

        outputs = []
        for _ in range(5):
            signal = strategy.generate_signals(sample_daily_df)
            outputs.append(signal)

        for i in range(1, len(outputs)):
            assert outputs[i].equals(outputs[0])

    def test_output_schema_validation(self, sample_daily_df):
        """Test output schema validation."""
        strategy = CrossSectionMomentumStrategy(lookback_days=1)
        signals = strategy.generate_signals(sample_daily_df)

        validate_signal_output("cross_section_momentum", signals)


class TestMeanReversion:
    """Tests for MeanReversionStrategy."""

    def test_instantiation(self):
        """Test strategy instantiation."""
        strategy = MeanReversionStrategy(lookback=20, threshold=2.0)
        assert strategy.strategy_id == "mean_reversion"

    def test_invalid_parameters(self):
        """Test invalid parameters."""
        with pytest.raises(ValueError):
            MeanReversionStrategy(lookback=1, threshold=2.0)

        with pytest.raises(ValueError):
            MeanReversionStrategy(lookback=20, threshold=-1.0)

    def test_generate_signals(self, sample_single_symbol_df):
        """Test signal generation."""
        strategy = MeanReversionStrategy(lookback=20, threshold=2.0)
        signals = strategy.generate_signals(sample_single_symbol_df)

        assert len(signals) == len(sample_single_symbol_df)
        assert set(signals.unique()) <= {-1, 0, 1}

    def test_determinism(self, sample_single_symbol_df):
        """Test deterministic output."""
        strategy = MeanReversionStrategy(lookback=20, threshold=2.0)

        outputs = []
        for _ in range(5):
            signal = strategy.generate_signals(sample_single_symbol_df)
            outputs.append(signal)

        for i in range(1, len(outputs)):
            assert outputs[i].equals(outputs[0])

    def test_output_schema_validation(self, sample_single_symbol_df):
        """Test output schema validation."""
        strategy = MeanReversionStrategy(lookback=20, threshold=2.0)
        signals = strategy.generate_signals(sample_single_symbol_df)

        validate_signal_output("mean_reversion", signals)


class TestBreakout:
    """Tests for BreakoutStrategy."""

    def test_instantiation(self):
        """Test strategy instantiation."""
        strategy = BreakoutStrategy(lookback=20)
        assert strategy.strategy_id == "breakout"

    def test_generate_signals(self, sample_single_symbol_df):
        """Test signal generation."""
        strategy = BreakoutStrategy(lookback=20)
        signals = strategy.generate_signals(sample_single_symbol_df)

        assert len(signals) == len(sample_single_symbol_df)
        assert set(signals.unique()) <= {-1, 0, 1}

    def test_determinism(self, sample_single_symbol_df):
        """Test deterministic output."""
        strategy = BreakoutStrategy(lookback=20)

        outputs = []
        for _ in range(5):
            signal = strategy.generate_signals(sample_single_symbol_df)
            outputs.append(signal)

        for i in range(1, len(outputs)):
            assert outputs[i].equals(outputs[0])

    def test_output_schema_validation(self, sample_single_symbol_df):
        """Test output schema validation."""
        strategy = BreakoutStrategy(lookback=20)
        signals = strategy.generate_signals(sample_single_symbol_df)

        validate_signal_output("breakout", signals)


class TestPairsTrading:
    """Tests for PairsTradingStrategy."""

    def test_instantiation(self):
        """Test strategy instantiation."""
        strategy = PairsTradingStrategy(lookback=63, threshold=2.0)
        assert strategy.strategy_id == "pairs_trading"

    def test_generate_signals(self, sample_daily_df):
        """Test signal generation."""
        strategy = PairsTradingStrategy(lookback=63, threshold=2.0)
        signals = strategy.generate_signals(sample_daily_df)

        assert len(signals) == len(sample_daily_df)

    def test_determinism(self, sample_daily_df):
        """Test deterministic output."""
        strategy = PairsTradingStrategy(lookback=63, threshold=2.0)

        outputs = []
        for _ in range(5):
            signal = strategy.generate_signals(sample_daily_df)
            outputs.append(signal)

        for i in range(1, len(outputs)):
            assert outputs[i].equals(outputs[0])

    def test_output_schema_validation(self, sample_daily_df):
        """Test output schema validation."""
        strategy = PairsTradingStrategy(lookback=63, threshold=2.0)
        signals = strategy.generate_signals(sample_daily_df)

        validate_signal_output("pairs_trading", signals)


class TestResidualMomentum:
    """Tests for ResidualMomentumStrategy."""

    def test_instantiation(self):
        """Test strategy instantiation."""
        strategy = ResidualMomentumStrategy(lookback_days=20)
        assert strategy.strategy_id == "residual_momentum"

    def test_generate_signals(self, sample_daily_df):
        """Test signal generation."""
        # Add market_return column
        sample_daily_df["market_return"] = np.random.randn(len(sample_daily_df)) * 0.01

        strategy = ResidualMomentumStrategy(lookback_days=20)
        signals = strategy.generate_signals(sample_daily_df)

        assert len(signals) == len(sample_daily_df)

    def test_determinism(self, sample_daily_df):
        """Test deterministic output."""
        sample_daily_df["market_return"] = np.random.randn(len(sample_daily_df)) * 0.01

        strategy = ResidualMomentumStrategy(lookback_days=20)

        outputs = []
        for _ in range(5):
            signal = strategy.generate_signals(sample_daily_df)
            outputs.append(signal)

        for i in range(1, len(outputs)):
            assert outputs[i].equals(outputs[0])

    def test_output_schema_validation(self, sample_daily_df):
        """Test output schema validation."""
        sample_daily_df["market_return"] = np.random.randn(len(sample_daily_df)) * 0.01

        strategy = ResidualMomentumStrategy(lookback_days=20)
        signals = strategy.generate_signals(sample_daily_df)

        validate_signal_output("residual_momentum", signals)


class TestRegistryIntegration:
    """Tests for registry and validation integration."""

    def test_load_registry(self):
        """Test loading registry."""
        registry = load_strategies_registry()
        assert len(registry) == 6
        assert all("strategy_id" in entry for entry in registry)

    def test_list_available_strategies(self):
        """Test listing available strategies."""
        strategies = list_available_strategies()
        expected = [
            "time_series_momentum",
            "cross_section_momentum",
            "mean_reversion",
            "breakout",
            "pairs_trading",
            "residual_momentum",
        ]
        assert sorted(strategies) == sorted(expected)

    def test_get_strategies_by_archetype(self):
        """Test filtering strategies by archetype."""
        momentum = get_strategy_by_archetype("time_series_momentum")
        assert "time_series_momentum" in momentum

        cross_section = get_strategy_by_archetype("cross_section_momentum")
        assert "cross_section_momentum" in cross_section

    def test_validate_config(self, sample_single_symbol_df):
        """Test configuration validation."""
        config = {
            "dataset": "features_daily",
            "signal_type": "ternary_quantile",
            "position_constructor": {"name": "identity_weights", "params": {}},
            "parameters": {"lookback": 20, "threshold": 2.0},
        }

        validate_strategy_config("mean_reversion", config)

    def test_validate_config_invalid(self):
        """Test that invalid config raises error."""
        config = {
            "dataset": "features_daily",
            # Missing 'parameters'
        }

        with pytest.raises(Exception):
            validate_strategy_config("mean_reversion", config)

    def test_build_strategy_from_registry(self, sample_single_symbol_df):
        """Test building strategy from registry."""
        config = {
            "dataset": "features_daily",
            "signal_type": "ternary_quantile",
            "parameters": {"lookback": 20, "threshold": 2.0},
        }

        strategy = build_strategy("mean_reversion", config)
        assert strategy is not None
        assert strategy.strategy_id == "mean_reversion"

    def test_verify_determinism(self, sample_single_symbol_df):
        """Test determinism verification."""
        config = {
            "dataset": "features_daily",
            "signal_type": "ternary_quantile",
            "parameters": {"lookback": 20, "threshold": 2.0},
        }

        result = verify_determinism("mean_reversion", config, sample_single_symbol_df, repetitions=3)
        assert result is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, sample_single_symbol_df):
        """Test handling empty DataFrame."""
        empty_df = sample_single_symbol_df.iloc[:0].copy()
        strategy = TimeSeriesMomentumStrategy(lookback_short=10, lookback_long=30)

        signals = strategy.generate_signals(empty_df)
        assert len(signals) == 0

    def test_insufficient_lookback(self, sample_single_symbol_df):
        """Test with insufficient lookback."""
        short_df = sample_single_symbol_df.iloc[:5].copy()
        strategy = TimeSeriesMomentumStrategy(lookback_short=10, lookback_long=30)

        signals = strategy.generate_signals(short_df)
        # Should generate NaN for insufficient periods
        assert len(signals) == len(short_df)

    def test_single_row(self, sample_single_symbol_df):
        """Test with single row."""
        single_df = sample_single_symbol_df.iloc[:1].copy()
        strategy = TimeSeriesMomentumStrategy(lookback_short=10, lookback_long=30)

        signals = strategy.generate_signals(single_df)
        assert len(signals) == 1

    def test_nan_handling(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
            "symbol": "TEST",
            "close": [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
        })

        strategy = MeanReversionStrategy(lookback=5, threshold=2.0)
        signals = strategy.generate_signals(df)

        assert len(signals) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
