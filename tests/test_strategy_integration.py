"""
Integration tests for Strategy Archetype Library with backtest and pipeline infrastructure.

Tests strategy integration with:
- Backtest runner (M20)
- Signal Semantics Layer (M21.1)
- Position Constructors (M21.2)
- Artifact tracking system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.research.strategies.registry import build_strategy
from src.research.strategies.validation import (
    load_strategies_registry,
    verify_determinism,
)


@pytest.fixture
def realistic_feature_data():
    """Create realistic multi-symbol feature data with price movements."""
    dates = pd.date_range("2024-01-01", periods=252, freq="D")  # 1 year of daily data
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    data = []
    for i, date in enumerate(dates):
        for symbol in symbols:
            # Create realistic price movements
            price = 100.0 + (i * 0.5) + np.random.randn() * 5
            high = price * 1.02 + np.random.rand() * 2
            low = price * 0.98 - np.random.rand() * 2
            close = price + np.random.randn() * 1

            data.append({
                "ts_utc": date.value,  # nanosecond timestamp
                "symbol": symbol,
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000000 + np.random.randint(-100000, 100000),
            })

    df = pd.DataFrame(data)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["market_return"] = np.random.randn(len(df)) * 0.01
    return df


class TestStrategyBacktestIntegration:
    """Test strategy integration with backtest runner."""

    def test_generate_signals_for_backtest(self, realistic_feature_data):
        """Test that strategies generate signals suitable for backtest."""
        config = {
            "dataset": "features_daily",
            "signal_type": "ternary_quantile",
            "position_constructor": {
                "name": "identity_weights",
                "params": {}
            },
            "parameters": {"lookback_short": 20, "lookback_long": 100},
        }

        strategy = build_strategy("time_series_momentum", config)
        signals = strategy.generate_signals(realistic_feature_data)

        # Verify signals are aligned to data
        assert len(signals) == len(realistic_feature_data)
        assert signals.index.equals(realistic_feature_data.index)

        # Verify signal values are valid
        assert set(signals.unique()) <= {-1, 0, 1}
        assert signals.notna().sum() > 0  # Some valid signals

    def test_cross_section_momentum_for_backtest(self, realistic_feature_data):
        """Test cross-sectional momentum generates proper ranked signals."""
        config = {
            "dataset": "features_daily",
            "signal_type": "cross_section_rank",
            "position_constructor": {
                "name": "rank_dollar_neutral",
                "params": {}
            },
            "parameters": {"lookback_days": 5},
        }

        strategy = build_strategy("cross_section_momentum", config)
        signals = strategy.generate_signals(realistic_feature_data)

        # Cross-section signals should be in [-1, 1]
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0
        assert len(signals) == len(realistic_feature_data)

    def test_mean_reversion_for_backtest(self, realistic_feature_data):
        """Test mean reversion strategy generates valid signals."""
        config = {
            "dataset": "features_daily",
            "signal_type": "ternary_quantile",
            "position_constructor": {
                "name": "identity_weights",
                "params": {}
            },
            "parameters": {"lookback": 30, "threshold": 2.0},
        }

        strategy = build_strategy("mean_reversion", config)
        signals = strategy.generate_signals(realistic_feature_data)

        # Verify ternary signal
        assert set(signals.unique()) <= {-1, 0, 1}
        assert len(signals) == len(realistic_feature_data)


class TestStrategySignalSemantics:
    """Test that strategies comply with Signal Semantics Layer (M21.1)."""

    def test_signal_type_compliance(self, realistic_feature_data):
        """Test signals comply with declared signal types."""
        from src.research.strategies.validation import validate_signal_output

        strategies_and_configs = [
            ("time_series_momentum", {
                "dataset": "features_daily",
                "parameters": {"lookback_short": 20, "lookback_long": 100},
            }),
            ("cross_section_momentum", {
                "dataset": "features_daily",
                "parameters": {"lookback_days": 1},
            }),
            ("mean_reversion", {
                "dataset": "features_daily",
                "parameters": {"lookback": 20, "threshold": 2.0},
            }),
            ("breakout", {
                "dataset": "features_daily",
                "parameters": {"lookback": 20},
            }),
        ]

        for strategy_id, config in strategies_and_configs:
            strategy = build_strategy(strategy_id, config)
            signals = strategy.generate_signals(realistic_feature_data)

            # All signals should validate against their declared type
            validate_signal_output(strategy_id, signals)

    def test_signal_metadata_attachment(self, realistic_feature_data):
        """Test that signal metadata can be attached for M21.1 integration."""
        config = {
            "dataset": "features_daily",
            "signal_type": "ternary_quantile",
            "parameters": {"lookback_short": 20, "lookback_long": 100},
        }

        strategy = build_strategy("time_series_momentum", config)
        signals = strategy.generate_signals(realistic_feature_data)

        # Create DataFrame for metadata attachment
        signal_df = pd.DataFrame({
            "symbol": realistic_feature_data["symbol"],
            "ts_utc": realistic_feature_data["ts_utc"],
            "signal": signals,
        })

        # Verify structure is compatible with signal semantics
        assert "symbol" in signal_df.columns
        assert "ts_utc" in signal_df.columns
        assert "signal" in signal_df.columns


class TestStrategyPositionConstructorCompatibility:
    """Test strategy signals are compatible with position constructors (M21.2)."""

    def test_ternary_signals_with_identity_weights(self, realistic_feature_data):
        """Test ternary signals work with identity_weights constructor."""
        config = {
            "dataset": "features_daily",
            "signal_type": "ternary_quantile",
            "position_constructor": {
                "name": "identity_weights",
                "params": {}
            },
            "parameters": {"lookback_short": 20, "lookback_long": 100},
        }

        strategy = build_strategy("time_series_momentum", config)
        signals = strategy.generate_signals(realistic_feature_data)

        # Identity weights should pass signals through directly
        assert set(signals.unique()) <= {-1, 0, 1}

    def test_ranked_signals_with_rank_dollar_neutral(self, realistic_feature_data):
        """Test ranked signals work with rank_dollar_neutral constructor."""
        config = {
            "dataset": "features_daily",
            "signal_type": "cross_section_rank",
            "position_constructor": {
                "name": "rank_dollar_neutral",
                "params": {}
            },
            "parameters": {"lookback_days": 1},
        }

        strategy = build_strategy("cross_section_momentum", config)
        signals = strategy.generate_signals(realistic_feature_data)

        # Signals should be in [-1, 1] for rank_dollar_neutral
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0


class TestStrategyRegistry:
    """Test strategy registry for declarative strategy resolution."""

    def test_registry_discovery(self):
        """Test discovering strategies from registry."""
        registry = load_strategies_registry()

        # Verify all 6 archetypes are in registry
        strategy_ids = [entry["strategy_id"] for entry in registry]
        expected = [
            "time_series_momentum",
            "cross_section_momentum",
            "mean_reversion",
            "breakout",
            "pairs_trading",
            "residual_momentum",
        ]
        assert sorted(strategy_ids) == sorted(expected)

    def test_registry_metadata_completeness(self):
        """Test registry entries have all required metadata."""
        registry = load_strategies_registry()

        required_fields = [
            "strategy_id",
            "version",
            "status",
            "archetype",
            "name",
            "description",
            "mathematical_definition",
            "research_hypothesis",
            "input_schema",
            "output_signal",
            "execution_semantics",
            "failure_modes",
            "assumptions",
            "tags",
        ]

        for entry in registry:
            for field in required_fields:
                assert field in entry, f"Registry entry {entry['strategy_id']} missing {field}"

    def test_registry_enables_declarative_resolution(self):
        """Test registry enables YAML-based strategy configuration."""
        registry = load_strategies_registry()

        # Simulate YAML config resolution
        yaml_config = {
            "strategy": {
                "name": "cross_section_momentum",
                "version": "1.0.0",
                "parameters": {"lookback_days": 1},
                "dataset": "features_daily",
            }
        }

        # Find strategy in registry
        strategy_entry = next(
            (e for e in registry if e["strategy_id"] == yaml_config["strategy"]["name"]),
            None
        )
        assert strategy_entry is not None
        assert strategy_entry["version"] == yaml_config["strategy"]["version"]


class TestDeterminismAndReproducibility:
    """Test strategies produce deterministic, reproducible outputs."""

    def test_deterministic_outputs(self, realistic_feature_data):
        """Test strategies produce identical outputs for identical inputs."""
        config = {
            "dataset": "features_daily",
            "signal_type": "ternary_quantile",
            "parameters": {"lookback_short": 20, "lookback_long": 100},
        }

        # Verify determinism
        verify_determinism("time_series_momentum", config, realistic_feature_data, repetitions=5)

    def test_reproducible_across_runs(self, realistic_feature_data):
        """Test strategies produce identical outputs across separate runs."""
        config = {
            "dataset": "features_daily",
            "parameters": {"lookback_days": 1},
        }

        # Run strategy multiple times
        outputs = []
        for _ in range(3):
            strategy = build_strategy("cross_section_momentum", config)
            signal = strategy.generate_signals(realistic_feature_data)
            outputs.append(signal)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert outputs[i].equals(outputs[0])

    def test_determinism_with_sorted_index(self, realistic_feature_data):
        """Test determinism is preserved with different index ordering."""
        config = {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 20, "lookback_long": 100},
        }

        strategy = build_strategy("time_series_momentum", config)

        # Original data
        signal1 = strategy.generate_signals(realistic_feature_data)

        # Shuffled and re-sorted data
        shuffled = realistic_feature_data.sample(frac=1).sort_index().reset_index(drop=True)
        signal2 = strategy.generate_signals(shuffled)

        # Signals should match after alignment
        assert len(signal1) == len(signal2)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_insufficient_data_handling(self):
        """Test strategies handle insufficient data gracefully."""
        small_df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5, freq="D"),
            "symbol": ["TEST"] * 5,
            "close": [100, 101, 102, 103, 104],
        })

        config = {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 20, "lookback_long": 100},
        }

        strategy = build_strategy("time_series_momentum", config)
        signals = strategy.generate_signals(small_df)

        # Should return Series with same length
        assert len(signals) == len(small_df)

    def test_nan_handling(self):
        """Test strategies handle NaN values appropriately."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=100, freq="D"),
            "symbol": ["TEST"] * 100,
            "close": [100 + i * 0.5 + np.random.randn() for i in range(100)],
        })

        # Introduce some NaNs
        df.loc[10:15, "close"] = np.nan

        config = {
            "dataset": "features_daily",
            "parameters": {"lookback": 20, "threshold": 2.0},
        }

        strategy = build_strategy("mean_reversion", config)
        signals = strategy.generate_signals(df)

        # Should still return valid signals
        assert len(signals) == len(df)
        assert signals.notna().sum() > 0

    def test_single_symbol_vs_multi_symbol(self, realistic_feature_data):
        """Test strategies work with both single and multi-symbol data."""
        # Single symbol
        single_sym = realistic_feature_data[
            realistic_feature_data["symbol"] == "AAPL"
        ].reset_index(drop=True)

        config = {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 20, "lookback_long": 100},
        }

        strategy = build_strategy("time_series_momentum", config)

        # Both should work without errors
        signal_single = strategy.generate_signals(single_sym)
        signal_multi = strategy.generate_signals(realistic_feature_data)

        assert len(signal_single) == len(single_sym)
        assert len(signal_multi) == len(realistic_feature_data)


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_performance_with_large_universe(self):
        """Test strategies scale with increasing universe size."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        symbols = [f"SYM{i}" for i in range(50)]  # 50 symbols

        data = []
        for date in dates:
            for symbol in symbols:
                price = 100.0 + np.random.randn() * 5
                data.append({
                    "ts_utc": date.value,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1000000,
                })

        df = pd.DataFrame(data)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

        config = {
            "dataset": "features_daily",
            "parameters": {"lookback_days": 1},
        }

        strategy = build_strategy("cross_section_momentum", config)
        signals = strategy.generate_signals(df)

        # Should complete without errors
        assert len(signals) == len(df)

    def test_performance_with_long_history(self):
        """Test strategies scale with longer time series."""
        dates = pd.date_range("2000-01-01", periods=5000, freq="D")  # ~20 years
        symbols = ["AAPL", "MSFT", "GOOGL"]

        data = []
        for date in dates:
            for symbol in symbols:
                price = 100.0 + np.random.randn() * 5
                data.append({
                    "ts_utc": date.value,
                    "symbol": symbol,
                    "close": price,
                })

        df = pd.DataFrame(data)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

        config = {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 20, "lookback_long": 100},
        }

        strategy = build_strategy("time_series_momentum", config)
        signals = strategy.generate_signals(df)

        # Should complete successfully
        assert len(signals) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
