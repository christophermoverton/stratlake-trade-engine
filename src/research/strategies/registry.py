from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.research.position_constructors import normalize_position_constructor_config
from src.research.strategy_archetype import ArchetypeStrategy
from src.research.strategy_base import BaseStrategy

from src.research.strategies.baselines import BuyAndHoldStrategy, SMACrossoverStrategy, SeededRandomStrategy
from src.research.strategies.builtins import MeanReversionStrategy, MomentumStrategy
from src.research.strategies.archetypes import (
    TimeSeriesMomentumStrategy,
    CrossSectionMomentumStrategy,
    MeanReversionStrategy as MeanReversionArchetype,
    BreakoutStrategy,
    PairsTradingStrategy,
    ResidualMomentumStrategy,
    VolatilityRegimeMomentumStrategy,
    WeightedCrossSectionEnsembleStrategy,
)

StrategyBuilder = Callable[[dict[str, Any]], BaseStrategy]


def _build_momentum(parameters: dict[str, Any]) -> BaseStrategy:
    return MomentumStrategy(
        lookback_short=int(parameters["lookback_short"]),
        lookback_long=int(parameters["lookback_long"]),
    )


def _build_mean_reversion(parameters: dict[str, Any]) -> BaseStrategy:
    return MeanReversionStrategy(
        lookback=int(parameters["lookback"]),
        threshold=float(parameters["threshold"]),
    )


def _build_buy_and_hold(parameters: dict[str, Any]) -> BaseStrategy:
    return BuyAndHoldStrategy()


def _build_sma_crossover(parameters: dict[str, Any]) -> BaseStrategy:
    return SMACrossoverStrategy(
        fast_window=int(parameters["fast_window"]),
        slow_window=int(parameters["slow_window"]),
    )


def _build_seeded_random(parameters: dict[str, Any]) -> BaseStrategy:
    return SeededRandomStrategy(seed=int(parameters["seed"]))


def _build_time_series_momentum(parameters: dict[str, Any]) -> BaseStrategy:
    return TimeSeriesMomentumStrategy(
        lookback_short=int(parameters["lookback_short"]),
        lookback_long=int(parameters["lookback_long"]),
    )


def _build_cross_section_momentum(parameters: dict[str, Any]) -> BaseStrategy:
    return CrossSectionMomentumStrategy(
        lookback_days=int(parameters.get("lookback_days", 1)),
    )


def _build_mean_reversion_archetype(parameters: dict[str, Any]) -> BaseStrategy:
    return MeanReversionArchetype(
        lookback=int(parameters["lookback"]),
        threshold=float(parameters["threshold"]),
    )


def _build_breakout(parameters: dict[str, Any]) -> BaseStrategy:
    return BreakoutStrategy(
        lookback=int(parameters.get("lookback", 20)),
    )


def _build_pairs_trading(parameters: dict[str, Any]) -> BaseStrategy:
    return PairsTradingStrategy(
        lookback=int(parameters.get("lookback", 63)),
        threshold=float(parameters.get("threshold", 2.0)),
    )


def _build_residual_momentum(parameters: dict[str, Any]) -> BaseStrategy:
    return ResidualMomentumStrategy(
        lookback_days=int(parameters.get("lookback_days", 20)),
    )


def _build_volatility_regime_momentum(parameters: dict[str, Any]) -> BaseStrategy:
    return VolatilityRegimeMomentumStrategy(
        lookback_short=int(parameters.get("lookback_short", 10)),
        lookback_long=int(parameters.get("lookback_long", 30)),
        volatility_lookback=int(parameters.get("volatility_lookback", 20)),
        min_volatility=float(parameters.get("min_volatility", 0.0)),
        max_volatility=float(parameters.get("max_volatility", 0.03)),
    )


def _build_weighted_cross_section_ensemble(parameters: dict[str, Any]) -> BaseStrategy:
    return WeightedCrossSectionEnsembleStrategy(
        momentum_lookback_days=int(parameters.get("momentum_lookback_days", 5)),
        residual_lookback_days=int(parameters.get("residual_lookback_days", 20)),
        momentum_weight=float(parameters.get("momentum_weight", 0.5)),
        residual_weight=float(parameters.get("residual_weight", 0.5)),
    )


STRATEGY_BUILDERS: dict[str, StrategyBuilder] = {
    "momentum_v1": _build_momentum,
    "mean_reversion_v1": _build_mean_reversion,
    "mean_reversion_v1_safe_2026_q1": _build_mean_reversion,
    "buy_and_hold_v1": _build_buy_and_hold,
    "sma_crossover_v1": _build_sma_crossover,
    "seeded_random_v1": _build_seeded_random,
    # New archetype strategies
    "time_series_momentum": _build_time_series_momentum,
    "cross_section_momentum": _build_cross_section_momentum,
    "mean_reversion": _build_mean_reversion_archetype,
    "breakout": _build_breakout,
    "pairs_trading": _build_pairs_trading,
    "residual_momentum": _build_residual_momentum,
    "volatility_regime_momentum": _build_volatility_regime_momentum,
    "weighted_cross_section_ensemble": _build_weighted_cross_section_ensemble,
}


def build_strategy(strategy_name: str, config: dict[str, Any]) -> BaseStrategy:
    """Instantiate a configured strategy and attach the configured dataset name."""

    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"Strategy '{strategy_name}' parameters must be a dictionary.")

    try:
        builder = STRATEGY_BUILDERS[strategy_name]
    except KeyError as exc:
        raise ValueError(f"No strategy implementation is registered for '{strategy_name}'.") from exc

    strategy = builder(parameters)
    dataset = config.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"Strategy '{strategy_name}' must define a non-empty dataset name.")

    strategy.name = strategy_name
    strategy.dataset = dataset
    if isinstance(strategy, ArchetypeStrategy):
        strategy.required_input_columns = tuple(strategy.strategy_definition.input_schema.required_columns)
        strategy.requires_return_column = False
        strategy.signal_type = strategy.strategy_definition.output_signal_schema.signal_type
        strategy.signal_params = {}
        strategy.position_constructor_name = "identity_weights"
        strategy.position_constructor_params = {}
    signal_type = config.get("signal_type")
    if signal_type is not None:
        if not isinstance(signal_type, str) or not signal_type.strip():
            raise ValueError(f"Strategy '{strategy_name}' signal_type must be a non-empty string when provided.")
        strategy.signal_type = signal_type.strip()
    signal_params = config.get("signal_params", {})
    if signal_params is None:
        signal_params = {}
    if not isinstance(signal_params, dict):
        raise ValueError(f"Strategy '{strategy_name}' signal_params must be a dictionary when provided.")
    strategy.signal_params = dict(signal_params)
    position_constructor = normalize_position_constructor_config(config.get("position_constructor"))
    if position_constructor is not None:
        strategy.position_constructor_name = str(position_constructor["name"])
        strategy.position_constructor_params = dict(position_constructor["params"])
    return strategy
