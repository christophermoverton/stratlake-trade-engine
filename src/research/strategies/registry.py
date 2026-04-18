from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.research.position_constructors import normalize_position_constructor_config
from src.research.strategy_base import BaseStrategy

from src.research.strategies.baselines import BuyAndHoldStrategy, SMACrossoverStrategy, SeededRandomStrategy
from src.research.strategies.builtins import MeanReversionStrategy, MomentumStrategy

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


STRATEGY_BUILDERS: dict[str, StrategyBuilder] = {
    "momentum_v1": _build_momentum,
    "mean_reversion_v1": _build_mean_reversion,
    "mean_reversion_v1_safe_2026_q1": _build_mean_reversion,
    "buy_and_hold_v1": _build_buy_and_hold,
    "sma_crossover_v1": _build_sma_crossover,
    "seeded_random_v1": _build_seeded_random,
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

    strategy.dataset = dataset
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
