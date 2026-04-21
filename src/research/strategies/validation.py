"""
Strategy validation utilities and schema enforcement.

Provides utilities for:
- Loading and parsing strategy registry entries
- Validating strategy configurations against schemas
- Determinism verification
- Signal semantics validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


class StrategyRegistryError(ValueError):
    """Raised when strategy registry entry is invalid."""


class StrategyConfigError(ValueError):
    """Raised when strategy configuration is invalid."""


def load_strategies_registry(registry_path: Path | None = None) -> list[dict[str, Any]]:
    """
    Load strategy registry from JSONL file.

    Args:
        registry_path: Path to strategies.jsonl. If None, uses default location.

    Returns:
        List of strategy registry entries (dicts).

    Raises:
        StrategyRegistryError: If registry file not found or invalid.
    """
    if registry_path is None:
        registry_path = (
            Path(__file__).resolve().parents[3]
            / "artifacts"
            / "registry"
            / "strategies.jsonl"
        )

    if not registry_path.exists():
        raise StrategyRegistryError(f"Registry not found at {registry_path}")

    entries = []
    with open(registry_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise StrategyRegistryError(
                    f"Invalid JSON at registry line {line_num}: {e}"
                ) from e

    return entries


def get_strategy_registry_entry(
    strategy_id: str, version: str = "1.0.0", registry: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """
    Retrieve a strategy registry entry by ID and version.

    Args:
        strategy_id: Strategy identifier.
        version: Strategy version.
        registry: Pre-loaded registry. If None, loads from disk.

    Returns:
        Strategy registry entry.

    Raises:
        StrategyRegistryError: If strategy not found.
    """
    if registry is None:
        registry = load_strategies_registry()

    for entry in registry:
        if entry.get("strategy_id") == strategy_id and entry.get("version") == version:
            return entry

    raise StrategyRegistryError(
        f"Strategy '{strategy_id}' version '{version}' not found in registry"
    )


def validate_strategy_config(
    strategy_id: str, config: Mapping[str, Any], registry: list[dict[str, Any]] | None = None
) -> None:
    """
    Validate strategy configuration against registry schema.

    Args:
        strategy_id: Strategy identifier.
        config: Configuration dict with 'parameters', 'dataset', 'signal_type', etc.
        registry: Pre-loaded registry. If None, loads from disk.

    Raises:
        StrategyConfigError: If config invalid against schema.
    """
    if registry is None:
        registry = load_strategies_registry()

    entry = get_strategy_registry_entry(strategy_id, registry=registry)

    # Check required config fields
    if "dataset" not in config:
        raise StrategyConfigError(f"Strategy '{strategy_id}' config missing 'dataset'")

    if not isinstance(config.get("parameters"), dict):
        raise StrategyConfigError(
            f"Strategy '{strategy_id}' config 'parameters' must be a dictionary"
        )

    # Validate signal_type if provided
    signal_type = config.get("signal_type")
    if signal_type:
        expected_signal_type = entry.get("output_signal", {}).get("signal_type")
        if signal_type != expected_signal_type:
            raise StrategyConfigError(
                f"Strategy '{strategy_id}' config signal_type '{signal_type}' "
                f"does not match expected '{expected_signal_type}'"
            )


def validate_signal_output(
    strategy_id: str,
    signal: pd.Series,
    registry: list[dict[str, Any]] | None = None,
) -> None:
    """
    Validate output signal against expected schema.

    Args:
        strategy_id: Strategy identifier.
        signal: Output signal series.
        registry: Pre-loaded registry. If None, loads from disk.

    Raises:
        StrategyConfigError: If signal violates schema.
    """
    if registry is None:
        registry = load_strategies_registry()

    entry = get_strategy_registry_entry(strategy_id, registry=registry)
    output_signal = entry.get("output_signal", {})

    allowed_values = output_signal.get("allowed_values")
    if allowed_values is not None:
        invalid = signal[~signal.isin(allowed_values)]
        if not invalid.empty:
            raise StrategyConfigError(
                f"Signal contains invalid values for strategy '{strategy_id}': "
                f"{set(invalid.unique())} not in {allowed_values}"
            )

    value_range = output_signal.get("value_range")
    if value_range is not None:
        min_val, max_val = value_range
        out_of_range = signal[(signal < min_val) | (signal > max_val)]
        if not out_of_range.empty:
            raise StrategyConfigError(
                f"Signal values outside expected range [{min_val}, {max_val}] for "
                f"strategy '{strategy_id}': min={signal.min()}, max={signal.max()}"
            )


def verify_determinism(
    strategy_id: str,
    config: dict[str, Any],
    df: pd.DataFrame,
    repetitions: int = 3,
    registry: list[dict[str, Any]] | None = None,
) -> bool:
    """
    Verify strategy produces deterministic outputs.

    Args:
        strategy_id: Strategy identifier.
        config: Strategy configuration.
        df: Input feature data.
        repetitions: Number of times to run strategy.
        registry: Pre-loaded registry. If None, loads from disk.

    Returns:
        True if all runs produce identical output.

    Raises:
        StrategyConfigError: If strategy is non-deterministic.
    """
    from src.research.strategies.registry import build_strategy

    if registry is None:
        registry = load_strategies_registry()

    entry = get_strategy_registry_entry(strategy_id, registry=registry)

    # Check if strategy is supposed to be deterministic
    if not entry.get("execution_semantics", {}).get("deterministic", True):
        raise StrategyConfigError(f"Strategy '{strategy_id}' is not marked as deterministic")

    # Run strategy multiple times
    outputs = []
    for i in range(repetitions):
        strategy = build_strategy(strategy_id, config)
        signal = strategy.generate_signals(df)
        outputs.append(signal)

    # Compare all outputs
    for i in range(1, len(outputs)):
        if not outputs[i].equals(outputs[0]):
            raise StrategyConfigError(
                f"Strategy '{strategy_id}' produced non-deterministic outputs "
                f"(run 0 != run {i})"
            )

    return True


def list_available_strategies(registry: list[dict[str, Any]] | None = None) -> list[str]:
    """
    List all available strategy IDs.

    Args:
        registry: Pre-loaded registry. If None, loads from disk.

    Returns:
        List of strategy IDs.
    """
    if registry is None:
        registry = load_strategies_registry()

    return sorted(set(entry.get("strategy_id") for entry in registry))


def get_strategy_by_archetype(
    archetype: str, registry: list[dict[str, Any]] | None = None
) -> list[str]:
    """
    Get all strategies of a given archetype.

    Args:
        archetype: Archetype name (e.g., "momentum").
        registry: Pre-loaded registry. If None, loads from disk.

    Returns:
        List of strategy IDs matching the archetype.
    """
    if registry is None:
        registry = load_strategies_registry()

    return sorted(
        set(
            entry.get("strategy_id")
            for entry in registry
            if entry.get("archetype") == archetype
        )
    )
