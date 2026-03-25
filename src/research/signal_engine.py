from __future__ import annotations

import warnings

import pandas as pd

from src.research.input_validation import STRATEGY_INPUT_MIN_ROWS, validate_strategy_input
from src.research.integrity import validate_research_integrity
from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.strategy_base import BaseStrategy
from src.research.strategies._helpers import resolve_return_column


def generate_signals(df: pd.DataFrame, strategy: BaseStrategy) -> pd.DataFrame:
    """
    Apply a research strategy to a feature dataset and attach standardized signals.

    Args:
        df: Feature dataset consumed by the strategy.
        strategy: Strategy implementation that generates trading signals.

    Returns:
        A copy of ``df`` with an added ``signal`` column aligned to ``df.index``.

    Raises:
        TypeError: If the strategy output is not a pandas Series.
        ValueError: If the strategy output index does not match ``df.index`` or
            if the signal values are outside the standard ``{-1, 0, 1}`` set.
    """

    validate_strategy_input(
        df,
        required_columns=_required_columns_for_strategy(df, strategy),
        timeframe=_resolve_timeframe(df, strategy),
    )

    signals = strategy.generate_signals(df)

    if not isinstance(signals, pd.Series):
        raise TypeError(
            f"Strategy {strategy.__class__.__name__}.generate_signals() must return a pandas Series."
        )

    validate_research_integrity(df, signals)

    result = df.copy()
    result["signal"] = signals.rename("signal")
    diagnostics = compute_signal_diagnostics(result["signal"], result)
    result.attrs["signal_diagnostics"] = diagnostics
    result.attrs["input_validation"] = dict(df.attrs.get("input_validation", {}))
    _warn_on_degenerate_behavior(strategy.__class__.__name__, diagnostics)
    _warn_on_low_data(result)
    return result


def _warn_on_degenerate_behavior(strategy_name: str, diagnostics: dict[str, object]) -> None:
    flags = diagnostics.get("flags")
    if not isinstance(flags, dict):
        return

    active_flags = [
        flag_name.replace("_", " ")
        for flag_name, enabled in flags.items()
        if enabled
    ]
    if not active_flags:
        return

    warnings.warn(
        f"Strategy {strategy_name} produced degenerate signal behavior: {', '.join(active_flags)}.",
        RuntimeWarning,
        stacklevel=2,
    )


def _required_columns_for_strategy(df: pd.DataFrame, strategy: BaseStrategy) -> list[str]:
    required = list(strategy.required_input_columns)
    if strategy.requires_return_column:
        required.append(resolve_return_column(df))
    return list(dict.fromkeys(required))


def _resolve_timeframe(df: pd.DataFrame, strategy: BaseStrategy) -> str:
    if "timeframe" not in df.columns:
        fallback = df.attrs.get("timeframe")
        return "" if fallback is None else str(fallback)

    timeframe_values = df["timeframe"].dropna().astype("string").str.strip()
    if timeframe_values.empty:
        raise ValueError(f"Strategy {strategy.name} input must include a non-empty timeframe value.")

    return str(timeframe_values.iloc[0])


def _warn_on_low_data(df: pd.DataFrame) -> None:
    validation = df.attrs.get("input_validation")
    if not isinstance(validation, dict):
        return
    if not bool(validation.get("low_data")):
        return

    warnings.warn(
        "Strategy input is below the recommended minimum sample size of "
        f"{STRATEGY_INPUT_MIN_ROWS} rows; interpret results cautiously.",
        RuntimeWarning,
        stacklevel=2,
    )
