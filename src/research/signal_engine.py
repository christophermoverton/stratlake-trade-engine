from __future__ import annotations

import warnings

import pandas as pd

from src.research.integrity import validate_research_integrity
from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.strategy_base import BaseStrategy


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
    _warn_on_degenerate_behavior(strategy.__class__.__name__, diagnostics)
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
