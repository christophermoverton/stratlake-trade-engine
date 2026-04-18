from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any

import pandas as pd

from src.research.registry import canonicalize_value
from src.research.signal_semantics import Signal

_REQUIRED_POSITION_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc", "position")


class PositionConstructorError(ValueError):
    """Raised when position construction configuration or output is invalid."""


@dataclass(frozen=True)
class PositionConstructor(ABC):
    """Deterministic transformer from one typed signal into executable positions."""

    constructor_id: str
    params: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    @abstractmethod
    def construct(self, signal: Signal) -> pd.DataFrame:
        """
        Build one deterministic position frame aligned to the signal rows.

        Returns a DataFrame with at least ``symbol``, ``ts_utc``, and ``position``.
        Implementations are expected to preserve the input row order and index.
        """

    def metadata_payload(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "constructor_id": self.constructor_id,
                "constructor_params": dict(self.params),
                "constructor_version": self.version,
            }
        )


def base_position_frame(signal: Signal) -> pd.DataFrame:
    frame = signal.data.loc[:, [column for column in ("symbol", "ts_utc") if column in signal.data.columns]].copy(deep=True)
    frame.attrs = {}
    return frame


def signal_values(signal: Signal) -> pd.Series:
    try:
        values = pd.to_numeric(signal.data[signal.value_column], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise PositionConstructorError(
            f"Signal column {signal.value_column!r} must contain numeric values for position construction."
        ) from exc
    if values.isna().any():
        raise PositionConstructorError(
            f"Signal column {signal.value_column!r} must not contain NaN values during position construction."
        )
    finite_mask = values.map(math.isfinite)
    if not bool(finite_mask.all()):
        raise PositionConstructorError(
            f"Signal column {signal.value_column!r} must contain only finite numeric values during position construction."
        )
    return values


def finalize_positions(signal: Signal, positions: pd.Series) -> pd.DataFrame:
    if not isinstance(positions, pd.Series):
        raise PositionConstructorError("Constructed positions must be returned as a pandas Series before finalization.")
    if not positions.index.equals(signal.data.index):
        raise PositionConstructorError("Constructed positions must preserve the input signal index.")

    frame = base_position_frame(signal)
    try:
        numeric_positions = pd.to_numeric(positions, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise PositionConstructorError("Constructed positions must be numeric.") from exc

    if numeric_positions.isna().any():
        raise PositionConstructorError("Constructed positions must not contain NaN values.")
    finite_mask = numeric_positions.map(math.isfinite)
    if not bool(finite_mask.all()):
        raise PositionConstructorError("Constructed positions must contain only finite numeric values.")

    frame["position"] = numeric_positions
    validate_position_frame(frame, signal=signal)
    return frame


def validate_position_frame(frame: pd.DataFrame, *, signal: Signal) -> None:
    missing = [column for column in _REQUIRED_POSITION_COLUMNS if column not in frame.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise PositionConstructorError(f"Position frame is missing required columns: {formatted}.")
    if len(frame) != len(signal.data):
        raise PositionConstructorError("Position frame row count must match the input signal row count.")
    if not frame.index.equals(signal.data.index):
        raise PositionConstructorError("Position frame must preserve the input signal index.")

    actual_symbols = frame["symbol"].astype("string")
    expected_symbols = signal.data["symbol"].astype("string")
    if not actual_symbols.equals(expected_symbols):
        raise PositionConstructorError("Position frame must preserve input symbol ordering.")

    actual_ts = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    expected_ts = pd.to_datetime(signal.data["ts_utc"], utc=True, errors="coerce")
    if not actual_ts.equals(expected_ts):
        raise PositionConstructorError("Position frame must preserve input ts_utc ordering.")


def require_float(
    params: dict[str, Any],
    *,
    name: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if name not in params:
        raise PositionConstructorError(f"Position constructor parameter {name!r} is required.")
    try:
        value = float(params[name])
    except (TypeError, ValueError) as exc:
        raise PositionConstructorError(f"Position constructor parameter {name!r} must be numeric.") from exc
    if math.isnan(value) or math.isinf(value):
        raise PositionConstructorError(f"Position constructor parameter {name!r} must be finite.")
    if positive and value <= 0.0:
        raise PositionConstructorError(f"Position constructor parameter {name!r} must be greater than zero.")
    if non_negative and value < 0.0:
        raise PositionConstructorError(f"Position constructor parameter {name!r} must be non-negative.")
    return value


__all__ = [
    "PositionConstructor",
    "PositionConstructorError",
    "base_position_frame",
    "finalize_positions",
    "require_float",
    "signal_values",
    "validate_position_frame",
]
