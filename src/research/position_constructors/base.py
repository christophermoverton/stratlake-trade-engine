from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any

import pandas as pd

from src.research.registry import canonicalize_value
from src.research.short_availability import (
    ShortAvailabilityConstraint,
    apply_short_availability_constraints,
)
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


def require_optional_float(
    params: dict[str, Any],
    *,
    name: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float | None:
    if name not in params or params[name] is None:
        return None
    return require_float(params, name=name, positive=positive, non_negative=non_negative)


def require_optional_int(
    params: dict[str, Any],
    *,
    name: str,
    non_negative: bool = False,
) -> int | None:
    if name not in params or params[name] is None:
        return None
    value = params[name]
    if isinstance(value, bool):
        raise PositionConstructorError(f"Position constructor parameter {name!r} must be an integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise PositionConstructorError(f"Position constructor parameter {name!r} must be an integer.") from exc
    if normalized != value and not (isinstance(value, float) and value.is_integer()):
        raise PositionConstructorError(f"Position constructor parameter {name!r} must be an integer.")
    if non_negative and normalized < 0:
        raise PositionConstructorError(f"Position constructor parameter {name!r} must be non-negative.")
    return normalized


def apply_directional_position_controls(
    signal: Signal,
    positions: pd.Series,
    params: dict[str, Any],
    *,
    constructor_id: str,
) -> tuple[pd.Series, dict[str, Any]]:
    validation = validate_asymmetry_parameters(params, constructor_id)
    if not validation["valid"]:
        raise PositionConstructorError(validation["error_message"])

    controls = validation["asymmetry_params"]
    diagnostics: dict[str, Any] = {
        "enabled": bool(controls),
        "controls": canonicalize_value(dict(controls)),
        "warnings": list(validation["warnings"]),
        "short_availability": {},
    }
    if not controls:
        return positions, diagnostics

    adjusted = positions.astype("float64").copy()
    long_scale = float(controls.get("long_position_scale", 1.0))
    short_scale = float(controls.get("short_position_scale", 1.0))
    max_long_weight = controls.get("max_long_weight")
    max_short_weight = controls.get("max_short_weight")
    max_long_positions = controls.get("max_long_positions")
    max_short_positions = controls.get("max_short_positions")
    short_max_exposure = controls.get("short_max_exposure")
    exclude_short = bool(controls.get("exclude_short", False))
    short_constraint = ShortAvailabilityConstraint(
        short_available=controls.get("short_availability"),
        hard_to_borrow=controls.get("hard_to_borrow"),
        policy=str(controls.get("short_availability_policy", "exclude")),
        hard_to_borrow_penalty_bps=float(controls.get("hard_to_borrow_penalty_bps", 0.0)),
        max_short_positions_with_constraints=controls.get("max_short_positions_with_constraints"),
    )

    for ts_utc, group in signal.data.groupby("ts_utc", sort=False):
        group_positions = adjusted.loc[group.index].copy()
        if exclude_short:
            group_positions.loc[group_positions < 0.0] = 0.0
        if long_scale != 1.0:
            long_mask = group_positions > 0.0
            group_positions.loc[long_mask] = group_positions.loc[long_mask] * long_scale
        if short_scale != 1.0:
            short_mask = group_positions < 0.0
            group_positions.loc[short_mask] = group_positions.loc[short_mask] * short_scale
        if max_long_weight is not None:
            long_mask = group_positions > 0.0
            group_positions.loc[long_mask] = group_positions.loc[long_mask].clip(upper=float(max_long_weight))
        if max_short_weight is not None:
            short_mask = group_positions < 0.0
            clipped = group_positions.loc[short_mask].abs().clip(upper=float(max_short_weight))
            group_positions.loc[short_mask] = -clipped
        if max_long_positions is not None:
            long_index = group_positions.loc[group_positions > 0.0].abs().sort_values(ascending=False, kind="stable").index
            if len(long_index) > int(max_long_positions):
                group_positions.loc[long_index[int(max_long_positions):]] = 0.0
        if max_short_positions is not None:
            short_index = group_positions.loc[group_positions < 0.0].abs().sort_values(ascending=False, kind="stable").index
            if len(short_index) > int(max_short_positions):
                group_positions.loc[short_index[int(max_short_positions):]] = 0.0
        if short_max_exposure is not None:
            short_mask = group_positions < 0.0
            current_short_exposure = float(group_positions.loc[short_mask].abs().sum())
            if current_short_exposure > float(short_max_exposure) and current_short_exposure > 0.0:
                scale = float(short_max_exposure) / current_short_exposure
                group_positions.loc[short_mask] = group_positions.loc[short_mask] * scale

        constrained_positions, short_diagnostics = apply_short_availability_constraints(
            group_positions,
            group["symbol"].astype("string"),
            short_constraint,
        )
        diagnostics["short_availability"][_format_ts_key(ts_utc)] = short_diagnostics
        adjusted.loc[group.index] = constrained_positions

    diagnostics["net_exposure"] = float(adjusted.sum())
    diagnostics["gross_exposure"] = float(adjusted.abs().sum())
    diagnostics["long_exposure"] = float(adjusted.clip(lower=0.0).sum())
    diagnostics["short_exposure"] = float(adjusted.clip(upper=0.0).abs().sum())
    return adjusted.astype("float64"), diagnostics


def _format_ts_key(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_asymmetry_parameters(
    params: dict[str, Any],
    constructor_id: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": True,
        "constructor_id": constructor_id,
        "asymmetry_params": {},
        "warnings": [],
        "issues": [],
    }

    asymmetry_param_names = [
        "max_long_positions",
        "max_short_positions",
        "max_long_weight",
        "max_short_weight",
        "long_position_scale",
        "short_position_scale",
        "short_max_exposure",
        "exclude_short",
        "short_availability",
        "hard_to_borrow",
        "short_availability_policy",
        "hard_to_borrow_penalty_bps",
        "max_short_positions_with_constraints",
    ]

    for param_name in asymmetry_param_names:
        if param_name in params:
            result["asymmetry_params"][param_name] = params[param_name]

    if not result["asymmetry_params"]:
        return result

    for param_name, param_value in result["asymmetry_params"].items():
        try:
            if param_name in {"max_long_positions", "max_short_positions", "max_short_positions_with_constraints"}:
                if require_optional_int({param_name: param_value}, name=param_name, non_negative=True) is not None:
                    continue
            elif param_name in {
                "max_long_weight",
                "max_short_weight",
                "long_position_scale",
                "short_position_scale",
                "short_max_exposure",
                "hard_to_borrow_penalty_bps",
            }:
                normalized = require_optional_float({param_name: param_value}, name=param_name, non_negative=True)
                if normalized is not None and normalized > 10.0:
                    result["issues"].append(
                        f"Parameter {param_name!r} must be between 0 and 10, got {normalized}"
                    )
                    result["valid"] = False
            elif param_name == "exclude_short":
                if not isinstance(param_value, bool):
                    result["issues"].append(
                        f"Parameter {param_name!r} must be boolean."
                    )
                    result["valid"] = False
            elif param_name in {"short_availability", "hard_to_borrow"}:
                if not isinstance(param_value, dict):
                    result["issues"].append(f"Parameter {param_name!r} must be a symbol-to-flag mapping.")
                    result["valid"] = False
            elif param_name == "short_availability_policy":
                if str(param_value) not in {"exclude", "cap", "penalty"}:
                    result["issues"].append(
                        "Parameter 'short_availability_policy' must be one of ['exclude', 'cap', 'penalty']."
                    )
                    result["valid"] = False
        except (TypeError, ValueError) as exc:
            result["issues"].append(
                f"Parameter {param_name!r} has invalid type/value: {exc}"
            )
            result["valid"] = False

    if "max_long_positions" in result["asymmetry_params"] and "max_short_positions" in result["asymmetry_params"]:
        max_long_positions = int(result["asymmetry_params"]["max_long_positions"])
        max_short_positions = int(result["asymmetry_params"]["max_short_positions"])
        if max_long_positions == 0 and max_short_positions == 0:
            result["issues"].append("At least one of long or short positions must be permitted.")
            result["valid"] = False
    elif "max_long_positions" in result["asymmetry_params"] or "max_short_positions" in result["asymmetry_params"]:
        result["warnings"].append(
            "Only one side (long or short) has position constraints. "
            "Consider symmetrically constraining both sides."
        )

    if result["issues"]:
        result["error_message"] = "; ".join(result["issues"])

    return result


__all__ = [
    "PositionConstructor",
    "PositionConstructorError",
    "apply_directional_position_controls",
    "base_position_frame",
    "finalize_positions",
    "require_float",
    "require_optional_float",
    "require_optional_int",
    "signal_values",
    "validate_position_frame",
    "validate_asymmetry_parameters",
]
