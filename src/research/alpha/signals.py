from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

import pandas as pd

from src.research.alpha.cross_section import validate_cross_section_input
from src.research.signal_semantics import (
    Signal,
    create_signal,
    score_to_binary_long_only,
    score_to_cross_section_rank,
    score_to_signed_zscore,
    score_to_ternary_long_short,
)

SignalMappingPolicy = Literal[
    "rank_long_short",
    "zscore_continuous",
    "top_bottom_quantile",
    "long_only_top_quantile",
]

_SUPPORTED_POLICIES: tuple[str, ...] = (
    "rank_long_short",
    "zscore_continuous",
    "top_bottom_quantile",
    "long_only_top_quantile",
)
_OPTIONAL_STRUCTURAL_COLUMNS: tuple[str, ...] = ("timeframe",)
_POLICY_TO_SIGNAL_TYPE: dict[str, str] = {
    "rank_long_short": "cross_section_rank",
    "zscore_continuous": "signed_zscore",
    "top_bottom_quantile": "ternary_quantile",
    "long_only_top_quantile": "binary_signal",
}


class AlphaSignalMappingError(ValueError):
    """Raised when deterministic alpha-to-signal mapping is invalid."""


@dataclass(frozen=True)
class AlphaSignalMappingConfig:
    """Explicit deterministic alpha-to-signal mapping configuration."""

    policy: SignalMappingPolicy
    prediction_column: str = "prediction_score"
    output_column: str = "signal"
    quantile: float | None = None
    signal_type: str | None = None
    signal_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlphaSignalMappingResult:
    """Structured deterministic signal mapping output."""

    config: AlphaSignalMappingConfig
    row_count: int
    symbol_count: int
    timestamp_count: int
    signals: pd.DataFrame
    signal: Signal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def map_alpha_predictions_to_signals(
    df: pd.DataFrame,
    config: AlphaSignalMappingConfig | dict[str, Any],
) -> AlphaSignalMappingResult:
    """Map raw alpha prediction scores into explicit cross-sectional signals."""

    resolved_config = resolve_signal_mapping_config(config)
    validated = _validate_signal_mapping_input(df, prediction_column=resolved_config.prediction_column)
    prediction_signal = create_signal(
        validated.loc[:, _signal_input_columns(validated, prediction_column=resolved_config.prediction_column)].copy(deep=True),
        signal_type="prediction_score",
        value_column=resolved_config.prediction_column,
        parameters={"min_cross_section_size": 2},
        source={"layer": "alpha", "component": "prediction_mapping"},
        metadata={"mapping_policy": resolved_config.policy},
    )
    typed_signal = _map_typed_prediction_signal(prediction_signal, config=resolved_config)
    output = typed_signal.data.copy(deep=True)
    output.attrs = dict(typed_signal.data.attrs)

    return AlphaSignalMappingResult(
        config=resolved_config,
        row_count=int(len(output)),
        symbol_count=int(output["symbol"].astype("string").nunique()),
        timestamp_count=int(output["ts_utc"].nunique()),
        signals=output,
        signal=typed_signal,
        metadata={
            "cross_section_key": "ts_utc",
            "input_columns": list(validated.columns),
            "output_columns": list(output.columns),
            "tie_breaker": "prediction_score then symbol ascending",
            "signal_type": typed_signal.signal_type,
            "signal_version": typed_signal.version,
        },
    )


def resolve_signal_mapping_config(
    config: AlphaSignalMappingConfig | dict[str, Any],
) -> AlphaSignalMappingConfig:
    """Normalize one signal mapping config mapping or dataclass."""

    if isinstance(config, AlphaSignalMappingConfig):
        resolved = config
    elif isinstance(config, dict):
        resolved = AlphaSignalMappingConfig(
            policy=str(config.get("policy")),
            prediction_column=str(config.get("prediction_column", "prediction_score")),
            output_column=str(config.get("output_column", "signal")),
            quantile=config.get("quantile"),
            signal_type=None if config.get("signal_type") is None else str(config.get("signal_type")),
            signal_params={} if not isinstance(config.get("signal_params"), dict) else dict(config["signal_params"]),
            metadata={} if not isinstance(config.get("metadata"), dict) else dict(config["metadata"]),
        )
    else:
        raise TypeError("Signal mapping config must be an AlphaSignalMappingConfig or dictionary.")

    if resolved.policy not in _SUPPORTED_POLICIES:
        formatted = ", ".join(_SUPPORTED_POLICIES)
        raise AlphaSignalMappingError(f"Unsupported signal mapping policy {resolved.policy!r}. Expected one of: {formatted}.")

    prediction_column = resolved.prediction_column.strip()
    output_column = resolved.output_column.strip()
    if not prediction_column:
        raise AlphaSignalMappingError("prediction_column must be a non-empty string.")
    if not output_column:
        raise AlphaSignalMappingError("output_column must be a non-empty string.")
    if output_column in {"symbol", "ts_utc", "timeframe"}:
        raise AlphaSignalMappingError("output_column must not overwrite structural columns.")

    quantile = _validate_quantile(policy=resolved.policy, quantile=resolved.quantile)
    signal_type = resolved.signal_type or _POLICY_TO_SIGNAL_TYPE[resolved.policy]
    if signal_type != _POLICY_TO_SIGNAL_TYPE[resolved.policy]:
        raise AlphaSignalMappingError(
            f"Policy {resolved.policy!r} must declare signal_type {_POLICY_TO_SIGNAL_TYPE[resolved.policy]!r}, "
            f"received {signal_type!r}."
        )
    signal_params = dict(resolved.signal_params)
    if quantile is not None:
        signal_params.setdefault("quantile", quantile)
    return AlphaSignalMappingConfig(
        policy=resolved.policy,
        prediction_column=prediction_column,
        output_column=output_column,
        quantile=quantile,
        signal_type=signal_type,
        signal_params=signal_params,
        metadata=dict(resolved.metadata),
    )


def _validate_signal_mapping_input(df: pd.DataFrame, *, prediction_column: str) -> pd.DataFrame:
    validated = validate_cross_section_input(df)
    if prediction_column not in validated.columns:
        raise AlphaSignalMappingError(
            f"Signal mapping input must include prediction column {prediction_column!r}."
        )

    normalized = validated.copy(deep=True)
    normalized.attrs = {}
    try:
        normalized[prediction_column] = pd.to_numeric(normalized[prediction_column], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise AlphaSignalMappingError("Signal mapping prediction column must be numeric.") from exc

    if normalized[prediction_column].isna().any():
        raise AlphaSignalMappingError("Signal mapping prediction column must not contain NaN values.")
    is_finite = pd.Series(map(math.isfinite, normalized[prediction_column]), index=normalized.index)
    if not bool(is_finite.all()):
        raise AlphaSignalMappingError("Signal mapping prediction column must contain only finite values.")
    return normalized


def _validate_quantile(*, policy: str, quantile: float | None) -> float | None:
    if policy in {"rank_long_short", "zscore_continuous"}:
        if quantile is not None:
            raise AlphaSignalMappingError(f"Policy {policy!r} does not accept a quantile parameter.")
        return None

    if quantile is None:
        raise AlphaSignalMappingError(f"Policy {policy!r} requires a quantile parameter.")

    try:
        normalized = float(quantile)
    except (TypeError, ValueError) as exc:
        raise AlphaSignalMappingError("quantile must be numeric when provided.") from exc

    if math.isnan(normalized) or math.isinf(normalized):
        raise AlphaSignalMappingError("quantile must be finite when provided.")
    if policy == "top_bottom_quantile":
        if normalized <= 0.0 or normalized > 0.5:
            raise AlphaSignalMappingError("top_bottom_quantile requires quantile in the interval (0, 0.5].")
    elif normalized <= 0.0 or normalized > 1.0:
        raise AlphaSignalMappingError("long_only_top_quantile requires quantile in the interval (0, 1.0].")
    return normalized


def _signal_input_columns(df: pd.DataFrame, *, prediction_column: str) -> list[str]:
    columns = ["symbol", "ts_utc"]
    columns.extend(column for column in _OPTIONAL_STRUCTURAL_COLUMNS if column in df.columns)
    columns.append(prediction_column)
    return columns


def _map_typed_prediction_signal(
    signal: Signal,
    *,
    config: AlphaSignalMappingConfig,
) -> Signal:
    if config.policy == "rank_long_short":
        return score_to_cross_section_rank(signal, output_column=config.output_column)
    if config.policy == "zscore_continuous":
        clip = config.signal_params.get("clip")
        return score_to_signed_zscore(signal, output_column=config.output_column, clip=clip)
    if config.policy == "top_bottom_quantile":
        return score_to_ternary_long_short(
            signal,
            output_column=config.output_column,
            quantile=float(config.quantile),
        )
    if config.policy == "long_only_top_quantile":
        return score_to_binary_long_only(
            signal,
            output_column=config.output_column,
            quantile=float(config.quantile),
        )
    raise AlphaSignalMappingError(f"Unsupported signal mapping policy {config.policy!r}.")


__all__ = [
    "AlphaSignalMappingConfig",
    "AlphaSignalMappingError",
    "AlphaSignalMappingResult",
    "SignalMappingPolicy",
    "map_alpha_predictions_to_signals",
    "resolve_signal_mapping_config",
]
