from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

import pandas as pd

from src.research.alpha.cross_section import iter_cross_sections, validate_cross_section_input

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


class AlphaSignalMappingError(ValueError):
    """Raised when deterministic alpha-to-signal mapping is invalid."""


@dataclass(frozen=True)
class AlphaSignalMappingConfig:
    """Explicit deterministic alpha-to-signal mapping configuration."""

    policy: SignalMappingPolicy
    prediction_column: str = "prediction_score"
    output_column: str = "signal"
    quantile: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlphaSignalMappingResult:
    """Structured deterministic signal mapping output."""

    config: AlphaSignalMappingConfig
    row_count: int
    symbol_count: int
    timestamp_count: int
    signals: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def map_alpha_predictions_to_signals(
    df: pd.DataFrame,
    config: AlphaSignalMappingConfig | dict[str, Any],
) -> AlphaSignalMappingResult:
    """Map raw alpha prediction scores into explicit cross-sectional signals."""

    resolved_config = resolve_signal_mapping_config(config)
    validated = _validate_signal_mapping_input(df, prediction_column=resolved_config.prediction_column)

    signal_values = pd.Series(index=validated.index, dtype="float64")
    for _, cross_section in iter_cross_sections(validated, columns=[resolved_config.prediction_column, *[
        column for column in _OPTIONAL_STRUCTURAL_COLUMNS if column in validated.columns
    ]]):
        signal_values.loc[cross_section.index] = _map_one_cross_section(cross_section, config=resolved_config)

    output_columns = ["symbol", "ts_utc"]
    output_columns.extend(column for column in _OPTIONAL_STRUCTURAL_COLUMNS if column in validated.columns)
    output = validated.loc[:, output_columns].copy(deep=True)
    output[resolved_config.prediction_column] = validated[resolved_config.prediction_column].astype("float64")
    output[resolved_config.output_column] = signal_values.astype("float64")
    output.attrs = {}

    return AlphaSignalMappingResult(
        config=resolved_config,
        row_count=int(len(output)),
        symbol_count=int(output["symbol"].astype("string").nunique()),
        timestamp_count=int(output["ts_utc"].nunique()),
        signals=output,
        metadata={
            "cross_section_key": "ts_utc",
            "input_columns": list(validated.columns),
            "output_columns": list(output.columns),
            "tie_breaker": "prediction_score then symbol ascending",
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
    return AlphaSignalMappingConfig(
        policy=resolved.policy,
        prediction_column=prediction_column,
        output_column=output_column,
        quantile=quantile,
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


def _map_one_cross_section(
    cross_section: pd.DataFrame,
    *,
    config: AlphaSignalMappingConfig,
) -> pd.Series:
    values = pd.to_numeric(cross_section[config.prediction_column], errors="raise").astype("float64")
    if config.policy == "rank_long_short":
        return _rank_long_short(values, symbols=cross_section["symbol"])
    if config.policy == "zscore_continuous":
        return _zscore_continuous(values)
    if config.policy == "top_bottom_quantile":
        return _top_bottom_quantile(values, symbols=cross_section["symbol"], quantile=float(config.quantile))
    if config.policy == "long_only_top_quantile":
        return _long_only_top_quantile(values, symbols=cross_section["symbol"], quantile=float(config.quantile))
    raise AlphaSignalMappingError(f"Unsupported signal mapping policy {config.policy!r}.")


def _rank_long_short(values: pd.Series, *, symbols: pd.Series) -> pd.Series:
    count = int(values.shape[0])
    if count <= 1:
        return pd.Series(0.0, index=values.index, dtype="float64")

    ordered = _ordered_index(values, symbols=symbols, ascending_scores=False)
    signals = pd.Series(index=values.index, dtype="float64")
    denominator = float(count - 1)
    for rank_position, row_index in enumerate(ordered, start=1):
        signals.loc[row_index] = 1.0 - (2.0 * float(rank_position - 1) / denominator)
    return signals.astype("float64")


def _zscore_continuous(values: pd.Series) -> pd.Series:
    count = int(values.shape[0])
    if count <= 1:
        return pd.Series(0.0, index=values.index, dtype="float64")
    mean_value = float(values.mean())
    std_value = float(values.std(ddof=0))
    if std_value == 0.0:
        return pd.Series(0.0, index=values.index, dtype="float64")
    return ((values - mean_value) / std_value).astype("float64")


def _top_bottom_quantile(values: pd.Series, *, symbols: pd.Series, quantile: float) -> pd.Series:
    count = int(values.shape[0])
    if count <= 1:
        return pd.Series(0.0, index=values.index, dtype="float64")

    selected_count = min(max(int(math.ceil(float(count) * quantile)), 1), count // 2)
    signals = pd.Series(0.0, index=values.index, dtype="float64")
    if selected_count == 0:
        return signals

    top_rows = _ordered_index(values, symbols=symbols, ascending_scores=False)[:selected_count]
    bottom_rows = _ordered_index(values, symbols=symbols, ascending_scores=True)[:selected_count]
    signals.loc[top_rows] = 1.0
    signals.loc[bottom_rows] = -1.0
    return signals.astype("float64")


def _long_only_top_quantile(values: pd.Series, *, symbols: pd.Series, quantile: float) -> pd.Series:
    count = int(values.shape[0])
    if count == 0:
        return pd.Series(dtype="float64", index=values.index)

    selected_count = min(max(int(math.ceil(float(count) * quantile)), 1), count)
    signals = pd.Series(0.0, index=values.index, dtype="float64")
    top_rows = _ordered_index(values, symbols=symbols, ascending_scores=False)[:selected_count]
    signals.loc[top_rows] = 1.0
    return signals.astype("float64")


def _ordered_index(values: pd.Series, *, symbols: pd.Series, ascending_scores: bool) -> list[Any]:
    order_frame = pd.DataFrame(
        {
            "_index": values.index,
            "_prediction_score": values.astype("float64"),
            "_symbol": symbols.astype("string"),
        }
    )
    ordered = order_frame.sort_values(
        ["_prediction_score", "_symbol"],
        ascending=[ascending_scores, True],
        kind="stable",
    )
    return ordered["_index"].tolist()


__all__ = [
    "AlphaSignalMappingConfig",
    "AlphaSignalMappingError",
    "AlphaSignalMappingResult",
    "SignalMappingPolicy",
    "map_alpha_predictions_to_signals",
    "resolve_signal_mapping_config",
]
