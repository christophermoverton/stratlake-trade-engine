from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.research.regimes.taxonomy import (
    REGIME_DIMENSIONS,
    REGIME_OUTPUT_COLUMNS,
    REGIME_STATE_COLUMNS,
    TAXONOMY_VERSION,
    UNDEFINED_REGIME_LABEL,
    taxonomy_payload,
)
from src.research.regimes.validation import validate_regime_labels


class RegimeClassificationError(ValueError):
    """Raised when deterministic regime classification cannot be completed."""


@dataclass(frozen=True)
class RegimeClassificationConfig:
    """Deterministic first-pass regime classification parameters."""

    volatility_window: int = 20
    volatility_low_quantile: float = 0.33
    volatility_high_quantile: float = 0.67
    trend_window: int = 20
    trend_return_threshold: float = 0.02
    drawdown_threshold: float = 0.10
    near_peak_drawdown_threshold: float = 0.01
    stress_window: int = 20
    stress_correlation_threshold: float = 0.75
    stress_dispersion_quantile: float = 0.80
    min_cross_section_size: int = 2
    time_column: str = "ts_utc"
    symbol_column: str = "symbol"
    price_column: str = "close"
    return_column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drawdown_threshold": self.drawdown_threshold,
            "metadata": dict(sorted(self.metadata.items())),
            "min_cross_section_size": self.min_cross_section_size,
            "near_peak_drawdown_threshold": self.near_peak_drawdown_threshold,
            "price_column": self.price_column,
            "return_column": self.return_column,
            "stress_correlation_threshold": self.stress_correlation_threshold,
            "stress_dispersion_quantile": self.stress_dispersion_quantile,
            "stress_window": self.stress_window,
            "symbol_column": self.symbol_column,
            "time_column": self.time_column,
            "trend_return_threshold": self.trend_return_threshold,
            "trend_window": self.trend_window,
            "volatility_high_quantile": self.volatility_high_quantile,
            "volatility_low_quantile": self.volatility_low_quantile,
            "volatility_window": self.volatility_window,
        }


@dataclass(frozen=True)
class RegimeClassificationResult:
    """Structured output for deterministic market-regime classification."""

    labels: pd.DataFrame
    metrics: pd.DataFrame
    metadata: dict[str, Any]
    config: RegimeClassificationConfig
    taxonomy_version: str = TAXONOMY_VERSION


def classify_market_regimes(
    market_data: pd.DataFrame,
    *,
    config: RegimeClassificationConfig | dict[str, Any] | None = None,
) -> RegimeClassificationResult:
    """Classify timestamp-level market regimes from local price or return data."""

    resolved_config = resolve_regime_classification_config(config)
    _validate_config(resolved_config)
    returns_matrix = _market_return_matrix(market_data, resolved_config)
    market_returns = returns_matrix.mean(axis=1, skipna=True).astype("float64")
    market_returns.loc[returns_matrix.notna().sum(axis=1).eq(0)] = float("nan")

    volatility_metric = market_returns.rolling(
        resolved_config.volatility_window,
        min_periods=resolved_config.volatility_window,
    ).std(ddof=1)
    trend_metric = market_returns.rolling(
        resolved_config.trend_window,
        min_periods=resolved_config.trend_window,
    ).apply(_compounded_return, raw=False)
    drawdown_metric = _drawdown_series(market_returns)
    stress_correlation_metric = _rolling_average_pairwise_correlation(
        returns_matrix,
        window=resolved_config.stress_window,
        min_cross_section_size=resolved_config.min_cross_section_size,
    )
    stress_dispersion_metric = returns_matrix.std(axis=1, ddof=1).astype("float64")
    stress_dispersion_metric.loc[
        returns_matrix.notna().sum(axis=1) < resolved_config.min_cross_section_size
    ] = float("nan")
    stress_dispersion_metric = stress_dispersion_metric.rolling(
        resolved_config.stress_window,
        min_periods=resolved_config.stress_window,
    ).mean()

    volatility_thresholds = _quantile_thresholds(
        volatility_metric,
        low_quantile=resolved_config.volatility_low_quantile,
        high_quantile=resolved_config.volatility_high_quantile,
    )
    stress_dispersion_threshold = _single_quantile_threshold(
        stress_dispersion_metric,
        quantile=resolved_config.stress_dispersion_quantile,
    )

    labels = pd.DataFrame(
        {
            "ts_utc": returns_matrix.index,
            "volatility_state": _classify_volatility(volatility_metric, thresholds=volatility_thresholds),
            "trend_state": _classify_trend(trend_metric, threshold=resolved_config.trend_return_threshold),
            "drawdown_recovery_state": _classify_drawdown_recovery(
                drawdown_metric,
                market_returns=market_returns,
                drawdown_threshold=resolved_config.drawdown_threshold,
                near_peak_threshold=resolved_config.near_peak_drawdown_threshold,
            ),
            "stress_state": _classify_stress(
                stress_correlation_metric,
                stress_dispersion_metric,
                correlation_threshold=resolved_config.stress_correlation_threshold,
                dispersion_threshold=stress_dispersion_threshold,
            ),
            "volatility_metric": volatility_metric.to_numpy(copy=True),
            "trend_metric": trend_metric.to_numpy(copy=True),
            "drawdown_metric": drawdown_metric.to_numpy(copy=True),
            "stress_correlation_metric": stress_correlation_metric.to_numpy(copy=True),
            "stress_dispersion_metric": stress_dispersion_metric.to_numpy(copy=True),
        }
    ).reset_index(drop=True)
    labels["regime_label"] = labels.apply(_composite_label_for_row, axis=1)
    labels["is_defined"] = labels.loc[:, list(REGIME_STATE_COLUMNS.values())].ne(UNDEFINED_REGIME_LABEL).all(axis=1)
    labels = labels.loc[:, list(REGIME_OUTPUT_COLUMNS)]
    labels = validate_regime_labels(labels)

    metrics = labels.loc[:, ["ts_utc", *[column for column in REGIME_OUTPUT_COLUMNS if column.endswith("_metric")]]]
    metadata = _classification_metadata(
        labels=labels,
        returns_matrix=returns_matrix,
        config=resolved_config,
        volatility_thresholds=volatility_thresholds,
        stress_dispersion_threshold=stress_dispersion_threshold,
    )
    labels.attrs["regime_metadata"] = metadata
    metrics.attrs["regime_metadata"] = metadata

    return RegimeClassificationResult(
        labels=labels,
        metrics=metrics,
        metadata=metadata,
        config=resolved_config,
    )


def resolve_regime_classification_config(
    payload: RegimeClassificationConfig | dict[str, Any] | None,
) -> RegimeClassificationConfig:
    if payload is None:
        return RegimeClassificationConfig()
    if isinstance(payload, RegimeClassificationConfig):
        return payload
    if not isinstance(payload, dict):
        raise RegimeClassificationError("Regime classification config must be a dictionary when provided.")
    allowed_fields = set(RegimeClassificationConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed_fields)
    if unknown:
        raise RegimeClassificationError(
            f"Regime classification config contains unsupported fields: {unknown}."
        )
    return RegimeClassificationConfig(**payload)


def _validate_config(config: RegimeClassificationConfig) -> None:
    for field_name in ("volatility_window", "trend_window", "stress_window", "min_cross_section_size"):
        value = getattr(config, field_name)
        if not isinstance(value, int) or value < 1:
            raise RegimeClassificationError(f"Regime config field {field_name!r} must be a positive integer.")
    if config.min_cross_section_size < 2:
        raise RegimeClassificationError("Regime config field 'min_cross_section_size' must be at least 2.")
    for field_name in ("volatility_low_quantile", "volatility_high_quantile", "stress_dispersion_quantile"):
        value = float(getattr(config, field_name))
        if value < 0.0 or value > 1.0:
            raise RegimeClassificationError(f"Regime config field {field_name!r} must be between 0.0 and 1.0.")
    if config.volatility_low_quantile > config.volatility_high_quantile:
        raise RegimeClassificationError(
            "Regime config field 'volatility_low_quantile' must be <= 'volatility_high_quantile'."
        )
    for field_name in (
        "trend_return_threshold",
        "drawdown_threshold",
        "near_peak_drawdown_threshold",
        "stress_correlation_threshold",
    ):
        value = float(getattr(config, field_name))
        if value < 0.0:
            raise RegimeClassificationError(f"Regime config field {field_name!r} must be non-negative.")
    if config.near_peak_drawdown_threshold > config.drawdown_threshold:
        raise RegimeClassificationError(
            "Regime config field 'near_peak_drawdown_threshold' must be <= 'drawdown_threshold'."
        )


def _market_return_matrix(market_data: pd.DataFrame, config: RegimeClassificationConfig) -> pd.DataFrame:
    if not isinstance(market_data, pd.DataFrame):
        raise TypeError("Market data must be provided as a pandas DataFrame.")
    if market_data.empty:
        raise RegimeClassificationError("Market data must not be empty.")
    if config.time_column not in market_data.columns:
        raise RegimeClassificationError(
            f"Market data must include timestamp column {config.time_column!r}."
        )
    value_column = config.return_column if config.return_column is not None else config.price_column
    if value_column not in market_data.columns:
        raise RegimeClassificationError(
            f"Market data must include {value_column!r} for regime classification."
        )

    normalized = market_data.copy(deep=True)
    normalized.attrs = {}
    normalized[config.time_column] = pd.to_datetime(normalized[config.time_column], utc=True, errors="coerce")
    if normalized[config.time_column].isna().any():
        raise RegimeClassificationError(f"Market data column {config.time_column!r} contains unparsable timestamps.")
    normalized[value_column] = pd.to_numeric(normalized[value_column], errors="coerce").astype("float64")

    if config.symbol_column in normalized.columns:
        normalized[config.symbol_column] = normalized[config.symbol_column].astype("string").str.strip()
        if normalized[config.symbol_column].isna().any() or normalized[config.symbol_column].eq("").any():
            raise RegimeClassificationError(f"Market data column {config.symbol_column!r} contains blank symbols.")
        duplicate_mask = normalized.duplicated(subset=[config.time_column, config.symbol_column], keep=False)
        if duplicate_mask.any():
            row = normalized.loc[duplicate_mask, [config.time_column, config.symbol_column]].iloc[0]
            raise RegimeClassificationError(
                "Market data contains duplicate (ts_utc, symbol) rows. "
                f"First duplicate key: ({row[config.time_column]}, {row[config.symbol_column]!r})."
            )
        normalized = normalized.sort_values([config.time_column, config.symbol_column], kind="stable")
        if config.return_column is None:
            normalized["_regime_return"] = normalized.groupby(config.symbol_column, sort=False)[value_column].pct_change()
        else:
            normalized["_regime_return"] = normalized[value_column]
        matrix = normalized.pivot(index=config.time_column, columns=config.symbol_column, values="_regime_return")
        matrix.columns = pd.Index([str(column) for column in matrix.columns], name=None)
        matrix = matrix.loc[:, sorted(matrix.columns)]
    else:
        normalized = normalized.sort_values(config.time_column, kind="stable")
        if normalized[config.time_column].duplicated().any():
            duplicate_ts = normalized.loc[normalized[config.time_column].duplicated(keep=False), config.time_column].iloc[0]
            raise RegimeClassificationError(
                f"Market data contains duplicate timestamps. First duplicate ts_utc={duplicate_ts}."
            )
        series = normalized[value_column] if config.return_column is not None else normalized[value_column].pct_change()
        matrix = pd.DataFrame({"market": series.to_numpy(copy=True)}, index=normalized[config.time_column])

    matrix.index = pd.DatetimeIndex(matrix.index, name="ts_utc")
    matrix = matrix.sort_index(kind="stable").astype("float64")
    return matrix


def _compounded_return(values: pd.Series) -> float:
    if values.isna().any():
        return float("nan")
    return float((1.0 + values.astype("float64")).prod() - 1.0)


def _drawdown_series(market_returns: pd.Series) -> pd.Series:
    clean_returns = market_returns.fillna(0.0).astype("float64")
    equity = (1.0 + clean_returns).cumprod()
    drawdown = 1.0 - (equity / equity.cummax())
    drawdown.loc[market_returns.isna()] = float("nan")
    return drawdown.astype("float64")


def _rolling_average_pairwise_correlation(
    returns_matrix: pd.DataFrame,
    *,
    window: int,
    min_cross_section_size: int,
) -> pd.Series:
    values: list[float] = []
    for position in range(len(returns_matrix)):
        if position + 1 < window:
            values.append(float("nan"))
            continue
        window_frame = returns_matrix.iloc[position + 1 - window : position + 1]
        complete = window_frame.dropna(axis=1, how="any")
        if complete.shape[1] < min_cross_section_size:
            values.append(float("nan"))
            continue
        corr = complete.corr()
        pair_values: list[float] = []
        columns = list(corr.columns)
        for left_index, left_column in enumerate(columns):
            for right_column in columns[left_index + 1 :]:
                value = corr.loc[left_column, right_column]
                if pd.notna(value):
                    pair_values.append(float(value))
        values.append(float("nan") if not pair_values else float(pd.Series(pair_values, dtype="float64").mean()))
    return pd.Series(values, index=returns_matrix.index, dtype="float64", name="stress_correlation_metric")


def _quantile_thresholds(
    values: pd.Series,
    *,
    low_quantile: float,
    high_quantile: float,
) -> tuple[float | None, float | None]:
    valid = values.dropna().astype("float64")
    if valid.empty:
        return None, None
    return float(valid.quantile(low_quantile)), float(valid.quantile(high_quantile))


def _single_quantile_threshold(values: pd.Series, *, quantile: float) -> float | None:
    valid = values.dropna().astype("float64")
    if valid.empty:
        return None
    return float(valid.quantile(quantile))


def _classify_volatility(values: pd.Series, *, thresholds: tuple[float | None, float | None]) -> pd.Series:
    low_threshold, high_threshold = thresholds
    labels: list[str] = []
    for value in values:
        if pd.isna(value) or low_threshold is None or high_threshold is None:
            labels.append(UNDEFINED_REGIME_LABEL)
        elif float(value) <= low_threshold:
            labels.append("low_volatility")
        elif float(value) >= high_threshold:
            labels.append("high_volatility")
        else:
            labels.append("normal_volatility")
    return pd.Series(labels, index=values.index, dtype="string")


def _classify_trend(values: pd.Series, *, threshold: float) -> pd.Series:
    labels: list[str] = []
    for value in values:
        if pd.isna(value):
            labels.append(UNDEFINED_REGIME_LABEL)
        elif float(value) >= threshold:
            labels.append("uptrend")
        elif float(value) <= -threshold:
            labels.append("downtrend")
        else:
            labels.append("sideways")
    return pd.Series(labels, index=values.index, dtype="string")


def _classify_drawdown_recovery(
    drawdown: pd.Series,
    *,
    market_returns: pd.Series,
    drawdown_threshold: float,
    near_peak_threshold: float,
) -> pd.Series:
    labels: list[str] = []
    previous_drawdown = float("nan")
    for ts_utc, value in drawdown.items():
        if pd.isna(value) or pd.isna(market_returns.loc[ts_utc]):
            labels.append(UNDEFINED_REGIME_LABEL)
        elif float(value) <= near_peak_threshold:
            labels.append("near_peak")
        elif pd.notna(previous_drawdown) and float(value) < previous_drawdown:
            labels.append("recovery")
        elif float(value) >= drawdown_threshold:
            labels.append("drawdown")
        else:
            labels.append("underwater")
        if pd.notna(value):
            previous_drawdown = float(value)
    return pd.Series(labels, index=drawdown.index, dtype="string")


def _classify_stress(
    correlation: pd.Series,
    dispersion: pd.Series,
    *,
    correlation_threshold: float,
    dispersion_threshold: float | None,
) -> pd.Series:
    labels: list[str] = []
    for ts_utc, corr_value in correlation.items():
        dispersion_value = dispersion.loc[ts_utc]
        if pd.isna(corr_value) and (pd.isna(dispersion_value) or dispersion_threshold is None):
            labels.append(UNDEFINED_REGIME_LABEL)
        elif pd.notna(corr_value) and float(corr_value) >= correlation_threshold:
            labels.append("correlation_stress")
        elif (
            dispersion_threshold is not None
            and pd.notna(dispersion_value)
            and float(dispersion_value) >= dispersion_threshold
        ):
            labels.append("dispersion_stress")
        elif pd.notna(corr_value) or pd.notna(dispersion_value):
            labels.append("normal_stress")
        else:
            labels.append(UNDEFINED_REGIME_LABEL)
    return pd.Series(labels, index=correlation.index, dtype="string")


def _composite_label_for_row(row: pd.Series) -> str:
    return "|".join(
        f"{dimension}={row[REGIME_STATE_COLUMNS[dimension]]}"
        for dimension in REGIME_DIMENSIONS
    )


def _classification_metadata(
    *,
    labels: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    config: RegimeClassificationConfig,
    volatility_thresholds: tuple[float | None, float | None],
    stress_dispersion_threshold: float | None,
) -> dict[str, Any]:
    defined_counts = {
        dimension: int(labels[REGIME_STATE_COLUMNS[dimension]].ne(UNDEFINED_REGIME_LABEL).sum())
        for dimension in REGIME_DIMENSIONS
    }
    return {
        "config": config.to_dict(),
        "defined_counts": defined_counts,
        "dimensions": list(REGIME_DIMENSIONS),
        "input": {
            "row_count": int(len(returns_matrix)),
            "symbol_count": int(len(returns_matrix.columns)),
            "symbols": list(returns_matrix.columns),
            "ts_utc_end": None if returns_matrix.empty else returns_matrix.index.max().isoformat().replace("+00:00", "Z"),
            "ts_utc_start": None if returns_matrix.empty else returns_matrix.index.min().isoformat().replace("+00:00", "Z"),
        },
        "label_columns": list(REGIME_OUTPUT_COLUMNS),
        "taxonomy": taxonomy_payload(),
        "taxonomy_version": TAXONOMY_VERSION,
        "thresholds": {
            "drawdown": {
                "drawdown_threshold": config.drawdown_threshold,
                "near_peak_drawdown_threshold": config.near_peak_drawdown_threshold,
            },
            "stress": {
                "correlation_threshold": config.stress_correlation_threshold,
                "dispersion_threshold": stress_dispersion_threshold,
                "dispersion_quantile": config.stress_dispersion_quantile,
            },
            "trend": {
                "negative_threshold": -config.trend_return_threshold,
                "positive_threshold": config.trend_return_threshold,
            },
            "volatility": {
                "high_threshold": volatility_thresholds[1],
                "low_threshold": volatility_thresholds[0],
            },
        },
    }


__all__ = [
    "RegimeClassificationConfig",
    "RegimeClassificationError",
    "RegimeClassificationResult",
    "classify_market_regimes",
    "resolve_regime_classification_config",
]
