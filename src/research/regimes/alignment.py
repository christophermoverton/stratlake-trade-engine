from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.research.regimes.taxonomy import (
    REGIME_AUDIT_COLUMNS,
    REGIME_DIMENSIONS,
    REGIME_STATE_COLUMNS,
    UNDEFINED_REGIME_LABEL,
)
from src.research.regimes.validation import validate_regime_labels

RegimeUnavailablePolicy = Literal["raise", "mark_unmatched"]

_REGIME_ALIGNMENT_STATUS_DEFINED = "matched_defined"
_REGIME_ALIGNMENT_STATUS_UNDEFINED = "matched_undefined"
_REGIME_ALIGNMENT_STATUS_UNMATCHED = "unmatched_timestamp"
_REGIME_ALIGNMENT_STATUS_UNAVAILABLE = "regime_labels_unavailable"


class RegimeAlignmentError(ValueError):
    """Raised when deterministic regime alignment cannot be completed."""


@dataclass(frozen=True)
class RegimeAlignmentConfig:
    """Deterministic alignment behavior for attaching regime context to target datasets."""

    target_timestamp_column: str = "ts_utc"
    include_metric_columns: bool = False
    unavailable_policy: RegimeUnavailablePolicy = "raise"
    surface: str = "generic"
    output_prefix: str = "regime_"
    metadata: dict[str, object] = field(default_factory=dict)


def align_regime_labels(
    target: pd.DataFrame,
    regime_labels: pd.DataFrame | None,
    *,
    config: RegimeAlignmentConfig | dict[str, object] | None = None,
) -> pd.DataFrame:
    """Attach timestamp-exact regime context to a target dataset in stable row order."""

    resolved_config = _resolve_alignment_config(config)
    normalized_target = _normalize_target(target, config=resolved_config)

    if regime_labels is None:
        if resolved_config.unavailable_policy == "raise":
            raise RegimeAlignmentError(
                "Regime labels were not provided and unavailable_policy='raise'."
            )
        return _attach_unavailable_labels(normalized_target, config=resolved_config)

    normalized_labels = validate_regime_labels(regime_labels)
    projection = _regime_projection(
        normalized_labels,
        include_metrics=resolved_config.include_metric_columns,
        config=resolved_config,
    )

    aligned = normalized_target.copy(deep=True)
    aligned.attrs = {}
    label_columns = list(projection.columns)
    mapped = projection.reindex(aligned[resolved_config.target_timestamp_column])

    for column in label_columns:
        aligned[column] = mapped[column].to_numpy(copy=True)

    matched_mask = aligned[_label_column(resolved_config, "label")].notna()
    aligned[_label_column(resolved_config, "has_exact_timestamp_match")] = matched_mask.astype("bool")

    _fill_unmatched_rows(aligned, matched_mask=matched_mask, config=resolved_config)

    defined_mask = aligned[_label_column(resolved_config, "is_defined")].astype("bool")
    status = pd.Series(_REGIME_ALIGNMENT_STATUS_UNMATCHED, index=aligned.index, dtype="string")
    status.loc[matched_mask & ~defined_mask] = _REGIME_ALIGNMENT_STATUS_UNDEFINED
    status.loc[matched_mask & defined_mask] = _REGIME_ALIGNMENT_STATUS_DEFINED

    aligned[_label_column(resolved_config, "alignment_status")] = status
    aligned[_label_column(resolved_config, "surface")] = str(resolved_config.surface)
    return aligned


def align_regimes_to_strategy_timeseries(
    strategy_frame: pd.DataFrame,
    regime_labels: pd.DataFrame | None,
    *,
    timestamp_column: str = "ts_utc",
    include_metric_columns: bool = False,
    unavailable_policy: RegimeUnavailablePolicy = "raise",
) -> pd.DataFrame:
    """Attach regimes to strategy timestamp-level outputs."""

    return align_regime_labels(
        strategy_frame,
        regime_labels,
        config=RegimeAlignmentConfig(
            target_timestamp_column=timestamp_column,
            include_metric_columns=include_metric_columns,
            unavailable_policy=unavailable_policy,
            surface="strategy",
        ),
    )


def align_regimes_to_alpha_windows(
    alpha_frame: pd.DataFrame,
    regime_labels: pd.DataFrame | None,
    *,
    timestamp_column: str = "ts_utc",
    include_metric_columns: bool = False,
    unavailable_policy: RegimeUnavailablePolicy = "raise",
) -> pd.DataFrame:
    """Attach regimes to alpha-evaluation windows."""

    return align_regime_labels(
        alpha_frame,
        regime_labels,
        config=RegimeAlignmentConfig(
            target_timestamp_column=timestamp_column,
            include_metric_columns=include_metric_columns,
            unavailable_policy=unavailable_policy,
            surface="alpha",
        ),
    )


def align_regimes_to_portfolio_windows(
    portfolio_frame: pd.DataFrame,
    regime_labels: pd.DataFrame | None,
    *,
    timestamp_column: str = "ts_utc",
    include_metric_columns: bool = False,
    unavailable_policy: RegimeUnavailablePolicy = "raise",
) -> pd.DataFrame:
    """Attach regimes to portfolio timestamp-level outputs."""

    return align_regime_labels(
        portfolio_frame,
        regime_labels,
        config=RegimeAlignmentConfig(
            target_timestamp_column=timestamp_column,
            include_metric_columns=include_metric_columns,
            unavailable_policy=unavailable_policy,
            surface="portfolio",
        ),
    )


def _resolve_alignment_config(
    payload: RegimeAlignmentConfig | dict[str, object] | None,
) -> RegimeAlignmentConfig:
    if payload is None:
        return RegimeAlignmentConfig()
    if isinstance(payload, RegimeAlignmentConfig):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("Regime alignment config must be a dictionary when provided.")

    allowed_fields = set(RegimeAlignmentConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed_fields)
    if unknown:
        raise RegimeAlignmentError(
            f"Regime alignment config contains unsupported fields: {unknown}."
        )
    return RegimeAlignmentConfig(**payload)


def _normalize_target(target: pd.DataFrame, *, config: RegimeAlignmentConfig) -> pd.DataFrame:
    if not isinstance(target, pd.DataFrame):
        raise TypeError("Regime alignment target must be provided as a pandas DataFrame.")
    if config.target_timestamp_column not in target.columns:
        raise RegimeAlignmentError(
            f"Regime alignment target must include timestamp column {config.target_timestamp_column!r}."
        )

    normalized = target.copy(deep=True)
    normalized.attrs = {}
    normalized[config.target_timestamp_column] = pd.to_datetime(
        normalized[config.target_timestamp_column],
        utc=True,
        errors="coerce",
    )
    if normalized[config.target_timestamp_column].isna().any():
        raise RegimeAlignmentError(
            f"Regime alignment target contains unparsable timestamps in {config.target_timestamp_column!r}."
        )

    reserved = set(_reserved_output_columns(config=config))
    collisions = sorted(reserved.intersection(set(normalized.columns)))
    if collisions:
        raise RegimeAlignmentError(
            "Regime alignment target already contains reserved output columns: "
            f"{collisions}."
        )
    return normalized

def _regime_projection(
    labels: pd.DataFrame,
    *,
    include_metrics: bool,
    config: RegimeAlignmentConfig,
) -> pd.DataFrame:
    projected_columns = ["ts_utc", *REGIME_STATE_COLUMNS.values(), "regime_label", "is_defined"]
    if include_metrics:
        projected_columns.extend(REGIME_AUDIT_COLUMNS)
    projection = labels.loc[:, projected_columns].set_index("ts_utc", drop=True)

    renamed: dict[str, str] = {}
    for dimension in REGIME_DIMENSIONS:
        state_column = REGIME_STATE_COLUMNS[dimension]
        renamed[state_column] = _label_column(config, state_column)
    renamed["regime_label"] = _label_column(config, "label")
    renamed["is_defined"] = _label_column(config, "is_defined")
    if include_metrics:
        renamed.update({metric: _label_column(config, metric) for metric in REGIME_AUDIT_COLUMNS})
    return projection.rename(columns=renamed)


def _fill_unmatched_rows(
    aligned: pd.DataFrame,
    *,
    matched_mask: pd.Series,
    config: RegimeAlignmentConfig,
) -> None:
    unmatched = ~matched_mask
    for dimension in REGIME_DIMENSIONS:
        aligned.loc[unmatched, _label_column(config, REGIME_STATE_COLUMNS[dimension])] = UNDEFINED_REGIME_LABEL

    aligned.loc[unmatched, _label_column(config, "label")] = _undefined_regime_label()
    aligned.loc[unmatched, _label_column(config, "is_defined")] = False

    if config.include_metric_columns:
        for metric in REGIME_AUDIT_COLUMNS:
            aligned.loc[unmatched, _label_column(config, metric)] = float("nan")


def _attach_unavailable_labels(target: pd.DataFrame, *, config: RegimeAlignmentConfig) -> pd.DataFrame:
    aligned = target.copy(deep=True)
    aligned.attrs = {}

    for dimension in REGIME_DIMENSIONS:
        aligned[_label_column(config, REGIME_STATE_COLUMNS[dimension])] = UNDEFINED_REGIME_LABEL
    aligned[_label_column(config, "label")] = _undefined_regime_label()
    aligned[_label_column(config, "is_defined")] = False

    if config.include_metric_columns:
        for metric in REGIME_AUDIT_COLUMNS:
            aligned[_label_column(config, metric)] = float("nan")

    aligned[_label_column(config, "has_exact_timestamp_match")] = False
    aligned[_label_column(config, "alignment_status")] = _REGIME_ALIGNMENT_STATUS_UNAVAILABLE
    aligned[_label_column(config, "surface")] = str(config.surface)
    return aligned


def _reserved_output_columns(*, config: RegimeAlignmentConfig) -> tuple[str, ...]:
    columns: list[str] = [
        _label_column(config, REGIME_STATE_COLUMNS[dimension])
        for dimension in REGIME_DIMENSIONS
    ]
    columns.extend(
        [
            _label_column(config, "label"),
            _label_column(config, "is_defined"),
            _label_column(config, "has_exact_timestamp_match"),
            _label_column(config, "alignment_status"),
            _label_column(config, "surface"),
        ]
    )
    if config.include_metric_columns:
        columns.extend(_label_column(config, metric) for metric in REGIME_AUDIT_COLUMNS)
    return tuple(columns)


def _label_column(config: RegimeAlignmentConfig, base: str) -> str:
    if base == "label":
        return f"{config.output_prefix}label"
    if base == "is_defined":
        return f"{config.output_prefix}is_defined"
    if base == "has_exact_timestamp_match":
        return f"{config.output_prefix}has_exact_timestamp_match"
    if base == "alignment_status":
        return f"{config.output_prefix}alignment_status"
    if base == "surface":
        return f"{config.output_prefix}surface"
    return f"{config.output_prefix}{base}"


def _undefined_regime_label() -> str:
    return "|".join(f"{dimension}={UNDEFINED_REGIME_LABEL}" for dimension in REGIME_DIMENSIONS)


__all__ = [
    "RegimeAlignmentConfig",
    "RegimeAlignmentError",
    "RegimeUnavailablePolicy",
    "align_regime_labels",
    "align_regimes_to_alpha_windows",
    "align_regimes_to_portfolio_windows",
    "align_regimes_to_strategy_timeseries",
]
