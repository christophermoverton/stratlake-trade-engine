from __future__ import annotations

import math

import pandas as pd

from src.research.regimes.taxonomy import (
    REGIME_DIMENSIONS,
    REGIME_METRIC_COLUMNS,
    REGIME_OUTPUT_COLUMNS,
    REGIME_STATE_COLUMNS,
    REGIME_TAXONOMY,
    UNDEFINED_REGIME_LABEL,
)


class RegimeValidationError(ValueError):
    """Raised when regime labels violate the deterministic output contract."""


def validate_regime_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a deterministic regime-label frame.

    The contract is timestamp-level, UTC aligned, sorted by ``ts_utc``, and one
    row per timestamp. Missing metrics are allowed only when the corresponding
    dimension is explicitly labelled ``undefined``.
    """

    if not isinstance(labels, pd.DataFrame):
        raise TypeError("Regime labels must be provided as a pandas DataFrame.")
    if labels.empty:
        raise RegimeValidationError("Regime labels must not be empty.")

    missing = [column for column in REGIME_OUTPUT_COLUMNS if column not in labels.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise RegimeValidationError(f"Regime labels are missing required columns: {formatted}.")

    normalized = labels.copy(deep=True)
    normalized.attrs = {}
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    if normalized["ts_utc"].isna().any():
        raise RegimeValidationError("Regime labels contain unparsable 'ts_utc' values.")
    if normalized["ts_utc"].duplicated().any():
        duplicate_ts = normalized.loc[normalized["ts_utc"].duplicated(keep=False), "ts_utc"].iloc[0]
        raise RegimeValidationError(
            f"Regime labels contain duplicate timestamps. First duplicate ts_utc={duplicate_ts}."
        )

    expected_order = normalized.sort_values("ts_utc", kind="stable")
    if not normalized.index.equals(expected_order.index):
        first_mismatch = next(
            position
            for position, (actual_index, expected_index) in enumerate(
                zip(normalized.index, expected_order.index, strict=False)
            )
            if actual_index != expected_index
        )
        bad_ts = normalized.iloc[first_mismatch]["ts_utc"]
        raise RegimeValidationError(
            "Regime labels must be sorted by ts_utc. "
            f"First out-of-order timestamp={bad_ts}."
        )

    for dimension in REGIME_DIMENSIONS:
        state_column = REGIME_STATE_COLUMNS[dimension]
        allowed_labels = set(REGIME_TAXONOMY[dimension].labels)
        normalized[state_column] = normalized[state_column].astype("string")
        invalid = ~normalized[state_column].isin(allowed_labels)
        if invalid.any():
            bad_label = normalized.loc[invalid, state_column].iloc[0]
            raise RegimeValidationError(
                f"Regime dimension {dimension!r} contains unsupported label {bad_label!r}."
            )

    for column in _all_metric_columns():
        normalized[column] = _normalize_optional_float_series(normalized[column], column_name=column)

    _validate_metric_label_alignment(normalized)
    _validate_composite_labels(normalized)
    _validate_defined_flag(normalized)

    normalized = normalized.loc[:, list(REGIME_OUTPUT_COLUMNS)]
    normalized.attrs["regime_contract"] = {
        "contract": "regime_labels",
        "row_count": int(len(normalized)),
        "taxonomy_version": "regime_taxonomy_v1",
    }
    return normalized


def _all_metric_columns() -> tuple[str, ...]:
    metric_columns: list[str] = []
    for dimension in REGIME_DIMENSIONS:
        metric_columns.extend(REGIME_METRIC_COLUMNS[dimension])
    return tuple(metric_columns)


def _normalize_optional_float_series(series: pd.Series, *, column_name: str) -> pd.Series:
    try:
        normalized = pd.to_numeric(series, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise RegimeValidationError(
            f"Regime metric column {column_name!r} must contain float-compatible values."
        ) from exc
    finite_mask = normalized.dropna().map(math.isfinite)
    if not finite_mask.all():
        raise RegimeValidationError(
            f"Regime metric column {column_name!r} must contain only finite values or NaN."
        )
    return normalized


def _validate_metric_label_alignment(labels: pd.DataFrame) -> None:
    for dimension in (REGIME_DIMENSIONS[0], REGIME_DIMENSIONS[1], REGIME_DIMENSIONS[2]):
        state_column = REGIME_STATE_COLUMNS[dimension]
        metric_column = REGIME_METRIC_COLUMNS[dimension][0]
        bad_defined = labels[metric_column].isna() & labels[state_column].ne(UNDEFINED_REGIME_LABEL)
        if bad_defined.any():
            row = labels.loc[bad_defined].iloc[0]
            raise RegimeValidationError(
                f"Regime dimension {dimension!r} must use label 'undefined' when "
                f"{metric_column!r} is missing. First failing ts_utc={row['ts_utc']}."
            )

    stress_state_column = REGIME_STATE_COLUMNS["stress"]
    stress_metrics = list(REGIME_METRIC_COLUMNS["stress"])
    stress_missing = labels.loc[:, stress_metrics].isna().all(axis=1)
    bad_stress = stress_missing & labels[stress_state_column].ne(UNDEFINED_REGIME_LABEL)
    if bad_stress.any():
        row = labels.loc[bad_stress].iloc[0]
        raise RegimeValidationError(
            "Regime dimension 'stress' must use label 'undefined' when stress metrics are missing. "
            f"First failing ts_utc={row['ts_utc']}."
        )


def _validate_composite_labels(labels: pd.DataFrame) -> None:
    expected = labels.apply(_composite_label_for_row, axis=1)
    mismatch = labels["regime_label"].astype("string").ne(expected.astype("string"))
    if mismatch.any():
        row = labels.loc[mismatch].iloc[0]
        raise RegimeValidationError(
            "Regime composite labels must match canonical dimension ordering. "
            f"First failing ts_utc={row['ts_utc']}."
        )


def _composite_label_for_row(row: pd.Series) -> str:
    return "|".join(
        f"{dimension}={row[REGIME_STATE_COLUMNS[dimension]]}"
        for dimension in REGIME_DIMENSIONS
    )


def _validate_defined_flag(labels: pd.DataFrame) -> None:
    expected = pd.Series(True, index=labels.index)
    for dimension in REGIME_DIMENSIONS:
        expected &= labels[REGIME_STATE_COLUMNS[dimension]].ne(UNDEFINED_REGIME_LABEL)
    actual = labels["is_defined"].astype("bool")
    if not actual.equals(expected.astype("bool")):
        bad_index = actual.ne(expected).idxmax()
        raise RegimeValidationError(
            "Regime labels column 'is_defined' must be true only when all dimensions are defined. "
            f"First failing ts_utc={labels.loc[bad_index, 'ts_utc']}."
        )


__all__ = ["RegimeValidationError", "validate_regime_labels"]
