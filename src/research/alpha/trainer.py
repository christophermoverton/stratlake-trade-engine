from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.feature_names import resolve_feature_names
from src.research.alpha.base import AlphaModelValidationError, BaseAlphaModel, validate_alpha_model_input
from src.research.alpha.registry import get_alpha_model

STRUCTURAL_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc")


class AlphaTrainingError(ValueError):
    """Raised when an alpha model cannot be trained safely or deterministically."""


@dataclass(frozen=True)
class TrainedAlphaModel:
    """Structured result returned from deterministic alpha-model training."""

    model_name: str
    model: BaseAlphaModel
    target_column: str
    feature_columns: list[str]
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    row_count: int
    symbol_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def train_alpha_model(
    df: pd.DataFrame,
    model_name: str,
    target_column: str,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    train_start: str | pd.Timestamp | None = None,
    train_end: str | pd.Timestamp | None = None,
) -> TrainedAlphaModel:
    """Fit one registered alpha model on a deterministic half-open time slice `[start, end)`."""

    normalized_df = _validate_training_frame(df, target_column=target_column)
    normalized_start, normalized_end = _normalize_train_bounds(train_start=train_start, train_end=train_end)
    selected_features = _resolve_feature_columns(
        normalized_df,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    training_frame = _select_training_slice(
        normalized_df,
        target_column=target_column,
        feature_columns=selected_features,
        train_start=normalized_start,
        train_end=normalized_end,
    )

    try:
        model = get_alpha_model(model_name)
    except ValueError as exc:
        raise AlphaTrainingError(str(exc)) from exc

    model.fit(training_frame)
    return TrainedAlphaModel(
        model_name=model_name.strip(),
        model=model,
        target_column=target_column,
        feature_columns=selected_features,
        train_start=normalized_start,
        train_end=normalized_end,
        row_count=len(training_frame),
        symbol_count=int(training_frame["symbol"].astype("string").nunique()),
        metadata={
            "fit_columns": [*STRUCTURAL_COLUMNS, target_column, *selected_features],
            "window_semantics": "[train_start, train_end)",
        },
    )


def _validate_training_frame(df: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Alpha training input must be a pandas DataFrame.")
    if df.empty:
        raise AlphaTrainingError("Alpha training input must not be empty.")
    if not isinstance(target_column, str) or not target_column.strip():
        raise AlphaTrainingError("target_column must be a non-empty string.")
    if target_column not in df.columns:
        if target_column.startswith("target_"):
            raise AlphaTrainingError(
                "Alpha training input is missing required target column "
                f"'{target_column}'. Rebuild or reload the feature dataset with canonical alpha targets."
            )
        raise AlphaTrainingError(f"Alpha training input must include target column '{target_column}'.")

    try:
        validated = validate_alpha_model_input(df)
    except AlphaModelValidationError as exc:
        raise AlphaTrainingError(str(exc)) from exc

    if df[target_column].isna().all():
        raise AlphaTrainingError(
            f"Alpha training input target column '{target_column}' must contain at least one non-null value."
        )

    return validated.copy(deep=True)


def _normalize_train_bounds(
    *,
    train_start: str | pd.Timestamp | None,
    train_end: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    normalized_start = _coerce_timestamp(train_start, field_name="train_start")
    normalized_end = _coerce_timestamp(train_end, field_name="train_end")

    if normalized_start is not None and normalized_end is not None and normalized_start >= normalized_end:
        raise AlphaTrainingError("train_start must be earlier than train_end.")

    return normalized_start, normalized_end


def _coerce_timestamp(value: str | pd.Timestamp | None, *, field_name: str) -> pd.Timestamp | None:
    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise AlphaTrainingError(f"{field_name} must be a valid timestamp or timestamp-like string.") from exc

    if pd.isna(timestamp):
        raise AlphaTrainingError(f"{field_name} must be a valid timestamp or timestamp-like string.")

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _resolve_feature_columns(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if feature_columns is None:
        derived = sorted(
            column
            for column in df.columns
            if column.startswith("feature_") and column not in STRUCTURAL_COLUMNS and column != target_column
        )
        if not derived:
            raise AlphaTrainingError(
                "Could not derive any training feature columns. Pass feature_columns explicitly or add feature_* columns."
            )
        return derived

    if not isinstance(feature_columns, (list, tuple)):
        raise TypeError("feature_columns must be a list, tuple, or None.")
    if not feature_columns:
        raise AlphaTrainingError("feature_columns must contain at least one column when provided explicitly.")

    normalized: list[str] = []
    seen: set[str] = set()
    resolved_columns = resolve_feature_names(feature_columns, df.columns)
    for requested_column, column in zip(feature_columns, resolved_columns, strict=False):
        if not isinstance(column, str) or not column.strip():
            raise AlphaTrainingError("feature_columns entries must be non-empty strings.")
        if column in seen:
            raise AlphaTrainingError(
                "feature_columns must not contain duplicates after alias resolution. "
                f"Found duplicate: '{column}' from '{requested_column}'."
            )
        if column not in df.columns:
            raise AlphaTrainingError(
                f"Alpha training input must include feature column '{requested_column}' "
                f"(resolved to '{column}')."
            )
        if column in STRUCTURAL_COLUMNS:
            raise AlphaTrainingError(f"feature_columns must not include structural column '{column}'.")
        if column == target_column:
            raise AlphaTrainingError("feature_columns must not include the target column.")
        normalized.append(column)
        seen.add(column)

    return normalized


def _select_training_slice(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp | None,
) -> pd.DataFrame:
    ts_utc = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    mask = pd.Series(True, index=df.index, dtype="bool")
    if train_start is not None:
        mask &= ts_utc >= train_start
    if train_end is not None:
        mask &= ts_utc < train_end

    selected_columns = [*STRUCTURAL_COLUMNS, target_column, *feature_columns]
    training_frame = df.loc[mask, selected_columns].copy(deep=True)
    if training_frame.empty:
        raise AlphaTrainingError("Alpha training window produced no rows after applying train_start/train_end filters.")

    training_frame["ts_utc"] = pd.to_datetime(training_frame["ts_utc"], utc=True, errors="coerce")
    training_frame = training_frame.sort_values(["symbol", "ts_utc"], kind="stable").copy(deep=True)
    training_frame.attrs = {}
    return training_frame
