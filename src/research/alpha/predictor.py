from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.feature_names import resolve_feature_names
from src.research.alpha.base import AlphaModelValidationError, validate_alpha_model_input
from src.research.alpha.trainer import STRUCTURAL_COLUMNS, TrainedAlphaModel

OPTIONAL_STRUCTURAL_COLUMNS: tuple[str, ...] = ("timeframe",)


class AlphaPredictionError(ValueError):
    """Raised when alpha-model prediction cannot be performed safely."""


@dataclass(frozen=True)
class AlphaPredictionResult:
    """Structured deterministic alpha-model prediction output."""

    model_name: str
    trained_model: TrainedAlphaModel
    target_column: str
    feature_columns: list[str]
    predict_start: pd.Timestamp | None
    predict_end: pd.Timestamp | None
    row_count: int
    symbol_count: int
    predictions: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def predict_alpha_model(
    trained_model: TrainedAlphaModel,
    df: pd.DataFrame,
    predict_start: str | pd.Timestamp | None = None,
    predict_end: str | pd.Timestamp | None = None,
) -> AlphaPredictionResult:
    """Generate deterministic half-open predictions on `[predict_start, predict_end)`."""

    _validate_trained_model(trained_model)
    prediction_input = _validate_prediction_frame(df, feature_columns=trained_model.feature_columns)
    normalized_start, normalized_end = _normalize_predict_bounds(
        predict_start=predict_start,
        predict_end=predict_end,
    )
    prediction_frame = _select_prediction_slice(
        prediction_input,
        feature_columns=trained_model.feature_columns,
        predict_start=normalized_start,
        predict_end=normalized_end,
    )

    try:
        predictions = trained_model.model.predict(prediction_frame)
    except AlphaModelValidationError as exc:
        raise AlphaPredictionError(str(exc)) from exc

    validated_predictions = _validate_prediction_series(prediction_frame, predictions)
    prediction_output = _build_prediction_output(prediction_frame, validated_predictions)

    return AlphaPredictionResult(
        model_name=trained_model.model_name,
        trained_model=trained_model,
        target_column=trained_model.target_column,
        feature_columns=list(trained_model.feature_columns),
        predict_start=normalized_start,
        predict_end=normalized_end,
        row_count=len(prediction_output),
        symbol_count=int(prediction_output["symbol"].astype("string").nunique()),
        predictions=prediction_output,
        metadata={
            "predict_columns": list(prediction_frame.columns),
            "prediction_output_columns": list(prediction_output.columns),
            "window_semantics": "[predict_start, predict_end)",
        },
    )


def _validate_trained_model(trained_model: TrainedAlphaModel) -> None:
    if not isinstance(trained_model, TrainedAlphaModel):
        raise TypeError("trained_model must be a TrainedAlphaModel instance.")
    if not trained_model.feature_columns:
        raise AlphaPredictionError("trained_model must include at least one feature column.")


def _validate_prediction_frame(df: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Alpha prediction input must be a pandas DataFrame.")
    if df.empty:
        raise AlphaPredictionError("Alpha prediction input must not be empty.")

    resolved_feature_columns = resolve_feature_names(feature_columns, df.columns)
    missing_features = [
        requested
        for requested, resolved in zip(feature_columns, resolved_feature_columns, strict=False)
        if resolved not in df.columns
    ]
    if missing_features:
        formatted = ", ".join(repr(column) for column in missing_features)
        raise AlphaPredictionError(
            f"Alpha prediction input must include trained feature columns: {formatted}."
        )

    try:
        validated = validate_alpha_model_input(df)
    except AlphaModelValidationError as exc:
        raise AlphaPredictionError(str(exc)) from exc

    prediction_columns = [*STRUCTURAL_COLUMNS, *resolved_feature_columns]
    if any(column == "timeframe" for column in df.columns):
        prediction_columns.append("timeframe")
    narrowed = validated.loc[:, list(dict.fromkeys(prediction_columns))].copy(deep=True)
    narrowed["ts_utc"] = pd.to_datetime(narrowed["ts_utc"], utc=True, errors="coerce")
    narrowed.attrs = {}
    return narrowed


def _normalize_predict_bounds(
    *,
    predict_start: str | pd.Timestamp | None,
    predict_end: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    normalized_start = _coerce_timestamp(predict_start, field_name="predict_start")
    normalized_end = _coerce_timestamp(predict_end, field_name="predict_end")

    if normalized_start is not None and normalized_end is not None and normalized_start >= normalized_end:
        raise AlphaPredictionError("predict_start must be earlier than predict_end.")

    return normalized_start, normalized_end


def _coerce_timestamp(value: str | pd.Timestamp | None, *, field_name: str) -> pd.Timestamp | None:
    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise AlphaPredictionError(
            f"{field_name} must be a valid timestamp or timestamp-like string."
        ) from exc

    if pd.isna(timestamp):
        raise AlphaPredictionError(f"{field_name} must be a valid timestamp or timestamp-like string.")

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _select_prediction_slice(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    predict_start: pd.Timestamp | None,
    predict_end: pd.Timestamp | None,
) -> pd.DataFrame:
    ts_utc = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    mask = pd.Series(True, index=df.index, dtype="bool")
    if predict_start is not None:
        mask &= ts_utc >= predict_start
    if predict_end is not None:
        mask &= ts_utc < predict_end

    selected_columns = [*STRUCTURAL_COLUMNS, *feature_columns]
    preserved_optional = [column for column in OPTIONAL_STRUCTURAL_COLUMNS if column in df.columns]
    prediction_frame = df.loc[mask, [*selected_columns, *preserved_optional]].copy(deep=True)
    if prediction_frame.empty:
        raise AlphaPredictionError(
            "Alpha prediction window produced no rows after applying predict_start/predict_end filters."
        )

    prediction_frame["ts_utc"] = pd.to_datetime(prediction_frame["ts_utc"], utc=True, errors="coerce")
    prediction_frame = prediction_frame.sort_values(["symbol", "ts_utc"], kind="stable").copy(deep=True)
    prediction_frame.attrs = {}
    return prediction_frame


def _validate_prediction_series(df: pd.DataFrame, predictions: pd.Series) -> pd.Series:
    if not isinstance(predictions, pd.Series):
        raise AlphaPredictionError("Alpha model predict() must return a pandas Series.")
    if len(predictions) != len(df):
        raise AlphaPredictionError(
            "Alpha model predictions must have the same length as the prediction input DataFrame."
        )
    if not predictions.index.equals(df.index):
        raise AlphaPredictionError(
            "Alpha model predictions must align exactly with the prediction input DataFrame index."
        )

    try:
        normalized = predictions.astype("float64")
    except (TypeError, ValueError) as exc:
        raise AlphaPredictionError(
            "Alpha model predictions must be numeric and convertible to float."
        ) from exc

    if normalized.isna().any():
        first_bad_index = normalized[normalized.isna()].index[0]
        raise AlphaPredictionError(
            "Alpha model predictions must not contain NaN values. "
            f"First invalid index: {first_bad_index!r}."
        )

    return normalized.rename("prediction_score")


def _build_prediction_output(df: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    output_columns = [*STRUCTURAL_COLUMNS]
    output_columns.extend(column for column in OPTIONAL_STRUCTURAL_COLUMNS if column in df.columns)
    output = df.loc[:, output_columns].copy(deep=True)
    output["prediction_score"] = predictions
    output.attrs = {}
    return output
