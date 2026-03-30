from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal


class AlphaModelValidationError(ValueError):
    """Raised when an alpha model violates deterministic interface guarantees."""


class BaseAlphaModel(ABC):
    """
    Deterministic base contract for ML-style alpha models.

    Alpha models consume a canonical research frame, may fit on historical data,
    and must return numeric prediction scores aligned exactly to the input
    index. Input frames are expected to be sorted by ``(symbol, ts_utc)``.
    """

    name: str
    warmup_rows: int = 0

    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on a canonical historical research frame."""

        baseline = _copy_frame(df)
        validated = validate_alpha_model_input(df)
        self._fit(validated)
        _validate_input_not_mutated(baseline, df, stage="fit")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return deterministic numeric predictions aligned to ``df.index``."""

        baseline = _copy_frame(df)
        validated = validate_alpha_model_input(df)
        predictions = self._predict(validated)
        _validate_input_not_mutated(baseline, df, stage="predict")
        validated_predictions = validate_alpha_predictions(
            df=validated,
            predictions=predictions,
            warmup_rows=self.warmup_rows,
        )

        repeat_predictions = self._predict(validated)
        _validate_input_not_mutated(baseline, df, stage="predict")
        repeat_validated = validate_alpha_predictions(
            df=validated,
            predictions=repeat_predictions,
            warmup_rows=self.warmup_rows,
        )
        _validate_deterministic_predictions(validated_predictions, repeat_validated)
        return validated_predictions

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> None:
        """Subclass training hook."""

    @abstractmethod
    def _predict(self, df: pd.DataFrame) -> pd.Series:
        """Subclass prediction hook."""


class DummyAlphaModel(BaseAlphaModel):
    """Deterministic zero-signal alpha model for tests and scaffolding."""

    name = "dummy_alpha_model"

    def _fit(self, df: pd.DataFrame) -> None:
        return None

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=df.index, dtype="float64", name="prediction")


def validate_alpha_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the canonical alpha-model input contract."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Alpha model input must be a pandas DataFrame.")

    missing = [column for column in ("symbol", "ts_utc") if column not in df.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise AlphaModelValidationError(
            f"Alpha model input must include required columns: {formatted}."
        )

    if df["symbol"].isna().any():
        raise AlphaModelValidationError("Alpha model input contains null values in 'symbol'.")

    ts_utc = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    if ts_utc.isna().any():
        raise AlphaModelValidationError("Alpha model input contains unparsable 'ts_utc' values.")

    keys = pd.DataFrame(
        {
            "symbol": df["symbol"].astype("string"),
            "ts_utc": ts_utc,
        },
        index=df.index,
    )
    sorted_keys = keys.sort_values(["symbol", "ts_utc"], kind="stable")
    if not keys.index.equals(sorted_keys.index):
        first_mismatch = next(
            position
            for position, (actual_index, expected_index) in enumerate(
                zip(keys.index, sorted_keys.index, strict=False)
            )
            if actual_index != expected_index
        )
        bad_row = keys.iloc[first_mismatch]
        raise AlphaModelValidationError(
            "Alpha model input must be sorted by (symbol, ts_utc). "
            f"First out-of-order row: symbol={bad_row['symbol']}, ts_utc={bad_row['ts_utc']}."
        )

    return df


def validate_alpha_predictions(
    *,
    df: pd.DataFrame,
    predictions: pd.Series,
    warmup_rows: int = 0,
) -> pd.Series:
    """Validate prediction alignment, type, and post-warmup completeness."""

    if not isinstance(predictions, pd.Series):
        raise TypeError("Alpha model predict() must return a pandas Series.")

    if len(predictions) != len(df):
        raise AlphaModelValidationError(
            "Alpha model predictions must have the same length as the input DataFrame."
        )

    if not predictions.index.equals(df.index):
        raise AlphaModelValidationError(
            "Alpha model predictions must align exactly with the input DataFrame index."
        )

    try:
        normalized = predictions.astype("float64")
    except (TypeError, ValueError) as exc:
        raise AlphaModelValidationError(
            "Alpha model predictions must be numeric and convertible to float."
        ) from exc

    if warmup_rows < 0:
        raise AlphaModelValidationError("warmup_rows must be greater than or equal to zero.")

    if normalized.empty:
        return normalized.rename("prediction")

    valid_mask = pd.Series(True, index=df.index, dtype="bool")
    if warmup_rows > 0:
        row_number = df.groupby("symbol", sort=False).cumcount()
        valid_mask = row_number >= warmup_rows

    invalid = normalized[valid_mask & normalized.isna()]
    if not invalid.empty:
        first_bad_index = invalid.index[0]
        raise AlphaModelValidationError(
            "Alpha model predictions must not contain NaN values after warmup handling. "
            f"First invalid index: {first_bad_index!r}."
        )

    return normalized.rename(predictions.name or "prediction")


def _validate_deterministic_predictions(first: pd.Series, second: pd.Series) -> None:
    try:
        assert_series_equal(first, second, check_dtype=True, check_exact=True)
    except AssertionError as exc:
        raise AlphaModelValidationError(
            "Alpha model predict() must be deterministic for the same input."
        ) from exc


def _validate_input_not_mutated(baseline: pd.DataFrame, current: pd.DataFrame, *, stage: str) -> None:
    try:
        assert_frame_equal(baseline, current, check_dtype=True, check_exact=True)
    except AssertionError as exc:
        raise AlphaModelValidationError(
            f"Alpha model {stage}() must not mutate the input DataFrame."
        ) from exc
    if baseline.attrs != current.attrs:
        raise AlphaModelValidationError(
            f"Alpha model {stage}() must not mutate the input DataFrame attrs."
        )


def _copy_frame(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df.copy(deep=True)
    baseline.attrs = dict(df.attrs)
    return baseline
