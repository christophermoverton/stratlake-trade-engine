from __future__ import annotations

import pandas as pd

from src.research.alpha.cross_section import AlphaCrossSectionError, iter_cross_sections

STRUCTURAL_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc", "timeframe")


class AlphaEvaluationError(ValueError):
    """Raised when alpha evaluation input is invalid or ambiguous."""


def validate_alpha_evaluation_input(
    df: pd.DataFrame,
    *,
    prediction_column: str = "prediction_score",
    forward_return_column: str = "forward_return",
    min_cross_section_size: int = 2,
) -> pd.DataFrame:
    """Validate the aligned alpha-evaluation contract without mutating caller input."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Alpha evaluation input must be a pandas DataFrame.")
    if df.empty:
        raise AlphaEvaluationError("Alpha evaluation input must not be empty.")
    if not isinstance(min_cross_section_size, int) or min_cross_section_size < 2:
        raise AlphaEvaluationError("min_cross_section_size must be an integer greater than or equal to 2.")

    missing = [
        column
        for column in (*STRUCTURAL_COLUMNS, prediction_column, forward_return_column)
        if column not in df.columns
    ]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise AlphaEvaluationError(
            f"Alpha evaluation input must include required columns: {formatted}."
        )

    normalized = df.copy(deep=True)
    normalized.attrs = {}
    normalized["symbol"] = normalized["symbol"].astype("string")
    normalized["timeframe"] = normalized["timeframe"].astype("string")

    _validate_non_null_key_columns(normalized)

    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    if normalized["ts_utc"].isna().any():
        raise AlphaEvaluationError("Alpha evaluation input contains unparsable 'ts_utc' values.")

    exact_duplicate_mask = normalized.duplicated(keep=False)
    if exact_duplicate_mask.any():
        first_duplicate = normalized.loc[exact_duplicate_mask].iloc[0]
        raise AlphaEvaluationError(
            "Alpha evaluation input must not contain duplicate rows. "
            f"First duplicate row key: symbol={first_duplicate['symbol']}, "
            f"ts_utc={first_duplicate['ts_utc']}, timeframe={first_duplicate['timeframe']}."
        )

    duplicate_mask = normalized.duplicated(subset=list(STRUCTURAL_COLUMNS), keep=False)
    if duplicate_mask.any():
        first_duplicate = normalized.loc[duplicate_mask, list(STRUCTURAL_COLUMNS)].iloc[0]
        raise AlphaEvaluationError(
            "Alpha evaluation input must not contain duplicate "
            "(symbol, ts_utc, timeframe) rows. "
            f"First duplicate key: symbol={first_duplicate['symbol']}, "
            f"ts_utc={first_duplicate['ts_utc']}, timeframe={first_duplicate['timeframe']}."
        )

    unique_timeframes = normalized["timeframe"].drop_duplicates().tolist()
    if len(unique_timeframes) != 1:
        raise AlphaEvaluationError(
            "Alpha evaluation input must contain exactly one timeframe so cross-sectional "
            "evaluation matches the current alpha prediction contract."
        )

    _validate_sorted_order(normalized)

    normalized[prediction_column] = _coerce_numeric_column(
        normalized[prediction_column],
        column_name=prediction_column,
    )
    normalized[forward_return_column] = _coerce_numeric_column(
        normalized[forward_return_column],
        column_name=forward_return_column,
    )

    _validate_cross_section_contract(
        normalized,
        prediction_column=prediction_column,
        forward_return_column=forward_return_column,
        min_cross_section_size=min_cross_section_size,
    )
    return normalized


def _validate_non_null_key_columns(df: pd.DataFrame) -> None:
    if df["symbol"].isna().any():
        raise AlphaEvaluationError("Alpha evaluation input contains null values in 'symbol'.")
    if df["timeframe"].isna().any():
        raise AlphaEvaluationError("Alpha evaluation input contains null values in 'timeframe'.")


def _validate_sorted_order(df: pd.DataFrame) -> None:
    keys = df.loc[:, list(STRUCTURAL_COLUMNS)]
    sorted_keys = keys.sort_values(list(STRUCTURAL_COLUMNS), kind="stable")
    if keys.index.equals(sorted_keys.index):
        return

    first_mismatch = next(
        position
        for position, (actual_index, expected_index) in enumerate(
            zip(keys.index, sorted_keys.index, strict=False)
        )
        if actual_index != expected_index
    )
    bad_row = keys.iloc[first_mismatch]
    raise AlphaEvaluationError(
        "Alpha evaluation input must be sorted by (symbol, ts_utc, timeframe) before IC evaluation. "
        f"First out-of-order row: symbol={bad_row['symbol']}, ts_utc={bad_row['ts_utc']}, "
        f"timeframe={bad_row['timeframe']}."
    )


def _validate_cross_section_contract(
    df: pd.DataFrame,
    *,
    prediction_column: str,
    forward_return_column: str,
    min_cross_section_size: int,
) -> None:
    cross_section_view = df.loc[:, ["symbol", "ts_utc", prediction_column, forward_return_column]].copy(deep=True)
    cross_section_view.attrs = {}

    try:
        cross_sections = iter_cross_sections(
            cross_section_view,
            columns=[prediction_column, forward_return_column],
        )
        for ts_utc, group in cross_sections:
            evaluation_slice = group.loc[:, [prediction_column, forward_return_column]].dropna(
                subset=[prediction_column, forward_return_column]
            )
            if evaluation_slice.empty:
                raise AlphaEvaluationError(
                    "Alpha evaluation input contains a timestamp with no usable non-null prediction/forward-return pairs. "
                    f"First failing ts_utc={ts_utc}."
                )
            sample_size = int(len(evaluation_slice))
            if sample_size < min_cross_section_size:
                raise AlphaEvaluationError(
                    "Alpha evaluation input contains an insufficient cross-section for IC evaluation. "
                    f"First failing ts_utc={ts_utc}, usable_rows={sample_size}, "
                    f"required_rows={min_cross_section_size}."
                )
            if evaluation_slice[prediction_column].nunique(dropna=True) <= 1:
                raise AlphaEvaluationError(
                    "Alpha evaluation input requires cross-sectional variation in the prediction column. "
                    f"First failing ts_utc={ts_utc}, column={prediction_column!r}."
                )
            if evaluation_slice[forward_return_column].nunique(dropna=True) <= 1:
                raise AlphaEvaluationError(
                    "Alpha evaluation input requires cross-sectional variation in the forward return column. "
                    f"First failing ts_utc={ts_utc}, column={forward_return_column!r}."
                )
    except AlphaCrossSectionError as exc:
        raise AlphaEvaluationError(str(exc)) from exc


def _coerce_numeric_column(values: pd.Series, *, column_name: str) -> pd.Series:
    try:
        return pd.to_numeric(values, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise AlphaEvaluationError(
            f"Alpha evaluation column '{column_name}' must be numeric."
        ) from exc
