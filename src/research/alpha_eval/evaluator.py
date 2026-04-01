from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.research.alpha.cross_section import AlphaCrossSectionError, iter_cross_sections

STRUCTURAL_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc", "timeframe")


class AlphaEvaluationError(ValueError):
    """Raised when alpha evaluation input is invalid or ambiguous."""


@dataclass(frozen=True)
class AlphaEvaluationResult:
    """Structured deterministic alpha-evaluation output for the research alpha workflow."""

    prediction_column: str
    forward_return_column: str
    min_cross_section_size: int
    row_count: int
    timestamp_count: int
    symbol_count: int
    ic_timeseries: pd.DataFrame
    summary: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_alpha_predictions(
    df: pd.DataFrame,
    *,
    prediction_column: str = "prediction_score",
    forward_return_column: str = "forward_return",
    min_cross_section_size: int = 2,
) -> AlphaEvaluationResult:
    """Evaluate aligned alpha predictions with cross-sectional IC diagnostics."""

    validated = validate_alpha_evaluation_input(
        df,
        prediction_column=prediction_column,
        forward_return_column=forward_return_column,
        min_cross_section_size=min_cross_section_size,
    )

    cross_section_frame = validated.sort_values(["symbol", "ts_utc"], kind="stable").copy(deep=True)
    cross_section_frame.attrs = {}

    rows: list[dict[str, Any]] = []
    for ts_utc, group in iter_cross_sections(
        cross_section_frame,
        columns=[prediction_column, forward_return_column],
    ):
        evaluation_slice = group.loc[:, [prediction_column, forward_return_column]].dropna(
            subset=[prediction_column, forward_return_column]
        )
        sample_size = int(len(evaluation_slice))

        rows.append(
            {
                "ts_utc": ts_utc,
                "ic": _correlation_or_nan(
                    evaluation_slice[prediction_column],
                    evaluation_slice[forward_return_column],
                    min_cross_section_size=min_cross_section_size,
                    method="pearson",
                ),
                "rank_ic": _correlation_or_nan(
                    evaluation_slice[prediction_column],
                    evaluation_slice[forward_return_column],
                    min_cross_section_size=min_cross_section_size,
                    method="spearman",
                ),
                "sample_size": sample_size,
            }
        )

    ic_timeseries = pd.DataFrame(rows, columns=["ts_utc", "ic", "rank_ic", "sample_size"])
    if not ic_timeseries.empty:
        ic_timeseries["ts_utc"] = pd.to_datetime(ic_timeseries["ts_utc"], utc=True, errors="coerce")
        ic_timeseries["ic"] = ic_timeseries["ic"].astype("float64")
        ic_timeseries["rank_ic"] = ic_timeseries["rank_ic"].astype("float64")
        ic_timeseries["sample_size"] = ic_timeseries["sample_size"].astype("int64")
    ic_timeseries.attrs = {}

    valid_mask = ic_timeseries["ic"].notna() & ic_timeseries["rank_ic"].notna()
    summary = {
        "mean_ic": float(ic_timeseries["ic"].mean()) if not ic_timeseries.empty else float("nan"),
        "mean_rank_ic": float(ic_timeseries["rank_ic"].mean()) if not ic_timeseries.empty else float("nan"),
        "valid_timestamps": float(valid_mask.sum()),
    }

    timeframe = str(validated["timeframe"].iloc[0])
    metadata = {
        "input_columns": list(validated.columns),
        "ic_timeseries_columns": list(ic_timeseries.columns),
        "artifact_scaffold": {
            "ic_timeseries": "ic_timeseries.csv",
            "alpha_metrics": "alpha_metrics.json",
        },
        "timeframe": timeframe,
        "ts_utc_start": validated["ts_utc"].min(),
        "ts_utc_end": validated["ts_utc"].max(),
    }

    return AlphaEvaluationResult(
        prediction_column=prediction_column,
        forward_return_column=forward_return_column,
        min_cross_section_size=min_cross_section_size,
        row_count=int(len(validated)),
        timestamp_count=int(validated["ts_utc"].nunique()),
        symbol_count=int(validated["symbol"].nunique()),
        ic_timeseries=ic_timeseries,
        summary=summary,
        metadata=metadata,
    )


def evaluate_information_coefficient(
    df: pd.DataFrame,
    *,
    prediction_column: str = "prediction_score",
    forward_return_column: str = "forward_return",
    min_cross_section_size: int = 2,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return the IC timeseries and summary scaffold for aligned alpha predictions."""

    result = evaluate_alpha_predictions(
        df,
        prediction_column=prediction_column,
        forward_return_column=forward_return_column,
        min_cross_section_size=min_cross_section_size,
    )
    return result.ic_timeseries, result.summary


def validate_alpha_evaluation_input(
    df: pd.DataFrame,
    *,
    prediction_column: str = "prediction_score",
    forward_return_column: str = "forward_return",
    min_cross_section_size: int = 2,
) -> pd.DataFrame:
    """Validate and normalize alpha prediction output joined with aligned forward returns."""

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

    if normalized["symbol"].isna().any():
        raise AlphaEvaluationError("Alpha evaluation input contains null values in 'symbol'.")
    if normalized["timeframe"].isna().any():
        raise AlphaEvaluationError("Alpha evaluation input contains null values in 'timeframe'.")

    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    if normalized["ts_utc"].isna().any():
        raise AlphaEvaluationError("Alpha evaluation input contains unparsable 'ts_utc' values.")

    keys = normalized.loc[:, list(STRUCTURAL_COLUMNS)]
    duplicate_mask = keys.duplicated(subset=list(STRUCTURAL_COLUMNS), keep=False)
    if duplicate_mask.any():
        first_duplicate = keys.loc[duplicate_mask, list(STRUCTURAL_COLUMNS)].iloc[0]
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

    normalized[prediction_column] = _coerce_numeric_column(
        normalized[prediction_column],
        column_name=prediction_column,
    )
    normalized[forward_return_column] = _coerce_numeric_column(
        normalized[forward_return_column],
        column_name=forward_return_column,
    )

    normalized = normalized.sort_values(["ts_utc", "symbol"], kind="stable").copy(deep=True)
    normalized.attrs = {}

    _validate_cross_section_contract(
        normalized,
        prediction_column=prediction_column,
        forward_return_column=forward_return_column,
    )
    return normalized


def _validate_cross_section_contract(
    df: pd.DataFrame,
    *,
    prediction_column: str,
    forward_return_column: str,
) -> None:
    cross_section_view = df.loc[:, ["symbol", "ts_utc", prediction_column, forward_return_column]].sort_values(
        ["symbol", "ts_utc"],
        kind="stable",
    )
    cross_section_view.attrs = {}

    try:
        list(
            iter_cross_sections(
                cross_section_view,
                columns=[prediction_column, forward_return_column],
            )
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


def _correlation_or_nan(
    left: pd.Series,
    right: pd.Series,
    *,
    min_cross_section_size: int,
    method: str,
) -> float:
    if len(left) < min_cross_section_size:
        return float("nan")
    if left.nunique(dropna=False) <= 1 or right.nunique(dropna=False) <= 1:
        return float("nan")

    if method == "spearman":
        left = left.rank(method="average")
        right = right.rank(method="average")
        correlation = left.corr(right, method="pearson")
    else:
        correlation = left.corr(right, method=method)

    if pd.isna(correlation):
        return float("nan")
    return float(correlation)
