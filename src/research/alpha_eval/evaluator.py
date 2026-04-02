from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.research.alpha.cross_section import iter_cross_sections
from src.research.alpha_eval.validation import AlphaEvaluationError, validate_alpha_evaluation_input


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
    summary: dict[str, float | int]
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

    cross_section_frame = validated.copy(deep=True)
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

    summary = _summarize_ic_timeseries(ic_timeseries)

    timeframe = str(validated["timeframe"].iloc[0])
    metadata = {
        "input_columns": list(validated.columns),
        "ic_timeseries_columns": list(ic_timeseries.columns),
        "artifact_scaffold": {
            "coefficients": "coefficients.json",
            "ic_timeseries": "ic_timeseries.csv",
            "alpha_metrics": "alpha_metrics.json",
            "cross_section_diagnostics": "cross_section_diagnostics.json",
            "predictions": "predictions.parquet",
            "training_summary": "training_summary.json",
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
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return the IC timeseries and summary scaffold for aligned alpha predictions."""

    result = evaluate_alpha_predictions(
        df,
        prediction_column=prediction_column,
        forward_return_column=forward_return_column,
        min_cross_section_size=min_cross_section_size,
    )
    return result.ic_timeseries, result.summary
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


def _summarize_ic_timeseries(ic_timeseries: pd.DataFrame) -> dict[str, float | int]:
    if ic_timeseries.empty:
        valid_ic = pd.Series(dtype="float64")
        valid_rank_ic = pd.Series(dtype="float64")
    else:
        valid_ic = ic_timeseries.loc[ic_timeseries["ic"].notna(), "ic"].astype("float64")
        valid_rank_ic = ic_timeseries.loc[ic_timeseries["rank_ic"].notna(), "rank_ic"].astype("float64")

    n_periods = int(valid_ic.shape[0])
    mean_ic = float(valid_ic.mean()) if n_periods > 0 else float("nan")
    std_ic = _sample_std_or_nan(valid_ic)
    ic_ir = _information_ratio(mean_ic, std_ic, n_periods=n_periods)

    n_rank_periods = int(valid_rank_ic.shape[0])
    mean_rank_ic = float(valid_rank_ic.mean()) if n_rank_periods > 0 else float("nan")
    std_rank_ic = _sample_std_or_nan(valid_rank_ic)
    rank_ic_ir = _information_ratio(mean_rank_ic, std_rank_ic, n_periods=n_rank_periods)

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir": ic_ir,
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "rank_ic_ir": rank_ic_ir,
        "n_periods": n_periods,
        "ic_positive_rate": _positive_rate(valid_ic),
        "valid_timestamps": float(n_periods),
    }


def _sample_std_or_nan(values: pd.Series) -> float:
    if int(values.shape[0]) < 2:
        return float("nan")
    return float(values.std(ddof=1))


def _information_ratio(mean_value: float, std_value: float, *, n_periods: int) -> float:
    if n_periods < 2 or pd.isna(mean_value) or pd.isna(std_value):
        return float("nan")
    if std_value == 0.0:
        return 0.0
    return float(mean_value / std_value)


def _positive_rate(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float((values > 0.0).mean())
