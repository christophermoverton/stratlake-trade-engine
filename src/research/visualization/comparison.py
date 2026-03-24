"""Plotting interfaces for comparing multiple strategy runs and outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from src.research.visualization.equity import normalize_equity_input
from src.research.visualization.plot_utils import (
    DEFAULT_BAR_ALPHA,
    DEFAULT_FIGSIZE,
    DEFAULT_LINEWIDTH,
    REFERENCE_LINEWIDTH,
    apply_axis_style,
    create_figure,
    finalize_figure,
    save_or_return_figure,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

PlotInput = pd.Series | pd.DataFrame
PlotResult = Path | Figure
InputType = Literal["returns", "equity"]
SeriesInput = Mapping[str, PlotInput] | Sequence[tuple[str, PlotInput]]
MetricInput = pd.DataFrame | Sequence[Mapping[str, object]]

_TIME_INDEX_TYPES = (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)
_DEFAULT_LABEL_COLUMNS = ("strategy", "strategy_name", "run_id")


def normalize_strategy_series_inputs(
    strategy_data: SeriesInput,
    *,
    input_type: InputType,
) -> list[tuple[str, pd.Series]]:
    """Normalize named strategy inputs into numeric time-indexed equity series."""

    items = _coerce_named_series_inputs(strategy_data)
    normalized: list[tuple[str, pd.Series]] = []

    for name, data in items:
        series = normalize_equity_input(data, series_name=name)
        _validate_time_index(series.index, strategy_name=name)
        if input_type == "returns":
            series = compute_equity_from_returns(series)
        elif input_type != "equity":
            raise ValueError("input_type must be either 'returns' or 'equity'.")
        series.name = name
        normalized.append((name, series))

    return normalized


def compute_equity_from_returns(returns: PlotInput) -> pd.Series:
    """Compound periodic returns into a cumulative equity series starting at 1.0."""

    normalized_returns = normalize_equity_input(returns, series_name="Strategy")
    cumulative = (1.0 + normalized_returns).cumprod()
    cumulative.name = normalized_returns.name
    return cumulative


def align_series_collection(strategy_series: Sequence[tuple[str, pd.Series]]) -> pd.DataFrame:
    """Align strategy series on the shared index using an inner join."""

    if len(strategy_series) == 0:
        raise ValueError("Strategy comparison input must be non-empty.")

    aligned: pd.DataFrame | None = None
    for name, series in strategy_series:
        frame = series.to_frame(name=name)
        aligned = frame if aligned is None else aligned.join(frame, how="inner")

    assert aligned is not None
    aligned = aligned.dropna(how="any")
    if aligned.empty:
        raise ValueError("Strategy comparison inputs must share at least one aligned index value.")
    return aligned


def plot_equity_comparison(
    strategy_data: SeriesInput,
    *,
    title: str = "Equity Comparison",
    output_path: Path | None = None,
    input_type: InputType = "equity",
) -> PlotResult:
    """Plot aligned equity curves for multiple strategy runs.

    Args:
        strategy_data: Either a mapping of run names to pandas Series or
            single-column DataFrames, or a sequence of ``(name, data)`` pairs.
            Inputs must be time-indexed. ``input_type`` controls whether each
            series represents periodic returns or already compounded equity.
        title: Chart title.
        output_path: Optional output location. When provided, the figure is
            saved as a PNG and the saved path is returned.
        input_type: Explicitly identifies whether the input values are periodic
            returns or cumulative equity values.
    """

    normalized = normalize_strategy_series_inputs(strategy_data, input_type=input_type)
    aligned = align_series_collection(normalized)

    figure, axis = create_figure(figsize=DEFAULT_FIGSIZE)
    for name in aligned.columns:
        axis.plot(aligned.index, aligned[name].values, label=name, linewidth=DEFAULT_LINEWIDTH)

    apply_axis_style(axis, title=title, x_label="Date", y_label="Equity", legend=True)
    finalize_figure(figure, axis, use_date_axis=True)

    return save_or_return_figure(figure, output_path)


def plot_strategy_overlays(
    strategy_frames: Sequence[pd.DataFrame],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Strategy Overlays",
    output_path: Path | None = None,
) -> PlotResult:
    """Overlay multiple strategy performance series on a shared visualization."""

    if len(strategy_frames) == 0:
        raise ValueError("Strategy comparison input must be non-empty.")
    if labels is not None and len(labels) != len(strategy_frames):
        raise ValueError("labels must match the number of strategy frames.")

    named_inputs: list[tuple[str, PlotInput]] = []
    for index, frame in enumerate(strategy_frames):
        label = labels[index] if labels is not None else f"Strategy {index + 1}"
        named_inputs.append((label, frame))

    return plot_equity_comparison(
        named_inputs,
        title=title,
        output_path=output_path,
        input_type="equity",
    )


def validate_metric_dataframe(
    metrics_frame: MetricInput,
    *,
    metric_name: str,
) -> pd.DataFrame:
    """Return a copied metric table containing one label column and one numeric metric column."""

    frame = _coerce_metric_frame(metrics_frame)
    if metric_name not in frame.columns:
        raise ValueError(f"Metrics input is missing required column: {metric_name}.")

    label_column = _resolve_label_column(frame)
    result = frame.loc[:, [label_column, metric_name]].copy(deep=True)
    result[metric_name] = pd.to_numeric(result[metric_name], errors="coerce")

    if result[metric_name].isna().any():
        raise ValueError(f"Metric column '{metric_name}' must contain only numeric values.")

    result = result.rename(columns={label_column: "label"})
    return result.reset_index(drop=True)


def plot_metric_comparison(
    metrics_frame: MetricInput,
    *,
    metric_name: str,
    title: str | None = None,
    output_path: Path | None = None,
) -> PlotResult:
    """Plot a comparison view for one metric across multiple strategy runs."""

    normalized = validate_metric_dataframe(metrics_frame, metric_name=metric_name)
    resolved_title = title or f"Strategy {metric_name} Comparison"
    values = normalized[metric_name].astype("float64")
    colors = ["tab:green" if value >= 0.0 else "tab:red" for value in values]

    figure, axis = create_figure(figsize=DEFAULT_FIGSIZE)
    axis.bar(
        normalized["label"].astype(str),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=DEFAULT_BAR_ALPHA,
    )
    axis.axhline(0.0, color="black", linewidth=REFERENCE_LINEWIDTH, linestyle="--", alpha=0.8)
    apply_axis_style(
        axis,
        title=resolved_title,
        x_label="Strategy",
        y_label=metric_name,
        grid_axis="y",
        rotate_x_labels=True,
    )
    finalize_figure(figure, axis)

    return save_or_return_figure(figure, output_path)


def plot_strategy_metric_bars(
    metrics_frame: MetricInput,
    *,
    metrics: Sequence[str],
    title: str = "Strategy Metric Comparison",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot grouped bar comparisons for selected metrics across strategy runs."""

    if len(metrics) == 0:
        raise ValueError("metrics must be non-empty.")

    frame = _coerce_metric_frame(metrics_frame)
    label_column = _resolve_label_column(frame)
    missing_metrics = [metric for metric in metrics if metric not in frame.columns]
    if missing_metrics:
        missing_list = ", ".join(missing_metrics)
        raise ValueError(f"Metrics input is missing required columns: {missing_list}.")

    result = frame.loc[:, [label_column, *metrics]].copy(deep=True)
    for metric in metrics:
        result[metric] = pd.to_numeric(result[metric], errors="coerce")
        if result[metric].isna().any():
            raise ValueError(f"Metric column '{metric}' must contain only numeric values.")

    labels = result[label_column].astype(str).tolist()
    positions = list(range(len(labels)))
    width = 0.8 / len(metrics)

    figure, axis = create_figure(figsize=DEFAULT_FIGSIZE)
    for metric_index, metric in enumerate(metrics):
        offsets = [position - 0.4 + (metric_index + 0.5) * width for position in positions]
        axis.bar(
            offsets,
            result[metric].astype("float64"),
            width=width,
            label=metric,
            alpha=DEFAULT_BAR_ALPHA,
            edgecolor="black",
            linewidth=0.8,
        )

    axis.axhline(0.0, color="black", linewidth=REFERENCE_LINEWIDTH, linestyle="--", alpha=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    apply_axis_style(
        axis,
        title=title,
        x_label="Strategy",
        y_label="Metric Value",
        legend=True,
        grid_axis="y",
        rotate_x_labels=True,
    )
    finalize_figure(figure, axis)

    return save_or_return_figure(figure, output_path)


def _coerce_named_series_inputs(strategy_data: SeriesInput) -> list[tuple[str, PlotInput]]:
    """Copy supported multi-strategy inputs into a normalized named list."""

    if isinstance(strategy_data, Mapping):
        if len(strategy_data) == 0:
            raise ValueError("Strategy comparison input must be non-empty.")
        return [(str(name), data) for name, data in strategy_data.items()]

    if not isinstance(strategy_data, Sequence) or isinstance(strategy_data, (str, bytes)):
        raise ValueError("Strategy comparison input must be a mapping or sequence of (name, data) pairs.")
    if len(strategy_data) == 0:
        raise ValueError("Strategy comparison input must be non-empty.")

    pairs: list[tuple[str, PlotInput]] = []
    for item in strategy_data:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError("Each strategy input must be a (name, data) pair.")
        name, data = item
        pairs.append((str(name), data))
    return pairs


def _validate_time_index(index: pd.Index, *, strategy_name: str) -> None:
    """Validate that a series index is time-based for plotting."""

    if not isinstance(index, _TIME_INDEX_TYPES):
        raise ValueError(f"Strategy '{strategy_name}' must be time-indexed.")


def _coerce_metric_frame(metrics_frame: MetricInput) -> pd.DataFrame:
    """Copy supported metric records into a DataFrame without mutating the caller."""

    if isinstance(metrics_frame, pd.DataFrame):
        frame = metrics_frame.copy(deep=True)
    elif isinstance(metrics_frame, Sequence) and not isinstance(metrics_frame, (str, bytes)):
        if len(metrics_frame) == 0:
            raise ValueError("Metrics input must be non-empty.")
        frame = pd.DataFrame(list(metrics_frame))
    else:
        raise ValueError("Metrics input must be a pandas DataFrame or a sequence of records.")

    if frame.empty:
        raise ValueError("Metrics input must be non-empty.")
    return frame


def _resolve_label_column(frame: pd.DataFrame) -> str:
    """Choose a strategy label column from common metric table conventions."""

    for column in _DEFAULT_LABEL_COLUMNS:
        if column in frame.columns:
            return column
    candidates = ", ".join(_DEFAULT_LABEL_COLUMNS)
    raise ValueError(f"Metrics input must contain one of the required label columns: {candidates}.")


