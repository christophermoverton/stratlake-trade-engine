"""Plotting interfaces for comparing multiple strategy runs and outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

from src.research.visualization.equity import normalize_equity_input
from src.research.visualization.plot_utils import (
    DEFAULT_BAR_ALPHA,
    DEFAULT_FILL_ALPHA,
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
ComparisonView = Literal["aggregate", "raw"]
SeriesInput = Mapping[str, PlotInput] | Sequence[tuple[str, PlotInput]]
MetricInput = pd.DataFrame | Sequence[Mapping[str, object]]

_TIME_INDEX_TYPES = (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)
_DEFAULT_LABEL_COLUMNS = ("strategy", "strategy_name", "run_id")
_RAW_LINE_ALPHA = 0.16
_RAW_LINEWIDTH = 1.0
_SUMMARY_LINEWIDTH = DEFAULT_LINEWIDTH + 0.35


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
        series = _collapse_duplicate_index(series)
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
    view: ComparisonView = "aggregate",
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
        view: ``"aggregate"`` renders one summary line per strategy and adds an
            interquartile band when multiple runs share the same strategy label.
            ``"raw"`` keeps faint individual run overlays and a highlighted
            median summary per strategy for dense diagnostic inspection.
    """

    normalized = normalize_strategy_series_inputs(strategy_data, input_type=input_type)
    grouped = _group_strategy_series(normalized)
    summaries = _summarize_grouped_series(grouped)
    aligned = align_series_collection(
        [(name, summary["median"]) for name, summary in summaries]
    )

    figure, axis = create_figure(figsize=DEFAULT_FIGSIZE)

    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle_colors:
        cycle_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    legend_handles: list[Line2D] = []

    if view == "aggregate":
        for index, (name, summary) in enumerate(summaries):
            color = cycle_colors[index % len(cycle_colors)]
            median = aligned[name]
            line = axis.plot(
                median.index,
                median.values,
                color=color,
                linewidth=_SUMMARY_LINEWIDTH,
                label=_summary_label(name=name, run_count=int(summary["run_count"])),
            )[0]
            legend_handles.append(line)

            lower = summary["lower"].reindex(median.index)
            upper = summary["upper"].reindex(median.index)
            if int(summary["run_count"]) > 1:
                axis.fill_between(
                    median.index,
                    lower.values,
                    upper.values,
                    color=color,
                    alpha=DEFAULT_FILL_ALPHA,
                    linewidth=0.0,
                )

        if any(int(summary["run_count"]) > 1 for _, summary in summaries):
            axis.text(
                0.01,
                0.02,
                "Shaded bands show the interquartile equity range across runs.",
                transform=axis.transAxes,
                fontsize=9,
                alpha=0.85,
                va="bottom",
            )
    elif view == "raw":
        include_raw_legend = any(int(summary["run_count"]) > 1 for _, summary in summaries)
        if include_raw_legend:
            legend_handles.append(
                Line2D(
                    [0.0],
                    [0.0],
                    color="0.5",
                    linewidth=_RAW_LINEWIDTH,
                    alpha=_RAW_LINE_ALPHA,
                    label="Individual runs",
                )
            )
        for index, (name, summary) in enumerate(summaries):
            color = cycle_colors[index % len(cycle_colors)]
            if int(summary["run_count"]) > 1:
                for series in summary["runs"]:
                    axis.plot(
                        series.index,
                        series.values,
                        color=color,
                        linewidth=_RAW_LINEWIDTH,
                        alpha=_RAW_LINE_ALPHA,
                        label="_nolegend_",
                    )

            median = aligned[name]
            line = axis.plot(
                median.index,
                median.values,
                color=color,
                linewidth=_SUMMARY_LINEWIDTH,
                label=_summary_label(name=name, run_count=int(summary["run_count"])),
            )[0]
            legend_handles.append(line)
    else:
        raise ValueError("view must be either 'aggregate' or 'raw'.")

    axis.legend(handles=legend_handles, loc="best", frameon=False)
    apply_axis_style(axis, title=title, x_label="Date", y_label="Equity")
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
        view="raw",
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
    normalized = normalized.sort_values(by=metric_name, ascending=False, kind="mergesort").reset_index(drop=True)
    values = normalized[metric_name].astype("float64")
    top_value = values.iloc[0]
    colors = [
        "tab:blue" if value == top_value else ("tab:green" if value >= 0.0 else "tab:red")
        for value in values
    ]

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
    axis.yaxis.set_major_locator(MaxNLocator(nbins="auto", min_n_ticks=4))
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


def _group_strategy_series(strategy_series: Sequence[tuple[str, pd.Series]]) -> list[tuple[str, list[pd.Series]]]:
    """Group normalized series by strategy name while preserving first-seen order."""

    grouped: dict[str, list[pd.Series]] = {}
    for name, series in strategy_series:
        grouped.setdefault(name, []).append(series.copy(deep=True))
    return [(name, grouped[name]) for name in grouped]


def _summarize_grouped_series(
    grouped_series: Sequence[tuple[str, list[pd.Series]]],
) -> list[tuple[str, dict[str, object]]]:
    """Build deterministic median and variability summaries for each strategy."""

    summaries: list[tuple[str, dict[str, object]]] = []
    for name, runs in grouped_series:
        combined = pd.concat(list(runs), axis=1, join="outer").sort_index()
        combined.columns = [f"run_{index}" for index in range(len(runs))]
        median = combined.median(axis=1, skipna=True).dropna().astype("float64")
        lower = combined.quantile(0.25, axis=1, interpolation="linear").dropna().astype("float64")
        upper = combined.quantile(0.75, axis=1, interpolation="linear").dropna().astype("float64")
        summaries.append(
            (
                name,
                {
                    "median": median.rename(name),
                    "lower": lower.rename(name),
                    "upper": upper.rename(name),
                    "run_count": len(runs),
                    "runs": [run.copy(deep=True).rename(name) for run in runs],
                },
            )
        )
    return summaries


def _summary_label(*, name: str, run_count: int) -> str:
    """Return a legend label that matches the rendered summary encoding."""

    if run_count > 1:
        return f"{name} median (n={run_count})"
    return name


def _validate_time_index(index: pd.Index, *, strategy_name: str) -> None:
    """Validate that a series index is time-based for plotting."""

    if not isinstance(index, _TIME_INDEX_TYPES):
        raise ValueError(f"Strategy '{strategy_name}' must be time-indexed.")


def _collapse_duplicate_index(series: pd.Series) -> pd.Series:
    """Reduce duplicate timestamps to the last observed value for stable plotting."""

    if not series.index.has_duplicates:
        return series
    collapsed = series.groupby(level=0, sort=True).last()
    collapsed.name = series.name
    return collapsed


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


