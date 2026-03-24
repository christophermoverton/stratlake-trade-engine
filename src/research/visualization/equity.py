"""Plotting interfaces for equity curves and cumulative return visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from src.research.visualization.plot_utils import (
    DEFAULT_FIGSIZE,
    DEFAULT_LINEWIDTH,
    SECONDARY_LINEWIDTH,
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


def normalize_equity_input(data: PlotInput, *, series_name: str) -> pd.Series:
    """Return a copied float series from a Series or single-column DataFrame input."""

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Plot input must be non-empty.")
        if len(data.columns) != 1:
            raise ValueError("DataFrame plot inputs must contain exactly one column.")
        series = data.iloc[:, 0]
    elif isinstance(data, pd.Series):
        series = data
    else:
        raise ValueError("Plot input must be a pandas Series or single-column DataFrame.")

    if series.empty:
        raise ValueError("Plot input must be non-empty.")

    normalized = series.copy(deep=True)
    normalized = pd.to_numeric(normalized, errors="coerce")
    normalized = normalized.dropna()

    if normalized.empty:
        raise ValueError("Plot input must contain at least one numeric value.")
    if not isinstance(normalized.index, pd.Index):
        raise ValueError("Plot input must provide an index for plotting.")

    normalized = normalized.astype("float64")
    normalized.name = series.name or series_name
    return normalized


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compound periodic returns into a cumulative equity series starting at 1.0."""

    normalized_returns = normalize_equity_input(returns, series_name="Strategy")
    cumulative = (1.0 + normalized_returns).cumprod()
    cumulative.name = normalized_returns.name
    return cumulative


def align_strategy_and_benchmark(
    strategy: pd.Series,
    benchmark: pd.Series | None,
) -> tuple[pd.Series, pd.Series | None]:
    """Align strategy and benchmark series on the shared index."""

    if benchmark is None:
        return strategy, None

    aligned_strategy, aligned_benchmark = strategy.align(benchmark, join="inner")
    aligned_strategy = aligned_strategy.dropna()
    aligned_benchmark = aligned_benchmark.dropna()
    aligned_strategy, aligned_benchmark = aligned_strategy.align(aligned_benchmark, join="inner")

    if aligned_strategy.empty or aligned_benchmark.empty:
        raise ValueError("Strategy and benchmark inputs must share at least one aligned index value.")

    return aligned_strategy, aligned_benchmark


def plot_equity_curve(
    equity_data: PlotInput,
    *,
    benchmark_data: PlotInput | None = None,
    title: str = "Equity Curve",
    output_path: Path | None = None,
    input_type: InputType = "equity",
) -> PlotResult:
    """Plot strategy equity over time and optionally save the resulting PNG artifact.

    Args:
        equity_data: Time-indexed strategy series. Pass periodic returns when
            ``input_type="returns"`` or an already compounded equity series when
            ``input_type="equity"``.
        benchmark_data: Optional benchmark series in the same format as
            ``equity_data``. When provided, it is aligned on the shared index.
        title: Chart title.
        output_path: Optional output location. When provided, the figure is saved
            as a PNG and the saved path is returned.
        input_type: Explicitly controls whether the inputs represent periodic
            returns or already compounded equity values.
    """

    strategy_series = _prepare_plot_series(equity_data, input_type=input_type, default_name="Strategy")
    benchmark_series = (
        _prepare_plot_series(benchmark_data, input_type=input_type, default_name="Benchmark")
        if benchmark_data is not None
        else None
    )
    strategy_series, benchmark_series = align_strategy_and_benchmark(strategy_series, benchmark_series)

    return _plot_performance_series(
        strategy_series,
        benchmark_series=benchmark_series,
        title=title,
        y_label="Equity",
        output_path=output_path,
    )


def plot_cumulative_returns(
    returns_data: PlotInput,
    *,
    benchmark_data: PlotInput | None = None,
    title: str = "Cumulative Returns",
    output_path: Path | None = None,
    input_type: InputType = "returns",
) -> PlotResult:
    """Plot cumulative performance from returns or equity inputs.

    Args:
        returns_data: Time-indexed strategy series. Periodic returns are assumed
            by default and compounded internally into cumulative performance.
        benchmark_data: Optional benchmark series in the same format as
            ``returns_data``. When provided, it is aligned on the shared index.
        title: Chart title.
        output_path: Optional output location. When provided, the figure is saved
            as a PNG and the saved path is returned.
        input_type: Explicitly controls whether the inputs represent periodic
            returns or already compounded equity values.
    """

    strategy_series = _prepare_plot_series(returns_data, input_type=input_type, default_name="Strategy")
    benchmark_series = (
        _prepare_plot_series(benchmark_data, input_type=input_type, default_name="Benchmark")
        if benchmark_data is not None
        else None
    )
    strategy_series, benchmark_series = align_strategy_and_benchmark(strategy_series, benchmark_series)

    cumulative_strategy = strategy_series - 1.0
    cumulative_strategy.name = strategy_series.name

    cumulative_benchmark = None
    if benchmark_series is not None:
        cumulative_benchmark = benchmark_series - 1.0
        cumulative_benchmark.name = benchmark_series.name

    return _plot_performance_series(
        cumulative_strategy,
        benchmark_series=cumulative_benchmark,
        title=title,
        y_label="Cumulative Return",
        output_path=output_path,
    )


def _prepare_plot_series(data: PlotInput, *, input_type: InputType, default_name: str) -> pd.Series:
    """Normalize an input series and convert returns to equity when needed."""

    normalized = normalize_equity_input(data, series_name=default_name)
    if input_type == "returns":
        return compute_cumulative_returns(normalized)
    if input_type == "equity":
        return normalized
    raise ValueError("input_type must be either 'returns' or 'equity'.")


def _plot_performance_series(
    strategy_series: pd.Series,
    *,
    benchmark_series: pd.Series | None,
    title: str,
    y_label: str,
    output_path: Path | None,
) -> PlotResult:
    """Render and optionally save a deterministic matplotlib performance chart."""

    figure, axis = create_figure(figsize=DEFAULT_FIGSIZE)
    axis.plot(
        strategy_series.index,
        strategy_series.values,
        label=strategy_series.name or "Strategy",
        linewidth=DEFAULT_LINEWIDTH,
    )

    if benchmark_series is not None:
        axis.plot(
            benchmark_series.index,
            benchmark_series.values,
            label=benchmark_series.name or "Benchmark",
            linewidth=SECONDARY_LINEWIDTH,
            linestyle="--",
        )

    apply_axis_style(axis, title=title, x_label="Date", y_label=y_label, legend=True)
    finalize_figure(figure, axis, use_date_axis=True)
    return save_or_return_figure(figure, output_path)
