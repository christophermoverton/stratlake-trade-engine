"""Plotting interfaces for drawdowns, rolling metrics, and signal diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

PlotResult = Path | Figure


def plot_drawdown(
    equity_data: pd.DataFrame,
    *,
    title: str = "Drawdown",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot drawdown series derived from strategy equity or returns data."""

    # TODO: Compute drawdown from the provided series and render the chart.
    raise NotImplementedError("Drawdown plotting is not implemented yet.")


def plot_rolling_metric(
    metric_data: pd.Series,
    *,
    metric_name: str,
    window_label: str | None = None,
    title: str | None = None,
    output_path: Path | None = None,
) -> PlotResult:
    """Plot a rolling metric time series for diagnostic analysis."""

    # TODO: Render the rolling metric series with appropriate labels and scales.
    raise NotImplementedError("Rolling metric plotting is not implemented yet.")


def plot_signal_diagnostics(
    signals: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    title: str = "Signal Diagnostics",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot diagnostic views for generated signals and related feature inputs."""

    # TODO: Visualize signal behavior, coverage, and optional feature relationships.
    raise NotImplementedError("Signal diagnostics plotting is not implemented yet.")
