"""Plotting interfaces for comparing multiple strategy runs and outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

PlotResult = Path | Figure


def plot_strategy_overlays(
    strategy_frames: Sequence[pd.DataFrame],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Strategy Overlays",
    output_path: Path | None = None,
) -> PlotResult:
    """Overlay multiple strategy performance series on a shared visualization."""

    # TODO: Align strategy series and render a multi-run overlay plot.
    raise NotImplementedError("Strategy overlay plotting is not implemented yet.")


def plot_metric_comparison(
    metrics_frame: pd.DataFrame,
    *,
    metric_name: str,
    title: str | None = None,
    output_path: Path | None = None,
) -> PlotResult:
    """Plot a comparison view for one metric across multiple strategy runs."""

    # TODO: Render a metric comparison chart for the selected performance metric.
    raise NotImplementedError("Metric comparison plotting is not implemented yet.")


def plot_strategy_metric_bars(
    metrics_frame: pd.DataFrame,
    *,
    metrics: Sequence[str],
    title: str = "Strategy Metric Comparison",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot bar-based comparisons for selected metrics across strategy runs."""

    # TODO: Render grouped bar comparisons for the requested strategy metrics.
    raise NotImplementedError("Strategy metric bar plotting is not implemented yet.")
