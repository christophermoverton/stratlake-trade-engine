"""Shared plotting defaults and helper utilities for visualization artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any
    Figure = Any

DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 100
DEFAULT_LINEWIDTH = 2.0
SECONDARY_LINEWIDTH = 1.75
REFERENCE_LINEWIDTH = 1.0
DEFAULT_GRID_ALPHA = 0.7
DEFAULT_FILL_ALPHA = 0.25
DEFAULT_BAR_ALPHA = 0.85
DEFAULT_HIST_ALPHA = 0.8
DEFAULT_MARKER = "o"

def create_figure(*, figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> tuple[Figure, Axes]:
    """Create a matplotlib figure using repository-wide sizing defaults."""

    return plt.subplots(figsize=figsize)


def apply_axis_style(
    axis: Axes,
    *,
    title: str,
    x_label: str,
    y_label: str,
    legend: bool = False,
    grid: bool = True,
    grid_axis: str = "both",
    rotate_x_labels: bool = False,
) -> None:
    """Apply consistent axis labels, legend behavior, and grid styling."""

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    if legend:
        handles, labels = axis.get_legend_handles_labels()
        if handles and labels:
            axis.legend(loc="best", frameon=False)

    if grid:
        axis.grid(True, axis=grid_axis, linestyle=":", linewidth=0.75, alpha=DEFAULT_GRID_ALPHA)

    if rotate_x_labels:
        for tick in axis.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")


def apply_date_axis(axis: Axes) -> None:
    """Apply a consistent date axis formatter when the x-axis is time-based."""

    locator = mdates.AutoDateLocator()
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def finalize_figure(figure: Figure, axis: Axes, *, use_date_axis: bool = False) -> None:
    """Apply final layout and axis formatting before save or return."""

    if use_date_axis:
        apply_date_axis(axis)
        figure.autofmt_xdate()
    figure.tight_layout()


def save_or_return_figure(figure: Figure, output_path: Path | None) -> Path | Figure:
    """Save a figure to a deterministic PNG artifact or return it directly."""

    if output_path is None:
        return figure

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, format="png", dpi=DEFAULT_DPI)
    plt.close(figure)
    return resolved_output_path
