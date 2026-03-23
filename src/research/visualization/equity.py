"""Plotting interfaces for equity curves and cumulative return visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

PlotResult = Path | Figure


def plot_equity_curve(
    equity_data: pd.DataFrame,
    *,
    title: str = "Equity Curve",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot a strategy equity curve and return a figure or saved artifact path."""

    # TODO: Validate expected columns and render the equity curve visualization.
    raise NotImplementedError("Equity curve plotting is not implemented yet.")


def plot_cumulative_returns(
    returns_data: pd.DataFrame,
    *,
    title: str = "Cumulative Returns",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot cumulative returns and return a figure or saved artifact path."""

    # TODO: Transform returns into cumulative performance and render the plot.
    raise NotImplementedError("Cumulative return plotting is not implemented yet.")
