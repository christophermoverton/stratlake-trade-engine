"""Plotting interfaces for walk-forward splits and fold-level evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

PlotResult = Path | Figure


def plot_walk_forward_splits(
    splits_frame: pd.DataFrame,
    *,
    title: str = "Walk-Forward Splits",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot train and test windows for each walk-forward split."""

    # TODO: Visualize split boundaries and train/test windows across time.
    raise NotImplementedError("Walk-forward split plotting is not implemented yet.")


def plot_fold_level_metrics(
    fold_metrics: pd.DataFrame,
    *,
    metric_name: str,
    title: str | None = None,
    output_path: Path | None = None,
) -> PlotResult:
    """Plot fold-level evaluation metrics across walk-forward iterations."""

    # TODO: Render metric values for each fold to inspect stability over time.
    raise NotImplementedError("Fold-level metric plotting is not implemented yet.")


def plot_walk_forward_results(
    fold_results: Sequence[pd.DataFrame],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Walk-Forward Results",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot aggregated or per-fold performance outputs from walk-forward runs."""

    # TODO: Combine fold results into a single visualization for inspection.
    raise NotImplementedError("Walk-forward results plotting is not implemented yet.")
