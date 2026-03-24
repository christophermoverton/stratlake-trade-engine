"""Visualization interfaces for research outputs and evaluation artifacts.

This package groups plotting helpers by domain so the research layer can keep
visualization logic modular, reusable, and easy to extend over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.research.visualization.artifacts import (
    get_plot_dir,
    get_plot_filename,
    get_plot_path,
    is_standard_plot_dir,
)
from src.research.visualization.plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE
from src.research.visualization.comparison import (
    plot_equity_comparison,
    plot_metric_comparison,
    plot_strategy_metric_bars,
    plot_strategy_overlays,
)
from src.research.visualization.diagnostics import (
    compute_trade_statistics,
    compute_long_short_counts,
    plot_drawdown,
    plot_exposure_over_time,
    plot_long_short_counts,
    plot_trade_return_distribution,
    plot_underwater_curve,
    plot_win_loss_distribution,
    plot_signal_distribution,
    plot_rolling_metric,
    plot_rolling_sharpe,
    plot_signal_diagnostics,
)
from src.research.visualization.equity import (
    plot_cumulative_returns,
    plot_equity_curve,
)
from src.research.visualization.walk_forward import (
    plot_fold_level_metrics,
    plot_walk_forward_results,
    plot_walk_forward_splits,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

PlotResult = Path | Figure

__all__ = [
    "PlotResult",
    "compute_long_short_counts",
    "compute_trade_statistics",
    "DEFAULT_DPI",
    "DEFAULT_FIGSIZE",
    "get_plot_dir",
    "get_plot_filename",
    "get_plot_path",
    "is_standard_plot_dir",
    "plot_cumulative_returns",
    "plot_drawdown",
    "plot_equity_comparison",
    "plot_equity_curve",
    "plot_exposure_over_time",
    "plot_fold_level_metrics",
    "plot_long_short_counts",
    "plot_metric_comparison",
    "plot_rolling_metric",
    "plot_rolling_sharpe",
    "plot_signal_distribution",
    "plot_signal_diagnostics",
    "plot_strategy_metric_bars",
    "plot_strategy_overlays",
    "plot_trade_return_distribution",
    "plot_underwater_curve",
    "plot_win_loss_distribution",
    "plot_walk_forward_results",
    "plot_walk_forward_splits",
]
