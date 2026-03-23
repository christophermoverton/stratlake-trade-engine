"""Visualization interfaces for research outputs and evaluation artifacts.

This package groups plotting helpers by domain so the research layer can keep
visualization logic modular, reusable, and easy to extend over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.research.visualization.comparison import (
    plot_metric_comparison,
    plot_strategy_metric_bars,
    plot_strategy_overlays,
)
from src.research.visualization.diagnostics import (
    plot_drawdown,
    plot_underwater_curve,
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
    "plot_cumulative_returns",
    "plot_drawdown",
    "plot_equity_curve",
    "plot_fold_level_metrics",
    "plot_metric_comparison",
    "plot_rolling_metric",
    "plot_rolling_sharpe",
    "plot_signal_diagnostics",
    "plot_strategy_metric_bars",
    "plot_strategy_overlays",
    "plot_underwater_curve",
    "plot_walk_forward_results",
    "plot_walk_forward_splits",
]
