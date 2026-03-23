"""Plotting interfaces for drawdowns, rolling metrics, and signal diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.research.visualization.equity import DEFAULT_FIGSIZE, normalize_equity_input

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

PlotInput = pd.Series | pd.DataFrame
PlotResult = Path | Figure
InputType = Literal["returns", "equity"]


def normalize_drawdown_input(data: PlotInput, *, series_name: str = "Strategy") -> pd.Series:
    """Return a copied numeric series suitable for drawdown computation."""

    return normalize_equity_input(data, series_name=series_name)


def compute_equity_from_returns(returns: PlotInput) -> pd.Series:
    """Compound periodic returns into a cumulative equity series starting at 1.0."""

    normalized_returns = normalize_drawdown_input(returns)
    cumulative_equity = (1.0 + normalized_returns).cumprod()
    cumulative_equity.name = normalized_returns.name
    return cumulative_equity


def compute_drawdown_series(data: PlotInput, *, input_type: InputType = "equity") -> pd.Series:
    """Compute a drawdown series from periodic returns or cumulative equity values.

    Args:
        data: Time-indexed pandas Series or single-column DataFrame.
        input_type: Explicitly identifies whether ``data`` contains periodic
            returns or cumulative equity values.

    Returns:
        A float series on the original index where peaks are ``0.0`` and
        drawdowns are negative decimal values.
    """

    normalized = normalize_drawdown_input(data)
    if input_type == "returns":
        cumulative_equity = compute_equity_from_returns(normalized)
    elif input_type == "equity":
        cumulative_equity = normalized
    else:
        raise ValueError("input_type must be either 'returns' or 'equity'.")

    running_peak = cumulative_equity.cummax()
    drawdown = cumulative_equity / running_peak - 1.0
    drawdown.name = f"{cumulative_equity.name or 'Strategy'} Drawdown"
    return drawdown


def plot_drawdown(
    equity_data: PlotInput,
    *,
    title: str = "Drawdown",
    output_path: Path | None = None,
    input_type: InputType = "equity",
) -> PlotResult:
    """Plot a drawdown time series and optionally save the resulting PNG artifact."""

    drawdown_series = compute_drawdown_series(equity_data, input_type=input_type)
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)

    axis.plot(
        drawdown_series.index,
        drawdown_series.values,
        label=drawdown_series.name or "Drawdown",
        linewidth=2.0,
        color="tab:red",
    )
    axis.fill_between(drawdown_series.index, drawdown_series.values, 0.0, color="tab:red", alpha=0.25)
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.legend()
    axis.grid(True, linestyle=":", linewidth=0.75, alpha=0.7)
    figure.autofmt_xdate()
    figure.tight_layout()

    if output_path is None:
        return figure

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, format="png", dpi=100)
    plt.close(figure)
    return resolved_output_path


def plot_underwater_curve(
    equity_data: PlotInput,
    *,
    title: str = "Underwater Curve",
    output_path: Path | None = None,
    input_type: InputType = "equity",
) -> PlotResult:
    """Plot drawdowns as an underwater curve.

    This is a semantic alias for ``plot_drawdown`` with an underwater-focused
    default title.
    """

    return plot_drawdown(
        equity_data,
        title=title,
        output_path=output_path,
        input_type=input_type,
    )


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
