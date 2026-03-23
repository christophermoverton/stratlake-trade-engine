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
MetricInput = pd.Series | pd.DataFrame
PositionsInput = pd.Series | pd.DataFrame


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


def normalize_returns_input(data: MetricInput, *, series_name: str = "Strategy Returns") -> pd.Series:
    """Return a copied numeric returns series from a Series or single-column DataFrame."""

    normalized = normalize_equity_input(data, series_name=series_name)
    normalized.name = normalized.name or series_name
    return normalized


def normalize_signal_input(data: PlotInput, *, series_name: str = "Signal") -> pd.Series:
    """Return a copied numeric signal series from a Series or single-column DataFrame."""

    normalized = normalize_equity_input(data, series_name=series_name)
    normalized.name = normalized.name or series_name
    return normalized


def normalize_exposure_input(data: PlotInput, *, series_name: str = "Exposure") -> pd.Series:
    """Return a copied numeric exposure series from a Series or single-column DataFrame."""

    normalized = normalize_equity_input(data, series_name=series_name)
    normalized.name = normalized.name or series_name
    return normalized


def compute_long_short_counts(positions: PositionsInput) -> pd.DataFrame:
    """Compute long and short counts by timestamp from aggregate or per-asset positions.

    Args:
        positions: Either a time-indexed aggregate position Series or single-column
            DataFrame, or a multi-column DataFrame where each column represents an
            asset position at each timestamp.

    Returns:
        A DataFrame indexed like the input with ``long_count`` and ``short_count``
        integer columns.
    """

    if isinstance(positions, pd.Series):
        normalized = normalize_equity_input(positions, series_name="Position")
        counts = pd.DataFrame(index=normalized.index)
        counts["long_count"] = (normalized > 0.0).astype("int64")
        counts["short_count"] = (normalized < 0.0).astype("int64")
        return counts

    if not isinstance(positions, pd.DataFrame):
        raise ValueError("Positions input must be a pandas Series or DataFrame.")
    if positions.empty:
        raise ValueError("Positions input must be non-empty.")

    numeric_positions = positions.copy(deep=True).apply(pd.to_numeric, errors="coerce")
    numeric_positions = numeric_positions.dropna(axis=1, how="all")

    if numeric_positions.empty:
        raise ValueError("Positions input must contain at least one numeric value.")
    if numeric_positions.notna().sum().sum() == 0:
        raise ValueError("Positions input must contain at least one numeric value.")

    if len(numeric_positions.columns) == 1:
        single_position = numeric_positions.iloc[:, 0].astype("float64")
        counts = pd.DataFrame(index=single_position.index)
        counts["long_count"] = (single_position > 0.0).astype("int64")
        counts["short_count"] = (single_position < 0.0).astype("int64")
        return counts

    counts = pd.DataFrame(index=numeric_positions.index.copy())
    counts["long_count"] = numeric_positions.gt(0.0).sum(axis=1).astype("int64")
    counts["short_count"] = numeric_positions.lt(0.0).sum(axis=1).astype("int64")
    return counts


def _save_or_return_figure(figure: Figure, output_path: Path | None) -> PlotResult:
    """Save a figure as a PNG artifact or return it directly."""

    if output_path is None:
        return figure

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, format="png", dpi=100)
    plt.close(figure)
    return resolved_output_path


def _validate_rolling_window(window: int) -> None:
    """Validate a rolling window size."""

    if window <= 0:
        raise ValueError("window must be greater than zero.")


def compute_rolling_sharpe(
    returns: MetricInput,
    *,
    window: int,
) -> pd.Series:
    """Compute a rolling Sharpe ratio from periodic returns.

    Args:
        returns: Time-indexed periodic returns as a pandas Series or
            single-column DataFrame.
        window: Rolling window size. Must be greater than zero.

    Returns:
        A float series on the original index where values are ``NaN`` until the
        rolling window is fully populated. Windows with zero standard deviation
        also produce ``NaN`` values.
    """

    _validate_rolling_window(window)
    normalized_returns = normalize_returns_input(returns)
    rolling_mean = normalized_returns.rolling(window=window, min_periods=window).mean()
    rolling_std = normalized_returns.rolling(window=window, min_periods=window).std()
    rolling_std = rolling_std.mask(rolling_std == 0.0)
    rolling_sharpe = rolling_mean / rolling_std
    rolling_sharpe.name = f"{normalized_returns.name or 'Strategy'} Rolling Sharpe"
    return rolling_sharpe


def compute_rolling_volatility(
    returns: MetricInput,
    *,
    window: int,
) -> pd.Series:
    """Compute rolling return volatility from periodic returns."""

    _validate_rolling_window(window)
    normalized_returns = normalize_returns_input(returns)
    rolling_volatility = normalized_returns.rolling(window=window, min_periods=window).std()
    rolling_volatility.name = f"{normalized_returns.name or 'Strategy'} Rolling Volatility"
    return rolling_volatility


def plot_rolling_metric(
    metric_data: MetricInput,
    *,
    metric_name: str,
    window_label: str | None = None,
    title: str | None = None,
    output_path: Path | None = None,
) -> PlotResult:
    """Plot a rolling metric time series for diagnostic analysis.

    Args:
        metric_data: Time-indexed rolling metric values as a pandas Series or
            single-column DataFrame.
        metric_name: Metric label used for chart titling, axis labeling, and
            legend text.
        window_label: Optional label such as ``"20-period"`` shown in the
            legend and title.
        title: Optional explicit chart title. When omitted, a deterministic
            title is generated from ``metric_name`` and ``window_label``.
        output_path: Optional output location. When provided, the figure is
            saved as a PNG and the saved path is returned.
    """

    normalized_metric = normalize_equity_input(metric_data, series_name=metric_name)
    label = metric_name if window_label is None else f"{metric_name} ({window_label})"
    resolved_title = title or label

    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)
    axis.plot(
        normalized_metric.index,
        normalized_metric.values,
        label=label,
        linewidth=2.0,
        color="tab:blue",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    axis.set_title(resolved_title)
    axis.set_xlabel("Date")
    axis.set_ylabel(metric_name)
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


def plot_rolling_sharpe(
    returns: MetricInput,
    *,
    window: int,
    title: str | None = None,
    output_path: Path | None = None,
) -> PlotResult:
    """Compute and plot a rolling Sharpe ratio from periodic returns."""

    rolling_sharpe = compute_rolling_sharpe(returns, window=window)
    return plot_rolling_metric(
        rolling_sharpe,
        metric_name="Rolling Sharpe",
        window_label=f"{window}-period",
        title=title or f"Rolling Sharpe ({window}-period)",
        output_path=output_path,
    )


def plot_signal_distribution(
    signals: PlotInput,
    *,
    title: str = "Signal Distribution",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot a histogram of signal or target exposure values."""

    normalized_signals = normalize_signal_input(signals)
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)

    unique_values = pd.Index(normalized_signals.dropna().unique())
    is_discrete = len(unique_values) <= 20 and all(float(value).is_integer() for value in unique_values)

    hist_kwargs: dict[str, Any] = {
        "color": "tab:blue",
        "edgecolor": "black",
        "alpha": 0.8,
    }
    if is_discrete:
        sorted_unique = sorted(float(value) for value in unique_values)
        bins = [sorted_unique[0] - 0.5]
        bins.extend(value + 0.5 for value in sorted_unique)
        hist_kwargs["bins"] = bins
    else:
        hist_kwargs["bins"] = min(30, max(10, len(normalized_signals) // 5))

    axis.hist(normalized_signals.values, **hist_kwargs)
    axis.set_title(title)
    axis.set_xlabel(normalized_signals.name or "Signal")
    axis.set_ylabel("Frequency")
    axis.grid(True, linestyle=":", linewidth=0.75, alpha=0.7, axis="y")
    figure.tight_layout()

    return _save_or_return_figure(figure, output_path)


def plot_exposure_over_time(
    exposure: PlotInput,
    *,
    title: str = "Exposure Over Time",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot aggregate strategy exposure, weight, or net exposure over time."""

    normalized_exposure = normalize_exposure_input(exposure)
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)

    axis.plot(
        normalized_exposure.index,
        normalized_exposure.values,
        label=normalized_exposure.name or "Exposure",
        linewidth=2.0,
        color="tab:green",
    )
    if normalized_exposure.min() <= 0.0 <= normalized_exposure.max():
        axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(normalized_exposure.name or "Exposure")
    axis.legend()
    axis.grid(True, linestyle=":", linewidth=0.75, alpha=0.7)
    figure.autofmt_xdate()
    figure.tight_layout()

    return _save_or_return_figure(figure, output_path)


def plot_long_short_counts(
    positions: PositionsInput,
    *,
    title: str = "Long/Short Counts",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot long and short position counts over time.

    Args:
        positions: Either a time-indexed aggregate position Series or
            single-column DataFrame, or a multi-column DataFrame whose columns
            represent per-asset positions and rows represent timestamps.
    """

    counts = compute_long_short_counts(positions)
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)

    axis.plot(counts.index, counts["long_count"].values, label="Long Count", linewidth=2.0, color="tab:green")
    axis.plot(counts.index, counts["short_count"].values, label="Short Count", linewidth=2.0, color="tab:red")
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Count")
    axis.legend()
    axis.grid(True, linestyle=":", linewidth=0.75, alpha=0.7)
    figure.autofmt_xdate()
    figure.tight_layout()

    return _save_or_return_figure(figure, output_path)


def plot_signal_diagnostics(
    signals: PlotInput,
    *,
    feature_columns: Sequence[str] | None = None,
    title: str = "Signal Diagnostics",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot a diagnostic histogram for strategy signals.

    ``feature_columns`` is accepted for backward compatibility but is not used by
    this artifact-focused plotting interface.
    """

    del feature_columns
    return plot_signal_distribution(signals, title=title, output_path=output_path)
