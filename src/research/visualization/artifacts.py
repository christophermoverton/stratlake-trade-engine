"""Helpers for deterministic visualization artifact names and locations."""

from __future__ import annotations

from pathlib import Path

_PLOTS_DIRNAME = "plots"
_STATIC_PLOT_FILENAMES: dict[str, str] = {
    "cumulative_returns": "cumulative_returns.png",
    "drawdown": "drawdown.png",
    "equity_comparison": "equity_comparison.png",
    "equity_curve": "equity_curve.png",
    "exposure_over_time": "exposure_over_time.png",
    "long_short_counts": "long_short_counts.png",
    "rolling_sharpe": "rolling_sharpe.png",
    "signal_diagnostics": "signal_distribution.png",
    "signal_distribution": "signal_distribution.png",
    "strategy_metric_bars": "strategy_metric_comparison.png",
    "strategy_overlays": "equity_comparison.png",
    "trade_return_distribution": "trade_return_distribution.png",
    "underwater_curve": "drawdown.png",
    "walk_forward_results": "walk_forward_results.png",
    "walk_forward_splits": "walk_forward_splits.png",
    "win_loss_distribution": "win_loss_distribution.png",
}


def get_plot_dir(run_dir: Path) -> Path:
    """Return the standardized run-scoped directory for plot artifacts."""

    return Path(run_dir) / _PLOTS_DIRNAME


def get_plot_path(run_dir: Path, plot_name: str, *, metric_name: str | None = None) -> Path:
    """Return the standardized PNG artifact path for a supported plot."""

    return get_plot_dir(run_dir) / get_plot_filename(plot_name, metric_name=metric_name)


def get_plot_filename(plot_name: str, *, metric_name: str | None = None) -> str:
    """Return the standardized PNG filename for a supported plot."""

    if plot_name in _STATIC_PLOT_FILENAMES:
        return _STATIC_PLOT_FILENAMES[plot_name]
    if plot_name == "fold_metric":
        return f"fold_metric_{_normalize_metric_name(metric_name)}.png"
    if plot_name == "metric_comparison":
        return f"metric_comparison_{_normalize_metric_name(metric_name)}.png"
    raise ValueError(f"Unsupported plot artifact name: {plot_name}")


def is_standard_plot_dir(run_dir: Path, plots_dir: Path) -> bool:
    """Return whether a provided directory matches the standardized run plot directory."""

    return Path(plots_dir) == get_plot_dir(run_dir)


def _normalize_metric_name(metric_name: str | None) -> str:
    if metric_name is None:
        raise ValueError("metric_name is required for metric-qualified plot artifacts.")

    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in metric_name.strip()
    ).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    if not normalized:
        raise ValueError("metric_name must contain at least one alphanumeric character.")
    return normalized


__all__ = [
    "get_plot_dir",
    "get_plot_filename",
    "get_plot_path",
    "is_standard_plot_dir",
]
