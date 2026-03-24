"""Helpers for deterministic visualization artifact names, intents, and locations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

_PLOTS_DIRNAME = "plots"
PlotIntent = Literal["report", "debug"]

_PLOT_SPECS: dict[str, tuple[str, PlotIntent]] = {
    "cumulative_returns": ("cumulative_returns.png", "report"),
    "drawdown": ("drawdown.png", "report"),
    "equity_comparison": ("equity_comparison.png", "report"),
    "equity_comparison_debug": ("equity_comparison_debug.png", "debug"),
    "equity_curve": ("equity_curve.png", "report"),
    "exposure_over_time": ("exposure_over_time_debug.png", "debug"),
    "long_short_counts": ("long_short_counts_debug.png", "debug"),
    "rolling_sharpe_debug": ("rolling_sharpe_debug.png", "debug"),
    "signal_diagnostics": ("signal_distribution_debug.png", "debug"),
    "signal_distribution": ("signal_distribution_debug.png", "debug"),
    "strategy_metric_bars": ("strategy_metric_comparison.png", "report"),
    "strategy_overlays": ("equity_comparison.png", "report"),
    "trade_return_distribution_debug": ("trade_return_distribution_debug.png", "debug"),
    "underwater_curve": ("drawdown.png", "report"),
    "walk_forward_results": ("walk_forward_results.png", "report"),
    "walk_forward_splits": ("walk_forward_splits.png", "report"),
    "win_loss_distribution_debug": ("win_loss_distribution_debug.png", "debug"),
}
_PLOT_NAME_ALIASES: dict[str, str] = {
    "rolling_sharpe": "rolling_sharpe_debug",
    "trade_return_distribution": "trade_return_distribution_debug",
    "win_loss_distribution": "win_loss_distribution_debug",
}


def get_plot_dir(run_dir: Path) -> Path:
    """Return the standardized run-scoped directory for plot artifacts."""

    return Path(run_dir) / _PLOTS_DIRNAME


def get_plot_path(run_dir: Path, plot_name: str, *, metric_name: str | None = None) -> Path:
    """Return the standardized PNG artifact path for a supported plot."""

    return get_plot_dir(run_dir) / get_plot_filename(plot_name, metric_name=metric_name)


def get_plot_filename(plot_name: str, *, metric_name: str | None = None) -> str:
    """Return the standardized PNG filename for a supported plot."""

    canonical_name = _resolve_plot_name(plot_name)
    if canonical_name in _PLOT_SPECS:
        return _PLOT_SPECS[canonical_name][0]
    if plot_name == "fold_metric":
        return f"fold_metric_{_normalize_metric_name(metric_name)}.png"
    if plot_name == "metric_comparison":
        return f"metric_comparison_{_normalize_metric_name(metric_name)}.png"
    raise ValueError(f"Unsupported plot artifact name: {plot_name}")


def get_plot_intent(plot_name: str) -> PlotIntent:
    """Return whether a supported plot artifact is report-quality or debug-oriented."""

    canonical_name = _resolve_plot_name(plot_name)
    if canonical_name in _PLOT_SPECS:
        return _PLOT_SPECS[canonical_name][1]
    if canonical_name in {"fold_metric", "metric_comparison"}:
        return "report"
    raise ValueError(f"Unsupported plot artifact name: {plot_name}")


def get_canonical_plot_name(plot_name: str) -> str:
    """Return the canonical plot name used for deterministic artifact wiring."""

    canonical_name = _resolve_plot_name(plot_name)
    if canonical_name in _PLOT_SPECS or canonical_name in {"fold_metric", "metric_comparison"}:
        return canonical_name
    raise ValueError(f"Unsupported plot artifact name: {plot_name}")


def list_plot_names(*, intent: PlotIntent | None = None) -> tuple[str, ...]:
    """Return supported plot names, optionally filtered by artifact intent."""

    names = tuple(_PLOT_SPECS)
    if intent is None:
        return names
    return tuple(name for name in names if _PLOT_SPECS[name][1] == intent)


def is_standard_plot_dir(run_dir: Path, plots_dir: Path) -> bool:
    """Return whether a provided directory matches the standardized run plot directory."""

    return Path(plots_dir) == get_plot_dir(run_dir)


def _resolve_plot_name(plot_name: str) -> str:
    return _PLOT_NAME_ALIASES.get(plot_name, plot_name)


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
    "PlotIntent",
    "get_canonical_plot_name",
    "get_plot_dir",
    "get_plot_filename",
    "get_plot_intent",
    "get_plot_path",
    "is_standard_plot_dir",
    "list_plot_names",
]
