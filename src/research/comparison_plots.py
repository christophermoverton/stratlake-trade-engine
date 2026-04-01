from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.research.visualization import get_plot_dir, get_plot_filename, plot_equity_comparison, plot_metric_comparison

if TYPE_CHECKING:
    from src.research.compare import ComparisonResult
    from src.research.review import ResearchReviewEntry

_MAX_EQUITY_COMPARISON_LINES = 6
_MAX_METRIC_COMPARISON_BARS = 10


def generate_strategy_comparison_plots(
    *,
    comparison_result: ComparisonResult,
    run_dirs_by_run_id: Mapping[str, Path],
    comparison_dir: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Write the bounded plot set used by the primary strategy comparison workflow."""

    plots_dir = get_plot_dir(comparison_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}
    skipped_plots: dict[str, str] = {}

    metric_count = len(comparison_result.leaderboard)
    if metric_count >= 2 and metric_count <= _MAX_METRIC_COMPARISON_BARS:
        metric_name = comparison_result.metric
        metric_plot_path = plots_dir / get_plot_filename("metric_comparison", metric_name=metric_name)
        plot_metric_comparison(
            [
                {
                    "strategy_name": entry.strategy_name,
                    metric_name: entry.selected_metric_value,
                }
                for entry in comparison_result.leaderboard
            ],
            metric_name=metric_name,
            title=f"Strategy {metric_name} Comparison",
            output_path=metric_plot_path,
        )
        plot_paths["metric_comparison"] = metric_plot_path
    else:
        skipped_plots["metric_comparison"] = _bounded_plot_reason(
            label="metric comparison",
            item_count=metric_count,
            max_items=_MAX_METRIC_COMPARISON_BARS,
        )

    equity_count = len(comparison_result.leaderboard)
    if equity_count >= 2 and equity_count <= _MAX_EQUITY_COMPARISON_LINES:
        equity_inputs: list[tuple[str, pd.Series]] = []
        for entry in comparison_result.leaderboard:
            run_dir = run_dirs_by_run_id.get(entry.run_id)
            if run_dir is None:
                skipped_plots["equity_comparison"] = f"Skipped because run directory metadata was missing for '{entry.run_id}'."
                equity_inputs = []
                break
            equity_csv = run_dir / "equity_curve.csv"
            if not equity_csv.exists():
                skipped_plots["equity_comparison"] = f"Skipped because '{equity_csv.as_posix()}' was not found."
                equity_inputs = []
                break
            equity_inputs.append((entry.strategy_name, _load_equity_series(equity_csv)))

        if equity_inputs:
            equity_plot_path = plots_dir / get_plot_filename("equity_comparison")
            plot_equity_comparison(
                equity_inputs,
                title="Strategy Equity Comparison",
                output_path=equity_plot_path,
                input_type="equity",
            )
            plot_paths["equity_comparison"] = equity_plot_path
    elif "equity_comparison" not in skipped_plots:
        skipped_plots["equity_comparison"] = _bounded_plot_reason(
            label="equity comparison",
            item_count=equity_count,
            max_items=_MAX_EQUITY_COMPARISON_LINES,
        )

    return plot_paths, skipped_plots


def generate_research_review_plots(
    *,
    entries: Sequence[ResearchReviewEntry],
    review_dir: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Write bounded per-run-type metric plots for the unified review workflow."""

    plots_dir = get_plot_dir(review_dir)
    plot_paths: dict[str, Path] = {}
    skipped_plots: dict[str, str] = {}
    grouped: dict[str, list[ResearchReviewEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.run_type].append(entry)

    for run_type, run_entries in sorted(grouped.items()):
        plot_key = f"{run_type}_metric_comparison"
        if len(run_entries) < 2 or len(run_entries) > _MAX_METRIC_COMPARISON_BARS:
            skipped_plots[plot_key] = _bounded_plot_reason(
                label=f"{run_type} metric comparison",
                item_count=len(run_entries),
                max_items=_MAX_METRIC_COMPARISON_BARS,
            )
            continue

        metric_name = run_entries[0].selected_metric_name
        run_type_dir = plots_dir / run_type
        run_type_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_type_dir / get_plot_filename("metric_comparison", metric_name=metric_name)
        plot_metric_comparison(
            [
                {
                    "strategy": entry.entity_name,
                    metric_name: entry.selected_metric_value,
                }
                for entry in run_entries
            ],
            metric_name=metric_name,
            title=f"{run_type.replace('_', ' ').title()} {metric_name} Comparison",
            output_path=output_path,
        )
        plot_paths[plot_key] = output_path

    return plot_paths, skipped_plots


def _load_equity_series(equity_csv_path: Path) -> pd.Series:
    frame = pd.read_csv(equity_csv_path)
    column = "equity" if "equity" in frame.columns else "equity_curve"
    index = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64", na_value=float("nan"))
    series = pd.Series(values, index=index, name=column)
    if series.index.has_duplicates:
        series = series.groupby(level=0, sort=True).last()
    return series.dropna()


def _bounded_plot_reason(*, label: str, item_count: int, max_items: int) -> str:
    if item_count < 2:
        return f"Skipped {label} because at least 2 rows are required."
    return f"Skipped {label} because {item_count} rows exceeds the readability limit of {max_items}."
