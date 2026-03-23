from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.research.compare import ComparisonResult, compare_strategies
from src.research.experiment_tracker import ARTIFACTS_ROOT
from src.research.reporting import generate_strategy_plots, generate_strategy_report
from src.research.visualization import get_plot_filename, plot_equity_comparison, plot_metric_comparison

DEFAULT_STRATEGIES = ("momentum_v1", "mean_reversion_v1", "buy_and_hold_v1")
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-03-01"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "comparisons" / "strategy_comparison_example"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the reproducible strategy comparison example."""

    parser = argparse.ArgumentParser(
        description="Run a bounded end-to-end research example across multiple strategies."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        help="Strategy names to include in the example workflow.",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Inclusive example start date. Defaults to {DEFAULT_START}.",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"Exclusive example end date. Defaults to {DEFAULT_END}.",
    )
    parser.add_argument(
        "--metric",
        default="sharpe_ratio",
        help="Metric used for leaderboard ranking and the comparison bar chart.",
    )
    parser.add_argument(
        "--report-strategy",
        help="Optional strategy name used for report.md generation. Defaults to the first strategy.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used for leaderboard artifacts and comparison plots.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Execute the full comparison example and return a structured summary."""

    args = parse_args(argv)
    strategies = list(dict.fromkeys(args.strategies))
    if len(strategies) < 2:
        raise ValueError("The example workflow requires at least two strategies.")

    comparison_dir = Path(args.output_dir)
    comparison_result = compare_strategies(
        strategies,
        metric=args.metric,
        start=args.start,
        end=args.end,
        output_path=comparison_dir,
    )

    run_dirs = {entry.strategy_name: ARTIFACTS_ROOT / entry.run_id for entry in comparison_result.leaderboard}
    per_run_plots = {
        strategy_name: generate_strategy_plots(run_dir)
        for strategy_name, run_dir in run_dirs.items()
    }

    report_strategy = args.report_strategy or strategies[0]
    if report_strategy not in run_dirs:
        raise ValueError(
            f"Report strategy '{report_strategy}' was not part of the comparison result: {sorted(run_dirs)}."
        )
    report_path = generate_strategy_report(run_dirs[report_strategy])

    comparison_plot_paths = _generate_comparison_plots(
        comparison_result=comparison_result,
        run_dirs=run_dirs,
        comparison_dir=comparison_dir,
    )

    summary = {
        "strategies": strategies,
        "start": args.start,
        "end": args.end,
        "metric": args.metric,
        "leaderboard_csv": comparison_result.csv_path.as_posix(),
        "leaderboard_json": comparison_result.json_path.as_posix(),
        "report_strategy": report_strategy,
        "report_path": report_path.as_posix(),
        "run_dirs": {name: path.as_posix() for name, path in run_dirs.items()},
        "run_plots": {
            name: {plot_name: plot_path.as_posix() for plot_name, plot_path in plots.items()}
            for name, plots in per_run_plots.items()
        },
        "comparison_plots": {name: path.as_posix() for name, path in comparison_plot_paths.items()},
    }
    summary_path = comparison_dir / "example_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"start: {args.start}")
    print(f"end: {args.end}")
    print(f"strategies: {', '.join(strategies)}")
    print(f"leaderboard_csv: {comparison_result.csv_path}")
    print(f"leaderboard_json: {comparison_result.json_path}")
    print(f"report_strategy: {report_strategy}")
    print(f"report_path: {report_path}")
    for strategy_name in strategies:
        print(f"run_dir[{strategy_name}]: {run_dirs[strategy_name]}")
    for plot_name, plot_path in sorted(comparison_plot_paths.items()):
        print(f"{plot_name}: {plot_path}")
    print(f"summary_path: {summary_path}")

    return summary


def _generate_comparison_plots(
    *,
    comparison_result: ComparisonResult,
    run_dirs: dict[str, Path],
    comparison_dir: Path,
) -> dict[str, Path]:
    plots_dir = comparison_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    equity_inputs: list[tuple[str, pd.Series]] = []
    metric_rows: list[dict[str, object]] = []
    metric_name = comparison_result.metric

    for entry in comparison_result.leaderboard:
        run_dir = run_dirs[entry.strategy_name]
        equity_curve = pd.read_csv(run_dir / "equity_curve.csv")
        equity_series = _select_equity_series(equity_curve)
        equity_inputs.append((entry.strategy_name, equity_series))
        metric_rows.append(
            {
                "strategy_name": entry.strategy_name,
                metric_name: entry.selected_metric_value,
            }
        )

    equity_plot_path = plots_dir / get_plot_filename("equity_comparison")
    plot_equity_comparison(
        equity_inputs,
        title="Strategy Equity Comparison",
        output_path=equity_plot_path,
        input_type="equity",
    )

    metric_plot_path = plots_dir / get_plot_filename("metric_comparison", metric_name=metric_name)
    plot_metric_comparison(
        metric_rows,
        metric_name=metric_name,
        title=f"Strategy {metric_name} Comparison",
        output_path=metric_plot_path,
    )

    return {
        "equity_comparison": equity_plot_path,
        "metric_comparison": metric_plot_path,
    }


def _select_equity_series(frame: pd.DataFrame) -> pd.Series:
    column = "equity" if "equity" in frame.columns else "equity_curve"
    index = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64", na_value=float("nan"))
    series = pd.Series(values, index=index, name=column)
    return series.dropna()


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
