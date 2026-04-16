from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.cli.comparison_cli import (
    add_dual_flag_argument,
    optional_output_path,
    parse_csv_or_space_separated,
    print_comparison_summary,
)
from src.config.evaluation import EVALUATION_CONFIG
from src.research.compare import (
    DEFAULT_METRIC,
    ComparisonResult,
    compare_strategies,
    render_leaderboard_table,
)
from src.pipeline.cli_adapter import build_pipeline_cli_result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for multi-strategy comparison."""

    parser = argparse.ArgumentParser(
        description="Compare multiple strategies and generate a deterministic leaderboard."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        required=True,
        help="Strategy names defined in configs/strategies.yml. Accepts comma-separated and/or space-separated values.",
    )
    parser.add_argument(
        "--evaluation",
        nargs="?",
        const=str(EVALUATION_CONFIG),
        help="Enable walk-forward comparison using configs/evaluation.yml or a provided path.",
    )
    parser.add_argument("--start", dest="start", help="Inclusive comparison start date (YYYY-MM-DD).")
    parser.add_argument("--end", dest="end", help="Exclusive comparison end date (YYYY-MM-DD).")
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric used to rank strategies. Defaults to '{DEFAULT_METRIC}'.",
    )
    add_dual_flag_argument(
        parser,
        "--top-k",
        "--top_k",
        dest="top_k",
        type=int,
        help="Limit the leaderboard to the top N strategies.",
    )
    add_dual_flag_argument(
        parser,
        "--from-registry",
        "--from_registry",
        dest="from_registry",
        action="store_true",
        help="Load the latest matching run per strategy from the registry instead of executing runs.",
    )
    add_dual_flag_argument(
        parser,
        "--output-path",
        "--output_path",
        dest="output_path",
        help="Optional leaderboard CSV path or output directory override.",
    )
    return parser.parse_args(argv)


def parse_strategy_names(raw_values: Sequence[str]) -> list[str]:
    """Return normalized strategy names from comma-separated and/or repeated CLI values."""

    return parse_csv_or_space_separated(raw_values) or []


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    state: dict[str, object] | None = None,
    pipeline_context: dict[str, object] | None = None,
) -> ComparisonResult | dict[str, object]:
    """Execute the comparison CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    strategies = parse_strategy_names(args.strategies)
    result = compare_strategies(
        strategies,
        metric=args.metric,
        evaluation_path=None if args.evaluation is None else Path(args.evaluation),
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        from_registry=args.from_registry,
        output_path=optional_output_path(args.output_path),
    )
    print_comparison_summary(
        identifier_label="comparison_id",
        identifier=result.comparison_id,
        row_count=len(result.leaderboard),
        table=render_leaderboard_table(result.leaderboard),
        csv_path=result.csv_path,
        json_path=result.json_path,
        extra_fields=(
            ("metric", result.metric),
            ("evaluation_mode", result.evaluation_mode),
            ("selection_mode", result.selection_mode),
            ("selection_rule", result.selection_rule),
            ("plot_count", len(result.plot_paths)),
        ),
    )
    if pipeline_context is not None:
        return build_pipeline_cli_result(
            identifier=result.comparison_id,
            name="strategy_comparison",
            artifact_dir=result.csv_path.parent,
            output_paths={
                "leaderboard_csv": result.csv_path,
                "summary_json": result.json_path,
            },
            extra={
                "comparison_id": result.comparison_id,
                "metric": result.metric,
                "evaluation_mode": result.evaluation_mode,
                "selection_mode": result.selection_mode,
                "row_count": int(len(result.leaderboard)),
            },
        )
    return result


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
