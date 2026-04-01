from __future__ import annotations

import argparse
from typing import Sequence

from src.cli.comparison_cli import (
    add_dual_flag_argument,
    optional_output_path,
    print_comparison_summary,
    require_registry_mode,
)
from src.research.alpha_eval import (
    DEFAULT_ALPHA_COMPARISON_METRIC,
    AlphaEvaluationComparisonResult,
    compare_alpha_evaluation_runs,
    render_alpha_leaderboard_table,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for alpha-evaluation comparison."""

    parser = argparse.ArgumentParser(
        description="Compare registered alpha evaluation runs and generate a deterministic leaderboard."
    )
    add_dual_flag_argument(
        parser,
        "--from-registry",
        "--from_registry",
        dest="from_registry",
        action="store_true",
        help="Load alpha evaluation runs from the registry. Required for alpha comparison.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_ALPHA_COMPARISON_METRIC,
        help=f"Primary metric used for ranking. Defaults to '{DEFAULT_ALPHA_COMPARISON_METRIC}'.",
    )
    add_dual_flag_argument(parser, "--alpha-name", "--alpha_name", dest="alpha_name", help="Optional alpha name filter.")
    parser.add_argument("--dataset", help="Optional dataset filter.")
    parser.add_argument("--timeframe", help="Optional timeframe filter.")
    add_dual_flag_argument(
        parser,
        "--evaluation-horizon",
        "--evaluation_horizon",
        dest="evaluation_horizon",
        type=int,
        help="Optional evaluation horizon filter.",
    )
    parser.add_argument("--artifacts-root", help="Optional alpha artifacts root override.")
    add_dual_flag_argument(
        parser,
        "--output-path",
        "--output_path",
        dest="output_path",
        help="Optional leaderboard CSV path or output directory override.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> AlphaEvaluationComparisonResult:
    """Execute the alpha comparison CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    require_registry_mode(args.from_registry, surface_name="Alpha comparison")

    result = compare_alpha_evaluation_runs(
        metric=args.metric,
        alpha_name=args.alpha_name,
        dataset=args.dataset,
        timeframe=args.timeframe,
        evaluation_horizon=args.evaluation_horizon,
        artifacts_root="artifacts/alpha" if args.artifacts_root is None else args.artifacts_root,
        output_path=optional_output_path(args.output_path),
    )
    print_comparison_summary(
        identifier_label="comparison_id",
        identifier=result.comparison_id,
        row_count=len(result.leaderboard),
        table=render_alpha_leaderboard_table(result.leaderboard),
        csv_path=result.csv_path,
        json_path=result.json_path,
        extra_fields=(("metric", result.metric), ("filters", result.filters)),
    )
    return result


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
