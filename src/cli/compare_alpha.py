from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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
    parser.add_argument(
        "--from-registry",
        action="store_true",
        help="Load alpha evaluation runs from the registry. Required for alpha comparison.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_ALPHA_COMPARISON_METRIC,
        help=f"Primary metric used for ranking. Defaults to '{DEFAULT_ALPHA_COMPARISON_METRIC}'.",
    )
    parser.add_argument("--alpha-name", help="Optional alpha name filter.")
    parser.add_argument("--dataset", help="Optional dataset filter.")
    parser.add_argument("--timeframe", help="Optional timeframe filter.")
    parser.add_argument("--evaluation-horizon", type=int, help="Optional evaluation horizon filter.")
    parser.add_argument("--artifacts-root", help="Optional alpha artifacts root override.")
    parser.add_argument("--output-path", help="Optional leaderboard CSV path or output directory override.")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> AlphaEvaluationComparisonResult:
    """Execute the alpha comparison CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    if not args.from_registry:
        raise ValueError("Alpha comparison currently supports registry-backed inputs only. Pass --from-registry.")

    result = compare_alpha_evaluation_runs(
        metric=args.metric,
        alpha_name=args.alpha_name,
        dataset=args.dataset,
        timeframe=args.timeframe,
        evaluation_horizon=args.evaluation_horizon,
        artifacts_root="artifacts/alpha" if args.artifacts_root is None else args.artifacts_root,
        output_path=None if args.output_path is None else Path(args.output_path),
    )
    print(f"comparison_id: {result.comparison_id}")
    print(f"metric: {result.metric}")
    print(f"filters: {result.filters}")
    print(render_alpha_leaderboard_table(result.leaderboard[:10]))
    print(f"leaderboard_csv: {result.csv_path}")
    print(f"leaderboard_json: {result.json_path}")
    return result


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
