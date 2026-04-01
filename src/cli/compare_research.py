from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.research.review import (
    ResearchReviewResult,
    compare_research_runs,
    render_research_review_table,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for unified registry-backed research review."""

    parser = argparse.ArgumentParser(
        description="Review alpha, strategy, and portfolio runs through one registry-backed comparison surface."
    )
    parser.add_argument(
        "--from-registry",
        action="store_true",
        help="Load alpha, strategy, and portfolio runs from their registries. Required for unified review.",
    )
    parser.add_argument(
        "--run-types",
        nargs="+",
        help="Optional run types to include. Supported values: alpha_evaluation, strategy, portfolio.",
    )
    parser.add_argument("--timeframe", help="Optional timeframe filter shared across supported run types.")
    parser.add_argument("--dataset", help="Optional dataset filter for alpha and strategy review rows.")
    parser.add_argument("--alpha-name", help="Optional alpha name filter.")
    parser.add_argument("--strategy-name", help="Optional strategy name filter.")
    parser.add_argument("--portfolio-name", help="Optional portfolio name filter.")
    parser.add_argument("--top-k", type=int, help="Optional top-K limit applied within each run type.")
    parser.add_argument("--output-path", help="Optional leaderboard CSV path or output directory override.")
    return parser.parse_args(argv)


def _parse_run_types(raw_values: Sequence[str] | None) -> list[str] | None:
    if raw_values is None:
        return None
    run_types: list[str] = []
    for raw_value in raw_values:
        for run_type in raw_value.split(","):
            normalized = run_type.strip()
            if normalized:
                run_types.append(normalized)
    return run_types


def run_cli(argv: Sequence[str] | None = None) -> ResearchReviewResult:
    """Execute the unified research review CLI flow from parsed arguments."""

    args = parse_args(argv)
    if not args.from_registry:
        raise ValueError("Unified research review currently supports registry-backed inputs only. Pass --from-registry.")

    result = compare_research_runs(
        run_types=_parse_run_types(args.run_types),
        timeframe=args.timeframe,
        dataset=args.dataset,
        alpha_name=args.alpha_name,
        strategy_name=args.strategy_name,
        portfolio_name=args.portfolio_name,
        top_k_per_type=args.top_k,
        output_path=None if args.output_path is None else Path(args.output_path),
    )
    print(f"review_id: {result.review_id}")
    print(f"filters: {result.filters}")
    print(render_research_review_table(result.entries))
    print(f"leaderboard_csv: {result.csv_path}")
    print(f"leaderboard_json: {result.json_path}")
    return result


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
