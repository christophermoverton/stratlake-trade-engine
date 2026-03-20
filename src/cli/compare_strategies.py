from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.config.evaluation import EVALUATION_CONFIG
from src.research.compare import (
    DEFAULT_METRIC,
    ComparisonResult,
    compare_strategies,
    render_leaderboard_table,
)


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
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric used to rank strategies. Defaults to '{DEFAULT_METRIC}'.",
    )
    parser.add_argument("--top_k", type=int, help="Limit the leaderboard to the top N strategies.")
    parser.add_argument(
        "--from_registry",
        action="store_true",
        help="Load the latest matching run per strategy from the registry instead of executing runs.",
    )
    parser.add_argument(
        "--output_path",
        help="Optional leaderboard CSV path or output directory override.",
    )
    return parser.parse_args(argv)


def parse_strategy_names(raw_values: Sequence[str]) -> list[str]:
    """Return normalized strategy names from comma-separated and/or repeated CLI values."""

    names: list[str] = []
    for raw_value in raw_values:
        for name in raw_value.split(","):
            normalized = name.strip()
            if normalized:
                names.append(normalized)
    return names


def run_cli(argv: Sequence[str] | None = None) -> ComparisonResult:
    """Execute the comparison CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    strategies = parse_strategy_names(args.strategies)
    result = compare_strategies(
        strategies,
        metric=args.metric,
        evaluation_path=None if args.evaluation is None else Path(args.evaluation),
        top_k=args.top_k,
        from_registry=args.from_registry,
        output_path=None if args.output_path is None else Path(args.output_path),
    )
    print(f"metric: {result.metric}")
    print(f"evaluation_mode: {result.evaluation_mode}")
    print(f"selection_mode: {result.selection_mode}")
    print(f"selection_rule: {result.selection_rule}")
    print(render_leaderboard_table(result.leaderboard))
    print(f"leaderboard_csv: {result.csv_path}")
    print(f"leaderboard_json: {result.json_path}")
    return result


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
