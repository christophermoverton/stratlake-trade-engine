from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.cli.comparison_cli import (
    add_dual_flag_argument,
    optional_output_path,
    parse_csv_or_space_separated,
    print_comparison_summary,
    require_registry_mode,
)
from src.config.review import load_review_config
from src.research.promotion import load_promotion_gate_config
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
    add_dual_flag_argument(
        parser,
        "--from-registry",
        "--from_registry",
        dest="from_registry",
        action="store_true",
        help="Load alpha, strategy, and portfolio runs from their registries. Required for unified review.",
    )
    add_dual_flag_argument(
        parser,
        "--run-types",
        "--run_types",
        dest="run_types",
        nargs="+",
        help="Optional run types to include. Supported values: alpha_evaluation, strategy, portfolio.",
    )
    parser.add_argument("--timeframe", help="Optional timeframe filter shared across supported run types.")
    parser.add_argument("--dataset", help="Optional dataset filter for alpha and strategy review rows.")
    add_dual_flag_argument(parser, "--alpha-name", "--alpha_name", dest="alpha_name", help="Optional alpha name filter.")
    add_dual_flag_argument(
        parser,
        "--strategy-name",
        "--strategy_name",
        dest="strategy_name",
        help="Optional strategy name filter.",
    )
    add_dual_flag_argument(
        parser,
        "--portfolio-name",
        "--portfolio_name",
        dest="portfolio_name",
        help="Optional portfolio name filter.",
    )
    add_dual_flag_argument(
        parser,
        "--top-k",
        "--top_k",
        dest="top_k",
        type=int,
        help="Optional top-K limit applied within each run type.",
    )
    add_dual_flag_argument(
        parser,
        "--output-path",
        "--output_path",
        dest="output_path",
        help="Optional leaderboard CSV path or output directory override.",
    )
    add_dual_flag_argument(
        parser,
        "--alpha-artifacts-root",
        "--alpha_artifacts_root",
        dest="alpha_artifacts_root",
        help="Optional alpha artifacts root override.",
    )
    add_dual_flag_argument(
        parser,
        "--strategy-artifacts-root",
        "--strategy_artifacts_root",
        dest="strategy_artifacts_root",
        help="Optional strategy artifacts root override.",
    )
    add_dual_flag_argument(
        parser,
        "--portfolio-artifacts-root",
        "--portfolio_artifacts_root",
        dest="portfolio_artifacts_root",
        help="Optional portfolio artifacts root override.",
    )
    add_dual_flag_argument(
        parser,
        "--review-config",
        "--review_config",
        dest="review_config",
        help="Optional YAML/JSON review config file. CLI flags override file values.",
    )
    parser.add_argument("--alpha-metric", help="Override the primary alpha ranking metric.")
    parser.add_argument("--alpha-secondary-metric", help="Override the secondary alpha ranking metric.")
    parser.add_argument("--strategy-metric", help="Override the primary strategy ranking metric.")
    parser.add_argument("--strategy-secondary-metric", help="Override the secondary strategy ranking metric.")
    parser.add_argument("--portfolio-metric", help="Override the primary portfolio ranking metric.")
    parser.add_argument("--portfolio-secondary-metric", help="Override the secondary portfolio ranking metric.")
    add_dual_flag_argument(
        parser,
        "--promotion-gates",
        "--promotion_gates",
        dest="promotion_gates",
        help="Optional YAML/JSON review-level promotion gate config override.",
    )
    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Disable review comparison plot generation even when config defaults enable it.",
    )
    return parser.parse_args(argv)


def _parse_run_types(raw_values: Sequence[str] | None) -> list[str] | None:
    return parse_csv_or_space_separated(raw_values)


def run_cli(argv: Sequence[str] | None = None) -> ResearchReviewResult:
    """Execute the unified research review CLI flow from parsed arguments."""

    args = parse_args(argv)
    require_registry_mode(args.from_registry, surface_name="Unified research review")

    file_review_config = None if args.review_config is None else load_review_config(Path(args.review_config)).to_dict()
    result = compare_research_runs(
        run_types=_parse_run_types(args.run_types),
        timeframe=args.timeframe,
        dataset=args.dataset,
        alpha_name=args.alpha_name,
        strategy_name=args.strategy_name,
        portfolio_name=args.portfolio_name,
        top_k_per_type=args.top_k,
        alpha_artifacts_root=args.alpha_artifacts_root,
        strategy_artifacts_root=args.strategy_artifacts_root,
        portfolio_artifacts_root=args.portfolio_artifacts_root,
        output_path=optional_output_path(args.output_path),
        review_config=file_review_config,
        alpha_metric=args.alpha_metric,
        alpha_secondary_metric=args.alpha_secondary_metric,
        strategy_metric=args.strategy_metric,
        strategy_secondary_metric=args.strategy_secondary_metric,
        portfolio_metric=args.portfolio_metric,
        portfolio_secondary_metric=args.portfolio_secondary_metric,
        promotion_gate_config=(
            None if args.promotion_gates is None else load_promotion_gate_config(Path(args.promotion_gates))
        ),
        emit_plots=False if args.disable_plots else None,
    )
    print_comparison_summary(
        identifier_label="review_id",
        identifier=result.review_id,
        row_count=len(result.entries),
        table=render_research_review_table(result.entries),
        csv_path=result.csv_path,
        json_path=result.json_path,
        extra_fields=(("filters", result.filters), ("plot_count", len(result.plot_paths))),
    )
    return result


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
