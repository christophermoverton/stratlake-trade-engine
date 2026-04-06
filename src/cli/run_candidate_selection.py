"""CLI entry point for candidate selection workflow."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.research.candidate_selection import run_candidate_selection

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "candidate_selection"


@dataclass(frozen=True)
class CandidateSelectionRunResult:
    """Structured result returned from candidate selection CLI run."""

    run_id: str
    universe_count: int
    selected_count: int
    primary_metric: str
    filters: dict[str, Any]
    universe_csv: Path
    selected_csv: Path
    summary_json: Path
    manifest_json: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for candidate selection."""

    parser = argparse.ArgumentParser(
        description="Run a deterministic candidate selection workflow to select alpha candidates for allocation."
    )
    parser.add_argument(
        "--artifacts-root",
        help="Optional root directory of alpha evaluation artifacts. Defaults to artifacts/alpha.",
    )
    parser.add_argument("--alpha-name", help="Optional filter by alpha name.")
    parser.add_argument("--dataset", help="Optional filter by dataset.")
    parser.add_argument("--timeframe", help="Optional filter by timeframe.")
    parser.add_argument(
        "--evaluation-horizon",
        type=int,
        help="Optional filter by evaluation horizon (in bars).",
    )
    parser.add_argument("--mapping-name", help="Optional filter by signal mapping name.")
    parser.add_argument(
        "--metric",
        default="ic_ir",
        choices=("ic_ir", "mean_ic", "mean_rank_ic", "rank_ic_ir"),
        help="Primary ranking metric. Defaults to 'ic_ir'.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Optional maximum number of candidates to select. Defaults to all.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional candidate selection artifacts root. Defaults to artifacts/candidate_selection.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> CandidateSelectionRunResult:
    """Execute the candidate selection CLI flow from parsed command-line arguments."""

    args = parse_args(argv)

    artifacts_root = Path(args.artifacts_root) if args.artifacts_root else Path("artifacts") / "alpha"
    output_path = Path(args.output_path) if args.output_path else DEFAULT_ARTIFACTS_ROOT

    result = run_candidate_selection(
        artifacts_root=artifacts_root,
        alpha_name=args.alpha_name,
        dataset=args.dataset,
        timeframe=args.timeframe,
        evaluation_horizon=args.evaluation_horizon,
        mapping_name=args.mapping_name,
        primary_metric=args.metric,
        max_candidate_count=args.max_candidates,
        output_artifacts_root=output_path,
    )

    return CandidateSelectionRunResult(
        run_id=result["run_id"],
        universe_count=result["universe_count"],
        selected_count=result["selected_count"],
        primary_metric=result["primary_metric"],
        filters=result["filters"],
        universe_csv=Path(result["universe_csv"]),
        selected_csv=Path(result["selected_csv"]),
        summary_json=Path(result["summary_json"]),
        manifest_json=Path(result["manifest_json"]),
    )


def print_candidate_selection_summary(result: CandidateSelectionRunResult) -> None:
    """Print a concise CLI summary of candidate selection results."""

    print(f"Candidate Selection Run: {result.run_id}")
    print(f"  Universe count: {result.universe_count}")
    print(f"  Selected count: {result.selected_count}")
    print(f"  Primary metric: {result.primary_metric}")
    print()
    print(f"  Universe CSV: {result.universe_csv}")
    print(f"  Selected CSV: {result.selected_csv}")
    print(f"  Summary JSON: {result.summary_json}")
    print()
    if result.filters.get("alpha_name"):
        print(f"  Filter alpha_name: {result.filters['alpha_name']}")
    if result.filters.get("dataset"):
        print(f"  Filter dataset: {result.filters['dataset']}")
    if result.filters.get("timeframe"):
        print(f"  Filter timeframe: {result.filters['timeframe']}")
    if result.filters.get("evaluation_horizon") is not None:
        print(f"  Filter evaluation_horizon: {result.filters['evaluation_horizon']}")
    if result.filters.get("mapping_name"):
        print(f"  Filter mapping_name: {result.filters['mapping_name']}")


if __name__ == "__main__":
    result = run_cli()
    print_candidate_selection_summary(result)
