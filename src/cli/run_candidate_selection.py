"""CLI entry point for candidate selection workflow."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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
    eligible_count: int
    rejected_count: int
    selected_count: int
    primary_metric: str
    filters: dict[str, Any]
    thresholds: dict[str, Any]
    universe_csv: Path
    selected_csv: Path
    rejected_csv: Path
    eligibility_csv: Path
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
        help="Optional maximum number of candidates to select after eligibility gating. Defaults to all.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional candidate selection artifacts root. Defaults to artifacts/candidate_selection.",
    )

    # Eligibility gate threshold arguments
    gate_group = parser.add_argument_group(
        "eligibility gates",
        "Configurable quality thresholds for candidate eligibility filtering. "
        "Omit any threshold to disable that gate check.",
    )
    gate_group.add_argument(
        "--min-ic",
        type=float,
        dest="min_mean_ic",
        help="Minimum mean IC threshold. Candidates below this value are rejected.",
    )
    gate_group.add_argument(
        "--min-rank-ic",
        type=float,
        dest="min_mean_rank_ic",
        help="Minimum mean Rank IC threshold.",
    )
    gate_group.add_argument(
        "--min-ic-ir",
        type=float,
        dest="min_ic_ir",
        help="Minimum IC information ratio threshold.",
    )
    gate_group.add_argument(
        "--min-rank-ic-ir",
        type=float,
        dest="min_rank_ic_ir",
        help="Minimum Rank IC information ratio threshold.",
    )
    gate_group.add_argument(
        "--min-history-length",
        type=int,
        dest="min_history_length",
        help="Minimum number of evaluation periods (observation count).",
    )
    gate_group.add_argument(
        "--require-mean-ic",
        action="store_true",
        default=False,
        dest="require_mean_ic",
        help="Reject candidates whose mean_ic metric is missing.",
    )
    gate_group.add_argument(
        "--require-ic-ir",
        action="store_true",
        default=False,
        dest="require_ic_ir",
        help="Reject candidates whose ic_ir metric is missing.",
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
        min_mean_ic=args.min_mean_ic,
        min_mean_rank_ic=args.min_mean_rank_ic,
        min_ic_ir=args.min_ic_ir,
        min_rank_ic_ir=args.min_rank_ic_ir,
        min_history_length=args.min_history_length,
        require_mean_ic=args.require_mean_ic,
        require_ic_ir=args.require_ic_ir,
    )

    return CandidateSelectionRunResult(
        run_id=result["run_id"],
        universe_count=result["universe_count"],
        eligible_count=result["eligible_count"],
        rejected_count=result["rejected_count"],
        selected_count=result["selected_count"],
        primary_metric=result["primary_metric"],
        filters=result["filters"],
        thresholds=result["thresholds"],
        universe_csv=Path(result["universe_csv"]),
        selected_csv=Path(result["selected_csv"]),
        rejected_csv=Path(result["rejected_csv"]),
        eligibility_csv=Path(result["eligibility_csv"]),
        summary_json=Path(result["summary_json"]),
        manifest_json=Path(result["manifest_json"]),
    )


def print_candidate_selection_summary(result: CandidateSelectionRunResult) -> None:
    """Print a concise CLI summary of candidate selection results."""

    print(f"Candidate Selection Run: {result.run_id}")
    print(f"  Universe count:  {result.universe_count}")
    print(f"  Eligible count:  {result.eligible_count}")
    print(f"  Rejected count:  {result.rejected_count}")
    print(f"  Selected count:  {result.selected_count}")
    print(f"  Primary metric:  {result.primary_metric}")
    print()
    print(f"  Universe CSV:    {result.universe_csv}")
    print(f"  Eligibility CSV: {result.eligibility_csv}")
    print(f"  Selected CSV:    {result.selected_csv}")
    print(f"  Rejected CSV:    {result.rejected_csv}")
    print(f"  Summary JSON:    {result.summary_json}")
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
    print()
    active_thresholds = {k: v for k, v in result.thresholds.items() if v is not None and v is not False}
    if active_thresholds:
        print("  Active eligibility thresholds:")
        for k, v in sorted(active_thresholds.items()):
            print(f"    {k}: {v}")


if __name__ == "__main__":
    result = run_cli()
    print_candidate_selection_summary(result)
