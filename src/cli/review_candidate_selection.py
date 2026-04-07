"""CLI entry point for candidate selection review and explainability artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.research.candidate_review import review_candidate_selection

DEFAULT_CANDIDATE_ARTIFACTS_ROOT = Path("artifacts") / "candidate_selection"
DEFAULT_PORTFOLIO_ARTIFACTS_ROOT = Path("artifacts") / "portfolios"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic candidate selection explainability artifacts from existing candidate and portfolio runs."
    )
    parser.add_argument("--candidate-selection-run-id", help="Candidate selection run id under artifacts/candidate_selection/.")
    parser.add_argument("--candidate-selection-path", help="Path to candidate selection artifact directory.")
    parser.add_argument("--portfolio-run-id", help="Portfolio run id under artifacts/portfolios/.")
    parser.add_argument("--portfolio-path", help="Path to portfolio artifact directory.")
    parser.add_argument(
        "--candidate-artifacts-root",
        default=str(DEFAULT_CANDIDATE_ARTIFACTS_ROOT),
        help="Root dir used when resolving --candidate-selection-run-id.",
    )
    parser.add_argument(
        "--portfolio-artifacts-root",
        default=str(DEFAULT_PORTFOLIO_ARTIFACTS_ROOT),
        help="Root dir used when resolving --portfolio-run-id.",
    )
    parser.add_argument("--output-path", help="Optional override output directory for review artifacts.")
    parser.add_argument(
        "--no-markdown-report",
        action="store_true",
        default=False,
        help="Disable generation of candidate_review_report.md.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N rows to include in markdown summary tables.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    candidate_dir = _resolve_candidate_path(
        run_id=args.candidate_selection_run_id,
        path=args.candidate_selection_path,
        root=Path(args.candidate_artifacts_root),
    )
    portfolio_dir = _resolve_portfolio_path(
        run_id=args.portfolio_run_id,
        path=args.portfolio_path,
        root=Path(args.portfolio_artifacts_root),
    )

    result = review_candidate_selection(
        candidate_selection_artifact_dir=candidate_dir,
        portfolio_artifact_dir=portfolio_dir,
        output_dir=None if args.output_path is None else Path(args.output_path),
        include_markdown_report=not bool(args.no_markdown_report),
        top_n=max(1, int(args.top_n)),
    )

    print(f"Candidate Selection Run: {result.candidate_selection_run_id}")
    print(f"Portfolio Run: {result.portfolio_run_id}")
    print(f"Total Candidates: {result.total_candidates}")
    print(f"Selected: {result.selected_candidates}")
    print(f"Rejected: {result.rejected_candidates}")
    print()
    print(f"Review Dir: {result.review_dir}")
    print(f"Candidate Decisions CSV: {result.candidate_decisions_csv}")
    print(f"Candidate Summary CSV: {result.candidate_summary_csv}")
    print(f"Candidate Contributions CSV: {result.candidate_contributions_csv}")
    print(f"Diversification JSON: {result.diversification_summary_json}")
    print(f"Review Summary JSON: {result.candidate_review_summary_json}")
    if result.candidate_review_report_md is not None:
        print(f"Review Report MD: {result.candidate_review_report_md}")
    print(f"Manifest JSON: {result.manifest_json}")
    return result


def _resolve_candidate_path(*, run_id: str | None, path: str | None, root: Path) -> Path:
    has_id = run_id is not None
    has_path = path is not None
    if has_id == has_path:
        raise ValueError("Provide exactly one of --candidate-selection-run-id or --candidate-selection-path.")
    if has_id:
        return root / str(run_id)
    return Path(str(path))


def _resolve_portfolio_path(*, run_id: str | None, path: str | None, root: Path) -> Path:
    has_id = run_id is not None
    has_path = path is not None
    if has_id == has_path:
        raise ValueError("Provide exactly one of --portfolio-run-id or --portfolio-path.")
    if has_id:
        return root / str(run_id)
    return Path(str(path))


def main() -> None:
    try:
        run_cli()
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
