from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.config.regime_aware_candidate_selection import RegimeAwareCandidateSelectionConfigError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic regime-aware candidate selection."
    )
    parser.add_argument("--config", required=True, help="Regime-aware candidate-selection YAML/JSON config path.")
    parser.add_argument("--source-review-pack", help="Optional source review-pack directory override.")
    parser.add_argument("--candidate-metrics-path", help="Optional candidate metrics CSV/JSON override.")
    parser.add_argument("--output-root", help="Optional candidate-selection artifact root override.")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    from src.execution.regime_aware_candidate_selection import (
        run_regime_aware_candidate_selection_from_cli_args,
    )

    result = run_regime_aware_candidate_selection_from_cli_args(args).raw_result
    print_summary(result)
    return result


def print_summary(result) -> None:
    counts = result.selection_summary.get("selected_count_by_category", {})
    formatted_counts = ", ".join(f"{key}={counts.get(key, 0)}" for key in sorted(counts))
    print("Regime-Aware Candidate Selection Summary")
    print("----------------------------------------")
    print(f"Selection run id: {result.selection_run_id}")
    print(f"Output directory: {result.output_dir.as_posix()}")
    print(f"Selected candidates: {result.selection_summary.get('selected_count', 0)}")
    print(f"Selected by category: {formatted_counts}")
    print(f"candidate_selection.csv: {result.candidate_selection_csv_path.as_posix()}")
    print(f"selection_summary.json: {result.selection_summary_path.as_posix()}")


def main() -> None:
    try:
        run_cli()
    except (RegimeAwareCandidateSelectionConfigError, ValueError, FileNotFoundError) as exc:
        print(str(exc).strip() or exc.__class__.__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
