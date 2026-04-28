from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.config.regime_review_pack import RegimeReviewPackConfigError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic regime review-pack artifacts.")
    parser.add_argument("--config", help="Regime review-pack YAML/JSON config path.")
    parser.add_argument("--benchmark-path", help="Regime benchmark-pack artifact directory.")
    parser.add_argument("--promotion-gates-path", help="Promotion-gate artifact directory.")
    parser.add_argument("--output-root", help="Output root for regime review artifacts.")
    parser.add_argument("--review-name", help="Optional review name override for explicit path usage.")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    from src.execution.regime_review_pack import generate_regime_review_pack_from_cli_args

    result = generate_regime_review_pack_from_cli_args(args).raw_result
    print_summary(result)
    return result


def print_summary(result) -> None:
    counts = result.review_summary.get("decision_counts", {})
    formatted_counts = ", ".join(
        f"{key}={counts.get(key, 0)}"
        for key in ("accepted", "accepted_with_warnings", "needs_review", "rejected")
    )
    print("Regime Review Pack Summary")
    print("--------------------------")
    print(
        f"Review run id: {result.review_run_id} | "
        f"benchmark={result.source_benchmark_run_id} | gate_config={result.source_gate_config_name}"
    )
    print(f"Output directory: {result.output_dir.as_posix()}")
    print(f"Decision counts: {formatted_counts}")
    print(
        "Artifacts: "
        f"leaderboard={result.leaderboard_csv_path.as_posix()} | "
        f"decision_log={result.decision_log_path.as_posix()} | "
        f"review_summary={result.review_summary_path.as_posix()}"
    )


def main() -> None:
    try:
        run_cli()
    except (RegimeReviewPackConfigError, ValueError, FileNotFoundError) as exc:
        print(str(exc).strip() or exc.__class__.__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

