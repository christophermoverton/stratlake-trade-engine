from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.config.regime_promotion_gates import RegimePromotionGateConfigError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic promotion gates for a regime benchmark pack."
    )
    parser.add_argument(
        "--benchmark-path",
        required=True,
        help="Regime benchmark-pack artifact directory containing benchmark_matrix.csv/json.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Regime promotion-gate YAML/JSON config path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory override. Defaults to <benchmark-path>/promotion_gates.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    from src.execution.regime_promotion_gates import run_regime_promotion_gates_from_cli_args

    result = run_regime_promotion_gates_from_cli_args(args).raw_result
    print_summary(result)
    return result


def print_summary(result) -> None:
    counts = result.decision_summary.get("decision_counts", {})
    formatted_counts = ", ".join(
        f"{key}={counts.get(key, 0)}"
        for key in ("accepted", "accepted_with_warnings", "needs_review", "rejected")
    )
    print("Regime Promotion Gate Summary")
    print("-----------------------------")
    print(
        f"Benchmark run id: {result.benchmark_run_id} | "
        f"gate_config={result.gate_config_name}"
    )
    print(f"Output directory: {result.output_dir.as_posix()}")
    print(f"Decision counts: {formatted_counts}")
    print(
        "Artifacts: "
        f"decision_summary={result.decision_summary_path.as_posix()} | "
        f"gate_results={result.gate_results_csv_path.as_posix()}"
    )


def main() -> None:
    try:
        run_cli()
    except (RegimePromotionGateConfigError, ValueError, FileNotFoundError) as exc:
        print(str(exc).strip() or exc.__class__.__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
