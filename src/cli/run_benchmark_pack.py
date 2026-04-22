from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.config.benchmark_pack import BenchmarkPackConfigError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic scale/reproducibility benchmark pack."
    )
    parser.add_argument("--config", required=True, help="Benchmark-pack YAML/JSON config path.")
    parser.add_argument(
        "--output-root",
        help="Optional output directory override for the benchmark pack.",
    )
    parser.add_argument(
        "--compare-to",
        help="Optional benchmark inventory JSON to compare the current run against.",
    )
    parser.add_argument(
        "--stop-after-batches",
        type=int,
        help="Optional deterministic stop point used to simulate an interrupted benchmark pass.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    from src.execution.benchmark import run_benchmark_pack_from_cli_args

    result = run_benchmark_pack_from_cli_args(args).raw_result
    print_summary(result)
    return result


def print_summary(result) -> None:
    summary = result.summary
    print("Benchmark Pack Summary")
    print("----------------------")
    print(
        f"Pack: {result.pack_run_id} | "
        f"status={summary.get('status', 'unknown')} | "
        f"dir={result.output_root.as_posix()}"
    )
    print(
        f"Batches: total={summary.get('batch_count', 0)} | "
        + " | ".join(
            f"{state}={count}"
            for state, count in dict(summary.get("batch_status_counts", {})).items()
        )
    )
    print(
        f"Scenarios: total={summary.get('scenario_count', 0)} | "
        f"matrix={result.benchmark_matrix_summary_path.as_posix()}"
    )
    print(
        "Artifacts: "
        f"summary={result.summary_path.as_posix()} | "
        f"manifest={result.manifest_path.as_posix()} | "
        f"checkpoint={result.checkpoint_path.as_posix()} | "
        f"inventory={result.inventory_path.as_posix()}"
    )
    comparison = summary.get("comparison")
    if isinstance(comparison, dict):
        print(
            "Comparison: "
            f"matches={comparison.get('matches')} | "
            f"reference={comparison.get('reference_inventory_path')}"
        )


def main() -> None:
    try:
        run_cli()
    except (BenchmarkPackConfigError, ValueError) as exc:
        print(str(exc).strip() or exc.__class__.__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
