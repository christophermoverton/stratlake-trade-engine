from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.config.regime_benchmark_pack import RegimeBenchmarkPackConfigError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic regime benchmark pack."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Regime benchmark-pack YAML/JSON config path.",
    )
    parser.add_argument(
        "--output-root",
        help="Optional output directory override for the regime benchmark pack.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    from src.execution.regime_benchmark import run_regime_benchmark_pack_from_cli_args

    result = run_regime_benchmark_pack_from_cli_args(args).raw_result
    print_summary(result)
    return result


def print_summary(result) -> None:
    print("Regime Benchmark Pack Summary")
    print("-----------------------------")
    print(
        f"Benchmark: {result.benchmark_name} | "
        f"run_id={result.benchmark_run_id} | "
        f"dir={result.output_root.as_posix()}"
    )
    print(
        f"Variants: total={result.variant_count} | "
        f"matrix={result.benchmark_matrix_csv_path.as_posix()}"
    )
    print(
        "Artifacts: "
        f"summary={result.summary_path.as_posix()} | "
        f"manifest={result.manifest_path.as_posix()} | "
        f"stability={result.stability_summary_path.as_posix()} | "
        f"transition={result.transition_summary_path.as_posix()}"
    )


def main() -> None:
    try:
        run_cli()
    except (RegimeBenchmarkPackConfigError, ValueError, FileNotFoundError) as exc:
        print(str(exc).strip() or exc.__class__.__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
