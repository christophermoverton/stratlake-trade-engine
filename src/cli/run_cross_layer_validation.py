from __future__ import annotations

import argparse
from typing import Sequence

from src.validation.cross_layer import DEFAULT_SCENARIOS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M28 cross-layer validation across representative entry points."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory. Defaults to current working directory.",
    )
    parser.add_argument(
        "--workdir",
        default="artifacts/qa/m28_cross_layer_validation",
        help="Working directory used for per-layer validation outputs.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/qa/cross_layer_validation_report.json",
        help="JSON output report path.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=DEFAULT_SCENARIOS,
        help="Scenario to run. May be provided more than once. Defaults to the lightweight M28 set.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    from src.execution.validation import run_cross_layer_validation_from_cli_args

    execution_result = run_cross_layer_validation_from_cli_args(args)
    report = execution_result.raw_result
    output_path = execution_result.output_paths["report_json"]
    print(f"cross_layer_validation_status: {report['status']}")
    print(f"scenario_count: {report['scenario_count']}")
    print(f"pass_count: {report['pass_count']}")
    print(f"report_path: {output_path.as_posix()}")
    if report["status"] != "passed":
        raise SystemExit(1)
    return report


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
