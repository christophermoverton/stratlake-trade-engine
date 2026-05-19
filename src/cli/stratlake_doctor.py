from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from src.config.doctor import run_environment_doctor, write_environment_doctor_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StratLake advisory environment-readiness checks."
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        choices=("local", "ci", "notebook", "pipeline"),
        help="Supported runtime profile name to inspect.",
    )
    profile_group.add_argument(
        "--profile-path",
        help="Explicit runtime profile YAML path to inspect.",
    )
    parser.add_argument(
        "--output",
        help="Optional deterministic JSON report path. Prefer artifacts/_derived/environment_readiness/.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = run_environment_doctor(
        profile=args.profile,
        profile_path=args.profile_path,
        output_path=args.output,
    )
    if args.output:
        write_environment_doctor_report(report, args.output)
        _print_summary(report.to_json_dict())
    else:
        print(report.to_json())
        _print_summary(report.to_json_dict())
    if report.status != "passed":
        raise SystemExit(1)
    return report.to_json_dict()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_cli(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1
    return 0


def _print_summary(report: dict[str, Any]) -> None:
    counts = report.get("finding_counts", {})
    print(f"environment_doctor_status: {report['status']}", file=sys.stderr)
    print(f"fail_count: {counts.get('fail', 0)}", file=sys.stderr)
    print(f"warning_count: {counts.get('warning', 0)}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
