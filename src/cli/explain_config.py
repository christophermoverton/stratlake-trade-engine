from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from src.config.explain import (
    SUPPORTED_EXPLAIN_WORKFLOWS,
    build_runtime_explain_report,
    write_runtime_explain_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain StratLake resolved runtime configuration without executing workflows."
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        choices=("local", "ci", "notebook", "pipeline"),
        help="Supported runtime profile name to explain.",
    )
    profile_group.add_argument(
        "--profile-path",
        help="Explicit runtime profile YAML path to explain.",
    )
    parser.add_argument(
        "--workflow",
        default="generic",
        choices=tuple(sorted(SUPPORTED_EXPLAIN_WORKFLOWS)),
        help="Workflow subject for assumptions included in the explain report.",
    )
    parser.add_argument(
        "--output",
        help="Optional deterministic JSON report path. Prefer artifacts/_derived/config_explain/.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = build_runtime_explain_report(
        profile=args.profile,
        profile_path=args.profile_path,
        workflow=args.workflow,
        output_path=args.output,
    )
    if args.output:
        write_runtime_explain_report(report, args.output)
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
    print(f"runtime_explain_status: {report['status']}", file=sys.stderr)
    print(f"workflow: {report['workflow']}", file=sys.stderr)
    print(f"finding_count: {len(report.get('findings', []))}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
