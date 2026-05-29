from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from src.validation.notebook_doctor import run_notebook_doctor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run read-only notebook doctor checks for Colab/session/data readiness."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="StratLake project root to inspect. Defaults to current working directory.",
    )
    parser.add_argument(
        "--marketlake-root",
        help="Optional curated data root to inspect (read-only).",
    )
    parser.add_argument(
        "--drive-root",
        help="Optional Drive root to inspect (read-only).",
    )
    parser.add_argument(
        "--archive-root",
        help="Optional archive pack root to inspect for M43 marker files.",
    )
    parser.add_argument(
        "--archive-destination-root",
        help="Optional restore destination/archive target root to inspect.",
    )
    parser.add_argument(
        "--universe",
        help="Optional universe config path for --check-universe.",
    )
    parser.add_argument(
        "--check-configs",
        action="store_true",
        help="Check readability/parsing of baseline YAML configs.",
    )
    parser.add_argument(
        "--check-universe",
        action="store_true",
        help="Resolve symbols from universe config and ensure non-empty universe.",
    )
    parser.add_argument(
        "--check-drive",
        action="store_true",
        help="Require a readable --drive-root and inspect Colab-like mount shape.",
    )
    parser.add_argument(
        "--check-archives",
        action="store_true",
        help="Inspect archive-root/destination-root for restore safety and marker files.",
    )
    parser.add_argument(
        "--check-marketlake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable MarketLake root checks (default: enabled).",
    )
    parser.add_argument(
        "--check-secrets",
        action="store_true",
        help="Check whether expected secrets are set without printing secret values.",
    )
    parser.add_argument(
        "--secret-name",
        action="append",
        default=[],
        help="Secret env var name to include with --check-secrets (repeatable).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic JSON to stdout.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_notebook_doctor(
        root=args.root,
        marketlake_root=args.marketlake_root,
        drive_root=args.drive_root,
        archive_root=args.archive_root,
        archive_destination_root=args.archive_destination_root,
        universe=args.universe,
        check_configs=args.check_configs,
        check_universe=args.check_universe,
        check_drive=args.check_drive,
        check_archives=args.check_archives,
        check_marketlake=bool(args.check_marketlake),
        check_secrets=args.check_secrets,
        secret_names=args.secret_name,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    _print_summary(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    report = run_cli(argv)
    return 1 if report.get("status") == "fail" else 0


def _print_human(report: dict[str, Any]) -> None:
    print(f"notebook_doctor_status: {report['status']}")
    print(f"root: {report['root']}")
    for check in report.get("checks", []):
        print(f"- {check['name']}: {check['status']} - {check['message']}")


def _print_summary(report: dict[str, Any]) -> None:
    counts = report.get("summary", {}).get("check_counts", {})
    print(f"notebook_doctor_status: {report['status']}", file=sys.stderr)
    print(f"pass_count: {counts.get('pass', 0)}", file=sys.stderr)
    print(f"warn_count: {counts.get('warn', 0)}", file=sys.stderr)
    print(f"fail_count: {counts.get('fail', 0)}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
