from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from src.validation.marketlake_handoff import validate_marketlake_handoff


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a restored MarketLake handoff before StratLake feature builds."
    )
    parser.add_argument("--root", required=True, help="StratLake project root to validate.")
    parser.add_argument(
        "--marketlake-root",
        required=True,
        help="Curated MarketLake root to validate against the selected universe.",
    )
    parser.add_argument(
        "--universe",
        required=True,
        help="Universe YAML that defines the requested symbols or ticker-file reference.",
    )
    parser.add_argument("--start", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="Exclusive end date in YYYY-MM-DD format.")
    parser.add_argument(
        "--timeframe",
        required=True,
        choices=("1D", "1Min"),
        help="Curated timeframe to validate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured report as deterministic JSON to stdout.",
    )
    parser.add_argument(
        "--output",
        help="Optional deterministic JSON report path. Prefer artifacts/_derived/handoff_validation/.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = validate_marketlake_handoff(
        root=args.root,
        marketlake_root=args.marketlake_root,
        universe=args.universe,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        output=args.output,
    )
    _emit_report(report.to_dict(), args.json)
    _print_summary(report.to_dict())
    return report.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    report = run_cli(argv)
    return 1 if report["status"] == "fail" else 0


def _emit_report(report: dict[str, Any], emit_json: bool) -> None:
    if emit_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"status: {report['status']}")
    print(f"validated: {report['validated']}")
    print(f"dataset_name: {report['dataset_name']}")
    print(f"requested_symbols: {len(report['symbols']['requested'])}")
    print(f"missing_symbols: {len(report['symbols']['missing'])}")


def _print_summary(report: dict[str, Any]) -> None:
    print(f"handoff_validation_status: {report['status']}", file=sys.stderr)
    print(f"check_count: {len(report.get('checks', []))}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
