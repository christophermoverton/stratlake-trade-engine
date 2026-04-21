from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.validation.milestone_bundle import build_milestone_validation_bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run milestone-grade validation and write a release-traceability bundle."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory. Defaults to current working directory.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="artifacts/qa/milestone_validation_bundle",
        help="Directory where bundle artifacts are written.",
    )
    parser.add_argument(
        "--include-full-pytest",
        action="store_true",
        help="Also run the full pytest suite in addition to the milestone test slice.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    summary = build_milestone_validation_bundle(
        repo_root=Path(args.repo_root),
        bundle_dir=Path(args.bundle_dir),
        include_full_pytest=bool(args.include_full_pytest),
    )
    print(f"milestone_validation_status: {summary['status']}")
    print(f"bundle_dir: {summary['bundle_dir']}")
    print(f"started_at_utc: {summary['started_at_utc']}")
    print(f"finished_at_utc: {summary['finished_at_utc']}")
    if summary["status"] != "passed":
        raise SystemExit(1)
    return summary


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
