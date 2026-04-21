from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.validation.deterministic_rerun import (
    run_deterministic_rerun_validation,
    write_deterministic_rerun_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic rerun validation against canonical pipeline examples."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory. Defaults to current working directory.",
    )
    parser.add_argument(
        "--workdir",
        default="artifacts/qa/rerun_workdir",
        help="Working directory used for first/second rerun outputs.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/qa/deterministic_rerun.json",
        help="JSON output report path.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = run_deterministic_rerun_validation(
        repo_root=Path(args.repo_root),
        output_root=Path(args.workdir),
    )
    output_path = write_deterministic_rerun_report(report, Path(args.output))
    print(f"deterministic_rerun_status: {report['status']}")
    print(f"target_count: {report['target_count']}")
    print(f"pass_count: {report['pass_count']}")
    print(f"report_path: {output_path.as_posix()}")
    if report["status"] != "passed":
        raise SystemExit(1)
    return report


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
