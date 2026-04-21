from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.validation.docs_path_lint import lint_guarded_surfaces, write_docs_path_lint_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint guarded release-facing docs/examples for absolute-path leakage."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory. Defaults to current working directory.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/qa/docs_path_lint.json",
        help="JSON output report path.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = lint_guarded_surfaces(repo_root=Path(args.repo_root))
    output_path = write_docs_path_lint_report(report, Path(args.output))
    print(f"docs_path_lint_status: {report['status']}")
    print(f"guarded_file_count: {report['guarded_file_count']}")
    print(f"finding_count: {report['finding_count']}")
    print(f"report_path: {output_path.as_posix()}")
    if report["status"] != "passed":
        raise SystemExit(1)
    return report


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
