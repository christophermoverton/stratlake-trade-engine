from __future__ import annotations

import argparse
from typing import Sequence


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
    from src.execution.validation import run_docs_path_lint_from_cli_args

    execution_result = run_docs_path_lint_from_cli_args(args)
    report = execution_result.raw_result
    output_path = execution_result.output_paths["report_json"]
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
