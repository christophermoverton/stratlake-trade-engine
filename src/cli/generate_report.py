from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.research.reporting import generate_strategy_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Markdown report generation."""

    parser = argparse.ArgumentParser(
        description="Generate a deterministic Markdown report from an existing strategy run directory."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing strategy run artifact directory under artifacts/strategies/.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional report.md output path override.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> Path:
    """Execute the report-generation CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    report_path = generate_strategy_report(
        run_dir=Path(args.run_dir),
        output_path=None if args.output_path is None else Path(args.output_path),
    )
    print(f"run_dir: {Path(args.run_dir)}")
    print(f"report_path: {report_path}")
    return report_path


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
