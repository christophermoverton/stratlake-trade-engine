from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.research.reporting import generate_campaign_milestone_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for campaign milestone report generation."""

    parser = argparse.ArgumentParser(
        description="Generate a deterministic milestone report pack from an existing research campaign artifact directory."
    )
    parser.add_argument(
        "--campaign-artifact-path",
        required=True,
        help="Completed research campaign artifact directory or its summary.json path.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional milestone report output directory or summary.json path override.",
    )
    parser.add_argument(
        "--milestone-name",
        help="Optional milestone name override. Defaults to the campaign run id.",
    )
    parser.add_argument(
        "--title",
        help="Optional report title override.",
    )
    parser.add_argument(
        "--owner",
        help="Optional owner metadata value written into the milestone report summary.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> Path:
    """Execute the milestone report generation CLI flow."""

    args = parse_args(argv)
    summary_path, _decision_log_path, _manifest_path = generate_campaign_milestone_report(
        campaign_artifact_path=Path(args.campaign_artifact_path),
        output_path=None if args.output_path is None else Path(args.output_path),
        milestone_name=args.milestone_name,
        title=args.title,
        owner=args.owner,
    )
    print(f"campaign_artifact_path: {Path(args.campaign_artifact_path)}")
    print(f"milestone_report_path: {summary_path}")
    return summary_path


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
