from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.pipeline.pipeline_runner import PipelineRunResult, PipelineRunner, PipelineSpec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for YAML-driven pipeline execution."""

    parser = argparse.ArgumentParser(
        description="Run a deterministic YAML-driven pipeline by invoking module run_cli(argv) steps."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML pipeline spec path.",
    )
    return parser.parse_args(argv)


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    """Load one pipeline specification from YAML."""

    return PipelineSpec.from_yaml(path)


def run_cli(argv: Sequence[str] | None = None) -> PipelineRunResult:
    """Execute one pipeline from CLI arguments."""

    args = parse_args(argv)
    spec = load_pipeline_spec(args.config)
    result = PipelineRunner(spec).run()
    print_summary(result)
    return result


def print_summary(result: PipelineRunResult) -> None:
    """Print a concise pipeline execution summary."""

    print(f"pipeline_id: {result.pipeline_id}")
    print(f"pipeline_run_id: {result.pipeline_run_id}")
    print(f"steps_executed: {len(result.step_results)}")
    print(f"execution_order: {', '.join(result.execution_order)}")


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    try:
        run_cli()
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"Pipeline run failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
