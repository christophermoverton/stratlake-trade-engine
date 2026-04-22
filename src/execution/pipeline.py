from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_pipeline(config_path: str | Path) -> ExecutionResult:
    """Run one YAML pipeline spec through the shared Python API."""

    from src.pipeline.pipeline_runner import PipelineRunner, PipelineSpec

    spec = PipelineSpec.from_yaml(config_path)
    raw_result = PipelineRunner(spec).run()
    return summarize_execution_result(
        workflow="pipeline",
        raw_result=raw_result,
        run_id=raw_result.pipeline_run_id,
        name=raw_result.pipeline_id,
        extra={
            "steps_executed": len(raw_result.step_results),
            "execution_order": list(raw_result.execution_order),
        },
    )


def run_pipeline_from_cli_args(args) -> ExecutionResult:
    return run_pipeline(args.config)


def run_pipeline_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_pipeline import parse_args

    return run_pipeline_from_cli_args(parse_args(argv))
