from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_pipeline(config_path: str | Path) -> ExecutionResult:
    """Run one YAML pipeline spec through the shared Python API."""

    from src.pipeline.pipeline_runner import PipelineRunner, PipelineSpec

    spec = PipelineSpec.from_yaml(config_path)
    raw_result = PipelineRunner(spec).run()
    metrics_payload = _read_json(raw_result.pipeline_metrics_path)
    state_payload = _read_json(raw_result.state_path)
    return summarize_execution_result(
        workflow="pipeline",
        raw_result=raw_result,
        run_id=raw_result.pipeline_run_id,
        name=raw_result.pipeline_id,
        artifact_dir=raw_result.artifact_dir,
        manifest_path=raw_result.manifest_path,
        output_paths={
            "manifest_json": raw_result.manifest_path,
            "pipeline_metrics_json": raw_result.pipeline_metrics_path,
            "lineage_json": raw_result.lineage_path,
            "state_json": raw_result.state_path,
        },
        metrics={
            "status": raw_result.status,
            "steps_executed": len(raw_result.step_results),
            "duration_seconds": metrics_payload.get("duration_seconds"),
            "status_counts": metrics_payload.get("status_counts", {}),
            "row_counts": metrics_payload.get("row_counts", {}),
        },
        extra={
            "status": raw_result.status,
            "execution_order": list(raw_result.execution_order),
            "state": state_payload.get("state", {}),
            "state_path": raw_result.state_path.as_posix(),
        },
    )


def run_pipeline_from_cli_args(args) -> ExecutionResult:
    return run_pipeline(args.config)


def run_pipeline_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_pipeline import parse_args

    return run_pipeline_from_cli_args(parse_args(argv))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
