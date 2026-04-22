from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_benchmark_pack(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    compare_to: str | Path | None = None,
    stop_after_batches: int | None = None,
) -> ExecutionResult:
    """Run one deterministic benchmark pack through the shared Python API."""

    from src.config.benchmark_pack import load_benchmark_pack_config
    from src.research.benchmark_pack import run_benchmark_pack as run_pack

    config = load_benchmark_pack_config(Path(config_path))
    raw_result = run_pack(
        config,
        output_root=None if output_root is None else Path(output_root),
        compare_to=None if compare_to is None else Path(compare_to),
        stop_after_batches=stop_after_batches,
    )
    return summarize_execution_result(
        workflow="benchmark_pack",
        raw_result=raw_result,
        run_id=raw_result.pack_run_id,
        name=raw_result.pack_run_id,
        artifact_dir=raw_result.output_root,
        manifest_path=raw_result.manifest_path,
        metrics=dict(raw_result.summary),
        output_paths={
            "summary_json": raw_result.summary_path,
            "manifest_json": raw_result.manifest_path,
            "checkpoint_json": raw_result.checkpoint_path,
            "inventory_json": raw_result.inventory_path,
            "benchmark_matrix_summary": raw_result.benchmark_matrix_summary_path,
        },
    )


def run_benchmark_pack_from_cli_args(args) -> ExecutionResult:
    return run_benchmark_pack(
        args.config,
        output_root=args.output_root,
        compare_to=args.compare_to,
        stop_after_batches=args.stop_after_batches,
    )


def run_benchmark_pack_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_benchmark_pack import parse_args

    return run_benchmark_pack_from_cli_args(parse_args(argv))
