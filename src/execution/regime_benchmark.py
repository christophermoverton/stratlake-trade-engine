from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_regime_benchmark_pack(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> ExecutionResult:
    from src.config.regime_benchmark_pack import load_regime_benchmark_pack_config
    from src.research.regime_benchmark_pack import run_regime_benchmark_pack as run_pack

    config = load_regime_benchmark_pack_config(Path(config_path))
    raw_result = run_pack(
        config,
        output_root=None if output_root is None else Path(output_root),
    )
    return summarize_execution_result(
        workflow="regime_benchmark_pack",
        raw_result=raw_result,
        run_id=raw_result.benchmark_run_id,
        name=raw_result.benchmark_name,
        artifact_dir=raw_result.output_root,
        manifest_path=raw_result.manifest_path,
        metrics=dict(raw_result.summary),
        output_paths={
            "config_json": raw_result.config_path,
            "benchmark_matrix_csv": raw_result.benchmark_matrix_csv_path,
            "benchmark_matrix_json": raw_result.benchmark_matrix_json_path,
            "model_comparison_csv": raw_result.model_comparison_csv_path,
            "calibration_comparison_csv": raw_result.calibration_comparison_csv_path,
            "policy_comparison_csv": raw_result.policy_comparison_csv_path,
            "conditional_performance_summary_json": raw_result.conditional_performance_summary_path,
            "stability_summary_json": raw_result.stability_summary_path,
            "transition_summary_json": raw_result.transition_summary_path,
            "manifest_json": raw_result.manifest_path,
            "benchmark_summary_json": raw_result.summary_path,
            "input_inventory_json": raw_result.input_inventory_path,
            "artifact_provenance_json": raw_result.provenance_path,
        },
        extra={
            "variant_count": raw_result.variant_count,
        },
    )


def run_regime_benchmark_pack_from_cli_args(args) -> ExecutionResult:
    return run_regime_benchmark_pack(
        args.config,
        output_root=args.output_root,
    )


def run_regime_benchmark_pack_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_regime_benchmark_pack import parse_args

    return run_regime_benchmark_pack_from_cli_args(parse_args(argv))
