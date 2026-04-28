from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_regime_promotion_gates(
    benchmark_path: str | Path,
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> ExecutionResult:
    from src.config.regime_promotion_gates import load_regime_promotion_gate_config
    from src.research.regime_promotion_gates import run_regime_promotion_gates as run_gates

    config = load_regime_promotion_gate_config(Path(config_path))
    raw_result = run_gates(
        Path(benchmark_path),
        config,
        output_dir=None if output_dir is None else Path(output_dir),
    )
    return summarize_execution_result(
        workflow="regime_promotion_gates",
        raw_result=raw_result,
        run_id=raw_result.benchmark_run_id,
        name=raw_result.gate_config_name,
        artifact_dir=raw_result.output_dir,
        manifest_path=raw_result.manifest_path,
        metrics=dict(raw_result.decision_summary.get("decision_counts", {})),
        output_paths={
            "gate_config_json": raw_result.gate_config_path,
            "gate_results_csv": raw_result.gate_results_csv_path,
            "gate_results_json": raw_result.gate_results_json_path,
            "decision_summary_json": raw_result.decision_summary_path,
            "failed_gates_csv": raw_result.failed_gates_csv_path,
            "warning_gates_csv": raw_result.warning_gates_csv_path,
            "manifest_json": raw_result.manifest_path,
        },
        extra={
            "decision_policy": raw_result.decision_policy,
            "source_benchmark_path": raw_result.source_benchmark_path,
        },
    )


def run_regime_promotion_gates_from_cli_args(args) -> ExecutionResult:
    return run_regime_promotion_gates(
        args.benchmark_path,
        args.config,
        output_dir=args.output_dir,
    )


def run_regime_promotion_gates_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_regime_promotion_gates import parse_args

    return run_regime_promotion_gates_from_cli_args(parse_args(argv))
