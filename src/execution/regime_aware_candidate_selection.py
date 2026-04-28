from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_regime_aware_candidate_selection(
    *,
    config_path: str | Path,
    source_review_pack: str | Path | None = None,
    candidate_metrics_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> ExecutionResult:
    from src.config.regime_aware_candidate_selection import (
        apply_regime_aware_candidate_selection_overrides,
        load_regime_aware_candidate_selection_config,
    )
    from src.research.regime_aware_candidate_selection import run_regime_aware_candidate_selection as _run

    config = load_regime_aware_candidate_selection_config(Path(config_path))
    config = apply_regime_aware_candidate_selection_overrides(
        config,
        source_review_pack=source_review_pack,
        candidate_metrics_path=candidate_metrics_path,
        output_root=output_root,
    )
    raw_result = _run(config)
    return summarize_execution_result(
        workflow="regime_aware_candidate_selection",
        raw_result=raw_result,
        run_id=raw_result.selection_run_id,
        name=raw_result.selection_name,
        artifact_dir=raw_result.output_dir,
        manifest_path=raw_result.manifest_path,
        metrics=dict(raw_result.selection_summary.get("selected_count_by_category", {})),
        output_paths={
            "candidate_selection_csv": raw_result.candidate_selection_csv_path,
            "candidate_selection_json": raw_result.candidate_selection_json_path,
            "regime_candidate_scores_csv": raw_result.regime_candidate_scores_csv_path,
            "regime_candidate_scores_json": raw_result.regime_candidate_scores_json_path,
            "category_assignments_csv": raw_result.category_assignments_path,
            "transition_resilience_csv": raw_result.transition_resilience_path,
            "defensive_fallback_candidates_csv": raw_result.defensive_fallback_candidates_path,
            "redundancy_report_csv": raw_result.redundancy_report_path,
            "allocation_hints_json": raw_result.allocation_hints_path,
            "selection_summary_json": raw_result.selection_summary_path,
            "config_json": raw_result.config_path,
            "manifest_json": raw_result.manifest_path,
        },
        extra={
            "source_review_run_id": raw_result.source_review_run_id,
            "source_benchmark_run_id": raw_result.source_benchmark_run_id,
        },
    )


def run_regime_aware_candidate_selection_from_cli_args(args) -> ExecutionResult:
    return run_regime_aware_candidate_selection(
        config_path=args.config,
        source_review_pack=args.source_review_pack,
        candidate_metrics_path=args.candidate_metrics_path,
        output_root=args.output_root,
    )


def run_regime_aware_candidate_selection_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_regime_aware_candidate_selection import parse_args

    return run_regime_aware_candidate_selection_from_cli_args(parse_args(argv))
