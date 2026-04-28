from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_regime_policy_stress_tests(
    *,
    config_path: str | Path,
    source_review_pack: str | Path | None = None,
    policy_metrics_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> ExecutionResult:
    from src.config.regime_policy_stress_tests import (
        apply_regime_policy_stress_test_overrides,
        load_regime_policy_stress_test_config,
    )
    from src.research.regime_policy_stress_tests import run_regime_policy_stress_tests as _run

    config = load_regime_policy_stress_test_config(Path(config_path))
    config = apply_regime_policy_stress_test_overrides(
        config,
        source_review_pack=source_review_pack,
        policy_metrics_path=policy_metrics_path,
        output_root=output_root,
    )
    raw_result = _run(config)
    return summarize_execution_result(
        workflow="regime_policy_stress_tests",
        raw_result=raw_result,
        run_id=raw_result.stress_run_id,
        name=raw_result.stress_test_name,
        artifact_dir=raw_result.output_dir,
        manifest_path=raw_result.manifest_path,
        metrics={
            "scenario_count": raw_result.scenario_summary.get("scenario_count"),
            "policy_count": raw_result.scenario_summary.get("policy_count"),
            "stress_pass_count": raw_result.manifest.get("stress_pass_count"),
            "stress_fail_count": raw_result.manifest.get("stress_fail_count"),
        },
        output_paths={
            "stress_matrix_csv": raw_result.stress_matrix_csv_path,
            "stress_matrix_json": raw_result.stress_matrix_json_path,
            "stress_leaderboard_csv": raw_result.stress_leaderboard_csv_path,
            "scenario_summary_json": raw_result.scenario_summary_path,
            "policy_stress_summary_json": raw_result.policy_stress_summary_path,
            "scenario_catalog_json": raw_result.scenario_catalog_path,
            "scenario_results_csv": raw_result.scenario_results_csv_path,
            "fallback_usage_csv": raw_result.fallback_usage_csv_path,
            "policy_turnover_under_stress_csv": raw_result.policy_turnover_csv_path,
            "adaptive_vs_static_stress_comparison_csv": raw_result.adaptive_vs_static_comparison_csv_path,
            "config_json": raw_result.config_path,
            "manifest_json": raw_result.manifest_path,
        },
        extra={
            "baseline_policy": raw_result.scenario_summary.get("baseline_policy"),
            "worst_scenario_type": raw_result.scenario_summary.get("worst_scenario_type"),
            "most_resilient_policy": raw_result.policy_stress_summary.get("most_resilient_policy"),
        },
    )


def run_regime_policy_stress_tests_from_cli_args(args) -> ExecutionResult:
    return run_regime_policy_stress_tests(
        config_path=args.config,
        source_review_pack=args.source_review_pack,
        policy_metrics_path=args.policy_metrics_path,
        output_root=args.output_root,
    )


def run_regime_policy_stress_tests_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_regime_policy_stress_tests import parse_args

    return run_regime_policy_stress_tests_from_cli_args(parse_args(argv))
