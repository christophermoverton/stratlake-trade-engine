from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_market_simulation_scenarios(
    *,
    config_path: str | Path,
) -> ExecutionResult:
    from src.config.market_simulation import load_market_simulation_config
    from src.research.market_simulation.artifacts import run_market_simulation_framework

    config = load_market_simulation_config(Path(config_path))
    raw_result = run_market_simulation_framework(config, config_path=config_path)
    historical_outputs = {
        f"historical_episode_replay_{index}_{name}": path
        for index, replay in enumerate(raw_result.historical_episode_replay_results, start=1)
        for name, path in (
            ("catalog_csv", replay.historical_episode_catalog_path),
            ("replay_results_csv", replay.episode_replay_results_path),
            ("policy_comparison_csv", replay.episode_policy_comparison_path),
            ("summary_json", replay.episode_summary_path),
            ("manifest_json", replay.manifest_path),
        )
    }
    shock_outputs = {
        f"shock_overlay_{index}_{name}": path
        for index, overlay in enumerate(raw_result.shock_overlay_results, start=1)
        for name, path in (
            ("config_json", overlay.shock_overlay_config_path),
            ("results_csv", overlay.shock_overlay_results_path),
            ("log_csv", overlay.shock_overlay_log_path),
            ("summary_json", overlay.shock_overlay_summary_path),
            ("manifest_json", overlay.manifest_path),
        )
    }
    bootstrap_outputs = {
        f"regime_block_bootstrap_{index}_{name}": path
        for index, bootstrap in enumerate(raw_result.block_bootstrap_results, start=1)
        for name, path in (
            ("config_json", bootstrap.bootstrap_config_path),
            ("source_block_catalog_csv", bootstrap.source_block_catalog_path),
            ("sampled_block_inventory_csv", bootstrap.sampled_block_inventory_path),
            ("path_catalog_csv", bootstrap.bootstrap_path_catalog_path),
            ("simulated_return_paths_parquet", bootstrap.simulated_return_paths_path),
            ("simulated_regime_paths_parquet", bootstrap.simulated_regime_paths_path),
            ("summary_json", bootstrap.bootstrap_sampling_summary_path),
            ("manifest_json", bootstrap.manifest_path),
        )
    }
    monte_carlo_outputs = {
        f"regime_transition_monte_carlo_{index}_{name}": path
        for index, monte_carlo in enumerate(raw_result.monte_carlo_results, start=1)
        for name, path in (
            ("transition_matrix_json", monte_carlo.transition_matrix_path),
            ("regime_paths_parquet", monte_carlo.regime_paths_path),
            ("path_catalog_csv", monte_carlo.path_catalog_path),
            ("summary_json", monte_carlo.summary_path),
            ("manifest_json", monte_carlo.manifest_path),
        )
    }
    monte_carlo_optional_outputs = {
        f"regime_transition_monte_carlo_{index}_{name}": path
        for index, monte_carlo in enumerate(raw_result.monte_carlo_results, start=1)
        for name, path in (
            ("adjusted_transition_matrix_json", monte_carlo.adjusted_transition_matrix_path),
            ("transition_counts_csv", monte_carlo.transition_counts_path),
        )
        if path is not None
    }
    metrics_result = raw_result.simulation_stress_metrics_result
    metrics_outputs = (
        {}
        if metrics_result is None
        else {
            "simulation_metrics_path_metrics_csv": metrics_result.path_metrics_path,
            "simulation_metrics_summary_csv": metrics_result.summary_path,
            "simulation_metrics_leaderboard_csv": metrics_result.leaderboard_path,
            "simulation_metrics_policy_failure_summary_json": metrics_result.policy_failure_summary_path,
            "simulation_metrics_config_json": metrics_result.metric_config_path,
            "simulation_metrics_manifest_json": metrics_result.manifest_path,
        }
    )
    return summarize_execution_result(
        workflow="market_simulation_scenarios",
        raw_result=raw_result,
        run_id=raw_result.simulation_run_id,
        name=raw_result.simulation_name,
        artifact_dir=raw_result.output_dir,
        manifest_path=raw_result.simulation_manifest_path,
        metrics={
            "scenario_count": raw_result.simulation_manifest.get("scenario_count"),
            "enabled_scenario_count": raw_result.simulation_manifest.get("enabled_scenario_count"),
            "disabled_scenario_count": raw_result.simulation_manifest.get("disabled_scenario_count"),
            "historical_episode_replay_count": len(raw_result.historical_episode_replay_results),
            "historical_episode_count": sum(
                replay.episode_count for replay in raw_result.historical_episode_replay_results
            ),
            "historical_episode_replayed_rows": sum(
                replay.replayed_row_count for replay in raw_result.historical_episode_replay_results
            ),
            "shock_overlay_count": len(raw_result.shock_overlay_results),
            "shock_overlay_rows": sum(overlay.row_count for overlay in raw_result.shock_overlay_results),
            "regime_block_bootstrap_count": len(raw_result.block_bootstrap_results),
            "regime_block_bootstrap_paths": sum(
                bootstrap.path_count for bootstrap in raw_result.block_bootstrap_results
            ),
            "regime_block_bootstrap_rows": sum(
                bootstrap.simulated_row_count for bootstrap in raw_result.block_bootstrap_results
            ),
            "regime_transition_monte_carlo_count": len(raw_result.monte_carlo_results),
            "regime_transition_monte_carlo_paths": sum(
                monte_carlo.path_count for monte_carlo in raw_result.monte_carlo_results
            ),
            "regime_transition_monte_carlo_rows": sum(
                monte_carlo.generated_row_count for monte_carlo in raw_result.monte_carlo_results
            ),
            "simulation_metrics_path_rows": 0
            if metrics_result is None
            else metrics_result.path_metric_row_count,
            "simulation_metrics_summary_rows": 0 if metrics_result is None else metrics_result.summary_row_count,
            "simulation_metrics_leaderboard_rows": 0
            if metrics_result is None
            else metrics_result.leaderboard_row_count,
            "simulation_metrics_policy_failure_rate": 0.0
            if metrics_result is None
            else metrics_result.policy_failure_rate,
        },
        output_paths={
            "scenario_catalog_csv": raw_result.scenario_catalog_csv_path,
            "scenario_catalog_json": raw_result.scenario_catalog_json_path,
            "simulation_config_json": raw_result.simulation_config_path,
            "input_inventory_json": raw_result.input_inventory_path,
            "simulation_manifest_json": raw_result.simulation_manifest_path,
            **historical_outputs,
            **bootstrap_outputs,
            **monte_carlo_outputs,
            **monte_carlo_optional_outputs,
            **shock_outputs,
            **metrics_outputs,
        },
    )


def run_market_simulation_scenarios_from_cli_args(args) -> ExecutionResult:
    return run_market_simulation_scenarios(config_path=args.config)


def run_market_simulation_scenarios_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_market_simulation_scenarios import parse_args

    return run_market_simulation_scenarios_from_cli_args(parse_args(argv))
