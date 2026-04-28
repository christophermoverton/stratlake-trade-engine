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
        },
        output_paths={
            "scenario_catalog_csv": raw_result.scenario_catalog_csv_path,
            "scenario_catalog_json": raw_result.scenario_catalog_json_path,
            "simulation_config_json": raw_result.simulation_config_path,
            "input_inventory_json": raw_result.input_inventory_path,
            "simulation_manifest_json": raw_result.simulation_manifest_path,
        },
    )


def run_market_simulation_scenarios_from_cli_args(args) -> ExecutionResult:
    return run_market_simulation_scenarios(config_path=args.config)


def run_market_simulation_scenarios_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_market_simulation_scenarios import parse_args

    return run_market_simulation_scenarios_from_cli_args(parse_args(argv))
