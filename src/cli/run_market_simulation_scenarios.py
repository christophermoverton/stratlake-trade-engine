from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.config.market_simulation import MarketSimulationConfigError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate deterministic market-simulation scenario configs and write framework artifacts."
    )
    parser.add_argument("--config", required=True, help="Market simulation YAML/JSON config path.")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    from src.execution.market_simulation import run_market_simulation_scenarios_from_cli_args

    result = run_market_simulation_scenarios_from_cli_args(args).raw_result
    print_summary(result)
    return result


def print_summary(result) -> None:
    print("Market Simulation Scenario Framework Summary")
    print("--------------------------------------------")
    print(f"Simulation run id: {result.simulation_run_id}")
    print(f"Output directory: {result.output_dir.as_posix()}")
    print(f"Scenario count: {result.simulation_manifest.get('scenario_count', 0)}")
    print(f"Enabled scenarios: {result.simulation_manifest.get('enabled_scenario_count', 0)}")
    print(f"Disabled scenarios: {result.simulation_manifest.get('disabled_scenario_count', 0)}")
    print(f"scenario_catalog.csv: {result.scenario_catalog_csv_path.as_posix()}")
    print(f"simulation_manifest.json: {result.simulation_manifest_path.as_posix()}")


def main() -> None:
    try:
        run_cli()
    except (MarketSimulationConfigError, ValueError, FileNotFoundError) as exc:
        print(str(exc).strip() or exc.__class__.__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
