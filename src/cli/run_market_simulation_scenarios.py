from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
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
    print(f"Output directory: {_display_path(result.output_dir)}")
    print(f"Scenario count: {result.simulation_manifest.get('scenario_count', 0)}")
    print(f"Enabled scenarios: {result.simulation_manifest.get('enabled_scenario_count', 0)}")
    print(f"Disabled scenarios: {result.simulation_manifest.get('disabled_scenario_count', 0)}")
    print(f"scenario_catalog.csv: {_display_path(result.scenario_catalog_csv_path)}")
    print(f"simulation_manifest.json: {_display_path(result.simulation_manifest_path)}")


def _display_path(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        digest = hashlib.sha256(resolved.as_posix().encode("utf-8")).hexdigest()[:12]
        return f"external/{resolved.name}_{digest}"


def main() -> None:
    try:
        run_cli()
    except (MarketSimulationConfigError, ValueError, FileNotFoundError) as exc:
        print(str(exc).strip() or exc.__class__.__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
