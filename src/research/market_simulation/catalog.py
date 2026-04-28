from __future__ import annotations

from typing import Any

from src.research.market_simulation.config import MarketSimulationConfig
from src.research.market_simulation.registry import get_simulation_type_metadata
from src.research.registry import canonicalize_value

SCENARIO_CATALOG_COLUMNS = (
    "scenario_id",
    "scenario_name",
    "simulation_type",
    "enabled",
    "seed",
    "path_count",
    "source_window_start",
    "source_window_end",
    "uses_historical_data",
    "uses_synthetic_generation",
    "uses_shock_overlay",
    "source_config_name",
    "notes",
)


def build_scenario_catalog(config: MarketSimulationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in config.market_simulations:
        metadata = get_simulation_type_metadata(scenario.simulation_type)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "simulation_type": scenario.simulation_type,
                "enabled": scenario.enabled,
                "seed": scenario.seed,
                "path_count": scenario.path_count,
                "source_window_start": scenario.source_window_start,
                "source_window_end": scenario.source_window_end,
                "uses_historical_data": metadata.uses_historical_data,
                "uses_synthetic_generation": metadata.uses_synthetic_generation,
                "uses_shock_overlay": metadata.uses_shock_overlay,
                "source_config_name": scenario.source_config_name or config.simulation_name,
                "notes": scenario.notes,
            }
        )
    rows.sort(key=lambda row: (str(row["scenario_id"]), str(row["scenario_name"])))
    return canonicalize_value(rows)
