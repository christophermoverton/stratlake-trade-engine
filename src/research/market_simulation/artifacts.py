from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.research.market_simulation.catalog import SCENARIO_CATALOG_COLUMNS, build_scenario_catalog
from src.research.market_simulation.config import MarketSimulationConfig
from src.research.market_simulation.historical_replay import (
    HistoricalEpisodeReplayResult,
    run_historical_episode_replays,
)
from src.research.market_simulation.shock_overlay import ShockOverlayResult, run_shock_overlay_scenarios
from src.research.registry import canonicalize_value, serialize_canonical_json, stable_timestamp_from_run_id


@dataclass(frozen=True)
class MarketSimulationFrameworkResult:
    simulation_run_id: str
    simulation_name: str
    output_dir: Path
    scenario_catalog_csv_path: Path
    scenario_catalog_json_path: Path
    simulation_config_path: Path
    input_inventory_path: Path
    simulation_manifest_path: Path
    scenario_catalog: list[dict[str, Any]]
    input_inventory: dict[str, Any]
    simulation_manifest: dict[str, Any]
    historical_episode_replay_results: list[HistoricalEpisodeReplayResult]
    shock_overlay_results: list[ShockOverlayResult]


def run_market_simulation_framework(
    config: MarketSimulationConfig,
    *,
    config_path: str | Path | None = None,
) -> MarketSimulationFrameworkResult:
    scenario_catalog = build_scenario_catalog(config)
    simulation_run_id = generate_simulation_run_id(config, scenario_catalog)
    output_dir = Path(config.output_root).resolve() / simulation_run_id / "market_simulations"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "scenario_catalog_csv": output_dir / "scenario_catalog.csv",
        "scenario_catalog_json": output_dir / "scenario_catalog.json",
        "simulation_config_json": output_dir / "simulation_config.json",
        "input_inventory_json": output_dir / "input_inventory.json",
        "simulation_manifest_json": output_dir / "simulation_manifest.json",
    }
    normalized_config = config.to_dict()
    input_inventory = build_input_inventory(config)

    _write_csv(paths["scenario_catalog_csv"], scenario_catalog, SCENARIO_CATALOG_COLUMNS)
    _write_json(
        paths["scenario_catalog_json"],
        {"row_count": len(scenario_catalog), "columns": list(SCENARIO_CATALOG_COLUMNS), "rows": scenario_catalog},
    )
    _write_json(paths["simulation_config_json"], normalized_config)
    _write_json(paths["input_inventory_json"], input_inventory)

    manifest = build_simulation_manifest(
        config=config,
        simulation_run_id=simulation_run_id,
        scenario_catalog=scenario_catalog,
        input_inventory=input_inventory,
        config_path=config_path,
    )
    _write_json(paths["simulation_manifest_json"], manifest)
    historical_results = run_historical_episode_replays(
        config,
        simulation_run_id=simulation_run_id,
        market_simulations_output_dir=output_dir,
    )
    shock_overlay_results = run_shock_overlay_scenarios(
        config,
        simulation_run_id=simulation_run_id,
        market_simulations_output_dir=output_dir,
        historical_episode_replay_results=historical_results,
    )

    return MarketSimulationFrameworkResult(
        simulation_run_id=simulation_run_id,
        simulation_name=config.simulation_name,
        output_dir=output_dir,
        scenario_catalog_csv_path=paths["scenario_catalog_csv"],
        scenario_catalog_json_path=paths["scenario_catalog_json"],
        simulation_config_path=paths["simulation_config_json"],
        input_inventory_path=paths["input_inventory_json"],
        simulation_manifest_path=paths["simulation_manifest_json"],
        scenario_catalog=scenario_catalog,
        input_inventory=input_inventory,
        simulation_manifest=manifest,
        historical_episode_replay_results=historical_results,
        shock_overlay_results=shock_overlay_results,
    )


def generate_simulation_run_id(
    config: MarketSimulationConfig,
    scenario_catalog: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        "simulation_name": config.simulation_name,
        "random_seed": config.random_seed,
        "source_review_pack": config.source_review_pack,
        "baseline_policy": config.baseline_policy,
        "source_policy_candidates": list(config.source_policy_candidates),
        "market_simulations": [scenario.to_dict() for scenario in config.market_simulations],
        "scenario_catalog": scenario_catalog or build_scenario_catalog(config),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{_slugify(config.simulation_name)}_{digest}"


def build_input_inventory(config: MarketSimulationConfig) -> dict[str, Any]:
    source_review_pack = _input_path_record(config.source_review_pack)
    output_root = _input_path_record(config.output_root)
    return canonicalize_value(
        {
            "simulation_name": config.simulation_name,
            "source_review_pack": source_review_pack,
            "baseline_policy": config.baseline_policy,
            "source_policy_candidates": list(config.source_policy_candidates),
            "output_root": output_root,
            "required_inputs": {
                "source_review_pack": False,
                "source_policy_candidates": False,
            },
        }
    )


def build_simulation_manifest(
    *,
    config: MarketSimulationConfig,
    simulation_run_id: str,
    scenario_catalog: list[dict[str, Any]],
    input_inventory: Mapping[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    enabled_count = sum(1 for row in scenario_catalog if bool(row["enabled"]))
    disabled_count = len(scenario_catalog) - enabled_count
    generated_files = {
        "scenario_catalog_csv": "scenario_catalog.csv",
        "scenario_catalog_json": "scenario_catalog.json",
        "simulation_config_json": "simulation_config.json",
        "input_inventory_json": "input_inventory.json",
        "simulation_manifest_json": "simulation_manifest.json",
    }
    return canonicalize_value(
        {
            "artifact_type": "market_simulation_framework",
            "schema_version": "1.0",
            "simulation_run_id": simulation_run_id,
            "simulation_name": config.simulation_name,
            "timestamp": stable_timestamp_from_run_id(simulation_run_id),
            "random_seed": config.random_seed,
            "scenario_count": len(scenario_catalog),
            "enabled_scenario_count": enabled_count,
            "disabled_scenario_count": disabled_count,
            "generated_files": generated_files,
            "row_counts": {
                "scenario_catalog_csv": len(scenario_catalog),
                "scenario_catalog_json": len(scenario_catalog),
            },
            "source_config_metadata": {
                "config_path": None if config_path is None else _rel(Path(config_path)),
                "simulation_name": config.simulation_name,
            },
            "input_inventory": input_inventory,
            "source_paths": {
                "source_review_pack": None if config.source_review_pack is None else _rel(Path(config.source_review_pack)),
                "output_root": _rel(Path(config.output_root)),
            },
            "limitations": [
                "This framework writes deterministic metadata only; simulation methods are reserved for follow-up issues.",
                "Artifacts are for research stress testing, not live trading or market forecasting.",
            ],
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _input_path_record(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(value)
    return {
        "configured_path": _rel(path),
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "is_file": path.is_file() if path.exists() else False,
    }


def _rel(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:12]
        return f"external/{path.name}_{digest}"


def _slugify(value: str) -> str:
    chars = [character.lower() if character.isalnum() else "_" for character in value.strip()]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "market_simulation"
