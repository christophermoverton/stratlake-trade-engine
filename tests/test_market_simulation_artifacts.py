from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.cli.run_market_simulation_scenarios import run_cli
from src.config.market_simulation import MarketSimulationConfig
from src.research.market_simulation.artifacts import run_market_simulation_framework


def _config(tmp_path: Path) -> MarketSimulationConfig:
    return MarketSimulationConfig.from_mapping(
        {
            "simulation_name": "artifact_test",
            "output_root": (tmp_path / "outputs").as_posix(),
            "random_seed": 1729,
            "source_review_pack": "artifacts/regime_reviews/source_review",
            "baseline_policy": "static_baseline",
            "source_policy_candidates": ["adaptive_policy", "transition_policy"],
            "market_simulations": [
                {"name": "episode", "type": "historical_episode_replay", "enabled": True},
                {"name": "disabled_overlay", "type": "shock_overlay", "enabled": False},
                {"name": "monte_carlo", "type": "regime_transition_monte_carlo", "path_count": 25},
            ],
        }
    )


def test_input_inventory_records_configured_inputs_and_relative_paths(tmp_path: Path) -> None:
    result = run_market_simulation_framework(_config(tmp_path))

    inventory = json.loads(result.input_inventory_path.read_text(encoding="utf-8"))

    assert inventory["source_review_pack"]["configured_path"] == "artifacts/regime_reviews/source_review"
    assert inventory["source_review_pack"]["exists"] is False
    assert inventory["baseline_policy"] == "static_baseline"
    assert inventory["source_policy_candidates"] == ["adaptive_policy", "transition_policy"]
    assert not Path(inventory["source_review_pack"]["configured_path"]).is_absolute()


def test_manifest_records_generated_files_and_row_counts_without_absolute_paths(tmp_path: Path) -> None:
    result = run_market_simulation_framework(_config(tmp_path), config_path="configs/regime_stress_tests/example.yml")

    manifest_text = result.simulation_manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["generated_files"]["scenario_catalog_csv"] == "scenario_catalog.csv"
    assert manifest["row_counts"]["scenario_catalog_csv"] == 3
    assert manifest["scenario_count"] == 3
    assert manifest["enabled_scenario_count"] == 2
    assert manifest["disabled_scenario_count"] == 1
    assert "C:\\" not in manifest_text
    assert str(tmp_path) not in manifest_text


def test_normalized_config_artifact_is_written(tmp_path: Path) -> None:
    result = run_market_simulation_framework(_config(tmp_path))

    payload = json.loads(result.simulation_config_path.read_text(encoding="utf-8"))

    assert payload["simulation_name"] == "artifact_test"
    assert payload["market_simulations"][0]["scenario_id"]


def test_market_simulation_cli_smoke(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "market.yml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8", newline="\n")

    result = run_cli(["--config", config_path.as_posix()])

    assert result.scenario_catalog_csv_path.exists()
    assert result.simulation_manifest_path.exists()
