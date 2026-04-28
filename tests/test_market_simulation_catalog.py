from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config.market_simulation import MarketSimulationConfig
from src.research.market_simulation.artifacts import run_market_simulation_framework
from src.research.market_simulation.catalog import SCENARIO_CATALOG_COLUMNS, build_scenario_catalog


def _config(tmp_path: Path) -> MarketSimulationConfig:
    return MarketSimulationConfig.from_mapping(
        {
            "simulation_name": "catalog_test",
            "output_root": (tmp_path / "outputs").as_posix(),
            "random_seed": 1729,
            "source_review_pack": "artifacts/regime_reviews/source_review",
            "baseline_policy": "static_baseline",
            "source_policy_candidates": ["adaptive_policy"],
            "market_simulations": [
                {
                    "name": "bootstrap",
                    "type": "regime_block_bootstrap",
                    "path_count": 100,
                    "notes": "enabled bootstrap placeholder",
                },
                {
                    "name": "overlay",
                    "type": "shock_overlay",
                    "enabled": False,
                    "random_seed": 1730,
                },
            ],
        }
    )


def test_build_scenario_catalog_has_stable_order_and_disabled_row(tmp_path: Path) -> None:
    catalog = build_scenario_catalog(_config(tmp_path))

    assert catalog == sorted(catalog, key=lambda row: (row["scenario_id"], row["scenario_name"]))
    assert {row["scenario_name"]: row["enabled"] for row in catalog} == {
        "bootstrap": True,
        "overlay": False,
    }


def test_scenario_catalog_outputs_csv_and_json_with_stable_columns(tmp_path: Path) -> None:
    result = run_market_simulation_framework(_config(tmp_path))

    with result.scenario_catalog_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    payload = json.loads(result.scenario_catalog_json_path.read_text(encoding="utf-8"))

    assert result.scenario_catalog_csv_path.exists()
    assert result.scenario_catalog_json_path.exists()
    assert header == list(SCENARIO_CATALOG_COLUMNS)
    assert payload["columns"] == list(SCENARIO_CATALOG_COLUMNS)
    assert payload["row_count"] == 2


def test_resolved_seed_is_written_to_catalog(tmp_path: Path) -> None:
    result = run_market_simulation_framework(_config(tmp_path))

    seeds = {row["scenario_name"]: row["seed"] for row in result.scenario_catalog}

    assert seeds["bootstrap"] == 1729
    assert seeds["overlay"] == 1730


def test_deterministic_rerun_produces_identical_catalog_content(tmp_path: Path) -> None:
    config = _config(tmp_path)

    first = run_market_simulation_framework(config)
    second = run_market_simulation_framework(config)

    assert first.scenario_catalog_csv_path.read_text(encoding="utf-8") == second.scenario_catalog_csv_path.read_text(encoding="utf-8")
    assert first.scenario_catalog_json_path.read_text(encoding="utf-8") == second.scenario_catalog_json_path.read_text(encoding="utf-8")
