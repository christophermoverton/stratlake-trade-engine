from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.market_simulation import (
    DEFAULT_RANDOM_SEED,
    MarketSimulationConfig,
    MarketSimulationConfigError,
    load_market_simulation_config,
)
from src.research.market_simulation.registry import RESERVED_SIMULATION_TYPES, get_simulation_type_metadata


def _payload(overrides: dict | None = None) -> dict:
    payload = {
        "simulation_name": "test_market_simulation",
        "output_root": "artifacts/regime_stress_tests",
        "random_seed": 1729,
        "source_review_pack": "artifacts/regime_reviews/source_review",
        "baseline_policy": "static_baseline",
        "source_policy_candidates": ["adaptive_policy"],
        "market_simulations": [
            {
                "name": "episode",
                "type": "historical_episode_replay",
                "enabled": True,
            },
            {
                "name": "disabled_overlay",
                "type": "shock_overlay",
                "enabled": False,
            },
        ],
    }
    if overrides:
        payload.update(overrides)
    return payload


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "market_simulation.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def test_market_simulation_config_loads_valid_config(tmp_path: Path) -> None:
    config = load_market_simulation_config(_write_config(tmp_path, _payload()))

    assert isinstance(config, MarketSimulationConfig)
    assert config.simulation_name == "test_market_simulation"
    assert config.market_simulations[0].seed == 1729
    assert config.market_simulations[1].enabled is False


def test_market_simulation_config_missing_required_field_fails(tmp_path: Path) -> None:
    payload = _payload()
    payload.pop("baseline_policy")

    with pytest.raises(MarketSimulationConfigError, match="baseline_policy"):
        load_market_simulation_config(_write_config(tmp_path, payload))


def test_market_simulation_config_accepts_disabled_scenarios(tmp_path: Path) -> None:
    config = load_market_simulation_config(_write_config(tmp_path, _payload()))

    assert [scenario.enabled for scenario in config.market_simulations] == [True, False]


def test_reserved_simulation_types_pass_validation() -> None:
    for simulation_type in RESERVED_SIMULATION_TYPES:
        assert get_simulation_type_metadata(simulation_type).simulation_type == simulation_type


def test_unsupported_simulation_type_fails(tmp_path: Path) -> None:
    payload = _payload(
        {
            "market_simulations": [
                {"name": "bad", "type": "unknown_simulation", "enabled": True},
            ]
        }
    )

    with pytest.raises(MarketSimulationConfigError, match="unsupported"):
        load_market_simulation_config(_write_config(tmp_path, payload))


def test_disabled_unsupported_simulation_type_still_fails(tmp_path: Path) -> None:
    payload = _payload(
        {
            "market_simulations": [
                {"name": "bad_disabled", "type": "unknown_simulation", "enabled": False},
            ]
        }
    )

    with pytest.raises(MarketSimulationConfigError, match="unsupported"):
        load_market_simulation_config(_write_config(tmp_path, payload))


def test_missing_global_seed_uses_deterministic_default(tmp_path: Path) -> None:
    payload = _payload()
    payload.pop("random_seed")

    config = load_market_simulation_config(_write_config(tmp_path, payload))

    assert config.random_seed == DEFAULT_RANDOM_SEED
    assert config.market_simulations[0].seed == DEFAULT_RANDOM_SEED


def test_scenario_seed_overrides_global_seed(tmp_path: Path) -> None:
    payload = _payload(
        {
            "market_simulations": [
                {"name": "episode", "type": "historical_episode_replay", "random_seed": 99},
            ]
        }
    )

    config = load_market_simulation_config(_write_config(tmp_path, payload))

    assert config.random_seed == 1729
    assert config.market_simulations[0].seed == 99
