from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGIES_CONFIG = REPO_ROOT / "configs" / "strategies.yml"


def load_strategies_config() -> dict:
    with STRATEGIES_CONFIG.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def test_strategies_config_file_exists() -> None:
    assert STRATEGIES_CONFIG.exists()


def test_strategies_config_parses_successfully() -> None:
    strategies = load_strategies_config()

    assert isinstance(strategies, dict)


def test_strategies_config_contains_example_strategy() -> None:
    strategies = load_strategies_config()

    assert strategies
    assert "momentum_v1" in strategies


def test_each_strategy_entry_has_required_fields() -> None:
    strategies = load_strategies_config()

    for strategy_name, strategy_config in strategies.items():
        assert isinstance(strategy_config, dict), f"{strategy_name} must map to a dictionary"
        assert "dataset" in strategy_config, f"{strategy_name} is missing dataset"
        assert "parameters" in strategy_config, f"{strategy_name} is missing parameters"
        assert isinstance(strategy_config["dataset"], str), f"{strategy_name} dataset must be a string"
        assert strategy_config["dataset"], f"{strategy_name} dataset must not be empty"
        assert isinstance(
            strategy_config["parameters"], dict
        ), f"{strategy_name} parameters must be a dictionary"
