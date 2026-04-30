from __future__ import annotations

from src.research.market_simulation.ids import generate_path_id, generate_scenario_id


def test_generate_scenario_id_is_deterministic() -> None:
    kwargs = {
        "simulation_name": "sim",
        "scenario_name": "episode",
        "simulation_type": "historical_episode_replay",
        "seed": 1729,
        "path_count": 0,
        "method_config": {"b": 2, "a": 1},
    }

    assert generate_scenario_id(**kwargs) == generate_scenario_id(**kwargs)


def test_generate_scenario_id_changes_when_name_type_or_seed_changes() -> None:
    base = {
        "simulation_name": "sim",
        "scenario_name": "episode",
        "simulation_type": "historical_episode_replay",
        "seed": 1729,
        "path_count": 0,
    }

    base_id = generate_scenario_id(**base)

    assert generate_scenario_id(**{**base, "scenario_name": "other"}) != base_id
    assert generate_scenario_id(**{**base, "simulation_type": "shock_overlay"}) != base_id
    assert generate_scenario_id(**{**base, "seed": 1730}) != base_id


def test_generate_path_id_is_deterministic() -> None:
    first = generate_path_id(
        scenario_id="episode_abc123",
        path_index=7,
        seed=1729,
        metadata={"regime": "high_vol"},
    )
    second = generate_path_id(
        scenario_id="episode_abc123",
        path_index=7,
        seed=1729,
        metadata={"regime": "high_vol"},
    )

    assert first == second
    assert first.startswith("episode_abc123_path_000007_")
