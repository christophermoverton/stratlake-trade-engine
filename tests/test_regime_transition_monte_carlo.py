from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.run_market_simulation_scenarios import run_cli
from src.config.market_simulation import MarketSimulationConfig, MarketSimulationConfigError
from src.research.market_simulation.artifacts import run_market_simulation_framework
from src.research.market_simulation.monte_carlo import (
    run_regime_transition_monte_carlo,
    validate_transition_matrix,
)
from src.research.market_simulation.registry import get_simulation_type_metadata

FIXTURE_PATH = "tests/fixtures/market_simulation/monte_carlo_regime_source_fixture.csv"


def _matrix() -> dict:
    return {
        "low_vol": {"low_vol": 0.7, "high_vol": 0.2, "stress": 0.1},
        "high_vol": {"low_vol": 0.3, "high_vol": 0.5, "stress": 0.2},
        "stress": {"low_vol": 0.25, "high_vol": 0.35, "stress": 0.4},
    }


def _payload(output_root: Path, *, method_overrides: dict | None = None) -> dict:
    method_config = {
        "path_count": 2,
        "path_length_bars": 8,
        "path_start": "2000-01-01",
        "initial_regime": "low_vol",
        "normalize_transition_rows": False,
        "transition_matrix": _matrix(),
        "duration_constraints": {"min_duration_bars": 1, "max_duration_bars": 4},
        "stress_bias": {"enabled": False, "target_regimes": ["stress"], "multiplier": 1.25},
    }
    if method_overrides:
        method_config.update(method_overrides)
    return {
        "simulation_name": "regime_transition_monte_carlo_test",
        "output_root": output_root.as_posix(),
        "random_seed": 3107,
        "source_review_pack": "artifacts/regime_reviews/source_review",
        "baseline_policy": "static_baseline",
        "source_policy_candidates": ["adaptive_policy"],
        "market_simulations": [
            {
                "name": "regime_transition_mc",
                "type": "regime_transition_monte_carlo",
                "enabled": True,
                "random_seed": 3107,
                "path_count": 2,
                "method_config": method_config,
            }
        ],
    }


def _config(tmp_path: Path, **kwargs) -> MarketSimulationConfig:
    return MarketSimulationConfig.from_mapping(_payload(tmp_path / "outputs", **kwargs))


def _run_monte_carlo(tmp_path: Path, **kwargs):
    config = _config(tmp_path, **kwargs)
    scenario = config.market_simulations[0]
    return (
        config,
        scenario,
        run_regime_transition_monte_carlo(
            scenario,
            simulation_run_id="regime_transition_monte_carlo_test_run",
            market_simulations_output_dir=tmp_path / "outputs" / "run" / "market_simulations",
        ),
    )


def test_registry_marks_regime_transition_monte_carlo_implemented_and_config_loads(tmp_path: Path) -> None:
    metadata = get_simulation_type_metadata("regime_transition_monte_carlo")
    config = _config(tmp_path)

    assert metadata.status == "implemented"
    assert metadata.uses_synthetic_generation is True
    assert config.market_simulations[0].method_config["initial_regime"] == "low_vol"


def test_transition_matrix_validation_accepts_valid_matrix() -> None:
    normalized = validate_transition_matrix(_matrix())

    assert normalized["low_vol"]["stress"] == 0.1


def test_negative_probability_fails() -> None:
    matrix = _matrix()
    matrix["low_vol"]["stress"] = -0.1

    with pytest.raises(MarketSimulationConfigError, match="non-negative"):
        validate_transition_matrix(matrix)


def test_non_numeric_probability_fails() -> None:
    matrix = _matrix()
    matrix["low_vol"]["stress"] = "often"

    with pytest.raises(MarketSimulationConfigError, match="numeric"):
        validate_transition_matrix(matrix)


def test_empty_row_fails() -> None:
    matrix = _matrix()
    matrix["low_vol"] = {}

    with pytest.raises(MarketSimulationConfigError, match="non-empty"):
        validate_transition_matrix(matrix)


def test_row_sum_fails_when_normalization_disabled() -> None:
    matrix = _matrix()
    matrix["low_vol"]["stress"] = 0.2

    with pytest.raises(MarketSimulationConfigError, match="sums to"):
        validate_transition_matrix(matrix, normalize_transition_rows=False)


def test_row_normalization_works_when_enabled() -> None:
    matrix = _matrix()
    matrix["low_vol"]["stress"] = 0.2

    normalized = validate_transition_matrix(matrix, normalize_transition_rows=True)

    assert normalized["low_vol"]["stress"] == pytest.approx(0.2 / 1.1)
    assert sum(normalized["low_vol"].values()) == pytest.approx(1.0)


def test_path_generation_counts_lengths_stable_ids_and_determinism(tmp_path: Path) -> None:
    _, _, first = _run_monte_carlo(tmp_path / "first")
    _, _, second = _run_monte_carlo(tmp_path / "second")
    first_catalog = pd.read_csv(first.path_catalog_path)
    second_catalog = pd.read_csv(second.path_catalog_path)
    rows = pd.read_parquet(first.regime_paths_path)
    second_rows = pd.read_parquet(second.regime_paths_path)

    assert len(first_catalog) == 2
    assert rows.groupby("path_id").size().tolist() == [8, 8]
    assert first_catalog["path_id"].tolist() == second_catalog["path_id"].tolist()
    pd.testing.assert_frame_equal(rows, second_rows)


def test_duration_constraints_enforce_minimum_and_maximum(tmp_path: Path) -> None:
    sticky_matrix = {
        "low_vol": {"low_vol": 0.9, "high_vol": 0.1},
        "high_vol": {"low_vol": 1.0, "high_vol": 0.0},
    }
    _, _, result = _run_monte_carlo(
        tmp_path,
        method_overrides={
            "transition_matrix": sticky_matrix,
            "initial_regime": "low_vol",
            "path_count": 1,
            "path_length_bars": 6,
            "duration_constraints": {"min_duration_bars": 2, "max_duration_bars": 3},
        },
    )
    rows = pd.read_parquet(result.regime_paths_path)

    assert rows.loc[1, "regime_label"] == "low_vol"
    assert rows["duration_in_regime"].max() <= 3
    assert rows["transitioned"].sum() >= 1


def test_stress_bias_adjusts_target_probabilities_and_keeps_rows_normalized(tmp_path: Path) -> None:
    _, _, result = _run_monte_carlo(
        tmp_path,
        method_overrides={
            "stress_bias": {"enabled": True, "target_regimes": ["stress"], "multiplier": 2.0}
        },
    )
    payload = json.loads(result.transition_matrix_path.read_text(encoding="utf-8"))

    assert payload["adjusted_transition_matrix"]["low_vol"]["stress"] > 0.1
    assert sum(payload["adjusted_transition_matrix"]["low_vol"].values()) == pytest.approx(1.0)
    assert result.adjusted_transition_matrix_path is not None
    assert result.adjusted_transition_matrix_path.exists()


def test_empirical_matrix_counts_by_symbol_and_writes_counts(tmp_path: Path) -> None:
    _, _, result = _run_monte_carlo(
        tmp_path,
        method_overrides={
            "transition_matrix": None,
            "transition_source": {
                "dataset_path": FIXTURE_PATH,
                "timestamp_column": "ts_utc",
                "regime_column": "regime_label",
                "symbol_column": "symbol",
            },
        },
    )
    payload = json.loads(result.transition_matrix_path.read_text(encoding="utf-8"))
    counts = pd.read_csv(result.transition_counts_path)

    assert payload["source_type"] == "empirical"
    assert payload["base_transition_matrix"]["low_vol"]["high_vol"] == pytest.approx(2 / 3)
    assert int(counts["transition_count"].sum()) == 10


def test_artifact_writing_creates_expected_files(tmp_path: Path) -> None:
    _, _, result = _run_monte_carlo(tmp_path)

    assert result.transition_matrix_path.exists()
    assert result.regime_paths_path.exists()
    assert result.path_catalog_path.exists()
    assert result.summary_path.exists()
    assert result.manifest_path.exists()


def test_json_artifacts_do_not_leak_absolute_tmp_paths(tmp_path: Path) -> None:
    _, _, result = _run_monte_carlo(tmp_path)

    for path in (result.transition_matrix_path, result.summary_path, result.manifest_path):
        text = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "C:\\" not in text


def test_framework_and_execution_result_include_monte_carlo_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_market_simulation_framework(config)

    assert len(result.monte_carlo_results) == 1
    assert result.monte_carlo_results[0].generated_row_count == 16


def test_manifest_records_relative_paths_and_counts(tmp_path: Path) -> None:
    _, _, result = _run_monte_carlo(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["artifact_type"] == "regime_transition_monte_carlo"
    assert manifest["row_counts"]["monte_carlo_regime_paths_parquet"] == 16
    assert manifest["relative_paths"]["scenario_dir"] == result.scenario_id


def test_cli_smoke_writes_monte_carlo_artifacts(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "regime_transition_monte_carlo.yml"
    config_path.write_text(
        yaml.safe_dump(_payload(tmp_path / "outputs"), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    result = run_cli(["--config", config_path.as_posix()])
    output = capsys.readouterr().out

    assert result.monte_carlo_results[0].path_catalog_path.exists()
    assert "Regime-transition Monte Carlo scenarios: 1" in output
    assert "Regime-transition Monte Carlo generated rows: 16" in output
    assert str(tmp_path) not in output
