from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.run_market_simulation_scenarios import run_cli
from src.config.market_simulation import MarketSimulationConfig, MarketSimulationConfigError
from src.research.market_simulation.artifacts import run_market_simulation_framework
from src.research.market_simulation.block_bootstrap import run_regime_block_bootstrap
from src.research.market_simulation.registry import get_simulation_type_metadata

FIXTURE_PATH = "tests/fixtures/market_simulation/bootstrap_source_fixture.csv"


def _payload(
    output_root: Path,
    *,
    method_overrides: dict | None = None,
    sampling_overrides: dict | None = None,
    scenario_path_count: int = 2,
) -> dict:
    sampling = {
        "mode": "regime_bucketed",
        "target_regimes": ["high_vol", "stress"],
        "include_transition_windows": True,
        "transition_window_bars": 1,
    }
    if sampling_overrides:
        sampling.update(sampling_overrides)
    method_config = {
        "dataset_path": FIXTURE_PATH,
        "timestamp_column": "ts_utc",
        "symbol_column": "symbol",
        "return_column": "return",
        "regime_column": "regime_label",
        "confidence_column": "gmm_confidence",
        "entropy_column": "gmm_entropy",
        "path_count": 2,
        "path_length_bars": 6,
        "block_length_bars": 2,
        "path_start": "2000-01-01",
        "sampling": sampling,
    }
    if method_overrides:
        method_config.update(method_overrides)
    return {
        "simulation_name": "regime_block_bootstrap_test",
        "output_root": output_root.as_posix(),
        "random_seed": 2701,
        "source_review_pack": "artifacts/regime_reviews/source_review",
        "baseline_policy": "static_baseline",
        "source_policy_candidates": ["adaptive_policy"],
        "market_simulations": [
            {
                "name": "high_vol_regime_bootstrap",
                "type": "regime_block_bootstrap",
                "enabled": True,
                "random_seed": 2701,
                "path_count": scenario_path_count,
                "method_config": method_config,
            }
        ],
    }


def _config(tmp_path: Path, **kwargs) -> MarketSimulationConfig:
    return MarketSimulationConfig.from_mapping(_payload(tmp_path / "outputs", **kwargs))


def _run_bootstrap(tmp_path: Path, **kwargs):
    config = _config(tmp_path, **kwargs)
    scenario = config.market_simulations[0]
    return (
        config,
        scenario,
        run_regime_block_bootstrap(
            scenario,
            simulation_run_id="regime_block_bootstrap_test_run",
            market_simulations_output_dir=tmp_path / "outputs" / "run" / "market_simulations",
        ),
    )


def test_registry_marks_regime_block_bootstrap_implemented_and_config_loads(tmp_path: Path) -> None:
    metadata = get_simulation_type_metadata("regime_block_bootstrap")
    config = _config(tmp_path)

    assert metadata.status == "implemented"
    assert metadata.uses_synthetic_generation is True
    assert config.market_simulations[0].method_config["sampling"]["mode"] == "regime_bucketed"


def test_missing_timestamp_column_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(MarketSimulationConfigError, match="timestamp_missing"):
        _run_bootstrap(tmp_path, method_overrides={"timestamp_column": "timestamp_missing"})


def test_missing_return_column_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(MarketSimulationConfigError, match="return_missing"):
        _run_bootstrap(tmp_path, method_overrides={"return_column": "return_missing"})


def test_missing_regime_column_fails_for_regime_bucketed(tmp_path: Path) -> None:
    with pytest.raises(MarketSimulationConfigError, match="regime_column"):
        _run_bootstrap(tmp_path, method_overrides={"regime_column": "missing_regime"})


def test_null_timestamps_fail_clearly(tmp_path: Path) -> None:
    dataset = tmp_path / "null_ts.csv"
    dataset.write_text(
        "ts_utc,symbol,return,regime_label\n"
        "2026-01-01T00:00:00Z,AAPL,0.01,calm\n"
        ",AAPL,0.02,high_vol\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(MarketSimulationConfigError, match="contains null values"):
        _run_bootstrap(tmp_path, method_overrides={"dataset_path": dataset.as_posix()})


def test_source_block_catalog_is_deterministic_and_tags_transitions(tmp_path: Path) -> None:
    _, _, result = _run_bootstrap(tmp_path)
    catalog = pd.read_csv(result.source_block_catalog_path)

    assert len(catalog) == 14
    assert list(catalog["source_block_index"].head(3)) == [0, 1, 2]
    assert catalog["contains_transition_window"].any()
    assert {"high_vol", "stress"}.issubset(set(catalog["primary_regime"].dropna()))


def test_fixed_sampling_generates_requested_paths_lengths_and_stable_ids(tmp_path: Path) -> None:
    _, _, first = _run_bootstrap(
        tmp_path / "first",
        method_overrides={"path_count": 3},
        sampling_overrides={"mode": "fixed", "target_regimes": []},
    )
    _, _, second = _run_bootstrap(
        tmp_path / "second",
        method_overrides={"path_count": 3},
        sampling_overrides={"mode": "fixed", "target_regimes": []},
    )
    paths = pd.read_csv(first.bootstrap_path_catalog_path)
    rows = pd.read_parquet(first.simulated_return_paths_path)
    second_paths = pd.read_csv(second.bootstrap_path_catalog_path)

    assert len(paths) == 3
    assert rows.groupby("path_id").size().tolist() == [6, 6, 6]
    assert paths["path_id"].tolist() == second_paths["path_id"].tolist()


def test_regime_bucketed_sampling_filters_target_regime_blocks(tmp_path: Path) -> None:
    _, _, result = _run_bootstrap(tmp_path)
    inventory = pd.read_csv(result.sampled_block_inventory_path)

    assert set(inventory["primary_regime"]).issubset({"high_vol", "stress"})


def test_transition_window_sampling_uses_transition_blocks(tmp_path: Path) -> None:
    _, _, result = _run_bootstrap(
        tmp_path,
        sampling_overrides={"mode": "transition_window", "target_regimes": []},
    )
    source_catalog = pd.read_csv(result.source_block_catalog_path)
    inventory = pd.read_csv(result.sampled_block_inventory_path)
    rows = pd.read_parquet(result.simulated_return_paths_path)

    assert source_catalog["contains_transition_window"].any()
    assert inventory["contains_transition_window"].all()
    assert rows["is_transition_window"].any()


def test_repeated_runs_produce_identical_catalog_and_simulated_paths(tmp_path: Path) -> None:
    _, _, first = _run_bootstrap(tmp_path / "first")
    _, _, second = _run_bootstrap(tmp_path / "second")

    assert first.bootstrap_path_catalog_path.read_text(encoding="utf-8") == second.bootstrap_path_catalog_path.read_text(encoding="utf-8")
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.simulated_return_paths_path),
        pd.read_parquet(second.simulated_return_paths_path),
    )


def test_artifact_writing_creates_expected_files(tmp_path: Path) -> None:
    _, _, result = _run_bootstrap(tmp_path)

    assert result.bootstrap_path_catalog_path.exists()
    assert result.source_block_catalog_path.exists()
    assert result.sampled_block_inventory_path.exists()
    assert result.simulated_return_paths_path.exists()
    assert result.simulated_regime_paths_path.exists()
    assert result.bootstrap_sampling_summary_path.exists()
    assert result.bootstrap_config_path.exists()
    assert result.manifest_path.exists()


def test_parquet_outputs_are_written_for_return_and_regime_paths(tmp_path: Path) -> None:
    _, _, result = _run_bootstrap(tmp_path)

    assert result.simulated_return_paths_path.name == "simulated_return_paths.parquet"
    assert result.simulated_regime_paths_path.name == "simulated_regime_paths.parquet"
    assert result.simulated_return_paths_path.exists()
    assert result.simulated_regime_paths_path.exists()
    assert len(pd.read_parquet(result.simulated_return_paths_path)) == 12
    assert len(pd.read_parquet(result.simulated_regime_paths_path)) == 12


def test_json_artifacts_do_not_leak_absolute_tmp_paths(tmp_path: Path) -> None:
    _, _, result = _run_bootstrap(tmp_path)

    for path in (result.bootstrap_config_path, result.bootstrap_sampling_summary_path, result.manifest_path):
        text = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "C:\\" not in text


def test_framework_and_execution_result_include_bootstrap_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_market_simulation_framework(config)

    assert len(result.block_bootstrap_results) == 1
    assert result.block_bootstrap_results[0].simulated_row_count == 12


def test_shock_overlay_does_not_resolve_bootstrap_outputs_as_inputs(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "outputs")
    payload["market_simulations"].append(
        {
            "name": "bootstrap_overlay_attempt",
            "type": "shock_overlay",
            "enabled": True,
            "method_config": {
                "input_source": {
                    "type": "historical_episode_replay",
                    "scenario_name": "high_vol_regime_bootstrap",
                },
                "timestamp_column": "ts_utc",
                "symbol_column": "symbol",
                "base_return_column": "source_return",
                "overlays": [
                    {
                        "name": "return_drawdown_shock",
                        "type": "return_bps",
                        "column": "source_return",
                        "bps": -50,
                    }
                ],
            },
        }
    )
    config = MarketSimulationConfig.from_mapping(payload)

    with pytest.raises(MarketSimulationConfigError, match="was not found"):
        run_market_simulation_framework(config)


def test_manifest_records_relative_paths_and_counts(tmp_path: Path) -> None:
    _, _, result = _run_bootstrap(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["artifact_type"] == "regime_block_bootstrap"
    assert manifest["source_dataset_metadata"]["dataset_path"] == FIXTURE_PATH
    assert manifest["row_counts"]["simulated_return_paths_parquet"] == 12


def test_cli_smoke_writes_bootstrap_artifacts(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "regime_block_bootstrap.yml"
    config_path.write_text(
        yaml.safe_dump(_payload(tmp_path / "outputs"), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    result = run_cli(["--config", config_path.as_posix()])
    output = capsys.readouterr().out

    assert result.block_bootstrap_results[0].bootstrap_path_catalog_path.exists()
    assert "Block bootstraps: 1" in output
    assert "Block bootstrap simulated rows: 12" in output
    assert str(tmp_path) not in output
