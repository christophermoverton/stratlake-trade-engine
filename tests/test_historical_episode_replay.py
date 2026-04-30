from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.run_market_simulation_scenarios import run_cli
from src.config.market_simulation import MarketSimulationConfig, MarketSimulationConfigError
from src.research.market_simulation.historical_replay import (
    generate_episode_id,
    run_historical_episode_replay,
)


FIXTURE_PATH = "tests/fixtures/market_simulation/historical_episode_fixture.csv"


def _scenario_payload(output_root: Path, method_overrides: dict | None = None) -> dict:
    method_config = {
        "dataset_path": FIXTURE_PATH,
        "timestamp_column": "ts_utc",
        "symbol_column": "symbol",
        "return_column": "return",
        "regime_column": "regime_label",
        "confidence_column": "gmm_confidence",
        "entropy_column": "gmm_entropy",
        "adaptive_policy_return_column": "adaptive_policy_return",
        "static_baseline_return_column": "static_baseline_return",
        "episodes": [
            {
                "episode_name": "volatility_spike_fixture",
                "episode_type": "volatility_spike",
                "start": "2026-01-15",
                "end": "2026-01-16",
                "description": "Fixture volatility spike.",
            },
            {
                "episode_name": "sideways_whipsaw_fixture",
                "episode_type": "sideways_whipsaw",
                "start": "2026-02-01",
                "end": "2026-02-02",
            },
        ],
    }
    if method_overrides:
        method_config.update(method_overrides)
    return {
        "simulation_name": "historical_episode_test",
        "output_root": output_root.as_posix(),
        "random_seed": 1729,
        "source_review_pack": "artifacts/regime_reviews/source_review",
        "baseline_policy": "static_baseline",
        "source_policy_candidates": ["adaptive_policy"],
        "market_simulations": [
            {
                "name": "historical_volatility_episode",
                "type": "historical_episode_replay",
                "enabled": True,
                "source_window_start": "2026-01-15",
                "source_window_end": "2026-02-15",
                "method_config": method_config,
            }
        ],
    }


def _config(tmp_path: Path, method_overrides: dict | None = None) -> MarketSimulationConfig:
    return MarketSimulationConfig.from_mapping(
        _scenario_payload(tmp_path / "outputs", method_overrides=method_overrides)
    )


def _run_replay(tmp_path: Path, method_overrides: dict | None = None):
    config = _config(tmp_path, method_overrides=method_overrides)
    scenario = config.market_simulations[0]
    return (
        config,
        scenario,
        run_historical_episode_replay(
            scenario,
            simulation_run_id="historical_episode_test_run",
            market_simulations_output_dir=tmp_path / "outputs" / "run" / "market_simulations",
        ),
    )


def test_config_loads_historical_replay_scenario_with_inline_episodes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scenario = config.market_simulations[0]

    assert scenario.simulation_type == "historical_episode_replay"
    assert scenario.scenario_id == MarketSimulationConfig.from_mapping(config.to_dict()).market_simulations[0].scenario_id
    assert scenario.method_config["episodes"][0]["episode_name"] == "volatility_spike_fixture"


def test_episode_window_selection_and_deterministic_sorting(tmp_path: Path) -> None:
    _, _, result = _run_replay(tmp_path)

    replay = pd.read_csv(result.episode_replay_results_path)
    first_episode = replay[replay["episode_name"] == "volatility_spike_fixture"]

    assert list(first_episode["ts_utc"]) == [
        "2026-01-15T00:00:00Z",
        "2026-01-15T00:00:00Z",
        "2026-01-16T00:00:00Z",
        "2026-01-16T00:00:00Z",
    ]
    assert list(first_episode["symbol"]) == ["AAPL", "MSFT", "AAPL", "MSFT"]


def test_date_only_episode_end_includes_full_utc_day(tmp_path: Path) -> None:
    dataset = tmp_path / "intraday_episode.csv"
    dataset.write_text(
        "ts_utc,symbol,return\n"
        "2026-01-16T00:00:00Z,AAPL,0.01\n"
        "2026-01-16T12:30:00Z,AAPL,0.02\n"
        "2026-01-17T00:00:00Z,AAPL,0.03\n",
        encoding="utf-8",
        newline="\n",
    )

    _, _, result = _run_replay(
        tmp_path,
        {
            "dataset_path": dataset.as_posix(),
            "episodes": [
                {
                    "episode_name": "intraday_fixture",
                    "episode_type": "recovery",
                    "start": "2026-01-16",
                    "end": "2026-01-16",
                }
            ],
        },
    )

    replay = pd.read_csv(result.episode_replay_results_path)

    assert list(replay["ts_utc"]) == ["2026-01-16T00:00:00Z", "2026-01-16T12:30:00Z"]


def test_missing_timestamp_column_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(MarketSimulationConfigError, match="timestamp_missing"):
        _run_replay(tmp_path, {"timestamp_column": "timestamp_missing"})


def test_missing_return_column_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(MarketSimulationConfigError, match="return_missing"):
        _run_replay(tmp_path, {"return_column": "return_missing"})


def test_optional_columns_can_be_unavailable_without_failure(tmp_path: Path) -> None:
    dataset = tmp_path / "minimal_episode.csv"
    dataset.write_text(
        "ts_utc,symbol,return\n"
        "2026-01-15T00:00:00Z,MSFT,-0.01\n"
        "2026-01-15T00:00:00Z,AAPL,0.02\n",
        encoding="utf-8",
        newline="\n",
    )

    _, _, result = _run_replay(
        tmp_path,
        {
            "dataset_path": dataset.as_posix(),
            "confidence_column": "missing_confidence",
            "entropy_column": "missing_entropy",
            "adaptive_policy_return_column": "missing_adaptive",
            "static_baseline_return_column": "missing_static",
            "episodes": [
                {
                    "episode_name": "minimal_fixture",
                    "episode_type": "sideways",
                    "start": "2026-01-15",
                    "end": "2026-01-15",
                }
            ],
        },
    )

    catalog = pd.read_csv(result.historical_episode_catalog_path)
    comparison = pd.read_csv(result.episode_policy_comparison_path)
    replay = pd.read_csv(result.episode_replay_results_path)

    assert not bool(catalog.loc[0, "has_confidence"])
    assert not bool(catalog.loc[0, "has_entropy"])
    assert comparison.loc[0, "comparison_status"] == "insufficient_policy_columns"
    assert "missing_adaptive" in comparison.loc[0, "primary_reason"]
    assert pd.isna(replay.loc[0, "adaptive_policy_return"])


def test_artifact_writing_creates_expected_files(tmp_path: Path) -> None:
    _, _, result = _run_replay(tmp_path)

    assert result.historical_episode_catalog_path.exists()
    assert result.episode_replay_results_path.exists()
    assert result.episode_policy_comparison_path.exists()
    assert result.episode_summary_path.exists()
    assert result.manifest_path.exists()


def test_policy_comparison_computes_totals_delta_and_drawdown(tmp_path: Path) -> None:
    _, _, result = _run_replay(tmp_path)

    comparison = pd.read_csv(result.episode_policy_comparison_path)
    spike = comparison[comparison["episode_name"] == "volatility_spike_fixture"].iloc[0]

    assert spike["comparison_status"] == "available"
    assert spike["adaptive_return_total"] == pytest.approx(-0.02603896624)
    assert spike["static_baseline_return_total"] == pytest.approx(-0.049377932416)
    assert spike["adaptive_vs_static_return_delta"] == pytest.approx(0.023338966176)
    assert spike["adaptive_max_drawdown"] == pytest.approx(-0.018)


def test_json_artifacts_do_not_leak_absolute_tmp_paths(tmp_path: Path) -> None:
    _, _, result = _run_replay(tmp_path)

    for path in (result.episode_summary_path, result.manifest_path):
        text = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "C:\\" not in text


def test_episode_ids_are_deterministic() -> None:
    first = generate_episode_id(
        scenario_id="scenario_123",
        episode_name="volatility_spike_fixture",
        start="2026-01-15",
        end="2026-01-16",
        dataset_path=FIXTURE_PATH,
    )
    second = generate_episode_id(
        scenario_id="scenario_123",
        episode_name="volatility_spike_fixture",
        start="2026-01-15",
        end="2026-01-16",
        dataset_path=FIXTURE_PATH,
    )

    assert first == second


def test_rerun_produces_identical_catalog_and_comparison_outputs(tmp_path: Path) -> None:
    _, _, first = _run_replay(tmp_path / "first")
    _, _, second = _run_replay(tmp_path / "second")

    assert first.historical_episode_catalog_path.read_text(encoding="utf-8") == second.historical_episode_catalog_path.read_text(encoding="utf-8")
    assert first.episode_policy_comparison_path.read_text(encoding="utf-8") == second.episode_policy_comparison_path.read_text(encoding="utf-8")


def test_cli_smoke_writes_historical_replay_artifacts(tmp_path: Path, capsys) -> None:
    payload = _scenario_payload(tmp_path / "outputs")
    config_path = tmp_path / "historical_episode.yml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")

    result = run_cli(["--config", config_path.as_posix()])
    output = capsys.readouterr().out

    replay_result = result.historical_episode_replay_results[0]
    assert replay_result.episode_policy_comparison_path.exists()
    assert str(tmp_path) not in output
    assert "Historical episode replays: 1" in output
    assert "external/" in output


def test_manifest_records_relative_paths_and_row_counts(tmp_path: Path) -> None:
    _, _, result = _run_replay(tmp_path)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["artifact_type"] == "historical_episode_replay"
    assert manifest["episode_count"] == 2
    assert manifest["row_counts"]["episode_replay_results_csv"] == 6
    assert manifest["source_dataset_metadata"]["dataset_path"] == FIXTURE_PATH
