from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.run_market_simulation_scenarios import run_cli
from src.config.market_simulation import MarketSimulationConfig, MarketSimulationConfigError
from src.research.market_simulation.artifacts import run_market_simulation_framework
from src.research.market_simulation.registry import get_simulation_type_metadata

FIXTURE_PATH = "tests/fixtures/market_simulation/historical_episode_fixture.csv"


def _payload(
    output_root: Path,
    *,
    overlay_enabled: bool = True,
    source_name: str = "historical_volatility_episode",
    overlay_overrides: dict | None = None,
) -> dict:
    overlay_method = {
        "input_source": {
            "type": "historical_episode_replay",
            "scenario_name": source_name,
        },
        "timestamp_column": "ts_utc",
        "symbol_column": "symbol",
        "base_return_column": "source_return",
        "adaptive_policy_return_column": "adaptive_policy_return",
        "static_baseline_return_column": "static_baseline_return",
        "confidence_column": "gmm_confidence",
        "entropy_column": "gmm_entropy",
        "overlays": [
            {
                "name": "return_drawdown_shock",
                "type": "return_bps",
                "columns": [
                    "source_return",
                    "adaptive_policy_return",
                    "static_baseline_return",
                ],
                "bps": -50,
            },
            {
                "name": "volatility_amplification",
                "type": "volatility_multiplier",
                "columns": [
                    "source_return",
                    "adaptive_policy_return",
                    "static_baseline_return",
                ],
                "multiplier": 1.50,
            },
            {
                "name": "confidence_degradation",
                "type": "confidence_multiplier",
                "column": "gmm_confidence",
                "multiplier": 0.70,
            },
            {
                "name": "entropy_amplification",
                "type": "entropy_multiplier",
                "column": "gmm_entropy",
                "multiplier": 1.25,
            },
        ],
    }
    if overlay_overrides:
        overlay_method.update(overlay_overrides)
    return {
        "simulation_name": "shock_overlay_test",
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
                "method_config": {
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
                        }
                    ],
                },
            },
            {
                "name": "volatility_spike_overlay",
                "type": "shock_overlay",
                "enabled": overlay_enabled,
                "method_config": overlay_method,
            },
        ],
    }


def _run(tmp_path: Path, **kwargs):
    config = MarketSimulationConfig.from_mapping(_payload(tmp_path / "outputs", **kwargs))
    return run_market_simulation_framework(config)


def test_registry_marks_shock_overlay_implemented_and_config_loads(tmp_path: Path) -> None:
    metadata = get_simulation_type_metadata("shock_overlay")
    config = MarketSimulationConfig.from_mapping(_payload(tmp_path / "outputs"))

    assert metadata.status == "implemented"
    assert metadata.uses_shock_overlay is True
    assert config.market_simulations[1].method_config["overlays"][0]["type"] == "return_bps"


def test_overlay_resolves_historical_replay_source_by_name(tmp_path: Path) -> None:
    result = _run(tmp_path)

    overlay = result.shock_overlay_results[0]

    assert overlay.source_scenario_name == "historical_volatility_episode"
    assert overlay.shock_overlay_results_path.exists()
    assert len(result.historical_episode_replay_results) == 1


def test_missing_source_scenario_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(MarketSimulationConfigError, match="was not found"):
        _run(tmp_path, source_name="missing_replay")


def test_disabled_overlay_does_not_run(tmp_path: Path) -> None:
    result = _run(tmp_path, overlay_enabled=False)

    assert result.shock_overlay_results == []


def test_return_bps_preserves_source_and_writes_stressed_column(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        overlay_overrides={
            "overlays": [
                {
                    "name": "return_drawdown_shock",
                    "type": "return_bps",
                    "columns": ["source_return"],
                    "bps": -50,
                }
            ]
        },
    )
    rows = pd.read_csv(result.shock_overlay_results[0].shock_overlay_results_path)

    assert "regime_label" in rows.columns
    assert rows.loc[0, "source_return"] == pytest.approx(-0.02)
    assert rows.loc[0, "stressed_source_return"] == pytest.approx(-0.025)


def test_volatility_multiplier_increases_dispersion_deterministically(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        overlay_overrides={
            "overlays": [
                {
                    "name": "volatility_amplification",
                    "type": "volatility_multiplier",
                    "columns": ["source_return"],
                    "multiplier": 1.50,
                }
            ]
        },
    )
    rows = pd.read_csv(result.shock_overlay_results[0].shock_overlay_results_path)

    assert rows["stressed_source_return"].std(ddof=0) > rows["source_return"].std(ddof=0)
    assert rows.loc[0, "stressed_source_return"] == pytest.approx(-0.023125)


def test_confidence_multiplier_clamps_to_unit_interval(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        overlay_overrides={
            "overlays": [
                {
                    "name": "confidence_boost",
                    "type": "confidence_multiplier",
                    "column": "gmm_confidence",
                    "multiplier": 2.0,
                }
            ]
        },
    )
    rows = pd.read_csv(result.shock_overlay_results[0].shock_overlay_results_path)

    assert rows["stressed_gmm_confidence"].max() == pytest.approx(1.0)
    assert rows["stressed_gmm_confidence"].min() >= 0.0


def test_entropy_multiplier_clamps_lower_bound(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        overlay_overrides={
            "overlays": [
                {
                    "name": "entropy_negative",
                    "type": "entropy_multiplier",
                    "column": "gmm_entropy",
                    "multiplier": -2.0,
                }
            ]
        },
    )
    rows = pd.read_csv(result.shock_overlay_results[0].shock_overlay_results_path)

    assert rows["stressed_gmm_entropy"].min() == pytest.approx(0.0)


def test_composable_overlays_apply_in_order_and_log_each_step(tmp_path: Path) -> None:
    result = _run(tmp_path)
    overlay = result.shock_overlay_results[0]
    rows = pd.read_csv(overlay.shock_overlay_results_path)
    log = pd.read_csv(overlay.shock_overlay_log_path)

    assert rows.loc[0, "stressed_source_return"] == pytest.approx(-0.028125)
    assert rows.loc[0, "overlay_stack"] == (
        "return_drawdown_shock|volatility_amplification|"
        "confidence_degradation|entropy_amplification"
    )
    assert list(log["overlay_index"].head(6)) == [1, 1, 1, 2, 2, 2]
    assert set(log["status"]) == {"applied"}


def test_artifact_writing_and_policy_comparison_summary(tmp_path: Path) -> None:
    result = _run(tmp_path)
    overlay = result.shock_overlay_results[0]
    summary = json.loads(overlay.shock_overlay_summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(overlay.manifest_path.read_text(encoding="utf-8"))

    assert overlay.shock_overlay_config_path.exists()
    assert overlay.shock_overlay_results_path.exists()
    assert overlay.shock_overlay_log_path.exists()
    assert overlay.shock_overlay_summary_path.exists()
    assert overlay.manifest_path.exists()
    assert summary["policy_comparison_available"] is True
    assert summary["stressed_adaptive_return_total"] is not None
    assert summary["stressed_static_baseline_return_total"] is not None
    assert summary["stressed_adaptive_vs_static_return_delta"] is not None
    assert manifest["artifact_type"] == "shock_overlay"
    assert manifest["row_counts"]["shock_overlay_results_csv"] == 4


def test_missing_optional_policy_columns_are_reported_without_policy_summary(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        overlay_overrides={
            "adaptive_policy_return_column": "missing_adaptive",
            "static_baseline_return_column": "missing_static",
            "overlays": [
                {
                    "name": "return_drawdown_shock",
                    "type": "return_bps",
                    "columns": ["source_return"],
                    "bps": -50,
                }
            ],
        },
    )
    summary = json.loads(result.shock_overlay_results[0].shock_overlay_summary_path.read_text(encoding="utf-8"))

    assert summary["policy_comparison_available"] is False
    assert summary["stressed_adaptive_return_total"] is None


def test_json_artifacts_do_not_leak_tmp_absolute_paths(tmp_path: Path) -> None:
    result = _run(tmp_path)
    overlay = result.shock_overlay_results[0]

    for path in (overlay.shock_overlay_config_path, overlay.shock_overlay_summary_path, overlay.manifest_path):
        text = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "C:\\" not in text


def test_rerun_produces_identical_results_and_summary(tmp_path: Path) -> None:
    first = _run(tmp_path / "first").shock_overlay_results[0]
    second = _run(tmp_path / "second").shock_overlay_results[0]

    assert first.shock_overlay_results_path.read_text(encoding="utf-8") == second.shock_overlay_results_path.read_text(encoding="utf-8")
    assert first.shock_overlay_summary_path.read_text(encoding="utf-8") == second.shock_overlay_summary_path.read_text(encoding="utf-8")


def test_cli_smoke_writes_historical_replay_and_shock_overlay_artifacts(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "shock_overlay.yml"
    config_path.write_text(
        yaml.safe_dump(_payload(tmp_path / "outputs"), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    result = run_cli(["--config", config_path.as_posix()])
    output = capsys.readouterr().out

    assert result.historical_episode_replay_results[0].episode_replay_results_path.exists()
    assert result.shock_overlay_results[0].shock_overlay_results_path.exists()
    assert "Shock overlays: 1" in output
    assert str(tmp_path) not in output
