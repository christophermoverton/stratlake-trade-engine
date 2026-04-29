from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.run_market_simulation_scenarios import run_cli
from src.config.market_simulation import MarketSimulationConfig
from src.research.market_simulation.artifacts import run_market_simulation_framework


def _matrix() -> dict:
    return {
        "low_vol": {"low_vol": 0.7, "high_vol": 0.2, "stress": 0.1},
        "high_vol": {"low_vol": 0.3, "high_vol": 0.5, "stress": 0.2},
        "stress": {"low_vol": 0.25, "high_vol": 0.35, "stress": 0.4},
    }


def _payload(output_root: Path, *, threshold_overrides: dict | None = None) -> dict:
    thresholds = {
        "max_drawdown_limit": -0.10,
        "min_total_return": -0.05,
        "max_transition_count": 20,
        "max_stress_regime_share": 0.50,
        "max_policy_underperformance": -0.02,
    }
    if threshold_overrides:
        thresholds.update(threshold_overrides)
    return {
        "simulation_name": "simulation_stress_metrics_test",
        "output_root": output_root.as_posix(),
        "random_seed": 308,
        "source_review_pack": "artifacts/regime_reviews/source_review",
        "baseline_policy": "static_baseline",
        "source_policy_candidates": ["adaptive_policy"],
        "stress_metrics": {
            "enabled": True,
            "failure_thresholds": thresholds,
            "leaderboard": {"ranking_metric": "mean_stress_score", "ascending": True},
            "stress_regimes": ["stress", "high_vol"],
            "tail_quantile": 0.05,
        },
        "market_simulations": [
            {
                "name": "metrics_historical_episode",
                "type": "historical_episode_replay",
                "enabled": True,
                "random_seed": 308,
                "path_count": 1,
                "method_config": {
                    "dataset_path": "tests/fixtures/market_simulation/historical_episode_fixture.csv",
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
                "name": "metrics_regime_bootstrap",
                "type": "regime_block_bootstrap",
                "enabled": True,
                "random_seed": 308,
                "path_count": 2,
                "method_config": {
                    "dataset_path": "tests/fixtures/market_simulation/bootstrap_source_fixture.csv",
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
                    "sampling": {
                        "mode": "regime_bucketed",
                        "target_regimes": ["high_vol", "stress"],
                        "include_transition_windows": True,
                        "transition_window_bars": 1,
                    },
                },
            },
            {
                "name": "metrics_regime_transition_mc",
                "type": "regime_transition_monte_carlo",
                "enabled": True,
                "random_seed": 308,
                "path_count": 2,
                "method_config": {
                    "path_count": 2,
                    "path_length_bars": 8,
                    "path_start": "2000-01-01",
                    "initial_regime": "low_vol",
                    "normalize_transition_rows": False,
                    "transition_matrix": _matrix(),
                    "duration_constraints": {"min_duration_bars": 1, "max_duration_bars": 4},
                    "stress_bias": {"enabled": False, "target_regimes": ["stress"], "multiplier": 1.25},
                },
            },
            {
                "name": "metrics_shock_overlay",
                "type": "shock_overlay",
                "enabled": True,
                "random_seed": 308,
                "path_count": 1,
                "method_config": {
                    "input_source": {
                        "type": "historical_episode_replay",
                        "scenario_name": "metrics_historical_episode",
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
                        }
                    ],
                },
            },
        ],
    }


def _run(tmp_path: Path, *, threshold_overrides: dict | None = None):
    config = MarketSimulationConfig.from_mapping(
        _payload(tmp_path / "outputs", threshold_overrides=threshold_overrides)
    )
    return run_market_simulation_framework(config)


def test_stress_metrics_config_defaults_load_deterministically(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "outputs")
    payload.pop("stress_metrics")
    config = MarketSimulationConfig.from_mapping(payload)

    assert config.stress_metrics.enabled is True
    assert config.stress_metrics.output_dir_name == "simulation_metrics"
    assert config.stress_metrics.failure_thresholds["max_drawdown_limit"] == -0.10
    assert config.stress_metrics.stress_regimes == ("stress",)


def test_historical_replay_metrics_compute_policy_delta(tmp_path: Path) -> None:
    result = _run(tmp_path)
    metrics = pd.read_csv(result.simulation_stress_metrics_result.path_metrics_path)
    row = metrics.loc[metrics["simulation_type"] == "historical_episode_replay"].iloc[0]

    assert row["has_policy_metrics"]
    assert row["adaptive_vs_static_return_delta"] == pytest.approx(
        row["adaptive_return_total"] - row["static_baseline_return_total"]
    )


def test_shock_overlay_metrics_use_stressed_returns_and_overlay_count(tmp_path: Path) -> None:
    result = _run(tmp_path)
    metrics = pd.read_csv(result.simulation_stress_metrics_result.path_metrics_path)
    row = metrics.loc[metrics["simulation_type"] == "shock_overlay"].iloc[0]

    assert row["overlay_count"] == 1
    assert row["has_return_metrics"]
    assert row["adaptive_return_total"] < 0


def test_block_bootstrap_metrics_compute_return_drawdown_tail_and_transitions(tmp_path: Path) -> None:
    result = _run(tmp_path)
    metrics = pd.read_csv(result.simulation_stress_metrics_result.path_metrics_path)
    rows = metrics.loc[metrics["simulation_type"] == "regime_block_bootstrap"]

    assert len(rows) == 2
    assert rows["total_return"].notna().all()
    assert rows["max_drawdown"].notna().all()
    assert rows["tail_5pct_return"].notna().all()
    assert rows["regime_transition_count"].notna().all()


def test_monte_carlo_metrics_are_regime_only_without_synthetic_returns(tmp_path: Path) -> None:
    result = _run(tmp_path)
    metrics = pd.read_csv(result.simulation_stress_metrics_result.path_metrics_path)
    rows = metrics.loc[metrics["simulation_type"] == "regime_transition_monte_carlo"]

    assert len(rows) == 2
    assert not rows["has_return_metrics"].any()
    assert not rows["has_policy_metrics"].any()
    assert rows["total_return"].isna().all()
    assert rows["has_regime_metrics"].all()


def test_failure_thresholds_set_failure_and_deterministic_reasons(tmp_path: Path) -> None:
    result = _run(tmp_path, threshold_overrides={"max_transition_count": 0})
    metrics = pd.read_csv(result.simulation_stress_metrics_result.path_metrics_path)
    failed = metrics.loc[metrics["failure_reason"].fillna("").str.contains("max_transition_count")]

    assert not failed.empty
    assert failed["policy_failure"].all()
    for reason in failed["failure_reason"]:
        assert reason.split(";") == sorted(reason.split(";"), key=reason.split(";").index)


def test_leaderboard_ranking_is_deterministic_with_stable_tie_breakers(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    first_text = first.simulation_stress_metrics_result.leaderboard_path.read_text(encoding="utf-8")
    second_text = second.simulation_stress_metrics_result.leaderboard_path.read_text(encoding="utf-8")
    leaderboard = pd.read_csv(first.simulation_stress_metrics_result.leaderboard_path)

    assert first_text == second_text
    assert leaderboard["tie_breaker"].str.contains(r"\|").all()
    assert not leaderboard["tie_breaker"].str.contains("artifacts|tmp", regex=True).any()


def test_metrics_artifacts_are_written(tmp_path: Path) -> None:
    result = _run(tmp_path)
    metrics = result.simulation_stress_metrics_result

    assert metrics.path_metrics_path.exists()
    assert metrics.summary_path.exists()
    assert metrics.leaderboard_path.exists()
    assert metrics.policy_failure_summary_path.exists()
    assert metrics.manifest_path.exists()


def test_metrics_json_artifacts_do_not_leak_absolute_tmp_paths(tmp_path: Path) -> None:
    result = _run(tmp_path)
    metrics = result.simulation_stress_metrics_result

    for path in (metrics.policy_failure_summary_path, metrics.manifest_path, metrics.metric_config_path):
        text = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "C:\\" not in text


def test_policy_failure_summary_and_manifest_counts(tmp_path: Path) -> None:
    result = _run(tmp_path)
    metrics = result.simulation_stress_metrics_result
    summary = json.loads(metrics.policy_failure_summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(metrics.manifest_path.read_text(encoding="utf-8"))

    assert summary["total_scenarios_evaluated"] == 4
    assert summary["total_paths_evaluated"] == metrics.path_metric_row_count
    assert manifest["artifact_type"] == "simulation_stress_metrics"
    assert manifest["row_counts"]["simulation_leaderboard_csv"] == 4


def test_cli_smoke_writes_metrics_artifacts(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "simulation_metrics.yml"
    config_path.write_text(
        yaml.safe_dump(_payload(tmp_path / "outputs"), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    result = run_cli(["--config", config_path.as_posix()])
    output = capsys.readouterr().out

    assert result.simulation_stress_metrics_result.path_metrics_path.exists()
    assert "Metrics output directory:" in output
    assert "Metrics leaderboard rows: 4" in output
    assert str(tmp_path) not in output
