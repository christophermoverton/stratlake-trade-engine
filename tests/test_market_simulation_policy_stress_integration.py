from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config.regime_policy_stress_tests import RegimePolicyStressTestConfig
from src.research.market_simulation.metrics import (
    LEADERBOARD_COLUMNS,
    PATH_METRIC_COLUMNS,
    SUMMARY_COLUMNS,
)
from src.research.market_simulation.policy_stress_integration import (
    REGIME_ONLY_MONTE_CARLO_NOTE,
    load_market_simulation_stress_summary,
)
from src.research.regime_policy_stress_tests import run_regime_policy_stress_tests
from tests.test_regime_policy_stress_tests import _config, _write_policy_csv, _write_review_pack


def _write_metrics_dir(tmp_path: Path) -> Path:
    metrics_dir = tmp_path / "m27" / "market_simulations" / "simulation_metrics"
    metrics_dir.mkdir(parents=True)

    path_rows = [
        {
            "simulation_run_id": "sim_fixture_001",
            "scenario_id": "historical_001",
            "scenario_name": "historical_fixture",
            "simulation_type": "historical_episode_replay",
            "source_artifact_type": "episode_policy_comparison",
            "path_id": None,
            "episode_id": "episode_1",
            "path_index": None,
            "path_length": 2,
            "row_count": 2,
            "policy_name": "adaptive_policy",
            "tail_quantile": 0.05,
            "has_return_metrics": True,
            "has_policy_metrics": True,
            "has_regime_metrics": False,
            "total_return": 0.01,
            "annualized_return": 0.2,
            "volatility": 0.02,
            "max_drawdown": -0.01,
            "min_period_return": -0.01,
            "max_period_return": 0.02,
            "mean_period_return": 0.005,
            "tail_quantile_return": -0.009,
            "adaptive_return_total": 0.01,
            "static_baseline_return_total": 0.0,
            "adaptive_vs_static_return_delta": 0.01,
            "adaptive_vs_static_win": True,
            "policy_failure": False,
            "failure_reason": "",
            "regime_transition_count": None,
            "unique_regime_count": None,
            "stress_regime_share": None,
            "max_regime_duration": None,
            "mean_regime_duration": None,
            "fallback_activation_count": 0,
            "overlay_count": 0,
            "stress_score": 0.25,
        },
        {
            "simulation_run_id": "sim_fixture_001",
            "scenario_id": "mc_001",
            "scenario_name": "regime_only_mc",
            "simulation_type": "regime_transition_monte_carlo",
            "source_artifact_type": "monte_carlo_regime_paths",
            "path_id": "path_1",
            "episode_id": None,
            "path_index": 1,
            "path_length": 3,
            "row_count": 3,
            "policy_name": "regime_path_only",
            "tail_quantile": 0.05,
            "has_return_metrics": False,
            "has_policy_metrics": False,
            "has_regime_metrics": True,
            "total_return": None,
            "annualized_return": None,
            "volatility": None,
            "max_drawdown": None,
            "min_period_return": None,
            "max_period_return": None,
            "mean_period_return": None,
            "tail_quantile_return": None,
            "adaptive_return_total": None,
            "static_baseline_return_total": None,
            "adaptive_vs_static_return_delta": None,
            "adaptive_vs_static_win": None,
            "policy_failure": True,
            "failure_reason": "max_transition_count",
            "regime_transition_count": 4,
            "unique_regime_count": 2,
            "stress_regime_share": 0.67,
            "max_regime_duration": 2,
            "mean_regime_duration": 1.5,
            "fallback_activation_count": 0,
            "overlay_count": 0,
            "stress_score": 1.5,
        },
    ]
    summary_rows = [
        {
            "simulation_run_id": "sim_fixture_001",
            "scenario_id": "historical_001",
            "scenario_name": "historical_fixture",
            "simulation_type": "historical_episode_replay",
            "path_count": 1,
            "row_count": 2,
            "tail_quantile": 0.05,
            "paths_with_return_metrics": 1,
            "paths_with_policy_metrics": 1,
            "paths_with_regime_metrics": 0,
            "mean_total_return": 0.01,
            "median_total_return": 0.01,
            "tail_quantile_total_return": 0.01,
            "worst_total_return": 0.01,
            "mean_max_drawdown": -0.01,
            "worst_max_drawdown": -0.01,
            "mean_volatility": 0.02,
            "policy_failure_rate": 0.0,
            "adaptive_vs_static_win_rate": 1.0,
            "mean_adaptive_vs_static_delta": 0.01,
            "mean_regime_transition_count": None,
            "mean_stress_regime_share": None,
            "worst_stress_score": 0.25,
            "mean_stress_score": 0.25,
            "ranking_metric": "mean_stress_score",
            "notes": "regime_metrics_unavailable",
        },
        {
            "simulation_run_id": "sim_fixture_001",
            "scenario_id": "mc_001",
            "scenario_name": "regime_only_mc",
            "simulation_type": "regime_transition_monte_carlo",
            "path_count": 1,
            "row_count": 3,
            "tail_quantile": 0.05,
            "paths_with_return_metrics": 0,
            "paths_with_policy_metrics": 0,
            "paths_with_regime_metrics": 1,
            "mean_total_return": None,
            "median_total_return": None,
            "tail_quantile_total_return": None,
            "worst_total_return": None,
            "mean_max_drawdown": None,
            "worst_max_drawdown": None,
            "mean_volatility": None,
            "policy_failure_rate": 1.0,
            "adaptive_vs_static_win_rate": None,
            "mean_adaptive_vs_static_delta": None,
            "mean_regime_transition_count": 4,
            "mean_stress_regime_share": 0.67,
            "worst_stress_score": 1.5,
            "mean_stress_score": 1.5,
            "ranking_metric": "mean_stress_score",
            "notes": "return_metrics_unavailable;policy_metrics_unavailable",
        },
    ]
    leaderboard_rows = [
        {
            "rank": 1,
            "simulation_run_id": "sim_fixture_001",
            "scenario_id": "historical_001",
            "scenario_name": "historical_fixture",
            "simulation_type": "historical_episode_replay",
            "policy_name": "all",
            "path_count": 1,
            "tail_quantile": 0.05,
            "policy_failure_rate": 0.0,
            "adaptive_vs_static_win_rate": 1.0,
            "tail_quantile_total_return": 0.01,
            "worst_max_drawdown": -0.01,
            "mean_stress_score": 0.25,
            "ranking_metric": "mean_stress_score",
            "ranking_value": 0.25,
            "tie_breaker": "historical_fixture|historical_episode_replay|historical_001",
            "decision_label": "monitor",
            "primary_reason": "No configured failure threshold breached.",
        },
        {
            "rank": 2,
            "simulation_run_id": "sim_fixture_001",
            "scenario_id": "mc_001",
            "scenario_name": "regime_only_mc",
            "simulation_type": "regime_transition_monte_carlo",
            "policy_name": "all",
            "path_count": 1,
            "tail_quantile": 0.05,
            "policy_failure_rate": 1.0,
            "adaptive_vs_static_win_rate": None,
            "tail_quantile_total_return": None,
            "worst_max_drawdown": None,
            "mean_stress_score": 1.5,
            "ranking_metric": "mean_stress_score",
            "ranking_value": 1.5,
            "tie_breaker": "regime_only_mc|regime_transition_monte_carlo|mc_001",
            "decision_label": "review",
            "primary_reason": "Policy failure threshold breached in one or more paths.",
        },
    ]

    pd.DataFrame(path_rows).loc[:, PATH_METRIC_COLUMNS].to_csv(
        metrics_dir / "simulation_path_metrics.csv",
        index=False,
    )
    pd.DataFrame(summary_rows).loc[:, SUMMARY_COLUMNS].to_csv(
        metrics_dir / "simulation_summary.csv",
        index=False,
    )
    pd.DataFrame(leaderboard_rows).loc[:, LEADERBOARD_COLUMNS].to_csv(
        metrics_dir / "simulation_leaderboard.csv",
        index=False,
    )
    (metrics_dir / "policy_failure_summary.json").write_text(
        json.dumps({"simulation_run_id": "sim_fixture_001", "policy_failure_rate": 0.5}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    (metrics_dir / "simulation_metric_config.json").write_text(
        json.dumps({"tail_quantile": 0.05}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    (metrics_dir / "manifest.json").write_text(
        json.dumps({"artifact_type": "simulation_stress_metrics", "simulation_run_id": "sim_fixture_001"}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return metrics_dir


def _config_with_market_simulation(
    tmp_path: Path,
    *,
    enabled: bool,
    metrics_dir: Path | None,
) -> RegimePolicyStressTestConfig:
    payload = _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path)).to_dict()
    payload["market_simulation_stress"] = {"enabled": enabled}
    if metrics_dir is not None:
        payload["market_simulation_stress"].update(
            {
                "mode": "existing_artifacts",
                "simulation_metrics_dir": metrics_dir.as_posix(),
                "include_in_policy_stress_summary": True,
                "include_in_case_study_report": True,
            }
        )
    return RegimePolicyStressTestConfig.from_mapping(payload)


def test_absent_market_simulation_stress_config_is_unchanged(tmp_path: Path) -> None:
    config = _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))

    result = run_regime_policy_stress_tests(config)

    assert "market_simulation_stress" not in result.manifest
    assert result.market_simulation_stress_summary_path is None
    assert "market_simulation_stress_summary.json" not in result.manifest["generated_files"]


def test_disabled_market_simulation_stress_is_noop(tmp_path: Path) -> None:
    base = _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    disabled = RegimePolicyStressTestConfig.from_mapping(
        {**base.to_dict(), "market_simulation_stress": {"enabled": False}}
    )

    first = run_regime_policy_stress_tests(base)
    second = run_regime_policy_stress_tests(disabled)

    assert first.stress_run_id == second.stress_run_id
    assert "market_simulation_stress" not in second.manifest


def test_enabled_existing_artifacts_writes_summary_and_leaderboard(tmp_path: Path) -> None:
    metrics_dir = _write_metrics_dir(tmp_path)
    config = _config_with_market_simulation(tmp_path, enabled=True, metrics_dir=metrics_dir)

    result = run_regime_policy_stress_tests(config)

    assert result.market_simulation_stress_summary_path is not None
    assert result.market_simulation_stress_leaderboard_path is not None
    assert result.market_simulation_stress_summary_path.exists()
    assert result.market_simulation_stress_leaderboard_path.exists()
    summary = json.loads(result.market_simulation_stress_summary_path.read_text(encoding="utf-8"))
    assert summary["source_market_simulation_run_id"] == "sim_fixture_001"
    assert summary["simulation_types"] == [
        "historical_episode_replay",
        "regime_transition_monte_carlo",
    ]
    assert summary["policy_failure_rate"] == pytest.approx(0.5)
    assert summary["regime_only_monte_carlo_note"] == REGIME_ONLY_MONTE_CARLO_NOTE
    assert result.manifest["market_simulation_stress"] == summary
    assert "market_simulation_stress" in result.policy_stress_summary


def test_missing_metrics_directory_fails_clearly(tmp_path: Path) -> None:
    config = _config_with_market_simulation(
        tmp_path,
        enabled=True,
        metrics_dir=tmp_path / "missing_metrics",
    )

    with pytest.raises(FileNotFoundError, match="Market simulation metrics directory does not exist"):
        run_regime_policy_stress_tests(config)


def test_missing_required_metric_file_fails_clearly(tmp_path: Path) -> None:
    metrics_dir = _write_metrics_dir(tmp_path)
    (metrics_dir / "simulation_summary.csv").unlink()
    config = _config_with_market_simulation(tmp_path, enabled=True, metrics_dir=metrics_dir)

    with pytest.raises(FileNotFoundError, match="missing required file"):
        run_regime_policy_stress_tests(config)


def test_loader_rejects_missing_required_columns(tmp_path: Path) -> None:
    metrics_dir = _write_metrics_dir(tmp_path)
    pd.DataFrame([{"simulation_run_id": "sim_fixture_001"}]).to_csv(
        metrics_dir / "simulation_summary.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="simulation_summary.csv is missing required column"):
        load_market_simulation_stress_summary(metrics_dir)


def test_market_simulation_policy_stress_integration_is_deterministic(tmp_path: Path) -> None:
    metrics_dir = _write_metrics_dir(tmp_path)
    config = _config_with_market_simulation(tmp_path, enabled=True, metrics_dir=metrics_dir)

    first = run_regime_policy_stress_tests(config)
    second = run_regime_policy_stress_tests(config)

    assert first.stress_run_id == second.stress_run_id
    assert first.market_simulation_stress_summary_path is not None
    assert second.market_simulation_stress_summary_path is not None
    assert first.market_simulation_stress_summary_path.read_text(encoding="utf-8") == (
        second.market_simulation_stress_summary_path.read_text(encoding="utf-8")
    )
