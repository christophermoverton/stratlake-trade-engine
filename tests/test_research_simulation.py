from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.simulation import (
    SimulationError,
    evaluate_return_path,
    run_return_simulation,
    write_simulation_artifacts,
)


def _returns() -> pd.Series:
    return pd.Series(
        [0.01, -0.02, 0.03, -0.01],
        index=pd.date_range("2025-01-01", periods=4, tz="UTC"),
        dtype="float64",
    )


def test_bootstrap_simulation_is_deterministic_and_preserves_path_order() -> None:
    config = {"method": "bootstrap", "num_paths": 3, "path_length": 4, "seed": 7}

    first = run_return_simulation(_returns(), config=config, periods_per_year=252)
    second = run_return_simulation(_returns(), config=config, periods_per_year=252)

    pdt.assert_frame_equal(first.paths, second.paths)
    pdt.assert_frame_equal(first.path_metrics, second.path_metrics)
    assert first.summary == second.summary
    assert first.paths["path_id"].drop_duplicates().tolist() == ["path_0000", "path_0001", "path_0002"]
    assert first.paths.groupby("path_id").size().tolist() == [4, 4, 4]
    assert "source_observation_index" in first.paths.columns


def test_bootstrap_simulation_rejects_non_finite_inputs() -> None:
    with pytest.raises(SimulationError, match="finite numeric values"):
        run_return_simulation(
            pd.Series([0.01, float("nan")], dtype="float64"),
            config={"method": "bootstrap", "num_paths": 2, "seed": 1},
        )


def test_monte_carlo_simulation_is_deterministic_for_fixed_seed() -> None:
    config = {
        "method": "monte_carlo",
        "num_paths": 2,
        "path_length": 3,
        "seed": 11,
        "monte_carlo_mean": 0.01,
        "monte_carlo_volatility": 0.02,
    }

    first = run_return_simulation(_returns(), config=config, periods_per_year=252)
    second = run_return_simulation(_returns(), config=config, periods_per_year=252)

    pdt.assert_frame_equal(first.paths, second.paths)
    pdt.assert_frame_equal(first.path_metrics, second.path_metrics)
    assert first.summary["assumptions"]["parameter_source"] == "explicit"


def test_monte_carlo_simulation_rejects_insufficient_samples_for_estimation() -> None:
    with pytest.raises(SimulationError, match="requires sufficient return observations"):
        run_return_simulation(
            pd.Series([0.01], dtype="float64"),
            config={"method": "monte_carlo", "num_paths": 2, "seed": 3, "min_samples": 2},
        )


def test_evaluate_return_path_uses_existing_metric_primitives() -> None:
    metrics = evaluate_return_path(
        pd.Series([0.10, -0.05], dtype="float64"),
        periods_per_year=252,
        var_confidence_level=0.95,
        cvar_confidence_level=0.95,
    )

    assert metrics["cumulative_return"] == pytest.approx(0.045)
    assert metrics["max_drawdown"] == pytest.approx(0.05)
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["value_at_risk"] == pytest.approx(0.05)
    assert metrics["conditional_value_at_risk"] == pytest.approx(0.05)
    assert metrics["final_equity"] == pytest.approx(1.045)


def test_simulation_summary_reports_loss_probability_and_drawdown_exceedance() -> None:
    result = run_return_simulation(
        _returns(),
        config={
            "method": "bootstrap",
            "num_paths": 5,
            "path_length": 4,
            "seed": 5,
            "drawdown_threshold": 0.02,
        },
        periods_per_year=252,
    )

    assert 0.0 <= result.summary["probability_of_loss"] <= 1.0
    assert 0.0 <= result.summary["drawdown_exceedance_probability"] <= 1.0
    assert "cumulative_return" in result.summary["metric_statistics"]
    assert "p05" in result.summary["metric_statistics"]["cumulative_return"]


def test_write_simulation_artifacts_updates_parent_manifest(tmp_path: Path) -> None:
    parent_dir = tmp_path / "strategy_run"
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "manifest.json").write_text(
        json.dumps({"artifact_files": ["config.json"], "run_id": "run-1"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = run_return_simulation(
        _returns(),
        config={"method": "bootstrap", "num_paths": 2, "path_length": 3, "seed": 1},
        periods_per_year=252,
    )
    manifest = write_simulation_artifacts(parent_dir / "simulation", result, parent_manifest_dir=parent_dir)

    assert sorted(manifest["artifact_files"]) == [
        "assumptions.json",
        "config.json",
        "manifest.json",
        "path_metrics.csv",
        "simulated_paths.csv",
        "summary.json",
    ]
    parent_manifest = json.loads((parent_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "simulation/config.json" in parent_manifest["artifact_files"]
    assert parent_manifest["simulation"]["summary_path"] == "simulation/summary.json"
