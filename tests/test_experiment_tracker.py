from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research import experiment_tracker
from src.research.experiment_tracker import save_experiment


def _experiment_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_alpha": [0.15, -0.10, 0.25],
            "signal": [1, 0, -1],
            "strategy_return": [0.0, 0.02, -0.01],
            "equity_curve": [1.0, 1.02, 1.0098],
        },
        index=pd.Index(["row_a", "row_b", "row_c"], name="row_id"),
    )


def test_save_experiment_writes_expected_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    metrics = {"cumulative_return": 0.0098, "sharpe_ratio": 0.41}
    config = {"lookback": 20, "threshold": 0.75}
    results_df = _experiment_results()

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_experiment("mean_reversion", results_df, metrics, config)

    assert experiment_dir.exists()
    assert experiment_dir.parent == artifact_root

    signals_path = experiment_dir / "signals.parquet"
    equity_curve_path = experiment_dir / "equity_curve.parquet"
    metrics_path = experiment_dir / "metrics.json"
    config_path = experiment_dir / "config.json"

    assert signals_path.exists()
    assert equity_curve_path.exists()
    assert metrics_path.exists()
    assert config_path.exists()

    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    assert json.loads(config_path.read_text(encoding="utf-8")) == config

    signals_df = pd.read_parquet(signals_path)
    equity_curve_df = pd.read_parquet(equity_curve_path)

    assert list(signals_df.columns) == ["feature_alpha", "signal"]
    assert list(equity_curve_df.columns) == ["signal", "strategy_return", "equity_curve"]
    assert signals_df.index.tolist() == results_df.index.tolist()
    assert equity_curve_df.index.tolist() == results_df.index.tolist()
