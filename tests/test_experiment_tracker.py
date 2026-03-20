from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research import experiment_tracker
from src.research.experiment_tracker import save_experiment, save_walk_forward_experiment


def _metrics() -> dict[str, float | None]:
    return {
        "cumulative_return": 0.0098,
        "total_return": 0.0098,
        "volatility": 0.012,
        "annualized_return": 0.19,
        "annualized_volatility": 0.21,
        "sharpe_ratio": 0.41,
        "max_drawdown": 0.05,
        "win_rate": 0.5,
        "hit_rate": 0.5,
        "profit_factor": 1.1,
        "turnover": 0.33,
        "exposure_pct": 66.7,
    }


def _experiment_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "feature_alpha": [0.15, -0.10, 0.25, 0.05],
            "signal": [1, 1, 0, 0],
            "strategy_return": [0.0, 0.02, -0.01, 0.0],
            "equity_curve": [1.0, 1.02, 1.0098, 1.0098],
        },
        index=pd.Index(["row_a", "row_b", "row_c", "row_d"], name="row_id"),
    )


def test_save_experiment_writes_standardized_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config = {"strategy_name": "mean_reversion", "lookback": 20, "threshold": 0.75}
    results_df = _experiment_results()

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_experiment("mean_reversion", results_df, _metrics(), config)

    assert experiment_dir.exists()
    assert experiment_dir.parent == artifact_root

    expected_files = {
        "config.json",
        "metrics.json",
        "equity_curve.csv",
        "equity_curve.parquet",
        "signals.parquet",
        "trades.parquet",
        "manifest.json",
    }
    assert expected_files.issubset({path.name for path in experiment_dir.iterdir() if path.is_file()})

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == experiment_dir.name
    assert manifest["strategy_name"] == "mean_reversion"
    assert manifest["evaluation_mode"] == "single"
    assert manifest["evaluation_config_path"] is None
    assert manifest["split_count"] is None
    assert manifest["primary_metric"] == "sharpe_ratio"
    assert "config.json" in manifest["artifact_files"]
    assert "equity_curve.csv" in manifest["artifact_files"]
    assert "trades.parquet" in manifest["artifact_files"]

    assert json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8")) == _metrics()
    assert json.loads((experiment_dir / "config.json").read_text(encoding="utf-8")) == config

    signals_df = pd.read_parquet(experiment_dir / "signals.parquet")
    equity_curve_df = pd.read_csv(experiment_dir / "equity_curve.csv")
    legacy_equity_curve_df = pd.read_parquet(experiment_dir / "equity_curve.parquet")
    trades_df = pd.read_parquet(experiment_dir / "trades.parquet")

    assert list(signals_df.columns[:5]) == ["ts_utc", "date", "symbol", "signal", "position"]
    assert signals_df["ts_utc"].tolist() == [
        "2022-01-01T00:00:00Z",
        "2022-01-02T00:00:00Z",
        "2022-01-03T00:00:00Z",
        "2022-01-04T00:00:00Z",
    ]
    assert signals_df["position"].tolist() == [0.0, 1.0, 1.0, 0.0]

    assert list(equity_curve_df.columns) == ["ts_utc", "symbol", "equity", "strategy_return", "signal", "position"]
    assert equity_curve_df["ts_utc"].tolist() == signals_df["ts_utc"].tolist()
    assert equity_curve_df["equity"].tolist() == [1.0, 1.02, 1.0098, 1.0098]
    assert equity_curve_df["position"].tolist() == [0.0, 1.0, 1.0, 0.0]

    assert list(legacy_equity_curve_df.columns) == ["signal", "strategy_return", "equity_curve"]
    assert trades_df.to_dict(orient="records") == [
        {
            "entry_ts_utc": "2022-01-02T00:00:00Z",
            "exit_ts_utc": "2022-01-03T00:00:00Z",
            "symbol": "SPY",
            "direction": "long",
            "return": pytest.approx(0.0098),
        }
    ]


def test_save_experiment_handles_empty_results_without_optional_trades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    empty_results = pd.DataFrame(
        columns=["ts_utc", "symbol", "signal", "strategy_return", "equity_curve"]
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_experiment(
        "empty_strategy",
        empty_results,
        _metrics(),
        {"strategy_name": "empty_strategy"},
    )

    assert (experiment_dir / "signals.parquet").exists()
    assert (experiment_dir / "equity_curve.csv").exists()
    assert (experiment_dir / "manifest.json").exists()
    assert not (experiment_dir / "trades.parquet").exists()

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "trades.parquet" not in manifest["artifact_files"]


def test_save_walk_forward_experiment_supports_empty_split_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config = {
        "strategy_name": "empty_walk_forward",
        "evaluation_config_path": "configs/evaluation.yml",
        "evaluation": {"mode": "rolling"},
    }

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_walk_forward_experiment(
        "empty_walk_forward",
        [],
        {"split_count": 0, **_metrics()},
        config,
    )

    metrics_by_split = pd.read_csv(experiment_dir / "metrics_by_split.csv")
    assert metrics_by_split.empty
    assert list(metrics_by_split.columns[:9]) == [
        "split_id",
        "mode",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "split_rows",
        "train_rows",
        "test_rows",
    ]

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_mode"] == "walk_forward"
    assert manifest["split_count"] == 0
    assert "metrics_by_split.csv" in manifest["artifact_files"]
