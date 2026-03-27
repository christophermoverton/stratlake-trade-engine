from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.robustness import RobustnessConfig, load_robustness_config
from src.research import experiment_tracker
from src.research.robustness import (
    RobustnessAnalysisError,
    expand_parameter_sweeps,
    run_robustness_experiment,
)


def _write_daily_features_dataset(root: Path, symbol: str = "AAPL", periods: int = 60) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    close_values = [100.0]
    for index in range(1, periods):
        drift = 0.8 if index % 6 not in {0, 1} else -1.2
        close_values.append(close_values[-1] + drift)
    closes = pd.Series(close_values, dtype="float64")
    feature_df = pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * periods, dtype="string"),
            "ts_utc": ts,
            "timeframe": pd.Series(["1D"] * periods, dtype="string"),
            "date": pd.Series(ts.strftime("%Y-%m-%d"), dtype="string"),
            "close": closes,
            "feature_ret_1d": closes.div(closes.shift(1)).sub(1.0).fillna(0.0),
        }
    )
    dataset_path = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
    dataset_path.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(dataset_path / "part-0.parquet", index=False)
    return feature_df


def test_expand_parameter_sweeps_is_deterministic_for_multi_parameter_inputs() -> None:
    strategy_config = {
        "dataset": "features_daily",
        "parameters": {"lookback_short": 5, "lookback_long": 20},
    }
    robustness_config = RobustnessConfig.from_mapping(
        {
            "ranking_metric": "sharpe_ratio",
            "sweep": [
                {"parameter": "lookback_short", "values": [5, 10]},
                {"parameter": "lookback_long", "values": [20, 30]},
            ],
        }
    )

    variants = expand_parameter_sweeps("momentum_v1", strategy_config, robustness_config)

    assert [variant.variant_id for variant in variants] == [
        "variant_0000_lookback_short_5__lookback_long_20",
        "variant_0001_lookback_short_5__lookback_long_30",
        "variant_0002_lookback_short_10__lookback_long_20",
        "variant_0003_lookback_short_10__lookback_long_30",
    ]
    assert [variant.parameter_values for variant in variants] == [
        {"lookback_short": 5, "lookback_long": 20},
        {"lookback_short": 5, "lookback_long": 30},
        {"lookback_short": 10, "lookback_long": 20},
        {"lookback_short": 10, "lookback_long": 30},
    ]


def test_expand_parameter_sweeps_rejects_invalid_or_duplicate_inputs() -> None:
    strategy_config = {
        "dataset": "features_daily",
        "parameters": {"lookback_short": 5, "lookback_long": 20},
    }

    invalid_parameter = RobustnessConfig.from_mapping(
        {
            "ranking_metric": "sharpe_ratio",
            "sweep": [{"parameter": "threshold", "values": [1.0, 2.0]}],
        }
    )
    with pytest.raises(RobustnessAnalysisError, match="is not defined"):
        expand_parameter_sweeps("momentum_v1", strategy_config, invalid_parameter)

    duplicate_values = RobustnessConfig.from_mapping(
        {
            "ranking_metric": "sharpe_ratio",
            "sweep": [{"parameter": "lookback_short", "values": [5, 5]}],
        }
    )
    with pytest.raises(RobustnessAnalysisError, match="contains duplicate values"):
        expand_parameter_sweeps("momentum_v1", strategy_config, duplicate_values)


def test_load_robustness_config_parses_top_level_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "robustness.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robustness": {
                    "strategy_name": "momentum_v1",
                    "ranking_metric": "sharpe_ratio",
                    "sweep": [{"parameter": "lookback_short", "values": [5, 10]}],
                    "stability": {"mode": "subperiods", "periods": 3},
                    "thresholds": {"sharpe_ratio": {"min": 0.0}},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_robustness_config(config_path)

    assert config.strategy_name == "momentum_v1"
    assert config.stability.mode == "subperiods"
    assert config.stability.periods == 3
    assert config.thresholds[0].metric == "sharpe_ratio"
    assert config.thresholds[0].min_value == pytest.approx(0.0)


def test_run_robustness_experiment_writes_deterministic_subperiod_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_daily_features_dataset(tmp_path)
    artifact_root = tmp_path / "artifacts" / "strategies"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(experiment_tracker, "ROBUSTNESS_ARTIFACTS_ROOT", artifact_root / "robustness")

    robustness_config = RobustnessConfig.from_mapping(
        {
            "strategy_name": "momentum_v1",
            "ranking_metric": "sharpe_ratio",
            "sweep": [
                {"parameter": "lookback_short", "values": [3, 5]},
                {"parameter": "lookback_long", "values": [12, 18]},
            ],
            "stability": {"mode": "subperiods", "periods": 3},
            "thresholds": {"sharpe_ratio": {"min": -5.0}},
        }
    )

    first = run_robustness_experiment("momentum_v1", robustness_config=robustness_config)
    second = run_robustness_experiment("momentum_v1", robustness_config=robustness_config)

    assert first.run_id == second.run_id
    assert first.summary == second.summary
    assert first.summary["variant_count"] == 4
    assert first.summary["split_count"] == 3
    assert [variant.variant_id for variant in first.variants] == [
        "variant_0000_lookback_short_3__lookback_long_12",
        "variant_0001_lookback_short_3__lookback_long_18",
        "variant_0002_lookback_short_5__lookback_long_12",
        "variant_0003_lookback_short_5__lookback_long_18",
    ]
    assert (first.experiment_dir / "config.json").exists()
    assert (first.experiment_dir / "variants.json").exists()
    assert (first.experiment_dir / "summary.json").exists()
    assert (first.experiment_dir / "metrics_by_variant.csv").exists()
    assert (first.experiment_dir / "stability_metrics.csv").exists()
    assert (first.experiment_dir / "neighbor_metrics.csv").exists()
    manifest = json.loads((first.experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_mode"] == "robustness"
    assert manifest["variant_count"] == 4
    assert manifest["split_count"] == 3
    assert "metrics_by_variant.csv" in manifest["artifact_files"]
    assert "stability_metrics.csv" in manifest["artifact_files"]
    assert "neighbor_metrics.csv" in manifest["artifact_files"]

    first_snapshot = {
        path.relative_to(first.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(first.experiment_dir.rglob("*"))
        if path.is_file()
    }
    second_snapshot = {
        path.relative_to(second.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(second.experiment_dir.rglob("*"))
        if path.is_file()
    }
    assert first_snapshot == second_snapshot


def test_run_robustness_experiment_supports_walk_forward_stability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_daily_features_dataset(tmp_path)
    artifact_root = tmp_path / "artifacts" / "strategies"
    evaluation_path = tmp_path / "evaluation.yml"
    evaluation_path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "mode": "rolling",
                    "timeframe": "1d",
                    "start": "2025-01-01",
                    "end": "2025-02-20",
                    "train_window": "20D",
                    "test_window": "10D",
                    "step": "10D",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(experiment_tracker, "ROBUSTNESS_ARTIFACTS_ROOT", artifact_root / "robustness")

    result = run_robustness_experiment(
        "momentum_v1",
        robustness_config=RobustnessConfig.from_mapping(
            {
                "strategy_name": "momentum_v1",
                "ranking_metric": "sharpe_ratio",
                "sweep": [{"parameter": "lookback_short", "values": [3, 5]}],
                "stability": {"mode": "walk_forward", "evaluation_path": str(evaluation_path)},
            }
        ),
    )

    assert result.summary["stability_mode"] == "walk_forward"
    assert result.summary["split_count"] >= 1
    assert set(result.stability_metrics["period_id"]) == {"rolling_0000", "rolling_0001", "rolling_0002"}
    assert "period_rank" in result.stability_metrics.columns
