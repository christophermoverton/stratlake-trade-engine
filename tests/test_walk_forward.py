from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research import experiment_tracker
from src.research.splits import EvaluationSplit
from src.research.strategy_base import BaseStrategy
from src.research.walk_forward import (
    WalkForwardExecutionError,
    build_aggregate_summary,
    execute_split,
    run_walk_forward_experiment,
)


class SignStrategy(BaseStrategy):
    name = "sign_v1"
    dataset = "features_daily"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return df["feature_ret_1d"].apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0)).rename(
            "signal"
        )


def _feature_frame() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=7, freq="D")
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates),
            "ts_utc": pd.to_datetime(dates, utc=True),
            "timeframe": ["1d"] * len(dates),
            "date": dates.strftime("%Y-%m-%d"),
            "feature_ret_1d": [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03],
        }
    )


def _write_evaluation_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"evaluation": payload}, sort_keys=False), encoding="utf-8")


def _expected_metric_keys() -> set[str]:
    return {
        "cumulative_return",
        "total_return",
        "volatility",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "hit_rate",
        "profit_factor",
        "turnover",
        "total_turnover",
        "average_turnover",
        "trade_count",
        "rebalance_count",
        "percent_periods_traded",
        "average_trade_size",
        "total_transaction_cost",
        "total_slippage_cost",
        "total_execution_friction",
        "average_execution_friction_per_trade",
        "exposure_pct",
    }


def test_execute_split_scores_only_test_window_and_preserves_metadata() -> None:
    dataset = _feature_frame()
    strategy = SignStrategy()
    split = EvaluationSplit(
        split_id="fixed_0000",
        mode="fixed",
        train_start="2022-01-01",
        train_end="2022-01-04",
        test_start="2022-01-04",
        test_end="2022-01-06",
    )

    result = execute_split(strategy, dataset, split)

    assert result.train_rows == 3
    assert result.test_rows == 2
    assert result.split_rows == 5
    assert result.results_df["date"].tolist() == ["2022-01-04", "2022-01-05"]
    assert result.results_df["split_id"].tolist() == ["fixed_0000", "fixed_0000"]
    assert result.results_df["train_start"].tolist() == ["2022-01-01", "2022-01-01"]
    assert _expected_metric_keys().issubset(result.metrics)
    assert result.metrics["sanity_issue_count"] == 0.0
    assert result.metrics["cumulative_return"] == pytest.approx(-0.0298)


def test_build_aggregate_summary_uses_concatenated_split_test_returns() -> None:
    dataset = _feature_frame()
    strategy = SignStrategy()
    first = execute_split(
        strategy,
        dataset,
        EvaluationSplit(
            split_id="rolling_0000",
            mode="rolling",
            train_start="2022-01-01",
            train_end="2022-01-03",
            test_start="2022-01-03",
            test_end="2022-01-04",
        ),
    )
    second = execute_split(
        strategy,
        dataset,
        EvaluationSplit(
            split_id="rolling_0001",
            mode="rolling",
            train_start="2022-01-02",
            train_end="2022-01-04",
            test_start="2022-01-04",
            test_end="2022-01-05",
        ),
    )

    summary = build_aggregate_summary([first, second])

    assert summary["split_count"] == 2
    assert summary["mode"] == "rolling"
    assert summary["test_rows"] == 2
    assert summary["aggregation_method"] == "metrics computed on concatenated split test windows in split order"
    assert _expected_metric_keys().issubset(summary)
    assert summary["cumulative_return"] == pytest.approx(-0.0397)


def test_run_walk_forward_experiment_writes_split_and_aggregate_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        config_path,
        {
            "mode": "rolling",
            "timeframe": "1d",
            "start": "2022-01-01",
            "end": "2022-01-07",
            "train_window": "2D",
            "test_window": "1D",
            "step": "1D",
        },
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr("src.research.walk_forward.load_features", lambda dataset, start=None, end=None: _feature_frame())

    result = run_walk_forward_experiment(
        "sign_v1",
        SignStrategy(),
        evaluation_path=config_path,
        strategy_config={"parameters": {"lookback": 2}},
    )

    assert result.run_id == result.experiment_dir.name
    assert result.aggregate_summary["split_count"] == 4
    assert len(result.splits) == 4
    assert (result.experiment_dir / "metrics.json").exists()
    assert (result.experiment_dir / "config.json").exists()
    assert (result.experiment_dir / "manifest.json").exists()
    assert (result.experiment_dir / "equity_curve.csv").exists()
    assert (result.experiment_dir / "metrics_by_split.csv").exists()
    assert (result.experiment_dir / "splits" / "rolling_0000" / "signals.parquet").exists()
    assert (result.experiment_dir / "splits" / "rolling_0000" / "equity_curve.csv").exists()
    assert (result.experiment_dir / "splits" / "rolling_0000" / "equity_curve.parquet").exists()
    assert (result.experiment_dir / "splits" / "rolling_0000" / "metrics.json").exists()
    assert (result.experiment_dir / "splits" / "rolling_0000" / "split.json").exists()

    metrics_by_split = pd.read_csv(result.experiment_dir / "metrics_by_split.csv")
    assert metrics_by_split["split_id"].tolist() == [
        "rolling_0000",
        "rolling_0001",
        "rolling_0002",
        "rolling_0003",
    ]
    assert metrics_by_split["train_rows"].tolist() == [2, 2, 2, 2]
    assert metrics_by_split["test_rows"].tolist() == [1, 1, 1, 1]
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
    for metric_name in _expected_metric_keys():
        assert metric_name in metrics_by_split.columns

    aggregate_metrics = json.loads((result.experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    assert aggregate_metrics["split_count"] == 4
    assert _expected_metric_keys().issubset(aggregate_metrics)
    assert aggregate_metrics["cumulative_return"] == pytest.approx(result.metrics["cumulative_return"])

    manifest = json.loads((result.experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == result.run_id
    assert manifest["strategy_name"] == "sign_v1"
    assert manifest["evaluation_mode"] == "walk_forward"
    assert manifest["evaluation_config_path"] == str(config_path)
    assert manifest["split_count"] == 4
    assert "metrics_by_split.csv" in manifest["artifact_files"]
    assert "splits/rolling_0000/equity_curve.csv" in manifest["artifact_files"]
    assert "splits/rolling_0000/signals.parquet" in manifest["artifact_files"]

    aggregate_equity_curve = pd.read_csv(result.experiment_dir / "equity_curve.csv")
    split_equity_curve = pd.read_csv(result.experiment_dir / "splits" / "rolling_0000" / "equity_curve.csv")
    split_signals = pd.read_parquet(result.experiment_dir / "splits" / "rolling_0000" / "signals.parquet")
    split_metadata = json.loads(
        (result.experiment_dir / "splits" / "rolling_0000" / "split.json").read_text(encoding="utf-8")
    )

    assert aggregate_equity_curve.columns.tolist() == [
        "ts_utc",
        "symbol",
        "equity",
        "strategy_return",
        "signal",
        "position",
    ]
    assert aggregate_equity_curve["ts_utc"].tolist() == sorted(aggregate_equity_curve["ts_utc"].tolist())
    assert split_equity_curve.columns.tolist() == aggregate_equity_curve.columns.tolist()
    assert split_signals.columns.tolist()[:6] == ["ts_utc", "date", "symbol", "signal", "executed_signal", "position"]
    assert split_signals["split_id"].tolist() == ["rolling_0000"]
    assert split_metadata == {
        "split_id": "rolling_0000",
        "mode": "rolling",
        "train_start": "2022-01-01",
        "train_end": "2022-01-03",
        "test_start": "2022-01-03",
        "test_end": "2022-01-04",
    }


def test_run_walk_forward_experiment_rejects_invalid_evaluation_config(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation.yml"
    config_path.write_text(yaml.safe_dump({"not_evaluation": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="must define an 'evaluation' mapping"):
        run_walk_forward_experiment("sign_v1", SignStrategy(), evaluation_path=config_path)


def test_run_walk_forward_experiment_rejects_empty_split_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        config_path,
        {
            "mode": "fixed",
            "timeframe": "1d",
            "train_start": "2022-01-01",
            "train_end": "2022-01-03",
            "test_start": "2022-01-03",
            "test_end": "2022-01-05",
        },
    )
    sparse_frame = _feature_frame().loc[lambda df: df["date"] >= "2022-01-03"].reset_index(drop=True)
    monkeypatch.setattr("src.research.walk_forward.load_features", lambda dataset, start=None, end=None: sparse_frame)

    with pytest.raises(WalkForwardExecutionError, match="produced no training rows"):
        run_walk_forward_experiment("sign_v1", SignStrategy(), evaluation_path=config_path)


def test_run_walk_forward_experiment_reports_flagged_splits_in_non_strict_sanity_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        config_path,
        {
            "mode": "rolling",
            "timeframe": "1d",
            "start": "2022-01-01",
            "end": "2022-01-07",
            "train_window": "2D",
            "test_window": "1D",
            "step": "1D",
        },
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr("src.research.walk_forward.load_features", lambda dataset, start=None, end=None: _feature_frame())

    result = run_walk_forward_experiment(
        "sign_v1",
        SignStrategy(),
        evaluation_path=config_path,
        strategy_config={
            "parameters": {"lookback": 2},
            "sanity": {
                "strict_sanity_checks": False,
                "max_abs_period_return": 0.005,
            },
        },
    )

    assert result.aggregate_summary["flagged_split_count"] >= 1
    assert result.sanity_summary["flagged_split_count"] >= 1
    assert any(split.sanity["issue_count"] > 0 for split in result.splits)

    metrics_payload = json.loads((result.experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["flagged_split_count"] >= 1
    assert metrics_payload["sanity_status"] in {"pass", "warn"}
