from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research import experiment_tracker
from src.research.consistency import (
    ConsistencyError,
    validate_features_to_signals_consistency,
    validate_run_consistency,
    validate_signals_to_backtest_consistency,
)
from src.research.experiment_tracker import save_experiment, save_walk_forward_experiment
from src.research.metrics import compute_performance_metrics
from src.research.walk_forward import SplitExecutionResult, build_aggregate_summary


def _single_results() -> pd.DataFrame:
    returns = pd.Series([0.0, 0.02, -0.01, 0.0], dtype="float64")
    frame = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "timeframe": ["1D", "1D", "1D", "1D"],
            "signal": [1, 1, 0, 0],
            "strategy_return": returns,
            "equity_curve": (1.0 + returns).cumprod(),
        }
    )
    frame.attrs["dataset"] = "features_daily"
    frame.attrs["timeframe"] = "1D"
    return frame


def _single_config() -> dict[str, object]:
    return {
        "strategy_name": "mean_reversion",
        "dataset": "features_daily",
        "parameters": {"lookback": 20},
        "start": "2022-01-01",
        "end": "2022-01-04",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _create_valid_single_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    artifact_root = tmp_path / "artifacts" / "strategies"
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    results = _single_results()
    metrics = compute_performance_metrics(results)
    return save_experiment("mean_reversion", results, metrics, _single_config())


def _split_result(
    split_id: str,
    dates: list[str],
    signals: list[int],
    returns: list[float],
    *,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> SplitExecutionResult:
    return_series = pd.Series(returns, dtype="float64")
    results_df = pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates),
            "date": dates,
            "timeframe": ["1D"] * len(dates),
            "signal": signals,
            "strategy_return": return_series,
            "equity_curve": (1.0 + return_series).cumprod(),
        }
    )
    results_df.attrs["dataset"] = "features_daily"
    results_df.attrs["timeframe"] = "1D"
    return SplitExecutionResult(
        split_id=split_id,
        mode="rolling",
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        split_rows=3,
        train_rows=2,
        test_rows=len(results_df),
        metrics=compute_performance_metrics(results_df),
        results_df=results_df,
    )


def _create_valid_walk_forward_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    artifact_root = tmp_path / "artifacts" / "strategies"
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    first = _split_result(
        "rolling_0000",
        ["2022-01-03", "2022-01-04"],
        [1, 0],
        [0.01, -0.02],
        train_start="2022-01-01",
        train_end="2022-01-03",
        test_start="2022-01-03",
        test_end="2022-01-05",
    )
    second = _split_result(
        "rolling_0001",
        ["2022-01-05", "2022-01-06"],
        [0, -1],
        [0.0, 0.03],
        train_start="2022-01-02",
        train_end="2022-01-04",
        test_start="2022-01-05",
        test_end="2022-01-07",
    )
    split_results = [first, second]
    aggregate_summary = build_aggregate_summary(split_results)

    return save_walk_forward_experiment(
        "wf_strategy",
        [
            {
                "split_id": result.split_id,
                "split_metadata": {
                    "split_id": result.split_id,
                    "mode": result.mode,
                    "train_start": result.train_start,
                    "train_end": result.train_end,
                    "test_start": result.test_start,
                    "test_end": result.test_end,
                },
                "split_rows": result.split_rows,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "metrics": result.metrics,
                "results_df": result.results_df,
            }
            for result in split_results
        ],
        aggregate_summary,
        {
            "strategy_name": "wf_strategy",
            "dataset": "features_daily",
            "parameters": {"lookback": 10},
            "evaluation_config_path": "configs/evaluation.yml",
            "evaluation": {
                "mode": "rolling",
                "timeframe": "1D",
                "start": "2022-01-01",
                "end": "2022-01-07",
            },
        },
    )


def test_validate_run_consistency_accepts_valid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)

    validate_run_consistency(run_dir)


def test_validate_run_consistency_fails_for_missing_required_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)
    (run_dir / "qa_summary.json").unlink()

    with pytest.raises(ConsistencyError, match="required artifact not found: qa_summary.json"):
        validate_run_consistency(run_dir)


def test_validate_run_consistency_fails_for_metrics_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_return"] = 0.25
    _write_json(run_dir / "metrics.json", metrics)

    with pytest.raises(ConsistencyError, match="total_return mismatch"):
        validate_run_consistency(run_dir)


def test_validate_run_consistency_fails_for_manifest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_files"] = [path for path in manifest["artifact_files"] if path != "metrics.json"]
    _write_json(run_dir / "manifest.json", manifest)

    with pytest.raises(ConsistencyError, match="manifest.json is missing artifact entries"):
        validate_run_consistency(run_dir)


def test_validate_run_consistency_fails_for_registry_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)
    registry_path = run_dir.parent / "registry.jsonl"
    entries = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries[0]["strategy_name"] = "wrong_strategy"
    registry_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")

    with pytest.raises(ConsistencyError, match="registry strategy_name mismatch"):
        validate_run_consistency(run_dir)


def test_validate_run_consistency_fails_for_qa_inconsistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)
    qa_summary = json.loads((run_dir / "qa_summary.json").read_text(encoding="utf-8"))
    qa_summary["row_count"] = 999
    _write_json(run_dir / "qa_summary.json", qa_summary)

    with pytest.raises(ConsistencyError, match="qa_summary.json row_count mismatch"):
        validate_run_consistency(run_dir)


def test_validate_run_consistency_warns_for_minor_signal_diagnostic_issue_in_non_strict_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)
    diagnostics = json.loads((run_dir / "signal_diagnostics.json").read_text(encoding="utf-8"))
    diagnostics["turnover"] = 1.5
    _write_json(run_dir / "signal_diagnostics.json", diagnostics)

    with pytest.warns(UserWarning, match="turnover should be within \\[0, 1\\]"):
        validate_run_consistency(run_dir, strict=False)


def test_validate_run_consistency_accepts_warn_status_for_sanity_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)
    qa_summary = json.loads((run_dir / "qa_summary.json").read_text(encoding="utf-8"))
    qa_summary["flags"]["sanity_warning"] = True
    qa_summary["overall_status"] = "warn"
    _write_json(run_dir / "qa_summary.json", qa_summary)

    validate_run_consistency(run_dir)


def test_validate_run_consistency_accepts_warn_status_for_plausibility_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_single_run(tmp_path, monkeypatch)
    qa_summary = json.loads((run_dir / "qa_summary.json").read_text(encoding="utf-8"))
    qa_summary["flags"]["low_excess_return"] = True
    qa_summary["overall_status"] = "warn"
    _write_json(run_dir / "qa_summary.json", qa_summary)

    validate_run_consistency(run_dir)


def test_validate_run_consistency_fails_for_walk_forward_split_inconsistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _create_valid_walk_forward_run(tmp_path, monkeypatch)
    metrics_by_split = pd.read_csv(run_dir / "metrics_by_split.csv")
    metrics_by_split.loc[0, "total_return"] = 0.5
    metrics_by_split.to_csv(run_dir / "metrics_by_split.csv", index=False)

    with pytest.raises(ConsistencyError, match="metrics_by_split\\.rolling_0000\\.total_return mismatch"):
        validate_run_consistency(run_dir)


def test_validate_features_to_signals_consistency_fails_for_reordered_index() -> None:
    features = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL", "AAPL"], dtype="string"),
            "ts_utc": pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
            "timeframe": pd.Series(["1D", "1D", "1D"], dtype="string"),
        },
        index=pd.Index(["row_a", "row_b", "row_c"], name="row_id"),
    )
    signals = pd.Series([1, 0, -1], index=features.index[::-1], dtype="int64")

    with pytest.raises(ConsistencyError, match="index mismatch"):
        validate_features_to_signals_consistency(features, signals)


def test_validate_signals_to_backtest_consistency_fails_for_dropped_row() -> None:
    signal_df = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL", "AAPL"], dtype="string"),
            "ts_utc": pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
            "signal": [1, 0, -1],
            "feature_ret_1d": [0.01, 0.02, -0.01],
        }
    )
    backtest_df = signal_df.iloc[:2].copy()
    backtest_df["executed_signal"] = [0.0, 1.0]

    with pytest.raises(ConsistencyError, match="row count mismatch"):
        validate_signals_to_backtest_consistency(signal_df, backtest_df, return_column="feature_ret_1d")


def test_save_experiment_blocks_artifact_write_when_consistency_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    results = _single_results()
    metrics = compute_performance_metrics(results)
    metrics["total_return"] = 999.0

    with pytest.raises(ConsistencyError, match="total_return mismatch"):
        save_experiment("mean_reversion", results, metrics, _single_config())

    assert not artifact_root.exists()
