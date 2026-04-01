from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.alpha_eval import (
    alpha_evaluation_registry_path,
    evaluate_alpha_predictions,
    filter_by_alpha_name,
    filter_by_timeframe,
    get_alpha_evaluation_run,
    load_alpha_evaluation_registry,
    register_alpha_evaluation_run,
    write_alpha_evaluation_artifacts,
)
from src.research.registry import filter_by_metric_threshold, load_registry


def _alpha_eval_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "timeframe": ["1d", "1d", "1d", "1d", "1d", "1d"],
            "prediction_score": [0.1, 0.6, 0.2, 0.5, 0.3, 0.4],
            "forward_return": [0.2, 0.3, 0.1, 0.4, 0.0, 0.2],
        }
    )


def test_register_alpha_evaluation_run_writes_stable_registry_entry(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    result = evaluate_alpha_predictions(_alpha_eval_frame())
    artifact_dir = artifacts_root / "demo_alpha_eval_run"
    manifest = write_alpha_evaluation_artifacts(
        artifact_dir,
        result,
        run_id="demo_alpha_eval_run",
        alpha_name="demo_alpha",
    )
    effective_config = {
        "alpha_model": "demo_alpha",
        "dataset": "features_daily",
        "target_column": "target_ret_1d",
        "price_column": "close",
        "alpha_horizon": 3,
        "min_cross_section_size": 2,
    }

    entry = register_alpha_evaluation_run(
        run_id="demo_alpha_eval_run",
        alpha_name="demo_alpha",
        effective_config=effective_config,
        evaluation_result=result,
        artifact_dir=artifact_dir,
        manifest=manifest,
    )

    registry_path = alpha_evaluation_registry_path(artifacts_root)
    entries = load_registry(registry_path)

    assert entry["run_id"] == "demo_alpha_eval_run"
    assert len(entries) == 1
    assert entries[0] == entry
    assert entry["run_type"] == "alpha_evaluation"
    assert entry["alpha_name"] == "demo_alpha"
    assert entry["dataset"] == "features_daily"
    assert entry["timeframe"] == "1d"
    assert entry["evaluation_horizon"] == 3
    assert entry["artifact_path"] == artifact_dir.as_posix()
    assert entry["ic_timeseries_path"] == (artifact_dir / "ic_timeseries.csv").as_posix()
    assert entry["metrics_path"] == (artifact_dir / "alpha_metrics.json").as_posix()
    assert entry["manifest_path"] == (artifact_dir / "manifest.json").as_posix()
    assert entry["data_range"] == {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-02T00:00:00Z",
    }
    assert entry["metrics_summary"] == {
        "mean_ic": result.summary["mean_ic"],
        "ic_ir": result.summary["ic_ir"],
        "mean_rank_ic": result.summary["mean_rank_ic"],
        "rank_ic_ir": result.summary["rank_ic_ir"],
        "n_periods": result.summary["n_periods"],
    }
    assert entry["manifest"]["metrics_path"] == "alpha_metrics.json"
    assert entry["metadata"]["artifact_scaffold"]["ic_timeseries"] == "ic_timeseries.csv"


def test_register_alpha_evaluation_run_upserts_identical_reruns_without_duplicates(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    result = evaluate_alpha_predictions(_alpha_eval_frame())
    artifact_dir = artifacts_root / "demo_alpha_eval_run"
    manifest = write_alpha_evaluation_artifacts(
        artifact_dir,
        result,
        run_id="demo_alpha_eval_run",
        alpha_name="demo_alpha",
    )
    effective_config = {
        "alpha_model": "demo_alpha",
        "dataset": "features_daily",
        "alpha_horizon": 1,
        "price_column": "close",
        "target_column": "target_ret_1d",
    }

    register_alpha_evaluation_run(
        run_id="demo_alpha_eval_run",
        alpha_name="demo_alpha",
        effective_config=effective_config,
        evaluation_result=result,
        artifact_dir=artifact_dir,
        manifest=manifest,
    )
    first_snapshot = alpha_evaluation_registry_path(artifacts_root).read_text(encoding="utf-8")

    register_alpha_evaluation_run(
        run_id="demo_alpha_eval_run",
        alpha_name="demo_alpha",
        effective_config=effective_config,
        evaluation_result=result,
        artifact_dir=artifact_dir,
        manifest=manifest,
    )
    second_snapshot = alpha_evaluation_registry_path(artifacts_root).read_text(encoding="utf-8")

    assert first_snapshot == second_snapshot
    assert len(load_alpha_evaluation_registry(artifacts_root)) == 1


def test_alpha_evaluation_registry_helpers_support_lightweight_queries(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    registry_path = alpha_evaluation_registry_path(artifacts_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": "alpha-run-a",
            "run_type": "alpha_evaluation",
            "timestamp": "2026-03-19T00:00:00Z",
            "alpha_name": "demo_alpha",
            "dataset": "features_daily",
            "timeframe": "1d",
            "evaluation_horizon": 1,
            "artifact_path": "artifacts/alpha/alpha-run-a",
            "ic_timeseries_path": "artifacts/alpha/alpha-run-a/ic_timeseries.csv",
            "metrics_path": "artifacts/alpha/alpha-run-a/alpha_metrics.json",
            "manifest_path": "artifacts/alpha/alpha-run-a/manifest.json",
            "metrics_summary": {"mean_ic": 0.4, "ic_ir": 1.2, "mean_rank_ic": 0.3, "rank_ic_ir": 1.1, "n_periods": 4},
        },
        {
            "run_id": "alpha-run-b",
            "run_type": "alpha_evaluation",
            "timestamp": "2026-03-19T00:01:00Z",
            "alpha_name": "other_alpha",
            "dataset": "features_minute",
            "timeframe": "1Min",
            "evaluation_horizon": 5,
            "artifact_path": "artifacts/alpha/alpha-run-b",
            "ic_timeseries_path": "artifacts/alpha/alpha-run-b/ic_timeseries.csv",
            "metrics_path": "artifacts/alpha/alpha-run-b/alpha_metrics.json",
            "manifest_path": "artifacts/alpha/alpha-run-b/manifest.json",
            "metrics_summary": {"mean_ic": 0.1, "ic_ir": 0.4, "mean_rank_ic": 0.2, "rank_ic_ir": 0.5, "n_periods": 8},
        },
    ]
    registry_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    entries = load_alpha_evaluation_registry(artifacts_root)

    assert [entry["run_id"] for entry in filter_by_alpha_name(entries, "demo_alpha")] == ["alpha-run-a"]
    assert [entry["run_id"] for entry in filter_by_timeframe(entries, "1Min")] == ["alpha-run-b"]
    assert get_alpha_evaluation_run(entries, "alpha-run-a")["alpha_name"] == "demo_alpha"
    assert [entry["run_id"] for entry in filter_by_metric_threshold(entries, "mean_ic", min_value=0.2)] == [
        "alpha-run-a"
    ]
