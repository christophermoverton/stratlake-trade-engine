from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha import AlphaSignalMappingConfig, map_alpha_predictions_to_signals
from src.research.alpha_eval import evaluate_alpha_predictions, write_alpha_evaluation_artifacts


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


def test_write_alpha_evaluation_artifacts_persists_deterministic_outputs(tmp_path: Path) -> None:
    result = evaluate_alpha_predictions(_alpha_eval_frame())
    output_dir = tmp_path / "artifacts" / "alpha" / "run_123"

    manifest = write_alpha_evaluation_artifacts(
        output_dir,
        result,
        run_id="run_123",
        alpha_name="demo_alpha",
    )

    assert manifest["artifact_files"] == [
        "alpha_metrics.json",
        "coefficients.json",
        "cross_section_diagnostics.json",
        "ic_timeseries.csv",
        "manifest.json",
        "predictions.parquet",
        "training_summary.json",
    ]
    assert manifest["artifact_groups"]["alpha_evaluation"] == [
        "alpha_metrics.json",
        "coefficients.json",
        "cross_section_diagnostics.json",
        "ic_timeseries.csv",
        "manifest.json",
        "predictions.parquet",
        "training_summary.json",
    ]
    assert manifest["run_id"] == "run_123"
    assert manifest["alpha_name"] == "demo_alpha"
    assert manifest["artifact_paths"]["predictions"] == "predictions.parquet"
    assert manifest["artifact_paths"]["training_summary"] == "training_summary.json"
    assert manifest["timeseries_columns"] == ["ts_utc", "ic", "rank_ic", "n_obs", "sample_size"]

    timeseries = pd.read_csv(output_dir / "ic_timeseries.csv")
    assert list(timeseries.columns) == ["ts_utc", "ic", "rank_ic", "n_obs", "sample_size"]
    assert timeseries["ts_utc"].tolist() == ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"]
    assert timeseries["n_obs"].tolist() == [3, 3]
    assert timeseries["sample_size"].tolist() == [3, 3]
    assert timeseries["ic"].tolist() == pytest.approx(result.ic_timeseries["ic"].tolist())
    assert timeseries["rank_ic"].tolist() == pytest.approx(result.ic_timeseries["rank_ic"].tolist())

    metrics_payload = json.loads((output_dir / "alpha_metrics.json").read_text(encoding="utf-8"))
    assert list(metrics_payload) == sorted(metrics_payload)
    assert metrics_payload["mean_ic"] == result.summary["mean_ic"]
    assert metrics_payload["std_ic"] == result.summary["std_ic"]
    assert metrics_payload["ic_ir"] == result.summary["ic_ir"]
    assert metrics_payload["mean_rank_ic"] == result.summary["mean_rank_ic"]
    assert metrics_payload["std_rank_ic"] == result.summary["std_rank_ic"]
    assert metrics_payload["rank_ic_ir"] == result.summary["rank_ic_ir"]
    assert metrics_payload["n_periods"] == result.summary["n_periods"]
    assert metrics_payload["metadata"]["run_id"] == "run_123"
    assert metrics_payload["metadata"]["alpha_name"] == "demo_alpha"
    assert metrics_payload["metadata"]["timeframe"] == "1d"

    training_summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert training_summary["alpha_name"] == "demo_alpha"
    assert training_summary["run_id"] == "run_123"
    assert training_summary["model_name"] is None
    assert training_summary["training"]["train_row_count"] is None

    coefficients_payload = json.loads((output_dir / "coefficients.json").read_text(encoding="utf-8"))
    assert coefficients_payload["representation"] == "unavailable"
    assert coefficients_payload["values"] == {}

    diagnostics_payload = json.loads((output_dir / "cross_section_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics_payload["row_count"] == result.row_count
    assert diagnostics_payload["timestamp_count"] == result.timestamp_count
    assert diagnostics_payload["valid_periods"]["ic"] == result.summary["n_periods"]

    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    assert list(predictions.columns) == ["symbol", "ts_utc", "timeframe", "prediction_score"]
    assert predictions.empty

    manifest_payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload == manifest


def test_write_alpha_evaluation_artifacts_persists_explicit_signal_mapping_outputs(tmp_path: Path) -> None:
    frame = _alpha_eval_frame()
    result = evaluate_alpha_predictions(frame)
    signal_mapping_result = map_alpha_predictions_to_signals(
        frame.loc[:, ["symbol", "ts_utc", "timeframe", "prediction_score"]],
        AlphaSignalMappingConfig(policy="top_bottom_quantile", quantile=0.34),
    )
    output_dir = tmp_path / "artifacts" / "alpha" / "run_456"

    manifest = write_alpha_evaluation_artifacts(
        output_dir,
        result,
        signal_mapping_result=signal_mapping_result,
        run_id="run_456",
        alpha_name="demo_alpha",
    )

    assert "signals.parquet" in manifest["artifact_files"]
    assert "signal_mapping.json" in manifest["artifact_files"]
    assert manifest["artifact_paths"]["signals"] == "signals.parquet"
    assert manifest["artifact_paths"]["signal_mapping"] == "signal_mapping.json"
    assert manifest["signals_path"] == "signals.parquet"
    assert manifest["signal_mapping_path"] == "signal_mapping.json"

    signal_mapping_payload = json.loads((output_dir / "signal_mapping.json").read_text(encoding="utf-8"))
    assert signal_mapping_payload["config"]["policy"] == "top_bottom_quantile"
    assert signal_mapping_payload["config"]["quantile"] == pytest.approx(0.34)

    signals = pd.read_parquet(output_dir / "signals.parquet")
    assert list(signals.columns) == ["symbol", "ts_utc", "timeframe", "prediction_score", "signal"]
    assert signals["signal"].tolist() == [-1.0, 1.0, 0.0, 0.0, 1.0, -1.0]

    training_summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert training_summary["signal_mapping"]["policy"] == "top_bottom_quantile"
    assert training_summary["signal_mapping"]["signal_column"] == "signal"


def test_write_alpha_evaluation_artifacts_updates_parent_manifest_idempotently(tmp_path: Path) -> None:
    parent_dir = tmp_path / "strategy_run"
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_files": ["config.json"],
                "artifact_groups": {"core": ["config.json"]},
                "run_id": "run-1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = evaluate_alpha_predictions(_alpha_eval_frame())
    alpha_dir = parent_dir / "alpha_eval"

    first_manifest = write_alpha_evaluation_artifacts(
        alpha_dir,
        result,
        parent_manifest_dir=parent_dir,
        run_id="run-1",
        alpha_name="demo_alpha",
    )
    first_snapshot = {
        path.relative_to(parent_dir).as_posix(): path.read_bytes()
        for path in sorted(parent_dir.rglob("*"))
        if path.is_file()
    }

    second_manifest = write_alpha_evaluation_artifacts(
        alpha_dir,
        result,
        parent_manifest_dir=parent_dir,
        run_id="run-1",
        alpha_name="demo_alpha",
    )
    second_snapshot = {
        path.relative_to(parent_dir).as_posix(): path.read_bytes()
        for path in sorted(parent_dir.rglob("*"))
        if path.is_file()
    }

    assert first_manifest == second_manifest
    assert first_snapshot == second_snapshot

    parent_manifest = json.loads((parent_dir / "manifest.json").read_text(encoding="utf-8"))
    assert parent_manifest["alpha_evaluation"]["enabled"] is True
    assert parent_manifest["alpha_evaluation"]["artifact_path"] == "alpha_eval"
    assert parent_manifest["alpha_evaluation"]["metrics_path"] == "alpha_eval/alpha_metrics.json"
    assert parent_manifest["alpha_evaluation"]["predictions_path"] == "alpha_eval/predictions.parquet"
    assert parent_manifest["alpha_evaluation"]["training_summary_path"] == "alpha_eval/training_summary.json"
    assert parent_manifest["alpha_evaluation"]["timeseries_path"] == "alpha_eval/ic_timeseries.csv"
    assert "alpha_eval/alpha_metrics.json" in parent_manifest["artifact_files"]
    assert parent_manifest["artifact_groups"]["alpha_evaluation"] == [
        "alpha_eval/alpha_metrics.json",
        "alpha_eval/coefficients.json",
        "alpha_eval/cross_section_diagnostics.json",
        "alpha_eval/ic_timeseries.csv",
        "alpha_eval/manifest.json",
        "alpha_eval/predictions.parquet",
        "alpha_eval/training_summary.json",
    ]


def test_write_alpha_evaluation_artifacts_does_not_mutate_result_frame(tmp_path: Path) -> None:
    result = evaluate_alpha_predictions(_alpha_eval_frame())
    baseline = result.ic_timeseries.copy(deep=True)
    baseline.attrs = dict(result.ic_timeseries.attrs)

    write_alpha_evaluation_artifacts(tmp_path / "alpha" / "run_123", result)

    pdt.assert_frame_equal(result.ic_timeseries, baseline, check_dtype=True, check_exact=True)
    assert result.ic_timeseries.attrs == baseline.attrs
