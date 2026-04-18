from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from src.research.alpha_eval.sleeves import generate_alpha_sleeve, write_alpha_sleeve_artifacts


def _signals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"], dtype="string"),
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "timeframe": pd.Series(["1D", "1D", "1D", "1D", "1D", "1D"], dtype="string"),
            "prediction_score": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "signal": [1.0, 0.0, -1.0, -1.0, 0.0, 1.0],
        }
    )


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"], dtype="string"),
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "timeframe": pd.Series(["1D", "1D", "1D", "1D", "1D", "1D"], dtype="string"),
            "feature_ret_1d": [0.02, 0.01, -0.01, -0.03, 0.01, 0.04],
        }
    )


def test_generate_alpha_sleeve_builds_expected_return_stream() -> None:
    result = generate_alpha_sleeve(
        signals=_signals_frame(),
        dataset=_dataset_frame(),
        realized_return_column="feature_ret_1d",
        alpha_name="demo_alpha",
        run_id="demo_run",
    )

    assert result.sleeve_returns["ts_utc"].tolist() == ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"]
    assert result.sleeve_returns["sleeve_return"].tolist() == pytest.approx([0.0, -0.07])
    assert result.sleeve_equity_curve["sleeve_equity_curve"].tolist() == pytest.approx([1.0, 0.93])
    assert result.metrics["alpha_name"] == "demo_alpha"
    assert result.metrics["run_id"] == "demo_run"
    assert result.metrics["return_source"] == "realized_return_column:feature_ret_1d"
    assert result.metrics["constructor_id"] == "backtest_numeric_exposure"


def test_write_alpha_sleeve_artifacts_is_deterministic_and_updates_manifest(tmp_path: Path) -> None:
    result = generate_alpha_sleeve(
        signals=_signals_frame(),
        dataset=_dataset_frame(),
        realized_return_column="feature_ret_1d",
        alpha_name="demo_alpha",
        run_id="demo_run",
    )
    output_dir = tmp_path / "artifacts" / "alpha" / "demo_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_files": ["alpha_metrics.json", "manifest.json", "signals.parquet"],
                "artifact_groups": {"alpha_evaluation": ["alpha_metrics.json", "manifest.json", "signals.parquet"]},
                "artifact_paths": {"metrics": "alpha_metrics.json", "signals": "signals.parquet"},
                "files_written": 3,
                "run_id": "demo_run",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    first_manifest = write_alpha_sleeve_artifacts(output_dir, result, update_manifest=True)
    first_snapshot = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    second_manifest = write_alpha_sleeve_artifacts(output_dir, result, update_manifest=True)
    second_snapshot = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }

    assert first_manifest == second_manifest
    assert first_snapshot == second_snapshot

    returns_frame = pd.read_csv(output_dir / "sleeve_returns.csv")
    equity_frame = pd.read_csv(output_dir / "sleeve_equity_curve.csv")
    metrics_payload = json.loads((output_dir / "sleeve_metrics.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    pdt.assert_frame_equal(returns_frame, result.sleeve_returns, check_dtype=False)
    pdt.assert_frame_equal(equity_frame, result.sleeve_equity_curve, check_dtype=False)
    assert metrics_payload == result.metrics
    assert manifest_payload["constructor_id"] == "backtest_numeric_exposure"
    assert manifest_payload["artifact_paths"]["sleeve_returns"] == "sleeve_returns.csv"
    assert manifest_payload["artifact_paths"]["sleeve_equity_curve"] == "sleeve_equity_curve.csv"
    assert manifest_payload["artifact_paths"]["sleeve_metrics"] == "sleeve_metrics.json"
    assert manifest_payload["sleeve"]["metric_summary"] == result.metrics
