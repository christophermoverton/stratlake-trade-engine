from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    REGIME_OUTPUT_COLUMNS,
    RegimeAlignmentError,
    align_regime_labels,
    align_regimes_to_alpha_windows,
    align_regimes_to_portfolio_windows,
    align_regimes_to_strategy_timeseries,
    attach_regime_artifacts_to_manifest,
    classify_market_regimes,
    load_regime_labels,
    load_regime_manifest,
    load_regime_summary,
    write_regime_artifacts,
)


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC"),
            "market_return": [0.00, 0.01, 0.01, -0.04, -0.03, 0.02, 0.03, 0.00],
        }
    )


def _regime_labels() -> pd.DataFrame:
    result = classify_market_regimes(
        _market_frame(),
        config={
            "return_column": "market_return",
            "volatility_window": 3,
            "trend_window": 3,
            "trend_return_threshold": 0.02,
            "drawdown_threshold": 0.05,
            "near_peak_drawdown_threshold": 0.0,
            "stress_window": 3,
        },
    )
    return result.labels


def test_write_regime_artifacts_persists_canonical_files(tmp_path: Path) -> None:
    labels = _regime_labels()
    output_dir = tmp_path / "regimes" / "regime_run"

    paths, manifest = write_regime_artifacts(
        output_dir,
        labels,
        metadata={"source": "unit_test", "taxonomy_version": "regime_taxonomy_v1"},
    )

    assert paths.labels_csv_path.exists()
    assert paths.summary_json_path.exists()
    assert paths.manifest_json_path.exists()

    loaded_labels = load_regime_labels(output_dir)
    loaded_summary = load_regime_summary(output_dir)
    loaded_manifest = load_regime_manifest(output_dir)

    pd.testing.assert_frame_equal(
        labels,
        loaded_labels,
        check_exact=False,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert loaded_summary["columns"] == list(REGIME_OUTPUT_COLUMNS)
    assert loaded_summary["taxonomy_version"] == "regime_taxonomy_v1"
    assert loaded_manifest["artifacts"]["regime_labels_csv"] == "regime_labels.csv"
    assert loaded_manifest["summary"]["row_count"] == len(labels)
    assert manifest == loaded_manifest


def test_regime_artifact_serialization_is_deterministic_for_same_output(tmp_path: Path) -> None:
    labels = _regime_labels()
    output_dir = tmp_path / "regimes" / "deterministic_run"

    first_paths, first_manifest = write_regime_artifacts(output_dir, labels, metadata={"source": "test"})
    first_csv = first_paths.labels_csv_path.read_text(encoding="utf-8")
    first_summary = first_paths.summary_json_path.read_text(encoding="utf-8")
    first_manifest_text = first_paths.manifest_json_path.read_text(encoding="utf-8")

    second_paths, second_manifest = write_regime_artifacts(output_dir, labels, metadata={"source": "test"})
    second_csv = second_paths.labels_csv_path.read_text(encoding="utf-8")
    second_summary = second_paths.summary_json_path.read_text(encoding="utf-8")
    second_manifest_text = second_paths.manifest_json_path.read_text(encoding="utf-8")

    assert first_csv == second_csv
    assert first_summary == second_summary
    assert first_manifest_text == second_manifest_text
    assert first_manifest == second_manifest


def test_attach_regime_artifacts_to_manifest_uses_relative_paths(tmp_path: Path) -> None:
    labels = _regime_labels()
    regime_dir = tmp_path / "run" / "regimes"
    _, _ = write_regime_artifacts(regime_dir, labels)

    parent_manifest_path = tmp_path / "run" / "manifest.json"
    parent_manifest_path.write_text('{"run_id":"demo"}', encoding="utf-8")

    updated = attach_regime_artifacts_to_manifest(
        parent_manifest_path,
        regime_manifest_path=regime_dir / "manifest.json",
    )

    section = updated["regime_artifacts"]
    assert section["manifest_path"] == "regimes/manifest.json"
    assert section["taxonomy_version"] == "regime_taxonomy_v1"


def test_align_regime_labels_exact_overlap_preserves_rows_and_order() -> None:
    labels = _regime_labels()
    target = pd.DataFrame(
        {
            "ts_utc": labels["ts_utc"].iloc[[2, 0, 3, 1]].reset_index(drop=True),
            "strategy_return": [0.03, 0.01, -0.02, 0.00],
        }
    )

    aligned = align_regime_labels(target, labels)

    assert aligned["strategy_return"].tolist() == [0.03, 0.01, -0.02, 0.00]
    assert aligned["regime_has_exact_timestamp_match"].all()
    assert aligned["regime_alignment_status"].isin(["matched_defined", "matched_undefined"]).all()


def test_align_regime_labels_partial_overlap_marks_unmatched_rows() -> None:
    labels = _regime_labels()
    target = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2024-12-30", periods=5, freq="D", tz="UTC"),
            "portfolio_return": [0.00, 0.01, 0.02, -0.01, 0.03],
        }
    )

    aligned = align_regime_labels(target, labels)

    assert aligned["regime_has_exact_timestamp_match"].tolist() == [False, False, True, True, True]
    assert aligned["regime_alignment_status"].tolist()[:2] == ["unmatched_timestamp", "unmatched_timestamp"]
    assert aligned.loc[0, "regime_label"].startswith("volatility=undefined")


def test_align_regime_labels_unavailable_policy_mark_unmatched() -> None:
    target = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
            "alpha_score": [0.4, 0.2, 0.7],
        }
    )

    aligned = align_regime_labels(
        target,
        None,
        config={"surface": "alpha", "unavailable_policy": "mark_unmatched"},
    )

    assert aligned["regime_alignment_status"].eq("regime_labels_unavailable").all()
    assert aligned["regime_has_exact_timestamp_match"].eq(False).all()
    assert aligned["regime_is_defined"].eq(False).all()


def test_align_regime_labels_unavailable_policy_raise() -> None:
    target = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC"),
            "alpha_score": [0.5, 0.1],
        }
    )

    with pytest.raises(RegimeAlignmentError, match="not provided"):
        align_regime_labels(target, None)


def test_align_regime_labels_warmup_rows_are_matched_undefined() -> None:
    labels = _regime_labels()
    target = pd.DataFrame(
        {
            "ts_utc": labels["ts_utc"].iloc[:3].reset_index(drop=True),
            "portfolio_equity": [1.0, 1.01, 1.02],
        }
    )

    aligned = align_regime_labels(target, labels)

    assert aligned["regime_has_exact_timestamp_match"].all()
    assert aligned.loc[0, "regime_alignment_status"] == "matched_undefined"


def test_align_regime_labels_sparse_target_with_metrics() -> None:
    labels = _regime_labels()
    target = pd.DataFrame(
        {
            "ts_utc": labels["ts_utc"].iloc[[0, 2, 5, 7]].reset_index(drop=True),
            "weight": [0.1, 0.2, 0.3, 0.4],
        }
    )

    aligned = align_regime_labels(
        target,
        labels,
        config={"include_metric_columns": True, "surface": "portfolio"},
    )

    assert len(aligned) == len(target)
    assert aligned["weight"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert "regime_volatility_metric" in aligned.columns
    assert "regime_stress_dispersion_metric" in aligned.columns


def test_surface_specific_alignment_helpers_set_surface() -> None:
    labels = _regime_labels()
    base = pd.DataFrame({"ts_utc": labels["ts_utc"].iloc[:2].reset_index(drop=True), "value": [1, 2]})

    strategy_aligned = align_regimes_to_strategy_timeseries(base, labels)
    alpha_aligned = align_regimes_to_alpha_windows(base, labels)
    portfolio_aligned = align_regimes_to_portfolio_windows(base, labels)

    assert strategy_aligned["regime_surface"].eq("strategy").all()
    assert alpha_aligned["regime_surface"].eq("alpha").all()
    assert portfolio_aligned["regime_surface"].eq("portfolio").all()
