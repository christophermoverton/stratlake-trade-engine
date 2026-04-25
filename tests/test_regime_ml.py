from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regime_ml import (
    REGIME_CLUSTER_DIAGNOSTICS_FILENAME,
    REGIME_CLUSTER_MAP_FILENAME,
    REGIME_CONFIDENCE_FILENAME,
    REGIME_LABEL_MAPPING_FILENAME,
    REGIME_ML_DIAGNOSTICS_FILENAME,
    REGIME_MODEL_MANIFEST_FILENAME,
    RegimeMLError,
    run_regime_ml_pipeline,
)

_BULL = "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress"
_BEAR = "volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress"
_RECOVERY = "volatility=normal_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress"


def _regime_ml_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=18, freq="D", tz="UTC")
    patterns = [
        (_BULL, (3.0, 0.2, 0.1)),
        (_BEAR, (-3.0, 3.2, 2.9)),
        (_RECOVERY, (0.2, -2.8, 2.5)),
    ]
    rows: list[dict[str, object]] = []
    for ts_index, ts_utc in enumerate(timestamps):
        regime_label, centers = patterns[ts_index % len(patterns)]
        for symbol_index, symbol in enumerate(("AAA", "BBB"), start=1):
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "regime_label": regime_label,
                    "feature_a": centers[0] + (0.03 * symbol_index),
                    "feature_b": centers[1] + (0.02 * symbol_index),
                    "feature_c": centers[2] + (0.01 * symbol_index),
                }
            )
    return pd.DataFrame(rows)


def test_regime_ml_pipeline_is_deterministic_and_taxonomy_compatible(tmp_path: Path) -> None:
    frame = _regime_ml_frame()
    output_dir = tmp_path / "regime_ml" / "baseline"

    first = run_regime_ml_pipeline(
        frame,
        feature_columns=["feature_a", "feature_b", "feature_c"],
        output_dir=output_dir,
    )
    second = run_regime_ml_pipeline(
        frame,
        feature_columns=["feature_a", "feature_b", "feature_c"],
        output_dir=output_dir,
    )

    pd.testing.assert_frame_equal(first.regime_confidence, second.regime_confidence, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(first.class_probabilities, second.class_probabilities, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(first.cluster_map, second.cluster_map, check_exact=False, rtol=1e-12, atol=1e-12)
    assert first.diagnostics == second.diagnostics
    assert first.manifest == second.manifest
    assert set(first.regime_confidence["predicted_label"]) <= {_BULL, _BEAR, _RECOVERY}
    assert set(first.regime_confidence["regime_label"]) <= {_BULL, _BEAR, _RECOVERY}
    assert (output_dir / REGIME_CONFIDENCE_FILENAME).exists()
    assert (output_dir / REGIME_ML_DIAGNOSTICS_FILENAME).exists()
    assert (output_dir / REGIME_MODEL_MANIFEST_FILENAME).exists()
    assert (output_dir / REGIME_CLUSTER_MAP_FILENAME).exists()
    assert (output_dir / REGIME_CLUSTER_DIAGNOSTICS_FILENAME).exists()
    assert (output_dir / REGIME_LABEL_MAPPING_FILENAME).exists()


def test_regime_ml_normalizes_feature_order_and_persists_actual_schema() -> None:
    result = run_regime_ml_pipeline(
        _regime_ml_frame(),
        feature_columns=["feature_c", "feature_a", "feature_b"],
    )

    assert result.manifest["feature_columns"] == ["feature_a", "feature_b", "feature_c"]
    assert result.diagnostics["feature_columns"] == ["feature_a", "feature_b", "feature_c"]


def test_regime_ml_probabilities_sum_to_one_and_thresholds_are_applied() -> None:
    result = run_regime_ml_pipeline(
        _regime_ml_frame(),
        feature_columns=["feature_a", "feature_b", "feature_c"],
    )

    probability_sums = result.class_probabilities.sum(axis=1)
    assert probability_sums.between(0.999999999, 1.000000001).all()

    high_mask = result.regime_confidence["confidence_score"] >= 0.70
    medium_mask = result.regime_confidence["confidence_score"].between(0.40, 0.70, inclusive="left")
    low_mask = result.regime_confidence["confidence_score"] < 0.40

    assert result.regime_confidence.loc[high_mask, "confidence_bucket"].eq("high").all()
    assert result.regime_confidence.loc[medium_mask, "confidence_bucket"].eq("medium").all()
    assert result.regime_confidence.loc[low_mask, "confidence_bucket"].eq("low").all()


def test_regime_ml_logs_calibration_metrics_and_cluster_alignment() -> None:
    result = run_regime_ml_pipeline(
        _regime_ml_frame(),
        feature_columns=["feature_a", "feature_b", "feature_c"],
    )

    calibration = result.diagnostics["calibration"]
    assert calibration["calibration_method"] == "platt"
    assert "pre_calibration_brier" in calibration
    assert "post_calibration_brier" in calibration
    assert "brier_improved" in calibration

    cluster_diagnostics = result.cluster_diagnostics
    assert cluster_diagnostics["n_clusters"] == 3
    assert 0.0 <= cluster_diagnostics["purity_mean"] <= 1.0
    assert 0.0 <= cluster_diagnostics["nmi"] <= 1.0
    assert "ari" in cluster_diagnostics
    assert not result.cluster_map.empty


def test_regime_ml_flags_rows_with_missing_features_as_unsupported() -> None:
    frame = _regime_ml_frame()
    frame.loc[5, "feature_b"] = None

    result = run_regime_ml_pipeline(
        frame,
        feature_columns=["feature_a", "feature_b", "feature_c"],
    )

    flagged = result.regime_confidence.loc[result.regime_confidence["fallback_reason"] == "unsupported"]
    assert not flagged.empty
    assert result.diagnostics["missing_features"]["row_count"] == 1
    imputation = result.diagnostics["feature_imputation"]
    assert imputation["method"] == "median"
    assert imputation["missing_count_by_feature"]["feature_b"] == 1
    assert imputation["rows_with_any_missing_feature"] == 1
    assert sorted(imputation["values"]) == ["feature_a", "feature_b", "feature_c"]
    assert result.manifest["feature_imputation"] == imputation


def test_regime_ml_rejects_insufficient_class_coverage() -> None:
    frame = _regime_ml_frame()
    frame["regime_label"] = _BULL

    with pytest.raises(RegimeMLError, match="at least two distinct labels"):
        run_regime_ml_pipeline(frame, feature_columns=["feature_a", "feature_b", "feature_c"])


def test_regime_ml_marks_unseen_evaluation_labels_as_unsupported() -> None:
    frame = _regime_ml_frame()
    extra_label = "volatility=undefined|trend=undefined|drawdown_recovery=undefined|stress=undefined"
    frame.loc[frame["ts_utc"] >= pd.Timestamp("2026-01-16", tz="UTC"), "regime_label"] = extra_label

    result = run_regime_ml_pipeline(
        frame,
        feature_columns=["feature_a", "feature_b", "feature_c"],
    )

    assert extra_label in result.diagnostics["unseen_labels"]
    unsupported = result.regime_confidence.loc[result.regime_confidence["regime_label"] == extra_label]
    assert unsupported["fallback_reason"].eq("unsupported").all()


@pytest.mark.parametrize(
    ("feature_columns", "message"),
    [
        ([], "at least one feature"),
        (["feature_a", "feature_a", "feature_b"], "must not contain duplicates"),
    ],
)
def test_regime_ml_rejects_invalid_feature_column_inputs(
    feature_columns: list[str],
    message: str,
) -> None:
    with pytest.raises(RegimeMLError, match=message):
        run_regime_ml_pipeline(_regime_ml_frame(), feature_columns=feature_columns)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"validation_fraction": 0.0}, "validation_fraction"),
        ({"test_fraction": 0.0}, "test_fraction"),
        ({"validation_fraction": 0.60, "test_fraction": 0.40}, "validation_fraction \\+ test_fraction < 1"),
        ({"confidence_thresholds": {"high": 0.7}}, "missing required keys"),
        ({"confidence_thresholds": {"medium": 0.8, "high": 0.7}}, "medium <= high"),
        ({"confidence_thresholds": {"medium": -0.1, "high": 0.7}}, "between 0 and 1"),
        ({"n_clusters": 0}, "positive integer"),
        ({"model_version": ""}, "model_version"),
    ],
)
def test_regime_ml_rejects_invalid_config_values(config: dict[str, object], message: str) -> None:
    with pytest.raises(RegimeMLError, match=message):
        run_regime_ml_pipeline(
            _regime_ml_frame(),
            feature_columns=["feature_a", "feature_b", "feature_c"],
            config=config,
        )


def test_regime_ml_rejects_unseen_validation_labels_before_calibration() -> None:
    frame = _regime_ml_frame()
    extra_label = "volatility=undefined|trend=undefined|drawdown_recovery=undefined|stress=undefined"
    frame.loc[frame["ts_utc"] == pd.Timestamp("2026-01-13", tz="UTC"), "regime_label"] = extra_label
    frame.loc[frame["ts_utc"] == pd.Timestamp("2026-01-14", tz="UTC"), "regime_label"] = extra_label

    with pytest.raises(
        RegimeMLError,
        match="Validation labels contain classes not seen during training",
    ):
        run_regime_ml_pipeline(
            frame,
            feature_columns=["feature_a", "feature_b", "feature_c"],
        )
