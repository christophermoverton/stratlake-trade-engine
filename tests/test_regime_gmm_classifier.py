from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    REGIME_AUDIT_COLUMNS,
    REGIME_GMM_LABELS_FILENAME,
    REGIME_GMM_MANIFEST_FILENAME,
    REGIME_GMM_POSTERIOR_FILENAME,
    REGIME_GMM_SHIFT_EVENTS_FILENAME,
    REGIME_GMM_SUMMARY_FILENAME,
    RegimeGMMClassifierError,
    apply_regime_calibration,
    apply_regime_policy,
    classify_regime_shifts_with_gmm,
    load_regime_gmm_labels,
    load_regime_gmm_manifest,
    load_regime_gmm_posteriors,
    load_regime_gmm_shift_events,
    load_regime_gmm_summary,
    write_regime_gmm_artifacts,
)


_BULL = "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress"
_BEAR = "volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress"
_RECOVERY = "volatility=normal_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress"


def _regime_labels_for_integration() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=18, freq="D", tz="UTC")
    pattern = [_BULL, _BULL, _RECOVERY, _BEAR, _BEAR, _RECOVERY]
    rows: list[dict[str, object]] = []
    for index, ts_utc in enumerate(timestamps):
        label = pattern[index % len(pattern)]
        mapping = dict(part.split("=", 1) for part in label.split("|"))
        rows.append(
            {
                "ts_utc": ts_utc,
                "volatility_state": mapping["volatility"],
                "trend_state": mapping["trend"],
                "drawdown_recovery_state": mapping["drawdown_recovery"],
                "stress_state": mapping["stress"],
                "regime_label": label,
                "is_defined": True,
                "volatility_metric": 0.10 + (0.03 * (index % 3)) + (0.18 if mapping["volatility"] == "high_volatility" else 0.0),
                "trend_metric": 0.03 if mapping["trend"] == "uptrend" else (-0.04 if mapping["trend"] == "downtrend" else 0.00),
                "drawdown_metric": 0.01 if mapping["drawdown_recovery"] == "near_peak" else (0.15 if mapping["drawdown_recovery"] == "drawdown" else 0.07),
                "stress_correlation_metric": 0.25 if mapping["stress"] == "normal_stress" else 0.82,
                "stress_dispersion_metric": 0.22 if mapping["stress"] == "normal_stress" else 0.48,
            }
        )
    return pd.DataFrame(rows)


def _gmm_input() -> pd.DataFrame:
    return _regime_labels_for_integration().loc[:, ["ts_utc", *list(REGIME_AUDIT_COLUMNS)]].copy(deep=True)


def _synthetic_shift_features() -> pd.DataFrame:
    ts_utc = pd.date_range("2026-03-01", periods=14, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    for index, ts in enumerate(ts_utc):
        if index < 7:
            center = (0.10, 0.02, 0.01, 0.15, 0.20)
        else:
            center = (0.45, -0.06, 0.20, 0.75, 0.55)
        rows.append(
            {
                "ts_utc": ts,
                "volatility_metric": center[0] + (0.002 * (index % 3)),
                "trend_metric": center[1] - (0.001 * (index % 2)),
                "drawdown_metric": center[2] + (0.003 * (index % 2)),
                "stress_correlation_metric": center[3] + (0.002 * (index % 2)),
                "stress_dispersion_metric": center[4] + (0.004 * (index % 2)),
            }
        )
    return pd.DataFrame(rows)


def test_gmm_classifier_is_deterministic_and_preserves_input() -> None:
    features = _gmm_input()
    original = features.copy(deep=True)

    shuffled = features.sample(frac=1.0, random_state=7).reset_index(drop=True)
    first = classify_regime_shifts_with_gmm(
        shuffled,
        config={"n_components": 2, "random_state": 31, "feature_columns": REGIME_AUDIT_COLUMNS},
    )
    second = classify_regime_shifts_with_gmm(
        shuffled,
        config={"n_components": 2, "random_state": 31, "feature_columns": REGIME_AUDIT_COLUMNS},
    )

    pd.testing.assert_frame_equal(features, original)
    pd.testing.assert_frame_equal(first.labels, second.labels, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(
        first.posterior_probabilities,
        second.posterior_probabilities,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(first.shift_events, second.shift_events)
    assert first.summary == second.summary
    assert first.labels["ts_utc"].is_monotonic_increasing


def test_gmm_posterior_probabilities_confidence_and_entropy_contract() -> None:
    result = classify_regime_shifts_with_gmm(
        _gmm_input(),
        config={"n_components": 3, "random_state": 19, "feature_columns": REGIME_AUDIT_COLUMNS},
    )

    posterior_columns = [column for column in result.posterior_probabilities.columns if column.startswith("posterior_cluster_")]
    assert len(posterior_columns) == 3
    assert len(result.posterior_probabilities) == len(result.labels)

    row_sums = result.posterior_probabilities.loc[:, posterior_columns].sum(axis=1)
    assert row_sums.between(0.999999999, 1.000000001).all()

    top_probability = result.posterior_probabilities.loc[:, posterior_columns].max(axis=1)
    assert result.labels["gmm_posterior_probability"].equals(top_probability)
    assert result.labels["gmm_confidence_score"].equals(top_probability)

    entropy = result.labels["gmm_posterior_entropy"]
    assert entropy.ge(0.0).all()
    assert entropy.le(math.log(3.0) + 1.0e-9).all()
    assert result.labels["gmm_entropy_confidence"].between(0.0, 1.0).all()


def test_gmm_detects_cluster_shift_events() -> None:
    result = classify_regime_shifts_with_gmm(
        _synthetic_shift_features(),
        config={
            "n_components": 2,
            "random_state": 11,
            "min_shift_probability": 0.5,
            "shift_probability_delta_threshold": 0.05,
            "feature_columns": REGIME_AUDIT_COLUMNS,
        },
    )

    assert result.labels["gmm_shift_flag"].any()
    assert not result.shift_events.empty
    assert result.shift_events["previous_cluster_label"].notna().all()
    assert result.shift_events["event_order"].tolist() == list(range(1, len(result.shift_events) + 1))


def test_gmm_artifacts_manifest_and_inventory_are_deterministic(tmp_path: Path) -> None:
    result = classify_regime_shifts_with_gmm(
        _gmm_input(),
        config={"n_components": 2, "random_state": 13, "feature_columns": REGIME_AUDIT_COLUMNS},
    )
    output_dir = tmp_path / "regimes" / "gmm_case"

    first_manifest = write_regime_gmm_artifacts(
        output_dir,
        result,
        run_id="gmm_case",
        source_regime_artifact_references={"manifest_path": "../manifest.json"},
    )
    second_manifest = write_regime_gmm_artifacts(
        output_dir,
        result,
        run_id="gmm_case",
        source_regime_artifact_references={"manifest_path": "../manifest.json"},
    )

    assert (output_dir / REGIME_GMM_LABELS_FILENAME).exists()
    assert (output_dir / REGIME_GMM_POSTERIOR_FILENAME).exists()
    assert (output_dir / REGIME_GMM_SHIFT_EVENTS_FILENAME).exists()
    assert (output_dir / REGIME_GMM_SUMMARY_FILENAME).exists()
    assert (output_dir / REGIME_GMM_MANIFEST_FILENAME).exists()

    loaded_labels = load_regime_gmm_labels(output_dir)
    loaded_posterior = load_regime_gmm_posteriors(output_dir)
    loaded_events = load_regime_gmm_shift_events(output_dir)
    loaded_summary = load_regime_gmm_summary(output_dir)
    loaded_manifest = load_regime_gmm_manifest(output_dir)

    assert len(loaded_labels) == len(result.labels)
    assert len(loaded_posterior) == len(result.posterior_probabilities)
    assert len(loaded_events) == len(result.shift_events)
    assert loaded_summary["cluster_count"] == result.summary["cluster_count"]
    assert loaded_manifest["file_inventory"][REGIME_GMM_MANIFEST_FILENAME]["path"] == REGIME_GMM_MANIFEST_FILENAME
    assert "sha256" not in loaded_manifest["file_inventory"][REGIME_GMM_MANIFEST_FILENAME]
    assert first_manifest == second_manifest == loaded_manifest


def test_gmm_confidence_integrates_with_calibration_and_policy() -> None:
    labels = _regime_labels_for_integration()
    features = labels.loc[:, ["ts_utc", *list(REGIME_AUDIT_COLUMNS)]].copy(deep=True)
    gmm_result = classify_regime_shifts_with_gmm(
        features,
        config={"n_components": 2, "random_state": 23, "feature_columns": REGIME_AUDIT_COLUMNS},
    )

    labels_with_confidence = labels.merge(
        gmm_result.labels.loc[:, ["ts_utc", "gmm_confidence_score"]],
        on="ts_utc",
        how="left",
        sort=False,
    )
    calibration = apply_regime_calibration(
        labels_with_confidence,
        profile="baseline",
        confidence_column="gmm_confidence_score",
        low_confidence_threshold=0.75,
    )

    assert "is_low_confidence" in calibration.audit.columns
    assert str(calibration.audit["is_low_confidence"].dtype) == "bool"

    confidence_frame = gmm_result.labels.loc[
        :,
        ["ts_utc", "confidence_score", "confidence_bucket", "fallback_flag", "fallback_reason"],
    ]
    policy_result = apply_regime_policy(
        labels,
        confidence_frame=confidence_frame,
    )

    assert len(policy_result.decisions) == len(labels)
    assert set(policy_result.decisions["confidence_bucket"].dropna().unique()).issubset({"high", "medium", "low"})


@pytest.mark.parametrize(
    "config, message",
    [
        ({"n_components": 1}, "n_components"),
        ({"low_confidence_threshold": 1.1}, "between 0 and 1"),
        ({"confidence_thresholds": {"high": 0.8}}, "missing required keys"),
        ({"confidence_thresholds": {"medium": 0.9, "high": 0.8}}, "medium <= high"),
    ],
)
def test_gmm_rejects_invalid_config(config: dict[str, object], message: str) -> None:
    with pytest.raises(RegimeGMMClassifierError, match=message):
        classify_regime_shifts_with_gmm(_synthetic_shift_features(), config=config)
