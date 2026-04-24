from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    REGIME_CALIBRATION_FILENAME,
    REGIME_CALIBRATION_SUMMARY_FILENAME,
    REGIME_STABILITY_METRICS_FILENAME,
    RegimeCalibrationError,
    apply_regime_calibration,
    builtin_regime_calibration_profiles,
    load_regime_calibration,
    load_regime_calibration_summary,
    load_regime_stability_metrics,
    resolve_regime_calibration_profile,
    write_regime_calibration_artifacts,
)
from src.research.regimes.taxonomy import REGIME_DIMENSIONS, REGIME_STATE_COLUMNS, REGIME_TAXONOMY

_BULL = "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress"
_BEAR = "volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress"
_RECOVERY = "volatility=normal_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress"
_UNDEFINED = "volatility=undefined|trend=undefined|drawdown_recovery=undefined|stress=undefined"


def _labels(sequence: list[str]) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=len(sequence), freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    for index, label in enumerate(sequence):
        mapping = dict(part.split("=", 1) for part in label.split("|"))
        rows.append(
            {
                "ts_utc": ts[index],
                "volatility_state": mapping["volatility"],
                "trend_state": mapping["trend"],
                "drawdown_recovery_state": mapping["drawdown_recovery"],
                "stress_state": mapping["stress"],
                "regime_label": label,
                "is_defined": all(mapping[dimension] != "undefined" for dimension in REGIME_DIMENSIONS),
                "volatility_metric": 0.10 + (0.01 * index),
                "trend_metric": 0.02 - (0.01 * (index % 3)),
                "drawdown_metric": 0.01 * (index % 5),
                "stress_correlation_metric": 0.20 + (0.02 * (index % 4)),
                "stress_dispersion_metric": 0.25 + (0.01 * (index % 4)),
            }
        )
    return pd.DataFrame(rows)


def test_builtin_regime_calibration_profiles_are_available() -> None:
    profiles = builtin_regime_calibration_profiles()

    assert set(profiles) == {"baseline", "conservative", "reactive", "crisis_sensitive"}
    assert resolve_regime_calibration_profile(None).name == "baseline"
    assert resolve_regime_calibration_profile("conservative").transition_smoothing_window == 3


def test_invalid_regime_calibration_profiles_fail_fast() -> None:
    with pytest.raises(RegimeCalibrationError, match="Unknown regime calibration profile"):
        resolve_regime_calibration_profile("not_a_profile")

    with pytest.raises(RegimeCalibrationError, match="missing required fields"):
        resolve_regime_calibration_profile({"name": "custom_only"})

    with pytest.raises(RegimeCalibrationError, match="unsupported taxonomy label"):
        resolve_regime_calibration_profile(
            {
                "name": "bad_fallback",
                "min_regime_duration_days": 2,
                "transition_smoothing_window": 1,
                "allow_single_day_flips": False,
                "max_flip_rate": 0.2,
                "max_single_day_flip_share": 0.1,
                "min_observations_per_regime": 2,
                "min_observations_for_attribution": 3,
                "low_confidence_share_threshold": 0.3,
                "unstable_regime_fallback": "volatility=bad|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress",
                "unknown_regime_fallback": _UNDEFINED,
                "require_stability_for_attribution": True,
            }
        )


def test_apply_regime_calibration_is_deterministic_and_does_not_mutate_inputs() -> None:
    labels = _labels([_BULL, _BULL, _BEAR, _BULL, _BULL, _RECOVERY])
    original = labels.copy(deep=True)
    profile = {
        "name": "custom_baseline",
        "min_regime_duration_days": 2,
        "transition_smoothing_window": 1,
        "allow_single_day_flips": False,
        "max_flip_rate": 0.5,
        "max_single_day_flip_share": 0.5,
        "min_observations_per_regime": 2,
        "min_observations_for_attribution": 2,
        "low_confidence_share_threshold": 0.6,
        "unstable_regime_fallback": None,
        "unknown_regime_fallback": _UNDEFINED,
        "require_stability_for_attribution": True,
    }

    first = apply_regime_calibration(labels, profile=profile)
    second = apply_regime_calibration(labels, profile=profile)

    pd.testing.assert_frame_equal(labels, original)
    pd.testing.assert_frame_equal(first.labels, second.labels)
    pd.testing.assert_frame_equal(first.audit, second.audit)
    pd.testing.assert_frame_equal(first.attribution_summary, second.attribution_summary)
    assert first.stability_metrics == second.stability_metrics
    assert first.profile_flags == second.profile_flags
    assert first.warnings == second.warnings


def test_apply_regime_calibration_suppresses_short_lived_flips() -> None:
    labels = _labels([_BULL, _BULL, _BEAR, _BULL, _BULL])

    result = apply_regime_calibration(
        labels,
        profile={
            "name": "min_duration_gate",
            "min_regime_duration_days": 2,
            "transition_smoothing_window": 1,
            "allow_single_day_flips": False,
            "max_flip_rate": 1.0,
            "max_single_day_flip_share": 1.0,
            "min_observations_per_regime": 2,
            "min_observations_for_attribution": 2,
            "low_confidence_share_threshold": 1.0,
            "unstable_regime_fallback": None,
            "unknown_regime_fallback": _UNDEFINED,
            "require_stability_for_attribution": True,
        },
    )

    assert result.labels["regime_label"].tolist() == [_BULL, _BULL, _BULL, _BULL, _BULL]
    assert result.stability_metrics["transition_count"] == 0
    assert result.stability_metrics["minimum_regime_duration"] == 5


def test_apply_regime_calibration_detects_unstable_profiles_and_applies_fallback() -> None:
    labels = _labels([_BULL, _BEAR, _BULL, _BEAR, _BULL])

    result = apply_regime_calibration(
        labels,
        profile={
            "name": "unstable_gate",
            "min_regime_duration_days": 1,
            "transition_smoothing_window": 1,
            "allow_single_day_flips": True,
            "max_flip_rate": 0.2,
            "max_single_day_flip_share": 0.2,
            "min_observations_per_regime": 1,
            "min_observations_for_attribution": 1,
            "low_confidence_share_threshold": 1.0,
            "unstable_regime_fallback": _UNDEFINED,
            "unknown_regime_fallback": _UNDEFINED,
            "require_stability_for_attribution": True,
        },
    )

    assert result.profile_flags["is_unstable_profile"] is True
    assert result.audit["used_unstable_profile_fallback"].all()
    assert result.labels["regime_label"].eq(_UNDEFINED).all()


def test_apply_regime_calibration_handles_unknown_fallback_without_new_labels() -> None:
    labels = _labels([_UNDEFINED, _BULL, _BULL])

    result = apply_regime_calibration(
        labels,
        profile={
            "name": "unknown_fill",
            "min_regime_duration_days": 1,
            "transition_smoothing_window": 1,
            "allow_single_day_flips": True,
            "max_flip_rate": 1.0,
            "max_single_day_flip_share": 1.0,
            "min_observations_per_regime": 1,
            "min_observations_for_attribution": 1,
            "low_confidence_share_threshold": 1.0,
            "unstable_regime_fallback": None,
            "unknown_regime_fallback": _BULL,
            "require_stability_for_attribution": False,
        },
    )

    assert result.labels.loc[0, "regime_label"] == _BULL
    for dimension, column in REGIME_STATE_COLUMNS.items():
        allowed = {state.label for state in REGIME_TAXONOMY[dimension].states}
        assert result.labels[column].isin(allowed).all()
    assert bool(result.labels.loc[0, "is_defined"]) is True


def test_apply_regime_calibration_tracks_low_confidence_share_when_present() -> None:
    labels = _labels([_BULL, _BULL, _RECOVERY, _RECOVERY])
    labels["regime_confidence"] = [0.95, 0.20, 0.10, 0.90]

    result = apply_regime_calibration(
        labels,
        profile={
            "name": "confidence_gate",
            "min_regime_duration_days": 1,
            "transition_smoothing_window": 1,
            "allow_single_day_flips": True,
            "max_flip_rate": 1.0,
            "max_single_day_flip_share": 1.0,
            "min_observations_per_regime": 1,
            "min_observations_for_attribution": 1,
            "low_confidence_share_threshold": 0.25,
            "unstable_regime_fallback": None,
            "unknown_regime_fallback": _UNDEFINED,
            "require_stability_for_attribution": False,
        },
        confidence_column="regime_confidence",
        low_confidence_threshold=0.50,
    )

    assert result.stability_metrics["low_confidence_share"] == pytest.approx(0.5)
    assert result.profile_flags["exceeds_low_confidence_share"] is True
    assert result.audit["is_low_confidence"].tolist() == [False, True, True, False]


def test_write_regime_calibration_artifacts_persists_expected_json_contract(tmp_path: Path) -> None:
    labels = _labels([_BULL, _BULL, _RECOVERY, _RECOVERY, _RECOVERY])
    result = apply_regime_calibration(labels, profile="baseline")
    output_dir = tmp_path / "regime_calibration" / "case_a"

    manifest = write_regime_calibration_artifacts(
        output_dir,
        result,
        run_id="case_a",
        source_regime_artifact_references={"manifest_path": "regimes/manifest.json"},
        taxonomy_metadata={"taxonomy_version": "regime_taxonomy_v1"},
    )

    assert (output_dir / REGIME_CALIBRATION_FILENAME).exists()
    assert (output_dir / REGIME_CALIBRATION_SUMMARY_FILENAME).exists()
    assert (output_dir / REGIME_STABILITY_METRICS_FILENAME).exists()

    loaded = load_regime_calibration(output_dir)
    summary = load_regime_calibration_summary(output_dir)
    metrics = load_regime_stability_metrics(output_dir)

    assert loaded == manifest
    assert loaded["artifacts"]["regime_calibration_json"] == REGIME_CALIBRATION_FILENAME
    assert summary["profile_name"] == "baseline"
    assert metrics["total_observations"] == 5
    assert loaded["source_regime_artifact_references"]["manifest_path"] == "regimes/manifest.json"


def test_calibration_output_preserves_supported_taxonomy_labels_only() -> None:
    labels = _labels([_BULL, _RECOVERY, _BEAR, _UNDEFINED])
    result = apply_regime_calibration(labels, profile="baseline")

    for dimension, column in REGIME_STATE_COLUMNS.items():
        allowed = {state.label for state in REGIME_TAXONOMY[dimension].states}
        assert result.labels[column].map(lambda value: isinstance(value, str)).all()
        assert result.labels[column].map(lambda value: "=" not in value).all()
        assert result.labels[column].isin(allowed).all()
