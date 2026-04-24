from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    CALIBRATION_PROFILE_RESULTS_FILENAME,
    REGIME_SENSITIVITY_MATRIX_FILENAME,
    REGIME_SENSITIVITY_SUMMARY_FILENAME,
    REGIME_STABILITY_REPORT_FILENAME,
    load_calibration_profile_results,
    load_regime_sensitivity_matrix,
    load_regime_sensitivity_summary,
    load_regime_stability_report,
    run_regime_calibration_sensitivity,
    write_regime_sensitivity_artifacts,
)

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
                "is_defined": all(value != "undefined" for value in mapping.values()),
                "volatility_metric": 0.10 + (0.01 * index) if mapping["volatility"] != "undefined" else float("nan"),
                "trend_metric": 0.02 - (0.01 * (index % 3)) if mapping["trend"] != "undefined" else float("nan"),
                "drawdown_metric": 0.01 * (index % 5) if mapping["drawdown_recovery"] != "undefined" else float("nan"),
                "stress_correlation_metric": 0.20 + (0.02 * (index % 4))
                if mapping["stress"] != "undefined"
                else float("nan"),
                "stress_dispersion_metric": 0.25 + (0.01 * (index % 4))
                if mapping["stress"] != "undefined"
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _returns_frame(length: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.date_range("2026-01-01", periods=length, freq="D", tz="UTC"),
            "strategy_return": [0.01, 0.012, -0.02, -0.015, 0.005, 0.018, -0.004, 0.007][:length],
        }
    )


def test_multi_profile_sensitivity_run_is_deterministic_and_exposes_required_columns() -> None:
    labels = _labels([_BULL, _BULL, _BEAR, _BULL, _RECOVERY, _RECOVERY, _BEAR, _BEAR])

    first = run_regime_calibration_sensitivity(
        labels,
        profiles=["baseline", "conservative", "reactive", "crisis_sensitive"],
    )
    second = run_regime_calibration_sensitivity(
        labels,
        profiles=["baseline", "conservative", "reactive", "crisis_sensitive"],
    )

    assert first.matrix["profile_name"].tolist() == [
        "baseline",
        "conservative",
        "reactive",
        "crisis_sensitive",
    ]
    pd.testing.assert_frame_equal(first.matrix, second.matrix)
    required_columns = {
        "profile_name",
        "transition_count",
        "flip_rate",
        "single_day_flip_share",
        "average_regime_duration",
        "median_regime_duration",
        "minimum_regime_duration",
        "maximum_regime_duration",
        "unstable_regime_share",
        "low_confidence_share",
        "attribution_eligible_regime_count",
        "attribution_ineligible_regime_count",
        "warning_count",
        "fallback_rows_total",
        "stable_profile_rank",
        "stability_score",
        "eligible_for_downstream_decisioning",
    }
    assert required_columns.issubset(set(first.matrix.columns))


def test_stable_profile_ranking_prefers_stable_profiles_and_uses_input_order_tie_break() -> None:
    labels = _labels([_BULL, _BULL, _BULL, _RECOVERY, _RECOVERY, _RECOVERY])

    result = run_regime_calibration_sensitivity(
        labels,
        profiles=[
            {
                "name": "tie_second",
                "min_regime_duration_days": 2,
                "transition_smoothing_window": 1,
                "allow_single_day_flips": False,
                "max_flip_rate": 1.0,
                "max_single_day_flip_share": 1.0,
                "min_observations_per_regime": 1,
                "min_observations_for_attribution": 1,
                "low_confidence_share_threshold": 1.0,
                "unstable_regime_fallback": None,
                "unknown_regime_fallback": _UNDEFINED,
                "require_stability_for_attribution": True,
            },
            {
                "name": "tie_first",
                "min_regime_duration_days": 2,
                "transition_smoothing_window": 1,
                "allow_single_day_flips": False,
                "max_flip_rate": 1.0,
                "max_single_day_flip_share": 1.0,
                "min_observations_per_regime": 1,
                "min_observations_for_attribution": 1,
                "low_confidence_share_threshold": 1.0,
                "unstable_regime_fallback": None,
                "unknown_regime_fallback": _UNDEFINED,
                "require_stability_for_attribution": True,
            },
            {
                "name": "unstable_last",
                "min_regime_duration_days": 1,
                "transition_smoothing_window": 1,
                "allow_single_day_flips": True,
                "max_flip_rate": 0.0,
                "max_single_day_flip_share": 0.0,
                "min_observations_per_regime": 1,
                "min_observations_for_attribution": 1,
                "low_confidence_share_threshold": 1.0,
                "unstable_regime_fallback": None,
                "unknown_regime_fallback": _UNDEFINED,
                "require_stability_for_attribution": False,
            },
        ],
    )

    ranked = result.matrix.set_index("profile_name")
    assert int(ranked.loc["tie_second", "stable_profile_rank"]) == 1
    assert int(ranked.loc["tie_first", "stable_profile_rank"]) == 2
    assert int(ranked.loc["unstable_last", "stable_profile_rank"]) == 3


def test_unstable_profile_detection_flows_into_matrix() -> None:
    labels = _labels(
        [
            _RECOVERY,
            _BULL, _BULL, _BULL,
            _RECOVERY, _RECOVERY, _RECOVERY,
            _BULL, _BULL, _BULL,
            _RECOVERY,
        ]
    )

    result = run_regime_calibration_sensitivity(
        labels,
        profiles=[
            {
                "name": "unstable_runs",
                "min_regime_duration_days": 3,
                "transition_smoothing_window": 1,
                "allow_single_day_flips": False,
                "max_flip_rate": 1.0,
                "max_single_day_flip_share": 1.0,
                "min_observations_per_regime": 2,
                "min_observations_for_attribution": 4,
                "low_confidence_share_threshold": 1.0,
                "unstable_regime_fallback": None,
                "unknown_regime_fallback": _UNDEFINED,
                "require_stability_for_attribution": True,
            },
            "reactive",
        ],
    )

    matrix = result.matrix.set_index("profile_name")
    assert bool(matrix.loc["unstable_runs", "has_unstable_runs"]) is True
    assert bool(matrix.loc["unstable_runs", "is_unstable_profile"]) is True
    assert bool(matrix.loc["reactive", "is_unstable_profile"]) is False


def test_fallback_row_counts_are_surfaced_in_matrix() -> None:
    labels = _labels([_UNDEFINED, _BULL, _BEAR, _BULL, _BEAR])
    labels["regime_confidence"] = [0.9, 0.1, 0.9, 0.1, 0.9]

    result = run_regime_calibration_sensitivity(
        labels,
        profiles=[
            {
                "name": "fallback_profile",
                "min_regime_duration_days": 1,
                "transition_smoothing_window": 1,
                "allow_single_day_flips": True,
                "max_flip_rate": 0.1,
                "max_single_day_flip_share": 0.1,
                "min_observations_per_regime": 1,
                "min_observations_for_attribution": 1,
                "low_confidence_share_threshold": 0.1,
                "unstable_regime_fallback": _UNDEFINED,
                "unknown_regime_fallback": _UNDEFINED,
                "require_stability_for_attribution": True,
            }
        ],
        confidence_column="regime_confidence",
        low_confidence_threshold=0.5,
    )

    row = result.matrix.iloc[0]
    assert int(row["unknown_fallback_rows"]) == 1
    assert int(row["low_confidence_fallback_rows"]) == 2
    assert int(row["unstable_profile_fallback_rows"]) == 5
    assert int(row["fallback_rows_total"]) == 8


def test_optional_performance_attribution_emits_deterministic_profile_summaries() -> None:
    labels = _labels([_BULL, _BULL, _BEAR, _BEAR, _RECOVERY, _RECOVERY, _BULL, _BULL])
    returns_frame = _returns_frame(len(labels))

    result = run_regime_calibration_sensitivity(
        labels,
        profiles=["baseline", "conservative"],
        performance_frame=returns_frame,
        performance_surface="strategy",
        performance_value_column="strategy_return",
    )

    assert not result.performance_summary.empty
    assert set(result.performance_summary.columns) >= {
        "profile_name",
        "best_regime_by_mean_return",
        "worst_regime_by_mean_return",
        "profile_mean_return",
        "profile_volatility",
        "profile_cumulative_return",
    }
    rerun = run_regime_calibration_sensitivity(
        labels,
        profiles=["baseline", "conservative"],
        performance_frame=returns_frame,
        performance_surface="strategy",
        performance_value_column="strategy_return",
    )
    pd.testing.assert_frame_equal(result.performance_summary, rerun.performance_summary)


def test_artifact_persistence_writes_expected_files_and_relative_inventory(tmp_path: Path) -> None:
    labels = _labels([_BULL, _BULL, _RECOVERY, _RECOVERY, _BEAR, _BEAR])
    result = run_regime_calibration_sensitivity(labels, profiles=["baseline", "conservative"])

    manifest = write_regime_sensitivity_artifacts(
        tmp_path,
        result,
        run_id="sensitivity_case",
        source_regime_artifact_references={"manifest_path": "../regimes/manifest.json"},
    )

    assert (tmp_path / REGIME_SENSITIVITY_MATRIX_FILENAME).exists()
    assert (tmp_path / REGIME_SENSITIVITY_SUMMARY_FILENAME).exists()
    assert (tmp_path / REGIME_STABILITY_REPORT_FILENAME).exists()
    assert (tmp_path / CALIBRATION_PROFILE_RESULTS_FILENAME).exists()

    loaded_matrix = load_regime_sensitivity_matrix(tmp_path)
    loaded_summary = load_regime_sensitivity_summary(tmp_path)
    loaded_profiles = load_calibration_profile_results(tmp_path)
    loaded_report = load_regime_stability_report(tmp_path)

    assert len(loaded_matrix) == 2
    assert loaded_summary == manifest
    assert loaded_summary["source_regime_artifact_references"]["manifest_path"] == "../regimes/manifest.json"
    assert loaded_profiles["profile_count"] == 2
    assert "# Regime Stability Report" in loaded_report
    assert "sha256" not in loaded_summary["file_inventory"][REGIME_SENSITIVITY_SUMMARY_FILENAME]
    assert loaded_summary["file_inventory"][REGIME_SENSITIVITY_MATRIX_FILENAME]["path"] == REGIME_SENSITIVITY_MATRIX_FILENAME


def test_sensitivity_run_does_not_mutate_inputs() -> None:
    labels = _labels([_BULL, _RECOVERY, _RECOVERY, _BEAR])
    labels["regime_confidence"] = [0.8, 0.4, 0.7, 0.3]
    returns_frame = _returns_frame(len(labels))
    original_labels = labels.copy(deep=True)
    original_returns = returns_frame.copy(deep=True)

    run_regime_calibration_sensitivity(
        labels,
        profiles=["baseline", "reactive"],
        confidence_column="regime_confidence",
        low_confidence_threshold=0.5,
        performance_frame=returns_frame,
        performance_surface="strategy",
        performance_value_column="strategy_return",
    )

    pd.testing.assert_frame_equal(labels, original_labels)
    pd.testing.assert_frame_equal(returns_frame, original_returns)
