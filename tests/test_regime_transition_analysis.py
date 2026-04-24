from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    REGIME_TRANSITION_EVENTS_FILENAME,
    REGIME_TRANSITION_SUMMARY_FILENAME,
    REGIME_TRANSITION_WINDOWS_FILENAME,
    RegimeTransitionConfig,
    RegimeTransitionError,
    align_regimes_to_alpha_windows,
    align_regimes_to_portfolio_windows,
    align_regimes_to_strategy_timeseries,
    analyze_alpha_regime_transitions,
    analyze_portfolio_regime_transitions,
    analyze_strategy_regime_transitions,
    classify_transition_category,
    detect_regime_transitions,
    load_regime_transition_events,
    load_regime_transition_manifest,
    load_regime_transition_summary,
    load_regime_transition_windows,
    tag_transition_windows,
    write_regime_transition_artifacts,
)


def _regime_labels() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    rows = [
        {
            "ts_utc": ts[0],
            "volatility_state": "undefined",
            "trend_state": "undefined",
            "drawdown_recovery_state": "undefined",
            "stress_state": "undefined",
            "regime_label": "volatility=undefined|trend=undefined|drawdown_recovery=undefined|stress=undefined",
            "is_defined": False,
            "volatility_metric": float("nan"),
            "trend_metric": float("nan"),
            "drawdown_metric": float("nan"),
            "stress_correlation_metric": float("nan"),
            "stress_dispersion_metric": float("nan"),
        },
        {
            "ts_utc": ts[1],
            "volatility_state": "low_volatility",
            "trend_state": "uptrend",
            "drawdown_recovery_state": "near_peak",
            "stress_state": "normal_stress",
            "regime_label": "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress",
            "is_defined": True,
            "volatility_metric": 0.10,
            "trend_metric": 0.03,
            "drawdown_metric": 0.00,
            "stress_correlation_metric": 0.20,
            "stress_dispersion_metric": 0.30,
        },
        {
            "ts_utc": ts[2],
            "volatility_state": "high_volatility",
            "trend_state": "uptrend",
            "drawdown_recovery_state": "near_peak",
            "stress_state": "normal_stress",
            "regime_label": "volatility=high_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress",
            "is_defined": True,
            "volatility_metric": 0.40,
            "trend_metric": 0.04,
            "drawdown_metric": 0.00,
            "stress_correlation_metric": 0.25,
            "stress_dispersion_metric": 0.31,
        },
        {
            "ts_utc": ts[3],
            "volatility_state": "high_volatility",
            "trend_state": "downtrend",
            "drawdown_recovery_state": "drawdown",
            "stress_state": "correlation_stress",
            "regime_label": "volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress",
            "is_defined": True,
            "volatility_metric": 0.42,
            "trend_metric": -0.05,
            "drawdown_metric": 0.12,
            "stress_correlation_metric": 0.85,
            "stress_dispersion_metric": 0.40,
        },
        {
            "ts_utc": ts[4],
            "volatility_state": "high_volatility",
            "trend_state": "downtrend",
            "drawdown_recovery_state": "recovery",
            "stress_state": "normal_stress",
            "regime_label": "volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress",
            "is_defined": True,
            "volatility_metric": 0.39,
            "trend_metric": -0.03,
            "drawdown_metric": 0.08,
            "stress_correlation_metric": 0.30,
            "stress_dispersion_metric": 0.28,
        },
        {
            "ts_utc": ts[5],
            "volatility_state": "low_volatility",
            "trend_state": "uptrend",
            "drawdown_recovery_state": "near_peak",
            "stress_state": "normal_stress",
            "regime_label": "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress",
            "is_defined": True,
            "volatility_metric": 0.12,
            "trend_metric": 0.02,
            "drawdown_metric": 0.00,
            "stress_correlation_metric": 0.24,
            "stress_dispersion_metric": 0.22,
        },
        {
            "ts_utc": ts[6],
            "volatility_state": "undefined",
            "trend_state": "undefined",
            "drawdown_recovery_state": "undefined",
            "stress_state": "undefined",
            "regime_label": "volatility=undefined|trend=undefined|drawdown_recovery=undefined|stress=undefined",
            "is_defined": False,
            "volatility_metric": float("nan"),
            "trend_metric": float("nan"),
            "drawdown_metric": float("nan"),
            "stress_correlation_metric": float("nan"),
            "stress_dispersion_metric": float("nan"),
        },
        {
            "ts_utc": ts[7],
            "volatility_state": "high_volatility",
            "trend_state": "downtrend",
            "drawdown_recovery_state": "drawdown",
            "stress_state": "dispersion_stress",
            "regime_label": "volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=dispersion_stress",
            "is_defined": True,
            "volatility_metric": 0.45,
            "trend_metric": -0.06,
            "drawdown_metric": 0.14,
            "stress_correlation_metric": 0.35,
            "stress_dispersion_metric": 0.50,
        },
    ]
    return pd.DataFrame(rows)


def _strategy_aligned() -> pd.DataFrame:
    labels = _regime_labels()
    frame = pd.DataFrame(
        {
            "ts_utc": labels["ts_utc"],
            "strategy_return": [0.00, 0.01, -0.03, -0.04, 0.02, 0.01, 0.00, -0.02],
        }
    )
    return align_regimes_to_strategy_timeseries(frame, labels)


def _alpha_aligned() -> pd.DataFrame:
    labels = _regime_labels()
    frame = pd.DataFrame(
        {
            "ts_utc": labels["ts_utc"],
            "ic": [0.00, 0.05, -0.02, -0.04, 0.02, 0.03, 0.00, -0.01],
            "rank_ic": [0.01, 0.06, -0.01, -0.03, 0.03, 0.02, 0.00, -0.02],
        }
    )
    return align_regimes_to_alpha_windows(frame, labels)


def _portfolio_aligned() -> pd.DataFrame:
    labels = _regime_labels()
    frame = pd.DataFrame(
        {
            "ts_utc": labels["ts_utc"],
            "portfolio_return": [0.00, 0.015, -0.02, -0.03, 0.01, 0.02, 0.00, -0.015],
        }
    )
    return align_regimes_to_portfolio_windows(frame, labels)


def test_detect_regime_transitions_returns_deterministic_events() -> None:
    labels = _regime_labels()
    events = detect_regime_transitions(labels)

    assert not events.empty
    assert list(events.columns) == [
        "event_id",
        "event_order",
        "dimension_event_order",
        "ts_utc",
        "transition_dimension",
        "previous_state",
        "current_state",
        "transition_label",
        "transition_category",
        "transition_direction",
        "is_stress_transition",
        "taxonomy_version",
    ]
    assert events["event_order"].tolist() == list(range(1, len(events) + 1))
    assert "volatility:low_volatility->high_volatility" in events["transition_label"].tolist()


def test_detect_regime_transitions_supports_single_dimension() -> None:
    events = detect_regime_transitions(_regime_labels(), dimensions="stress")

    assert events["transition_dimension"].eq("stress").all()
    assert set(events["transition_category"]) >= {"stress_onset", "stress_relief"}


def test_detect_regime_transitions_skips_undefined_gap_bridging() -> None:
    events = detect_regime_transitions(_regime_labels(), dimensions="composite")

    assert events["ts_utc"].max() != pd.Timestamp("2025-01-08T00:00:00Z")


def test_detect_regime_transitions_no_transition_case() -> None:
    labels = _regime_labels().copy()
    labels.loc[:, ["volatility_state", "trend_state", "drawdown_recovery_state", "stress_state"]] = [
        "low_volatility",
        "uptrend",
        "near_peak",
        "normal_stress",
    ]
    labels.loc[:, "regime_label"] = "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress"
    labels.loc[:, "is_defined"] = True
    labels.loc[:, "volatility_metric"] = 0.10
    labels.loc[:, "trend_metric"] = 0.02
    labels.loc[:, "drawdown_metric"] = 0.00
    labels.loc[:, "stress_correlation_metric"] = 0.20
    labels.loc[:, "stress_dispersion_metric"] = 0.30
    events = detect_regime_transitions(labels, dimensions="all")

    assert events.empty


def test_classify_transition_category_maps_expected_shift_types() -> None:
    assert classify_transition_category(
        dimension="volatility",
        previous_state="low_volatility",
        current_state="high_volatility",
    ) == "volatility_upshift"
    assert classify_transition_category(
        dimension="trend",
        previous_state="uptrend",
        current_state="downtrend",
    ) == "trend_breakdown"
    assert classify_transition_category(
        dimension="drawdown_recovery",
        previous_state="near_peak",
        current_state="recovery",
    ) == "recovery_onset"


def test_tag_transition_windows_marks_pre_event_post_roles_and_overlap() -> None:
    frame = _strategy_aligned()
    events = detect_regime_transitions(_regime_labels(), dimensions="all")
    windows = tag_transition_windows(
        frame,
        events,
        config=RegimeTransitionConfig(pre_event_rows=1, post_event_rows=1),
    )

    assert set(windows["transition_window_role"]) == {"pre_window", "event_timestamp", "post_window"}
    assert windows["transition_has_window_overlap"].any()
    assert windows["transition_overlap_count"].max() >= 2


def test_tag_transition_windows_can_reject_overlap() -> None:
    frame = _strategy_aligned()
    events = detect_regime_transitions(_regime_labels(), dimensions="all")

    with pytest.raises(RegimeTransitionError, match="overlap"):
        tag_transition_windows(
            frame,
            events,
            config=RegimeTransitionConfig(pre_event_rows=1, post_event_rows=1, allow_window_overlap=False),
        )


def test_strategy_transition_summary_uses_only_matched_defined_rows() -> None:
    result = analyze_strategy_regime_transitions(
        _strategy_aligned(),
        _regime_labels(),
        config=RegimeTransitionConfig(pre_event_rows=1, post_event_rows=1, min_observations=2),
    )

    assert not result.event_summaries.empty
    assert result.event_summaries["surface"].eq("strategy").all()
    assert result.event_summaries["coverage_status"].isin(["sufficient", "sparse", "empty"]).all()
    assert (result.event_summaries["valid_observation_count"] <= result.event_summaries["window_observation_count"]).all()


def test_alpha_transition_summary_exposes_ic_metrics() -> None:
    result = analyze_alpha_regime_transitions(
        _alpha_aligned(),
        _regime_labels(),
        config=RegimeTransitionConfig(pre_event_rows=1, post_event_rows=1, min_observations=2),
    )

    assert "window_mean_ic" in result.event_summaries.columns
    assert "window_mean_rank_ic" in result.event_summaries.columns


def test_portfolio_transition_summary_exposes_return_metrics() -> None:
    result = analyze_portfolio_regime_transitions(
        _portfolio_aligned(),
        _regime_labels(),
        config=RegimeTransitionConfig(pre_event_rows=1, post_event_rows=1, min_observations=2),
    )

    assert "window_cumulative_return" in result.event_summaries.columns
    assert "window_max_drawdown" in result.event_summaries.columns


def test_transition_artifacts_roundtrip(tmp_path: Path) -> None:
    result = analyze_strategy_regime_transitions(
        _strategy_aligned(),
        _regime_labels(),
        config=RegimeTransitionConfig(pre_event_rows=1, post_event_rows=1, min_observations=2),
    )
    manifest = write_regime_transition_artifacts(tmp_path, result, run_id="transition_test")

    assert (tmp_path / REGIME_TRANSITION_EVENTS_FILENAME).exists()
    assert (tmp_path / REGIME_TRANSITION_WINDOWS_FILENAME).exists()
    assert (tmp_path / REGIME_TRANSITION_SUMMARY_FILENAME).exists()
    assert (tmp_path / "regime_transition_manifest.json").exists()

    loaded_events = load_regime_transition_events(tmp_path)
    loaded_windows = load_regime_transition_windows(tmp_path)
    loaded_summary = load_regime_transition_summary(tmp_path)
    loaded_manifest = load_regime_transition_manifest(tmp_path)

    assert len(loaded_events) == len(result.events)
    assert len(loaded_windows) == len(result.windows)
    assert loaded_summary["event_count"] == len(result.events)
    assert loaded_manifest["summary"]["event_count"] == len(result.events)
    assert manifest == loaded_manifest


def test_transition_artifacts_are_deterministic(tmp_path: Path) -> None:
    result = analyze_strategy_regime_transitions(
        _strategy_aligned(),
        _regime_labels(),
        config=RegimeTransitionConfig(pre_event_rows=1, post_event_rows=1, min_observations=2),
    )
    write_regime_transition_artifacts(tmp_path, result, run_id="same")
    first_events = (tmp_path / REGIME_TRANSITION_EVENTS_FILENAME).read_text(encoding="utf-8")
    first_windows = (tmp_path / REGIME_TRANSITION_WINDOWS_FILENAME).read_text(encoding="utf-8")
    first_summary = (tmp_path / REGIME_TRANSITION_SUMMARY_FILENAME).read_text(encoding="utf-8")

    write_regime_transition_artifacts(tmp_path, result, run_id="same")
    second_events = (tmp_path / REGIME_TRANSITION_EVENTS_FILENAME).read_text(encoding="utf-8")
    second_windows = (tmp_path / REGIME_TRANSITION_WINDOWS_FILENAME).read_text(encoding="utf-8")
    second_summary = (tmp_path / REGIME_TRANSITION_SUMMARY_FILENAME).read_text(encoding="utf-8")

    assert first_events == second_events
    assert first_windows == second_windows
    assert first_summary == second_summary
