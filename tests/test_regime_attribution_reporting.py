from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    REGIME_ATTRIBUTION_MANIFEST_FILENAME,
    REGIME_ATTRIBUTION_REPORT_FILENAME,
    REGIME_ATTRIBUTION_SUMMARY_FILENAME,
    REGIME_ATTRIBUTION_TABLE_FILENAME,
    REGIME_COMPARISON_SUMMARY_FILENAME,
    REGIME_COMPARISON_TABLE_FILENAME,
    RegimeConditionalConfig,
    RegimeConditionalResult,
    RegimeTransitionAnalysisResult,
    RegimeTransitionConfig,
    compare_regime_results,
    load_regime_attribution_manifest,
    load_regime_attribution_summary,
    load_regime_attribution_table,
    load_regime_comparison_summary,
    load_regime_comparison_table,
    render_regime_attribution_report,
    summarize_regime_attribution,
    summarize_transition_attribution,
    write_regime_attribution_artifacts,
)


def _strategy_result() -> RegimeConditionalResult:
    frame = pd.DataFrame(
        [
            {
                "regime_label": "volatility=high_volatility|trend=downtrend",
                "dimension": "composite",
                "observation_count": 8,
                "coverage_status": "sufficient",
                "total_return": -0.06,
                "annualized_return": -0.20,
                "volatility": 0.03,
                "annualized_volatility": 0.48,
                "sharpe_ratio": -0.40,
                "max_drawdown": 0.10,
                "win_rate": 0.25,
            },
            {
                "regime_label": "volatility=low_volatility|trend=uptrend",
                "dimension": "composite",
                "observation_count": 12,
                "coverage_status": "sufficient",
                "total_return": 0.18,
                "annualized_return": 0.35,
                "volatility": 0.01,
                "annualized_volatility": 0.16,
                "sharpe_ratio": 1.40,
                "max_drawdown": 0.03,
                "win_rate": 0.75,
            },
            {
                "regime_label": "volatility=normal_volatility|trend=uptrend",
                "dimension": "composite",
                "observation_count": 3,
                "coverage_status": "sparse",
                "total_return": None,
                "annualized_return": None,
                "volatility": None,
                "annualized_volatility": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "win_rate": None,
            },
        ]
    )
    return RegimeConditionalResult(
        surface="strategy",
        dimension="composite",
        metrics_by_regime=frame,
        alignment_summary={
            "total_rows": 30,
            "matched_defined": 20,
            "matched_undefined": 5,
            "unmatched_timestamp": 5,
        },
        config=RegimeConditionalConfig(min_observations=5),
        metadata={"source": "unit_test"},
    )


def _alpha_result(run_id_suffix: str, strong_value: float) -> RegimeConditionalResult:
    frame = pd.DataFrame(
        [
            {
                "regime_label": "high_stress",
                "dimension": "stress",
                "observation_count": 10,
                "coverage_status": "sufficient",
                "mean_ic": strong_value,
                "mean_rank_ic": strong_value + 0.01,
                "ic_std": 0.10,
                "rank_ic_std": 0.10,
                "ic_ir": strong_value / 0.10,
                "rank_ic_ir": (strong_value + 0.01) / 0.10,
            },
            {
                "regime_label": "normal_stress",
                "dimension": "stress",
                "observation_count": 10,
                "coverage_status": "sufficient",
                "mean_ic": 0.01,
                "mean_rank_ic": 0.02,
                "ic_std": 0.10,
                "rank_ic_std": 0.10,
                "ic_ir": 0.10,
                "rank_ic_ir": 0.20,
            },
        ]
    )
    return RegimeConditionalResult(
        surface="alpha",
        dimension="stress",
        metrics_by_regime=frame,
        alignment_summary={
            "total_rows": 20,
            "matched_defined": 20,
            "matched_undefined": 0,
            "unmatched_timestamp": 0,
        },
        config=RegimeConditionalConfig(min_observations=5),
        metadata={"run_label": run_id_suffix},
    )


def _transition_result() -> RegimeTransitionAnalysisResult:
    event_summaries = pd.DataFrame(
        [
            {
                "event_id": "stress:0001",
                "event_order": 1,
                "dimension_event_order": 1,
                "ts_utc": pd.Timestamp("2025-01-02T00:00:00Z"),
                "transition_dimension": "stress",
                "previous_state": "normal_stress",
                "current_state": "correlation_stress",
                "transition_label": "stress:normal_stress->correlation_stress",
                "transition_category": "stress_onset",
                "transition_direction": "enter",
                "is_stress_transition": True,
                "taxonomy_version": "m24",
                "surface": "strategy",
                "pre_observation_count": 2,
                "event_observation_count": 1,
                "post_observation_count": 2,
                "window_observation_count": 5,
                "valid_observation_count": 5,
                "coverage_status": "sufficient",
                "pre_transition_return": 0.02,
                "event_window_return": -0.03,
                "post_transition_return": -0.02,
                "window_cumulative_return": -0.05,
                "window_volatility": 0.02,
                "window_max_drawdown": 0.06,
                "window_win_rate": 0.40,
            },
            {
                "event_id": "volatility:0001",
                "event_order": 2,
                "dimension_event_order": 1,
                "ts_utc": pd.Timestamp("2025-01-05T00:00:00Z"),
                "transition_dimension": "volatility",
                "previous_state": "normal_volatility",
                "current_state": "low_volatility",
                "transition_label": "volatility:normal_volatility->low_volatility",
                "transition_category": "volatility_downshift",
                "transition_direction": "down",
                "is_stress_transition": False,
                "taxonomy_version": "m24",
                "surface": "strategy",
                "pre_observation_count": 2,
                "event_observation_count": 1,
                "post_observation_count": 2,
                "window_observation_count": 5,
                "valid_observation_count": 5,
                "coverage_status": "sufficient",
                "pre_transition_return": 0.01,
                "event_window_return": 0.01,
                "post_transition_return": 0.03,
                "window_cumulative_return": 0.05,
                "window_volatility": 0.01,
                "window_max_drawdown": 0.01,
                "window_win_rate": 0.80,
            },
            {
                "event_id": "trend:0001",
                "event_order": 3,
                "dimension_event_order": 1,
                "ts_utc": pd.Timestamp("2025-01-08T00:00:00Z"),
                "transition_dimension": "trend",
                "previous_state": "uptrend",
                "current_state": "downtrend",
                "transition_label": "trend:uptrend->downtrend",
                "transition_category": "trend_breakdown",
                "transition_direction": "state_change",
                "is_stress_transition": False,
                "taxonomy_version": "m24",
                "surface": "strategy",
                "pre_observation_count": 1,
                "event_observation_count": 1,
                "post_observation_count": 0,
                "window_observation_count": 2,
                "valid_observation_count": 2,
                "coverage_status": "sparse",
                "pre_transition_return": 0.00,
                "event_window_return": -0.01,
                "post_transition_return": None,
                "window_cumulative_return": -0.01,
                "window_volatility": 0.02,
                "window_max_drawdown": 0.02,
                "window_win_rate": 0.00,
            },
        ]
    )
    return RegimeTransitionAnalysisResult(
        surface="strategy",
        events=pd.DataFrame(),
        windows=pd.DataFrame(),
        event_summaries=event_summaries,
        config=RegimeTransitionConfig(min_observations=3),
        metadata={"source": "unit_test"},
    )


def test_regime_attribution_summary_is_deterministic_and_sorted() -> None:
    result = summarize_regime_attribution(_strategy_result())

    assert list(result.attribution_table["regime_label"]) == sorted(result.attribution_table["regime_label"].tolist())
    assert result.summary["best_regime"]["regime_label"] == "volatility=low_volatility|trend=uptrend"
    assert result.summary["worst_regime"]["regime_label"] == "volatility=high_volatility|trend=downtrend"
    assert result.summary["dominant_regime_label"] == "volatility=low_volatility|trend=uptrend"


def test_regime_attribution_flags_fragility_when_one_regime_drives_positive_results() -> None:
    result = summarize_regime_attribution(_strategy_result())

    assert result.summary["fragility_flag"] is True
    assert result.summary["fragility_reason"] == "positive performance is isolated to one regime"
    assert result.summary["positive_regime_count"] == 1
    assert result.summary["sparse_regime_count"] == 1


def test_transition_attribution_summarizes_categories_and_sparse_warnings() -> None:
    result = summarize_transition_attribution(_transition_result())

    assert "stress_onset" in result.attribution_table["transition_category"].tolist()
    stress_row = result.attribution_table.loc[
        result.attribution_table["transition_category"] == "stress_onset"
    ].iloc[0]
    assert stress_row["average_post_transition_return"] == -0.02
    assert result.summary["worst_transition_category"]["transition_category"] == "stress_onset"
    assert "trend_breakdown" in result.summary["sparse_transition_categories"]


def test_compare_regime_results_ranks_runs_with_stable_tie_breaks() -> None:
    alpha_a = _alpha_result("a", 0.03)
    alpha_b = _alpha_result("b", 0.03)

    comparison = compare_regime_results(
        {"run_a": alpha_a, "run_b": alpha_b},
        dimension="stress",
    )

    high_stress = comparison.comparison_table.loc[
        comparison.comparison_table["regime_label"] == "high_stress"
    ]
    assert high_stress["run_id"].tolist() == ["run_a", "run_b"]
    assert high_stress["rank_within_regime"].tolist() == [1, 2]


def test_markdown_report_is_deterministic_and_contains_expected_sections() -> None:
    attribution = summarize_regime_attribution(_strategy_result())
    transition = summarize_transition_attribution(_transition_result())
    comparison = compare_regime_results({"run_a": _strategy_result()}, surface=None, dimension="composite")

    report = render_regime_attribution_report(
        attribution,
        transition=transition,
        comparison=comparison,
    )

    assert report == render_regime_attribution_report(attribution, transition=transition, comparison=comparison)
    assert "## Executive Summary" in report
    assert "## Transition Attribution Highlights" in report
    assert "## Comparison Tables" in report
    assert "## Limitations and Cautions" in report


def test_write_regime_attribution_artifacts_persists_expected_files(tmp_path: Path) -> None:
    attribution = summarize_regime_attribution(_strategy_result())
    transition = summarize_transition_attribution(_transition_result())
    comparison = compare_regime_results({"run_a": _strategy_result()}, dimension="composite")

    manifest = write_regime_attribution_artifacts(
        tmp_path,
        attribution,
        transition=transition,
        comparison=comparison,
        run_id="demo_run",
    )

    assert sorted(manifest["artifact_files"]) == [
        REGIME_ATTRIBUTION_MANIFEST_FILENAME,
        REGIME_ATTRIBUTION_REPORT_FILENAME,
        REGIME_ATTRIBUTION_SUMMARY_FILENAME,
        REGIME_ATTRIBUTION_TABLE_FILENAME,
        REGIME_COMPARISON_SUMMARY_FILENAME,
        REGIME_COMPARISON_TABLE_FILENAME,
    ]
    assert load_regime_attribution_summary(tmp_path)["run_id"] == "demo_run"
    assert not load_regime_attribution_table(tmp_path).empty
    assert load_regime_comparison_summary(tmp_path)["surface"] == "strategy"
    assert not load_regime_comparison_table(tmp_path).empty
    assert load_regime_attribution_manifest(tmp_path)["summary"]["comparison_included"] is True


def test_write_regime_attribution_artifacts_without_comparison_skips_comparison_files(tmp_path: Path) -> None:
    attribution = summarize_regime_attribution(_strategy_result())

    manifest = write_regime_attribution_artifacts(tmp_path, attribution, run_id="solo_run")

    assert REGIME_COMPARISON_SUMMARY_FILENAME not in manifest["artifact_files"]
    assert REGIME_COMPARISON_TABLE_FILENAME not in manifest["artifact_files"]
