from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.regimes import (  # noqa: E402
    RegimeConditionalConfig,
    RegimeConditionalResult,
    RegimeTransitionAnalysisResult,
    RegimeTransitionConfig,
    available_regime_labels,
    compare_regime_results,
    compare_runs_by_regime,
    inspect_regime_artifacts,
    list_transition_categories,
    load_regime_review_bundle,
    render_artifact_inventory_markdown,
    render_attribution_summary_markdown,
    render_comparison_summary_markdown,
    render_transition_highlights_markdown,
    slice_conditional_metrics,
    summarize_regime_attribution,
    summarize_transition_attribution,
    write_regime_artifacts,
    write_regime_attribution_artifacts,
    write_regime_conditional_artifacts_multi_dimension,
    write_regime_transition_artifacts,
)


def build_demo_regime_review_bundle(output_dir: str | Path) -> Any:
    """Persist one small review bundle and reload it through the notebook helpers."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
            "volatility_state": ["low_volatility", "high_volatility", "high_volatility", "low_volatility", "low_volatility"],
            "trend_state": ["uptrend", "downtrend", "downtrend", "uptrend", "uptrend"],
            "drawdown_recovery_state": ["near_peak", "drawdown", "recovery", "near_peak", "near_peak"],
            "stress_state": ["normal_stress", "correlation_stress", "normal_stress", "normal_stress", "normal_stress"],
            "regime_label": [
                "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress",
                "volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress",
                "volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress",
                "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress",
                "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress",
            ],
            "is_defined": [True, True, True, True, True],
            "volatility_metric": [0.10, 0.40, 0.35, 0.12, 0.11],
            "trend_metric": [0.03, -0.05, -0.01, 0.02, 0.01],
            "drawdown_metric": [0.00, 0.11, 0.08, 0.00, 0.00],
            "stress_correlation_metric": [0.20, 0.85, 0.25, 0.22, 0.21],
            "stress_dispersion_metric": [0.22, 0.40, 0.24, 0.20, 0.19],
        }
    )
    write_regime_artifacts(output_path, labels, metadata={"source": "demo"})

    metrics = pd.DataFrame(
        [
            {
                "regime_label": "high_volatility",
                "dimension": "volatility",
                "observation_count": 2,
                "coverage_status": "sparse",
                "total_return": None,
                "annualized_return": None,
                "volatility": None,
                "annualized_volatility": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "win_rate": None,
            },
            {
                "regime_label": "low_volatility",
                "dimension": "volatility",
                "observation_count": 3,
                "coverage_status": "sufficient",
                "total_return": 0.05,
                "annualized_return": 0.12,
                "volatility": 0.02,
                "annualized_volatility": 0.31,
                "sharpe_ratio": 1.10,
                "max_drawdown": 0.03,
                "win_rate": 0.67,
            },
        ]
    )
    conditional = RegimeConditionalResult(
        surface="strategy",
        dimension="volatility",
        metrics_by_regime=metrics,
        alignment_summary={"total_rows": 5, "matched_defined": 5, "matched_undefined": 0, "unmatched_timestamp": 0},
        config=RegimeConditionalConfig(min_observations=3),
        metadata={"source": "demo"},
    )
    write_regime_conditional_artifacts_multi_dimension(output_path, {"volatility": conditional}, run_id="demo_run")

    events = pd.DataFrame(
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
                "taxonomy_version": "regime_taxonomy_v1",
                "surface": "strategy",
                "pre_observation_count": 1,
                "event_observation_count": 1,
                "post_observation_count": 1,
                "window_observation_count": 3,
                "valid_observation_count": 3,
                "coverage_status": "sufficient",
                "pre_transition_return": 0.01,
                "event_window_return": -0.02,
                "post_transition_return": 0.01,
                "window_cumulative_return": 0.00,
                "window_volatility": 0.02,
                "window_max_drawdown": 0.02,
                "window_win_rate": 0.67,
            }
        ]
    )
    windows = pd.DataFrame(
        [
            {
                "transition_event_id": "stress:0001",
                "transition_event_order": 1,
                "transition_dimension_event_order": 1,
                "transition_ts_utc": pd.Timestamp("2025-01-02T00:00:00Z"),
                "transition_dimension": "stress",
                "transition_previous_state": "normal_stress",
                "transition_current_state": "correlation_stress",
                "transition_label": "stress:normal_stress->correlation_stress",
                "transition_category": "stress_onset",
                "transition_direction": "enter",
                "transition_is_stress_transition": True,
                "transition_window_role": "event_timestamp",
                "transition_row_offset": 0,
                "transition_timestamp_offset_seconds": 0.0,
                "transition_overlap_count": 1,
                "transition_has_window_overlap": False,
                "transition_is_valid_evidence": True,
                "ts_utc": pd.Timestamp("2025-01-02T00:00:00Z"),
                "strategy_return": -0.02,
            }
        ]
    )

    transition_result = RegimeTransitionAnalysisResult(
        surface="strategy",
        events=events.loc[:, [
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
        ]],
        windows=windows,
        event_summaries=events,
        config=RegimeTransitionConfig(min_observations=3, pre_event_rows=1, post_event_rows=1),
        metadata={"source": "demo"},
    )
    write_regime_transition_artifacts(output_path, transition_result, run_id="demo_run")

    attribution = summarize_regime_attribution(conditional, run_id="demo_run")
    transition_attribution = summarize_transition_attribution(transition_result, run_id="demo_run")
    comparison = compare_regime_results({"demo_run": conditional}, dimension="volatility")
    write_regime_attribution_artifacts(
        output_path,
        attribution,
        transition=transition_attribution,
        comparison=comparison,
        run_id="demo_run",
    )

    return load_regime_review_bundle(output_path)


def regime_review_notebook_cell(output_dir: str | Path) -> dict[str, Any]:
    """Load a bundle, inspect it, and render the common notebook summaries."""

    bundle = build_demo_regime_review_bundle(output_dir)
    inspection = inspect_regime_artifacts(bundle)
    return {
        "bundle": bundle,
        "inventory_markdown": render_artifact_inventory_markdown(bundle),
        "available_regimes": available_regime_labels(bundle),
        "coverage": inspection["coverage_summary"],
        "transition_categories": list_transition_categories(bundle),
        "attribution_markdown": render_attribution_summary_markdown(bundle),
        "comparison_markdown": render_comparison_summary_markdown(bundle),
        "transition_markdown": render_transition_highlights_markdown(bundle),
    }


def regime_slicing_notebook_cell(output_dir: str | Path) -> dict[str, Any]:
    """Show common slicing patterns for interactive review."""

    bundle = build_demo_regime_review_bundle(output_dir)
    return {
        "low_volatility": slice_conditional_metrics(bundle, dimension="volatility", regime_label="low_volatility"),
        "comparison": compare_runs_by_regime(bundle, dimension="volatility"),
    }
