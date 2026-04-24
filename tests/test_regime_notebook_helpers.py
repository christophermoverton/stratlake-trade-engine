from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd
import pytest

from src.research.regimes import (
    compare_runs_by_regime,
    inspect_regime_artifacts,
    load_regime_review_bundle,
    render_artifact_inventory_markdown,
    render_attribution_summary_markdown,
    render_comparison_summary_markdown,
    render_transition_highlights_markdown,
    slice_conditional_metrics,
    slice_regime_table,
    slice_transition_events,
    slice_transition_windows,
    summarize_fragility_flags,
    summarize_regime_coverage,
    summarize_run_regime_extremes,
    summarize_transition_category_extremes,
)


def _example_namespace() -> dict[str, object]:
    return runpy.run_path("docs/examples/regime_notebook_review_examples.py")


def test_regime_notebook_example_is_import_safe_and_exposes_cells() -> None:
    namespace = _example_namespace()

    assert callable(namespace["build_demo_regime_review_bundle"])
    assert callable(namespace["regime_review_notebook_cell"])
    assert callable(namespace["regime_slicing_notebook_cell"])


def test_load_regime_review_bundle_from_shared_root_and_explicit_paths(tmp_path: Path) -> None:
    bundle = _example_namespace()["build_demo_regime_review_bundle"](tmp_path)

    reloaded = load_regime_review_bundle(
        paths={
            "regime_labels": bundle.source_paths["regime_labels"],
            "conditional_metrics": bundle.source_paths["conditional_metrics"],
            "transition_events": bundle.source_paths["transition_events"],
            "transition_windows": bundle.source_paths["transition_windows"],
            "attribution_summary": bundle.source_paths["attribution_summary"],
            "comparison_table": bundle.source_paths["comparison_table"],
            "attribution_report": bundle.source_paths["attribution_report"],
        }
    )

    assert set(bundle.available_sections()) == {"regime", "conditional", "transition", "attribution", "comparison"}
    assert reloaded.regime_labels is not None
    assert reloaded.conditional_metrics is not None
    assert reloaded.transition_events is not None
    assert reloaded.transition_windows is not None
    assert reloaded.attribution_summary is not None
    assert reloaded.comparison_table is not None
    assert reloaded.attribution_report is not None


def test_inspection_helpers_surface_regimes_coverage_transitions_and_warnings(tmp_path: Path) -> None:
    bundle = _example_namespace()["build_demo_regime_review_bundle"](tmp_path)

    inspection = inspect_regime_artifacts(bundle)

    assert "available_regimes" in inspection
    assert "coverage_summary" in inspection
    assert "transition_categories" in inspection
    assert "attribution_warnings" in inspection
    assert inspection["available_regimes"]["dimension"].tolist() == sorted(
        inspection["available_regimes"]["dimension"].tolist()
    )
    assert "low_volatility" in inspection["available_regimes"]["regime_label"].tolist()
    assert "sufficient" in inspection["coverage_summary"]["coverage_status"].tolist()
    assert "stress_onset" in inspection["transition_categories"]["transition_category"].tolist()
    assert "fragility_reason" in inspection["attribution_warnings"]["warning_type"].tolist()
    assert inspection["alignment_statuses"]["dimension"].eq("volatility").all()


def test_slicing_helpers_preserve_input_and_support_common_filters(tmp_path: Path) -> None:
    bundle = _example_namespace()["build_demo_regime_review_bundle"](tmp_path)
    original_metrics = bundle.conditional_metrics.copy(deep=True)
    original_events = bundle.transition_events.copy(deep=True)

    conditional_slice = slice_conditional_metrics(bundle, dimension="volatility", regime_label="low_volatility")
    event_slice = slice_transition_events(bundle, transition_category="stress_onset")
    window_slice = slice_transition_windows(bundle, transition_window_role="event_timestamp")

    assert conditional_slice["regime_label"].tolist() == ["low_volatility"]
    assert event_slice["transition_category"].tolist() == ["stress_onset"]
    assert window_slice["transition_window_role"].tolist() == ["event_timestamp"]
    pd.testing.assert_frame_equal(bundle.conditional_metrics, original_metrics)
    pd.testing.assert_frame_equal(bundle.transition_events, original_events)


def test_compare_and_summary_helpers_use_persisted_surfaces(tmp_path: Path) -> None:
    bundle = _example_namespace()["build_demo_regime_review_bundle"](tmp_path)

    comparison = compare_runs_by_regime(bundle, dimension="volatility", run_id="demo_run")
    extremes = summarize_run_regime_extremes(bundle)
    transition_extremes = summarize_transition_category_extremes(bundle)
    fragility = summarize_fragility_flags({"demo_run": bundle})

    assert comparison["run_id"].tolist() == ["demo_run", "demo_run"]
    assert not extremes.empty
    assert not transition_extremes.empty
    assert fragility.loc[0, "run_id"] == "demo_run"
    assert fragility.loc[0, "fragility_flag"] in (True, False)


def test_markdown_helpers_are_deterministic(tmp_path: Path) -> None:
    bundle = _example_namespace()["build_demo_regime_review_bundle"](tmp_path)

    inventory = render_artifact_inventory_markdown(bundle)
    attribution = render_attribution_summary_markdown(bundle)
    comparison = render_comparison_summary_markdown(bundle)
    transition = render_transition_highlights_markdown(bundle)

    assert inventory == render_artifact_inventory_markdown(bundle)
    assert attribution == render_attribution_summary_markdown(bundle)
    assert comparison == render_comparison_summary_markdown(bundle)
    assert transition == render_transition_highlights_markdown(bundle)
    assert "## Artifact Inventory" in inventory
    assert "## Attribution Summary" in attribution
    assert "## Comparison Summary" in comparison
    assert "## Transition Highlights" in transition


def test_error_paths_are_clear_for_missing_files_and_columns(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="regime_labels"):
        load_regime_review_bundle(paths={"regime_labels": tmp_path / "missing.csv"})

    with pytest.raises(KeyError, match="coverage_status"):
        summarize_regime_coverage(pd.DataFrame({"dimension": ["volatility"], "regime_label": ["low_volatility"]}))

    with pytest.raises(KeyError, match="transition_category"):
        slice_regime_table(pd.DataFrame({"value": [1]}), transition_category="stress_onset")

    with pytest.raises(KeyError, match="missing"):
        slice_conditional_metrics(
            pd.DataFrame(
                {
                    "dimension": ["volatility"],
                    "regime_label": ["low_volatility"],
                    "coverage_status": ["sufficient"],
                    "observation_count": [3],
                }
            ),
            columns=["missing"],
        )


def test_example_document_mentions_review_workflow() -> None:
    source = Path("docs/examples/regime_notebook_review_examples.md").read_text(encoding="utf-8")

    assert "load_regime_review_bundle" in source
    assert "inspect_regime_artifacts" in source
    assert "slice_conditional_metrics" in source
    assert "compare_runs_by_regime" in source
    assert "render_transition_highlights_markdown" in source
    assert "bundle.inventory()" in source
