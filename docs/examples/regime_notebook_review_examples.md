# Regime Notebook Review Examples

This walkthrough shows the lightweight notebook helpers added for M24.6. The
goal is to keep regime-aware review explicit and artifact-first while removing
the repetitive file-loading boilerplate that otherwise clutters notebook cells.

## Import The Public Helpers

```python
from src.research.regimes import (
    available_regime_labels,
    compare_runs_by_regime,
    inspect_regime_artifacts,
    list_attribution_warnings,
    list_transition_categories,
    load_regime_review_bundle,
    render_artifact_inventory_markdown,
    render_attribution_summary_markdown,
    render_comparison_summary_markdown,
    render_transition_highlights_markdown,
    slice_conditional_metrics,
    slice_transition_events,
    slice_transition_windows,
    summarize_fragility_flags,
    summarize_run_regime_extremes,
    summarize_transition_category_extremes,
)
```

## Load One Review Bundle

Use one shared artifact directory when the M24 files are persisted together:

```python
bundle = load_regime_review_bundle("artifacts/regime_review/demo_run")

bundle.available_sections()
bundle.inventory()
```

If the artifact families live in different directories, pass them explicitly:

```python
bundle = load_regime_review_bundle(
    regime="artifacts/regimes/demo_run",
    conditional="artifacts/regime_conditional/demo_run",
    transition="artifacts/regime_transition/demo_run",
    attribution="artifacts/regime_attribution/demo_run",
)
```

`load_regime_review_bundle(...)` also accepts an `ExecutionResult` and explicit
`paths={...}` entries when a notebook already knows the exact files it wants to
load.

## Inspect What Is Available

```python
inspection = inspect_regime_artifacts(bundle)

inspection["artifact_inventory"]
inspection["available_regimes"]
inspection["coverage_summary"]
inspection["alignment_statuses"]
inspection["transition_categories"]
inspection["attribution_warnings"]
```

Common direct inspection helpers:

```python
available_regime_labels(bundle)
list_transition_categories(bundle)
list_attribution_warnings(bundle)
```

These helpers expose deterministic DataFrames and dictionaries. They do not
hide the underlying artifact files or recompute alternate metrics.

## Slice Conditional Metrics And Transition Tables

```python
low_vol = slice_conditional_metrics(
    bundle,
    dimension="volatility",
    regime_label="low_volatility",
)

stress_events = slice_transition_events(
    bundle,
    transition_category="stress_onset",
)

event_rows = slice_transition_windows(
    bundle,
    transition_window_role="event_timestamp",
)
```

`slice_regime_table(...)` powers the specific helpers and supports filters for:

* `dimension`
* `regime_label`
* `volatility_state`
* `trend_state`
* `drawdown_recovery_state`
* `stress_state`
* `coverage_status`
* `alignment_status`
* `transition_category`
* `transition_window_role`
* `surface`
* `run_id`

Missing required columns raise clear errors instead of silently falling back.

## Compare Runs Under One Regime Surface

If `regime_comparison_table.csv` is already present:

```python
compare_runs_by_regime(
    bundle,
    dimension="stress",
    regime_label="high_stress",
)
```

If a notebook already has in-memory `RegimeConditionalResult` objects, the
same helper can build the comparison through the existing M24.5 logic:

```python
compare_runs_by_regime(
    {
        "run_a": conditional_a,
        "run_b": conditional_b,
    },
    dimension="stress",
)
```

Useful follow-on summaries:

```python
summarize_run_regime_extremes(bundle)
summarize_transition_category_extremes(bundle)
summarize_fragility_flags({"run_a": bundle_a, "run_b": bundle_b})
```

## Render Notebook-Friendly Markdown

```python
render_artifact_inventory_markdown(bundle)
render_attribution_summary_markdown(bundle)
render_comparison_summary_markdown(bundle)
render_transition_highlights_markdown(bundle)
```

These helpers return plain markdown strings so notebooks can render compact
review summaries without any widget, dashboard, or plotting dependency.

## Case-Study Readiness Check

A practical notebook review loop for M24.7 should now look like:

1. Load the persisted regime bundle with `load_regime_review_bundle(...)`.
2. Audit available regimes and evidence coverage with `inspect_regime_artifacts(...)`.
3. Slice the conditional metrics and transition windows for the regime of interest.
4. Compare runs under one selected regime dimension or label.
5. Render attribution, comparison, and transition markdown summaries for notes.
6. Use `bundle.inventory()` to see which artifact families still need to be produced before a full case study.
