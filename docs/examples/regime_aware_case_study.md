# Canonical Regime-Aware Case Study (Milestone 24)

## Objective

This is the canonical Milestone 24 case study for StratLake's regime-aware
research stack. It demonstrates the intended end-to-end workflow:

1. classify deterministic market regimes with the M24.1 classifier
2. persist canonical regime artifacts with M24.2
3. align regime context to strategy and alpha evaluation surfaces
4. compute conditional metrics with M24.3
5. analyze transition windows and stress-state shifts with M24.4
6. generate attribution and comparison outputs with M24.5
7. review the resulting bundles through the M24.6 notebook helpers

The example is intentionally deterministic and validation-oriented. It uses
small repository fixtures rather than live market data, but it exercises the
shipped APIs and artifact contracts directly.

## What It Demonstrates

This case study covers:

1. one canonical strategy path from regime classification through transition
   attribution and final interpretation
2. one canonical alpha path with two deterministic variants
   (`alpha_baseline` and `alpha_defensive`) so M24.5 comparison surfaces are
   visible in a realistic review flow
3. notebook-ready artifact bundles that can be loaded through
   `load_regime_review_bundle(...)` without custom file plumbing

The portfolio surface is intentionally deferred here to keep the flagship M24
example concise, fast, and easy to validate. The same regime APIs already
support portfolio workflows through the dedicated surface functions introduced in
M24.3 and M24.4.

## Execute

```powershell
python docs/examples/regime_aware_case_study.py
```

To write the artifacts explicitly to the canonical committed output root:

```powershell
python docs/examples/regime_aware_case_study.py --output-root docs/examples/output/regime_aware_case_study
```

## Output Root

The example writes to:

```text
docs/examples/output/regime_aware_case_study/
```

Primary stitched summary:

```text
docs/examples/output/regime_aware_case_study/summary.json
```

Interpretation notes:

```text
docs/examples/output/regime_aware_case_study/final_interpretation.md
```

## Expected Output Tree

```text
docs/examples/output/regime_aware_case_study/
  summary.json
  final_interpretation.md
  notebook_review/
    strategy_inventory.md
    strategy_attribution_summary.md
    strategy_transition_highlights.md
    alpha_baseline_inventory.md
    alpha_baseline_attribution_summary.md
    alpha_baseline_transition_highlights.md
    alpha_baseline_comparison_summary.md
  regime_bundle/
    regime_labels.csv
    regime_summary.json
    manifest.json
  strategy_bundle/
    regime_labels.csv
    regime_summary.json
    manifest.json
    metrics_by_regime.csv
    regime_conditional_summary.json
    regime_conditional_manifest.json
    regime_transition_events.csv
    regime_transition_windows.csv
    regime_transition_summary.json
    regime_transition_manifest.json
    regime_attribution_summary.json
    regime_attribution_table.csv
    regime_attribution_report.md
    regime_attribution_manifest.json
  alpha_baseline_bundle/
    regime_labels.csv
    regime_summary.json
    manifest.json
    metrics_by_regime.csv
    regime_conditional_summary.json
    regime_conditional_manifest.json
    regime_transition_events.csv
    regime_transition_windows.csv
    regime_transition_summary.json
    regime_transition_manifest.json
    regime_attribution_summary.json
    regime_attribution_table.csv
    regime_comparison_summary.json
    regime_comparison_table.csv
    regime_attribution_report.md
    regime_attribution_manifest.json
  alpha_defensive_bundle/
    regime_labels.csv
    regime_summary.json
    manifest.json
    metrics_by_regime.csv
    regime_conditional_summary.json
    regime_conditional_manifest.json
    regime_transition_events.csv
    regime_transition_windows.csv
    regime_transition_summary.json
    regime_transition_manifest.json
    regime_attribution_summary.json
    regime_attribution_table.csv
    regime_attribution_report.md
    regime_attribution_manifest.json
```

## Artifact Guide

`regime_bundle/`
: M24.1 and M24.2 outputs only. This is the canonical persisted regime-label
surface and the root input to all later regime-aware analysis.

`strategy_bundle/metrics_by_regime.csv`
: M24.3 strategy conditional metrics across the composite label and each regime
dimension. Used later by attribution and notebook slicing.

`strategy_bundle/regime_transition_events.csv`
: M24.4 detected transition events from the persisted regime labels.

`strategy_bundle/regime_transition_windows.csv`
: M24.4 tagged evidence windows around each transition event.

`strategy_bundle/regime_attribution_summary.json`
: M24.5 summary used for interpretation notes and notebook review.

`strategy_bundle/regime_attribution_report.md`
: M24.5 markdown report that stitches regime attribution and transition
attribution into one readable surface.

`alpha_baseline_bundle/regime_comparison_table.csv`
: M24.5 comparison surface across the two alpha variants under one shared regime
taxonomy.

`notebook_review/*.md`
: M24.6-ready markdown snapshots rendered from the persisted bundles. These are
not required by the APIs, but they make review and release-note preparation
faster.

## Workflow

The script follows the intended public API path:

```python
classification = classify_market_regimes(...)
write_regime_artifacts(...)

strategy_aligned = align_regimes_to_strategy_timeseries(...)
strategy_results = evaluate_all_dimensions(strategy_aligned, surface="strategy")
strategy_transition = analyze_strategy_regime_transitions(...)
strategy_attribution = summarize_regime_attribution(strategy_results["composite"])
strategy_transition_attribution = summarize_transition_attribution(strategy_transition)
write_regime_attribution_artifacts(...)
```

The alpha path mirrors the same shape with
`align_regimes_to_alpha_windows(...)`,
`evaluate_all_dimensions(..., surface="alpha")`, and
`analyze_alpha_regime_transitions(...)`.

## Notebook Review Loop

The case study is structured so the notebook helpers can inspect the bundles
directly:

```python
from src.research.regimes import (
    inspect_regime_artifacts,
    load_regime_review_bundle,
    render_attribution_summary_markdown,
    render_transition_highlights_markdown,
    slice_conditional_metrics,
)

strategy_bundle = load_regime_review_bundle(
    "docs/examples/output/regime_aware_case_study/strategy_bundle"
)

inspect_regime_artifacts(strategy_bundle)
slice_conditional_metrics(strategy_bundle, dimension="stress")
render_attribution_summary_markdown(strategy_bundle)
render_transition_highlights_markdown(strategy_bundle)
```

For the comparison surface:

```python
alpha_bundle = load_regime_review_bundle(
    "docs/examples/output/regime_aware_case_study/alpha_baseline_bundle"
)
```

That bundle includes the persisted comparison files, so
`render_comparison_summary_markdown(alpha_bundle)` works without custom wiring.

## Reading The Result

The deterministic interpretation in this case study is meant to answer a small,
practical set of questions:

1. which regimes were strongest or weakest on the strategy surface
2. whether the positive evidence is concentrated in one regime or more evenly
   distributed
3. which transition categories were most adverse
4. whether the defensive alpha variant changes the regime profile compared with
   the baseline alpha

Read the outputs in this order:

1. `summary.json`
2. `strategy_bundle/regime_attribution_report.md`
3. `alpha_baseline_bundle/regime_attribution_report.md`
4. `final_interpretation.md`
5. the notebook review markdown files under `notebook_review/`

## Caution Boundaries

- This is a deterministic validation case study, not a live market claim.
- The interpretation is descriptive and artifact-backed; it does not claim that
  the regimes caused the observed strategy or alpha behavior.
- Sparse and empty slices are evidence warnings, not positive proof of
  robustness.
- The downstream strategy and alpha surfaces are small deterministic fixtures
  chosen to exercise the review stack clearly and repeatably.

## Related Docs

- [regime_taxonomy.md](../regime_taxonomy.md)
- [regime_conditional_evaluation.md](../regime_conditional_evaluation.md)
- [regime_transition_analysis.md](../regime_transition_analysis.md)
- [regime_attribution_and_comparison.md](../regime_attribution_and_comparison.md)
- [regime_notebook_review_examples.md](regime_notebook_review_examples.md)
