# Regime Attribution and Comparison (M24.5)

## Overview

M24.5 adds a deterministic interpretation layer on top of:

* M24.3 regime-conditional metrics
* M24.4 transition-aware event summaries

The goal is to make regime analysis readable in release notes, case studies,
docs, and notebook summaries without changing the underlying classification,
alignment, or evaluation semantics.

This layer is descriptive. It explains where observed performance came from,
where it weakened, how concentrated it appears, and how multiple runs compare
under the same regime dimension. It does not claim causality or statistical
significance.

## Static Regime Attribution

Use `summarize_regime_attribution()` with a `RegimeConditionalResult` from
M24.3.

```python
from src.research.regimes import (
    evaluate_strategy_metrics_by_regime,
    summarize_regime_attribution,
)

conditional = evaluate_strategy_metrics_by_regime(aligned_frame)
attribution = summarize_regime_attribution(conditional, run_id="run_001")
```

The attribution summary is built from the existing `metrics_by_regime` table.
It does not recompute incompatible return or IC logic.

Summary fields include:

* `best_regime`
* `worst_regime`
* `dominant_regime_label`
* `dominant_regime_share`
* `positive_regime_count`
* `negative_regime_count`
* `sparse_regime_count`
* `empty_regime_count`
* `regime_concentration_score`
* `fragility_flag`
* `fragility_reason`

### Concentration and Fragility Heuristics

The heuristics are deterministic and intentionally simple:

* positive contribution share is computed from the selected primary metric over
  positive metric values only
* absolute contribution share is computed from absolute primary metric values
* concentration score is the sum of squared positive contribution shares
* fragility is flagged when one of the following holds:
  * only one regime has sufficient evidence
  * only one regime contributes positively
  * one regime contributes at least 60% of positive contribution
  * concentration score is at least 0.70
  * positive regimes exist but negative regimes are more numerous

These flags are interpretation aids, not proof of overfitting.

### Sparse, Empty, and Alignment Caveats

M24.5 preserves M24.3 evidence semantics:

* `matched_defined` rows remain the only rows used for metric attribution
* sparse and empty regimes remain visible in the attribution table
* `matched_undefined` and `unmatched_timestamp` rows are surfaced as caveats,
  not hidden evidence

## Transition Attribution

Use `summarize_transition_attribution()` with a
`RegimeTransitionAnalysisResult` from M24.4.

```python
from src.research.regimes import (
    analyze_strategy_regime_transitions,
    summarize_transition_attribution,
)

transition_result = analyze_strategy_regime_transitions(aligned_frame, regime_labels)
transition_attribution = summarize_transition_attribution(transition_result)
```

The transition attribution table aggregates M24.4 event summaries by:

* `transition_category`
* `transition_dimension`

For strategy and portfolio surfaces it reports deterministic averages for:

* pre-transition return
* event-window return
* post-transition return
* window cumulative return
* window max drawdown
* window win rate

For alpha surfaces it reports deterministic averages for:

* pre/event/post/window mean IC
* pre/event/post/window mean Rank IC

Additional fields include:

* event counts by coverage state
* adverse event counts
* degradation event counts
* sparse transition warnings
* best and worst transition categories
* `stress_onset` summary when present

These summaries describe behavior around deterministic regime transitions. They
do not imply that the transition caused the behavior.

## Run-to-Run Comparison

Use `compare_regime_results()` to compare multiple runs under one shared regime
dimension.

```python
from src.research.regimes import compare_regime_results

comparison = compare_regime_results(
    {
        "run_a": conditional_a,
        "run_b": conditional_b,
    },
    dimension="stress",
)
```

Comparison ranking rules are explicit and stable:

* compare only runs aligned on the same surface and regime dimension
* rank `sufficient` coverage ahead of `sparse`, and `sparse` ahead of `empty`
* then rank by the selected primary metric
* then break ties by `run_id`

This produces deterministic tables for docs, markdown reports, and later
notebook helpers.

## Markdown Reports and Artifact Outputs

Use `write_regime_attribution_artifacts()` to persist a bundle:

```python
from src.research.regimes import write_regime_attribution_artifacts

manifest = write_regime_attribution_artifacts(
    "artifacts/regime_attribution/run_001",
    attribution,
    transition=transition_attribution,
    comparison=comparison,
    run_id="run_001",
)
```

Files written:

* `regime_attribution_summary.json`
* `regime_attribution_table.csv`
* `regime_comparison_summary.json` when comparison is supplied
* `regime_comparison_table.csv` when comparison is supplied
* `regime_attribution_report.md`
* `regime_attribution_manifest.json`

The markdown report includes:

* Executive Summary
* Regime Attribution Highlights
* Best / Worst Regimes
* Concentration and Fragility Warnings
* Transition Attribution Highlights when provided
* Comparison Tables when provided
* Interpretation Notes
* Limitations and Cautions

## Interpretation Boundaries

Use these surfaces as decision aids:

* to identify regime concentration
* to compare runs on shared regime buckets
* to spot weak or sparse evidence
* to summarize transition sensitivity

Do not use them as:

* proof of causality
* proof of statistical significance
* replacements for underlying M24.3 or M24.4 evidence tables
* live adaptive trading logic
