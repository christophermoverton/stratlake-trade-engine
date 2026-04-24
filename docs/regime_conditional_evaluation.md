# Regime-Conditional Evaluation (M24.3)

This document describes the regime-conditional evaluation layer introduced in M24.3. It covers how conditional metrics are computed for strategy, alpha, and portfolio workflows, how aligned regime rows are selected, how edge cases are handled, and what later work will build on these outputs.

---

## Overview

M24.3 extends StratLake's evaluation stack by adding **regime-conditioned metrics** to the strategy, alpha, and portfolio workflows. These metrics answer the question: *How does this strategy, alpha, or portfolio behave inside each market regime?*

The implementation rests directly on M24.2's exact-timestamp alignment infrastructure (`align_regime_labels`, `align_regimes_to_strategy_timeseries`, etc.) and produces deterministic, stable-schema outputs compatible with the existing evaluation architecture.

---

## Regime Alignment Contract

Before calling any conditional evaluation function, the target frame must have regime context attached by an M24.2 alignment helper:

```python
from src.research.regimes import (
    align_regimes_to_strategy_timeseries,
    classify_market_regimes,
)

regime_result = classify_market_regimes(market_frame, config={...})
aligned = align_regimes_to_strategy_timeseries(strategy_frame, regime_result.labels)
```

The alignment adds the following columns (with default prefix `regime_`):

| Column | Description |
|--------|-------------|
| `regime_label` | Composite label joining all dimension states (e.g. `volatility=high_volatility\|trend=uptrend\|...`) |
| `regime_volatility_state` | Volatility dimension state |
| `regime_trend_state` | Trend dimension state |
| `regime_drawdown_recovery_state` | Drawdown-recovery dimension state |
| `regime_stress_state` | Stress dimension state |
| `regime_is_defined` | Whether all dimension labels are fully defined (not `undefined`) |
| `regime_has_exact_timestamp_match` | Whether a timestamp-exact match was found in the regime labels |
| `regime_alignment_status` | One of: `matched_defined`, `matched_undefined`, `unmatched_timestamp`, `regime_labels_unavailable` |
| `regime_surface` | Surface tag applied at alignment time |

---

## Row Selection for Metric Computation

**Only rows with `regime_alignment_status == "matched_defined"` contribute to regime-conditional metric calculations.**

| Status | Treatment |
|--------|-----------|
| `matched_defined` | Included in metric computations |
| `matched_undefined` | Excluded — regime was undefined at this timestamp |
| `unmatched_timestamp` | Excluded — no timestamp-exact regime match found |
| `regime_labels_unavailable` | Excluded — regime labels were not provided |

This is a strict contract: no undefined or unmatched rows are silently treated as valid regime evidence.

---

## Grouping Dimensions

Conditional evaluation can group by:

- `"composite"` — the full multi-dimension `regime_label` column
- `"volatility"` — the `regime_volatility_state` column
- `"trend"` — the `regime_trend_state` column
- `"drawdown_recovery"` — the `regime_drawdown_recovery_state` column
- `"stress"` — the `regime_stress_state` column

Use `evaluate_all_dimensions()` to compute all groupings in one call.

---

## Minimum Observations and Coverage Status

Each regime subset receives a **coverage status**:

| Status | Meaning |
|--------|---------|
| `sufficient` | `observation_count >= min_observations` — metrics are computed |
| `sparse` | `0 < observation_count < min_observations` — metrics are null |
| `empty` | `observation_count == 0` — metrics are null |

The default `min_observations` is **5**. This can be changed via `RegimeConditionalConfig(min_observations=N)`.

Sparse and empty subsets always appear in the output with null metric values rather than being silently dropped. This makes the absence of valid data explicit and auditable.

---

## Strategy Conditional Metrics

```python
from src.research.regimes import evaluate_strategy_metrics_by_regime

result = evaluate_strategy_metrics_by_regime(
    aligned_frame,
    return_column="strategy_return",   # default
    dimension="composite",              # default
    config={"min_observations": 5},
)
```

Metrics produced per regime label:

| Metric | Description |
|--------|-------------|
| `total_return` | Compounded return across regime period |
| `annualized_return` | Annualized using `periods_per_year` |
| `volatility` | Sample standard deviation of returns |
| `annualized_volatility` | Annualized volatility |
| `sharpe_ratio` | Annualized Sharpe (zero risk-free rate) |
| `max_drawdown` | Maximum peak-to-trough drawdown as positive fraction |
| `win_rate` | Fraction of periods with positive returns |

---

## Alpha Conditional Metrics

```python
from src.research.regimes import evaluate_alpha_metrics_by_regime

result = evaluate_alpha_metrics_by_regime(
    aligned_ic_frame,
    ic_column="ic",               # default
    rank_ic_column="rank_ic",     # default
    dimension="composite",
    config={"min_observations": 5},
)
```

Input should be a timestamp-level IC timeseries (as produced by `evaluate_information_coefficient`) with regime columns attached via `align_regimes_to_alpha_windows`.

Metrics produced per regime label:

| Metric | Description |
|--------|-------------|
| `mean_ic` | Mean Pearson IC across regime timestamps |
| `mean_rank_ic` | Mean Spearman Rank IC |
| `ic_std` | Standard deviation of IC values |
| `rank_ic_std` | Standard deviation of Rank IC values |
| `ic_ir` | IC Information Ratio (mean_ic / ic_std) |
| `rank_ic_ir` | Rank IC Information Ratio |

When a regime subset has fewer than `min_observations` IC values, all metrics are null and coverage status is `sparse` or `empty`.

---

## Portfolio Conditional Metrics

```python
from src.research.regimes import evaluate_portfolio_metrics_by_regime

result = evaluate_portfolio_metrics_by_regime(
    aligned_frame,
    return_column="portfolio_return",   # default
    dimension="composite",
    config={"min_observations": 5},
)
```

Metrics produced per regime label (same set as strategy):

| Metric | Description |
|--------|-------------|
| `total_return` | Compounded portfolio return |
| `annualized_return` | Annualized portfolio return |
| `volatility` | Sample standard deviation |
| `annualized_volatility` | Annualized volatility |
| `sharpe_ratio` | Annualized Sharpe |
| `max_drawdown` | Maximum drawdown |
| `win_rate` | Fraction of periods with positive returns |

---

## Evaluating All Dimensions at Once

```python
from src.research.regimes import evaluate_all_dimensions

results = evaluate_all_dimensions(
    aligned_frame,
    surface="strategy",   # or "alpha" or "portfolio"
    return_column="strategy_return",
    config={"min_observations": 5},
)
# results["composite"], results["volatility"], results["trend"], ...
```

Returns a `dict[str, RegimeConditionalResult]` keyed by dimension name.

---

## Artifact Outputs

### Single-Dimension Artifacts

```python
from src.research.regimes import write_regime_conditional_artifacts

manifest = write_regime_conditional_artifacts(
    output_dir="artifacts/regime_conditional/run_001",
    result=result,
    run_id="run_001",
)
```

Produces:

| File | Description |
|------|-------------|
| `metrics_by_regime.csv` | Stable-column-ordered regime metrics table |
| `regime_conditional_summary.json` | Coverage breakdown, alignment summary, metadata |
| `regime_conditional_manifest.json` | Artifact traceability manifest |

### Multi-Dimension Artifacts

```python
from src.research.regimes import write_regime_conditional_artifacts_multi_dimension

manifest = write_regime_conditional_artifacts_multi_dimension(
    output_dir="artifacts/regime_conditional/run_001_all",
    results=results,    # dict from evaluate_all_dimensions()
    run_id="run_001",
)
```

Produces the same three files, with `metrics_by_regime.csv` containing rows from all dimensions combined and a `dimension` column for filtering.

### Loading Artifacts

```python
from src.research.regimes import (
    load_regime_conditional_metrics,
    load_regime_conditional_summary,
    load_regime_conditional_manifest,
)

metrics = load_regime_conditional_metrics("artifacts/regime_conditional/run_001")
summary = load_regime_conditional_summary("artifacts/regime_conditional/run_001")
manifest = load_regime_conditional_manifest("artifacts/regime_conditional/run_001")
```

---

## Artifact Schema

### `metrics_by_regime.csv` — Strategy / Portfolio

```
regime_label, dimension, observation_count, coverage_status,
total_return, annualized_return, volatility, annualized_volatility,
sharpe_ratio, max_drawdown, win_rate
```

### `metrics_by_regime.csv` — Alpha

```
regime_label, dimension, observation_count, coverage_status,
mean_ic, mean_rank_ic, ic_std, rank_ic_std, ic_ir, rank_ic_ir
```

### `regime_conditional_summary.json`

```json
{
  "schema_version": 1,
  "surface": "strategy",
  "dimension": "composite",
  "taxonomy_version": "regime_taxonomy_v1",
  "min_observations": 5,
  "periods_per_year": 252,
  "regime_labels": ["..."],
  "regime_label_count": 4,
  "coverage_breakdown": { "label_X": "sufficient", "label_Y": "sparse" },
  "observation_counts": { "label_X": 18, "label_Y": 3 },
  "alignment_summary": {
    "total_rows": 60,
    "matched_defined": 55,
    "matched_undefined": 3,
    "unmatched_timestamp": 2,
    "regime_labels_unavailable": 0
  },
  "metric_columns": [...]
}
```

---

## Configuration Reference

```python
from src.research.regimes import RegimeConditionalConfig

config = RegimeConditionalConfig(
    min_observations=5,       # minimum obs per regime subset for metric computation
    regime_prefix="regime_",  # must match the alignment helper's output_prefix
    periods_per_year=252,     # annualization factor
    taxonomy_version="regime_taxonomy_v1",
    metadata={},              # propagated to artifact payloads
)
```

---

## Interpretation Guidance

### What conditional metrics tell you

Regime-conditional metrics show how a strategy, alpha, or portfolio performs *inside* each market regime. A strategy with strong aggregate Sharpe may show concentrated performance in one regime and flat or negative returns in others. Conditional evaluation makes this visible and auditable.

### What conditional metrics do not tell you

- **Transition effects.** A strategy may behave differently *around* regime changes. M24.4 adds deterministic transition analysis in [docs/regime_transition_analysis.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/regime_transition_analysis.md), but those outputs remain separate from the static regime-conditioned layer documented here.
- **Statistical significance.** A high `mean_ic` in a regime with sparse observations (coverage status `sparse`) may reflect noise. The `observation_count` and `coverage_status` columns exist precisely to flag this.
- **Causality.** Conditional metrics describe association, not regime-driven causality. Use them alongside narrative analysis.

### Low-sample regimes

When a regime label has fewer than `min_observations` matched-defined rows:
- All metric values are `null` / `None` in the CSV.
- `coverage_status` is `sparse` (1 to min_observations - 1 rows) or `empty` (0 rows).
- The row always appears so the absence is visible, not silent.

Do not interpret null metric values as zero. They are undefined.

### Undefined regime rows

Rows classified as `matched_undefined` by M24.2 represent timestamps where the regime label was valid but all dimension states were `undefined` (typically during warm-up periods or periods of insufficient history). These rows are excluded from metric computations.

### Unmatched timestamp rows

Rows classified as `unmatched_timestamp` had no exact timestamp match in the regime labels frame. These are also excluded.

---

## What Later M24 Work Will Add

M24.3 establishes the conditional evaluation layer. Later M24 issues will build on it:

- **Transition-aware analysis** — metrics around regime change events
- **Regime attribution summaries** — decomposing aggregate performance by regime contribution
- **Regime comparison surfaces** — cross-strategy regime performance comparison
- **Notebook inspection helpers** — interactive regime-aware visualization
- **Canonical regime-aware case studies**

The M24.3 outputs (particularly `metrics_by_regime.csv` and the `RegimeConditionalResult` dataclass) are the stable inputs for those layers.
