# Regime Taxonomy

## Overview

Milestone 24 introduces a deterministic market-regime interpretation layer.
The first-pass framework lives in `src.research.regimes` and is intentionally
rule based: no clustering, hidden-state inference, external macro data, or
live inputs are used.

The classifier accepts local market prices or returns, derives a timestamp-level
market return stream, and emits auditable labels plus the metrics that produced
them. Multi-symbol inputs are collapsed into an equal-weight market return for
volatility, trend, and drawdown/recovery states. Cross-sectional stress labels
are computed only when enough symbols and lookback history are available.

## Taxonomy Version

The canonical version is `regime_taxonomy_v1`.

The output schema is:

| Column | Meaning |
| --- | --- |
| `ts_utc` | Timezone-aware UTC timestamp. One row per timestamp, sorted ascending. |
| `volatility_state` | Volatility label for the timestamp. |
| `trend_state` | Trend label for the timestamp. |
| `drawdown_recovery_state` | Drawdown or recovery label for the timestamp. |
| `stress_state` | Correlation or dispersion stress label for the timestamp. |
| `regime_label` | Composite label in canonical dimension order. |
| `is_defined` | True only when every dimension is not `undefined`. |
| `*_metric` | Numeric audit metrics used to assign labels. Missing metrics require `undefined` labels. |

Composite labels use this stable ordering:

```text
volatility=<label>|trend=<label>|drawdown_recovery=<label>|stress=<label>
```

## Dimensions

### Volatility

Volatility uses rolling realized market-return volatility. The default label
thresholds are in-sample deterministic quantiles of the valid rolling
volatility series:

* `low_volatility`: metric is at or below the low threshold.
* `normal_volatility`: metric is between the low and high thresholds.
* `high_volatility`: metric is at or above the high threshold.
* `undefined`: insufficient lookback history or unusable returns.

Boundary comparisons are inclusive so exact threshold matches are stable.

### Trend

Trend uses rolling compounded market return over the configured lookback:

* `uptrend`: compounded return is at or above the positive trend threshold.
* `downtrend`: compounded return is at or below the negative trend threshold.
* `sideways`: compounded return is inside the threshold band.
* `undefined`: insufficient lookback history or unusable returns.

### Drawdown / Recovery

Drawdown/recovery uses an equity curve compounded from the timestamp-level
market return stream:

* `near_peak`: drawdown is at or below the near-peak threshold.
* `underwater`: drawdown is positive but below the drawdown threshold and not improving.
* `drawdown`: drawdown is at or above the drawdown threshold and not improving.
* `recovery`: drawdown is positive and improving relative to the previous timestamp.
* `undefined`: the current return cannot support an auditable drawdown state.

### Stress

Stress is cross-sectional and is only defined when enough symbols have complete
lookback history:

* `correlation_stress`: rolling average pairwise correlation is at or above threshold.
* `dispersion_stress`: current cross-sectional return dispersion is at or above threshold.
* `normal_stress`: at least one stress metric is available and neither stress threshold is breached.
* `undefined`: insufficient cross-section, insufficient lookback history, or unusable returns.

Correlation stress takes precedence over dispersion stress when both thresholds
are breached for the same timestamp.

## Usage

```python
from src.research.regimes import RegimeClassificationConfig, classify_market_regimes

result = classify_market_regimes(
    market_data,
    config=RegimeClassificationConfig(return_column="asset_return"),
)

labels = result.labels
metadata = result.metadata
```

For price inputs, omit `return_column` and provide `close` values. Returns are
then computed per symbol, or from the single timestamp series when no symbol
column is present.

## Validation

Use `validate_regime_labels()` to enforce the output contract. Validation
checks required columns, UTC timestamp parsing, duplicate timestamps, sorted
ordering, supported labels, composite-label construction, `is_defined`
semantics, and missing-metric handling.

The classifier validates its own output before returning, so persisted future
artifacts can reuse the same function as a fail-fast contract check.

## Canonical Artifact Outputs (M24.2)

Regime labels are now persisted as first-class deterministic artifacts.

One regime artifact directory contains:

* `regime_labels.csv`: canonical row-level labels and audit metrics.
* `regime_summary.json`: deterministic summary and taxonomy trace metadata.
* `manifest.json`: deterministic file inventory and shape metadata.

The canonical CSV schema is exactly the taxonomy output contract in stable
column order:

```text
ts_utc,
volatility_state,
trend_state,
drawdown_recovery_state,
stress_state,
regime_label,
is_defined,
volatility_metric,
trend_metric,
drawdown_metric,
stress_correlation_metric,
stress_dispersion_metric
```

Persistence guarantees:

* UTF-8 JSON and CSV with `\n` newlines.
* `ts_utc` rendered as `YYYY-MM-DDTHH:MM:SSZ` in UTC.
* deterministic column ordering.
* deterministic JSON key ordering.
* taxonomy-version traceability in summary and manifest payloads.
* file inventory metadata in `manifest.json` for traceability.

Use `src.research.regimes.artifacts.write_regime_artifacts()` to persist
labels and `load_regime_labels()` / `load_regime_summary()` to consume them.

When upstream workflows maintain manifest-style metadata, use
`attach_regime_artifacts_to_manifest()` so regime outputs appear in the same
traceability surface as other research artifacts.

## Deterministic Alignment Contract (M24.2)

`src.research.regimes.alignment` provides reusable exact-timestamp alignment
helpers for strategy, alpha, and portfolio surfaces.

Available helpers:

* `align_regime_labels()` (generic)
* `align_regimes_to_strategy_timeseries()`
* `align_regimes_to_alpha_windows()`
* `align_regimes_to_portfolio_windows()`

Alignment behavior is deterministic and explicit:

* join mode: exact timestamp equality only (`ts_utc`-based).
* row count: output row count always equals target row count.
* row order: output row order always matches target row order.
* no hidden merge keys beyond explicit timestamp.
* no forward-fill/backfill/interpolation.

Output columns are prefixed with `regime_` and include:

* regime state labels per taxonomy dimension.
* `regime_label` and `regime_is_defined`.
* `regime_has_exact_timestamp_match`.
* `regime_alignment_status`.
* `regime_surface` (`strategy`, `alpha`, `portfolio`, or `generic`).

`regime_alignment_status` values:

* `matched_defined`: exact timestamp match and `is_defined == true`.
* `matched_undefined`: exact timestamp match but regime row is undefined
    (typical warmup/insufficient-history conditions).
* `unmatched_timestamp`: target row has no exact regime timestamp match.
* `regime_labels_unavailable`: regime labels were omitted and caller selected
    `unavailable_policy="mark_unmatched"`.

Missing/partial-overlap semantics:

* if target timestamps start before regime warmup coverage, those rows are
    `unmatched_timestamp`.
* if regime rows exist but are undefined, rows are `matched_undefined`.
* unmatched rows receive explicit undefined regime labels; they are never
    silently dropped.
* sparse target series are aligned row-by-row without resampling.
* when no regime frame is provided:
    * `unavailable_policy="raise"` fails fast.
    * `unavailable_policy="mark_unmatched"` keeps all target rows and marks
        them as unavailable.

## Deferred Work

This foundation intentionally does not implement regime-aware strategy
adaptation, portfolio reallocation, attribution reports, transition event
analysis, notebook-specific renderers, macro data ingestion, clustering, or
hidden-state models. Later M24 work should preserve taxonomy versioning,
canonical output schemas, exact-timestamp alignment semantics, composite-label
order, and explicit `undefined` handling unless a deliberate taxonomy migration
is introduced.
