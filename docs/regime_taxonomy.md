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

## Deferred Work

This foundation intentionally does not implement regime-aware strategy
adaptation, portfolio reallocation, attribution reports, transition event
analysis, persistence manifests, notebook-specific renderers, macro data
ingestion, clustering, or hidden-state models. Later M24 work should preserve
the taxonomy version, column names, timestamp alignment, composite-label order,
and explicit `undefined` semantics unless a deliberate taxonomy migration is
introduced.
