# Regime GMM Classifier (M25)

## Overview

Milestone 25 adds a deterministic Gaussian Mixture Model (GMM) classifier for
regime-shift detection on top of the existing Milestone 24 taxonomy and the
Milestone 25 calibration/policy layers.

This layer does not replace `regime_taxonomy_v1` labels. It complements them by
adding model-based confidence and shift events from canonical regime feature
columns.

## Inputs and Outputs

The classifier consumes a timestamped frame with:

* `ts_utc` (UTC timestamp)
* canonical regime feature columns by default:
  * `volatility_metric`
  * `trend_metric`
  * `drawdown_metric`
  * `stress_correlation_metric`
  * `stress_dispersion_metric`

The classifier emits deterministic outputs sorted by `ts_utc`:

* cluster assignment and previous cluster
* posterior probability (`gmm_posterior_probability`)
* confidence (`gmm_confidence_score`)
* entropy (`gmm_posterior_entropy`)
* entropy-derived confidence (`gmm_entropy_confidence`)
* low-confidence flags
* shift flags and probability-delta shift strength
* policy-compatible confidence fields:
  * `confidence_score`
  * `confidence_bucket`
  * `fallback_flag`
  * `fallback_reason`

A separate posterior matrix is also returned with one column per cluster.

## Determinism Contract

Determinism is enforced through:

* required integer `random_state`
* fixed `n_init` and explicit config serialization
* stable UTC timestamp normalization and sorting
* stable cluster ordering based on fitted component means
* JSON persistence with sorted keys and canonicalized values
* CSV and JSON artifact paths recorded with relative inventory entries

The classifier copies input frames and does not mutate caller data.

## Shift Detection

A cluster-shift event is flagged when:

* cluster label changes versus previous timestamp
* current posterior probability exceeds `min_shift_probability`
* posterior delta exceeds `shift_probability_delta_threshold`
* optional gate: require non-low-confidence rows (`shift_requires_not_low_confidence`)

Shift events are persisted as deterministic event tables.

## Configuration

Main configuration knobs:

* `n_components`
* `random_state`
* `feature_columns`
* `low_confidence_threshold`
* `confidence_thresholds` (`medium`, `high`)
* `ambiguous_margin_threshold`
* `min_shift_probability`
* `shift_probability_delta_threshold`
* `shift_requires_not_low_confidence`

## Artifacts

`write_regime_gmm_artifacts()` writes:

* `regime_gmm_labels.csv`
* `regime_gmm_posteriors.csv`
* `regime_gmm_shift_events.csv`
* `regime_gmm_summary.json`
* `regime_gmm_manifest.json`

The manifest includes deterministic file inventory metadata and avoids recursive
self-hashing for `regime_gmm_manifest.json`.

## Minimal Usage

```python
from src.research.regimes import (
    REGIME_AUDIT_COLUMNS,
    apply_regime_calibration,
    apply_regime_policy,
    classify_market_regimes,
    classify_regime_shifts_with_gmm,
    write_regime_gmm_artifacts,
)

classification = classify_market_regimes(
    market_data,
    config={"return_column": "market_return"},
)

# Use rows with complete canonical regime metrics for GMM fitting.
feature_frame = classification.labels.dropna(subset=list(REGIME_AUDIT_COLUMNS)).reset_index(drop=True)

gmm = classify_regime_shifts_with_gmm(
    feature_frame,
    config={"n_components": 3, "random_state": 42},
)

# Feed confidence into calibration.
labels_with_confidence = classification.labels.merge(
    gmm.labels.loc[:, ["ts_utc", "gmm_confidence_score"]],
    on="ts_utc",
    how="left",
)
calibration = apply_regime_calibration(
    labels_with_confidence,
    profile="baseline",
    confidence_column="gmm_confidence_score",
    low_confidence_threshold=0.60,
)

# Feed policy-compatible confidence fields into policy routing.
confidence_frame = gmm.labels.loc[
    :,
    ["ts_utc", "confidence_score", "confidence_bucket", "fallback_flag", "fallback_reason"],
]
policy = apply_regime_policy(feature_frame, confidence_frame=confidence_frame)

manifest = write_regime_gmm_artifacts("artifacts/regimes/gmm_example", gmm, run_id="gmm_example")
```

## Non-Goals

This layer does not:

* redefine taxonomy labels
* replace `regime_taxonomy_v1`
* introduce live trading behavior
* ingest external data feeds
* make direct adaptive policy decisions inside the classifier
