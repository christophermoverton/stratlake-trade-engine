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
* posterior L1 delta (`gmm_posterior_l1_delta`) — L1 distance between consecutive posterior vectors
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
* deterministic feature standardization (when enabled)

The classifier copies input frames and does not mutate caller data.

## Feature Scaling (Standardization)

By default, feature standardization is **enabled** (`standardize_features=True`):

* Computes feature means and standard deviations from the input frame
* Replaces zero or near-zero std values using `standardization_epsilon` guard
* Fits and predicts GMM on standardized features
* Persists standardization metadata in `summary["feature_scaling"]`

To disable standardization:

```python
gmm = classify_regime_shifts_with_gmm(
    features,
    config={"standardize_features": False}
)
```

When disabled, `feature_scaling` metadata shows `enabled=False` and `feature_means/stds=None`.

## Minimum Observation Validation

The classifier enforces a configurable minimum observation count before fitting:

* `min_observations` defaults to 30
* Validated before sklearn GaussianMixture fitting
* Error raised if `len(normalized) < min(n_components, min_observations)`

For small deterministic test fixtures, override:

```python
gmm = classify_regime_shifts_with_gmm(
    features,
    config={"n_components": 2, "min_observations": 10}
)
```

## Stable Cluster Mapping

The fitted GMM component order may differ from the standard sklearn ordering.
The classifier enforces a deterministic stable ordering based on sorted component means.

This mapping is recorded in `summary["stable_cluster_mapping"]`:

```json
{
  "gmm_cluster_0": {"sklearn_component": 2},
  "gmm_cluster_1": {"sklearn_component": 0},
  "gmm_cluster_2": {"sklearn_component": 1}
}
```

Labels and shift events use stable cluster indices (0, 1, 2, ...).
This ensures reproducible cluster names across runs.

## Shift Detection

A cluster-shift event is flagged when:

* cluster label changes versus previous timestamp
* current posterior probability exceeds `min_shift_probability`
* posterior delta exceeds `shift_probability_delta_threshold`
* optional gate: require non-low-confidence rows (`shift_requires_not_low_confidence`)

### Shift Probability Delta

`gmm_shift_probability_delta` measures the absolute gain in posterior probability for
the current winning cluster compared to the previous timestamp's posterior for that cluster.

This metric helps distinguish high-confidence transitions from gradual drift.

### Posterior L1 Delta

`gmm_posterior_l1_delta` measures the L1 distance between consecutive posterior
probability vectors. Values range from 0 (no posterior change) to 2 (complete cluster swap).

This metric complements shift probability delta by measuring global posterior stability.

Shift events are persisted as deterministic event tables.

## Configuration

Main configuration knobs:

| Field | Default | Purpose |
|-------|---------|---------|
| `n_components` | 3 | Number of clusters |
| `random_state` | 42 | Random seed for reproducibility |
| `feature_columns` | `REGIME_AUDIT_COLUMNS` | Features to use |
| `standardize_features` | True | Enable mean/std normalization |
| `standardization_epsilon` | 1.0e-12 | Guard for zero std |
| `min_observations` | 30 | Minimum rows required |
| `low_confidence_threshold` | 0.55 | Posterior prob threshold for low confidence |
| `confidence_thresholds` | `{"medium": 0.50, "high": 0.75}` | Bucketing thresholds |
| `ambiguous_margin_threshold` | 0.10 | Top-2 margin for ambiguity detection |
| `min_shift_probability` | 0.55 | Posterior prob minimum to flag shift |
| `shift_probability_delta_threshold` | 0.15 | Delta minimum to flag shift |
| `shift_requires_not_low_confidence` | True | Gate shifts on confidence |

## Artifacts

`write_regime_gmm_artifacts()` writes:

* `regime_gmm_labels.csv` — per-timestamp labels and confidence
* `regime_gmm_posteriors.csv` — posterior probability matrix
* `regime_gmm_shift_events.csv` — detected shift events
* `regime_gmm_summary.json` — metadata and config snapshot
* `regime_gmm_manifest.json` — file inventory with SHA256 hashes

### Summary Metadata

The summary JSON includes:

* `row_count`, `cluster_count`, `posterior_shape` — data shape
* `cluster_order`, `stable_cluster_mapping` — cluster ordering
* `feature_columns`, `feature_scaling` — input features and standardization metadata
* `cluster_distribution`, `confidence_bucket_distribution` — label distributions
* `low_confidence_share`, `mean_entropy`, `shift_event_count` — quality metrics
* `config` — full config snapshot (sorted keys)

### Manifest File Inventory

The manifest includes relative file paths only (no absolute paths).
The manifest file itself is not recursively hashed in its own inventory entry.

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
    config={
        "n_components": 3,
        "random_state": 42,
        "standardize_features": True,
        "min_observations": 30,
    },
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
