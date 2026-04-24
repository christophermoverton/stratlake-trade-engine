# Regime Calibration Profiles

## Overview

Milestone 25 adds a calibration layer on top of the Milestone 24 regime-aware
stack. Calibration does not redefine the taxonomy, replace labels, or rebuild
classification logic. It consumes canonical regime-label outputs, applies
profile-driven stability controls, and emits auditable stability metrics plus
decision-ready artifact payloads.

Use calibration when the first-pass regime output is too reactive for a
downstream consumer, when transitions need to be smoothed, or when attribution
and comparison workflows need explicit stability gates.

For the conceptual design behind the layer, including the operating hypothesis
and the calibration model, see
`docs/regime_calibration_hypothesis_and_model.md`.

Calibration is designed to answer questions such as:

* is the current regime stream flipping too often?
* should short-lived transitions be suppressed?
* does a regime have enough observations for attribution?
* should undefined, unstable, or low-confidence rows fall back to an explicit
  safe state?

## What Calibration Does Not Do

Calibration does not:

* introduce a new taxonomy version.
* rename existing taxonomy labels.
* infer hidden states or run any ML model.
* use forward-fill, centered smoothing, or other lookahead-sensitive logic.

The output remains deterministic and traceable to the existing
`regime_taxonomy_v1` label contract.

## Built-In Profiles

The calibration module includes these built-in profiles:

* `baseline`: balanced defaults for general review and sensitivity analysis.
* `conservative`: stronger smoothing and longer minimum durations to suppress
  short-lived noise.
* `reactive`: minimal gating for workflows that want to retain fast shifts.
* `crisis_sensitive`: reactive settings intended to preserve abrupt stress
  transitions while keeping audit metrics visible.

Each profile defines:

* `min_regime_duration_days`
* `transition_smoothing_window`
* `allow_single_day_flips`
* `max_flip_rate`
* `max_single_day_flip_share`
* `min_observations_per_regime`
* `min_observations_for_attribution`
* `low_confidence_share_threshold`
* `unstable_regime_fallback`
* `unknown_regime_fallback`
* `require_stability_for_attribution`

## Stability Controls

Calibration operates on validated regime-label frames and applies the following
controls in order:

1. optional trailing-window smoothing over existing composite labels.
2. causal minimum-duration stabilization that only accepts a regime change
   after it persists for the configured duration.
3. optional fallback routing for undefined rows.
4. optional low-confidence fallback routing when a confidence column is
   supplied.
5. unstable-profile detection when flip-rate, one-day-flip share, or
   low-confidence-share thresholds are exceeded.
6. optional full-profile fallback when the output is classified as unstable.

The implementation is causal: smoothing only uses current and prior
observations, and minimum-duration gating never requires future rows to make a
decision.

## Stability Metrics

Calibration computes deterministic summary metrics for the final calibrated
output:

* total observations
* regime count
* transition count
* single-day flip count
* flip rate
* average regime duration
* median regime duration
* minimum regime duration
* maximum regime duration
* unstable regime share
* low-confidence share, when a confidence column is supplied
* attribution-eligible regime count
* attribution-ineligible regime count

Attribution eligibility is evaluated by regime label using the configured
minimum observation threshold and, when enabled, stability requirements.

## Fallback Behavior

Fallback labels must be valid composite labels in canonical taxonomy order:

```text
volatility=<label>|trend=<label>|drawdown_recovery=<label>|stress=<label>
```

No new labels are invented. If a fallback label is incompatible with the
underlying metric availability for a row, calibration fails fast.

Common patterns:

* set `unknown_regime_fallback` to the all-`undefined` composite label for
  warmup or unavailable periods.
* set `unstable_regime_fallback` to the all-`undefined` composite label when
  downstream consumers should avoid unstable classifications.
* leave fallbacks as `None` when the caller wants to audit instability without
  altering the label stream.

## Artifact Contract

Use `write_regime_calibration_artifacts()` to persist a deterministic artifact
bundle.

The bundle contains:

* `regime_calibration.json`: full calibration payload including profile,
  warnings, file inventory, stability metrics, fallback summary, and source
  regime artifact references.
* `regime_calibration_summary.json`: compact summary for review surfaces.
* `regime_stability_metrics.json`: stability metrics only.

Artifact payloads are JSON-only, use sorted keys, and keep persisted paths
portable and relative when callers provide source artifact references.

## Usage

```python
from src.research.regimes import (
    apply_regime_calibration,
    classify_market_regimes,
    write_regime_calibration_artifacts,
)

classification = classify_market_regimes(market_data)

calibration = apply_regime_calibration(
    classification.labels,
    profile="conservative",
)

write_regime_calibration_artifacts(
    "artifacts/regimes/example_run/calibration",
    calibration,
    source_regime_artifact_references={"manifest_path": "../manifest.json"},
)
```

Confidence-aware usage:

```python
calibration = apply_regime_calibration(
    labels_with_confidence,
    profile="baseline",
    confidence_column="regime_confidence",
    low_confidence_threshold=0.50,
)
```

## Review Guidance

Recommended profile starting points:

* sensitivity analysis: `baseline`
* ML confidence gating: `baseline` or `conservative`
* adaptive strategy scaling: `baseline` or `crisis_sensitive`
* portfolio optimization inputs: `conservative`
* stress monitoring: `crisis_sensitive`

If the profile is flagged unstable, treat calibration output as an audit signal
first and a portfolio or attribution input second.