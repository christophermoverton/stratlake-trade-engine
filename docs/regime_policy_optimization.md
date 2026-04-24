# Regime Policy Optimization

Milestone 25 Issue 4 adds a deterministic policy layer on top of the existing
Milestone 24 regime taxonomy and the Milestone 25 calibration, sensitivity,
and ML-confidence surfaces.

The goal is simple:

What should the research system do differently when regime conditions change?

This layer does not redefine regimes and it does not replace the existing
strategy, alpha, or portfolio engines. It consumes their regime-aware research
artifacts and produces explicit policy decisions that can be inspected,
compared, and persisted.

## What It Does

The regime policy layer:

* consumes canonical regime labels from Milestone 24
* respects calibration profile and stability metadata from Milestone 25 Issue 1
* accepts sensitivity-profile decisioning metadata from Milestone 25 Issue 2
* optionally joins ML confidence output from Milestone 25 Issue 3
* resolves config-driven policy rules for signal scaling, allocation scaling,
  alpha weighting, volatility targeting, gross exposure caps, component caps,
  rebalance gating, and optimizer/allocation overrides
* applies explicit fallback behavior for undefined, unstable, ineligible,
  low-confidence, ambiguous, and unsupported rows
* computes a lightweight adaptive-vs-static comparison for research diagnostics

Adaptive comparison is a deterministic research diagnostic, not a live trading
recommendation.

## What It Does Not Do

The policy layer does not:

* create a new taxonomy
* rename or redefine existing regime labels
* introduce a new ML model family
* replace the current portfolio optimizer
* place trades or act as brokerage logic
* download live market data

## Config Shape

```yaml
regime_aliases:
  risk_on:
    match:
      trend: uptrend
      stress: normal_stress

  risk_off:
    match:
      stress_state: correlation_stress

regime_policy:
  default:
    signal_scale: 1.0
    allocation_scale: 1.0
    alpha_weight_multiplier: 1.0
    volatility_target: 0.10
    gross_exposure_cap: 1.0
    max_component_weight: 1.0
    rebalance_enabled: true
    optimizer_override: null
    allocation_rule_override: null
    fallback_policy: baseline

  confidence:
    min_confidence: 0.60
    low_confidence_fallback: neutral
    ambiguous_fallback: reduce_exposure
    unsupported_fallback: baseline

  regimes:
    risk_on:
      signal_scale: 1.10
      allocation_scale: 1.00
      alpha_weight_multiplier: 1.05
      volatility_target: 0.12
      gross_exposure_cap: 1.0
      max_component_weight: 1.0
      rebalance_enabled: true
      optimizer_override: equal_weight
      allocation_rule_override: pro_rata
      fallback_policy: baseline

    risk_off:
      signal_scale: 0.50
      allocation_scale: 0.60
      alpha_weight_multiplier: 0.75
      volatility_target: 0.06
      gross_exposure_cap: 0.60
      max_component_weight: 0.50
      rebalance_enabled: true
      optimizer_override: risk_parity
      allocation_rule_override: null
      fallback_policy: reduce_exposure
```

## Alias Matching

Aliases are optional shortcuts over existing taxonomy dimensions and label
columns. They do not create new labels.

Supported alias match keys:

* canonical dimensions such as `trend`, `stress`, `volatility`
* canonical state columns such as `trend_state`, `stress_state`
* `regime_label`
* `is_defined`

If multiple aliases match, config order wins deterministically. Exact canonical
`regime_label` policy rules take priority over alias-based rules.

## Fallback Policies

Supported fallback policies:

* `baseline`: use the default static policy rule
* `neutral`: zero-out adaptive scales and exposure
* `cash_proxy`: route to a cash-like zero-exposure decision without inventing a new asset
* `reduce_exposure`: clamp scales and caps to reduced values
* `skip_adaptation`: preserve baseline row behavior

Fallback behavior is always recorded in `fallback_policy_applied` and
`policy_reason`.

## Confidence-Gated Adaptation

When an ML confidence frame is present, policy routing uses:

* `confidence_score`
* `confidence_bucket`
* `fallback_flag`
* `fallback_reason`
* `predicted_label` when available for diagnostics

Rows can be rerouted through explicit low-confidence, ambiguous, or unsupported
fallback behavior. Without a confidence frame, the policy layer still operates
from regime labels alone.

## Adaptive Comparison

The comparison layer is intentionally lightweight:

* strategy surfaces use `adaptive_return = baseline_return * signal_scale`
* portfolio surfaces use `adaptive_return = baseline_return * allocation_scale`
* alpha surfaces use `adaptive_return = baseline_return * alpha_weight_multiplier`

This keeps the output deterministic and avoids pretending to be a full
replacement for the underlying research engines.

## Artifact Contract

The policy layer writes:

* `regime_policy_decisions.csv`
* `regime_policy_summary.json`
* `adaptive_vs_static_comparison.csv`
* `adaptive_policy_manifest.json`

The manifest records:

* schema and taxonomy versions
* source regime artifact references
* calibration and sensitivity profile metadata
* confidence artifact references
* policy config
* output file inventory using relative paths only
* row counts, fallback counts, and decision counts
* comparison summary metrics

As with earlier manifest-style artifacts, the manifest includes its own path in
`file_inventory` but does not self-hash recursively.
