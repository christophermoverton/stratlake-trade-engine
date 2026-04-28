# Regime Promotion Gates

## Purpose

Milestone 26 Issue 2 adds a deterministic governance layer above regime
benchmark packs. It consumes an existing benchmark-pack artifact directory,
evaluates configured trust gates, and classifies each benchmarked variant into a
promotion outcome.

The workflow does not recompute benchmark packs and does not mutate benchmark
artifacts. It writes a separate `promotion_gates/` bundle that downstream review
pack work can consume.

## Relationship To Benchmark Packs

The evaluator reads:

* `benchmark_matrix.csv` or `benchmark_matrix.json`
* `benchmark_summary.json` when available
* `artifact_provenance.json` when available
* `transition_summary.json` when available
* `policy_comparison.csv` when available

The benchmark matrix provides the core variant rows. Transition and policy
artifacts add side metrics such as transition concentration, transition
instability, and high-volatility regime drawdown.

## Config Example

```yaml
gate_name: m26_regime_policy_governance
decision_policy: strict_with_warnings

outcomes:
  pass: accepted
  warn: accepted_with_warnings
  review: needs_review
  fail: rejected

missing_metric_policy:
  required_metric_missing: fail
  optional_metric_missing: warn
  not_applicable_metric: ignore

confidence:
  enabled: true
  required_for_variants:
    - gmm_classifier
    - gmm_calibrated_overlay
    - policy_optimized
  min_mean_confidence: 0.55
  min_median_confidence: 0.60
  max_low_confidence_pct: 0.25
```

The checked-in example is
`configs/regime_promotion_gates/m26_regime_policy_gates.yml`.

## Gate Categories

Supported categories:

* `confidence`: mean confidence, median confidence, low-confidence percentage
* `entropy`: mean entropy and high-entropy percentage
* `stability`: average regime duration, dominant regime share, transition rate
* `transition_behavior`: transition rate, instability score, concentration
* `adaptive_uplift`: return, Sharpe, and max-drawdown deltas versus static
* `drawdown`: max drawdown and high-volatility regime drawdown
* `policy_turnover`: policy turnover and policy state changes

Each category can be disabled with `enabled: false`. A category may also set
`failure_impact: review` to route threshold failures to `needs_review` instead
of rejection.

## Missing And Not-Applicable Metrics

Missing required metrics use `required_metric_missing`. Missing optional metrics
use `optional_metric_missing`. Metrics that do not apply to a variant, such as
classifier confidence on `static_baseline`, are preserved as `not_applicable`
rows and use `not_applicable_metric`.

This makes missing and not-applicable behavior visible in `gate_results.*`
instead of silently dropping unavailable metrics.

## Decision Mapping

Variant decisions are deterministic:

* any `fail` impact -> `rejected`
* otherwise any `review` impact -> `needs_review`
* otherwise any `warn` impact -> `accepted_with_warnings`
* only pass, info, ignored, or not-applicable rows -> `accepted`

The final outcome names come from the `outcomes` config block.

## Artifacts

By default, outputs are written under:

```text
<benchmark_path>/promotion_gates/
```

Generated files:

* `gate_config.json`
* `gate_results.csv`
* `gate_results.json`
* `decision_summary.json`
* `failed_gates.csv`
* `warning_gates.csv`
* `manifest.json`

The gate results contain one row per variant, gate, and metric. Persisted paths
are relative where possible, and row ordering follows benchmark matrix order and
fixed gate order.

## CLI

```powershell
python -m src.cli.run_regime_promotion_gates `
  --benchmark-path artifacts/regime_benchmarks/<benchmark_run_id> `
  --config configs/regime_promotion_gates/m26_regime_policy_gates.yml
```

The CLI prints the benchmark run id, gate config name, output directory,
decision counts, and primary artifact paths. It fails fast when the benchmark
path lacks `benchmark_matrix.csv` or `benchmark_matrix.json`, or when the gate
config is missing or malformed.

## Interpretation Guidance

Use `decision_summary.json` for the concise variant-level outcome. Use
`gate_results.csv` when auditing why a variant passed, warned, required review,
or failed. `failed_gates.csv` and `warning_gates.csv` are filtered convenience
surfaces for automation and later review-pack generation.

## Non-Goals

This workflow does not:

* create human-readable final review packs
* select candidates
* run stress tests
* alter regime taxonomy, calibration, GMM, benchmark-pack, or policy behavior
* add live-trading behavior
