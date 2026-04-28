# Regime Benchmark Packs

## Purpose

Milestone 26 adds a deterministic benchmark-pack layer for comparing multiple
regime-aware variants side by side without redefining the underlying taxonomy,
calibration logic, GMM classifier, or policy rules.

This workflow sits above the Milestone 24 taxonomy and the Milestone 25
calibration, classifier, and policy surfaces. It consumes those existing
outputs, computes a shared comparison row per variant, and persists one stable
artifact bundle for reproducible reruns.

## Supported Variants

The first benchmark surface supports variants such as:

* `static_baseline`
* `taxonomy_only`
* `calibrated_taxonomy`
* `gmm_classifier`
* `gmm_calibrated_overlay`
* `policy_optimized`

`gmm_classifier` keeps taxonomy labels as the canonical regime surface and uses
the deterministic GMM classifier as a confidence and transition-disagreement
overlay. The benchmark pack does not relabel GMM clusters as taxonomy states.

## Config Shape

Canonical example:

```yaml
benchmark_name: m26_regime_policy_benchmark
dataset: features_daily
timeframe: 1D
start: "2026-01-01"
end: "2026-12-31"
features_root: data
output_root: artifacts/regime_benchmarks

variants:
  - name: static_baseline
    regime_source: static

  - name: taxonomy_only
    regime_source: taxonomy

  - name: calibrated_taxonomy
    regime_source: taxonomy
    calibration_profile: baseline

  - name: gmm_classifier
    regime_source: gmm

  - name: gmm_calibrated_overlay
    regime_source: gmm
    calibration_profile: baseline

  - name: policy_optimized
    regime_source: policy
    calibration_profile: baseline
    policy_config: ../regime_policies/m25_policy_default.yml
```

Variant order is preserved in the benchmark matrix and is part of the
deterministic run identity.

When `policy_config` is provided as a relative path, the benchmark pack keeps
that relative form in persisted `config.json` and related metadata while still
resolving it internally for file access.

## Artifacts

Each run writes:

* `benchmark_matrix.csv`
* `benchmark_matrix.json`
* `model_comparison.csv`
* `calibration_comparison.csv`
* `policy_comparison.csv`
* `conditional_performance_summary.json`
* `stability_summary.json`
* `transition_summary.json`
* `config.json`
* `manifest.json`
* `input_inventory.json`
* `artifact_provenance.json`

Variant-specific artifacts are also written under `variants/`.

All persisted paths inside JSON and manifest outputs are relative to the run
root when possible.

## Metric Notes

The benchmark matrix records one row per variant and keeps a stable column set.

Missing optional metrics are represented as:

* JSON `null` values for unavailable scalar fields
* empty CSV cells for the corresponding matrix export
* explicit warning strings in the `warnings` column when needed

Examples:

* static variants do not emit regime-duration metrics
* taxonomy-only variants do not emit classifier confidence metrics
* non-policy variants do not emit adaptive-vs-static deltas

`taxonomy_ml_disagreement_rate` is defined as a transition-disagreement proxy:
it compares taxonomy regime changes against GMM cluster changes on shared
timestamps. It is not a direct label-equality score because GMM cluster labels
are not canonical taxonomy labels.

`include_conditional_performance` is currently reserved for a follow-up issue.
When enabled, the benchmark pack emits
`conditional_performance_summary.json` as an explicit placeholder instead of
silently pretending a dedicated conditional-performance comparison artifact
exists.

## CLI

```powershell
python -m src.cli.run_regime_benchmark_pack --config configs/regime_benchmark_packs/m26_regime_policy_benchmark.yml
```

The checked-in full-year style config may require local `features_daily`
coverage for its selected evaluation window. If that local dataset is absent
or incomplete, the run is expected to fail fast rather than fall back to
synthetic data.

## Interpretation Guidance

Use the benchmark pack to compare:

* how much calibration stabilizes regime durations and transitions
* where the GMM layer identifies low-confidence or high-entropy windows
* whether the configured policy changes adaptive-vs-static diagnostics
* which variants rely on warnings or unavailable optional surfaces

This is still a research comparison layer. It does not introduce live trading,
promotion gates, portfolio construction, or stress testing.

## Non-Goals

The benchmark pack does not:

* create a new taxonomy
* train a new classifier family
* replace calibration profile behavior
* replace the existing policy layer
* automatically select candidates or allocate capital
