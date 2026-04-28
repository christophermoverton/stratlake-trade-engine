# Regime Review Packs

## Purpose

Regime review packs turn completed regime benchmark-pack and promotion-gate
outputs into review-ready evidence. They answer what was reviewed, what passed,
what failed, why each decision was made, and which source artifacts support the
decision.

The workflow is read-only with respect to benchmark and promotion-gate inputs.
It does not rerun benchmark packs, rerun gates, select candidates, stress test
variants, or change regime taxonomy, calibration, GMM, policy, benchmark, or
gate semantics.

## Inputs

Required inputs:

* `benchmark_matrix.csv` or `benchmark_matrix.json`
* `promotion_gates/gate_results.csv`
* `promotion_gates/decision_summary.json`

Optional inputs are recorded as warnings when absent:

* `benchmark_summary.json`
* `artifact_provenance.json`
* `promotion_gates/gate_results.json`
* `promotion_gates/failed_gates.csv`
* `promotion_gates/warning_gates.csv`
* `promotion_gates/manifest.json`

## Config Example

```yaml
review_name: m26_regime_policy_review
benchmark_path: artifacts/regime_benchmarks/<benchmark_run_id>
promotion_gates_path: artifacts/regime_benchmarks/<benchmark_run_id>/promotion_gates
output_root: artifacts/regime_reviews

ranking:
  decision_order:
    - accepted
    - accepted_with_warnings
    - needs_review
    - rejected
  metric_priority:
    - adaptive_vs_static_sharpe_delta
    - adaptive_vs_static_return_delta
    - adaptive_vs_static_max_drawdown_delta
    - mean_confidence
    - transition_rate
    - policy_turnover
  tie_breakers:
    - variant_name

report:
  write_markdown: true
```

The checked-in example is
`configs/regime_reviews/m26_regime_review_pack.yml`. Replace the placeholder
benchmark run id or pass explicit CLI paths.

## Artifacts

Outputs are written under:

```text
artifacts/regime_reviews/<review_run_id>/
```

Generated files:

* `leaderboard.csv` and `leaderboard.json`
* `decision_log.json`
* `review_summary.json`
* `accepted_variants.csv`
* `warning_variants.csv`
* `needs_review_variants.csv`
* `rejected_candidates.csv`
* `evidence_index.json`
* `config.json`
* `manifest.json`
* `review_inputs.json`
* `report.md` when `report.write_markdown` is true

## Ranking

Ranking is deterministic. Variants are ordered by configured decision order,
then by configured metric priority, then by tie breakers. The default favors:

* higher adaptive-vs-static Sharpe delta
* higher adaptive-vs-static return delta
* higher adaptive-vs-static max-drawdown delta
* higher mean confidence
* lower transition rate
* lower policy turnover
* variant name ascending as the final tie breaker

## Leaderboard

The leaderboard joins benchmark metrics with promotion-gate decisions. It
contains decision status, gate counts, key regime and policy metrics, the
primary reason, source benchmark run id, and source gate config name.

Use `accepted_variants.csv`, `warning_variants.csv`,
`needs_review_variants.csv`, and `rejected_candidates.csv` for filtered review
surfaces derived from the same ranked leaderboard.

## Decision Log

`decision_log.json` is the primary machine-readable explanation artifact. Each
entry includes the variant, final decision, rank, summary text, passed gates,
warning gates, review gates, failed gates, missing metric gates,
not-applicable gates, primary reasons, key metrics, and evidence paths.

## Rejected Candidates

`rejected_candidates.csv` condenses rejected variants into an audit surface with
failure category, failed metrics, source metric values, source thresholds, and
review notes. Use `gate_results.csv` for full row-level gate evidence.

## Evidence Index

`evidence_index.json` maps source artifacts and generated artifacts to each
variant decision. Missing optional artifacts are listed explicitly so reviewers
can distinguish absent evidence from clean evidence.

## CLI

Explicit paths:

```powershell
python -m src.cli.generate_regime_review_pack `
  --benchmark-path artifacts/regime_benchmarks/<benchmark_run_id> `
  --promotion-gates-path artifacts/regime_benchmarks/<benchmark_run_id>/promotion_gates `
  --output-root artifacts/regime_reviews
```

Config-driven:

```powershell
python -m src.cli.generate_regime_review_pack `
  --config configs/regime_reviews/m26_regime_review_pack.yml
```

The CLI prints the review run id, output directory, decision counts, and main
artifact paths. It fails fast when required source artifacts are missing.

