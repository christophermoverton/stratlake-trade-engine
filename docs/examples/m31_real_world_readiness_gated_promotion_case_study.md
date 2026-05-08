# M31 Real-World Readiness-Gated Promotion Case Study

## Objective

Run a market-shaped companion to the synthetic M31 readiness-gated promotion
example. The case study computes M30 statistical diagnostics from pinned,
repository-local price and return rows, evaluates the existing
`configs/statistical_readiness_promotion_gates_example.yml` policy through
`source: metrics`, and writes the usual artifact-first promotion, registry,
review, campaign, and candidate-review summaries.

## Prior Pattern Reused

This example reuses the `real_world_campaign_case_study.py` pattern because it
is the closest existing real-world architecture surface: a runnable script under
`docs/examples/`, deterministic artifacts under `docs/examples/output/`, a
stitched summary over native workflow-style artifacts, and review/candidate
context preservation.

The full Milestone 16 case study can consume repository `features_daily`
partitions. This M31 companion deliberately uses a pinned market-shaped fixture
instead, so validation does not require network access, live market data, or
local feature partitions.

## Prior Case-Study Audit

Reviewed prior real-world and real-data examples before adding this companion:

- `real_world_campaign_case_study.py` and `.md`: best base pattern because it
  stitches campaign, candidate-selection, candidate-review, portfolio, review,
  milestone, and summary artifacts under one deterministic output root.
- `real_world_scenario_sweep_case_study.py` and `.md`: useful for scenario
  matrix shape, path sanitization, and fixture-backed test stubs.
- `real_data_scenario_sweep_case_study.md`: confirms the convention of
  documenting optional repository data prerequisites and inspecting
  orchestration-level `scenario_matrix` artifacts.
- `real_q1_2026_regime_aware_case_study.py`, `.md`, and report: confirms the
  pattern of returning a typed artifact bundle, using repository data when
  available, and keeping generated paths portable in summaries.
- `real_world_resume_workflow_case_study.py`: confirms deterministic local
  fixtures are acceptable for real-world workflow validation when tests should
  avoid optional data dependencies.

The selected pattern writes under `docs/examples/output/<case_study>/`, keeps
docs and artifacts path-relative, documents local prerequisites, and validates
through direct script execution plus focused pytest coverage. This companion
keeps that shape while swapping the full campaign runner for an artifact-level
M31 fixture, because the purpose is to demonstrate promotion-readiness
propagation rather than retest campaign orchestration.

## Execute

```powershell
python docs/examples/m31_real_world_readiness_gated_promotion_case_study.py
```

## Data Availability

The script has no external data prerequisite. It builds three deterministic
market-shaped snapshots locally:

- `broad_market_momentum`: promotion-ready fixture expected to pass the example
  readiness gates
- `balanced_rotation_review`: realistic mixed-return fixture that routes to
  manual review
- `short_history_breakout`: short-history fixture that blocks on effective
  sample size

The outputs are deterministic for the fixed fixture snapshot. The example is a
research workflow validation artifact, not a live-trading or production
promotion-readiness claim.

## Output Location

```text
docs/examples/output/m31_real_world_readiness_gated_promotion_case_study/
```

Artifacts to inspect:

- `summary.json`
- `manifest.json`
- `registry.jsonl`
- `review_summary.json`
- `campaign_summary.json`
- `candidate_review_summary.json`
- `runs/<scenario_id>/market_data.csv`
- `runs/<scenario_id>/strategy_returns.csv`
- `runs/<scenario_id>/metrics.json`
- `runs/<scenario_id>/promotion_gates.json`

## Readiness Fields

The example gates M30 diagnostics directly through `source: metrics`:

- `effective_n`
- `p_value`
- `hit_rate_p_value`
- `autocorr_lag1`
- `split_mean_diff_p`
- `sharpe_stability_ratio`

Downstream summaries consume the existing `promotion_gate_summary`. They do not
re-derive severity outside `src/research/promotion.py`, and they do not create
`promotion_decision.json` or `promotion_readiness.json`.

## Interpretation

The example policy thresholds are illustrative defaults only. They are useful
for showing how `eligible`, `needs_review`, and `blocked` outcomes propagate
through StratLake artifacts, but they are not universal statistical truth.
