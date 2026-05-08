# Milestone 31 Release Notes

Milestone 31 integrates Milestone 30 statistical readiness diagnostics into the
existing StratLake promotion policy workflow. It keeps `promotion_gates.json` as
the canonical promotion artifact and keeps `promotion_gate_summary` as the
canonical summary path through manifests, registries, campaigns, reviews, and
candidate-review context.

## What Changed

* Promotion gates now support optional per-gate `severity`:
  `warn`, `review`, `reject`, and `block`.
* Failed or non-skipped missing severity gates resolve deterministically by:
  `block > reject > review > warn`.
* Severity maps to promotion status as:
  `warn -> warn`, `review -> needs_review`, `reject -> rejected`, and
  `block -> blocked`.
* Existing configs without `severity` keep their legacy `status_on_pass` and
  `status_on_fail` behavior.
* Promotion summaries now include additive audit fields such as
  `highest_severity`, `severity_counts`, per-severity gate counts, and
  `decision_reason_codes`.
* Registry review metadata maps expanded promotion outcomes explicitly:
  `eligible -> candidate`, `warn -> needs_review`,
  `needs_review -> needs_review`, `rejected -> rejected`, and
  `blocked -> rejected`.
* Campaign summaries, scenario matrices, milestone reports, and candidate-review
  summaries preserve readiness-aware context from existing artifacts without
  re-evaluating severity downstream.

## Canonical Example

Run the deterministic example:

```bash
python docs/examples/m31_readiness_gated_promotion_case_study.py
```

The example uses only synthetic M30-style metric payloads and the example policy
in:

```text
configs/statistical_readiness_promotion_gates_example.yml
```

It writes artifacts under:

```text
docs/examples/output/m31_readiness_gated_promotion_case_study/
```

The example demonstrates `eligible`, `warn`, `needs_review`, and `blocked`
outcomes, stable reason codes, registry/review metadata propagation, campaign
final outcome propagation, scenario severity context, and candidate-review
`promotion_context` preservation.

## Validation

Recommended release checks:

```bash
python -m pytest tests/test_promotion_gates.py tests/test_experiment_registry.py tests/test_research_review.py tests/test_candidate_review.py tests/test_cli_run_research_campaign.py tests/test_campaign_milestone_reporting.py tests/test_m31_readiness_gated_promotion_case_study.py
python docs/examples/m31_readiness_gated_promotion_case_study.py
```

## Non-Goals

M31 does not add FDR/q-values, Deflated Sharpe Ratio, HAC/Newey-West adjusted
p-values, bootstrap confidence intervals, dashboards, database backends, live
market-data dependencies, a second promotion engine, or separate
`promotion_decision.json` / `promotion_readiness.json` artifacts.

The example thresholds are illustrative policy defaults. They are not universal
statistical truth and should be calibrated to the strategy family, portfolio
context, and review governance process.
