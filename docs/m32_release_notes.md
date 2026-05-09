# Milestone 32 Release Notes

Milestone title: `Milestone 32 - Promotion Governance Observability & Audit Reporting`

Milestone 32 adds deterministic governance reporting for the promotion outcomes
introduced and propagated by Milestone 31. It is an observability and audit
layer over existing artifacts. It does not evaluate promotion gates, replay
policy, or create a second promotion decision path.

## Relationship To Milestone 31

Milestone 31 made promotion gates severity-aware and propagated additive
promotion summaries through registries, manifests, campaign summaries, review
summaries, and candidate-review context. Milestone 32 reads those artifacts and
makes the outcomes easier to compare, audit, validate, and govern.

The source-of-truth boundary is unchanged:

* `promotion_gates.json` remains the canonical promotion policy artifact.
* `promotion_gate_summary` remains the canonical summary carried by manifests,
  registry rows, reviews, campaigns, and candidate-review context.
* Governance reports aggregate existing promotion outcomes. They do not
  recompute `promotion_status`.

## Major Capabilities

* Governance report bundles under `artifacts/promotion_governance/<report_id>/`.
* `promotion_outcome_matrix.csv` for row-level comparison across strategy,
  alpha, portfolio, campaign, scenario, review, and candidate-review records.
* Campaign and scenario observability for campaign summaries, scenario catalogs,
  checkpoints, manifests, and propagated promotion summaries.
* Candidate-selection visibility for candidate-review rows, selected candidate
  IDs, selected run IDs, upstream run IDs, and promotion context.
* Shared promotion/review status normalization with auditable
  `legacy_status_normalized` validation findings.
* Structured consistency validation in `consistency_validation.json`.
* Deterministic JSON, CSV, Markdown, and manifest output.
* Unit, regression, and M31-style integration test coverage.

## Governance Bundle

Each report writes the existing M32 bundle:

* `promotion_governance_summary.json`
* `promotion_outcome_matrix.csv`
* `reason_code_summary.csv`
* `severity_summary.csv`
* `workflow_summary.csv`
* `consistency_validation.json`
* `promotion_governance_report.md`
* `manifest.json`

M32 does not write `promotion_decision.json` or `promotion_readiness.json`.

## Non-Goals

M32 does not add:

* a second promotion engine
* policy replay or sensitivity analysis
* new statistical diagnostics
* new promotion status semantics
* campaign or candidate-selection execution changes
* dashboard, database, or UI services
* live market data dependencies

## Validation Summary

Recommended validation for the M32 governance surface:

```bash
python -m pytest tests/test_promotion_governance.py tests/test_promotion_governance_integration.py
python -m pytest tests/test_docs_path_portability.py
python -m pytest tests/test_promotion_gates.py tests/test_experiment_registry.py tests/test_research_review.py tests/test_candidate_review.py tests/test_cli_run_research_campaign.py tests/test_campaign_milestone_reporting.py tests/test_promotion_governance.py tests/test_promotion_governance_integration.py
```

The broader M32 slice may emit existing `signal_engine.py` runtime warnings from
low-sample or high-turnover fixtures. Those warnings are expected in the current
test fixtures and are not governance-report failures.
