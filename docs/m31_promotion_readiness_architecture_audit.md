# Milestone 31 Promotion Readiness Architecture Audit

Issue #341 audit scope: define the safest integration plan for M30 statistical readiness diagnostics in the existing StratLake promotion, registry, review, campaign, and milestone-reporting architecture. This document is intentionally documentation-only. It does not implement a new promotion engine.

## Executive Recommendation

Favor Option A: extend the existing `promotion_gates.json` architecture with statistical readiness checks.

The current repository already has a centralized gate evaluator in `src/research/promotion.py`, artifact writers in strategy, portfolio, alpha, and review workflows, registry propagation through `promotion_gate_summary`, and campaign/milestone consumers that read promotion gate outcomes. M31 should add readiness-aware policy to that path rather than creating a parallel promotion decision system.

The main implementation gap is not access to M30 metrics. The M30 fields are already present in `metrics.json` for strategy and portfolio return workflows and available to gate evaluation through the `metrics` source. The gap is that the current promotion schema is binary at the per-gate level (`pass`, `fail`, `missing`) and binary at the evaluation level (`pass`, `fail`), with configurable final labels (`status_on_pass`, `status_on_fail`). It can express `eligible` versus `blocked` today, but it cannot natively distinguish `warn`, `needs_review`, `rejected`, and `blocked` by severity without extending the schema.

## Current Architecture Summary

Promotion gate configuration is a plain JSON/YAML mapping with:

* `status_on_pass`, default `eligible`
* `status_on_fail`, default `blocked`
* `gates`, a non-empty list of metric comparisons

The evaluator supports these sources:

* `metrics`
* `qa_summary`
* `manifest`
* `config`
* `metadata`
* `split_metrics`
* `timeseries`
* `aggregate_metrics`

The evaluator supports scalar and aggregate statistics over source values:

* `value`
* `count`
* `non_null_count`
* `min`
* `max`
* `mean`
* `median`
* `std`
* `range`
* `abs_max`

Supported comparators are `gt`, `gte`, `lt`, `lte`, `eq`, and `ne`. Missing metrics either produce `missing` or pass via `missing_behavior: skip`.

Gate evaluation is centralized in `evaluate_promotion_gates()` and persisted with `write_promotion_gate_artifact()` in `src/research/promotion.py`. The output payload is deterministic, canonicalized, JSON-safe, and written as sorted JSON.

## Current Config Flow

Promotion configs enter the system through several existing paths:

* Strategy config: `configs/strategies.yml` entries may include `promotion_gates`; `src/cli/run_strategy.py` forwards those into `save_experiment()`.
* Strategy CLI/API override: `--promotion-gates` and `promotion_gates_path` load YAML/JSON through `load_promotion_gate_config()`.
* Portfolio config: portfolio definitions may include `promotion_gates`; `src/cli/run_portfolio.py` and `src/execution/portfolio.py` forward them to single-run or walk-forward portfolio artifact writers.
* Portfolio CLI/API override: `--promotion-gates` and `promotion_gates_path` also use `load_promotion_gate_config()`.
* Alpha evaluation config: `src/cli/run_alpha_evaluation.py` passes `promotion_gate_config` into `write_alpha_evaluation_artifacts()`.
* Unified review config: `src/config/review.py` resolves repository/default/CLI review config and supports a `promotion_gates` section.
* Existing examples include `configs/alpha_promotion_gates.yml` and `configs/review_gates_2026_q1.yml`.

No separate statistical readiness policy file exists today.

## Current Artifact Flow

`promotion_gates.json` is emitted when promotion gates are configured by:

* strategy single-run artifacts in `src/research/experiment_tracker.py`
* strategy walk-forward aggregate artifacts in `src/research/experiment_tracker.py`
* portfolio single-run artifacts in `src/portfolio/artifacts.py`
* portfolio walk-forward aggregate artifacts in `src/portfolio/walk_forward.py`
* alpha evaluation artifacts in `src/research/alpha_eval/artifacts.py`
* unified research review artifacts in `src/research/review.py`

Strategy walk-forward split directories do not independently evaluate promotion gates. They write metrics/readiness artifacts, and aggregate promotion can use `split_metrics`.

`metrics_readiness.json` is emitted by:

* strategy single-run artifacts
* strategy walk-forward aggregate artifacts
* strategy walk-forward split artifacts
* portfolio walk-forward split artifacts

Single-run portfolio artifacts currently do not write `metrics_readiness.json`, even though `compute_portfolio_metrics()` includes the M30 statistical fields in `metrics.json`. Portfolio walk-forward aggregate artifacts also do not write a root readiness manifest; split-level readiness manifests are written under `splits/<split_id>/`.

Alpha evaluation currently writes `alpha_metrics.json`, not M30 return-stream `metrics_readiness.json`.

## Fields Available To Promotion Evaluation Today

For strategy single-run and strategy walk-forward aggregate promotion, the evaluator receives:

* `metrics`: the full strategy or aggregate metric dictionary
* `qa_summary`
* `config`
* `split_metrics` when split results are available

The `metrics` source includes M30 fields such as `t_stat`, `p_value`, `conf_int_lower`, `conf_int_upper`, `hit_rate_p_value`, `autocorr_lag1`, `effective_n`, `split_mean_diff`, `split_mean_diff_p`, `rolling_sharpe_mean`, `rolling_sharpe_sd`, and `sharpe_stability_ratio`.

For portfolio single-run promotion, the evaluator receives:

* `metrics`
* `qa_summary`
* `config`

The portfolio `metrics` source also includes the M30 return-stream fields.

For portfolio walk-forward aggregate promotion, the evaluator receives:

* `metrics`: `aggregate_metrics["metric_summary"]`
* `aggregate_metrics`
* `config`
* `split_metrics`

The aggregate `metric_summary` is the mean of split-level M30 fields where available. `aggregate_metrics.metric_statistics.<metric>.<stat>` can be accessed through the `aggregate_metrics` source for mean/median/std/min/max style checks.

For unified research review promotion, the evaluator receives:

* `metrics`: review-level counts such as `entry_count`, `eligible_entry_count`, and `blocked_entry_count`
* `metadata`
* `aggregate_metrics`: best selected and secondary metric values by run type

Review promotion does not currently receive raw candidate run `metrics.json` or `metrics_readiness.json` payloads. It receives registry summaries and review-level aggregates.

`metrics_readiness.json` itself is not currently passed as a promotion source in any workflow. Its checks are derived from fields already available in `metrics`, but the grouped readiness `status`, `checks`, and `summary` are not directly available to the gate evaluator unless a future issue adds a `metrics_readiness` source or inlines readiness summaries into existing sources.

## Gate Schema Capability

The current gate schema supports M30 diagnostics directly through `source: metrics` when those diagnostics are present in the metrics payload. Example future gates can reference:

```yaml
promotion_gates:
  gates:
    - gate_id: return_p_value
      source: metrics
      metric_path: p_value
      comparator: lte
      threshold: 0.05
    - gate_id: minimum_effective_n
      source: metrics
      metric_path: effective_n
      comparator: gte
      threshold: 30
    - gate_id: split_stability_p_value
      source: metrics
      metric_path: split_mean_diff_p
      comparator: gte
      threshold: 0.05
```

The schema does not directly support multi-severity outcomes. Per-gate statuses are only `pass`, `fail`, or `missing`. Overall `evaluation_status` is only `pass` or `fail`. The final `promotion_status` can be configured to labels such as `eligible`, `blocked`, `needs_review`, or `rejected`, but only one status is available for all failures. There is no native way to say one failed gate is a warning, another is manual review, and a third is blocking.

## Registry And Review Propagation

Strategy registry propagation is handled in `src/research/experiment_tracker.py`:

* `_build_registry_entry()` stores `promotion_status`
* `promotion_gate_summary`
* `review_status`
* `review_metadata`

Portfolio registry propagation is handled in `src/research/registry.py` through `register_portfolio_run()` and `_build_portfolio_registry_entry()`:

* `promotion_status`
* `review_status`
* `review_metadata`
* `promotion_gate_summary`

`build_review_metadata()` maps promotion outcomes into review metadata:

* `eligible` -> review status `candidate`
* `blocked` -> review status `rejected`
* anything else or absent -> review status `needs_review`

Allowed explicit review statuses are currently:

* `candidate`
* `promoted`
* `rejected`
* `needs_review`

This means `warn` is not a review status today. `needs_review` is supported in review metadata but not natively produced by the gate evaluator unless configured as `status_on_fail` or provided manually in config review metadata.

Unified review uses registry rows and `resolve_review_status()` to build review entries. It includes `promotion_status`, `passed_gate_count`, and `gate_count` in the leaderboard and can emit a review-level `promotion_gates.json`.

## Campaign And Candidate Review Flow

Research campaigns orchestrate stages in `src/cli/run_research_campaign.py`:

* preflight
* research
* comparison
* candidate selection
* portfolio
* candidate review
* unified review
* optional milestone report

Campaign summaries include:

* stage state counts and checkpoint metadata
* selected run IDs
* key metrics for alpha, strategy, candidate selection, portfolio, and review
* final review promotion status through `_campaign_review_promotion_summary()`

Campaign review promotion propagation is currently limited to reading the unified review `promotion_gates.json` and carrying:

* `evaluation_status`
* `promotion_status`
* `gate_count`
* `passed_gate_count`
* `failed_gate_count`
* `missing_gate_count`

Candidate review in `src/research/candidate_review/review.py` is explainability-oriented. It consumes candidate selection artifacts and portfolio artifacts to write candidate decisions, summaries, contributions, diversification summaries, markdown, and a manifest. It does not evaluate promotion gates or readiness policy. After Issue #344 it preserves naturally available readiness context under `promotion_context`: candidate `promotion_status` counts, candidate `review_status` counts, and the portfolio `promotion_gate_summary` when present. Legacy artifacts without those fields remain valid and emit empty counts plus a `null` portfolio promotion summary.

Milestone reporting in `src/research/reporting/campaign_milestone_report.py` loads campaign `summary.json`, campaign `manifest.json`, review summary, review `promotion_gates.json`, and candidate review summary. It derives milestone decisions from campaign execution state and review promotion artifacts. Its current decision vocabulary includes accepted/rejected/deferred style milestone decisions, with review promotion status handling that recognizes values like `approved`, `blocked`, and `rejected`. Existing examples also mention `review_ready`, `pending`, and `deferred`. This is not yet aligned to the proposed M31 vocabulary.

## Reason Codes Today

Promotion reason strings are generated per gate in `_evaluate_definition()`:

* missing metric: `metric missing`
* skipped missing metric: `metric missing and gate skipped`
* passing comparison: `<actual> <comparator> <threshold>`
* failing comparison: `<actual> not <comparator> <threshold>`

Registry review reasons are generated in `build_review_metadata()` through `_resolve_review_reason()`, usually summarizing gate counts and evaluation status. There are no structured reason codes today, only human-readable reason strings and gate IDs.

## Status And Schema Gaps

Current support for the M31 outcome words:

| Desired outcome | Current support | Gap |
| --- | --- | --- |
| `eligible` | Supported as default `status_on_pass`; maps to review status `candidate` | No gap for pass case |
| `needs_review` | Supported as registry review status; can be used as `status_on_fail` | Not a first-class promotion severity; not distinguishable per gate |
| `warn` | Not a supported review status; can only be used as arbitrary `promotion_status` label | No per-gate or overall warning semantics |
| `rejected` | Supported as review status; can be used as `status_on_fail` | No distinction from `blocked` unless configured globally |
| `blocked` | Supported as default `status_on_fail`; maps to review status `rejected` | Blocking severity is binary with all failures |

Schema gaps to close for M31:

* per-gate severity or outcome is absent
* overall evaluation can only be `pass` or `fail`
* `missing_behavior` only supports `fail` and `skip`
* `warn` is not supported by `_VALID_REVIEW_STATUSES`
* campaign and milestone reporting expect older promotion status names in places
* no structured reason-code field beyond `gate_id` and text `reason`
* `metrics_readiness.json` is advisory and not a promotion source
* single-run portfolio readiness artifacts are missing

## Design Options

### Option A: Extend Existing `promotion_gates.json`

Add statistical readiness checks as normal promotion gates, then extend the existing schema only where necessary for severity. This could be implemented incrementally:

* M31.1: use current schema for hard readiness gates that should block, by referencing M30 fields through `source: metrics`
* M31.2: add optional gate severity such as `severity: warn | review | reject | block`
* M31.3: derive overall promotion status deterministically from the highest-severity non-passing gate
* M31.4: preserve existing behavior when severity is omitted

Pros:

* keeps one promotion artifact and one evaluator
* preserves artifact-first workflows
* preserves CLI/API/notebook/orchestrator reuse
* already works for many M30 metric checks
* preserves backward compatibility with existing gates

Cons:

* requires schema extension for true `warn` and `needs_review`
* requires registry/review/campaign consumers to understand expanded summaries

Recommendation: primary path.

### Option B: Add `promotion_readiness.json` Or `promotion_decision.json`

Create a new derived artifact that combines `promotion_gates.json` and `metrics_readiness.json`.

Pros:

* can model a richer decision separately from the legacy gate artifact
* avoids changing the existing gate payload immediately

Cons:

* creates a second promotion source of truth
* requires every registry/review/campaign/milestone consumer to choose between artifacts
* risks duplicating workflow logic and diverging CLI/API/notebook behavior

Recommendation: do not start here. Consider only later if external reporting needs a stable summarized decision artifact after the single promotion gate model is extended.

### Option C: Add A Separate Statistical Readiness Policy File

Examples: `configs/statistical_readiness_policy.yml`, `configs/promotion_readiness_policy.yml`, or `configs/research_promotion_policy.yml`.

Pros:

* provides reusable policy defaults
* can centralize M31 thresholds

Cons:

* not necessary for the first integration because promotion config already exists
* risks introducing a second policy layer beside `promotion_gates`
* requires new config resolution semantics

Recommendation: defer. If reusable policy templates are needed, prefer documented `promotion_gates` examples or included config fragments before adding a new policy file type.

### Option D: Integrate Only Into Research Campaigns And Candidate Selection

Use readiness-aware decisions only when selecting or reviewing campaign candidates.

Pros:

* narrows initial blast radius
* aligns with campaign-level decision workflows

Cons:

* duplicates promotion logic outside the central evaluator
* leaves strategy/portfolio direct runs inconsistent with campaigns
* undermines the one-execution-system principle

Recommendation: avoid as primary path. Campaigns should consume promotion outcomes; they should not own the readiness policy engine.

## Recommended M31 Implementation Path

1. Extend `src/research/promotion.py` without changing default behavior.

   Add optional schema fields such as `severity` and possibly `on_fail_status`. Existing configs without these fields should produce byte-for-byte compatible summary semantics except for any explicitly versioned additions.

2. Keep M30 checks in `promotion_gates`.

   Reference statistical diagnostics directly through `source: metrics` first. Do not require `metrics_readiness.json` as an evaluator source for M31 unless a later issue specifically needs grouped readiness checks or readiness summary counts.

3. Add deterministic outcome derivation.

   Preserve `eligible` on pass. On non-pass, derive the highest-severity outcome in a deterministic order such as:

   `blocked` > `rejected` > `needs_review` > `warn`

   The exact vocabulary should be settled in implementation, but it should remain explicit, sorted, and JSON-safe.

4. Propagate expanded summaries through existing fields.

   Keep `promotion_gate_summary` as the canonical summary in manifest and registry rows. Add fields such as `warning_gate_count`, `review_gate_count`, `rejected_gate_count`, `blocked_gate_count`, and `decision_reason_codes` only if needed by consumers.

5. Align review metadata status mapping.

   Update `build_review_metadata()` to map promotion statuses deterministically:

   * `eligible` -> `candidate`
   * `warn` -> likely `needs_review`
   * `needs_review` -> `needs_review`
   * `rejected` -> `rejected`
   * `blocked` -> `rejected`

   Avoid adding `warn` as a review status unless the product decision is that review status and promotion status should share one vocabulary.

6. Update campaign and milestone consumers to read the extended summary.

   They should continue reading `promotion_gates.json` and `promotion_gate_summary`, not a new artifact.

7. Fill portfolio readiness artifact parity.

   Add `metrics_readiness.json` for portfolio single-run artifacts and optionally root portfolio walk-forward aggregate artifacts for completeness. This is useful for auditability, even if M31 gates read M30 fields directly from `metrics`.

## Exact Follow-Up Touchpoints

Likely files/functions to modify:

* `src/research/promotion.py`
  * `PromotionGateDefinition`
  * `PromotionGateResult`
  * `PromotionGateEvaluation`
  * `_normalize_definition()`
  * `_normalize_promotion_gate_config()`
  * `_evaluate_definition()`
  * `evaluate_promotion_gates()`
  * `promotion_gate_config_digest()`
* `src/research/registry.py`
  * `_VALID_REVIEW_STATUSES` if review vocabulary changes
  * `build_review_metadata()`
  * `_resolve_review_status()`
  * `_resolve_review_reason()`
* `src/research/experiment_tracker.py`
  * `_build_registry_entry()`
  * `_build_manifest()`
  * `_prepare_run_outputs()` only if adding readiness source payloads
* `src/portfolio/artifacts.py`
  * `write_portfolio_artifacts()`
  * `_build_manifest()`
  * `build_portfolio_registry_metadata()`
* `src/portfolio/walk_forward.py`
  * `run_portfolio_walk_forward()`
  * `_build_manifest()`
  * `_build_aggregate_metrics()`
* `src/research/review.py`
  * `_review_metrics_payload()`
  * `_build_review_manifest()`
  * `_promotion_status()`
  * `_promotion_gate_count()`
* `src/cli/run_research_campaign.py`
  * `_campaign_review_promotion_summary()`
  * `_build_campaign_summary()`
  * `_scenario_matrix_row()`
* `src/research/reporting/campaign_milestone_report.py`
  * `_recommendations()`
  * `_review_decision()`
  * `_review_follow_up_actions()`
  * `_review_result_fragment()`
* `src/config/review.py`
  * `_resolve_promotion_gates()` if schema validation becomes stricter
* docs and config examples:
  * `docs/strategy_performance_metrics.md`
  * `docs/experiment_artifact_logging.md`
  * `docs/portfolio_artifact_logging.md`
  * `docs/strategy_evaluation_workflow.md`
  * `configs/review_gates_2026_q1.yml`
  * possible new example config using M30 readiness fields

## Test Coverage Recommendations

Update or add targeted tests:

* `tests/test_promotion_gates.py`
  * M30 metrics can be gated through `source: metrics`
  * omitted severity preserves current pass/fail behavior
  * severity derives `warn`, `needs_review`, `rejected`, and `blocked`
  * deterministic ordering when multiple severities fail
  * missing metric behavior with severity
* `tests/test_experiment_registry.py`
  * registry maps expanded promotion statuses to review metadata correctly
  * existing `eligible` and `blocked` behavior remains compatible
* `tests/test_experiment_tracker.py`
  * strategy manifest and registry summaries include expanded promotion summaries
* `tests/test_portfolio_artifacts.py` or existing portfolio artifact tests
  * single-run portfolio emits `metrics_readiness.json` if parity is added
  * portfolio promotion gates can reference M30 fields
* `tests/test_portfolio_walk_forward.py`
  * aggregate promotion can gate split-level M30 fields through `split_metrics` and `aggregate_metrics`
* `tests/test_research_review.py`
  * review-level promotion summary preserves expanded status and counts
* `tests/test_cli_run_research_campaign.py`
  * campaign `final_outcomes` carries expanded review promotion status
* `tests/test_campaign_milestone_reporting.py`
  * milestone report maps expanded promotion status to accepted/deferred/rejected decisions
* docs/path hygiene tests
  * no absolute local paths in new examples or artifacts

## Known Risks And Mitigations

Risk: Statistical metrics are present in `metrics.json`, but readiness manifests are not passed to the evaluator.

Mitigation: Use `source: metrics` for first M31 gates. Add a `metrics_readiness` source only if future policy truly needs readiness `checks` or grouped `diagnostics`.

Risk: Existing consumers assume binary `evaluation_status`.

Mitigation: Preserve `evaluation_status: pass|fail` for compatibility and add a separate `promotion_status` or `decision_status` expansion. Alternatively introduce a schema version and keep legacy keys stable.

Risk: `warn` is not a valid review status.

Mitigation: Keep `warn` as a promotion outcome and map it to `review_status: needs_review` unless review status vocabulary is intentionally expanded.

Risk: Milestone reporting uses older status names such as `approved` and `review_ready`.

Mitigation: Update milestone mapping to recognize `eligible`, `warn`, `needs_review`, `rejected`, and `blocked` while preserving old labels for existing artifacts.

Risk: Portfolio readiness artifact parity is incomplete.

Mitigation: Add `metrics_readiness.json` to single-run portfolio artifacts and consider root walk-forward aggregate readiness in a focused follow-up.

Risk: Review-level promotion does not have raw run readiness payloads.

Mitigation: Keep review-level gates focused on registry/review aggregates for M31. If review needs per-run readiness filtering later, add explicit registry summary fields rather than scanning arbitrary run directories in review code.

## Direct Answers To Audit Questions

1. Promotion gate configuration is loaded from run configs and optional YAML/JSON override paths via `load_promotion_gate_config()`, strategy/portfolio config payloads, alpha evaluation config, and review config resolution in `src/config/review.py`.
2. Promotion gate evaluation is performed centrally by `evaluate_promotion_gates()` in `src/research/promotion.py`.
3. `promotion_gates.json` is emitted by strategy single/walk-forward aggregate, portfolio single/walk-forward aggregate, alpha evaluation, and unified research review workflows when gates are configured.
4. `metrics_readiness.json` is emitted by strategy single, strategy walk-forward aggregate, strategy walk-forward splits, and portfolio walk-forward splits. It is not emitted by portfolio single-run artifacts today.
5. M30 fields in `metrics.json` are available through `source: metrics` for strategy and portfolio promotion. `metrics_readiness.json` fields are not currently passed to promotion evaluation.
6. Yes, the current gate schema supports M30 diagnostics directly through `metrics` when those fields are present.
7. The current gate schema supports only per-gate pass/fail/missing and overall pass/fail. Final labels are configurable but not severity-aware.
8. Promotion outcomes propagate through manifest `promotion_gate_summary`, registry `promotion_status`/`promotion_gate_summary`/`review_metadata`, campaign review summaries/final outcomes, and milestone reporting through review `promotion_gates.json`.
9. Promotion reason strings are generated in `_evaluate_definition()`; registry review reasons are generated in `build_review_metadata()`.
10. `eligible` and `blocked` are supported. `needs_review` is supported in review metadata but not as first-class gate severity. `warn` is missing from review status. `rejected` is supported as review status but not distinguishable from other failures without config or schema extension.
11. Yes. M31 can be implemented mostly by extending `promotion_gates.json`.
12. No separate `promotion_decision.json` or `promotion_readiness.json` is necessary for the primary path.
13. No separate statistical readiness policy file is necessary for the primary path.
14. Follow-up implementation issues should cover gate severity/schema extension, registry/review mapping, campaign/milestone propagation, portfolio readiness parity, and tests/docs examples.

## Proposed Follow-Up Issues

Issue #342: Extend promotion gate schema for readiness severity. Completed in this branch.

* Add optional per-gate severity/outcome fields.
* Preserve current binary behavior when omitted.
* Add tests for deterministic status derivation and M30 metric gates.

Issue #343: Propagate expanded promotion outcomes through registry, review, campaign, and milestone metadata. Completed in this branch.

* Map `eligible`, `warn`, `needs_review`, `rejected`, and `blocked`.
* Preserve existing registry schema fields.
* Preserve additive `promotion_gate_summary` fields in campaign final outcomes.
* Align milestone decision mapping to the new vocabulary while preserving older labels.
* Add registry, review metadata, campaign-summary, and milestone tests.

Issue #344: Complete readiness-aware propagation hardening. Completed in this branch.

* Audit remaining registry, review, campaign, candidate-review, manifest, and milestone surfaces.
* Preserve candidate-review promotion context without making candidate review own policy logic.
* Add tests for scenario matrix severity, milestone M31 status mapping, and legacy candidate-review compatibility.

Issue #345: Add canonical readiness-gated promotion examples and validation. Completed in this branch.

* Add `docs/examples/m31_readiness_gated_promotion_case_study.py`.
* Demonstrate synthetic M30 diagnostics gated through `source: metrics`.
* Validate `eligible`, `warn`, `needs_review`, and `blocked` propagation through promotion artifacts, registry/review metadata, campaign summaries, scenario rows, and candidate-review context.
* Add `docs/m31_release_notes.md`.

Issue #346: Close readiness artifact parity and review coverage gaps.

* Add portfolio single-run `metrics_readiness.json`.
* Consider root portfolio walk-forward aggregate readiness.
* Add targeted tests for portfolio readiness and candidate/review propagation.

## Issue #341 Closing Summary

The audit confirms that M31 should not introduce a new promotion engine. StratLake already has a centralized, deterministic promotion gate evaluator and artifact-first propagation path. M30 diagnostics are already available to strategy and portfolio promotion through `metrics.json`. The safest M31 path is to extend the existing promotion gate schema with readiness-aware severity while preserving current pass/fail compatibility, then propagate the expanded summary through the existing manifest, registry, review, campaign, and milestone-reporting fields.
