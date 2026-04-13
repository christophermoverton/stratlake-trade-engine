# Real-World Campaign Case Study (Milestone 16)

## Objective

Run the Milestone 16 campaign orchestration flow end to end on repository
`features_daily` data while comparing multiple real alpha models, selecting
candidates, constructing a candidate-driven portfolio, and reviewing the final
outputs from one campaign entry point.

This case study demonstrates:

1. alpha evaluation on real `features_daily` data
2. alpha comparison through the campaign comparison stage
3. candidate selection driven by campaign-produced alpha artifacts
4. candidate-driven portfolio construction
5. candidate review and unified research review
6. one stitched case-study summary artifact plus native campaign artifacts
7. auto-generated Milestone 18 milestone-report and decision-log artifacts
8. optional second-pass checkpoint reuse across the same campaign artifact root

## Execute

```powershell
python docs/examples/real_world_campaign_case_study.py
```

To demonstrate checkpoint creation and stage reuse on an immediate rerun:

```powershell
python docs/examples/real_world_campaign_case_study.py --checkpoint-demo
```

## Campaign Config

Checked-in campaign config:

```text
docs/examples/data/milestone_16_campaign_configs/real_world_campaign.yml
```

The config targets four real Q1 2026 alpha definitions from
`configs/alphas_2026_q1.yml`:

- `ml_cross_sectional_xgb_2026_q1`
- `ml_cross_sectional_lgbm_2026_q1`
- `ml_cross_sectional_elastic_net_2026_q1`
- `rank_composite_momentum_2026_q1`

Notable choices:

- the new Elastic Net alpha is included alongside XGBoost and LightGBM
- candidate selection intentionally runs across the full campaign alpha universe
  instead of forcing a single `alpha_name` filter
- portfolio construction chains directly from candidate selection outputs
- review runs on `alpha_evaluation` and `portfolio` artifacts only

## Workflow Surfaces Used

- `src.cli.run_research_campaign`
- `src.cli.run_alpha`
- `src.cli.compare_alpha`
- `src.cli.run_candidate_selection`
- `src.cli.run_portfolio`
- `src.cli.review_candidate_selection`
- `src.cli.compare_research`

## Output Location

The example writes all artifacts under:

```text
docs/examples/output/real_world_campaign_case_study/
```

Primary stitched summary:

```text
docs/examples/output/real_world_campaign_case_study/summary.json
```

Native campaign summary:

```text
docs/examples/output/real_world_campaign_case_study/artifacts/research_campaigns/<campaign_run_id>/summary.json
```

Auto-generated milestone pack:

```text
docs/examples/output/real_world_campaign_case_study/artifacts/research_campaigns/<campaign_run_id>/milestone_report/
```

## Outputs To Inspect

- campaign artifacts:
  - `campaign_config.json`
  - `preflight_summary.json`
  - `manifest.json`
  - `summary.json`
- milestone reporting:
  - `milestone_report/summary.json`
  - `milestone_report/decision_log.json`
  - `milestone_report/manifest.json`
  - `milestone_report/report.md`
- alpha comparison:
  - `artifacts/campaign_comparisons/alpha/leaderboard.csv`
  - `artifacts/campaign_comparisons/alpha/summary.json`
- candidate selection:
  - `selected_candidates.csv`
  - `rejected_candidates.csv`
  - `selection_summary.json`
  - `manifest.json`
- candidate-driven portfolio:
  - `portfolio_returns.csv`
  - `metrics.json`
  - `manifest.json`
- review:
  - `artifacts/reviews/candidate_review/candidate_review_summary.json`
  - `artifacts/reviews/candidate_review/candidate_review_report.md`
  - `artifacts/reviews/research_review/leaderboard.csv`
  - `artifacts/reviews/research_review/review_summary.json`

## Observed Results (Validated 2026-04-08)

### Campaign

- campaign run id: `research_campaign_c7361ba400ad`
- status: `completed`
- preflight: `passed`
- stage statuses:
  - `preflight`, `research`, `comparison`, `candidate_selection`, `portfolio`,
    `candidate_review`, and `review` all completed

### Alpha Runs

- `ml_cross_sectional_xgb_2026_q1`
  - run id: `ml_cross_sectional_xgb_2026_q1_alpha_eval_366f61a76d97`
  - mean_ic: `0.254604`
  - ic_ir: `1.919811`
  - sleeve sharpe: `12.465814`
  - sleeve total return: `2.87%`
- `ml_cross_sectional_lgbm_2026_q1`
  - run id: `ml_cross_sectional_lgbm_2026_q1_alpha_eval_b666eb92f4bd`
  - mean_ic: `0.248483`
  - ic_ir: `1.589419`
  - sleeve sharpe: `10.311079`
  - sleeve total return: `2.40%`
- `ml_cross_sectional_elastic_net_2026_q1`
  - run id: `ml_cross_sectional_elastic_net_2026_q1_alpha_eval_813a3a37e1a0`
  - mean_ic: `0.120844`
  - ic_ir: `0.827244`
  - sleeve sharpe: `2.453439`
  - sleeve total return: `0.71%`
- `rank_composite_momentum_2026_q1`
  - run id: `rank_composite_momentum_2026_q1_alpha_eval_81ef86e27306`
  - mean_ic: `-0.007926`
  - ic_ir: `-0.045432`
  - sleeve sharpe: `-2.828352`
  - sleeve total return: `-1.12%`

### Alpha Comparison

- comparison id: `registry_alpha_combined_af4ec0488c60`
- view: `combined`
- ranking metric: `ic_ir + sharpe_ratio`
- candidates compared: `4`

### Candidate Selection

- run id: `candidate_selection_e35992810577`
- candidate universe: `4`
- eligible: `4`
- rejected: `1`
- pruned by redundancy: `1`
- selected: `3`
- redundancy event:
  - rejected candidate: `ml_cross_sectional_lgbm_2026_q1_alpha_eval_b666eb92f4bd`
  - reason: `correlation_above_threshold`
  - observed correlation: `0.759760`
  - threshold: `0.75`

### Governed Allocation

- allocation method: `equal_weight`
- selected candidates: `3`
- final weights:
  - `ml_cross_sectional_xgb_2026_q1_alpha_eval_366f61a76d97`: `0.333333333334`
  - `ml_cross_sectional_elastic_net_2026_q1_alpha_eval_813a3a37e1a0`: `0.333333333333`
  - `rank_composite_momentum_2026_q1_alpha_eval_81ef86e27306`: `0.333333333333`
- weight sum: `1.00`

### Candidate-Driven Portfolio

- portfolio name: `real_world_m16_campaign_case_study`
- run id: `real_world_m16_campaign_case_study_portfolio_cc99ba561315`
- component count: `3`
- total return: `0.81%`
- sharpe ratio: `4.0672`
- max drawdown: `0.46%`
- realized volatility: `2.09%`

### Review Outputs

- candidate review dir:
  - `docs/examples/output/real_world_campaign_case_study/artifacts/reviews/candidate_review/`
- research review dir:
  - `docs/examples/output/real_world_campaign_case_study/artifacts/reviews/research_review/`
- research review id:
  - `registry_review_63c30be06cb7`
- review rows:
  - `4` alpha rows
  - `1` portfolio row

## Milestone 18 Reporting

Milestone reporting is enabled by default for campaign runs, so the real-world
campaign flow can produce a milestone pack without a second manual review
assembly step.

From an existing campaign directory, the equivalent follow-on CLI is:

```powershell
python -m src.cli.generate_milestone_report `
  --campaign-artifact-path docs/examples/output/real_world_campaign_case_study/artifacts/research_campaigns/research_campaign_c7361ba400ad `
  --milestone-name "Milestone 18" `
  --title "Milestone 18 Readiness Report"
```

### Artifact Locations

The milestone pack lives under the campaign artifact root:

```text
docs/examples/output/real_world_campaign_case_study/artifacts/research_campaigns/research_campaign_c7361ba400ad/milestone_report/
```

That location matters because the milestone files use relative links back into
the campaign, candidate-review, and research-review artifacts instead of
duplicating those payloads.

### How To Read The Pack

Read the files in this order:

1. `milestone_report/summary.json`
   Use this as the main machine-readable entrypoint. It gives the milestone
   id, milestone status, decision counts, key findings, recommendations, and
   relative paths to upstream evidence.
2. `milestone_report/decision_log.json`
   Use this for the auditable decision trail. It records each decision's
   status, rationale, follow-up actions, and source artifacts.
3. `milestone_report/report.md`
   Use this as the operator-facing milestone brief for review meetings or
   promotion handoff discussions.
4. `milestone_report/manifest.json`
   Use this when automation needs a deterministic inventory of the milestone
   pack itself.

### Practical Milestone Review Workflow

One practical review pass for this case study is:

1. open the stitched campaign `summary.json`
2. confirm the concrete `campaign_run_id`
3. jump into `milestone_report/summary.json`
4. scan `decision_counts_by_status`, `key_findings`, and `recommendations`
5. open `decision_log.json` for the exact decisions that need discussion
6. follow `related_artifacts` or per-decision `source_artifacts` into
   `artifacts/reviews/research_review/`,
   `artifacts/reviews/candidate_review/`, or the campaign root
7. use `report.md` as the final brief for the milestone review meeting

### What The Outputs Mean In Practice

For this campaign, the milestone pack should summarize:

* the campaign completed successfully on real `features_daily` Q1 2026 data
* four alpha evaluation runs fed one combined comparison leaderboard
* governed candidate selection retained three candidates after one redundancy
  rejection
* the candidate-driven portfolio and research review outputs are the main
  downstream evidence surfaces for milestone discussion

If promotion-gate outputs are not present in the linked research review
artifacts, the milestone decision log may defer the review-promotion decision
and surface follow-up actions instead of marking promotion as fully approved.

### Observed Milestone Results (Validated 2026-04-13)

- milestone id: `milestone_report_a2443a03ccea`
- milestone name: `Milestone 18`
- milestone status: `final`
- decision count: `2`
- decision counts by status:
  - `accepted`: `1`
  - `deferred`: `1`
- accepted decision:
  - `campaign_execution`
- deferred decision:
  - `review_promotion_outcome`
- deferred follow-up:
  - persist promotion-gate outputs for reviewed campaigns so milestone
    decisions remain auditable

In other words, the campaign itself is review-ready and fully auditable, but
promotion remains intentionally unresolved because the committed research
review pack does not include promotion-gate artifacts.

## Notes

- The campaign config intentionally leaves `candidate_selection.alpha_name` unset so candidate selection can operate across the full campaign alpha universe.
- The new Elastic Net alpha is meaningfully weaker than XGBoost and LightGBM on Q1 2026 `features_daily`, but it still adds a distinct linear-shrinkage baseline to the leaderboard and campaign review.
- In this run, the governed portfolio retained the weaker rank-composite sleeve because only the LightGBM sleeve breached the configured redundancy threshold.
- Milestone reporting is most useful after opening the campaign `summary.json`
  first; the milestone pack is a review layer on top of those stitched campaign
  outputs, not a replacement for them.
