# Milestone Report for research_campaign_1b8cdf9211cf

## Milestone Summary
- Milestone: `research_campaign_1b8cdf9211cf`
- Milestone ID: `milestone_report_39d8310e7734`
- Status: `final`
- Reporting Window: 2026-01-01 to 2026-04-03
- Summary: Campaign `research_campaign_1b8cdf9211cf` finished with status `completed`. 7 of 7 tracked stages completed or were reused. Review outcome: Review ranked 5 entries.

## Campaign Scope
- alpha:ml_cross_sectional_elastic_net_2026_q1
- alpha:ml_cross_sectional_lgbm_2026_q1
- alpha:ml_cross_sectional_xgb_2026_q1
- alpha:rank_composite_momentum_2026_q1
- portfolio:real_world_m16_campaign_case_study

## Selections
- Alpha Runs: ml_cross_sectional_elastic_net_2026_q1_alpha_eval_813a3a37e1a0, ml_cross_sectional_lgbm_2026_q1_alpha_eval_b666eb92f4bd, ml_cross_sectional_xgb_2026_q1_alpha_eval_366f61a76d97, rank_composite_momentum_2026_q1_alpha_eval_81ef86e27306
- Candidate Selection Run: candidate_selection_e35992810577
- Portfolio Run: real_world_m16_campaign_case_study_portfolio_cc99ba561315
- Review: registry_review_63c30be06cb7

## Key Findings
- Research scope resolved 4 alpha run(s), 0 strategy run(s), candidate_selection=candidate_selection_e35992810577, portfolio=real_world_m16_campaign_case_study_portfolio_cc99ba561315.
- Stage outcomes: completed=7, failed=0, partial=0, pending=0, reused=0, skipped=0.
- Portfolio metrics: Sharpe=4.06716156929312, Total Return=0.008101759628482474, Max Drawdown=0.004632462958028327.
- Candidate review evaluated 0 candidate(s) with 0 selected and 0 rejected.
- Review outcome: Review ranked 5 entries.

## Key Metrics
- Alpha Runs: alpha_name=ml_cross_sectional_elastic_net_2026_q1; ic_ir=0.8272437066429204; mean_ic=0.12084441007118236; n_periods=19; run_id=ml_cross_sectional_elastic_net_2026_q1_alpha_eval_813a3a37e1a0; sleeve_sharpe_ratio=NA; sleeve_total_return=NA; alpha_name=ml_cross_sectional_lgbm_2026_q1; ic_ir=1.589418842850813; mean_ic=0.24848289027411252; n_periods=19; run_id=ml_cross_sectional_lgbm_2026_q1_alpha_eval_b666eb92f4bd; sleeve_sharpe_ratio=NA; sleeve_total_return=NA; alpha_name=ml_cross_sectional_xgb_2026_q1; ic_ir=1.9198107890623903; mean_ic=0.2546042594447913; n_periods=19; run_id=ml_cross_sectional_xgb_2026_q1_alpha_eval_366f61a76d97; sleeve_sharpe_ratio=NA; sleeve_total_return=NA; alpha_name=rank_composite_momentum_2026_q1; ic_ir=-0.04543185604194747; mean_ic=-0.007925863633708358; n_periods=19; run_id=rank_composite_momentum_2026_q1_alpha_eval_81ef86e27306; sleeve_sharpe_ratio=NA; sleeve_total_return=NA
- Candidate Selection: eligible_count=4; primary_metric=ic_ir; pruned_by_redundancy=1; rejected_count=1; run_id=candidate_selection_e35992810577; selected_count=3; universe_count=4
- Portfolio: component_count=3; max_drawdown=0.004632462958028327; portfolio_name=real_world_m16_campaign_case_study; run_id=real_world_m16_campaign_case_study_portfolio_cc99ba561315; sharpe_ratio=4.06716156929312; total_return=0.008101759628482474
- Review: counts_by_run_type={'alpha_evaluation': 4, 'portfolio': 1}; entry_count=5; review_id=registry_review_63c30be06cb7

## Gate Outcomes
- Stage State Counts: completed=7, failed=0, partial=0, pending=0, reused=0, skipped=0
- Campaign execution outcome: status=accepted; summary=Campaign `research_campaign_1b8cdf9211cf` recorded 7 tracked stage outcomes with overall status `completed`.
- Review and promotion outcome: status=deferred; summary=Review `registry_review_63c30be06cb7` covered 5 ranked entries with counts_by_run_type=alpha_evaluation:4,portfolio:1

## Risks
- Decision `review_promotion_outcome` is `deferred` and still needs resolution.

## Next Steps
- No blocking follow-up actions were detected in the completed campaign artifacts.
- No additional campaign execution follow-up actions were required.
- Persist promotion-gate outputs for reviewed campaigns so milestone decisions remain auditable.

## Open Questions
- Should promotion gates be persisted for reviewed campaigns that stop at review readiness?

## Decision Snapshot
### 1. Campaign execution outcome
- Decision ID: `campaign_execution`
- Status: `accepted`
- Summary: Campaign `research_campaign_1b8cdf9211cf` recorded 7 tracked stage outcomes with overall status `completed`.
- Follow-Up Actions:
  - No additional campaign execution follow-up actions were required.
### 2. Review and promotion outcome
- Decision ID: `review_promotion_outcome`
- Status: `deferred`
- Summary: Review `registry_review_63c30be06cb7` covered 5 ranked entries with counts_by_run_type=alpha_evaluation:4,portfolio:1
- Follow-Up Actions:
  - Persist promotion-gate outputs for reviewed campaigns so milestone decisions remain auditable.

## Related Artifacts
- alpha_comparison_summary: ../../../campaign_comparisons/alpha/leaderboard.json
- campaign_checkpoint: ../checkpoint.json
- campaign_manifest: ../manifest.json
- campaign_manifest_summary_path: ../summary.json
- campaign_summary: ../summary.json
- candidate_review_manifest: ../../../reviews/candidate_review/manifest.json
- candidate_review_summary: ../../../reviews/candidate_review/candidate_review_summary.json
- candidate_selection_manifest: ../../../candidate_selection/candidate_selection_e35992810577/manifest.json
- candidate_selection_summary: ../../../candidate_selection/candidate_selection_e35992810577/selection_summary.json
- preflight_summary: ../preflight_summary.json
- review_manifest: ../../../reviews/research_review/manifest.json
- review_summary: ../../../reviews/research_review/review_summary.json
