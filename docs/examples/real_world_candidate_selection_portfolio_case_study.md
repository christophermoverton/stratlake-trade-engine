# Real-World Candidate Selection And Portfolio Case Study (Milestone 15)

## Objective

Run the full governed Milestone 15 path on real `features_daily` data using existing
built-in models:

1. run multiple built-in alpha models end to end
2. compare evaluated alpha sleeves
3. run governed candidate selection (eligibility + redundancy + allocation)
4. build candidate-driven portfolio from selected sleeves
5. generate candidate review/explainability artifacts

This case study uses existing production CLI surfaces and artifact contracts only.

## Execute

```powershell
python docs/examples/real_world_candidate_selection_portfolio_case_study.py
```

## Workflow Surfaces Used

- `src.cli.run_alpha`
- `src.cli.compare_alpha`
- `src.cli.run_candidate_selection`
- `src.cli.run_portfolio`

## Output Location

The script writes all artifacts under:

```text
docs/examples/output/real_world_candidate_selection_portfolio_case_study/
```

Primary stitched summary:

```text
docs/examples/output/real_world_candidate_selection_portfolio_case_study/summary.json
```

## Observed Results (Validated 2026-04-07)

### Alpha Runs

- `ml_cross_sectional_xgb_2026_q1`
  - run id: `ml_cross_sectional_xgb_2026_q1_alpha_eval_5bf43247572c`
  - mean_ic: `0.254604`
  - ic_ir: `1.919811`
  - sleeve sharpe: `12.465814`
  - sleeve total return: `2.87%`
- `ml_cross_sectional_lgbm_2026_q1`
  - run id: `ml_cross_sectional_lgbm_2026_q1_alpha_eval_359ae89d67b3`
  - mean_ic: `0.248483`
  - ic_ir: `1.589419`
  - sleeve sharpe: `10.311079`
  - sleeve total return: `2.40%`
- `rank_composite_momentum_2026_q1`
  - run id: `rank_composite_momentum_2026_q1_alpha_eval_ea1286623001`
  - mean_ic: `-0.007926`
  - ic_ir: `-0.045432`
  - sleeve sharpe: `-2.828352`
  - sleeve total return: `-1.12%`

### Alpha Comparison

- comparison id: `registry_alpha_combined_630ae8a1664f`
- view: `combined`
- ranking metric: `ic_ir + sharpe_ratio`
- candidates compared: `3`

### Candidate Selection

- run id: `candidate_selection_9361770baaf5`
- candidate universe: `3`
- eligible: `3`
- rejected: `1`
- pruned by redundancy: `1`
- selected: `2`
- redundancy event:
  - rejected candidate: `ml_cross_sectional_lgbm_2026_q1_alpha_eval_359ae89d67b3`
  - reason: `correlation_above_threshold`
  - observed correlation: `0.759760`
  - threshold: `0.75`

### Governed Allocation

- allocation method: `equal_weight`
- selected candidates: `2`
- final weights:
  - `ml_cross_sectional_xgb_2026_q1_alpha_eval_5bf43247572c`: `0.50`
  - `rank_composite_momentum_2026_q1_alpha_eval_ea1286623001`: `0.50`
- weight sum: `1.00`

### Candidate-Driven Portfolio

- portfolio name: `real_world_m15_candidate_driven`
- run id: `real_world_m15_candidate_driven_portfolio_ce632b3293ed`
- component count: `2`
- total return: `0.86%`
- sharpe ratio: `4.1368`
- max drawdown: `0.34%`
- realized volatility: `2.17%`

### Candidate Review Outputs

- review dir:
  - `docs/examples/output/real_world_candidate_selection_portfolio_case_study/artifacts/reviews/candidate_selection_9361770baaf5/`
- key outputs:
  - `candidate_decisions.csv`
  - `candidate_summary.csv`
  - `candidate_contributions.csv`
  - `candidate_review_summary.json`
  - `candidate_review_report.md`

## Notes

- This case study intentionally uses real built-in model configs from `configs/alphas_2026_q1.yml` and real `features_daily` data.
- Selection thresholds are permissive (`min_* = -1`) so governance behavior is primarily driven by deterministic ranking and redundancy filtering.
- The workflow remains deterministic and artifact-driven end to end under fixed inputs.
