# Real Q1 2026 Regime-Aware Case Study

## Objective

This case study is the real-data companion to the canonical deterministic
Milestone 24 example in
[`regime_aware_case_study.py`](regime_aware_case_study.py). It applies the full
M24 regime-aware stack to repository-available Q1 2026 research surfaces:

1. real `features_daily` market data for regime classification
2. a real Q1 2026 strategy run from the strategy registry
3. real Q1 2026 alpha runs from `configs/alphas_2026_q1.yml`
4. a real candidate-driven portfolio path extending the earlier real-world
   candidate-selection case study

## Relationship To Prior Work

This example extends:

- [real_world_candidate_selection_portfolio_case_study.py](real_world_candidate_selection_portfolio_case_study.py)
- [real_world_campaign_case_study.py](real_world_campaign_case_study.py)
- [regime_aware_case_study.py](regime_aware_case_study.py)

The workflow reuses the same Q1 2026 alpha catalog, the same real
`features_daily` input dataset, and the same candidate-selection-to-portfolio
path that earlier real-world examples already validated.

## Input Path Used

The case study uses repository-available data only:

- market regime input: `data/curated/features_daily`
- alpha configs: `configs/alphas_2026_q1.yml`
- strategy config: `configs/strategies.yml`
- strategy entry: `mean_reversion_v1_safe_2026_q1`
- portfolio path: real candidate-driven portfolio built from selected Q1 2026
  alpha sleeves
- date window: `2026-01-01` through `2026-04-03` with the repo's standard
  exclusive-end convention

No live external data is fetched.

## Surface Coverage

- Strategy: included
- Alpha: included
- Portfolio: included

The strategy surface is a real strategy registry run. The alpha and portfolio
surfaces extend the real-world candidate-selection workflow so the example shows
comparison, selection, and downstream portfolio behavior under one regime
taxonomy.

## Execute

```powershell
python docs/examples/real_q1_2026_regime_aware_case_study.py
```

To force the canonical committed output root:

```powershell
python docs/examples/real_q1_2026_regime_aware_case_study.py --output-root docs/examples/output/real_q1_2026_regime_aware_case_study
```

## Output Root

```text
docs/examples/output/real_q1_2026_regime_aware_case_study/
```

Primary stitched summary:

```text
docs/examples/output/real_q1_2026_regime_aware_case_study/summary.json
```

Interpretation notes:

```text
docs/examples/output/real_q1_2026_regime_aware_case_study/final_interpretation.md
```

## Expected Output Tree

```text
docs/examples/output/real_q1_2026_regime_aware_case_study/
  summary.json
  final_interpretation.md
  regime_bundle/
  strategy_bundle/
  portfolio_bundle/
  alpha_ml_cross_sectional_xgb_2026_q1_bundle/
  alpha_ml_cross_sectional_lgbm_2026_q1_bundle/
  alpha_rank_composite_momentum_2026_q1_bundle/
  notebook_review/
  native_artifacts/
```

Key bundle contents:

- `regime_bundle/`
  M24.1 and M24.2 regime labels plus summary and manifest
- `strategy_bundle/`
  conditional metrics, transitions, attribution, and report for the real Q1
  2026 strategy surface
- `portfolio_bundle/`
  conditional metrics, transitions, attribution, and report for the real
  candidate-driven portfolio surface
- `alpha_*_bundle/`
  one bundle per real Q1 2026 alpha run
- `alpha_ml_cross_sectional_xgb_2026_q1_bundle/regime_comparison_table.csv`
  the shared M24.5 alpha comparison surface across the Q1 2026 alpha runs
- `notebook_review/`
  rendered M24.6 review markdown snapshots for strategy, portfolio, alpha, and
  comparison

## Workflow Summary

The case study follows this high-level sequence:

1. load a deterministic market basket from real Q1 2026 `features_daily`
   partitions
2. classify and persist market regimes with M24.1 and M24.2
3. run `mean_reversion_v1_safe_2026_q1` on real Q1 2026 data and aggregate a
   timestamp-level strategy-return surface
4. run the Q1 2026 alpha suite from `configs/alphas_2026_q1.yml`
5. compare the alpha runs, perform candidate selection, and build the real
   candidate-driven portfolio
6. align the regime labels to strategy, alpha, and portfolio surfaces
7. compute M24.3 conditional metrics for each surface
8. compute M24.4 transition analysis for each surface
9. compute M24.5 attribution and alpha comparison outputs
10. load the persisted bundles through M24.6 notebook helpers and render review
    markdown snapshots

## Notebook Review Loop

Every bundle is loadable through the public notebook helpers:

```python
from src.research.regimes import (
    inspect_regime_artifacts,
    load_regime_review_bundle,
    render_comparison_summary_markdown,
    slice_conditional_metrics,
)

strategy_bundle = load_regime_review_bundle(
    "docs/examples/output/real_q1_2026_regime_aware_case_study/strategy_bundle"
)

portfolio_bundle = load_regime_review_bundle(
    "docs/examples/output/real_q1_2026_regime_aware_case_study/portfolio_bundle"
)

alpha_bundle = load_regime_review_bundle(
    "docs/examples/output/real_q1_2026_regime_aware_case_study/alpha_ml_cross_sectional_xgb_2026_q1_bundle"
)

inspect_regime_artifacts(strategy_bundle)
slice_conditional_metrics(portfolio_bundle, dimension="stress")
render_comparison_summary_markdown(alpha_bundle)
```

## Interpretation Highlights

Read the results in this order:

1. `summary.json`
2. `strategy_bundle/regime_attribution_report.md`
3. `portfolio_bundle/regime_attribution_report.md`
4. `alpha_ml_cross_sectional_xgb_2026_q1_bundle/regime_attribution_report.md`
5. `final_interpretation.md`
6. the review snapshots under `notebook_review/`

The case study is designed to answer:

- which regimes were strongest and weakest for the real strategy surface
- how the compared Q1 2026 alpha runs ranked across regime slices
- whether the candidate-driven portfolio retained or diluted the alpha-level
  regime profile
- which transition categories were most adverse on the strategy and portfolio
  surfaces

## Caveats And Limitations

- The market regime surface is built from a deterministic fixed basket of real
  repository symbols rather than a bespoke benchmark index series.
- The strategy surface is a single real strategy run, while the alpha and
  portfolio surfaces come from the candidate-selection workflow.
- Interpretation remains descriptive and evidence-based. The example does not
  claim that regimes caused any observed alpha, strategy, or portfolio result.
- Sparse and empty slices should be treated as evidence limits.

## Related Docs

- [regime_aware_case_study.md](regime_aware_case_study.md)
- [real_world_candidate_selection_portfolio_case_study.md](real_world_candidate_selection_portfolio_case_study.md)
- [real_world_campaign_case_study.md](real_world_campaign_case_study.md)
- [regime_notebook_review_examples.md](regime_notebook_review_examples.md)
