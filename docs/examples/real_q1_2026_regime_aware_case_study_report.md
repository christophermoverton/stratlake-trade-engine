# Real Q1 2026 Regime-Aware Case Study Report

## Executive Summary

The real Q1 2026 regime-aware case study demonstrates that StratLake's Milestone 24 regime stack can be applied end to end on repository-available market and research data, not just deterministic fixtures. The workflow uses real `features_daily` market data, the Q1 2026 alpha catalog, a real strategy registry run, and a candidate-selection-to-portfolio path to evaluate strategy, alpha, and portfolio behavior under a shared regime taxonomy.

The key finding is that performance was materially regime-dependent. The strategy surface was strongest in a high-volatility, sideways, recovery, normal-stress regime, but weakest in a similar high-volatility sideways regime when the drawdown/recovery state moved underwater. The strategy was also flagged as fragile because most positive contribution came from one dominant regime.

Across alpha runs, `ml_cross_sectional_lgbm_2026_q1` led the largest number of regime slices, while candidate selection advanced `ml_cross_sectional_xgb_2026_q1` and `rank_composite_momentum_2026_q1` into the portfolio path. The portfolio's strongest regime was low-volatility, downtrend, underwater, normal-stress, while its weakest regime was low-volatility, downtrend, recovery, normal-stress.

The interpretation is descriptive, not causal. The case study uses a fixed deterministic market basket from `features_daily`, and sparse or empty regime slices remain visible as evidence limitations.

## 1. Case Study Purpose

This case study is the real-data companion to the canonical deterministic Milestone 24 example. Its purpose is to show that the full regime-aware workflow can run against repository-available Q1 2026 research surfaces:

1. real `features_daily` market data for regime classification
2. a real Q1 2026 strategy run from the strategy registry
3. real Q1 2026 alpha runs from `configs/alphas_2026_q1.yml`
4. a candidate-driven portfolio path extending the earlier real-world candidate-selection case study

The study extends prior repository workflows, especially:

- `real_world_candidate_selection_portfolio_case_study.py`
- `real_world_campaign_case_study.py`
- `regime_aware_case_study.py`

This makes M24.8 more than a standalone example. It connects regime-aware interpretation back to the existing StratLake research process.

## 2. Input Data and Workflow Scope

The workflow uses only repository-available data and does not fetch live external data. The documented input path includes:

- market regime input: `data/curated/features_daily`
- alpha configs: `configs/alphas_2026_q1.yml`
- strategy config: `configs/strategies.yml`
- strategy entry: `mean_reversion_v1_safe_2026_q1`
- portfolio path: real candidate-driven portfolio built from selected Q1 2026 alpha sleeves
- date window: `2026-01-01` through `2026-04-03`, using the repository's exclusive-end convention

The structured summary confirms the market basket used for regime classification included symbols such as `AAPL`, `ABBV`, `ADBE`, `AMD`, `AMZN`, `AVGO`, `BA`, `BAC`, `C`, `CAT`, `COP`, and `COST`.

The case study covers all three research surfaces:

- Strategy: included
- Alpha: included
- Portfolio: included

## 3. Regime Classification Results

The classifier produced 63 timestamp rows for the Q1 2026 window. Of those, 43 rows were fully defined and 20 were undefined, mostly reflecting warmup or insufficient-history periods in the deterministic rolling regime calculations.

The state distribution was:

| Dimension | Distribution |
|---|---|
| Volatility | 14 high-volatility, 14 low-volatility, 15 normal-volatility, 20 undefined |
| Trend | 18 downtrend, 23 sideways, 2 uptrend, 20 undefined |
| Drawdown/recovery | 22 near-peak, 16 recovery, 24 underwater, 1 undefined |
| Stress | 9 dispersion-stress, 34 normal-stress, 20 undefined |

These distributions show that the Q1 2026 case was not dominated by a single state. The classification surface captured variation across volatility, trend, drawdown/recovery, and stress states.

## 4. Strategy Surface Results

The strategy surface used the real `mean_reversion_v1_safe_2026_q1` strategy path. The strategy output contained 63 rows and was evaluated across the M24 conditional dimensions: composite, drawdown/recovery, stress, trend, and volatility.

The strongest strategy regime was:

```text
volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress
```

with a `total_return` of:

```text
0.009944287812721608
```

The weakest strategy regime was:

```text
volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress
```

with a `total_return` of:

```text
-0.0033326061805848406
```

The strategy fragility flag was `true`, with the reason:

```text
most positive contribution comes from one dominant regime
```

This is a meaningful finding. The strategy did not simply perform uniformly across high-volatility sideways regimes. Its behavior changed depending on whether the drawdown/recovery dimension was in recovery or underwater.

The strategy transition analysis detected 45 events. The most adverse transition category for the strategy was `stress_relief`, with a `post_transition_return` metric value of `-0.0014651113147232109`.

### Strategy interpretation

The strategy appears sensitive to the path-dependent drawdown/recovery state. Its strongest result occurred when the market was high-volatility and sideways but recovering. Its weakest result occurred under a similar volatility/trend/stress setup but in an underwater state. This suggests that aggregate strategy performance could hide important differences across recovery versus underwater environments.

The fragility flag reinforces this: positive contribution was concentrated rather than evenly distributed across regimes. That does not invalidate the strategy, but it does make regime-aware review necessary before interpreting aggregate results.

## 5. Alpha Surface Results

The alpha workflow evaluated three Q1 2026 alpha runs:

| Alpha | Mean IC | IC IR | Selected in Portfolio |
|---|---:|---:|---|
| `ml_cross_sectional_xgb_2026_q1` | 0.2546042594447913 | 1.9198107890623903 | yes |
| `ml_cross_sectional_lgbm_2026_q1` | 0.24848289027411252 | 1.589418842850813 | no |
| `rank_composite_momentum_2026_q1` | -0.007925863633708358 | -0.04543185604194747 | yes |

The highest aggregate mean IC and IC IR belonged to `ml_cross_sectional_xgb_2026_q1`. However, the regime comparison found that `ml_cross_sectional_lgbm_2026_q1` led the largest number of regime slices, with `winner_count=7`.

The final interpretation summarizes this tension clearly:

- `ml_cross_sectional_lgbm_2026_q1` led the most alpha regime slices.
- Candidate selection nevertheless advanced `ml_cross_sectional_xgb_2026_q1` and `rank_composite_momentum_2026_q1` into the portfolio path.

This is an important result. Aggregate alpha strength and regime-slice leadership are not identical selection criteria.

### Alpha interpretation

The alpha results show why regime-aware evaluation matters. `ml_cross_sectional_xgb_2026_q1` looked strongest on aggregate mean IC and IC IR, while `ml_cross_sectional_lgbm_2026_q1` showed broader regime leadership across slices. That does not automatically mean LGBM was better, especially because many regime-winner rows include sparse evidence warnings. But it does show that alpha review benefits from seeing both aggregate metrics and conditional/regime-aware comparison.

The candidate-selection path selected XGBoost and rank-composite momentum, not LGBM. This provides a useful example of how a portfolio workflow may prioritize criteria beyond simple regime-winner count.

## 6. Portfolio Surface Results

The portfolio surface was built from the selected alpha candidates. The candidate-selection summary shows two selected alphas and one rejected alpha, with allocation weights summing to 1.0. The selected alpha names were:

- `ml_cross_sectional_xgb_2026_q1`
- `rank_composite_momentum_2026_q1`

The portfolio's strongest regime was:

```text
volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress
```

with `total_return`:

```text
0.015174656275256826
```

The weakest portfolio regime was:

```text
volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress
```

with `total_return`:

```text
-0.003646818401148222
```

The portfolio fragility flag was `false`, and the fragility reason was:

```text
no deterministic fragility heuristic triggered
```

The portfolio transition analysis detected 45 events. The most adverse portfolio transition category was `trend_breakdown`, with a `post_transition_return` metric value of `0.001337844704837221`.

### Portfolio interpretation

The portfolio showed a different regime profile than the strategy. Its best regime occurred in low-volatility, downtrend, underwater, normal-stress conditions. Its weakest regime was also low-volatility and downtrend, but in a recovery state. That makes the portfolio's behavior distinct from the strategy surface.

The portfolio's fragility flag was false, suggesting that the candidate-driven allocation did not trigger the same deterministic concentration warning observed in the strategy surface. However, this should still be read cautiously because sparse and empty regime slices remain evidence limits.

## 7. Transition Findings

The case study applies transition-aware analysis across strategy, alpha, and portfolio surfaces.

Key transition observations:

- Strategy: 45 transition events, most adverse category `stress_relief`, metric value `-0.0014651113147232109`
- Portfolio: 45 transition events, most adverse category `trend_breakdown`, metric value `0.001337844704837221`
- LGBM alpha example: transition attribution highlighted `generic_transition`, `recovery_onset`, `stress_relief`, `trend_breakdown`, `volatility_downshift`, and `volatility_upshift` categories, with sparse warnings on some categories

These transition outputs support a more detailed analysis than static regime buckets alone. They allow the researcher to ask not only where performance differed, but also what happened around changes in regime state.

The report should avoid treating these as causal effects. The transition layer identifies performance around deterministic regime changes; it does not prove that the regime transition caused the observed performance.

## 8. Artifact and Review Outputs

The case study generates a full artifact tree under:

```text
docs/examples/output/real_q1_2026_regime_aware_case_study/
```

The documented output tree includes:

```text
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

The docs explain that the `regime_bundle` contains M24.1/M24.2 labels and manifests, the strategy and portfolio bundles contain conditional metrics, transitions, attribution, and reports, and the alpha bundles contain per-alpha regime-aware outputs.

The structured summary confirms that generated artifacts include:

- regime labels and summaries
- conditional metrics
- transition events and windows
- transition summaries and manifests
- attribution summaries and tables
- comparison tables
- notebook review markdown snapshots

The notebook review loop is also documented. Bundles can be loaded with `load_regime_review_bundle`, inspected with `inspect_regime_artifacts`, sliced with `slice_conditional_metrics`, and rendered through markdown helpers such as `render_comparison_summary_markdown`.

## 9. Main Findings

### Finding 1: Strategy performance was regime-fragile

The strategy had a positive best regime and a negative weakest regime, but its positive contribution was concentrated enough to trigger the deterministic fragility flag. This suggests aggregate strategy performance should not be interpreted without regime context.

### Finding 2: Alpha ranking depends on the lens

`ml_cross_sectional_xgb_2026_q1` had the highest aggregate mean IC and IC IR, while `ml_cross_sectional_lgbm_2026_q1` led the largest number of regime slices. This shows how aggregate alpha evaluation and regime-aware comparison can produce different but complementary research signals.

### Finding 3: Portfolio behavior did not simply mirror alpha leadership

The portfolio was built from selected XGB and rank-composite momentum alphas, not from the LGBM alpha that led the most regime slices. The portfolio's strongest regime was low-volatility/downtrend/underwater/normal-stress, and its fragility flag was false.

### Finding 4: Transition analysis adds an additional risk surface

Strategy and portfolio transition summaries identified different most-adverse transition categories. The strategy's most adverse category was `stress_relief`, while the portfolio's was `trend_breakdown`. This suggests that static regime buckets and transition windows answer related but distinct questions.

## 10. Limitations and Caution Boundaries

The case study explicitly avoids causal claims. It describes observed performance across deterministic regime slices and transition windows. It does not prove that the regimes caused those outcomes.

Important limitations:

1. The market regime surface is built from a deterministic fixed basket of repository symbols from `features_daily`, not a bespoke benchmark index series.
2. Sparse and empty regime slices remain visible and should limit confidence in regime-specific conclusions.
3. The strategy surface is a direct real-data strategy run, while the alpha and portfolio surfaces extend the prior candidate-selection workflow.
4. Some transition categories have sparse warnings, so those transition-level summaries should be treated as directional review aids, not statistical proof.

## Conclusion

The real Q1 2026 regime-aware case study demonstrates that StratLake's M24 regime system works as an integrated research interpretation layer. It classifies market conditions, aligns those labels to strategy/alpha/portfolio surfaces, computes conditional metrics, analyzes transitions, produces attribution and comparison reports, and exposes the outputs through notebook-friendly review helpers.

The results show why this layer matters. Strategy performance was fragile and concentrated in one dominant regime. Alpha conclusions differed depending on whether the lens was aggregate IC/IC IR or regime-slice leadership. Portfolio behavior showed its own regime profile and did not simply mirror the alpha comparison winner. Transition analysis added another layer by identifying different adverse transition categories for strategy and portfolio surfaces.

This is the central value of M24: it moves StratLake from asking only whether a research object worked to asking where it worked, where it struggled, how stable the evidence was, and what changed around regime transitions.
