# Alpha Regime Attribution Report

## Executive Summary
- Best mean_ic came from `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` at 0.3045.
- Weakest regime was `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` at 0.1591.
- Positive contribution is led by `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` with share 0.6568.
- Fragility heuristic: most positive contribution comes from one dominant regime.

## Regime Attribution Highlights
- Primary metric: `mean_ic` (desc).
- Best regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` (0.3045).
- Worst regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` (0.1591).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | mean_ic | mean_rank_ic | ic_std | rank_ic_std | ic_ir | rank_ic_ir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | alpha | 5 | sufficient | mean_ic | 0.3045 | desc | 0.6568 | 0.6568 | positive | 1.0000 | true | false | NA | NA | 0.3045 | 0.2621 | 0.1595 | 0.0882 | 1.9086 | 2.9720 |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 5 | sufficient | mean_ic | 0.1591 | desc | 0.3432 | 0.3432 | positive | 2.0000 | false | true | NA | NA | 0.1591 | 0.1639 | 0.0283 | 0.0581 | 5.6225 | 2.8209 |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=underwater|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` with share 0.6568.
- Concentration score: 0.5492.
- Fragility flag: `true`.
- Fragility reason: most positive contribution comes from one dominant regime.
- Caveat: one or more regimes are sparse and carry null metrics

## Transition Attribution Highlights
- Best transition category was `stress_relief` at 0.3606.
- Worst transition category was `volatility_upshift` at 0.1519.
- Sparse transition categories: `generic_transition`, `recovery_onset`, `volatility_upshift`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_mean_ic | average_event_mean_ic | average_post_mean_ic | average_window_mean_ic | average_pre_mean_rank_ic | average_event_mean_rank_ic | average_post_mean_rank_ic | average_window_mean_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generic_transition | drawdown_recovery | alpha | 6 | 5 | 1 | 0 | post_mean_ic | 0.2136 | 5 | 0 | 3 | true | 0.2883 | 0.2745 | 0.2136 | 0.2684 | 0.2409 | 0.2033 | 0.2434 | 0.2349 |
| generic_transition | trend | alpha | 1 | 1 | 0 | 0 | post_mean_ic | 0.3158 | 2 | 0 | 1 | false | 0.3656 | 0.1294 | 0.3158 | 0.2984 | 0.2550 | 0.1714 | 0.2814 | 0.2489 |
| recovery_onset | drawdown_recovery | alpha | 6 | 5 | 1 | 0 | post_mean_ic | 0.2484 | 4 | 0 | 3 | true | 0.2384 | 0.2210 | 0.2484 | 0.2422 | 0.2127 | 0.2259 | 0.2288 | 0.2227 |
| stress_relief | stress | alpha | 1 | 1 | 0 | 0 | post_mean_ic | 0.3606 | 1 | 0 | 0 | false | 0.2581 | 0.2547 | 0.3606 | 0.2984 | 0.1982 | 0.2252 | 0.3550 | 0.2663 |
| trend_breakdown | trend | alpha | 2 | 2 | 0 | 0 | post_mean_ic | 0.1855 | 6 | 0 | 2 | false | 0.3301 | 0.3656 | 0.1855 | 0.2728 | 0.2833 | 0.2985 | 0.2007 | 0.2474 |
| volatility_downshift | volatility | alpha | 3 | 3 | 0 | 0 | post_mean_ic | 0.3052 | 3 | 0 | 2 | false | 0.2707 | 0.3180 | 0.3052 | 0.2896 | 0.2661 | 0.3015 | 0.2225 | 0.2518 |
| volatility_upshift | volatility | alpha | 2 | 1 | 1 | 0 | post_mean_ic | 0.1519 | 7 | 0 | 1 | true | 0.2929 | 0.3616 | 0.1519 | 0.3074 | 0.2293 | 0.2208 | 0.2317 | 0.2382 |

## Comparison Tables
- Comparison is aligned on shared regime dimension `composite`.
- Runs leading in more than one regime: `ml_cross_sectional_lgbm_2026_q1`.

| run_id | surface | dimension | regime_label | primary_metric | primary_metric_value | metric_direction | coverage_status | observation_count | rank_within_regime | warning_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_cross_sectional_lgbm_2026_q1 | alpha | composite | volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | mean_ic | NA | desc | sparse | 1 | 1 | sparse_evidence |
| ml_cross_sectional_xgb_2026_q1 | alpha | composite | volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | mean_ic | NA | desc | sparse | 1 | 2 | sparse_evidence |
| rank_composite_momentum_2026_q1 | alpha | composite | volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | mean_ic | NA | desc | sparse | 1 | 3 | sparse_evidence |
| ml_cross_sectional_xgb_2026_q1 | alpha | composite | volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | mean_ic | 0.3045 | desc | sufficient | 5 | 1 | NA |
| ml_cross_sectional_lgbm_2026_q1 | alpha | composite | volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | mean_ic | 0.2074 | desc | sufficient | 5 | 2 | NA |
| rank_composite_momentum_2026_q1 | alpha | composite | volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | mean_ic | 0.0041 | desc | sufficient | 5 | 3 | NA |
| ml_cross_sectional_lgbm_2026_q1 | alpha | composite | volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | mean_ic | 0.1810 | desc | sufficient | 5 | 1 | NA |
| ml_cross_sectional_xgb_2026_q1 | alpha | composite | volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | mean_ic | 0.1591 | desc | sufficient | 5 | 2 | NA |
| rank_composite_momentum_2026_q1 | alpha | composite | volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | mean_ic | -0.0713 | desc | sufficient | 5 | 3 | NA |
| ml_cross_sectional_lgbm_2026_q1 | alpha | composite | volatility=normal_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | mean_ic | NA | desc | sparse | 2 | 1 | sparse_evidence |
| ml_cross_sectional_xgb_2026_q1 | alpha | composite | volatility=normal_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | mean_ic | NA | desc | sparse | 2 | 2 | sparse_evidence |
| rank_composite_momentum_2026_q1 | alpha | composite | volatility=normal_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | mean_ic | NA | desc | sparse | 2 | 3 | sparse_evidence |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
