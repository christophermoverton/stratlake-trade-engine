# Alpha Regime Attribution Report

## Executive Summary
- Best mean_ic came from `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` at 0.2074.
- Weakest regime was `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` at 0.1810.
- Positive contribution is led by `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` with share 0.5340.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `mean_ic` (desc).
- Best regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` (0.2074).
- Worst regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` (0.1810).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | mean_ic | mean_rank_ic | ic_std | rank_ic_std | ic_ir | rank_ic_ir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | alpha | 5 | sufficient | mean_ic | 0.2074 | desc | 0.5340 | 0.5340 | positive | 1.0000 | true | false | NA | NA | 0.2074 | 0.2008 | 0.1974 | 0.1632 | 1.0505 | 1.2302 |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 5 | sufficient | mean_ic | 0.1810 | desc | 0.4660 | 0.4660 | positive | 2.0000 | false | true | NA | NA | 0.1810 | 0.1704 | 0.0515 | 0.0671 | 3.5169 | 2.5377 |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=underwater|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` with share 0.5340.
- Concentration score: 0.5023.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics

## Transition Attribution Highlights
- Best transition category was `generic_transition` at 0.3964.
- Worst transition category was `volatility_upshift` at 0.1022.
- Sparse transition categories: `generic_transition`, `recovery_onset`, `volatility_upshift`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_mean_ic | average_event_mean_ic | average_post_mean_ic | average_window_mean_ic | average_pre_mean_rank_ic | average_event_mean_rank_ic | average_post_mean_rank_ic | average_window_mean_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generic_transition | drawdown_recovery | alpha | 6 | 5 | 1 | 0 | post_mean_ic | 0.2210 | 6 | 0 | 4 | true | 0.2751 | 0.3056 | 0.2210 | 0.2684 | 0.2358 | 0.2437 | 0.2560 | 0.2483 |
| generic_transition | trend | alpha | 1 | 1 | 0 | 0 | post_mean_ic | 0.3964 | 1 | 0 | 1 | false | 0.4552 | 0.2136 | 0.3964 | 0.3833 | 0.3315 | 0.1980 | 0.3911 | 0.3287 |
| recovery_onset | drawdown_recovery | alpha | 6 | 5 | 1 | 0 | post_mean_ic | 0.2564 | 5 | 0 | 3 | true | 0.2487 | 0.2108 | 0.2564 | 0.2526 | 0.2197 | 0.2063 | 0.2545 | 0.2354 |
| stress_relief | stress | alpha | 1 | 1 | 0 | 0 | post_mean_ic | 0.3040 | 3 | 0 | 1 | false | 0.3373 | 0.4515 | 0.3040 | 0.3468 | 0.2676 | 0.4458 | 0.3168 | 0.3229 |
| trend_breakdown | trend | alpha | 2 | 2 | 0 | 0 | post_mean_ic | 0.2810 | 4 | 0 | 2 | false | 0.4229 | 0.3639 | 0.2810 | 0.3488 | 0.3585 | 0.3171 | 0.2671 | 0.3138 |
| volatility_downshift | volatility | alpha | 3 | 3 | 0 | 0 | post_mean_ic | 0.3521 | 2 | 0 | 2 | false | 0.3160 | 0.2938 | 0.3521 | 0.3223 | 0.3073 | 0.3077 | 0.2793 | 0.2962 |
| volatility_upshift | volatility | alpha | 2 | 1 | 1 | 0 | post_mean_ic | 0.1022 | 7 | 0 | 1 | true | 0.2366 | 0.3290 | 0.1022 | 0.2571 | 0.2256 | 0.1930 | 0.2048 | 0.2296 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
