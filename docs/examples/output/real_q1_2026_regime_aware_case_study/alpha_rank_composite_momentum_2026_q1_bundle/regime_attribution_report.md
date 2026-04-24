# Alpha Regime Attribution Report

## Executive Summary
- Best mean_ic came from `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` at 0.0041.
- Weakest regime was `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` at -0.0713.
- Positive contribution is led by `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` with share 1.0000.
- Fragility heuristic: positive performance is isolated to one regime.

## Regime Attribution Highlights
- Primary metric: `mean_ic` (desc).
- Best regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` (0.0041).
- Worst regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` (-0.0713).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | mean_ic | mean_rank_ic | ic_std | rank_ic_std | ic_ir | rank_ic_ir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | alpha | 5 | sufficient | mean_ic | 0.0041 | desc | 1.0000 | 0.0545 | positive | 1.0000 | true | false | NA | NA | 0.0041 | -0.0085 | 0.1580 | 0.1856 | 0.0260 | -0.0458 |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 5 | sufficient | mean_ic | -0.0713 | desc | 0.0000 | 0.9455 | negative | 2.0000 | false | true | NA | NA | -0.0713 | -0.0610 | 0.1384 | 0.1191 | -0.5148 | -0.5118 |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=underwater|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress | composite | alpha | 2 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` with share 1.0000.
- Concentration score: 1.0000.
- Fragility flag: `true`.
- Fragility reason: positive performance is isolated to one regime.
- Caveat: one or more regimes are sparse and carry null metrics

## Transition Attribution Highlights
- Best transition category was `generic_transition` at 0.2343.
- Worst transition category was `recovery_onset` at 0.0371.
- Sparse transition categories: `generic_transition`, `recovery_onset`, `volatility_upshift`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_mean_ic | average_event_mean_ic | average_post_mean_ic | average_window_mean_ic | average_pre_mean_rank_ic | average_event_mean_rank_ic | average_post_mean_rank_ic | average_window_mean_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generic_transition | drawdown_recovery | alpha | 6 | 5 | 1 | 0 | post_mean_ic | 0.0869 | 6 | 1 | 1 | true | -0.0658 | 0.0228 | 0.0869 | 0.0193 | -0.0398 | 0.0186 | 0.1001 | 0.0285 |
| generic_transition | trend | alpha | 1 | 1 | 0 | 0 | post_mean_ic | 0.2343 | 1 | 0 | 0 | false | -0.2779 | 0.1342 | 0.2343 | 0.0094 | -0.1728 | 0.1643 | 0.2590 | 0.0674 |
| recovery_onset | drawdown_recovery | alpha | 6 | 5 | 1 | 0 | post_mean_ic | 0.0371 | 7 | 2 | 2 | true | -0.0750 | -0.0037 | 0.0371 | -0.0203 | -0.0522 | 0.0202 | 0.0435 | 0.0000 |
| stress_relief | stress | alpha | 1 | 1 | 0 | 0 | post_mean_ic | 0.1285 | 3 | 0 | 0 | false | -0.0789 | 0.3839 | 0.1285 | 0.0966 | 0.0143 | 0.3610 | 0.1981 | 0.1571 |
| trend_breakdown | trend | alpha | 2 | 2 | 0 | 0 | post_mean_ic | 0.1668 | 2 | 0 | 1 | false | -0.0147 | -0.0599 | 0.1668 | 0.0743 | 0.0246 | 0.0517 | 0.1654 | 0.1118 |
| volatility_downshift | volatility | alpha | 3 | 3 | 0 | 0 | post_mean_ic | 0.1161 | 4 | 0 | 2 | false | 0.0211 | 0.0038 | 0.1161 | 0.0726 | 0.0400 | 0.0756 | 0.0813 | 0.0806 |
| volatility_upshift | volatility | alpha | 2 | 1 | 1 | 0 | post_mean_ic | 0.0929 | 5 | 0 | 0 | true | -0.0501 | -0.0238 | 0.0929 | 0.0045 | -0.0755 | -0.0838 | 0.0708 | -0.0361 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
