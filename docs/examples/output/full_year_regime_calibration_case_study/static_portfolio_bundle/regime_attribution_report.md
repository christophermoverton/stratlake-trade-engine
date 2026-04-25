# Portfolio Regime Attribution Report

## Executive Summary
- Best total_return came from `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress` at 0.0296.
- Weakest regime was `volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=normal_stress` at -0.0412.
- Positive contribution is led by `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress` with share 0.2965.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `total_return` (desc).
- Best regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress` (0.0296).
- Worst regime: `volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=normal_stress` (-0.0412).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | total_return | annualized_return | volatility | annualized_volatility | sharpe_ratio | max_drawdown | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | composite | portfolio | 4 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=dispersion_stress | composite | portfolio | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress | composite | portfolio | 5 | sufficient | total_return | 0.0030 | desc | 0.0297 | 0.0188 | positive | 8.0000 | false | false | NA | NA | 0.0030 | 0.1607 | 0.0062 | 0.0981 | 1.5586 | 0.0062 | 0.4000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | composite | portfolio | 7 | sufficient | total_return | 0.0296 | desc | 0.2965 | 0.1878 | positive | 1.0000 | true | false | NA | NA | 0.0296 | 1.8572 | 0.0028 | 0.0449 | 23.4656 | 0.0000 | 1.0000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress | composite | portfolio | 7 | sufficient | total_return | 0.0253 | desc | 0.2532 | 0.1604 | positive | 2.0000 | false | false | NA | NA | 0.0253 | 1.4550 | 0.0031 | 0.0495 | 18.1911 | 0.0005 | 0.8571 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | portfolio | 5 | sufficient | total_return | 0.0041 | desc | 0.0406 | 0.0257 | positive | 6.0000 | false | false | NA | NA | 0.0041 | 0.2262 | 0.0013 | 0.0205 | 9.9592 | 0.0011 | 0.6000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=dispersion_stress | composite | portfolio | 3 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | portfolio | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress` with share 0.2965.
- Concentration score: 0.1882.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics
- Caveat: matched_undefined rows were excluded from attribution

## Transition Attribution Highlights
- Best transition category was `generic_transition` at 0.0055.
- Worst transition category was `stress_relief` at -0.0026.
- `stress_onset` shows post_transition_return at 0.0034.
- Sparse transition categories: `generic_transition`, `generic_transition`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_transition_return | average_event_window_return | average_post_transition_return | average_window_cumulative_return | average_window_max_drawdown | average_window_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_onset | drawdown_recovery | portfolio | 9 | 9 | 0 | 0 | post_transition_return | 0.0024 | 3 | 3 | 3 | false | 0.0019 | 0.0002 | 0.0024 | 0.0045 | 0.0069 | 0.5238 |
| generic_transition | drawdown_recovery | portfolio | 43 | 42 | 1 | 0 | post_transition_return | 0.0004 | 7 | 20 | 21 | true | -0.0000 | -0.0000 | 0.0004 | 0.0003 | 0.0042 | 0.5497 |
| generic_transition | stress | portfolio | 1 | 1 | 0 | 0 | post_transition_return | 0.0055 | 1 | 0 | 0 | false | -0.0012 | 0.0084 | 0.0055 | 0.0128 | 0.0018 | 0.4286 |
| generic_transition | trend | portfolio | 22 | 21 | 1 | 0 | post_transition_return | -0.0000 | 8 | 12 | 16 | true | 0.0014 | 0.0002 | -0.0000 | 0.0016 | 0.0032 | 0.5731 |
| recovery_onset | drawdown_recovery | portfolio | 30 | 30 | 0 | 0 | post_transition_return | 0.0017 | 4 | 10 | 12 | false | -0.0009 | 0.0000 | 0.0017 | 0.0008 | 0.0059 | 0.5095 |
| stress_onset | stress | portfolio | 6 | 6 | 0 | 0 | post_transition_return | 0.0034 | 2 | 2 | 2 | false | -0.0006 | 0.0006 | 0.0034 | 0.0034 | 0.0041 | 0.6190 |
| stress_relief | stress | portfolio | 6 | 6 | 0 | 0 | post_transition_return | -0.0026 | 10 | 3 | 3 | false | 0.0010 | -0.0013 | -0.0026 | -0.0029 | 0.0098 | 0.5476 |
| trend_breakdown | trend | portfolio | 3 | 3 | 0 | 0 | post_transition_return | 0.0015 | 5 | 1 | 1 | false | -0.0046 | 0.0003 | 0.0015 | -0.0028 | 0.0045 | 0.4762 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
