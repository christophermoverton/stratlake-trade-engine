# Strategy Regime Attribution Report

## Executive Summary
- Best total_return came from `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` at 0.1140.
- Weakest regime was `volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress` at -0.0738.
- Positive contribution is led by `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` with share 0.5095.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `total_return` (desc).
- Best regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` (0.1140).
- Worst regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress` (-0.0738).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | total_return | annualized_return | volatility | annualized_volatility | sharpe_ratio | max_drawdown | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | composite | strategy | 4 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=dispersion_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress | composite | strategy | 5 | sufficient | total_return | -0.0738 | desc | 0.0000 | 0.2103 | negative | 16.0000 | false | true | NA | NA | -0.0738 | -0.9790 | 0.0159 | 0.2519 | -15.1122 | 0.0562 | 0.0000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | composite | strategy | 7 | sufficient | total_return | 0.0313 | desc | 0.1399 | 0.0893 | positive | 2.0000 | false | false | NA | NA | 0.0313 | 2.0338 | 0.0049 | 0.0776 | 14.3659 | 0.0032 | 0.7143 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress | composite | strategy | 7 | sufficient | total_return | 0.1140 | desc | 0.5095 | 0.3251 | positive | 1.0000 | true | false | NA | NA | 0.1140 | 47.7986 | 0.0332 | 0.5264 | 7.6563 | 0.0015 | 0.7143 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | strategy | 5 | sufficient | total_return | 0.0080 | desc | 0.0358 | 0.0229 | positive | 6.0000 | false | false | NA | NA | 0.0080 | 0.4954 | 0.0043 | 0.0679 | 5.9595 | 0.0039 | 0.6000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=dispersion_stress | composite | strategy | 3 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` with share 0.5095.
- Concentration score: 0.3049.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics
- Caveat: matched_undefined rows were excluded from attribution

## Transition Attribution Highlights
- Best transition category was `generic_transition` at 0.0120.
- Worst transition category was `trend_breakdown` at -0.0130.
- `stress_onset` shows post_transition_return at 0.0028.
- Sparse transition categories: `generic_transition`, `generic_transition`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_transition_return | average_event_window_return | average_post_transition_return | average_window_cumulative_return | average_window_max_drawdown | average_window_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_onset | drawdown_recovery | strategy | 9 | 9 | 0 | 0 | post_transition_return | 0.0074 | 2 | 2 | 2 | false | 0.0015 | -0.0065 | 0.0074 | 0.0018 | 0.0216 | 0.4286 |
| generic_transition | drawdown_recovery | strategy | 43 | 42 | 1 | 0 | post_transition_return | -0.0002 | 7 | 18 | 24 | true | 0.0013 | -0.0002 | -0.0002 | 0.0010 | 0.0062 | 0.5542 |
| generic_transition | stress | strategy | 1 | 1 | 0 | 0 | post_transition_return | 0.0120 | 1 | 0 | 1 | false | 0.0677 | -0.0121 | 0.0120 | 0.0675 | 0.0169 | 0.4286 |
| generic_transition | trend | strategy | 22 | 21 | 1 | 0 | post_transition_return | -0.0004 | 8 | 8 | 9 | true | 0.0003 | 0.0007 | -0.0004 | 0.0007 | 0.0057 | 0.5065 |
| recovery_onset | drawdown_recovery | strategy | 30 | 30 | 0 | 0 | post_transition_return | 0.0002 | 6 | 13 | 15 | false | -0.0026 | 0.0058 | 0.0002 | 0.0032 | 0.0112 | 0.5524 |
| stress_onset | stress | strategy | 6 | 6 | 0 | 0 | post_transition_return | 0.0028 | 3 | 1 | 2 | false | -0.0126 | 0.0150 | 0.0028 | 0.0044 | 0.0112 | 0.5238 |
| stress_relief | stress | strategy | 6 | 6 | 0 | 0 | post_transition_return | -0.0024 | 9 | 2 | 3 | false | -0.0012 | 0.0012 | -0.0024 | -0.0024 | 0.0107 | 0.5000 |
| trend_breakdown | trend | strategy | 3 | 3 | 0 | 0 | post_transition_return | -0.0130 | 10 | 2 | 2 | false | -0.0025 | -0.0038 | -0.0130 | -0.0194 | 0.0223 | 0.3333 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
