# Portfolio Regime Attribution Report

## Executive Summary
- Best total_return came from `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` at 0.0152.
- Weakest regime was `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` at -0.0036.
- Positive contribution is led by `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` with share 0.5878.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `total_return` (desc).
- Best regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` (0.0152).
- Worst regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` (-0.0036).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | total_return | annualized_return | volatility | annualized_volatility | sharpe_ratio | max_drawdown | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | composite | portfolio | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress | composite | portfolio | 2 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress | composite | portfolio | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | portfolio | 5 | sufficient | total_return | -0.0036 | desc | 0.0000 | 0.1238 | negative | 3.0000 | false | true | NA | NA | -0.0036 | -0.1682 | 0.0020 | 0.0310 | -5.9240 | 0.0057 | 0.4000 |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | portfolio | 5 | sufficient | total_return | 0.0152 | desc | 0.5878 | 0.5151 | positive | 1.0000 | true | false | NA | NA | 0.0152 | 1.1363 | 0.0043 | 0.0682 | 11.1785 | 0.0000 | 0.8000 |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | portfolio | 2 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=underwater|stress=dispersion_stress | composite | portfolio | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=normal_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | portfolio | 4 | sufficient | total_return | 0.0106 | desc | 0.4122 | 0.3612 | positive | 2.0000 | false | false | NA | NA | 0.0106 | 0.9481 | 0.0044 | 0.0696 | 9.6227 | 0.0026 | 0.7500 |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` with share 0.5878.
- Concentration score: 0.5154.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics

## Transition Attribution Highlights
- Best transition category was `generic_transition` at 0.0038.
- Worst transition category was `trend_breakdown` at 0.0013.
- Sparse transition categories: `generic_transition`, `recovery_onset`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_transition_return | average_event_window_return | average_post_transition_return | average_window_cumulative_return | average_window_max_drawdown | average_window_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generic_transition | drawdown_recovery | portfolio | 7 | 6 | 1 | 0 | post_transition_return | 0.0024 | 4 | 2 | 2 | true | -0.0004 | 0.0023 | 0.0024 | 0.0039 | 0.0020 | 0.5952 |
| generic_transition | trend | portfolio | 2 | 2 | 0 | 0 | post_transition_return | 0.0038 | 1 | 0 | 0 | false | 0.0021 | -0.0039 | 0.0038 | 0.0020 | 0.0052 | 0.6000 |
| recovery_onset | drawdown_recovery | portfolio | 7 | 6 | 1 | 0 | post_transition_return | 0.0018 | 6 | 3 | 4 | true | 0.0033 | -0.0017 | 0.0018 | 0.0030 | 0.0036 | 0.5333 |
| stress_relief | stress | portfolio | 1 | 1 | 0 | 0 | post_transition_return | 0.0032 | 3 | 0 | 0 | false | -0.0002 | 0.0015 | 0.0032 | 0.0044 | 0.0017 | 0.8000 |
| trend_breakdown | trend | portfolio | 2 | 2 | 0 | 0 | post_transition_return | 0.0013 | 7 | 1 | 2 | false | 0.0017 | 0.0014 | 0.0013 | 0.0044 | 0.0027 | 0.6500 |
| volatility_downshift | volatility | portfolio | 3 | 3 | 0 | 0 | post_transition_return | 0.0035 | 2 | 1 | 2 | false | 0.0007 | 0.0002 | 0.0035 | 0.0045 | 0.0032 | 0.6333 |
| volatility_upshift | volatility | portfolio | 3 | 3 | 0 | 0 | post_transition_return | 0.0019 | 5 | 1 | 1 | false | 0.0019 | 0.0003 | 0.0019 | 0.0042 | 0.0043 | 0.5333 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
