# Portfolio Regime Attribution Report

## Executive Summary
- Best total_return came from `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` at 0.0134.
- Weakest regime was `volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=normal_stress` at -0.0238.
- Positive contribution is led by `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` with share 0.1936.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `total_return` (desc).
- Best regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` (0.0134).
- Worst regime: `volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=normal_stress` (-0.0238).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | total_return | annualized_return | volatility | annualized_volatility | sharpe_ratio | max_drawdown | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | composite | portfolio | 4 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress | composite | portfolio | 6 | sufficient | total_return | 0.0122 | desc | 0.1759 | 0.0994 | positive | 4.0000 | false | false | NA | NA | 0.0122 | 0.6629 | 0.0034 | 0.0546 | 9.3490 | 0.0012 | 0.5000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | composite | portfolio | 6 | sufficient | total_return | 0.0124 | desc | 0.1795 | 0.1015 | positive | 3.0000 | false | false | NA | NA | 0.0124 | 0.6802 | 0.0031 | 0.0491 | 10.5900 | 0.0018 | 0.8333 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress | composite | portfolio | 14 | sufficient | total_return | 0.0134 | desc | 0.1936 | 0.1094 | positive | 1.0000 | true | false | NA | NA | 0.0134 | 0.2709 | 0.0045 | 0.0716 | 3.3812 | 0.0101 | 0.5000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | portfolio | 3 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress | composite | portfolio | 8 | sufficient | total_return | -0.0038 | desc | 0.0000 | 0.0312 | negative | 12.0000 | false | false | NA | NA | -0.0038 | -0.1138 | 0.0032 | 0.0515 | -2.3210 | 0.0102 | 0.6250 |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=dispersion_stress | composite | portfolio | 12 | sufficient | total_return | 0.0015 | desc | 0.0215 | 0.0122 | positive | 9.0000 | false | false | NA | NA | 0.0015 | 0.0318 | 0.0014 | 0.0224 | 1.4070 | 0.0043 | 0.5833 |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=normal_stress | composite | portfolio | 12 | sufficient | total_return | -0.0238 | desc | 0.0000 | 0.1945 | negative | 16.0000 | false | true | NA | NA | -0.0238 | -0.3973 | 0.0060 | 0.0947 | -5.2985 | 0.0267 | 0.3333 |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` with share 0.1936.
- Concentration score: 0.1505.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics
- Caveat: matched_undefined rows were excluded from attribution

## Transition Attribution Highlights
- Best transition category was `drawdown_onset` at 0.0038.
- Worst transition category was `stress_relief` at -0.0048.
- `stress_onset` shows post_transition_return at 0.0018.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_transition_return | average_event_window_return | average_post_transition_return | average_window_cumulative_return | average_window_max_drawdown | average_window_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_onset | drawdown_recovery | portfolio | 2 | 2 | 0 | 0 | post_transition_return | 0.0038 | 1 | 1 | 1 | false | 0.0067 | -0.0043 | 0.0038 | 0.0062 | 0.0074 | 0.4286 |
| generic_transition | drawdown_recovery | portfolio | 10 | 10 | 0 | 0 | post_transition_return | 0.0008 | 7 | 4 | 4 | false | 0.0004 | 0.0001 | 0.0008 | 0.0013 | 0.0033 | 0.5667 |
| generic_transition | trend | portfolio | 14 | 14 | 0 | 0 | post_transition_return | -0.0019 | 8 | 8 | 9 | false | 0.0004 | 0.0006 | -0.0019 | -0.0008 | 0.0052 | 0.5612 |
| recovery_onset | drawdown_recovery | portfolio | 10 | 10 | 0 | 0 | post_transition_return | 0.0018 | 5 | 5 | 5 | false | 0.0008 | 0.0012 | 0.0018 | 0.0037 | 0.0042 | 0.5714 |
| stress_onset | stress | portfolio | 4 | 4 | 0 | 0 | post_transition_return | 0.0018 | 4 | 1 | 3 | false | 0.0048 | 0.0032 | 0.0018 | 0.0098 | 0.0053 | 0.7500 |
| stress_relief | stress | portfolio | 4 | 4 | 0 | 0 | post_transition_return | -0.0048 | 9 | 2 | 3 | false | 0.0010 | 0.0017 | -0.0048 | -0.0020 | 0.0092 | 0.5000 |
| trend_breakdown | trend | portfolio | 2 | 2 | 0 | 0 | post_transition_return | 0.0027 | 2 | 1 | 0 | false | -0.0028 | -0.0005 | 0.0027 | -0.0005 | 0.0077 | 0.5714 |
| volatility_downshift | volatility | portfolio | 7 | 7 | 0 | 0 | post_transition_return | 0.0010 | 6 | 3 | 3 | false | 0.0002 | 0.0009 | 0.0010 | 0.0020 | 0.0022 | 0.6054 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
