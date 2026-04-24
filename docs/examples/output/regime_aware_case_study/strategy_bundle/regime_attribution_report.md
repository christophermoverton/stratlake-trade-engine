# Strategy Regime Attribution Report

## Executive Summary
- Best total_return came from `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress` at 0.0120.
- Weakest regime was `volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress` at -0.0189.
- Positive contribution is led by `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress` with share 0.4619.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `total_return` (desc).
- Best regime: `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress` (0.0120).
- Worst regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress` (-0.0189).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | total_return | annualized_return | volatility | annualized_volatility | sharpe_ratio | max_drawdown | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=correlation_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress | composite | strategy | 2 | sufficient | total_return | 0.0120 | desc | 0.4619 | 0.2273 | positive | 1.0000 | true | false | NA | NA | 0.0120 | 3.5153 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=correlation_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=correlation_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=underwater|stress=correlation_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | composite | strategy | 2 | sufficient | total_return | -0.0189 | desc | 0.0000 | 0.3572 | negative | 5.0000 | false | true | NA | NA | -0.0189 | -0.9098 | 0.0035 | 0.0561 | -42.6549 | 0.0070 | 0.0000 |
| volatility=low_volatility|trend=sideways|drawdown_recovery=near_peak|stress=correlation_stress | composite | strategy | 2 | sufficient | total_return | 0.0040 | desc | 0.1536 | 0.0756 | positive | 3.0000 | false | false | NA | NA | 0.0040 | 0.6543 | 0.0014 | 0.0224 | 22.4499 | 0.0000 | 1.0000 |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress` with share 0.4619.
- Concentration score: 0.3848.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics
- Caveat: matched_undefined rows were excluded from attribution

## Transition Attribution Highlights
- Best transition category was `recovery_onset` at 0.0073.
- Worst transition category was `generic_transition` at -0.0053.
- Sparse transition categories: `generic_transition`, `volatility_upshift`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_transition_return | average_event_window_return | average_post_transition_return | average_window_cumulative_return | average_window_max_drawdown | average_window_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_onset | drawdown_recovery | strategy | 2 | 2 | 0 | 0 | post_transition_return | -0.0005 | 4 | 1 | 0 | false | -0.0075 | -0.0095 | -0.0005 | -0.0174 | 0.0125 | 0.1667 |
| generic_transition | drawdown_recovery | strategy | 5 | 4 | 1 | 0 | post_transition_return | -0.0053 | 8 | 4 | 4 | true | 0.0042 | -0.0002 | -0.0053 | -0.0002 | 0.0064 | 0.3667 |
| generic_transition | stress | strategy | 2 | 2 | 0 | 0 | post_transition_return | -0.0030 | 6 | 2 | 2 | false | 0.0040 | 0.0050 | -0.0030 | 0.0060 | 0.0030 | 0.3333 |
| generic_transition | trend | strategy | 7 | 7 | 0 | 0 | post_transition_return | 0.0040 | 2 | 2 | 3 | false | 0.0013 | 0.0031 | 0.0040 | 0.0085 | 0.0023 | 0.6190 |
| recovery_onset | drawdown_recovery | strategy | 3 | 3 | 0 | 0 | post_transition_return | 0.0073 | 1 | 0 | 0 | false | -0.0030 | 0.0030 | 0.0073 | 0.0073 | 0.0007 | 0.5556 |
| trend_breakdown | trend | strategy | 2 | 2 | 0 | 0 | post_transition_return | -0.0035 | 7 | 1 | 1 | false | -0.0060 | -0.0080 | -0.0035 | -0.0174 | 0.0139 | 0.1667 |
| volatility_downshift | volatility | strategy | 5 | 5 | 0 | 0 | post_transition_return | 0.0012 | 3 | 3 | 2 | false | -0.0004 | -0.0022 | 0.0012 | -0.0014 | 0.0058 | 0.4000 |
| volatility_upshift | volatility | strategy | 9 | 8 | 1 | 0 | post_transition_return | -0.0008 | 5 | 4 | 4 | true | -0.0009 | -0.0008 | -0.0008 | -0.0023 | 0.0062 | 0.3889 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
