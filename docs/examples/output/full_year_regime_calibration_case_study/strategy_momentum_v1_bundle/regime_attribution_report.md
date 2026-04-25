# Strategy Regime Attribution Report

## Executive Summary
- Best total_return came from `volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress` at 0.0835.
- Weakest regime was `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` at -0.0644.
- Positive contribution is led by `volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress` with share 0.5011.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `total_return` (desc).
- Best regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress` (0.0835).
- Worst regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` (-0.0644).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | total_return | annualized_return | volatility | annualized_volatility | sharpe_ratio | max_drawdown | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | composite | strategy | 4 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=dispersion_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress | composite | strategy | 5 | sufficient | total_return | 0.0835 | desc | 0.5011 | 0.2432 | positive | 1.0000 | true | false | NA | NA | 0.0835 | 55.8284 | 0.0202 | 0.3212 | 12.8042 | 0.0115 | 0.8000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | composite | strategy | 7 | sufficient | total_return | 0.0278 | desc | 0.1666 | 0.0809 | positive | 2.0000 | false | false | NA | NA | 0.0278 | 1.6791 | 0.0058 | 0.0916 | 10.8151 | 0.0022 | 0.7143 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress | composite | strategy | 7 | sufficient | total_return | -0.0644 | desc | 0.0000 | 0.1875 | negative | 16.0000 | false | true | NA | NA | -0.0644 | -0.9088 | 0.0368 | 0.5849 | -3.8104 | 0.0914 | 0.7143 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | strategy | 5 | sufficient | total_return | 0.0000 | desc | 0.0003 | 0.0001 | positive | 9.0000 | false | false | NA | NA | 0.0000 | 0.0022 | 0.0041 | 0.0649 | 0.0591 | 0.0057 | 0.6000 |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=dispersion_stress | composite | strategy | 3 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress` with share 0.5011.
- Concentration score: 0.3124.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics
- Caveat: matched_undefined rows were excluded from attribution

## Transition Attribution Highlights
- Best transition category was `trend_breakdown` at 0.0164.
- Worst transition category was `stress_relief` at -0.0028.
- `stress_onset` shows post_transition_return at 0.0039.
- Sparse transition categories: `generic_transition`, `generic_transition`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_transition_return | average_event_window_return | average_post_transition_return | average_window_cumulative_return | average_window_max_drawdown | average_window_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_onset | drawdown_recovery | strategy | 9 | 9 | 0 | 0 | post_transition_return | -0.0027 | 9 | 6 | 4 | false | 0.0019 | 0.0070 | -0.0027 | 0.0054 | 0.0319 | 0.5238 |
| generic_transition | drawdown_recovery | strategy | 43 | 42 | 1 | 0 | post_transition_return | 0.0009 | 5 | 23 | 17 | true | -0.0014 | 0.0001 | 0.0009 | -0.0004 | 0.0090 | 0.4874 |
| generic_transition | stress | strategy | 1 | 1 | 0 | 0 | post_transition_return | -0.0009 | 7 | 1 | 0 | false | -0.0736 | 0.0288 | -0.0009 | -0.0478 | 0.0914 | 0.5714 |
| generic_transition | trend | strategy | 22 | 21 | 1 | 0 | post_transition_return | 0.0004 | 6 | 13 | 14 | true | 0.0025 | -0.0003 | 0.0004 | 0.0025 | 0.0088 | 0.5357 |
| recovery_onset | drawdown_recovery | strategy | 30 | 30 | 0 | 0 | post_transition_return | 0.0029 | 3 | 11 | 12 | false | 0.0007 | -0.0058 | 0.0029 | -0.0025 | 0.0195 | 0.4810 |
| stress_onset | stress | strategy | 6 | 6 | 0 | 0 | post_transition_return | 0.0039 | 2 | 4 | 3 | false | 0.0117 | -0.0138 | 0.0039 | 0.0006 | 0.0270 | 0.5476 |
| stress_relief | stress | strategy | 6 | 6 | 0 | 0 | post_transition_return | -0.0028 | 10 | 3 | 4 | false | 0.0031 | -0.0038 | -0.0028 | -0.0035 | 0.0165 | 0.4762 |
| trend_breakdown | trend | strategy | 3 | 3 | 0 | 0 | post_transition_return | 0.0164 | 1 | 2 | 1 | false | -0.0068 | 0.0045 | 0.0164 | 0.0137 | 0.0117 | 0.4762 |

## Comparison Tables
- Comparison is aligned on shared regime dimension `composite`.
- Runs leading in more than one regime: `mean_reversion_v1`, `momentum_v1`.

| run_id | surface | dimension | regime_label | primary_metric | primary_metric_value | metric_direction | coverage_status | observation_count | rank_within_regime | warning_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean_reversion_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | total_return | NA | desc | sparse | 4 | 1 | sparse_evidence |
| momentum_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | total_return | NA | desc | sparse | 4 | 2 | sparse_evidence |
| mean_reversion_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=dispersion_stress | total_return | NA | desc | sparse | 1 | 1 | sparse_evidence |
| momentum_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=dispersion_stress | total_return | NA | desc | sparse | 1 | 2 | sparse_evidence |
| momentum_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress | total_return | 0.0835 | desc | sufficient | 5 | 1 | NA |
| mean_reversion_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=normal_stress | total_return | -0.0738 | desc | sufficient | 5 | 2 | NA |
| mean_reversion_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | total_return | 0.0313 | desc | sufficient | 7 | 1 | NA |
| momentum_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | total_return | 0.0278 | desc | sufficient | 7 | 2 | NA |
| mean_reversion_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress | total_return | 0.1140 | desc | sufficient | 7 | 1 | NA |
| momentum_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress | total_return | -0.0644 | desc | sufficient | 7 | 2 | NA |
| mean_reversion_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | total_return | 0.0080 | desc | sufficient | 5 | 1 | NA |
| momentum_v1 | strategy | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | total_return | 0.0000 | desc | sufficient | 5 | 2 | NA |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
