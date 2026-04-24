# Alpha Regime Attribution Report

## Executive Summary
- Best mean_ic came from `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress` at 0.0400.
- Weakest regime was `volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress` at -0.0300.
- Positive contribution is led by `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress` with share 0.3810.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `mean_ic` (desc).
- Best regime: `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress` (0.0400).
- Worst regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress` (-0.0300).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | mean_ic | mean_rank_ic | ic_std | rank_ic_std | ic_ir | rank_ic_ir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress | composite | alpha | 2 | sufficient | mean_ic | 0.0400 | desc | 0.3810 | 0.2667 | positive | 1.0000 | true | false | NA | NA | 0.0400 | 0.0500 | 0.0000 | 0.0000 | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=underwater|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | composite | alpha | 2 | sufficient | mean_ic | -0.0300 | desc | 0.0000 | 0.2000 | negative | 5.0000 | false | true | NA | NA | -0.0300 | -0.0200 | 0.0000 | 0.0000 | NA | NA |
| volatility=low_volatility|trend=sideways|drawdown_recovery=near_peak|stress=correlation_stress | composite | alpha | 2 | sufficient | mean_ic | 0.0250 | desc | 0.2381 | 0.1667 | positive | 3.0000 | false | false | NA | NA | 0.0250 | 0.0350 | 0.0071 | 0.0071 | 3.5355 | 4.9497 |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress` with share 0.3810.
- Concentration score: 0.3469.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics
- Caveat: matched_undefined rows were excluded from attribution

## Transition Attribution Highlights
- Best transition category was `recovery_onset` at 0.0467.
- Worst transition category was `generic_transition` at -0.0125.
- Sparse transition categories: `generic_transition`, `volatility_upshift`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_mean_ic | average_event_mean_ic | average_post_mean_ic | average_window_mean_ic | average_pre_mean_rank_ic | average_event_mean_rank_ic | average_post_mean_rank_ic | average_window_mean_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_onset | drawdown_recovery | alpha | 2 | 2 | 0 | 0 | post_mean_ic | 0.0050 | 5 | 1 | 0 | false | -0.0200 | -0.0300 | 0.0050 | -0.0150 | -0.0100 | -0.0200 | 0.0150 | -0.0050 |
| generic_transition | drawdown_recovery | alpha | 5 | 4 | 1 | 0 | post_mean_ic | -0.0125 | 8 | 3 | 4 | true | 0.0320 | 0.0060 | -0.0125 | 0.0100 | 0.0420 | 0.0160 | -0.0025 | 0.0200 |
| generic_transition | stress | alpha | 2 | 2 | 0 | 0 | post_mean_ic | -0.0050 | 7 | 1 | 2 | false | 0.0300 | 0.0250 | -0.0050 | 0.0167 | 0.0400 | 0.0350 | 0.0050 | 0.0267 |
| generic_transition | trend | alpha | 7 | 7 | 0 | 0 | post_mean_ic | 0.0300 | 2 | 1 | 3 | false | 0.0114 | 0.0257 | 0.0300 | 0.0224 | 0.0214 | 0.0357 | 0.0400 | 0.0324 |
| recovery_onset | drawdown_recovery | alpha | 3 | 3 | 0 | 0 | post_mean_ic | 0.0467 | 1 | 0 | 0 | false | -0.0167 | 0.0233 | 0.0467 | 0.0178 | -0.0067 | 0.0333 | 0.0567 | 0.0278 |
| trend_breakdown | trend | alpha | 2 | 2 | 0 | 0 | post_mean_ic | 0.0000 | 6 | 1 | 1 | false | -0.0150 | -0.0250 | 0.0000 | -0.0133 | -0.0050 | -0.0150 | 0.0100 | -0.0033 |
| volatility_downshift | volatility | alpha | 5 | 5 | 0 | 0 | post_mean_ic | 0.0140 | 3 | 1 | 2 | false | 0.0080 | 0.0040 | 0.0140 | 0.0087 | 0.0180 | 0.0140 | 0.0240 | 0.0187 |
| volatility_upshift | volatility | alpha | 9 | 8 | 1 | 0 | post_mean_ic | 0.0075 | 4 | 5 | 4 | true | 0.0089 | 0.0078 | 0.0075 | 0.0081 | 0.0189 | 0.0178 | 0.0175 | 0.0181 |

## Comparison Tables
- Comparison is aligned on shared regime dimension `composite`.
- Runs leading in more than one regime: `alpha_baseline`, `alpha_defensive`.

| run_id | surface | dimension | regime_label | primary_metric | primary_metric_value | metric_direction | coverage_status | observation_count | rank_within_regime | warning_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha_baseline | alpha | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 1 | sparse_evidence |
| alpha_defensive | alpha | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 2 | sparse_evidence |
| alpha_baseline | alpha | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 1 | sparse_evidence |
| alpha_defensive | alpha | composite | volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 2 | sparse_evidence |
| alpha_baseline | alpha | composite | volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress | mean_ic | 0.0400 | desc | sufficient | 2 | 1 | NA |
| alpha_defensive | alpha | composite | volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress | mean_ic | 0.0150 | desc | sufficient | 2 | 2 | NA |
| alpha_baseline | alpha | composite | volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 1 | sparse_evidence |
| alpha_defensive | alpha | composite | volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 2 | sparse_evidence |
| alpha_baseline | alpha | composite | volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 1 | sparse_evidence |
| alpha_defensive | alpha | composite | volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 2 | sparse_evidence |
| alpha_baseline | alpha | composite | volatility=high_volatility|trend=uptrend|drawdown_recovery=underwater|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 1 | sparse_evidence |
| alpha_defensive | alpha | composite | volatility=high_volatility|trend=uptrend|drawdown_recovery=underwater|stress=correlation_stress | mean_ic | NA | desc | sparse | 1 | 2 | sparse_evidence |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
