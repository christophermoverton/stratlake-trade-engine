# Alpha Regime Attribution Report

## Executive Summary
- Best mean_ic came from `volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress` at 0.0375.
- Weakest regime was `volatility=low_volatility|trend=uptrend|drawdown_recovery=recovery|stress=correlation_stress` at 0.0150.
- Positive contribution is led by `volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress` with share 0.3488.
- Fragility heuristic: no deterministic fragility heuristic triggered.

## Regime Attribution Highlights
- Primary metric: `mean_ic` (desc).
- Best regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress` (0.0375).
- Worst regime: `volatility=low_volatility|trend=uptrend|drawdown_recovery=recovery|stress=correlation_stress` (0.0150).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | mean_ic | mean_rank_ic | ic_std | rank_ic_std | ic_ir | rank_ic_ir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=correlation_stress | composite | alpha | 2 | sufficient | mean_ic | 0.0150 | desc | 0.1395 | 0.1395 | positive | 4.0000 | false | false | NA | NA | 0.0150 | 0.0220 | 0.0071 | 0.0057 | 2.1213 | 3.8891 |
| volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=underwater|stress=correlation_stress | composite | alpha | 1 | sparse | mean_ic | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress | composite | alpha | 2 | sufficient | mean_ic | 0.0375 | desc | 0.3488 | 0.3488 | positive | 1.0000 | true | false | NA | NA | 0.0375 | 0.0475 | 0.0035 | 0.0035 | 10.6066 | 13.4350 |
| volatility=low_volatility|trend=sideways|drawdown_recovery=near_peak|stress=correlation_stress | composite | alpha | 2 | sufficient | mean_ic | 0.0175 | desc | 0.1628 | 0.1628 | positive | 3.0000 | false | false | NA | NA | 0.0175 | 0.0250 | 0.0035 | 0.0042 | 4.9497 | 5.8926 |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=low_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress` with share 0.3488.
- Concentration score: 0.2309.
- Fragility flag: `false`.
- Fragility reason: no deterministic fragility heuristic triggered.
- Caveat: one or more regimes are sparse and carry null metrics
- Caveat: matched_undefined rows were excluded from attribution

## Transition Attribution Highlights
- Best transition category was `generic_transition` at 0.0307.
- Worst transition category was `recovery_onset` at 0.0173.
- Sparse transition categories: `generic_transition`, `volatility_upshift`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_mean_ic | average_event_mean_ic | average_post_mean_ic | average_window_mean_ic | average_pre_mean_rank_ic | average_event_mean_rank_ic | average_post_mean_rank_ic | average_window_mean_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_onset | drawdown_recovery | alpha | 2 | 2 | 0 | 0 | post_mean_ic | 0.0215 | 5 | 0 | 2 | false | 0.0325 | 0.0375 | 0.0215 | 0.0305 | 0.0425 | 0.0475 | 0.0295 | 0.0398 |
| generic_transition | drawdown_recovery | alpha | 5 | 4 | 1 | 0 | post_mean_ic | 0.0307 | 1 | 0 | 1 | true | 0.0196 | 0.0210 | 0.0307 | 0.0226 | 0.0262 | 0.0292 | 0.0402 | 0.0305 |
| generic_transition | stress | alpha | 2 | 2 | 0 | 0 | post_mean_ic | 0.0290 | 2 | 0 | 1 | false | 0.0250 | 0.0215 | 0.0290 | 0.0252 | 0.0320 | 0.0290 | 0.0380 | 0.0330 |
| generic_transition | trend | alpha | 7 | 7 | 0 | 0 | post_mean_ic | 0.0189 | 7 | 0 | 5 | false | 0.0216 | 0.0200 | 0.0189 | 0.0201 | 0.0297 | 0.0274 | 0.0261 | 0.0278 |
| recovery_onset | drawdown_recovery | alpha | 3 | 3 | 0 | 0 | post_mean_ic | 0.0173 | 8 | 0 | 3 | false | 0.0283 | 0.0187 | 0.0173 | 0.0214 | 0.0380 | 0.0260 | 0.0240 | 0.0293 |
| trend_breakdown | trend | alpha | 2 | 2 | 0 | 0 | post_mean_ic | 0.0265 | 3 | 0 | 1 | false | 0.0275 | 0.0350 | 0.0265 | 0.0297 | 0.0375 | 0.0450 | 0.0345 | 0.0390 |
| volatility_downshift | volatility | alpha | 5 | 5 | 0 | 0 | post_mean_ic | 0.0210 | 6 | 0 | 4 | false | 0.0244 | 0.0266 | 0.0210 | 0.0240 | 0.0330 | 0.0352 | 0.0282 | 0.0321 |
| volatility_upshift | volatility | alpha | 9 | 8 | 1 | 0 | post_mean_ic | 0.0244 | 4 | 0 | 4 | true | 0.0229 | 0.0217 | 0.0244 | 0.0226 | 0.0313 | 0.0299 | 0.0330 | 0.0309 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
