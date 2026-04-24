# Strategy Regime Attribution Report

## Executive Summary
- Best total_return came from `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress` at 0.0099.
- Weakest regime was `volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress` at -0.0033.
- Positive contribution is led by `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress` with share 0.7908.
- Fragility heuristic: most positive contribution comes from one dominant regime.

## Regime Attribution Highlights
- Primary metric: `total_return` (desc).
- Best regime: `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress` (0.0099).
- Worst regime: `volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress` (-0.0033).

### Best / Worst Regimes
| regime_label | dimension | surface | observation_count | coverage_status | primary_metric | primary_metric_value | metric_direction | positive_contribution_share | absolute_contribution_share | contribution_sign | metric_rank | is_best_regime | is_worst_regime | evidence_warning | coverage_caveat | total_return | annualized_return | volatility | annualized_volatility | sharpe_ratio | max_drawdown | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volatility=high_volatility|trend=sideways|drawdown_recovery=near_peak|stress=dispersion_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=dispersion_stress | composite | strategy | 2 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress | composite | strategy | 4 | sufficient | total_return | 0.0099 | desc | 0.7908 | 0.5082 | positive | 1.0000 | true | false | NA | NA | 0.0099 | 0.8653 | 0.0046 | 0.0727 | 8.6131 | 0.0003 | 0.5000 |
| volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=dispersion_stress | composite | strategy | 2 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress | composite | strategy | 4 | sufficient | total_return | -0.0033 | desc | 0.0000 | 0.1703 | negative | 6.0000 | false | true | NA | NA | -0.0033 | -0.1897 | 0.0011 | 0.0182 | -11.5260 | 0.0032 | 0.0000 |
| volatility=high_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=dispersion_stress | composite | strategy | 1 | sparse | total_return | NA | desc | NA | NA | undefined | NA | false | false | sparse_evidence | regime has fewer than min_observations rows | NA | NA | NA | NA | NA | NA | NA |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress | composite | strategy | 5 | sufficient | total_return | 0.0026 | desc | 0.2092 | 0.1345 | positive | 2.0000 | false | false | NA | NA | 0.0026 | 0.1416 | 0.0006 | 0.0095 | 13.8771 | 0.0002 | 0.8000 |
| volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress | composite | strategy | 5 | sufficient | total_return | -0.0004 | desc | 0.0000 | 0.0213 | negative | 3.0000 | false | false | NA | NA | -0.0004 | -0.0208 | 0.0002 | 0.0025 | -8.3087 | 0.0006 | 0.2000 |

## Concentration and Fragility Warnings
- Dominant regime: `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress` with share 0.7908.
- Concentration score: 0.6691.
- Fragility flag: `true`.
- Fragility reason: most positive contribution comes from one dominant regime.
- Caveat: one or more regimes are sparse and carry null metrics
- Caveat: matched_undefined rows were excluded from attribution

## Transition Attribution Highlights
- Best transition category was `stress_onset` at 0.0007.
- Worst transition category was `stress_relief` at -0.0015.
- `stress_onset` shows post_transition_return at 0.0007.
- Sparse transition categories: `generic_transition`.

### Transition Categories
| transition_category | transition_dimension | surface | event_count | sufficient_event_count | sparse_event_count | empty_event_count | primary_metric | primary_metric_value | metric_rank | adverse_event_count | degradation_event_count | sparse_warning | average_pre_transition_return | average_event_window_return | average_post_transition_return | average_window_cumulative_return | average_window_max_drawdown | average_window_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generic_transition | drawdown_recovery | strategy | 13 | 12 | 1 | 0 | post_transition_return | -0.0000 | 5 | 6 | 8 | true | 0.0012 | 0.0003 | -0.0000 | 0.0015 | 0.0013 | 0.5333 |
| generic_transition | trend | strategy | 6 | 6 | 0 | 0 | post_transition_return | -0.0011 | 7 | 4 | 3 | false | 0.0008 | 0.0016 | -0.0011 | 0.0012 | 0.0018 | 0.4833 |
| recovery_onset | drawdown_recovery | strategy | 10 | 10 | 0 | 0 | post_transition_return | 0.0001 | 3 | 3 | 2 | false | -0.0007 | 0.0017 | 0.0001 | 0.0011 | 0.0009 | 0.5000 |
| stress_onset | stress | strategy | 2 | 2 | 0 | 0 | post_transition_return | 0.0007 | 1 | 0 | 0 | false | 0.0005 | 0.0017 | 0.0007 | 0.0030 | 0.0008 | 0.6000 |
| stress_relief | stress | strategy | 2 | 2 | 0 | 0 | post_transition_return | -0.0015 | 8 | 2 | 2 | false | 0.0024 | -0.0004 | -0.0015 | 0.0005 | 0.0025 | 0.5000 |
| trend_breakdown | trend | strategy | 3 | 3 | 0 | 0 | post_transition_return | 0.0005 | 2 | 0 | 1 | false | -0.0006 | 0.0007 | 0.0005 | 0.0006 | 0.0008 | 0.5333 |
| volatility_downshift | volatility | strategy | 4 | 4 | 0 | 0 | post_transition_return | 0.0001 | 4 | 1 | 2 | false | -0.0000 | 0.0006 | 0.0001 | 0.0007 | 0.0006 | 0.5500 |
| volatility_upshift | volatility | strategy | 5 | 5 | 0 | 0 | post_transition_return | -0.0001 | 6 | 2 | 2 | false | 0.0002 | 0.0026 | -0.0001 | 0.0027 | 0.0013 | 0.5600 |

## Interpretation Notes
- Results are descriptive regime slices, not causal claims.
- Sparse and empty regimes remain visible and should limit confidence.
- Undefined or unmatched rows are excluded from metric attribution.

## Limitations and Cautions
- Concentration and fragility flags are deterministic heuristics.
- Transition summaries describe behavior around regime changes only.
