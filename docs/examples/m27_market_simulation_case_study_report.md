# M27 Market Simulation Stress Testing Case Study Report

## Executive Summary

This report summarizes the deterministic M27 market simulation case-study run
`m27_market_simulation_case_study_b542a0934d5e`. The workflow evaluates one
fixture-backed adaptive policy evidence set across historical replay, regime
block bootstrap paths, regime-transition Monte Carlo paths, shock overlays, and
simulation-aware stress metrics.

The case study produced 6 path metric rows, 4 scenario summary rows, and 4
leaderboard rows. The aggregate policy failure rate under the configured
thresholds was `0.666667`.

These results validate artifact flow and stress-test plumbing. They do not
forecast market returns, estimate live trading performance, or provide trading
advice.

## Research Question

How does one adaptive policy evaluation workflow behave when the same
fixture-backed policy evidence is passed through historical replay, empirical
bootstrap paths, regime-only Monte Carlo paths, deterministic shock overlays,
and simulation-aware metrics?

## Source Run

| Field | Value |
| --- | --- |
| Case study | `m27_market_simulation_case_study` |
| Milestone | `M27` |
| Source simulation run id | `m27_market_simulation_case_study_b542a0934d5e` |
| Scenario count | `4` |
| Path metric rows | `6` |
| Summary rows | `4` |
| Leaderboard rows | `4` |
| Policy failure rate | `0.666667` |

## Scenario Coverage

| Scenario | Simulation type | Path rows | Policy failure rate |
| --- | --- | ---: | ---: |
| `case_historical_episode` | `historical_episode_replay` | 1 | 0.0 |
| `case_regime_bootstrap` | `regime_block_bootstrap` | 2 | 1.0 |
| `case_regime_transition_mc` | `regime_transition_monte_carlo` | 2 | 0.5 |
| `case_shock_overlay` | `shock_overlay` | 1 | 1.0 |

## Metric Availability

The metrics layer preserves unsupported metric families as blank values instead
of filling in assumptions. This is especially important for regime-only Monte
Carlo paths, where return and policy metrics are intentionally unavailable.

| Scenario | Return rows | Policy rows | Regime rows | Notes |
| --- | ---: | ---: | ---: | --- |
| `case_historical_episode` | 1 | 1 | 0 | `regime_metrics_unavailable` |
| `case_regime_bootstrap` | 2 | 0 | 2 | `policy_metrics_unavailable` |
| `case_regime_transition_mc` | 0 | 0 | 2 | `return_metrics_unavailable;policy_metrics_unavailable` |
| `case_shock_overlay` | 1 | 1 | 0 | `regime_metrics_unavailable` |

## Leaderboard

The leaderboard ranks scenarios by `mean_stress_score`, where lower values rank
better under the default M27 metrics config.

| Rank | Scenario | Type | Ranking value | Decision |
| ---: | --- | --- | ---: | --- |
| 1 | `case_historical_episode` | `historical_episode_replay` | 0.180000 | `monitor` |
| 2 | `case_shock_overlay` | `shock_overlay` | 1.495715 | `review` |
| 3 | `case_regime_transition_mc` | `regime_transition_monte_carlo` | 1.825000 | `review` |
| 4 | `case_regime_bootstrap` | `regime_block_bootstrap` | 3.423631 | `review` |

The best-ranked scenario was `case_historical_episode`. The worst-ranked
scenario was `case_regime_bootstrap`. This ranking is a deterministic stress
ordering for configured fixtures and thresholds, not a prediction of future
market outcomes.

## Scenario Metric Summary

| Scenario | Mean total return | Worst drawdown | Mean volatility | Mean stress share | Mean stress score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `case_historical_episode` | -0.026039 | -0.018000 | 0.012196 |  | 0.180000 |
| `case_regime_bootstrap` | -0.022897 | -0.037688 | 0.014281 | 1.000000 | 3.423631 |
| `case_regime_transition_mc` |  |  |  | 0.562500 | 1.825000 |
| `case_shock_overlay` | -0.073332 | -0.049571 | 0.014737 |  | 1.495715 |

The shock overlay lowered the stressed total return to `-0.073332` for the
replayed episode and triggered the `min_total_return` failure threshold. The
bootstrap paths stayed above the total-return threshold but spent all rows in
configured stress regimes, causing `max_stress_regime_share` failures.

## Path-Level Diagnostics

| Scenario | Path or episode | Total return | Max drawdown | Regime transitions | Stress share | Failure reason | Stress score |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `case_historical_episode` | `volatility_spike_fixture_703b7ea6ad15` | -0.026039 | -0.018000 |  |  | `none` | 0.180000 |
| `case_regime_bootstrap` | `path_000000` | -0.022326 | -0.037688 | 2 | 1.000000 | `max_stress_regime_share` | 3.476880 |
| `case_regime_bootstrap` | `path_000001` | -0.023467 | -0.032038 | 1 | 1.000000 | `max_stress_regime_share` | 3.370382 |
| `case_regime_transition_mc` | `path_000000` |  |  | 4 | 0.500000 | `none` | 1.200000 |
| `case_regime_transition_mc` | `path_000001` |  |  | 4 | 0.625000 | `max_stress_regime_share` | 2.450000 |
| `case_shock_overlay` | `volatility_spike_fixture_703b7ea6ad15` | -0.073332 | -0.049571 |  |  | `min_total_return` | 1.495715 |

## Failure Diagnostics

The configured stress thresholds produced 4 failing path metric rows out of 6.

| Failure reason | Count | Interpretation |
| --- | ---: | --- |
| `max_stress_regime_share` | 3 | Bootstrap and one Monte Carlo path exceeded the configured stress-regime share limit of `0.50`. |
| `min_total_return` | 1 | The shock-overlay episode fell below the configured total-return floor of `-0.05`. |

Thresholds used by the metrics layer:

| Threshold | Value |
| --- | ---: |
| `max_drawdown_limit` | -0.10 |
| `min_total_return` | -0.05 |
| `max_transition_count` | 20.0 |
| `max_stress_regime_share` | 0.50 |
| `max_policy_underperformance` | -0.02 |

## Method Interpretation

Historical replay uses configured fixture episode rows as-is and compares
adaptive policy returns with the static baseline where source columns are
available.

Regime-aware block bootstrap resamples observed contiguous fixture blocks into
deterministic empirical return paths. These paths preserve provenance to source
blocks and support return, drawdown, tail, and regime-transition metrics.

Regime-transition Monte Carlo generates regime-label paths only. It contributes
transition and stress-regime-share metrics while intentionally leaving return
and policy metrics unavailable. The case study does not infer synthetic returns
for Monte Carlo paths.

Shock overlay applies configured deterministic return and confidence
transformations to historical replay output, then feeds the stressed rows into
the same simulation-aware metrics layer.

## Artifact Flow

The runnable script calls the existing market simulation framework through
`configs/regime_stress_tests/m27_market_simulation_case_study.yml`, then
stitches the source framework outputs into the compact case-study directory:

```text
docs/examples/output/m27_market_simulation_case_study/
```

Primary generated artifacts:

- `simulation_summary.json`
- `leaderboard.csv`
- `case_study_report.md`
- `manifest.json`

Source framework artifacts are retained under:

```text
docs/examples/output/m27_market_simulation_case_study/source_simulation_artifacts/
```

## Limitations

- Fixture-backed examples validate workflow plumbing and artifact contracts only.
- Scenario metrics are generated from checked-in fixtures and configured overlays, not live data.
- The leaderboard ranks configured stress artifacts and does not forecast future returns.
- Regime-transition Monte Carlo contributes regime metrics only; returns and policy outcomes are not inferred.
- Policy metrics depend on source artifacts that include adaptive and static policy return columns.

## Recommended Follow-Ups

- Add larger fixture packs that exercise more regime transitions while staying CI-safe.
- Extend overlays to consume file-backed or bootstrap outputs when those framework contracts land.
- Compare multiple candidate policies once source review-pack integration is available.
