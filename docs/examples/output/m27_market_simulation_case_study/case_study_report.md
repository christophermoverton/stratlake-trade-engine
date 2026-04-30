# M27 Market Simulation Stress Testing Case Study

## Research Question

How does one adaptive policy evaluation workflow behave when the same fixture-backed policy evidence is passed through historical replay, empirical bootstrap paths, regime-only Monte Carlo paths, deterministic shock overlays, and simulation-aware metrics?

## Scenario Methods

The case study ran `m27_market_simulation_case_study_b542a0934d5e` with 4 configured scenarios covering: historical_episode_replay, regime_block_bootstrap, regime_transition_monte_carlo, shock_overlay.

- `case_historical_episode` (historical_episode_replay): 1 path row(s), policy failure rate 0.0.
- `case_regime_bootstrap` (regime_block_bootstrap): 2 path row(s), policy failure rate 1.0.
- `case_regime_transition_mc` (regime_transition_monte_carlo): 2 path row(s), policy failure rate 0.5.
- `case_shock_overlay` (shock_overlay): 1 path row(s), policy failure rate 1.0.

## Artifact Flow

The script calls the existing market simulation framework, then stitches the framework artifacts into a compact case-study output directory. Source artifacts remain under `source_simulation_artifacts/`; the root directory keeps only the summary, leaderboard, report, and manifest.

## Generated Outputs

- `simulation_summary.json`
- `leaderboard.csv`
- `case_study_report.md`
- `manifest.json`

## Leaderboard Interpretation

The leaderboard ranks scenarios by `mean_stress_score`. The best-ranked scenario is `case_historical_episode` (historical_episode_replay) with value 0.180000. The worst-ranked scenario is `case_regime_bootstrap` (regime_block_bootstrap) with value 3.423631.

| Rank | Scenario | Type | Ranking Value | Decision |
| --- | --- | --- | ---: | --- |
| 1 | `case_historical_episode` | `historical_episode_replay` | 0.180000 | `monitor` |
| 2 | `case_shock_overlay` | `shock_overlay` | 1.495715 | `review` |
| 3 | `case_regime_transition_mc` | `regime_transition_monte_carlo` | 1.825000 | `review` |
| 4 | `case_regime_bootstrap` | `regime_block_bootstrap` | 3.423631 | `review` |

## Scenario Metric Summary

This table shows the scenario-level metrics used by the leaderboard and diagnostics. Blank values indicate that the source artifact does not support that metric family.

| Scenario | Return rows | Policy rows | Regime rows | Mean total return | Worst drawdown | Mean stress share | Mean stress score | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `case_historical_episode` | 1 | 1 | 0 | -0.026039 | -0.018000 |  | 0.180000 | `regime_metrics_unavailable` |
| `case_regime_bootstrap` | 2 | 0 | 2 | -0.022897 | -0.037688 | 1.000000 | 3.423631 | `policy_metrics_unavailable` |
| `case_regime_transition_mc` | 0 | 0 | 2 |  |  | 0.562500 | 1.825000 | `return_metrics_unavailable;policy_metrics_unavailable` |
| `case_shock_overlay` | 1 | 1 | 0 | -0.073332 | -0.049571 |  | 1.495715 | `regime_metrics_unavailable` |

## Path-Level Diagnostics

The metrics layer emits one row per replay episode, overlay episode, bootstrap path, or Monte Carlo regime path. This compact view keeps the most actionable columns.

| Scenario | Path/Episode | Total return | Max drawdown | Regime transitions | Stress share | Failure | Reason | Stress score |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `case_historical_episode` | `volatility_spike_fixture_703b7ea6ad15` | -0.026039 | -0.018000 |  |  | `False` | `none` | 0.180000 |
| `case_regime_bootstrap` | `case_regime_bootstrap_2bf0fc923c47_path_000000_e68e7c219b5f` | -0.022326 | -0.037688 | 2 | 1.000000 | `True` | `max_stress_regime_share` | 3.476880 |
| `case_regime_bootstrap` | `case_regime_bootstrap_2bf0fc923c47_path_000001_864a32f31e0d` | -0.023467 | -0.032038 | 1 | 1.000000 | `True` | `max_stress_regime_share` | 3.370382 |
| `case_regime_transition_mc` | `case_regime_transition_mc_e83001a03dff_path_000000_feb5b26ee7e8` |  |  | 4 | 0.500000 | `False` | `none` | 1.200000 |
| `case_regime_transition_mc` | `case_regime_transition_mc_e83001a03dff_path_000001_30cb53b3c990` |  |  | 4 | 0.625000 | `True` | `max_stress_regime_share` | 2.450000 |
| `case_shock_overlay` | `volatility_spike_fixture_703b7ea6ad15` | -0.073332 | -0.049571 |  |  | `True` | `min_total_return` | 1.495715 |

## Failure Diagnostics

4 of 6 path metric rows breached at least one configured threshold.

| Failure reason | Count |
| --- | ---: |
| `max_stress_regime_share` | 3 |
| `min_total_return` | 1 |

## Historical Replay Interpretation

Historical replay uses the configured fixture episode rows as-is and compares adaptive policy returns with the static baseline where source columns are available. This is a deterministic replay of checked-in fixture data, not evidence of future performance.

## Block Bootstrap Interpretation

The regime-aware block bootstrap resamples observed contiguous fixture blocks into deterministic empirical return paths. These paths preserve provenance to source blocks and support return, drawdown, tail, and regime-transition metrics.

## Monte Carlo Interpretation

The regime-transition Monte Carlo scenario generates regime-label paths only. It contributes transition and stress-regime-share metrics, and intentionally leaves return and policy metrics unavailable instead of fabricating them.

## Shock Overlay Interpretation

The shock overlay applies configured deterministic return and confidence transformations to the historical replay output. It shows how overlay artifacts feed the same metrics layer without introducing live data or forecasting assumptions.

## Simulation-Aware Metrics Interpretation

The metrics layer produced 6 path metric rows, 4 scenario summary rows, and 4 leaderboard rows. The aggregate policy failure rate is 0.666667 under the configured thresholds.

## Limitations

- Fixture-backed examples validate workflow plumbing and artifact contracts only.
- Simulated paths do not forecast future returns or recommend trades.
- Monte Carlo outputs are regime-only and do not contain synthetic returns.
- Policy metrics depend on source artifacts that include adaptive and static policy columns.

## Recommended Follow-Ups

- Add larger fixture packs that exercise more regime transitions while staying CI-safe.
- Extend overlays to consume file-backed or bootstrap outputs when those framework contracts land.
- Compare multiple candidate policies once source review-pack integration is available.

## Manifest

See `docs/examples/output/m27_market_simulation_case_study/manifest.json` for relative source and output paths.
