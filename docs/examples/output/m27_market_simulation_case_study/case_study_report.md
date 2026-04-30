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
