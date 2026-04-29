# Market Simulation Stress Testing

## Purpose

The M27 market simulation framework is the deterministic artifact layer for adaptive policy stress testing. It validates scenario definitions, resolves seeds, assigns stable scenario and path identifiers, and writes catalogs, inventories, normalized config, and manifests.

Historical episode replay, regime-aware block bootstrap, regime-transition Monte Carlo, shock overlays, and simulation-aware stress metrics are the implemented M27 layers. Transition bootstrap as a standalone scenario and the end-to-end case study remain reserved for follow-up issues.

## Relationship to M26

M26 regime policy stress testing applies deterministic fixture-friendly transforms to candidate policy metrics. M27 adds the shared scenario framework that later market-simulation methods can use to feed richer stress paths into the same governance style:

- artifact-first outputs
- deterministic IDs and row ordering
- relative persisted paths where possible
- research-only assumptions

It does not alter M26 stress-test semantics or introduce live trading behavior.

## Simulation Types

The framework recognizes these identifiers:

- `historical_episode_replay`, implemented in Issue #304
- `shock_overlay`, implemented in Issue #305
- `regime_block_bootstrap`, implemented in Issue #306
- `transition_block_bootstrap`
- `regime_transition_monte_carlo`, implemented in Issue #307
- `hybrid_simulation`

Unsupported type names fail during config validation, including disabled scenarios. Disabled reserved scenarios are preserved in the catalog with `enabled=false`.

## Historical Episode Replay

`historical_episode_replay` deterministically replays configured historical market windows from a fixture-backed or research dataset. It is intended for regime-aware policy stress testing against real or curated episodes such as volatility spikes, drawdowns, recoveries, sideways whipsaws, and regime transitions.

Replay is intentionally conservative:

- source rows are selected as-is
- ordering is stable by timestamp, then symbol when a symbol column is configured
- episode windows use inclusive start and inclusive end timestamps; date-only end values include the full UTC date
- no shock overlays, bootstrapping, Monte Carlo generation, live data, broker integration, or forecasting claims are introduced

Required `method_config` fields:

```yaml
method_config:
  dataset_path: tests/fixtures/market_simulation/historical_episode_fixture.csv
  timestamp_column: ts_utc
  return_column: return
  episodes:
    - episode_name: volatility_spike_fixture
      episode_type: volatility_spike
      start: "2026-01-15"
      end: "2026-01-16"
      description: Fixture-backed volatility spike replay window.
```

Optional dataset columns can be configured with:

- `symbol_column`
- `regime_column`
- `confidence_column`
- `entropy_column`
- `adaptive_policy_return_column`
- `static_baseline_return_column`

CSV and Parquet datasets are supported. The timestamp column is parsed strictly, null timestamps fail validation, and the replay runner fails clearly when `timestamp_column` or `return_column` is missing.

Episodes can be configured inline under `method_config.episodes`. A `method_config.episode_catalog_path` can also point to a CSV, JSON, YAML, or YML catalog with equivalent episode fields.

When both adaptive and static policy return columns are available, the replay writes one comparison row per episode with total return, volatility, max drawdown, adaptive-vs-static return delta, confidence, and entropy summaries. When policy return columns are unavailable, artifacts are still emitted and `episode_policy_comparison.csv` sets `comparison_status` to `insufficient_policy_columns` with a concrete reason.

## Shock Overlays

`shock_overlay` applies deterministic stress transformations to replayed market path rows. It is designed as a composable robustness layer for adaptive policy stress testing, not a future-market prediction engine.

An overlay scenario runs after enabled historical replay scenarios in the same config. Its `method_config.input_source` can reference a replay by `scenario_name` or `scenario_id`:

```yaml
method_config:
  input_source:
    type: historical_episode_replay
    scenario_name: historical_volatility_episode
  timestamp_column: ts_utc
  symbol_column: symbol
  base_return_column: source_return
  adaptive_policy_return_column: adaptive_policy_return
  static_baseline_return_column: static_baseline_return
  confidence_column: gmm_confidence
  entropy_column: gmm_entropy
  overlays:
    - name: return_drawdown_shock
      type: return_bps
      columns:
        - source_return
        - adaptive_policy_return
        - static_baseline_return
      bps: -50
    - name: volatility_amplification
      type: volatility_multiplier
      columns:
        - source_return
        - adaptive_policy_return
        - static_baseline_return
      multiplier: 1.50
```

Supported overlay types:

- `return_bps`: adds `bps / 10000.0` to each configured return column.
- `volatility_multiplier`: applies `mean + multiplier * (value - mean)` to amplify or dampen dispersion while preserving the sample mean.
- `transaction_cost_multiplier`: multiplies configured transaction-cost columns.
- `slippage_multiplier`: multiplies configured slippage columns.
- `confidence_multiplier`: multiplies classifier confidence and clamps values to `[0, 1]`.
- `entropy_multiplier`: multiplies entropy and clamps the lower bound at `0`.

Overlays are applied in configured order. The output of one overlay becomes the input to the next for the same `stressed_<column_name>` column. Source columns are preserved, and adjusted values are written to `stressed_*` columns such as `stressed_source_return`, `stressed_gmm_confidence`, and `stressed_gmm_entropy`.

If a configured target column is missing, the default behavior is to fail clearly. An individual overlay can set `missing_column_policy: ignore` to log a skipped row for that column instead.

Enabled shock overlay scenarios write `shock_overlay_config.json`, `shock_overlay_results.csv`, `shock_overlay_log.csv`, `shock_overlay_summary.json`, and `manifest.json`.

`shock_overlay_results.csv` uses a stable schema with source scenario and episode metadata, original return/policy/confidence/entropy columns, stressed counterparts, `overlay_count`, and `overlay_stack`. `shock_overlay_log.csv` records one row per overlay target column with before/after min, max, and mean statistics.

The overlay summary includes stressed adaptive and static total returns when both stressed policy return columns are available. If policy columns are unavailable, artifacts are still written and policy comparison fields are null with `policy_comparison_available=false`.

File input sources are intentionally reserved for a follow-up issue. The implemented source mode consumes historical replay outputs generated in the same run.

## Regime-Aware Block Bootstrap

`regime_block_bootstrap` generates deterministic empirical market-return paths by sampling contiguous source blocks. It differs from historical replay because it creates new path orderings from observed blocks rather than replaying one configured date window. It differs from Monte Carlo because it does not assume Gaussian or parametric returns; every simulated return is copied from the source dataset with provenance fields.

Required `method_config` fields:

```yaml
method_config:
  dataset_path: tests/fixtures/market_simulation/bootstrap_source_fixture.csv
  timestamp_column: ts_utc
  return_column: return
  path_length_bars: 60
  block_length_bars: 5
```

Optional fields:

- `symbol_column`: when provided, source blocks are built within each symbol.
- `regime_column`: required for `regime_bucketed` and `stress_regime` sampling.
- `confidence_column` and `entropy_column`: copied into simulated path rows when present.
- `path_count`: overrides scenario-level `path_count`; precedence is `method_config.path_count > scenario.path_count > 1`.
- `path_start`: deterministic synthetic path start date, defaulting to `2000-01-01 UTC`.

Supported sampling modes:

- `fixed`: sample uniformly from all source blocks.
- `regime_bucketed`: sample blocks whose primary regime is one of `sampling.target_regimes`.
- `transition_window`: sample blocks that contain transition-window rows.
- `stress_regime`: alias-style specialization of `regime_bucketed`, defaulting `target_regimes` to `high_vol` and `stress` when not configured.

Transition windows are tagged when a regime column is available. A row is transition-adjacent when it falls within `sampling.transition_window_bars` bars around a regime-label change. If a symbol column is configured, transitions are computed independently within each symbol.

Transition-aware bootstrap terminology:

- `sampling.mode: transition_window` is used inside `regime_block_bootstrap` and selects source blocks containing regime-transition rows.
- `transition_block_bootstrap` is a standalone future scenario type and is not implemented in M27.

The current implementation supports transition-aware sampling within `regime_block_bootstrap`, but a dedicated `transition_block_bootstrap` scenario remains reserved for future work.

Bootstrap outputs use synthetic `ts_utc` values for path ordering and preserve original timestamps in `source_ts_utc`. Simulated rows also include `sampled_block_id`, `source_regime_label`, `source_symbol`, `source_row_index`, and `is_transition_window` so later stress metrics and shock overlays can trace the source of each bar.

Enabled bootstrap scenarios write:

- `bootstrap_config.json`
- `source_block_catalog.csv`
- `sampled_block_inventory.csv`
- `bootstrap_path_catalog.csv`
- `simulated_return_paths.parquet`
- `simulated_regime_paths.parquet`
- `bootstrap_sampling_summary.json`
- `bootstrap_manifest.json`

## Parquet Output Requirements

Block bootstrap writes:

- `simulated_return_paths.parquet`
- `simulated_regime_paths.parquet`

Monte Carlo writes:

- `monte_carlo_regime_paths.parquet`

These require a Parquet backend such as `pyarrow`.

Ensure your environment includes:

- `pandas`
- `pyarrow`

If unavailable, Parquet writes will fail.

## Bootstrap and Shock Overlay Compatibility

Current behavior:

- Shock overlays consume same-run historical replay outputs only.
- Bootstrap outputs are shaped for future compatibility but are not yet used as overlay inputs.

Future extension, not implemented:

- `input_source.type: file`
- overlay applied to bootstrap-generated paths

## Regime-Transition Monte Carlo

`regime_transition_monte_carlo` generates deterministic synthetic regime-label paths from a validated transition matrix. It differs from historical replay because it does not replay fixed historical windows. It differs from regime block bootstrap because it does not resample empirical return blocks or create simulated return paths; the generated artifact is a regime path suitable for later policy stress metrics and replay integrations.

This scenario produces regime sequences suitable for stress testing and policy evaluation. It does NOT generate simulated return paths or price paths.

Monte Carlo outputs are intended for downstream metrics and policy replay, not direct trading or backtesting without additional modeling layers.

Transition probabilities can be configured inline:

```yaml
market_simulations:
  - name: regime_transition_mc
    type: regime_transition_monte_carlo
    enabled: true
    random_seed: 3107
    path_count: 25
    method_config:
      path_count: 25
      path_length_bars: 120
      path_start: "2000-01-01"
      initial_regime: low_vol
      normalize_transition_rows: false
      transition_matrix:
        low_vol:
          low_vol: 0.70
          high_vol: 0.20
          stress: 0.10
        high_vol:
          low_vol: 0.30
          high_vol: 0.50
          stress: 0.20
        stress:
          low_vol: 0.25
          high_vol: 0.35
          stress: 0.40
      duration_constraints:
        min_duration_bars: 1
        max_duration_bars: 20
```

Rows must be non-empty, numeric, non-negative, and sum to 1.0 within tolerance unless `normalize_transition_rows: true` is set. Missing destinations are treated as zero, but destinations outside the known regime set fail validation.

The empirical matrix source builds transition counts by sorting source data deterministically by `symbol_column` then timestamp, or by timestamp alone when no symbol is configured:

```yaml
method_config:
  path_count: 25
  path_length_bars: 120
  transition_source:
    dataset_path: tests/fixtures/market_simulation/monte_carlo_regime_source_fixture.csv
    timestamp_column: ts_utc
    regime_column: regime_label
    symbol_column: symbol
```

`duration_constraints.min_duration_bars` keeps a path in the current regime until the minimum run length is reached. `duration_constraints.max_duration_bars` forces a transition away from the current regime once the maximum is reached when an alternative regime has positive probability.

Optional transition adjustments are applied after base matrix validation and are renormalized row by row:

```yaml
stress_bias:
  enabled: true
  target_regimes:
    - stress
  multiplier: 1.25
sticky_regime:
  enabled: true
  self_transition_multiplier: 1.10
```

Monte Carlo writes scenario-level artifacts under the standard layout:

```text
artifacts/regime_stress_tests/<simulation_run_id>/market_simulations/<scenario_id>/
```

Generated files:

- `transition_matrix.json`
- `adjusted_transition_matrix.json`, when stress or sticky adjustment is enabled
- `transition_counts.csv`, when the matrix is empirical
- `monte_carlo_regime_paths.parquet`
- `monte_carlo_path_catalog.csv`
- `monte_carlo_summary.json`
- `manifest.json`

`monte_carlo_regime_paths.parquet` includes stable columns for `scenario_id`, `path_id`, `path_index`, `path_step`, `ts_utc`, `regime_label`, `previous_regime_label`, `transitioned`, `duration_in_regime`, `transition_probability`, `seed`, adjustment flags, initial regime, and matrix source. Repeated runs with the same config, matrix, and seed produce identical path IDs, row ordering, paths, summaries, and manifests.

Limitations: this scenario intentionally simulates regime sequences only. It does not implement live trading, broker APIs, real-time feeds, order-book simulation, full return simulation, predictive market forecasting, or the M27 end-to-end case study.

## Monte Carlo and Shock Overlay Compatibility

Current behavior:

- Shock overlays consume same-run historical replay outputs only.
- Monte Carlo outputs are not currently used as overlay inputs.

Future extension, not implemented:

- overlay application to Monte Carlo paths
- `input_source.type: file` support

## Simulation-Aware Stress Metrics

`stress_metrics` is the deterministic evaluation layer for generated simulation artifacts. It runs after historical replay, regime block bootstrap, regime-transition Monte Carlo, and shock overlays in the same framework run. It does not add a new generator or scenario type.

Metrics summarize robustness under the configured replayed or simulated scenarios. They are not forecasts of future performance.

Default behavior:

```yaml
stress_metrics:
  enabled: true
  output_dir_name: simulation_metrics
  failure_thresholds:
    max_drawdown_limit: -0.10
    min_total_return: -0.05
    max_transition_count: 50
    max_stress_regime_share: 0.50
    max_policy_underperformance: -0.02
  leaderboard:
    ranking_metric: mean_stress_score
    ascending: true
  tail_quantile: 0.05
  stress_regimes:
    - stress
```

If the section is absent, safe deterministic defaults are used and metrics are enabled. Set `stress_metrics.enabled: false` to skip metrics artifacts.

Inputs by scenario type:

- Historical replay consumes `episode_policy_comparison.csv` and uses adaptive/static totals, policy deltas, volatility, and drawdown when available.
- Shock overlay consumes `shock_overlay_results.csv` and prefers stressed return columns such as `stressed_source_return`, `stressed_adaptive_policy_return`, and `stressed_static_baseline_return`.
- Regime block bootstrap consumes `simulated_return_paths.parquet`, `simulated_regime_paths.parquet`, and `bootstrap_path_catalog.csv` to compute per-path return and regime metrics.
- Regime-transition Monte Carlo consumes `monte_carlo_regime_paths.parquet` and `monte_carlo_path_catalog.csv` for regime-only metrics. It does not invent returns, so return and policy metric availability flags are false.

Metrics outputs are written under:

```text
artifacts/regime_stress_tests/<simulation_run_id>/market_simulations/simulation_metrics/
```

Generated files:

- `simulation_path_metrics.csv`
- `simulation_summary.csv`
- `simulation_leaderboard.csv`
- `policy_failure_summary.json`
- `simulation_metric_config.json`
- `manifest.json`

`simulation_path_metrics.csv` has one row per replay episode, overlay episode, bootstrap path, or Monte Carlo regime path. Stable columns include scenario identifiers, path or episode IDs, `tail_quantile`, availability flags, return metrics, adaptive-vs-static deltas, regime transition counts, stress regime share, failure reasons, overlay count, and `stress_score`.

`simulation_summary.csv` aggregates each scenario with path counts, row counts, `tail_quantile`, return availability, policy failure rate, adaptive-vs-static win rate, mean transition count, mean stress regime share, worst and mean stress score, and notes for unavailable metric families.

`simulation_leaderboard.csv` ranks scenarios deterministically from `simulation_summary.csv` rows. Because the leaderboard is scenario-summary based, `leaderboard.ranking_metric` must name a numeric summary column such as `mean_stress_score`, `policy_failure_rate`, `tail_quantile_total_return`, or `worst_max_drawdown`.

The default and checked-in M27 example use `mean_stress_score` with `ascending: true`, meaning lower average stress ranks better. Unknown ranking metric names fail clearly during metrics generation. If a known summary metric is unavailable for a specific scenario, that scenario uses deterministic `mean_stress_score` fallback for ordering while preserving the configured `ranking_metric` value in the output.

Ranking uses the configured metric first, then stable tie-breakers:

1. `scenario_name`
2. `simulation_type`
3. `scenario_id`

Tail-risk columns are quantile-aware. Earlier drafts used fixed names such as `tail_5pct_return`; the current schema uses `tail_quantile_return` for path-level period returns and `tail_quantile_total_return` for scenario summary and leaderboard total-return tails. The configured `tail_quantile` is persisted in `simulation_metric_config.json` and repeated in path, summary, and leaderboard rows for interpretability.

Volatile paths and timestamps are not used as tie-breakers.

Policy failure detection is threshold-based. A path is marked as failed when any configured threshold is breached:

- `max_drawdown_limit`: breached when max drawdown is more negative than the limit.
- `min_total_return`: breached when total return is below the limit.
- `max_transition_count`: breached when regime transitions exceed the limit.
- `max_stress_regime_share`: breached when configured stress regimes exceed the share limit.
- `max_policy_underperformance`: breached when adaptive-vs-static return delta is below the limit.

Failure reasons are deterministic semicolon-separated threshold names.

Stress score is intentionally transparent. Higher `stress_score` means a worse stress outcome:

```text
stress_score =
  drawdown_penalty
  + adaptive_underperformance_penalty
  + transition_instability_penalty
  + stress_regime_share_penalty
  + failure_penalty
```

Drawdown, underperformance, transition, and stress-regime penalties are scaled by the configured thresholds. A failed path adds a fixed penalty of `1.0`.

Limitations:

- Metrics compare configured artifacts; they do not claim predictive power.
- Monte Carlo outputs are regime-only until a future return or policy replay layer exists.
- Policy metrics are available only when source artifacts include adaptive and static policy return columns.
- Stress score is a simple ranking aid, not an optimized risk model.

## Config Example

Checked-in example:

```text
configs/regime_stress_tests/m27_market_simulation_framework.yml
configs/regime_stress_tests/m27_historical_episode_replay.yml
configs/regime_stress_tests/m27_shock_overlay.yml
configs/regime_stress_tests/m27_regime_block_bootstrap.yml
configs/regime_stress_tests/m27_regime_transition_monte_carlo.yml
configs/regime_stress_tests/m27_simulation_metrics.yml
```

Minimal shape:

```yaml
simulation_name: m27_market_simulation_stress
output_root: artifacts/regime_stress_tests
random_seed: 1729
source_review_pack: artifacts/regime_reviews/<review_run_id>
baseline_policy: static_baseline
source_policy_candidates:
  - hybrid_calibrated_gmm_policy
market_simulations:
  - name: historical_volatility_episode
    type: historical_episode_replay
    enabled: true
    source_window_start: "2026-01-15"
    source_window_end: "2026-02-15"
    method_config:
      dataset_path: tests/fixtures/market_simulation/historical_episode_fixture.csv
      timestamp_column: ts_utc
      symbol_column: symbol
      return_column: return
      regime_column: regime_label
      confidence_column: gmm_confidence
      entropy_column: gmm_entropy
      adaptive_policy_return_column: adaptive_policy_return
      static_baseline_return_column: static_baseline_return
      episodes:
        - episode_name: volatility_spike_fixture
          episode_type: volatility_spike
          start: "2026-01-15"
          end: "2026-01-16"
  - name: volatility_spike_overlay
    type: shock_overlay
    enabled: true
    method_config:
      input_source:
        type: historical_episode_replay
        scenario_name: historical_volatility_episode
      timestamp_column: ts_utc
      symbol_column: symbol
      base_return_column: source_return
      adaptive_policy_return_column: adaptive_policy_return
      static_baseline_return_column: static_baseline_return
      confidence_column: gmm_confidence
      entropy_column: gmm_entropy
      overlays:
        - name: return_drawdown_shock
          type: return_bps
          columns: [source_return, adaptive_policy_return, static_baseline_return]
          bps: -50
        - name: confidence_degradation
          type: confidence_multiplier
          column: gmm_confidence
          multiplier: 0.70
```

Scenario-level `random_seed` or `seed` overrides the global `random_seed`. When no seed is configured, the deterministic default is `0`.

Method-specific settings belong under `method_config`.

## Artifacts

Outputs are written under:

```text
artifacts/regime_stress_tests/<simulation_run_id>/market_simulations/
```

Generated files:

- `scenario_catalog.csv`
- `scenario_catalog.json`
- `simulation_config.json`
- `input_inventory.json`
- `simulation_manifest.json`

Enabled historical replay scenarios also write scenario-level artifacts under:

```text
artifacts/regime_stress_tests/<simulation_run_id>/market_simulations/<scenario_id>/
```

Generated replay files:

- `historical_episode_catalog.csv`
- `episode_replay_results.csv`
- `episode_policy_comparison.csv`
- `episode_summary.json`
- `manifest.json`

Enabled shock overlay scenarios write scenario-level artifacts under the same directory layout.

Generated shock overlay files:

- `shock_overlay_config.json`
- `shock_overlay_results.csv`
- `shock_overlay_log.csv`
- `shock_overlay_summary.json`
- `manifest.json`

Enabled regime block bootstrap scenarios write scenario-level artifacts under the same directory layout.

Generated bootstrap files:

- `bootstrap_config.json`
- `source_block_catalog.csv`
- `sampled_block_inventory.csv`
- `bootstrap_path_catalog.csv`
- `simulated_return_paths.parquet`
- `simulated_regime_paths.parquet`
- `bootstrap_sampling_summary.json`
- `bootstrap_manifest.json`

The scenario catalog uses stable row ordering and these CSV columns:

```text
scenario_id, scenario_name, simulation_type, enabled, seed, path_count,
source_window_start, source_window_end, uses_historical_data,
uses_synthetic_generation, uses_shock_overlay, source_config_name, notes
```

The manifest records generated files, catalog row counts, scenario counts, enabled/disabled counts, source config metadata, deterministic run metadata, and research-only limitations.

The historical episode catalog records scenario and episode IDs, episode names and types, inclusive window bounds, sanitized dataset path, row/symbol/regime counts, optional column availability, and descriptions.

Replay results use a stable schema:

```text
scenario_id, episode_id, episode_name, ts_utc, symbol, source_return,
regime_label, gmm_confidence, gmm_entropy, adaptive_policy_return,
static_baseline_return, adaptive_vs_static_return_delta
```

Unavailable optional columns are emitted as blank values in the stable CSV schema.

## Deterministic IDs

`generate_scenario_id(...)` hashes the normalized scenario definition, including simulation name, scenario name, simulation type, resolved seed, path count, source windows, and `method_config`.

`generate_path_id(...)` hashes scenario ID, path index, seed, and optional metadata. Regime block bootstrap uses it with normalized sampling metadata so repeated runs produce stable path IDs.

## CLI

```powershell
python -m src.cli.run_market_simulation_scenarios `
  --config configs/regime_stress_tests/m27_market_simulation_framework.yml
```

The CLI validates the config and writes framework artifacts. Enabled historical replay scenarios also produce their scenario-level replay artifacts in the same run.

For the implemented historical replay example:

```powershell
python -m src.cli.run_market_simulation_scenarios `
  --config configs/regime_stress_tests/m27_historical_episode_replay.yml
```

The CLI writes both framework artifacts and scenario-level historical replay artifacts.

For the implemented shock overlay example:

```powershell
python -m src.cli.run_market_simulation_scenarios `
  --config configs/regime_stress_tests/m27_shock_overlay.yml
```

The CLI writes framework artifacts, the referenced historical replay artifacts, and scenario-level shock overlay artifacts.

For the implemented regime block bootstrap example:

```powershell
python -m src.cli.run_market_simulation_scenarios `
  --config configs/regime_stress_tests/m27_regime_block_bootstrap.yml
```

The CLI writes framework artifacts and scenario-level bootstrap artifacts.

For the simulation-aware metrics example:

```powershell
python -m src.cli.run_market_simulation_scenarios `
  --config configs/regime_stress_tests/m27_simulation_metrics.yml
```

The CLI writes all supported scenario artifacts and the run-level `simulation_metrics/` directory. The summary reports the metrics output directory, path metric rows, summary rows, leaderboard rows, and policy failure rate.

## Non-Goals

This issue does not implement file-based overlay inputs, transition bootstrap as a separate scenario type, Monte Carlo return-path generation, case studies, broker integrations, live feeds, order-book simulation, or forecasting claims.
