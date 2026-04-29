# Market Simulation Stress Testing

## Purpose

The M27 market simulation framework is the deterministic artifact layer for adaptive policy stress testing. It validates scenario definitions, resolves seeds, assigns stable scenario and path identifiers, and writes catalogs, inventories, normalized config, and manifests.

Historical episode replay, regime-aware block bootstrap, and shock overlays are the implemented simulation scenario types. Transition bootstrap as a standalone scenario, Monte Carlo regime paths, simulation-aware leaderboards, and the end-to-end case study remain reserved for follow-up issues.

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
- `regime_transition_monte_carlo`
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

## Config Example

Checked-in example:

```text
configs/regime_stress_tests/m27_market_simulation_framework.yml
configs/regime_stress_tests/m27_historical_episode_replay.yml
configs/regime_stress_tests/m27_shock_overlay.yml
configs/regime_stress_tests/m27_regime_block_bootstrap.yml
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

## Non-Goals

This issue does not implement file-based overlay inputs, transition bootstrap as a separate scenario type, Monte Carlo path generation, simulation-aware stress leaderboards, case studies, broker integrations, live feeds, order-book simulation, or forecasting claims.
