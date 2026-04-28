# Market Simulation Stress Testing

## Purpose

The M27 market simulation framework is the deterministic artifact layer for adaptive policy stress testing. It validates scenario definitions, resolves seeds, assigns stable scenario and path identifiers, and writes catalogs, inventories, normalized config, and manifests.

This framework is deliberately metadata-only. Historical replay, shock overlays, regime-aware block bootstrap, transition bootstrap, Monte Carlo regime paths, simulation-aware metrics, and the end-to-end case study are reserved for follow-up issues.

## Relationship to M26

M26 regime policy stress testing applies deterministic fixture-friendly transforms to candidate policy metrics. M27 adds the shared scenario framework that later market-simulation methods can use to feed richer stress paths into the same governance style:

- artifact-first outputs
- deterministic IDs and row ordering
- relative persisted paths where possible
- research-only assumptions

It does not alter M26 stress-test semantics or introduce live trading behavior.

## Reserved Simulation Types

The framework recognizes these reserved identifiers:

- `historical_episode_replay`
- `shock_overlay`
- `regime_block_bootstrap`
- `transition_block_bootstrap`
- `regime_transition_monte_carlo`
- `hybrid_simulation`

Unsupported type names fail during config validation, including disabled scenarios. Disabled reserved scenarios are preserved in the catalog with `enabled=false`.

## Config Example

Checked-in example:

```text
configs/regime_stress_tests/m27_market_simulation_framework.yml
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
  - name: volatility_spike_overlay
    type: shock_overlay
    enabled: false
```

Scenario-level `random_seed` or `seed` overrides the global `random_seed`. When no seed is configured, the deterministic default is `0`.

Method-specific future settings belong under `method_config`.

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

The scenario catalog uses stable row ordering and these CSV columns:

```text
scenario_id, scenario_name, simulation_type, enabled, seed, path_count,
source_window_start, source_window_end, uses_historical_data,
uses_synthetic_generation, uses_shock_overlay, source_config_name, notes
```

The manifest records generated files, catalog row counts, scenario counts, enabled/disabled counts, source config metadata, deterministic run metadata, and research-only limitations.

## Deterministic IDs

`generate_scenario_id(...)` hashes the normalized scenario definition, including simulation name, scenario name, simulation type, resolved seed, path count, source windows, and `method_config`.

`generate_path_id(...)` hashes scenario ID, path index, seed, and optional metadata. Path generation is not implemented in this issue; the helper is provided for future simulation methods.

## CLI

```powershell
python -m src.cli.run_market_simulation_scenarios `
  --config configs/regime_stress_tests/m27_market_simulation_framework.yml
```

The CLI validates the config and writes framework artifacts only.

## Non-Goals

This framework does not implement historical replay, path generation, shock mutation logic, simulation-aware metrics, leaderboards, case studies, broker integrations, live feeds, or forecasting claims.
