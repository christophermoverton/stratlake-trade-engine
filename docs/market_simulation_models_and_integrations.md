# Market Simulation Models and Integrations

## Purpose

Milestone 27 adds deterministic market simulation stress-testing workflows for
research governance. The stack is artifact-first: configs define scenarios,
scenario runners write stable source artifacts, the metrics layer normalizes
scenario evidence, and optional M26 integration consumes those metrics without
changing deterministic adaptive policy stress semantics.

M27 evidence complements Issue #299 deterministic adaptive policy stress tests.
It does not replace deterministic regime shock, whipsaw, classifier
uncertainty, taxonomy/ML disagreement, fallback, turnover, or
adaptive-vs-static diagnostics.

## Implemented M27 Layers

| Layer | Purpose | Primary artifacts |
| --- | --- | --- |
| Scenario framework | Validates configs, resolves deterministic seeds, assigns stable scenario/path IDs, and writes run catalogs. | `scenario_catalog.csv`, `scenario_catalog.json`, `simulation_config.json`, `simulation_manifest.json`, `input_inventory.json` |
| Historical episode replay | Replays configured fixture or research windows as ordered source rows. | `episode_policy_comparison.csv`, `manifest.json` |
| Shock overlays | Applies deterministic return, cost, slippage, confidence, or entropy transforms to same-run replay artifacts. | `shock_overlay_results.csv`, `shock_overlay_log.csv`, `shock_overlay_summary.json`, `manifest.json` |
| Regime-aware block bootstrap | Samples contiguous observed blocks into empirical return and regime paths with source provenance. | `source_block_catalog.csv`, `sampled_block_inventory.csv`, `bootstrap_path_catalog.csv`, `simulated_return_paths.parquet`, `simulated_regime_paths.parquet`, `bootstrap_manifest.json` |
| Regime-transition Monte Carlo | Generates deterministic regime-label paths from configured transition probabilities. | `monte_carlo_path_catalog.csv`, `monte_carlo_regime_paths.parquet`, `manifest.json` |
| Simulation-aware metrics | Normalizes replay, overlay, bootstrap, and Monte Carlo evidence into path metrics, scenario summaries, policy failure summaries, and deterministic leaderboards. | `simulation_path_metrics.csv`, `simulation_summary.csv`, `simulation_leaderboard.csv`, `policy_failure_summary.json`, `simulation_metric_config.json`, `manifest.json` |

The M27 case-study workflow stitches these layers through
`docs/examples/m27_market_simulation_case_study.py` and writes compact checked-in
outputs under `docs/examples/output/m27_market_simulation_case_study/`.

## Canonical Metrics Artifacts

The canonical M27 metrics directory is:

```text
artifacts/regime_stress_tests/<simulation_run_id>/market_simulations/simulation_metrics/
```

The fixture-backed M27 case study writes the same directory shape under its
ignored source-artifact root:

```text
docs/examples/output/m27_market_simulation_case_study/source_simulation_artifacts/<run_id>/market_simulations/simulation_metrics/
```

Expected metrics files:

* `simulation_path_metrics.csv`
* `simulation_summary.csv`
* `simulation_leaderboard.csv`
* `policy_failure_summary.json`
* `simulation_metric_config.json`
* `manifest.json`

CSV row ordering is deterministic. JSON manifests and summaries persist relative
paths where possible so checked-in fixture outputs stay portable.

## M26/M27 Integration Bridge

M26 adaptive policy stress tests can optionally consume M27 metrics through the
`market_simulation_stress` config block.

Disabled or absent config is a no-op:

```yaml
market_simulation_stress:
  enabled: false
```

`existing_artifacts` mode validates and consumes a prebuilt M27
`simulation_metrics/` directory:

```yaml
market_simulation_stress:
  enabled: true
  mode: existing_artifacts
  simulation_metrics_dir: docs/examples/output/m27_market_simulation_case_study/source_simulation_artifacts/<run_id>/market_simulations/simulation_metrics
  include_in_policy_stress_summary: true
  include_in_case_study_report: true
```

`run_config` mode runs a fixture-backed M27 config through the existing M27
execution stack, then consumes its metrics artifacts:

```yaml
market_simulation_stress:
  enabled: true
  mode: run_config
  config_path: configs/regime_stress_tests/m27_market_simulation_case_study.yml
```

When market simulation evidence is enabled and available, M26 stress outputs
include:

* `market_simulation_stress_summary.json`
* `market_simulation_stress_leaderboard.csv`

The summary records `market_simulation_available`, row counts, source run ID,
simulation types, policy failure rate, best/worst ranked scenarios, source
artifact paths, and the regime-only Monte Carlo limitation note.

## Optional and Strict Behavior

The full-year M26 regime policy benchmark case study treats M27 evidence as
optional by default. If the canonical M27 fixture metrics are absent, the script
still completes, writes `market_simulation_available=false`, marks the mode as
`not_available`, records zero metric row counts, and emits a schema-only empty
`market_simulation_stress_leaderboard.csv`.

Use:

```powershell
python docs\examples\full_year_regime_policy_benchmark_case_study.py --require-market-simulation-stress
```

to make missing M27 evidence fail clearly.

## Interpretation Guardrails

* Fixture-backed examples validate workflow plumbing and artifact contracts only.
* Deterministic M26 stress transforms are governance diagnostics, not empirical
  market simulations.
* M27 evidence complements, but does not replace, Issue #299 deterministic
  adaptive policy stress tests.
* Regime-transition Monte Carlo remains regime-only unless return or policy
  replay artifacts are explicitly available.
* Simulation-aware metrics summarize configured replayed or simulated artifacts;
  they are not forecasts.
* Market simulation outputs are research diagnostics, not trading
  recommendations.

