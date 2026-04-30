# M27 Market Simulation Case Study

## Purpose

This case study demonstrates the Milestone 27 market simulation
stress-testing workflow end-to-end. It runs all configured simulation
scenarios (episode replay, market shock overlay, regime-aware block
bootstrap, and regime-transition Monte Carlo), computes simulation-aware
stress metrics, and stitches those metrics into a stitched case-study
report.

The case study is fixture-backed and deterministic. It does not require live
market data and produces the same output across environments given the same
config.

## Prerequisites

No live data or broker connections are required. The script reads checked-in
fixture CSVs and YAML configs from the repository.

## How To Run

```bash
python docs/examples/m27_market_simulation_case_study.py
```

Run from the repository root so that relative config and artifact paths
resolve correctly.

## Config Files

| File | Purpose |
|---|---|
| `configs/regime_stress_tests/m27_market_simulation_case_study.yml` | Scenario framework config |
| `configs/regime_stress_tests/m27_simulation_metrics.yml` | Simulation metrics and failure threshold config |

## Output Root

```text
docs/examples/output/m27_market_simulation_case_study/
```

Simulation source artifacts are written under a timestamped run directory:

```text
artifacts/market_simulation/<run_id>/
```

## Generated Artifacts

| Artifact | Location | Description |
|---|---|---|
| `simulation_summary.json` | output root | stitched cross-scenario summary |
| `simulation_leaderboard.csv` | output root | policy ranking by tail-risk |
| `policy_failure_summary.json` | output root | per-policy failure rates |
| `scenario_summaries/` | output root | per-scenario metric summaries |
| `manifest.json` | output root | stitched artifact manifest |
| `report.md` | output root | human-readable case-study report |
| `simulation_path_metrics.csv` | `artifacts/market_simulation/<run_id>/simulation_metrics/` | per-path metrics |
| `simulation_summary.csv` | `artifacts/market_simulation/<run_id>/simulation_metrics/` | per-scenario summaries |
| `simulation_leaderboard.csv` | `artifacts/market_simulation/<run_id>/simulation_metrics/` | source leaderboard |
| `policy_failure_summary.json` | `artifacts/market_simulation/<run_id>/simulation_metrics/` | source failure summary |
| `simulation_metric_config.json` | `artifacts/market_simulation/<run_id>/simulation_metrics/` | metrics config snapshot |

## Scenario Coverage

The checked-in config runs four scenario types:

| Scenario Type | Model | Purpose |
|---|---|---|
| Episode replay | Historical Episode Replay | real observed windows as stress paths |
| Shock overlay | Market Shock Overlay | return-bps and confidence shocks on known conditions |
| Block bootstrap | Regime-Aware Block Bootstrap | regime-respecting resampled paths |
| Monte Carlo | Regime-Transition Monte Carlo | hypothetical regime-sequence paths |

See [docs/market_simulation_models_and_integrations.md](../market_simulation_models_and_integrations.md)
for a detailed explanation of each model.

## Simulation Metrics and Leaderboard

After all scenarios complete, the metrics layer produces:

- per-path stress scores and policy failure flags
- per-scenario tail-risk quantiles (5th/10th percentile)
- cross-scenario policy leaderboard ranked by worst-case tail behavior

Monte Carlo paths have no return metrics. Return-metric columns are `null` for
Monte Carlo rows in `simulation_path_metrics.csv`.

See
[Simulation-Aware Stress Metrics](../market_simulation_models_and_integrations.md#6-simulation-aware-stress-metrics)
for column definitions and threshold interpretation.

## Interpretation Guardrails

- all paths are fixture-backed synthetic paths, not live or forward-looking data
- leaderboard rankings reflect configured stress scenarios, not unconditional performance
- Monte Carlo paths probe regime-transition dynamics only; no return claims are made
- case-study outputs are research governance artifacts, not trading signals

## Connecting M27 Evidence to M26 Integration

After running this case study, point the M26 full-year case study bridge to the
simulation metrics output:

```yaml
market_simulation_stress:
  enabled: true
  mode: existing_artifacts
  simulation_metrics_dir: artifacts/market_simulation/<run_id>/simulation_metrics
  include_in_policy_stress_summary: true
  include_in_case_study_report: true
```

See [docs/market_simulation_models_and_integrations.md — M26 Integration Bridge](../market_simulation_models_and_integrations.md#8-m26-integration-bridge)
and
[docs/examples/full_year_regime_policy_benchmark_case_study.md](full_year_regime_policy_benchmark_case_study.md)
for the full M26 integration workflow.
