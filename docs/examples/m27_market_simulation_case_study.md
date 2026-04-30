# M27 Market Simulation Stress Testing Case Study

## Purpose

This example is the flagship Milestone 27 workflow for deterministic adaptive
policy stress testing. It stitches together the implemented market simulation
layers:

- historical episode replay
- regime-aware block bootstrap
- regime-transition Monte Carlo paths
- deterministic shock overlays
- simulation-aware stress metrics and leaderboard ranking

The case study uses checked-in fixture data and existing framework APIs. It is
for research workflow validation only, not forecasting or trading advice.

## Run It

```powershell
python docs/examples/m27_market_simulation_case_study.py
```

## Expected Outputs

The script writes compact stitched outputs to:

```text
docs/examples/output/m27_market_simulation_case_study/
```

Primary files:

- `simulation_summary.json`
- `leaderboard.csv`
- `case_study_report.md`
- `manifest.json`

A stable docs copy of the generated report is available at
`docs/examples/m27_market_simulation_case_study_report.md`.

The framework source artifacts used to build those files are written under:

```text
docs/examples/output/m27_market_simulation_case_study/source_simulation_artifacts/
```

The source `simulation_metrics/` directory uses the canonical M27 metrics
artifact names:

- `simulation_path_metrics.csv`
- `simulation_summary.csv`
- `simulation_leaderboard.csv`
- `policy_failure_summary.json`
- `simulation_metric_config.json`
- `manifest.json`

## Interpretation Guardrails

The leaderboard ranks deterministic stress-test artifacts under the configured
thresholds. Lower `mean_stress_score` ranks better in the default config. The
regime-transition Monte Carlo scenario is regime-only; it does not fabricate
returns, prices, or policy outcomes. All outputs are fixture-backed examples and
do not predict future market behavior.

## Report

See [m27_market_simulation_case_study_report.md](m27_market_simulation_case_study_report.md)
for the generated case-study interpretation, leaderboard summary, limitations,
and recommended follow-ups.
