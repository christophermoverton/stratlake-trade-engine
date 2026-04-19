# Robustness and Scenario Sweeps Pipeline

## Purpose

Demonstrate Milestone 21 sweep expansion across signal semantics, constructor selection, and asymmetry controls.

## Components Exercised

- strategy archetype: `cross_section_momentum`
- signal sweep: `cross_section_rank`, `binary_signal`
- constructor sweep: `rank_dollar_neutral`, `identity_weights`
- asymmetry sweep: `exclude_short` toggles
- deterministic ranking: Sharpe ratio primary metric, total return tie-breaker

## Run

```powershell
python docs/examples/pipelines/robustness_scenario_sweep/pipeline.py
```

## Configuration

`config.yml` uses builder-native `sweep` structure and emits a standard robustness run under the M20 pipeline runner.

## Output Interpretation

- `metrics_by_config.csv` is the primary per-configuration matrix
- `ranked_configs.csv` is deterministic ranking output
- `summary.json` captures the full run and artifact references
