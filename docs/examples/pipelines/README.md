# Canonical Orchestration Pipelines (M21.7)

This directory is the canonical reference library for Milestone 21 composition.

Each example is runnable and demonstrates one or more integration points across:

- signal semantics
- position constructors
- strategy archetypes
- long/short asymmetry and risk controls
- robustness sweeps and scenario/campaign orchestration
- resume/reuse semantics
- declarative pipeline construction via Pipeline Builder

## Pipeline Library

- `baseline_reference/`
  Baseline signal semantics -> constructor -> strategy execution flow.
- `strategy_archetype_showcase/`
  Multiple archetypes (trend, mean reversion, cross-sectional) under one canonical execution pattern.
- `regime_ensemble_showcase/`
  Regime-aware and ensemble archetypes under the same builder-driven execution path.
- `long_short_risk_controls/`
  Directional asymmetry and constructor-level risk controls.
- `robustness_scenario_sweep/`
  Parameter and semantic sweep expansion with ranked comparable outputs.
- `research_campaign_orchestration/`
  Multi-run orchestration with scenario matrix and aggregation outputs.
- `resume_reuse/`
  Deterministic checkpoint, retry, and reuse behavior.
- `declarative_builder/`
  Config-defined execution path through the Pipeline Builder and equivalence to imperative composition.

## Run Examples

Run from repository root with relative paths:

```powershell
python docs/examples/pipelines/baseline_reference/pipeline.py
python docs/examples/pipelines/strategy_archetype_showcase/pipeline.py
python docs/examples/pipelines/regime_ensemble_showcase/pipeline.py
python docs/examples/pipelines/long_short_risk_controls/pipeline.py
python docs/examples/pipelines/robustness_scenario_sweep/pipeline.py
python docs/examples/pipelines/research_campaign_orchestration/pipeline.py
python docs/examples/pipelines/resume_reuse/pipeline.py
python docs/examples/pipelines/declarative_builder/pipeline.py
```

All scripts write outputs under `docs/examples/output/pipelines/` by default.
