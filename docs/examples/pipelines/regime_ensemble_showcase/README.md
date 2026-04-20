# Regime And Ensemble Showcase Pipeline

## Purpose

Demonstrate Milestone 22 regime-aware and ensemble strategy archetypes through the canonical Pipeline Builder flow.

## Components Exercised

- Regime-aware archetype: `volatility_regime_momentum`
- Ensemble archetype: `weighted_cross_section_ensemble`
- Typed signal semantics (`ternary_quantile`, `cross_section_rank`)
- Position constructors (`identity_weights`, `rank_dollar_neutral`)

## Run

```powershell
python docs/examples/pipelines/regime_ensemble_showcase/pipeline.py
```

## Configuration

`config.yml` lists each strategy profile with its strategy parameters, signal semantics, and position constructor.

## Output Interpretation

`summary.json` contains one deterministic run entry per profile, including:

- `pipeline_run_id`
- declared signal semantic
- selected constructor
- pipeline and strategy artifact references
