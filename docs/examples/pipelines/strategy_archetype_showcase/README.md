# Strategy Archetype Showcase Pipeline

## Purpose

Demonstrate multiple Milestone 21 strategy archetypes running through the same canonical composition layers.

## Components Exercised

- Trend archetype: `time_series_momentum`
- Mean reversion archetype: `mean_reversion`
- Cross-sectional archetype: `cross_section_momentum`
- Signal semantics per archetype (`ternary_quantile`, `cross_section_rank`)
- Position constructors (`identity_weights`, `rank_dollar_neutral`)

## Run

```powershell
python docs/examples/pipelines/strategy_archetype_showcase/pipeline.py
```

## Configuration

`config.yml` enumerates each archetype profile. The script loops through these entries and runs each as a builder-generated pipeline.

## Output Interpretation

`summary.json` contains one run entry per archetype, including:

- deterministic `pipeline_run_id`
- selected signal semantic
- selected position constructor
- pipeline and strategy artifact references
