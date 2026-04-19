# Baseline Reference Pipeline

## Purpose

This is the canonical minimal M21 composition path:

1. strategy archetype emits typed signal semantics
2. signal semantics resolve to a compatible position constructor
3. constructor output feeds deterministic strategy execution artifacts

## Components Exercised

- Signal semantics: `cross_section_rank`
- Position construction: `rank_dollar_neutral`
- Strategy archetype: `cross_section_momentum`
- Strict mode: enabled (`strict: true`)

## Run

From repository root:

```powershell
python docs/examples/pipelines/baseline_reference/pipeline.py
```

## Configuration

See `config.yml` for the declarative builder mapping used by this example.

## Execution Flow

1. Build deterministic synthetic `features_daily` fixture inside an isolated workspace.
2. Resolve registry-backed strategy/signal/constructor through `PipelineBuilder`.
3. Generate and run one M20 pipeline with `src.cli.run_builder_strategy`.
4. Rerun the same config to confirm deterministic pipeline run id behavior.

## Output Interpretation

- `summary.json` records the deterministic rerun check.
- Pipeline manifest confirms step ordering and artifact lineage.
- Strategy artifact directory contains metrics, QA summary, `signal_semantics.json`, and manifest.
