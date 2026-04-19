# Expected Outputs

Default root:

`docs/examples/output/pipelines/strategy_archetype_showcase/`

Key files:

- `summary.json`
- `workspace/configs/m21_archetype_time_series_momentum.pipeline.yml`
- `workspace/configs/m21_archetype_mean_reversion.pipeline.yml`
- `workspace/configs/m21_archetype_cross_section_momentum.pipeline.yml`
- `workspace/artifacts/pipelines/<pipeline_run_id>/manifest.json` (one per archetype)
- `workspace/artifacts/strategies/<strategy_run_id>/metrics.json` (one per archetype)

Behavior checks:

- all configured archetypes complete with status `completed`
- each run records signal semantic and constructor metadata
- JSON outputs are sanitized for portability
