# Expected Outputs

Default root:

`docs/examples/output/pipelines/robustness_scenario_sweep/`

Key files:

- `summary.json`
- `workspace/configs/m21_robustness_scenario_sweep.pipeline.yml`
- `workspace/artifacts/pipelines/<pipeline_run_id>/manifest.json`
- `workspace/artifacts/strategies/robustness/<run_id>/metrics_by_config.csv`
- `workspace/artifacts/strategies/robustness/<run_id>/ranked_configs.csv`
- `workspace/artifacts/strategies/robustness/<run_id>/manifest.json`

Behavior checks:

- the pipeline executes a single `research_sweep` step
- sweep artifacts include ranked and per-config outputs
- sweep artifacts include `statistical_validity` metadata in `summary.json`,
  `aggregate_metrics.json`, and `manifest.json`
- row-level sweep outputs include `raw_primary_metric`, correction fields,
  warning codes, and `validity_rank`
- output references remain relative in case-study summaries
