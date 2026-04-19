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
- output references remain relative in case-study summaries
