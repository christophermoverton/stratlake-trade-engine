# Expected Outputs

Default root:

`docs/examples/output/pipelines/baseline_reference/`

Key files:

- `summary.json`
- `workspace/configs/m21_baseline_reference.pipeline.yml`
- `workspace/artifacts/pipelines/<pipeline_run_id>/manifest.json`
- `workspace/artifacts/strategies/<strategy_run_id>/manifest.json`
- `workspace/artifacts/strategies/<strategy_run_id>/metrics.json`

Behavior checks:

- two consecutive runs produce the same `pipeline_run_id`
- the pipeline has a single strategy step (`run_strategy`)
- all JSON outputs in the example root are sanitized to avoid absolute path leakage
