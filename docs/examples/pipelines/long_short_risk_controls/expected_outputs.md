# Expected Outputs

Default root:

`docs/examples/output/pipelines/long_short_risk_controls/`

Key files:

- `summary.json`
- `workspace/configs/m21_long_short_risk_controls.pipeline.yml`
- `workspace/artifacts/pipelines/<pipeline_run_id>/manifest.json`
- `workspace/artifacts/strategies/<strategy_run_id>/manifest.json`

Behavior checks:

- run completes with deterministic single-step execution order
- asymmetry parameters are persisted in the summary and strategy signal semantics metadata
- JSON outputs are sanitized to avoid absolute path leakage
