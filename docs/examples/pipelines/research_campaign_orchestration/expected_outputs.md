# Expected Outputs

Default root:

`docs/examples/output/pipelines/research_campaign_orchestration/`

Key files:

- `summary.json`
- `orchestrated_campaign/summary.json`
- `orchestrated_campaign/artifacts/research_campaigns/<orchestration_run_id>/summary.json`
- `orchestrated_campaign/artifacts/research_campaigns/<orchestration_run_id>/scenario_catalog.json`
- `orchestrated_campaign/artifacts/research_campaigns/<orchestration_run_id>/scenario_matrix.csv`
- `orchestrated_campaign/artifacts/research_campaigns/<orchestration_run_id>/scenario_matrix.json`

Behavior checks:

- first and second orchestration pass complete successfully
- scenario matrix outputs are materialized and interpretable
- JSON outputs in the canonical wrapper root are sanitized
