# Expected Outputs

Default root:

`docs/examples/output/pipelines/resume_reuse/`

Key files:

- `summary.json`
- `resume_reuse_flow/summary.json`
- `resume_reuse_flow/snapshots/partial_summary.json`
- `resume_reuse_flow/snapshots/resumed_summary.json`
- `resume_reuse_flow/snapshots/stable_summary.json`
- `resume_reuse_flow/artifacts/research_campaigns/<campaign_run_id>/checkpoint.json`

Behavior checks:

- partial pass ends with `comparison` in `partial` state
- resumed pass marks `comparison` as `completed` with retry metadata
- stable pass marks prior stages as `reused`
- wrapper summary and source summaries avoid absolute path leakage
