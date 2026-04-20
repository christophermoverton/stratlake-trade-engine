# Expected Outputs

Default root:

`docs/examples/output/pipelines/regime_ensemble_showcase/`

Key files:

- `summary.json`
- `workspace/configs/m22_regime_volatility_regime_momentum.pipeline.yml`
- `workspace/configs/m22_regime_weighted_cross_section_ensemble.pipeline.yml`
- `workspace/artifacts/pipelines/<pipeline_run_id>/manifest.json`
- `workspace/artifacts/strategies/<strategy_run_id>/metrics.json`

Behavior checks:

- both configured profiles complete with status `completed`
- the regime-aware run records `ternary_quantile` semantics
- the ensemble run records `cross_section_rank` semantics
- JSON outputs are sanitized for portability
