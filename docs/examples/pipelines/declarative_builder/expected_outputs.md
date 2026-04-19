# Expected Outputs

Default root:

`docs/examples/output/pipelines/declarative_builder/`

Key files:

- `summary.json`
- `workspace/configs/m21_declarative_builder.pipeline.yml`
- `workspace/configs/m21_declarative_builder.strategy.builder.yml`
- `workspace/configs/m21_declarative_builder.portfolio.builder.yml`
- `workspace/artifacts/pipelines/<pipeline_run_id>/manifest.json`
- `workspace/artifacts/strategies/<strategy_run_id>/manifest.json`
- `workspace/artifacts/portfolios/<portfolio_run_id>/manifest.json`

Behavior checks:

- generated execution order includes both `run_strategy` and `run_portfolio`
- declarative and imperative composition produce equivalent generated YAML
- summaries avoid absolute path leakage

Sweep-plus-portfolio checks (when configured):

- generated execution order includes both `research_sweep` and `run_portfolio`
- downstream portfolio step uses `--from-sweep-top-ranked`
- sweep artifact directory contains `ranked_configs.csv` and `runs/<variant_id>/`
