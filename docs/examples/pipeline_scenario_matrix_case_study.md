# Pipeline Scenario-Matrix Case Study

## Objective

Recreate the scenario-matrix research workflow with the YAML-driven
`PipelineRunner` instead of the campaign orchestrator.

This pipeline keeps the workflow explicit and deterministic:

1. run multiple alpha models
2. run strategy baselines
3. write alpha and strategy comparison outputs
4. select candidates from alpha artifacts
5. construct a portfolio from the selected candidates
6. persist pipeline-level lineage, manifest, metrics, and state artifacts

## Pipeline Config

Shipped config:

```text
configs/pipelines/scenario_matrix_pipeline.yml
```

The pipeline uses only `python_module` steps and invokes existing CLI modules
through the standard `module.run_cli(argv)` adapter path.

Excerpt:

```yaml
id: scenario_matrix_pipeline
steps:
  - id: alpha_rank_composite
    adapter: python_module
    module: src.cli.run_alpha
    argv:
      - --config
      - configs/alphas_2026_q1.yml
      - --alpha-name
      - rank_composite_momentum_2026_q1
      - --mode
      - full
      - --artifacts-root
      - docs/examples/output/pipeline_scenario_matrix_case_study/artifacts/alpha

  - id: compare_alpha
    depends_on:
      - alpha_rank_composite
      - alpha_xgb
      - alpha_lgbm
    adapter: python_module
    module: src.cli.compare_alpha
    argv:
      - --from-registry
      - --view
      - combined
      - --metric
      - ic_ir
      - --output-path
      - docs/examples/output/pipeline_scenario_matrix_case_study/artifacts/comparisons/alpha

  - id: candidate_selection
    depends_on:
      - compare_alpha
    adapter: python_module
    module: src.cli.run_candidate_selection
    argv:
      - --artifacts-root
      - docs/examples/output/pipeline_scenario_matrix_case_study/artifacts/alpha
      - --metric
      - ic_ir
      - --max-candidates
      - "3"
      - --output-path
      - docs/examples/output/pipeline_scenario_matrix_case_study/artifacts/candidate_selection

  - id: construct_portfolio
    depends_on:
      - candidate_selection
    adapter: python_module
    module: src.cli.run_portfolio
    argv:
      - --portfolio-name
      - scenario_matrix_pipeline_portfolio
      - --timeframe
      - 1D
      - --output-dir
      - docs/examples/output/pipeline_scenario_matrix_case_study/artifacts/portfolios
```

## Execute

```powershell
python -m src.cli.run_pipeline --config configs/pipelines/scenario_matrix_pipeline.yml
```

On the verified run in this repository, the pipeline resolved to:

```text
pipeline_run_id = scenario_matrix_pipeline_pipeline_d4d50c3683a7
```

## Artifact Walkthrough

Pipeline-level artifacts are written to:

```text
artifacts/pipelines/scenario_matrix_pipeline_pipeline_d4d50c3683a7/
```

Required files:

- `manifest.json`
- `pipeline_metrics.json`
- `lineage.json`
- `state.json`

What each file contains:

- `manifest.json`: ordered step inventory, status, module, dependencies, and
  step artifact references
- `pipeline_metrics.json`: pipeline status plus deterministic step timing and
  status counts
- `lineage.json`: step dependency edges plus produced artifact nodes
- `state.json`: merged pipeline state, including the candidate-selection
  artifact path and run id consumed by the portfolio step

Workflow outputs referenced by the pipeline live under:

```text
docs/examples/output/pipeline_scenario_matrix_case_study/artifacts/
```

Key subdirectories:

- `alpha/`
- `comparisons/alpha/`
- `comparisons/strategy/`
- `candidate_selection/`
- `portfolios/`

## Determinism

The pipeline run id is derived from the canonicalized YAML spec, so the same
logical pipeline resolves to the same `pipeline_run_id`.

The pipeline layer also writes canonical JSON for:

- manifest ordering
- lineage node and edge ordering
- merged state serialization
- pipeline metrics payload structure

Focused tests cover:

- config structure and dependency ordering
- repeated pipeline runs producing the same pipeline artifact bundle
- repeated candidate-selection and portfolio outputs remaining byte-stable in
  the exercised pipeline flow

## Comparison To Campaign Runner

Use the campaign runner when you want one higher-level research orchestration
surface with campaign-specific stage logic, checkpoint semantics, review
integration, and campaign summaries.

Use this pipeline when you want:

- an explicit YAML DAG
- direct reuse of existing CLI modules without campaign-specific orchestration
- step-by-step lineage and state artifacts under `artifacts/pipelines/`
- a lighter orchestration layer for reproducible scenario-matrix execution

The pipeline version does not modify campaign orchestration and does not add a
new orchestration system alongside it. It simply composes the already-existing
CLI surfaces with `depends_on` ordering and pipeline artifact capture.
