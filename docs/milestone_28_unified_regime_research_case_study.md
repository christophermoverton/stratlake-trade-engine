# Milestone 28: Unified Regime Research Case Study

## Purpose

This document describes the M28.6 capstone case study: a unified, reproducible
regime research workflow that can be run from a script or CLI-style entrypoint,
inspected from notebook-friendly surfaces, wrapped by pipeline and orchestrator
examples, and validated through cross-layer comparison.

The case study proves the central Milestone 28 claim:

> StratLake has one execution system with multiple entry points and no
> divergence in logic or outputs.

The case study does not introduce a new workflow engine. It calls existing
StratLake CLI modules, `src.execution` APIs, existing pipeline and research
orchestration surfaces, regime benchmark workflows, and the M28.5 cross-layer
validation foundation.

**This case study is research and validation oriented. It does not imply live trading
or production deployment readiness.**

## Architecture

The case study has a single canonical callable:

```python
from docs.examples.m28_unified_regime_research_case_study import (
    run_m28_unified_regime_research_case_study,
)
```

Every entry point — script, notebook, pipeline wrapper — calls this same
callable or delegates to existing `src.execution` APIs. No workflow logic is
reimplemented in the notebook or orchestrator wrappers.

### Execution System Principle

```
Script / CLI
  │
  ├── run_m28_unified_regime_research_case_study(output_root=...)
  │
Notebook (docs/examples/notebooks/m28_unified_regime_research_case_study.py)
  │
  └── calls same function
  │
Pipeline wrappers (docs/examples/pipelines/)
  │
  └── thin wrappers around the same function or src.execution APIs
```

All entry points produce the same artifact contracts under their respective
output roots. Cross-layer validation normalizes root-specific paths and
transient fields, then compares stable contracts.

## How This Ties Together M28.1–M28.5

| Milestone | Component | How M28.6 Builds On It |
|-----------|-----------|------------------------|
| M28.1 | Implementation audit and methodology | M28.6 follows the audit's recommendation: harden and expose existing surfaces, do not introduce parallel pipeline systems |
| M28.2 | Artifact safety hardening | M28.6 uses relative output roots, `reset_output`, and safe output patterns consistent with M28.2 conventions |
| M28.3 | Airflow, Prefect, Dagster wrappers | M28.6 extends the existing pipeline wrappers with capstone-specific callables that delegate to the canonical function |
| M28.4 | Notebook integration guidance | M28.6 provides a script-backed notebook example and an `.ipynb` that follow M28.4 patterns without reimplementing workflow logic |
| M28.5 | Cross-layer validation | M28.6 invokes the existing `run_cross_layer_validation` as Stage 2, writing its report to the case study output root as `cross_layer_comparison.json` |

## Workflow Stages

The case study executes three stages:

### Stage 1: Regime Benchmark Pack

Calls `src.execution.regime_benchmark.run_regime_benchmark_pack` with the
existing M26 regime benchmark config:

```text
configs/regime_benchmark_packs/m26_regime_policy_benchmark.yml
```

This runs regime benchmark variants across static baseline, taxonomy,
calibrated taxonomy, GMM classifier, GMM calibrated overlay, and
policy-optimized regime sources. It generates a benchmark matrix, model
comparison, calibration comparison, policy comparison, and regime-conditional
performance summaries.

Artifacts are written to:

```text
docs/examples/output/m28_unified_regime_research_case_study/regime_benchmark/
```

### Stage 2: Cross-Layer Validation

Calls `src.execution.run_cross_layer_validation` (the M28.5 foundation). This
runs the three M28.5 scenarios:

1. `benchmark_pack_cli_api` — CLI vs `src.execution` API
2. `notebook_benchmark_api` — notebook-style callable vs `src.execution` API
3. `prefect_wrapper_api` — Prefect fallback callable vs `src.execution` API

The cross-layer validation report is written as `cross_layer_comparison.json`
in the case study output root.

This stage can be skipped with `--skip-cross-layer-validation` or
`include_cross_layer_validation=False` if the benchmark workflows are too
expensive for a given environment.

### Stage 3: Case Study Artifact Assembly

Assembles `manifest.json`, `summary.json`, `validation_report.json`, and
`artifact_index.json` from the outputs of Stage 1 and Stage 2. No new workflow
logic is introduced here — this stage reads artifacts produced by existing
execution surfaces and packages them into a machine-readable case study record.

## Exact Execution Surfaces Called

| Surface | Module path |
|---------|-------------|
| Regime benchmark pack | `src.execution.regime_benchmark.run_regime_benchmark_pack` |
| Cross-layer validation | `src.execution.run_cross_layer_validation` |

## Artifact Layout

```
docs/examples/output/m28_unified_regime_research_case_study/
├── manifest.json                  # Case study manifest
├── summary.json                   # Case study summary (workflow stages, surfaces called)
├── validation_report.json         # Case study validation report
├── cross_layer_comparison.json    # M28.5 cross-layer validation report
├── artifact_index.json            # Evidence index of all artifacts
├── regime_benchmark/              # Stage 1: regime benchmark artifacts
│   ├── benchmark_matrix.csv
│   ├── benchmark_matrix.json
│   ├── model_comparison.csv
│   ├── calibration_comparison.csv
│   ├── policy_comparison.csv
│   ├── benchmark_summary.json
│   ├── conditional_performance_summary.json
│   ├── stability_summary.json
│   ├── transition_summary.json
│   ├── manifest.json
│   └── ...
└── cross_layer_workdir/           # Stage 2: cross-layer validation working artifacts
    └── ...
```

### Machine-Readable Output Schema

All top-level JSON files follow these schema conventions:

**`summary.json`** includes:

- `case_study_name`
- `milestone`
- `workflow_stages` — list of stage names
- `execution_surfaces_called` — list of dotted module paths
- `regime_benchmark` — regime benchmark summary fields
- `cross_layer_validation` — cross-layer validation status and counts
- `limitations` — explicit documented limitations
- `research_orientation_note`

**`manifest.json`** includes:

- `case_study_name`
- `milestone`
- `workflow_stages`
- `execution_surfaces_called`
- `regime_benchmark_run_id`
- `regime_benchmark_name`
- `output_artifact_paths` — relative paths keyed by artifact name
- `regime_artifact_paths` — relative paths to regime benchmark sub-artifacts

**`validation_report.json`** includes:

- `run_type`
- `schema_version`
- `status` — `passed` or `warning`
- `regime_benchmark_checks`
- `cross_layer_validation_checks`
- `case_study_artifact_schema`
- `limitations`

**`cross_layer_comparison.json`** is the M28.5 cross-layer validation report
written by `src.execution.run_cross_layer_validation`. See
[`docs/cross_layer_validation.md`](cross_layer_validation.md) for its schema.

## Notebook Usage

The script-backed notebook example is at:

```text
docs/examples/notebooks/m28_unified_regime_research_case_study.py
```

It calls the same canonical case-study function. Notebook cells import and
call `run_m28_unified_regime_research_case_study`, then inspect returned paths
and load artifact JSON through `ExecutionResult` helpers.

Follow notebook integration guidance in
[`docs/notebook_integration.md`](notebook_integration.md) for patterns on
reusing output roots, inspecting status markers, and keeping notebooks as thin
inspection surfaces over artifact-first contracts.

An `.ipynb` version is also provided:

```text
docs/examples/notebooks/m28_unified_regime_research_case_study.ipynb
```

The `.ipynb` has no outputs committed. Run it with:

```python
run_m28_unified_regime_research_case_study(
    output_root="artifacts/notebooks/m28_unified_regime_research_case_study/attempt_001"
)
```

Inspect artifacts through the returned dict or the `ExecutionResult` helpers
from `src.execution`.

## Pipeline / Orchestrator Usage

Capstone-specific pipeline wrappers are provided in:

```text
docs/examples/pipelines/m28_prefect_regime_research_flow.py     (run_m28_capstone_prefect_example)
docs/examples/pipelines/m28_airflow_regime_research_dag.py      (run_m28_capstone_airflow_example)
docs/examples/pipelines/m28_dagster_regime_research_job.py      (run_m28_dagster_capstone_example)
```

These wrappers are thin callables that delegate to the canonical case-study
function. They are importable and testable without Airflow, Prefect, or Dagster
installed. When the optional scheduler packages are available, DAG/flow/job
objects are also registered.

For pipeline integration patterns, see
[`docs/pipeline_integration.md`](pipeline_integration.md).

Generic M28.3 benchmark-pack wrappers (`run_m28_prefect_example`,
`run_m28_airflow_example`, `run_m28_dagster_example`) remain unchanged.

## Cross-Layer Validation

The M28.6 case study calls the existing M28.5 `run_cross_layer_validation`
as a second stage. This compares:

- CLI vs `src.execution` API benchmark-pack artifacts
- Notebook-style callable vs `src.execution` API
- Prefect fallback callable vs `src.execution` API

See [`docs/cross_layer_validation.md`](cross_layer_validation.md) for the full
scenario descriptions, stable comparison contract, and limitations.

The cross-layer validation output is written to `cross_layer_comparison.json`
in the case study output root and is also available at the default M28.5 path:

```text
artifacts/qa/cross_layer_validation_report.json
```

### Running Cross-Layer Validation Independently

```powershell
python -m src.cli.run_cross_layer_validation
```

```powershell
python -m src.cli.run_milestone_validation --include-cross-layer-validation
```

## Running the Case Study

### From the command line

```powershell
python docs/examples/m28_unified_regime_research_case_study.py
```

### Skipping cross-layer validation

```powershell
python docs/examples/m28_unified_regime_research_case_study.py --skip-cross-layer-validation
```

### Dry-run structural check

```powershell
python docs/examples/m28_unified_regime_research_case_study.py --dry-run
```

### Custom output root

```powershell
python docs/examples/m28_unified_regime_research_case_study.py --output-root artifacts/qa/m28_6_case_study
```

### From Python

```python
from docs.examples.m28_unified_regime_research_case_study import (
    run_m28_unified_regime_research_case_study,
)

result = run_m28_unified_regime_research_case_study()
print(result["summary"])
print(result["validation_report"])
```

## Validation Commands

Run the following after implementing or updating the case study:

```powershell
python -m src.cli.run_docs_path_lint
pytest tests/test_m28_unified_regime_research_case_study.py
pytest tests/test_m28_notebook_integration_examples.py
pytest tests/test_m28_pipeline_integration_examples.py
pytest tests/test_cross_layer_validation.py
pytest tests/test_execution_api.py
python docs/examples/m28_unified_regime_research_case_study.py --dry-run
```

Optional milestone bundle:

```powershell
python -m src.cli.run_milestone_validation \
    --bundle-dir artifacts/qa/m28_6_milestone_validation_bundle \
    --include-cross-layer-validation
```

## Limitations

- This case study uses fixture-backed example configs and does not require live
  market data.
- Cross-layer validation covers representative benchmark-pack parity only (the
  three M28.5 scenarios).
- Stress tests, promotion gates, review packs, candidate selection, and market
  simulation are deferred workflow stages not included in this case study run.
  They are available as separate CLI/API surfaces and documented in
  [`docs/regime_policy_stress_testing.md`](regime_policy_stress_testing.md),
  [`docs/regime_review_packs.md`](regime_review_packs.md), and
  [`docs/regime_aware_candidate_selection.md`](regime_aware_candidate_selection.md).
- The regime benchmark variants are deterministic diagnostic comparisons, not
  trading recommendations.
- This case study does not prove distributed locking, production deployment
  readiness, or exhaustive parity for every config variant. For concurrency and
  idempotency guarantees, see
  [`docs/concurrency_and_idempotency.md`](concurrency_and_idempotency.md).
- Generated outputs under
  `docs/examples/output/m28_unified_regime_research_case_study/` are runtime
  artifacts and are not committed to the repository.
