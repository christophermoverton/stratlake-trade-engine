# Cross-Layer Validation

## Purpose

Cross-layer validation proves that representative StratLake CLI, API,
pipeline/orchestrator, and notebook-style entry points converge on the same
underlying execution logic and equivalent canonical artifacts.

The validation is artifact-first. It runs existing entry points, reads their
machine-readable outputs, normalizes expected run-local differences, and
compares the stable contract that downstream research and review workflows
depend on.

## Core Principle

StratLake has one execution system with multiple entry points. Cross-layer
validation compares stable artifacts from those entry points, not transient
runtime behavior.

This validation does not introduce a new execution framework, a new pipeline
framework, or notebook-only workflow logic. It calls existing CLI helpers,
`src.execution` APIs, and M28 example wrappers.

## Supported Layers

The current M28.5 implementation validates representative coverage for:

- CLI entrypoints, using the benchmark-pack CLI argument path.
- `src.execution` APIs, using `src.execution.run_benchmark_pack`.
- Notebook-style examples from `docs/examples/notebooks/`.
- External orchestrator fallback callables from `docs/examples/pipelines/`,
  without requiring Airflow, Prefect, or Dagster to be installed.

The benchmark pack is the canonical scenario because it is deterministic,
lightweight enough for focused validation, and already emits summary,
manifest, inventory, batch-plan, and benchmark-matrix artifacts.

## Stable Comparison Contract

Cross-layer validation compares stable machine-readable fields:

- workflow name
- logical benchmark-pack config identity
- named output keys
- manifest schema, run type, artifact files, and artifact groups
- summary status, batch counts, scenario counts, and resume state
- benchmark matrix columns, row counts, and deterministic rows
- batch-plan fields and scenario identities
- inventory relative entry paths
- validation pass/fail status

The validator normalizes or ignores expected unstable values:

- absolute workspace paths
- artifact root prefixes
- output-root-specific paths
- run IDs if an entry point generates them differently
- timestamps
- status marker `recorded_at_utc` values
- stdout and stderr
- ordering where the artifact contract does not require ordering
- temporary files
- `_RUNNING.json`, `_SUCCESS.json`, and `_FAILED.json` marker payload
  timestamps
- inventory hashes and byte sizes when path prefixes can change serialized
  artifact contents

## Validation Scenarios

Implemented scenarios:

- CLI vs `src.execution` API for benchmark-pack execution, using
  `configs/benchmark_packs/m22_scale_repro.yml` with a one-batch stop point.
- Notebook-style callable vs `src.execution` API, using
  `docs/examples/notebooks/m28_benchmark_pack_execution_api.py`.
- Prefect fallback callable vs `src.execution` API, using
  `docs/examples/pipelines/m28_prefect_regime_research_flow.py` without
  requiring Prefect.

Deferred scenario:

- Pipeline CLI/API parity is already covered by focused execution tests. A
  separate cross-layer pipeline artifact comparison can be added when a small
  M28-specific pipeline config is selected for this validator.

## Running Validation

Use the dedicated CLI:

```powershell
python -m src.cli.run_cross_layer_validation
```

The default report is:

```text
artifacts/qa/cross_layer_validation_report.json
```

The report is JSON and includes `run_type`, `schema_version`, `status`,
scenario counts, per-scenario digests, differences, compared stable fields,
normalized fields, and limitations.

The milestone bundle can include this report explicitly:

```powershell
python -m src.cli.run_milestone_validation --include-cross-layer-validation
```

Cross-layer validation remains separate from the default deterministic-rerun
command so M22 historical validation behavior stays unchanged.

## Limitations

This validation does not prove exhaustive parity for every optional config.

This validation does not prove distributed locking.

This validation does not replace full pytest, deterministic-rerun validation,
or milestone validation.

This validation does not imply live trading or production deployment
readiness.

M28.6 will provide the unified capstone case study.
