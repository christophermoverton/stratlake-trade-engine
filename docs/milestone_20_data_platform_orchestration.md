# Milestone 20 Data Platform Orchestration

## Overview

Milestone 20 adds a small, deterministic pipeline runner for YAML-defined
workflows.

Its purpose is to let the repository compose existing CLI modules into one
artifact-driven run with:

* deterministic pipeline ids
* deterministic artifact payloads
* schema-validated pipeline outputs
* lightweight state passing between steps
* shared registry tracking under `artifacts/pipelines/`

The current runner is implemented in
`src/pipeline/pipeline_runner.py` and exposed through
`python -m src.cli.run_pipeline`.

This orchestration layer complements
`src.cli.run_research_campaign`. It does not replace it.

Practical distinction:

* `run_research_campaign` is a specialized research workflow orchestrator with
  campaign-specific stage semantics, inheritance, preflight, checkpointing,
  and stitched campaign summaries.
* `run_pipeline` is a generalized abstraction that executes a DAG of
  `run_cli(argv)` steps and emits uniform pipeline artifacts.

## Pipeline YAML Specification

The committed minimal working example is
`configs/test_pipeline.yml`:

```yaml
id: test_pipeline
steps:
  - id: prepare
    adapter: python_module
    module: src.pipeline.testing
    argv:
      - --stage
      - prepare
  - id: evaluate
    depends_on:
      - prepare
    adapter: python_module
    module: src.pipeline.testing
    argv:
      - --stage
      - evaluate
```

The loader accepts either:

* a top-level mapping like the example above
* a nested `pipeline:` mapping, which is unwrapped before validation

### Supported Fields

Current implementation supports these spec fields:

* `id`
  Deterministic pipeline identifier. If omitted, the YAML file stem is used.
* `steps`
  Non-empty ordered list of step definitions.
* `steps[].id`
  Required unique step identifier.
* `steps[].depends_on`
  Optional string or list of step ids.
* `steps[].adapter`
  Must be `python_module`.
* `steps[].module`
  Import path for the Python module to execute.
* `steps[].argv`
  List of CLI-style arguments passed to the step module.
* `parameters`
  Optional mapping used by the current implementation for dataset lineage and
  prior-state loading.

Important implementation note:

* The runner currently supports `id`, not `pipeline.name`.
* The runner currently does not support a YAML `artifacts_root` field.
  Artifacts always write under `artifacts/pipelines/<pipeline_run_id>/`.
* The manifest artifact uses `pipeline_name` as its output field, but that
  value is sourced from the spec `id`.

### Execution Model

Execution is dependency ordered and deterministic.

Current behavior:

* steps are validated before execution
* duplicate step ids are rejected
* unknown dependencies are rejected
* self-dependencies are rejected
* dependency cycles are rejected
* the runner computes a topological order
* when multiple steps become ready together, declaration order is preserved

Each step module is imported and invoked through its exported
`run_cli(argv)` function.

The pipeline runner also passes extra keyword arguments when the target
signature accepts them:

* `state`
  Current pipeline state snapshot
* `pipeline_context`
  Deterministic context with `pipeline_id`, `pipeline_run_id`, `step_id`, and
  `step_depends_on`

## Running a Pipeline

Run a pipeline with:

```powershell
python -m src.cli.run_pipeline --config configs/test_pipeline.yml
```

The CLI is implemented in `src/cli/run_pipeline.py`.

Expected behavior:

* the YAML spec is loaded into `PipelineSpec`
* a deterministic `pipeline_run_id` is derived from the canonicalized spec
* steps run in dependency order
* artifacts are written under `artifacts/pipelines/<pipeline_run_id>/`
* a registry entry is appended to `artifacts/pipelines/registry.jsonl`
* the CLI prints `pipeline_id`, `pipeline_run_id`, `steps_executed`, and
  `execution_order`

Failure behavior:

* if a step raises, the pipeline status becomes `failed`
* already completed steps remain recorded
* downstream steps are marked `skipped`
* metrics, lineage, state, and manifest are still emitted for the failed run
* if contract validation fails, the runner fails fast before writing the
  pipeline artifact directory

## Artifact Structure

Each successful or failed pipeline run writes:

```text
artifacts/pipelines/<pipeline_run_id>/
  manifest.json
  pipeline_metrics.json
  lineage.json
  state.json
```

Pipeline runs are also tracked in the shared registry:

```text
artifacts/pipelines/
  registry.jsonl
```

Example tree for the committed test pipeline:

```text
artifacts/pipelines/test_pipeline_pipeline_f402d7b306f5/
  manifest.json
  pipeline_metrics.json
  lineage.json
  state.json
```

### `manifest.json`

`manifest.json` is the pipeline inventory artifact.

It records:

* `pipeline_run_id`
* `pipeline_name`
* top-level pipeline `status`
* ordered `steps`

Each step entry records:

* `step_id`
* `status`
* `module`
* `depends_on`
* normalized `outputs`
* `step_artifact_dir`
* `step_manifest_path`

`step_artifact_dir` and `step_manifest_path` are extracted from step return
values when available. The runner looks for fields such as `artifact_dir`,
`manifest_path`, and `manifest_json`.

For `configs/test_pipeline.yml`, the example step module
`src.pipeline.testing` returns only `argc` and `argv`, so both artifact
reference fields are `null`.

### `pipeline_metrics.json`

`pipeline_metrics.json` captures run timing and status summaries.

It records:

* `started_at_unix`
* `ended_at_unix`
* `duration_seconds`
* pipeline `status`
* ordered per-step `steps`
* `step_durations_seconds`
* `status_counts`
* optional `row_counts`

`row_counts` is included only when the runner can infer a row count from step
outputs, such as a CSV path or dataframe-like object.

### `lineage.json`

`lineage.json` is the pipeline graph artifact.

It records:

* `nodes`
* `edges`
* `pipeline_run_id`

Nodes and edges are emitted in stable sorted order.

The graph includes:

* one `step` node per pipeline step
* one `artifact` node for a step output directory when the step returns an
  artifact reference
* optional `dataset` nodes when `parameters.datasets` is provided

### `state.json`

`state.json` records merged incremental pipeline state:

* `schema_version`
* `pipeline_run_id`
* `state`

If no step emits state, the saved state is an empty object.

### `registry.jsonl`

`artifacts/pipelines/registry.jsonl` is the shared append-only registry for
pipeline runs.

Each line is one canonical JSON object containing:

* `artifact_dir`
* `pipeline_name`
* `pipeline_run_id`
* `status`

## Lineage Model

The lineage contract lives in `contracts/lineage.schema.json`.

### Node Types

Current node types are:

* `step`
  One node per declared pipeline step.
* `artifact`
  Emitted when a step result exposes an artifact directory reference.
* `dataset`
  Optional nodes derived from `parameters.datasets`.

### Edge Types

Current edge types are:

* `depends_on`
  Used for step-to-step dependencies and dataset-to-step consumption links.
* `produces`
  Used for step-to-artifact output links.

### Example Graph

For `configs/test_pipeline.yml`, the lineage graph is:

```text
step:prepare --depends_on--> step:evaluate
```

When a step returns an artifact directory, the graph expands to:

```text
step:prepare --depends_on--> step:evaluate
step:prepare --produces--> artifact:<prepare_dir>
step:evaluate --produces--> artifact:<evaluate_dir>
```

When datasets are declared in `parameters.datasets`, the runner adds dataset
consumer edges:

```yaml
parameters:
  datasets:
    prices:
      consumers:
        - prepare
```

This yields a lineage edge:

```text
dataset:prices --depends_on--> step:prepare
```

## Metrics & Observability

The pipeline runner captures:

* pipeline start and end timestamps
* total pipeline duration
* per-step start and end timestamps
* per-step durations
* per-step statuses
* aggregate `status_counts`
* optional `row_counts`

Observed statuses are:

* `completed`
* `failed`
* `skipped`
* `reused`

Current runner behavior around failures:

* metrics are still emitted on failure
* the failing step is recorded as `failed`
* downstream steps are recorded as `skipped`
* lineage still reflects the declared graph, with step node status showing
  `completed`, `failed`, or `skipped`

The runner does not currently emit a separate logging or tracing backend. The
main observability surface is the artifact set plus the registry.

## Contracts & Validation

Pipeline artifact contracts live under `contracts/`:

* `contracts/pipeline_manifest.schema.json`
* `contracts/pipeline_metrics.schema.json`
* `contracts/lineage.schema.json`

Validation is performed by `src/contracts/validate.py`.

Current behavior:

* manifest, metrics, and lineage payloads are built in memory first
* the runner validates all three payloads before writing artifacts
* if validation fails, execution raises `ContractValidationError`
* invalid artifact payloads are not partially written

Error messages are deterministic because validation errors are sorted before
the first failing error is reported.

Example message shape:

```text
JSON contract validation failed for 'lineage.schema.json' at '$': 'edges' is a required property
```

## State / Incremental Execution

The pipeline runner persists incremental state in `state.json`.

State behavior is implemented through:

* prior-state loading from `parameters.state`
* optional `state` keyword injection into step `run_cli` calls
* step-level `state_updates`
* deterministic merge and serialization

### Loading Prior State

The runner can load prior state from:

* `parameters.state.path`
* `parameters.state.state_path`
* `parameters.state.previous_pipeline_run_id`
* `parameters.state.previous_run_id`

When `previous_pipeline_run_id` is used, the runner loads:

```text
artifacts/pipelines/<previous_pipeline_run_id>/state.json
```

### Step-Level Updates

If a step returns:

```json
{
  "state_updates": {
    "watermark_end": "2025-01-10"
  }
}
```

the runner merges that object into the pipeline state after the step
completes successfully.

Failed steps do not apply their state updates.

### Merge Rules

Current merge behavior is simple and deterministic:

* the current state is copied before being passed into a step
* returned `state_updates` must be a JSON object
* keys from `state_updates` overwrite same-named keys in the current state
* nested objects are replaced at the key level rather than deep-merged
* the final state is normalized through canonical JSON serialization

### Watermark Example

One practical incremental pattern is:

```yaml
id: load_prior_state_pipeline
parameters:
  state:
    previous_pipeline_run_id: prior_pipeline_run
steps:
  - id: only
    adapter: python_module
    module: my_pipeline_module
    argv: []
```

If the prior state contains:

```json
{
  "state": {
    "watermark_start": "2025-01-01"
  }
}
```

and the current step returns:

```json
{
  "state_updates": {
    "watermark_end": "2025-01-10"
  }
}
```

the resulting `state.json` contains both values.

## Registry Integration

Pipeline registry behavior is implemented in `src/pipeline/registry.py`.

The registry file is:

```text
artifacts/pipelines/registry.jsonl
```

Current registry rules:

* append-only JSON Lines format
* entries are canonicalized before write
* artifact paths are normalized to stable POSIX-style relative paths when they
  live under the repository root
* identical duplicate `pipeline_run_id` entries are ignored
* conflicting duplicate `pipeline_run_id` entries raise `RegistryError`

Example entry:

```json
{
  "artifact_dir": "artifacts/pipelines/test_pipeline_pipeline_f402d7b306f5",
  "pipeline_name": "test_pipeline",
  "pipeline_run_id": "test_pipeline_pipeline_f402d7b306f5",
  "status": "completed"
}
```

Supported registry statuses are:

* `completed`
* `failed`
* `partial`

The current runner writes `completed` or `failed`.

## Determinism Guarantees

Determinism is the core design principle of this pipeline system.

Current deterministic guarantees include:

* deterministic `pipeline_run_id`
  Derived from a SHA-256 digest of the canonicalized pipeline spec.
* sorted JSON serialization
  Artifact payloads are written with `sort_keys=True` and canonicalized values.
* stable step ordering
  execution order is derived from dependency order with declaration-order tie
  breaking.
* stable lineage ordering
  nodes are sorted by node id and edges are sorted by `(from, to, type)`.
* stable registry entries
  registry rows are canonical JSON objects with normalized artifact paths.
* reproducibility across repeated runs
  unchanged spec plus unchanged step behavior yields the same
  `pipeline_run_id` and the same artifact payload structure.

What this means in practice:

* equivalent YAML specs produce the same `pipeline_run_id`
* repeated runs with the same deterministic step outputs overwrite the same
  pipeline directory contents with the same JSON structure
* contract validation errors are stable and comparable across reruns

This section is the main design philosophy for Milestone 20: pipeline runs are
treated as reproducible artifact graphs, not opaque ad hoc scripts.

## Relationship to `run_research_campaign`

`src.cli.run_research_campaign` remains the repository's specialized research
orchestration layer.

It still owns concerns such as:

* campaign-stage inheritance
* campaign preflight
* checkpoint and reuse policy
* research-specific stage semantics
* stitched campaign review outputs

`src.cli.run_pipeline` is lower-level and more general.

Integration patterns that fit the current implementation:

* campaign as a macro-step
  A pipeline step can target `src.cli.run_research_campaign` as its module and
  pass campaign CLI arguments through `argv`.
* pipeline wrapper around campaign
  A larger YAML pipeline can use a campaign run as one step among other
  deterministic steps, then capture the returned artifact references in
  pipeline manifest and lineage outputs.

Choose `run_research_campaign` when you need the repository's research-stage
workflow semantics.

Choose `run_pipeline` when you need a simple generalized DAG executor over
existing `run_cli(argv)` modules.

## Related Paths

Implementation and contracts referenced in this document:

* `src/cli/run_pipeline.py`
* `src/pipeline/pipeline_runner.py`
* `src/pipeline/registry.py`
* `src/pipeline/testing.py`
* `configs/test_pipeline.yml`
* `contracts/pipeline_manifest.schema.json`
* `contracts/pipeline_metrics.schema.json`
* `contracts/lineage.schema.json`
* `docs/milestone_16_campaign_workflow.md`
