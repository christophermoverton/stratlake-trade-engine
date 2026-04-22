# Notebook Execution API

## Overview

The execution API exposes the repository's highest-value deterministic
workflows through importable Python functions for notebooks and scripts. These
entrypoints use the same execution foundations as the CLI runners: validation,
runtime config precedence, artifact persistence, manifests, and registries are
preserved.

Use this surface when you want to run research workflows from Python without
building subprocess calls or emulating shell arguments.

```python
from src.execution import (
    load_json_artifact,
    run_strategy,
    run_alpha,
    run_alpha_evaluation,
    run_benchmark_pack,
    run_docs_path_lint,
    run_deterministic_rerun_validation,
    run_milestone_validation,
    run_portfolio,
    run_pipeline,
    run_research_campaign,
)
```

Each function returns an `ExecutionResult`, a small notebook-friendly summary
around the underlying workflow result.

## Entrypoints

### Strategy

```python
from src.execution import run_strategy

result = run_strategy(
    "momentum_v1",
    start="2022-01-01",
    end="2023-01-01",
    strict=True,
)

result.workflow
result.run_id
result.metrics["sharpe_ratio"]
result.artifact_dir
result.manifest_path
result.output_keys()
result.load_metrics_json()
```

Strategy runs write the same deterministic strategy artifacts as the CLI, such
as `metrics.json`, `qa_summary.json`, `equity_curve.csv`, and `manifest.json`.

### Alpha Evaluation

```python
from src.execution import run_alpha_evaluation

result = run_alpha_evaluation(
    config={
        "alpha_name": "cs_linear_ret_1d",
        "dataset": "features_daily",
        "target_column": "target_ret_1d",
        "price_column": "close",
        "start": "2022-01-01",
        "end": "2023-01-01",
    }
)

result.metrics["mean_ic"]
result.output_path("alpha_metrics_json")
result.output_path("predictions_parquet")
result.manifest_path
```

`run_alpha_evaluation` also accepts `config_path=...` plus keyword overrides.
The resolved run still follows the alpha evaluation validation, manifest, and
registry behavior used by the CLI.

### Alpha Full Run

```python
from src.execution import run_alpha

result = run_alpha(
    "cs_linear_ret_1d",
    mode="full",
    start="2022-01-01",
    end="2023-01-01",
)

result.metrics["ic_ir"]
result.output_path("signals_parquet")
result.output_path("sleeve_metrics_json")
result.extra["mode"]
```

Use `mode="evaluate"` when you only want the evaluation artifacts. Use
`mode="full"` when you also want deterministic signal mapping, sleeve artifacts,
and the full-run scaffold.

### Portfolio

Portfolio construction consumes completed component artifacts. A direct
run-id-driven portfolio can be launched from Python like this:

```python
from src.execution import run_portfolio

result = run_portfolio(
    portfolio_name="core_portfolio",
    run_ids=["momentum_run_id", "mean_reversion_run_id"],
    timeframe="1D",
    strict=True,
)

result.metrics["total_return"]
result.output_path("weights_csv")
result.output_path("portfolio_returns_csv")
result.manifest_path
```

Config-driven and registry-backed portfolio runs are also available:

```python
result = run_portfolio(
    portfolio_config_path="configs/portfolios.yml",
    portfolio_name="momentum_meanrev_equal",
    from_registry=True,
    timeframe="1D",
)
```

Portfolio outputs preserve the CLI artifact contract, including
`components.json`, `weights.csv`, `portfolio_returns.csv`,
`portfolio_equity_curve.csv`, `metrics.json`, `qa_summary.json`, and
`manifest.json`.

### Pipeline

YAML pipeline specs can be launched directly from Python:

```python
from src.execution import run_pipeline

result = run_pipeline("configs/test_pipeline.yml")

result.run_id
result.output_path("manifest_json")
result.output_path("pipeline_metrics_json")
result.output_path("lineage_json")
result.output_path("state_json")
result.extra["execution_order"]
result.extra["state"]
```

`run_pipeline` uses `PipelineSpec.from_yaml(...)` and `PipelineRunner.run()`,
the same path used by `python -m src.cli.run_pipeline`. It preserves
deterministic pipeline ids, dependency ordering, pipeline state handoff,
manifest writing, metrics, lineage, state artifacts, and registry updates.

### Research Campaigns And Orchestration

Campaign-style staged workflows are available through:

```python
from src.execution import run_research_campaign, run_campaign

result = run_research_campaign(config_path="configs/research_campaign.yml")

result.workflow
result.run_id
result.artifact_dir
result.manifest_path
result.output_path("checkpoint_json")
result.output_path("summary_json")
result.extra["stage_statuses"]
result.extra["stage_execution"]
result.load_summary_json()
```

`run_campaign(...)` is an alias. Both functions accept `config_path=...`, an
in-memory config mapping, or a resolved `ResearchCampaignConfig`. They resolve
configuration with the shared campaign config resolver and delegate to the
existing campaign runner, so preflight checks, stage ordering, checkpoints,
resume/reuse decisions, manifests, milestone reports, and downstream stage
invocation stay CLI-equivalent.

When the campaign config enables scenario expansion, the same function returns
an orchestration summary:

```python
result = run_research_campaign(
    config={
        "scenarios": {"enabled": True},
        # normal campaign config sections...
    }
)

result.workflow  # "research_campaign_orchestration"
result.output_path("scenario_catalog_json")
result.output_path("scenario_matrix_csv")
result.output_path("expansion_preflight_json")
result.extra["scenarios"]
result.extra["scenario_run_ids"]
```

Single-campaign results expose checkpoint and stage state references in
`extra`; scenario orchestration results expose scenario catalogs, matrix
artifacts, expansion preflight, and per-scenario run ids.

### Validation

Operational validation checks are exposed through the same execution layer:

```python
from src.execution import (
    run_docs_path_lint,
    run_deterministic_rerun_validation,
    run_milestone_validation,
)

docs_result = run_docs_path_lint(output="artifacts/qa/docs_path_lint.json")
rerun_result = run_deterministic_rerun_validation(
    workdir="artifacts/qa/rerun_workdir",
    output="artifacts/qa/deterministic_rerun.json",
)
bundle_result = run_milestone_validation(
    bundle_dir="artifacts/qa/milestone_validation_bundle",
    include_full_pytest=False,
)
```

These wrappers preserve the CLI validation contracts. `run_docs_path_lint`
writes the same docs/path lint JSON report and exposes it as
`output_paths["report_json"]`. `run_deterministic_rerun_validation` writes the
canonical rerun comparison report and exposes target/pass counts in `extra`.
`run_milestone_validation` builds the milestone validation bundle without
changing its schema; notebook callers can inspect
`output_paths["summary_json"]`, `output_paths["docs_path_lint_json"]`, and
`output_paths["deterministic_rerun_json"]`.

CLI fail conditions are unchanged: the CLI runners still exit non-zero when
their report status is not passed. The Python APIs return structured
`ExecutionResult` objects with the same report payloads in `metrics` and
`raw_result` so notebooks can inspect failures before deciding whether to
raise.

### Benchmark Packs

Benchmark-pack execution is available from Python:

```python
from src.execution import run_benchmark_pack

result = run_benchmark_pack(
    "configs/benchmark_packs/m22_scale_repro.yml",
    output_root="artifacts/benchmark_packs/m22_scale_repro",
)

result.run_id
result.metrics["status"]
result.output_path("summary_json")
result.output_path("manifest_json")
result.output_path("inventory_json")
result.output_path("benchmark_matrix_csv")
```

The wrapper loads the same benchmark-pack config and delegates to the existing
benchmark runner used by `python -m src.cli.run_benchmark_pack`. It preserves
the deterministic output layout, checkpoint/resume behavior, manifest,
inventory, benchmark matrix, and optional comparison report. Named paths include
`summary_json`, `manifest_json`, `checkpoint_json`, `inventory_json`,
`batch_plan_json`, `batch_plan_csv`, `benchmark_matrix_csv`,
`benchmark_matrix_summary`, and `comparison_json` when a comparison is written.

## `ExecutionResult` Contract

Notebook entrypoints return `ExecutionResult` with these stable fields:

* `workflow`: workflow family, such as `"strategy"`, `"alpha"`,
  `"alpha_evaluation"`, `"portfolio"`, `"pipeline"`,
  `"research_campaign"`, `"research_campaign_orchestration"`,
  `"docs_path_lint"`, `"deterministic_rerun_validation"`,
  `"milestone_validation"`, or `"benchmark_pack"`.
* `run_id`: deterministic run identifier produced by the workflow.
* `name`: strategy, alpha, portfolio, pipeline, or pack name.
* `artifact_dir`: root directory for persisted artifacts, when the workflow
  writes one.
* `manifest_path`: expected or existing `manifest.json` path for the run.
* `registry_path`: registry path when the execution wrapper exposes one.
* `metrics`: summary metrics useful for notebook inspection.
* `output_paths`: named artifact paths, such as `metrics_json`,
  `alpha_metrics_json`, `weights_csv`, or `predictions_parquet`.
* `extra`: deterministic workflow-specific metadata, such as alpha mode,
  portfolio timeframe, allocator, pipeline state, campaign stage execution,
  scenario summaries, validation check references, or benchmark-pack status.
* `raw_result`: the original workflow result object for deeper inspection.

The result also exposes lightweight inspection helpers for those same fields
and paths. They are conveniences for reading existing artifacts, not alternate
execution surfaces.

Use `to_dict()` when you want a JSON-safe summary:

```python
payload = result.to_dict()
payload["artifact_dir"]
payload["metrics"]
```

By default, `to_dict()` excludes `raw_result` because raw workflow results may
contain DataFrames, models, or other non-JSON objects. Pass
`include_raw_result=True` only for local debugging.

## Notebook Inspection Helpers

`ExecutionResult` keeps the artifact-first contract visible while removing the
small bits of notebook boilerplate that otherwise repeat across workflows.
These helpers only inspect local paths already exposed by the result. They do
not trigger execution, mutate files, create persistence, download data, or infer
global state.

Available helpers:

* `output_keys()`: list named outputs in deterministic order.
* `has_output(key)`: check whether an optional output was exposed.
* `output_path(key, must_exist=False)`: fetch a named output path and raise a
  clear `KeyError` when the key is absent. Set `must_exist=True` when the
  notebook should fail immediately if the file is missing.
* `artifact_path(*parts, must_exist=False)`: resolve a path under
  `artifact_dir` without creating it. Paths that escape the artifact root are
  rejected.
* `load_manifest()`: load `manifest_path` as JSON and require an object payload.
* `load_output_json(key)`: load JSON from an explicit named output path.
* `load_metrics_json(key=None)`: load a metrics/report JSON output. Without a
  key, it chooses the first available standard metrics key such as
  `metrics_json`, `alpha_metrics_json`, `aggregate_metrics_json`,
  `pipeline_metrics_json`, or `report_json`.
* `load_summary_json(key=None)`: load a summary-style JSON output. Without a
  key, it chooses the first available standard summary key such as
  `summary_json`, `benchmark_matrix_summary`, `qa_summary_json`,
  `training_summary_json`, or `report_json`.
* `load_comparison_json(key="comparison_json")`: load a comparison JSON payload
  when a workflow emitted one.
* `notebook_summary()`: return a compact JSON-safe payload with identity,
  artifact root, manifest path, metrics, output keys, and `extra`.
* `load_json_artifact(path)`: module-level helper for loading any explicit
  local JSON artifact path.

Example inspection flow:

```python
result = run_strategy("momentum_v1", start="2022-01-01", end="2023-01-01")

result.notebook_summary()
result.output_keys()

manifest = result.load_manifest()
metrics = result.load_metrics_json()
qa_summary = result.load_summary_json("qa_summary_json")

equity_curve_path = result.output_path("equity_curve_csv", must_exist=True)
```

Optional outputs should be checked explicitly:

```python
result = run_benchmark_pack("configs/benchmark_packs/m22_scale_repro.yml")

if result.has_output("comparison_json"):
    comparison = result.load_comparison_json()
```

For workflow-specific reports, prefer loading by the named output key so the
file being inspected remains obvious:

```python
summary = result.load_output_json("summary_json")
docs_lint = result.load_output_json("docs_path_lint_json")
```

These helpers are intentionally narrow. Use `metrics` for the in-memory summary
returned by the workflow, and use the `load_*` helpers when you want to inspect
the machine-readable artifact that was written to disk.

## CLI/API Parity Expectations

The notebook execution API is expected to stay aligned with the corresponding
CLI workflows on stable, machine-readable contracts. Parity tests cover
representative strategy, alpha evaluation, portfolio, pipeline, validation, and
benchmark-pack paths. They compare workflow identifiers, deterministic run ids,
names, summary metrics, manifest references, named output path keys, artifact
file names, and workflow-specific `extra` fields.

Parity intentionally does not compare raw stdout text, object identity,
absolute workspace prefixes, transient logs, or raw DataFrames/models attached
to `raw_result`. CLI entrypoints may also apply process-oriented behavior, such
as exiting non-zero when validation reports fail, while the API returns an
`ExecutionResult` so notebooks can inspect the failed report.

## Working With Artifacts

The API is intentionally artifact-first. In notebooks, prefer using the
structured paths instead of reconstructing filenames manually:

```python
metrics_path = result.output_paths["metrics_json"]
manifest_path = result.manifest_path

metrics_payload = result.metrics
summary_payload = result.to_dict()
```

This keeps notebook code aligned with the same deterministic artifact contracts
used by CLI and pipeline workflows.

The helper equivalent keeps the same explicit paths while adding clearer
missing-file behavior:

```python
metrics_path = result.output_path("metrics_json", must_exist=True)
manifest_payload = result.load_manifest()
metrics_artifact = result.load_metrics_json()
summary_payload = result.notebook_summary()
```

## Example Script

See
[`docs/examples/notebook_execution_api_examples.py`](examples/notebook_execution_api_examples.py)
for compact Python functions that demonstrate strategy, alpha evaluation, full
alpha, portfolio, pipeline, and campaign calls through the public execution API.
