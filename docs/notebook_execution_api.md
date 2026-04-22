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
result.output_paths["alpha_metrics_json"]
result.output_paths["predictions_parquet"]
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
result.output_paths["signals_parquet"]
result.output_paths["sleeve_metrics_json"]
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
result.output_paths["weights_csv"]
result.output_paths["portfolio_returns_csv"]
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
result.output_paths["manifest_json"]
result.output_paths["pipeline_metrics_json"]
result.output_paths["lineage_json"]
result.output_paths["state_json"]
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
result.output_paths["checkpoint_json"]
result.output_paths["summary_json"]
result.extra["stage_statuses"]
result.extra["stage_execution"]
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
result.output_paths["scenario_catalog_json"]
result.output_paths["scenario_matrix_csv"]
result.output_paths["expansion_preflight_json"]
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
result.output_paths["summary_json"]
result.output_paths["manifest_json"]
result.output_paths["inventory_json"]
result.output_paths["benchmark_matrix_csv"]
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

Use `to_dict()` when you want a JSON-safe summary:

```python
payload = result.to_dict()
payload["artifact_dir"]
payload["metrics"]
```

By default, `to_dict()` excludes `raw_result` because raw workflow results may
contain DataFrames, models, or other non-JSON objects. Pass
`include_raw_result=True` only for local debugging.

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

## Example Script

See
[`docs/examples/notebook_execution_api_examples.py`](examples/notebook_execution_api_examples.py)
for compact Python functions that demonstrate strategy, alpha evaluation, full
alpha, portfolio, pipeline, and campaign calls through the public execution API.
