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
from src.execution import run_strategy, run_alpha, run_alpha_evaluation, run_portfolio
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

## `ExecutionResult` Contract

Notebook entrypoints return `ExecutionResult` with these stable fields:

* `workflow`: workflow family, such as `"strategy"`, `"alpha"`,
  `"alpha_evaluation"`, or `"portfolio"`.
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
  portfolio timeframe, allocator, or split count.
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
alpha, and portfolio calls through the public execution API.
