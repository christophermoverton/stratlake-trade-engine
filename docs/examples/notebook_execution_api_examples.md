# Canonical Notebook Execution API Examples

## Purpose

This is the canonical notebook-oriented companion to
[`notebook_execution_api_examples.py`](notebook_execution_api_examples.py). It
shows how to run the Milestone 23 Python-facing execution APIs without shelling
out to CLI commands, then inspect the same deterministic artifacts that CLI
workflows write.

Use these examples as notebook cells. The Python file is import-safe: importing
it defines helper functions only. Calling a function executes the corresponding
workflow and writes artifacts under the normal repository output roots.

## Import The Stable API

```python
from src.execution import (
    ExecutionResult,
    compare_strategies,
    run_alpha,
    run_alpha_evaluation,
    run_benchmark_pack,
    run_docs_path_lint,
    run_pipeline,
    run_portfolio,
    run_research_campaign,
    run_strategy,
)
```

Each entrypoint returns an `ExecutionResult`. Treat execution and inspection as
separate steps: first run the workflow, then use named output keys and helper
methods to inspect the artifacts.

```python
result = run_strategy("momentum_v1", start="2022-01-01", end="2023-01-01")

result.notebook_summary()
result.output_keys()
result.output_path("metrics_json", must_exist=True)
result.load_manifest()
result.load_metrics_json()
```

## CLI Or Notebook

Use notebooks for exploration, interactive inspection, comparative analysis,
teaching, and one-off research review. They are a good fit when you want to
inspect manifests, metrics, summaries, validation reports, or benchmark
inventories immediately after a run.

Use the CLI for operational runs, automation, CI, milestone validation, release
bundles, scheduled benchmark packs, and workflows where process exit behavior is
part of the contract. Notebook support complements the CLI; it does not replace
the CLI as the operational interface.

## Canonical Cells

The companion Python module defines a small set of notebook cells:

* `strategy_notebook_cell()` runs a strategy and inspects `metrics.json`,
  `manifest.json`, and `equity_curve.csv`.
* `strategy_comparison_notebook_cell()` compares strategy runs and inspects the
  leaderboard summary and CSV path.
* `alpha_evaluation_notebook_cell()` evaluates an alpha and inspects
  `alpha_metrics.json`, predictions, and the manifest.
* `alpha_full_run_notebook_cell()` runs the full alpha surface and inspects
  mapped signals plus sleeve metrics.
* `portfolio_notebook_cell(component_run_ids)` builds a portfolio from explicit
  completed components and inspects weights and returns.
* `registry_backed_portfolio_notebook_cell()` shows registry-backed portfolio
  construction when completed component runs already exist.
* `pipeline_notebook_cell()` runs a YAML pipeline and inspects metrics, lineage,
  and state handoff.
* `research_campaign_notebook_cell()` runs one staged campaign and inspects the
  campaign summary, checkpoint, and stage statuses.
* `scenario_orchestration_notebook_cell()` shows scenario-enabled campaign
  orchestration and inspects scenario catalog and matrix artifacts.
* `validation_notebook_cell()` runs docs/path lint, deterministic rerun
  validation, and the milestone validation bundle while keeping reports
  inspectable even when status is failed.
* `benchmark_pack_notebook_cell()` runs a benchmark pack and inspects the
  summary, inventory, benchmark matrix, and optional comparison.

Example notebook usage:

```python
from docs.examples.notebook_execution_api_examples import (
    benchmark_pack_notebook_cell,
    inspect_result,
    strategy_notebook_cell,
    validation_notebook_cell,
)

strategy_result, strategy_view = strategy_notebook_cell()
strategy_view["sharpe_ratio"]
inspect_result(strategy_result)

validation_results = validation_notebook_cell()
validation_results["docs_path_lint"][1]["report"]

benchmark_result, benchmark_view = benchmark_pack_notebook_cell()
benchmark_view["summary"]["status"]
```

For a full `.ipynb` case-study artifact, open
[`ml_cross_sectional_xgb_2026_q1_notebook.ipynb`](ml_cross_sectional_xgb_2026_q1_notebook.ipynb).
It walks through the Q1 2026 XGBoost alpha-to-portfolio workflow using the same
stable execution API and `ExecutionResult` inspection helpers.

## Inspection Patterns

Prefer named outputs over reconstructed filenames:

```python
metrics = result.load_metrics_json()
summary = result.load_summary_json("summary_json")
manifest = result.load_manifest()
outputs = result.output_keys()
```

Check optional outputs explicitly:

```python
if result.has_output("comparison_json") and result.output_path("comparison_json").exists():
    comparison = result.load_comparison_json()
```

Use explicit artifact loading when a manifest points to a JSON file that is not
already a named output:

```python
from src.execution import load_json_artifact

manifest = result.load_manifest()
relative_path = manifest["artifact_paths"]["summary"]
payload = load_json_artifact(result.artifact_path(relative_path, must_exist=True))
```

## Parity Expectations

Notebook and CLI runs use the same underlying workflow logic. The stable
contracts are the machine-readable ones: run ids, artifact directories,
manifests, registries, metrics, reports, benchmark inventories, and summary
schemas. Representative CLI/API tests validate those contracts across strategy,
alpha evaluation, portfolio, pipeline, validation, and benchmark-pack flows.

Some process behavior intentionally differs. For example, validation CLIs can
exit non-zero when a report fails, while notebook APIs return an
`ExecutionResult` so the failed report can be inspected before the notebook
decides whether to raise.
