# Notebook Integration

## Purpose

Notebook integration makes StratLake easier to use for interactive research,
inspection, teaching, and review while preserving the same artifact contracts
used by CLI, pipeline, validation, and orchestrated workflows. Use this guide
with `docs/notebook_execution_api.md`, `docs/concurrency_and_idempotency.md`,
and `docs/pipeline_integration.md`.

Notebooks are an interactive surface over the existing StratLake execution
system. They should run established workflows, inspect returned
`ExecutionResult` objects, and read persisted artifacts. The source of truth is
the artifact root, manifest, summaries, metrics, inventories, and named output
paths, not hidden notebook cell state.

## Core Principle

StratLake has one execution system with multiple entry points. Notebook
workflows should call existing `src.execution` APIs or CLI-equivalent paths and
should not reimplement workflow logic.

Notebook examples must not duplicate strategy, alpha, portfolio, pipeline,
benchmark-pack, campaign, validation, regime, or stress-test behavior. They
should remain thin cells that import StratLake APIs, pass explicit inputs, and
inspect canonical outputs.

## Local Notebook Workspace Bootstrap

Use the installed bootstrap command to initialize a local notebook workspace
under an explicit root path:

```powershell
stratlake-init-notebook --root ./stratlake-notebooks
```

This creates local `notebooks/`, `configs/`, `docs/`, `contracts/`, and
`artifacts/` directories, copies a curated starter allowlist from repository
`configs/` and `docs/`, and skips existing files by default. Use `--force` to
overwrite copied starter templates only.

See [`docs/notebook_workspace_bootstrap.md`](notebook_workspace_bootstrap.md)
for command reference, installed CLI entry points, and package/workspace
boundary guidance.

## Notebook Project Sessions

Notebook project sessions make root selection explicit without becoming a
second source of truth for artifacts. The `src.session` package can create
`.stratlake/session.json` and `.stratlake/path_resolution.json` under a selected
project root. Those files record the notebook CWD, StratLake project root,
`configs/`, `artifacts/`, `data/curated` feature root, optional external
MarketLake root, optional Drive root, and path-resolution provenance.

Session metadata is diagnostic state for notebook ergonomics. Canonical
workflow state remains in manifests, summaries, metrics, inventories, and named
artifact outputs.

## Existing Notebook Surface

The public notebook-friendly execution surface is documented in
[`docs/notebook_execution_api.md`](notebook_execution_api.md). Verified
top-level imports from `src.execution` include:

- `run_strategy`
- `compare_strategies`
- `run_alpha`
- `run_alpha_evaluation`
- `run_portfolio`
- `run_pipeline`
- `run_research_campaign`
- `run_campaign`
- `run_benchmark_pack`
- `run_docs_path_lint`
- `run_deterministic_rerun_validation`
- `run_milestone_validation`
- `ExecutionResult`
- `load_json_artifact`

Specialized regime and market-simulation execution wrappers also exist under
module paths such as `src.execution.regime_benchmark`,
`src.execution.regime_policy_stress_tests`,
`src.execution.regime_promotion_gates`,
`src.execution.regime_review_pack`, and
`src.execution.market_simulation`. These are CLI-equivalent wrappers around
their corresponding research and CLI surfaces.

Every notebook-facing workflow should return or inspect an `ExecutionResult`.
The stable contract includes `workflow`, `run_id`, `name`, `artifact_dir`,
`manifest_path`, `metrics`, `output_paths`, `extra`, and `raw_result`.

Use named output helpers instead of reconstructing filenames:

```python
result.notebook_summary()
result.output_keys()
result.output_path("summary_json", must_exist=True)
result.load_manifest()
result.load_metrics_json()
result.load_summary_json("summary_json")
```

For explicit JSON artifacts that are not named outputs, use
`load_json_artifact(path)` with a path obtained from a manifest or
`ExecutionResult`.

Run status markers from the M28 artifact-safety work may appear in reused roots:
`_RUNNING.json`, `_SUCCESS.json`, and `_FAILED.json`. They are safety hints for
active, completed, or failed roots. Manifests and workflow summaries remain the
canonical contracts.

Pipeline and external-orchestrator examples should follow
[`docs/pipeline_integration.md`](pipeline_integration.md): wrap existing CLI or
`src.execution` surfaces, pass explicit paths, and avoid scheduler-specific
workflow logic in notebooks.

## Notebook Rerun Safety

Notebook cells are easy to rerun, so output-root isolation must be explicit:

- Use explicit output roots where the called API supports them.
- Use one logical notebook run per output root.
- Include run ids, attempt ids, or cell labels for repeated cells.
- Avoid writing repeatedly into broad shared roots such as `artifacts/qa/` or
  `artifacts/research_campaigns/`.
- When reusing roots intentionally, inspect `_RUNNING.json`, `_SUCCESS.json`,
  `_FAILED.json`, manifests, summaries, checkpoints, and inventories before
  deciding whether to resume or rerun.
- Prefer read-only artifact inspection cells after execution.

Good notebook roots include:

```python
"artifacts/notebooks/m28_benchmark_pack_execution_api/attempt_001"
"artifacts/notebooks/m28_regime_research_inspection/attempt_001"
```

Some workflows, such as benchmark packs and research campaigns, document
checkpoint or reuse behavior. Other workflows should be treated
conservatively: use a fresh root or the workflow's default deterministic output
layout rather than writing into a shared root from multiple notebook kernels.

## CLI / API / Notebook Mapping

| Workflow | Notebook API | CLI Equivalent | Primary Artifacts | Notes |
| --- | --- | --- | --- | --- |
| Strategy execution | `src.execution.run_strategy` | `python -m src.cli.run_strategy` | `manifest.json`, `metrics.json`, `qa_summary.json`, `equity_curve.csv` | Uses the same strategy runner and returns `ExecutionResult`. |
| Strategy comparison | `src.execution.compare_strategies` | `python -m src.cli.compare_strategies` | leaderboard CSV, summary JSON, optional comparison JSON | Compare persisted strategy outputs through the shared comparison surface. |
| Alpha evaluation | `src.execution.run_alpha_evaluation` | `python -m src.cli.run_alpha_evaluation` | `manifest.json`, `alpha_metrics.json`, predictions, signals | Config mapping or config path resolves through CLI-equivalent helpers. |
| Full alpha run | `src.execution.run_alpha` | `python -m src.cli.run_alpha` | evaluation artifacts, mapped signals, sleeve metrics, scaffold | Use notebooks for inspection, not for custom signal-mapping logic. |
| Portfolio execution | `src.execution.run_portfolio` | `python -m src.cli.run_portfolio` | `manifest.json`, `metrics.json`, weights, returns, equity curve | Consumes completed component artifacts or registry-backed inputs. |
| Pipeline execution | `src.execution.run_pipeline` | `python -m src.cli.run_pipeline` | `manifest.json`, `pipeline_metrics.json`, `lineage.json`, `state.json` | Delegates to `PipelineSpec.from_yaml(...)` and `PipelineRunner.run()`. |
| Benchmark-pack execution | `src.execution.run_benchmark_pack` | `python -m src.cli.run_benchmark_pack` | summary, manifest, checkpoint, inventory, batch plan, benchmark matrix | Supports explicit `output_root` and documented checkpoint/reuse semantics. |
| Research-campaign execution | `src.execution.run_research_campaign` | `python -m src.cli.run_research_campaign` | campaign config, checkpoint, manifest, summary, preflight summary | Preserves campaign stage ordering, preflight, checkpoints, and reuse policy. |
| Deterministic-rerun validation | `src.execution.run_deterministic_rerun_validation` | `python -m src.cli.run_deterministic_rerun_validation` | validation report JSON | Notebook API returns an inspectable result even when CLI policy would exit non-zero. |
| Milestone-validation bundle | `src.execution.run_milestone_validation` | `python -m src.cli.run_milestone_validation` | bundle summary, docs/path lint report, deterministic-rerun report | Use CLI for release validation; use notebooks for interactive report review. |
| Docs/path lint | `src.execution.run_docs_path_lint` | `python -m src.cli.run_docs_path_lint` | docs/path lint JSON report | Useful for notebook inspection of path-portability failures. |
| Regime benchmark pack | `src.execution.regime_benchmark.run_regime_benchmark_pack` | `python -m src.cli.run_regime_benchmark_pack` | benchmark matrix, model/calibration/policy comparisons, summaries, manifest | Specialized regime wrapper, not exported from top-level `src.execution`. |
| Regime review pack | `src.execution.regime_review_pack.generate_regime_review_pack` | `python -m src.cli.generate_regime_review_pack` | leaderboard, decision log, review summary, evidence index, manifest | Review artifact inspection should stay artifact-first. |
| Regime policy stress tests | `src.execution.regime_policy_stress_tests.run_regime_policy_stress_tests` | `python -m src.cli.run_regime_policy_stress_tests` | stress matrices, scenario summaries, policy stress summary, manifest | Diagnostic research evidence, not production readiness evidence. |
| Market simulation scenarios | `src.execution.market_simulation.run_market_simulation_scenarios` | `python -m src.cli.run_market_simulation_scenarios` | scenario catalogs, inventories, manifests, replay/bootstrap/Monte Carlo outputs | Simulation outputs are diagnostics, not forecasts. |

Only use rows whose configs and input artifacts exist for the notebook you are
running. Do not invent configs inside notebooks.

## Canonical Notebook Patterns

Import StratLake APIs inside a cell or helper function:

```python
from src.execution import run_benchmark_pack

result = run_benchmark_pack(
    "configs/benchmark_packs/m22_scale_repro.yml",
    output_root="artifacts/notebooks/m28_benchmark_pack_execution_api/attempt_001",
    stop_after_batches=1,
)
```

Inspect the `ExecutionResult` before loading larger artifacts:

```python
result.notebook_summary()
result.output_keys()
```

Load canonical artifacts through named outputs:

```python
summary = result.load_summary_json("summary_json")
inventory = result.load_output_json("inventory_json")
matrix_path = result.output_path("benchmark_matrix_csv", must_exist=True)
```

Compare notebook outputs to CLI-produced artifacts by comparing stable
machine-readable contracts: workflow, run id, manifest references, named output
keys, metrics, summaries, inventories, and matrix files. Do not compare object
identity, transient stdout, absolute workspace prefixes, or hidden notebook
state.

Use pandas, matplotlib, or similar libraries only for inspection and display.
Do not move workflow logic, artifact schema construction, signal generation,
portfolio construction, policy selection, or validation decisions into notebook
cells.

Keep notebook outputs deterministic and clearable. A clean notebook or
script-backed notebook example should be runnable from a fresh Python process
with explicit inputs.

## Examples

Canonical M28 notebook-style examples live under `docs/examples/notebooks/`:

- [`docs/examples/notebooks/m28_benchmark_pack_execution_api.ipynb`](examples/notebooks/m28_benchmark_pack_execution_api.ipynb)
- [`docs/examples/notebooks/m28_strategy_execution_api.py`](examples/notebooks/m28_strategy_execution_api.py)
- [`docs/examples/notebooks/m28_benchmark_pack_execution_api.py`](examples/notebooks/m28_benchmark_pack_execution_api.py)
- [`docs/examples/notebooks/m28_regime_research_inspection.py`](examples/notebooks/m28_regime_research_inspection.py)

The `.ipynb` file is output-free and lightweight. The `.py` files are
script-backed notebook examples that are intentionally import-safe and can be
copied into Jupyter cells without requiring Jupyter as a repository runtime
dependency.

## Limitations

Notebooks are not a replacement for CI, release validation, benchmark-pack
validation, milestone-validation bundles, or command-line automation.

Notebook execution does not prove production readiness, live deployment
readiness, future performance, or operational risk control.

Full cross-layer CLI/API/pipeline/notebook parity validation is deferred to
M28.5.

The unified capstone case study is deferred to M28.6.

Optional `.ipynb` execution depends on the local environment. This repository
does not add Jupyter as a required runtime dependency for M28.4.
