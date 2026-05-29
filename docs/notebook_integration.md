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

For a copy-friendly Colab walkthrough that keeps notebook CWD, project root,
external MarketLake root, and mounted Drive root explicit, see
[`docs/colab_project_sessions.md`](colab_project_sessions.md).

For fresh runtime recovery, that guide now includes a restore-first M44 pattern
using `ARCHIVE_ID` / `ARCHIVE_ROOT`, dry-run validation and inspection, then
intentional `stratlake-session-archive-restore-bootstrap` execution into an
explicit local `/content/...` workspace.

For M44 root stability in Colab, define a single profile cell once (for
`FINTECH_ROOT`, `STRATLAKE_ROOT`, `MARKETLAKE_ROOT`, `DRIVE_ROOT`, `START`,
`END`, `UNIVERSE_CONFIG`, and `PATHS_CONFIG`) and pass those variables through
all later commands. Prefer explicit `--root`, `--marketlake-root`, and
`--drive-root` arguments over notebook-CWD-dependent command behavior.

For read-only preflight diagnostics across roots/configs/universe/Drive/archive
markers and secret presence, run `stratlake-notebook-doctor --json` with
explicit roots. The command does not mutate `.env`/`os.environ`, call Google
APIs, or run hidden sync.

For restored Colab sessions, use the M44.6 notebook-native execution pattern
in `docs/colab_project_sessions.md`: keep CLI for setup/restore/doctor/handoff
flows, then switch to `src.execution` APIs (`run_strategy`, result helpers) for
interactive execution and artifact inspection without hard-coded run IDs.

For session-first notebooks, use `stratlake-init-session`. It delegates to the
same notebook bootstrap initializer, then writes `.stratlake/session.json` and
`.stratlake/path_resolution.json`:

```powershell
stratlake-init-session --root ./stratlake-notebooks --project-name stratlake-demo
```

Add `--notebook-configs` when you want deterministic starter files for
`configs/paths.yml`, `configs/universe.yml`, and `configs/tickers_sample.txt`.
Existing bundle files are preserved by default; use
`--force-notebook-configs` to overwrite only those three files.

Use `stratlake-init-notebook` when you only need the workspace layout and
starter templates. Use `stratlake-init-session` when the notebook CWD, project
root, MarketLake root, or optional Drive root may differ and should be recorded
explicitly.

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

`stratlake-init-session --enable-drive-persistence --drive-root ...` records
Drive persistence intent as metadata only. It does not perform Google Drive
sync, import, export, backup, OAuth, or API calls.

Notebook cells can consume session metadata through path helpers:

```python
from src.session import find_session_root, load_session, resolve_session_paths

root = find_session_root()
session = load_session(root)
paths = resolve_session_paths(session)

configs_root = paths["configs_root"].resolved_path
artifacts_root = paths["artifacts_root"].resolved_path
marketlake_root = paths["marketlake_root"].resolved_path
```

`resolve_session_paths(...)` applies explicit overrides first, then session
metadata, then environment-variable fallbacks such as `MARKETLAKE_ROOT`, and
then starter defaults. Any environment fallback is recorded with provenance;
the helpers do not mutate CWD, `.env`, `os.environ`, Drive files, or canonical
artifacts.

## Filesystem Drive Persistence

Mounted Drive persistence is available through explicit one-shot filesystem
copy commands:

```powershell
stratlake-session-export --root ./stratlake-notebooks --drive-root ./mounted-drive/stratlake-demo --include-configs
stratlake-session-import --root ./stratlake-notebooks --drive-root ./mounted-drive/stratlake-demo --include-configs
```

The Drive root is a local filesystem path. StratLake does not use Google APIs,
OAuth, credentials, network access, background sync, or automatic backup.
Copies are persistence snapshots and remain non-authoritative.

Session metadata is included by default. Configs, contracts, docs, artifacts,
and derived artifacts are opt-in. Feature data requires `--include-features`;
MarketLake data requires `--include-market-data`. `.env`, obvious
credential/secret/API-key files, notebook checkpoints, caches, bytecode, and
temporary files are excluded by default.

Use `--dry-run` to see the deterministic plan without copying files. Import
preserves existing files unless `--force` is supplied. Non-dry-run operations
write a manifest under `artifacts/_derived/notebook_sessions/...`.

Use `--operation-id` when you want distinct historical manifests instead of
reusing `latest`.

For the M42 release scope, validation checklist, and architecture boundaries,
see [`docs/m42_release_notes.md`](m42_release_notes.md) and
[`docs/m42_release_validation_checklist.md`](m42_release_validation_checklist.md).

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
| Feature building | `cli.build_features.run_cli` | `python -m cli.build_features` or `stratlake-build-features` | feature run `summary.json`, feature datasets, QA rollups | Use `--marketlake-root` for a cell-local curated-data root override. |
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

Feature-builder cells can use the same CLI-equivalent parser while returning
the summary artifact path. Environment-driven setup:

```python
import os

os.environ["MARKETLAKE_ROOT"] = "data/curated"

from cli.build_features import run_cli

summary_path = run_cli([
    "--timeframe", "1D",
    "--start", "2025-01-01",
    "--end", "2025-02-01",
    "--tickers", "configs/tickers_50.txt",
])
```

For a visible cell-local override, pass `--marketlake-root` directly:

```python
from cli.build_features import run_cli

summary_path = run_cli([
    "--timeframe", "1D",
    "--start", "2025-01-01",
    "--end", "2025-02-01",
    "--tickers", "configs/tickers_50.txt",
    "--marketlake-root", "data/curated",
])
```

The feature builder resolves the curated root as `--marketlake-root`,
`MARKETLAKE_ROOT`, then `configs/paths.yml`. The direct override is local to
that run and does not mutate `.env`, `configs/paths.yml`, or global process
configuration. The feature-run `summary.json` records the effective root and
source under `config_resolution`.

## Validate Curated Data Before Feature Builds

Run the handoff validator between curated-data restoration and feature builds:

```bash
stratlake-validate-marketlake-handoff \
    --root "{STRATLAKE_ROOT}" \
    --marketlake-root "{MARKETLAKE_ROOT}" \
    --universe "{UNIVERSE_CONFIG}" \
    --start "{START}" \
    --end "{END}" \
    --timeframe 1D \
    --json
```

The validator is read-only. It fails early on missing curated roots, archive
pack directories, missing symbols, missing date-window coverage, or notebook
profile/config mismatches.

Maintenance note: keep this `--marketlake-root` / `config_resolution`
guidance synchronized with the notebook workspace starter-template copy at
`src/resources/notebook_workspace/docs/notebook_integration.md`, because
`stratlake-init-notebook` copies that file into newly created workspaces.

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
