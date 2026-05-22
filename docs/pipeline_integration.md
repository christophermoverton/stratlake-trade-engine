# Pipeline Integration

## Purpose

StratLake already has native execution and pipeline surfaces for deterministic
research workflows. This guide explains how external orchestrators such as
Airflow, Prefect, and Dagster can call those existing surfaces without adding a
second StratLake workflow layer.

The examples in `docs/examples/pipelines/` are integration patterns. They show
how a scheduler task can invoke the same CLI modules or `src.execution` APIs
used by notebooks, validation, and repository tests.

## Core Principle

StratLake has one execution system with multiple entry points. Airflow,
Prefect, and Dagster examples must be thin wrappers around existing CLI or
`src.execution` calls.

External orchestrator code should not:

- introduce a new internal pipeline abstraction
- duplicate benchmark-pack, research-campaign, validation, or pipeline logic
- bypass existing `src.execution` APIs or CLI entrypoints
- add Airflow, Prefect, or Dagster as required runtime dependencies

## Supported Existing Surfaces

Verified `src.execution` surfaces include:

- `src.execution.run_pipeline`
- `src.execution.run_benchmark_pack`
- `src.execution.run_research_campaign`
- `src.execution.run_deterministic_rerun_validation`
- `src.execution.run_milestone_validation`

Verified CLI entrypoints include:

```powershell
python -m src.cli.run_pipeline --config configs/test_pipeline.yml
python -m src.cli.run_benchmark_pack --config configs/benchmark_packs/m22_scale_repro.yml
python -m src.cli.run_research_campaign --config configs/research_campaign.yml
python -m src.cli.run_deterministic_rerun_validation
python -m src.cli.run_milestone_validation
```

Prefer the Python API when the scheduler task wants an `ExecutionResult` with
named artifact paths. Prefer the CLI when process exit behavior, stdout/stderr,
or shell-friendly automation is the main contract.

## Corporate-Actions Dividend Evidence Step

M40 dividend evidence can be used in pipeline-style workflows as an explicit
local-file import step. It should remain a thin call into
`src.corporate_actions.import_dividend_events` or the matching CLI wrapper; it
should not be hidden inside strategy, alpha, portfolio, or backtest execution.

Python pattern:

```python
from src.corporate_actions import import_dividend_events


def import_dividend_evidence_step() -> dict[str, object]:
    result = import_dividend_events(
        source_data_path="data/external/corporate_actions/dividends/dividends.parquet",
        source_metadata_path="data/external/corporate_actions/dividends/metadata.json",
        output_root="data/curated/events/dividends",
        artifact_root="artifacts/corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
        strict=True,
    )
    return result.to_dict()
```

CLI pattern:

```bash
python -m src.cli.import_corporate_actions_dividends \
  --source-data data/external/corporate_actions/dividends/dividends.parquet \
  --source-metadata data/external/corporate_actions/dividends/metadata.json \
  --output-root data/curated/events/dividends \
  --artifact-root artifacts/corporate_actions \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --strict
```

This step consumes local upstream artifacts only. It does not call live APIs,
require credentials, adjust price bars, reconstruct adjusted prices, model
dividend reinvestment, or alter downstream research outputs. Generated import
artifacts are evidence context for review and catalog discovery.

See [Corporate Actions Dividend Evidence](corporate_actions_dividend_evidence.md)
and `docs/examples/m40_dividend_pipeline_step_example.py`.

## Artifact Safety and Idempotency

External orchestrators should follow
[`docs/concurrency_and_idempotency.md`](concurrency_and_idempotency.md):

- use a unique output root for each orchestrator attempt by default
- opt into explicit reuse only for workflows that document checkpoint/resume
  behavior
- do not share broad output roots across concurrent jobs
- include scheduler run ids and attempt ids in output roots
- inspect `_RUNNING.json`, `_SUCCESS.json`, and `_FAILED.json` markers when a
  root may have been reused or interrupted
- keep manifests, summaries, checkpoints, and inventories as the canonical
  workflow contracts; marker files are safety hints, not replacements

Research campaigns and benchmark packs have explicit checkpoint/reuse
semantics. Other workflows should be treated conservatively: rerun with a new
root unless the workflow documents safe reuse.

## Environment Guidance

Run orchestrated StratLake tasks from the repository root so relative config
paths resolve consistently. Use isolated virtual environments for scheduler
workers and keep scheduler-specific dependencies outside the core StratLake
runtime unless they are installed as optional tooling.

Set `MARKETLAKE_ROOT`, `ARTIFACTS_ROOT`, or other repository-supported runtime
configuration variables only where the called workflow expects them. Keep CLI
commands deterministic by passing explicit config paths, output roots, and
run identifiers where the surface supports them.

Do not rely on notebook state, imported module globals, or mutable process
state from a prior scheduler task. Each task should be able to run from a clean
Python process with explicit inputs.

## Airflow Pattern

Use a `PythonOperator` or `@task` to call a small local function. Import
StratLake inside that callable so the DAG can still be parsed when a scheduler
environment only needs to inspect the DAG structure.

The callable should pass explicit output roots where possible, usually derived
from Airflow `dag_run.run_id`, `task_instance.try_number`, or another attempt
identifier. Keep heavy workflow logic out of the DAG definition and avoid
placing Airflow-specific code in core StratLake modules.

See
[`docs/examples/pipelines/m28_airflow_regime_research_dag.py`](examples/pipelines/m28_airflow_regime_research_dag.py).

## Prefect Pattern

Use a Prefect `@task` as a thin call into `src.execution` or a CLI helper. A
`@flow` can compose one or more StratLake tasks, but it should not reimplement
StratLake workflow stages.

Use unique output roots per flow run or task run. Pass the same root only for
checkpoint/resume-aware workflows where reuse is intentional and documented.

See
[`docs/examples/pipelines/m28_prefect_regime_research_flow.py`](examples/pipelines/m28_prefect_regime_research_flow.py).

## Dagster Pattern

Use a Dagster `@op` to call an existing `src.execution` or CLI surface. A
`@job` can compose ops, while StratLake remains responsible for benchmark-pack,
research-campaign, pipeline, and validation logic.

Keep ops thin, pass explicit paths, and avoid duplicating StratLake pipeline
logic inside Dagster graphs.

See
[`docs/examples/pipelines/m28_dagster_regime_research_job.py`](examples/pipelines/m28_dagster_regime_research_job.py).

## Validation Guidance

Examples should remain import-safe without Airflow, Prefect, or Dagster
installed. The repository examples expose plain Python fallback callables so
unit tests can import them and inspect thin-wrapper behavior without optional
scheduler packages.

Tests can skip scheduler-object assertions when optional packages are absent.
They should still verify that fallback callables exist, optional imports are
guarded, and docs/path lint remains clean.

## Limitations

These examples are integration patterns, not production scheduler deployments.
They do not provide distributed locking, scheduler installation guidance, or a
full deployment topology.

Orchestrator retry safety depends on output-root isolation and the semantics
of the called workflow. Full cross-layer output parity is deferred to M28.5.
The capstone case study is deferred to M28.6.
