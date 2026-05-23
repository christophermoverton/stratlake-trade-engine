# Concurrency and Idempotency

## Purpose

M28 hardens existing StratLake execution surfaces for repeated execution,
notebook reruns, external orchestrator retries, shared output roots, and
partial metadata writes.

This work reinforces the existing CLI, `src.execution`, `src.pipeline`,
research-campaign, benchmark-pack, and validation surfaces. It does not add a
second pipeline framework, a replacement orchestration layer, or new workflow
semantics.

## Supported Guarantees

StratLake uses simple file-system safeguards:

- Artifact roots can be checked before writes with conservative collision
  policy.
- Metadata JSON/text writes use temp-file replacement where updated surfaces
  call shared artifact-safety helpers.
- Updated run roots can expose status markers:
  `_RUNNING.json`, `_SUCCESS.json`, and `_FAILED.json`.
- Completed roots are not considered successful until the final success marker
  or complete manifest/summary has been written.
- Interrupted or incomplete roots can be detected by missing success markers,
  failed markers, running markers, checkpoints, partial summaries, or
  incomplete manifests.

These guarantees are local file-system guarantees. They are intended for
research workflows, CI jobs, notebooks, and external scheduler retries that use
clear output-root isolation.

## Non-Goals

M28 does not provide distributed locking, a database, Redis, a queue, a daemon,
or a service. It does not require Airflow, Prefect, Dagster, or any other
orchestration dependency. It does not make every artifact writer in the
repository transactionally atomic.

## Output-Root Isolation

The safest pattern is one logical run per output root:

- Pipeline artifacts write below `artifacts/pipelines/<pipeline_run_id>/`.
- Research campaigns write below the configured `campaign_artifacts_root`.
- Benchmark packs write below the configured benchmark-pack output root.
- Validation reports write below the configured `artifacts/qa/` paths.

For external orchestrator retries, prefer a unique output root per scheduler
attempt unless the workflow explicitly supports checkpoint/resume. Good retry
roots include a scheduler run id, attempt id, or timestamp supplied by the
orchestrator.

## Repeated Runs

Some StratLake workflows intentionally support deterministic repeated runs:

- Research campaigns can reuse matching checkpoint stages when
  `reuse_policy.enable_checkpoint_reuse` is enabled.
- Benchmark packs can resume partial batches and reuse completed batch
  checkpoints.
- Deterministic rerun validation intentionally creates first/second run roots
  and may be rerun against the same validation workdir.

When a workflow does not document checkpoint/resume semantics, repeated writes
to a non-empty output root should be treated as a collision and should fail
fast or use a new output root.

## Concurrent Runs

Concurrent runs should not share the same output root unless a workflow
explicitly documents compatible checkpoint/reuse behavior. The M28 file-system
markers make active, failed, and completed roots visible, but they are not a
distributed lock and do not prevent all races between independent processes.

If two processes target the same root at the same time, the supported guidance
is to stop one run, inspect status markers and manifests, then rerun with a
fresh output root or an explicit resume path.

## Notebook Reruns

Notebook cells should pass explicit output roots for rerunnable examples. Use
`src.execution` helpers to run workflows and inspect `ExecutionResult`
artifacts, manifests, summaries, and named output paths.

For exploratory notebooks, prefer roots such as:

- `artifacts/notebooks/<notebook_name>/<run_id>/`
- `artifacts/notebooks/<notebook_name>/<cell_label>/attempt_<n>/`

Avoid repeatedly writing into broad shared roots such as `artifacts/qa/` or
`artifacts/research_campaigns/` unless the called workflow has documented
checkpoint/resume behavior.

## External Orchestrator Retries

Airflow, Prefect, Dagster, CI, and scheduler usage should call existing CLI
commands or `src.execution` APIs. They should not reimplement pipeline logic or
introduce a parallel execution layer.

Recommended retry behavior:

- Use unique output roots for each orchestrator attempt by default.
- Pass the same root only for workflows that explicitly support
  checkpoint/resume.
- Treat `_RUNNING.json` without `_SUCCESS.json` as an active or interrupted
  run requiring inspection.
- Treat `_FAILED.json` or partial checkpoints as resumable only when the
  workflow documents resume semantics.

## Checkpoint/Resume Compatibility

Checkpoint-aware workflows may opt into reuse behavior:

- Research campaigns validate stage fingerprints before reusing stages.
- Benchmark packs validate batch fingerprints and existing summaries before
  reusing batches.
- Validation and deterministic rerun commands preserve their existing stable
  rerun patterns.

Reuse is explicit workflow behavior. It is not the default assumption for an
arbitrary non-empty artifact root.

## Manifest And Marker Semantics

Updated surfaces write metadata artifacts atomically where practical and use
status markers on representative run roots.

Marker meanings:

- `_RUNNING.json`: a run has started and has not yet written its final marker.
- `_SUCCESS.json`: the run completed and final metadata was written.
- `_FAILED.json`: the run failed or stopped in a partial state.

Manifests and summaries remain the canonical workflow-specific artifact
contracts. Markers are lightweight safety hints and should not replace
manifest validation.

## Collision Policies

The shared artifact-safety helper supports:

- `fail`: allow missing or empty roots, and fail on non-empty roots.
- `reuse`: allow an existing root for workflows with explicit resume or
  checkpoint semantics.

The conservative default is `fail`. Workflows that already support
checkpoint/resume can opt into `reuse` while preserving their own validation
rules.

## Limitations

M28 hardening is intentionally local and practical. It does not prove
distributed concurrency safety, transactional multi-file commits, or full
cross-layer parity for every optional workflow. Broader parity validation and
external orchestrator examples are deferred to later M28 issues.
