# Milestone 17 Campaign Resume Workflow

## Overview

Milestone 17 extends the campaign orchestration layer with operator-facing
resume behavior that is explicit in persisted artifacts, not just implicit in
checkpoint reuse.

The campaign runner now persists enough state to answer four practical
questions from `summary.json`, `manifest.json`, and `checkpoint.json` alone:

* did the campaign stop on a `failed` or `partial` stage
* which stages are still resumable from the saved checkpoint
* which stage retried after a prior failed or interrupted pass
* which stages were later reused from the recovered stable run

Use this guide as the Milestone 17 companion to
[milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md). For
the config contract, see
[research_campaign_configuration.md](research_campaign_configuration.md). For a
checked-in operator example, see
[examples/real_world_resume_workflow_case_study.md](examples/real_world_resume_workflow_case_study.md).

## What Changed

Milestone 17 keeps the same seven canonical campaign stages:

1. `preflight`
2. `research`
3. `comparison`
4. `candidate_selection`
5. `portfolio`
6. `candidate_review`
7. `review`

The change is in what gets persisted around those stages.

Each stage now carries richer stitched metadata in both the campaign
`summary.json` and `manifest.json`:

* `stage_execution`
  Per-stage operational metadata for resume, retry, reuse, skip, failure, and
  fingerprint inspection.
* `execution_metadata`
  The same per-stage view embedded directly inside each `summary.json` stage
  entry.
* `retry_stage_names`
  Stages that ran after a prior `failed` or `partial` checkpoint state.
* `partial_stage_names`
  Stages interrupted before completion.
* `reused_stage_names`
  Stages restored from a matching checkpoint instead of rerunning.
* `resumable_stage_names`
  Stages whose saved checkpoint state is still resumable.
* `failures`
  Compact failure summaries that mirror the persisted stage failure payloads.

## Stage-State Model

The canonical checkpoint stage states remain:

* `completed`
* `reused`
* `failed`
* `partial`
* `skipped`
* `pending`

Milestone 17 makes the operational meaning of those states more explicit:

* `partial`
  The stage started, emitted resumable checkpoint state, and was interrupted
  before completion. This is the expected state for `KeyboardInterrupt` or
  operator-driven stop/restart flows.
* `failed`
  The stage ended with an exception and did not complete. The campaign still
  writes stitched artifacts so the failure is inspectable without rerunning.
* `reused`
  The stage did not execute in the current pass because a matching checkpoint
  was restored.
* `pending`
  The stage did not run yet. After a partial stop, downstream stages usually
  remain pending and resumable.

## Resume And Retry Metadata

`stage_execution.<stage_name>` now exposes four operational subviews:

* `resume`
  Whether the persisted checkpoint state is resumable, terminal, and what the
  saved checkpoint state/reason were.
* `retry`
  Whether the stage is running after a prior `failed` or `partial` attempt,
  including the previous state and previous failure payload when present.
* `reuse`
  Whether the current pass reused checkpoint state and the exact
  reuse-policy/fingerprint decision behind that choice.
* `failure`
  A structured failure record with exception type, message, kind, and
  retryability.

That means operators can distinguish these three similar-looking but different
cases:

* resumed after an interrupted partial stage
* reran after a failed stage
* reused the recovered stable stage on later identical reruns

## Reading The Artifacts

One practical Milestone 17 review flow is:

1. open campaign `summary.json`
2. inspect `final_outcomes.partial_stage_names`,
   `retry_stage_names`, and `reused_stage_names`
3. inspect `stage_execution.<stage_name>.resume` and `.retry` for the stage you
   care about
4. open `checkpoint.json` when you need canonical resumable state and stage
   fingerprints
5. open `manifest.json` when automation needs the same operational view from a
   deterministic top-level inventory file

In practice:

* `summary.json` is the richest stitched operational surface
* `checkpoint.json` is the canonical resumable contract
* `manifest.json` mirrors the same state in a deterministic file-inventory
  entrypoint for automation

## Real-World Resume Example

The committed example case study intentionally simulates this flow:

1. complete `preflight` and `research`
2. interrupt `comparison`, producing a `partial` campaign
3. rerun the same campaign root and resume from that checkpoint
4. rerun one more time and confirm every stage is now `reused`

Run it with:

```powershell
python docs/examples/real_world_resume_workflow_case_study.py
```

The example writes:

* one top-level case-study `summary.json`
* one native campaign artifact directory under
  `docs/examples/output/real_world_resume_workflow_case_study/artifacts/research_campaigns/<campaign_run_id>/`
* snapshot copies of the partial, resumed, and stable stitched
  `summary.json`, `manifest.json`, and `checkpoint.json` payloads under
  `docs/examples/output/real_world_resume_workflow_case_study/snapshots/`

## Observed Example Results (Validated 2026-04-12)

The committed resume case study shows:

* partial pass:
  `comparison=partial`, downstream stages `pending`, and
  `resumable_stage_names` includes `comparison`, `candidate_selection`,
  `portfolio`, `candidate_review`, and `review`
* resumed pass:
  `preflight` and `research` reused, `comparison` retried from prior
  `partial`, and downstream stages completed
* stable pass:
  all seven stages reused on an identical rerun, while
  `retry_stage_names` still records `comparison` for auditability

The important Milestone 17 behavior is that retry history survives into the
stable reused run. The stage no longer needs to rerun, but the stitched
artifacts still remember that it previously recovered from a partial state.

## Related Docs

* [../README.md](../README.md)
* [milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md)
* [research_campaign_configuration.md](research_campaign_configuration.md)
* [examples/milestone_17_resume_workflow.md](examples/milestone_17_resume_workflow.md)
* [examples/real_world_resume_workflow_case_study.md](examples/real_world_resume_workflow_case_study.md)
* [milestone_16_merge_readiness.md](milestone_16_merge_readiness.md)
