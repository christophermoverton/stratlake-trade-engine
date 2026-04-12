# Real-World Resume Workflow Case Study (Milestone 17)

## Objective

Demonstrate a realistic operator resume flow for campaign orchestration:

1. a campaign reaches `comparison`
2. the operator interrupts the run
3. the next pass resumes from the persisted checkpoint
4. a later identical rerun reuses every recovered stage

This case study focuses on campaign operational behavior rather than expensive
real-data model execution, so it uses deterministic stage stubs while still
driving the real campaign runner and real stitched artifact contracts.

## Execute

```powershell
python docs/examples/real_world_resume_workflow_case_study.py
```

## Campaign Config

Checked-in config set:

```text
docs/examples/data/milestone_17_campaign_configs/
```

Primary campaign file:

```text
docs/examples/data/milestone_17_campaign_configs/resume_reuse_campaign.yml
```

## Workflow Surfaces Used

* `src.cli.run_research_campaign`
* `src.cli.run_alpha`
* `src.cli.run_strategy`
* `src.cli.compare_alpha`
* `src.cli.compare_strategies`
* `src.cli.run_candidate_selection`
* `src.cli.run_portfolio`
* `src.cli.review_candidate_selection`
* `src.cli.compare_research`

## Output Location

The example writes all artifacts under:

```text
docs/examples/output/real_world_resume_workflow_case_study/
```

Primary stitched case-study summary:

```text
docs/examples/output/real_world_resume_workflow_case_study/summary.json
```

Native campaign summary:

```text
docs/examples/output/real_world_resume_workflow_case_study/artifacts/research_campaigns/<campaign_run_id>/summary.json
```

Snapshot copies for the three phases:

```text
docs/examples/output/real_world_resume_workflow_case_study/snapshots/
```

## Outputs To Inspect

* case-study summary:
  * `summary.json`
* partial-pass snapshots:
  * `snapshots/partial_summary.json`
  * `snapshots/partial_manifest.json`
  * `snapshots/partial_checkpoint.json`
* resumed-pass snapshots:
  * `snapshots/resumed_summary.json`
  * `snapshots/resumed_manifest.json`
  * `snapshots/resumed_checkpoint.json`
* stable reused-pass snapshots:
  * `snapshots/stable_summary.json`
  * `snapshots/stable_manifest.json`
  * `snapshots/stable_checkpoint.json`

## Observed Results (Validated 2026-04-12)

### Campaign

* campaign run id: `research_campaign_7327a8915d50`
* same campaign run id reused across all three passes: `true`
* interrupted stage: `comparison`
* comparison attempts: `2`

### Partial Pass

* campaign status: `partial`
* stage statuses:
  * `preflight=completed`
  * `research=completed`
  * `comparison=partial`
  * `candidate_selection=pending`
  * `portfolio=pending`
  * `candidate_review=pending`
  * `review=pending`
* `partial_stage_names`: `["comparison"]`
* `resumable_stage_names`:
  * `comparison`
  * `candidate_selection`
  * `portfolio`
  * `candidate_review`
  * `review`
* comparison failure kind: `interrupted`

### Resumed Pass

* campaign status: `completed`
* stage statuses:
  * `preflight=reused`
  * `research=reused`
  * `comparison=completed`
  * `candidate_selection=completed`
  * `portfolio=completed`
  * `candidate_review=completed`
  * `review=completed`
* `retry_stage_names`: `["comparison"]`
* comparison retry metadata:
  * `attempted=true`
  * `previous_state=partial`
  * reuse flag stayed `false` for the resumed comparison execution

### Stable Reused Pass

* campaign status: `completed`
* stage statuses:
  * all seven stages are `reused`
* `retry_stage_names`: `["comparison"]`
* `reused_stage_names`:
  * `preflight`
  * `research`
  * `comparison`
  * `candidate_selection`
  * `portfolio`
  * `candidate_review`
  * `review`
* comparison execution metadata now shows:
  * `state=reused`
  * `retry.attempted=true`
  * `retry.previous_state=partial`
  * `reuse.reused=true`

## Notes

* The case study keeps the same campaign root across all three passes so the
  checkpoint, manifest, and summary tell one continuous recovery story.
* The snapshot files are intentionally committed because they make the resume
  transitions easy to review in git without rerunning the example first.
* The committed JSON outputs keep file references relative to the case-study
  output root so the example stays portable across machines and repository
  locations.
* The stable reused pass still records the comparison retry history, which is
  the main Milestone 17 auditability improvement.
