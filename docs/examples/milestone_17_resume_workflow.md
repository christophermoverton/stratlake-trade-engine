# Milestone 17 Resume Workflow Example

This example set shows the operator-facing resume story added in Milestone 17.

Use this as the compact companion to
[../milestone_17_resume_workflow.md](../milestone_17_resume_workflow.md).

## Config Location

Committed example campaign files live under:

```text
docs/examples/data/milestone_17_campaign_configs/
```

Included files:

* [data/milestone_17_campaign_configs/resume_reuse_campaign.yml](data/milestone_17_campaign_configs/resume_reuse_campaign.yml)
* [data/milestone_17_campaign_configs/resume_reuse_alphas.yml](data/milestone_17_campaign_configs/resume_reuse_alphas.yml)
* [data/milestone_17_campaign_configs/resume_reuse_strategies.yml](data/milestone_17_campaign_configs/resume_reuse_strategies.yml)
* [data/milestone_17_campaign_configs/resume_reuse_portfolios.yml](data/milestone_17_campaign_configs/resume_reuse_portfolios.yml)

## Run It

```powershell
python docs/examples/real_world_resume_workflow_case_study.py
```

## What It Demonstrates

* a campaign that stops with one `partial` comparison stage
* downstream stages left `pending` and resumable after the interrupted pass
* a second pass that reuses `preflight` and `research`, retries
  `comparison`, then completes the remaining stages
* a third pass that reuses every stage from the recovered stable checkpoint
* committed snapshot artifacts for partial, resumed, and stable
  `summary.json`, `manifest.json`, and `checkpoint.json`

## Output Tree

The example writes under:

```text
docs/examples/output/real_world_resume_workflow_case_study/
```

Key files:

* `summary.json`
* `snapshots/partial_summary.json`
* `snapshots/resumed_summary.json`
* `snapshots/stable_summary.json`
* `snapshots/partial_manifest.json`
* `snapshots/resumed_manifest.json`
* `snapshots/stable_manifest.json`
