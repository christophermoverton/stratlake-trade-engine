# Milestone 18 Milestone Review Workflow

## Overview

Milestone 18 adds one more stitched layer on top of the campaign artifacts
introduced in Milestone 16 and the resume metadata added in Milestone 17:
deterministic milestone review packs.

The goal is simple. After a campaign finishes, operators should be able to
open one small pack and answer three practical questions without re-reading
every stage directory:

* what happened in the campaign
* which decisions were accepted, deferred, or rejected
* which upstream artifacts justify those decisions

Use this guide as the Milestone 18 companion to
[milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md),
[milestone_17_resume_workflow.md](milestone_17_resume_workflow.md), and
[milestone_reporting_artifacts.md](milestone_reporting_artifacts.md).

## What Changed

When `research_campaign.milestone_reporting.enabled` is `true`, campaign runs
now auto-generate a milestone pack after the stitched campaign
`summary.json` and `manifest.json` are written.

That pack lives under the campaign artifact directory:

```text
artifacts/research_campaigns/<campaign_run_id>/milestone_report/
    summary.json
    decision_log.json
    manifest.json
    report.md
```

The same pack can also be generated later from an already-existing campaign
artifact directory:

```powershell
python -m src.cli.generate_milestone_report --campaign-artifact-path artifacts/research_campaigns/<campaign_run_id>
```

That second path is useful when:

* the campaign finished before Milestone 18 fields were documented
* you want to regenerate only the milestone pack from stable campaign outputs
* you want a custom milestone name, title, or output directory

## Automatic Campaign Integration

Milestone reporting is enabled by default in
[configs/research_campaign.yml](configs/research_campaign.yml).

The campaign runner passes these controls into milestone generation:

* `milestone_reporting.output.include_markdown_report`
* `milestone_reporting.output.decision_log_render_formats`
* `milestone_reporting.sections`
* `milestone_reporting.summary`
* `milestone_reporting.decision_categories`

That means the campaign flow stays artifact-first:

1. run preflight and enabled campaign stages
2. persist campaign `checkpoint.json`, `summary.json`, and `manifest.json`
3. derive a milestone review pack from those saved campaign artifacts
4. add milestone artifact paths back into the campaign stitched outputs

In practice, Milestone 18 does not replace the campaign summary. It gives you
an operator-facing review packet built from it.

## Artifact Locations

There are two related locations to understand.

The campaign root:

```text
artifacts/research_campaigns/<campaign_run_id>/
```

The milestone review pack inside it:

```text
artifacts/research_campaigns/<campaign_run_id>/milestone_report/
```

The campaign `summary.json` and `manifest.json` record the milestone outputs
through:

* `output_paths.milestone_report_dir`
* `output_paths.milestone_report_summary`
* `output_paths.milestone_report_decision_log`
* `output_paths.milestone_report_manifest`
* `output_paths.milestone_report_markdown`

The milestone pack then links back to upstream campaign and review artifacts
through relative paths in `related_artifacts`, `evidence_artifacts`, and
`source_artifacts`.

## How To Read The Outputs

### `summary.json`

Open this first.

It is the main machine-readable review payload and answers:

* which milestone this pack represents
* whether the pack is `draft` or `final`
* how many decisions were recorded
* the counts by decision status
* the high-level findings, recommendations, and open questions

Practical fields to check first:

* `status`
* `summary`
* `decision_count`
* `decision_counts_by_status`
* `key_findings`
* `recommendations`
* `related_artifacts`

### `decision_log.json`

Open this when you need the auditable decision trail.

It answers:

* which decisions were made
* why each decision was made
* which source artifacts support each decision
* which follow-up actions remain

Practical fields to inspect:

* `decision_ids`
* `decisions[*].status`
* `decisions[*].summary`
* `decisions[*].rationale`
* `decisions[*].follow_up_actions`
* `decisions[*].source_artifacts`

### `report.md`

Open this when you want a human-readable brief for milestone review meetings,
handoffs, or PR context.

By default it renders:

* campaign scope
* selected runs
* key findings and key metrics
* gate outcomes
* risks
* next steps
* open questions
* a decision snapshot
* related artifacts

### `manifest.json`

Open this when you want the deterministic inventory of the pack itself.

It is most useful for:

* automation entrypoints
* checking whether `report.md` was intentionally omitted
* verifying the exact file set persisted for the review packet

## Practical Review Workflow

One practical Milestone 18 review flow is:

1. open campaign `summary.json`
2. confirm the campaign status and identify the concrete `campaign_run_id`
3. jump into `milestone_report/summary.json`
4. read `decision_counts_by_status` and `key_findings`
5. open `decision_log.json` for the accepted, deferred, or rejected items you
   need to discuss
6. follow the linked relative artifact paths into the campaign, candidate
   review, or research review outputs when you need deeper evidence
7. use `report.md` as the meeting-ready brief once the machine-readable review
   looks correct

In practice:

* campaign `summary.json` tells you what ran
* milestone `summary.json` tells you what matters for review
* `decision_log.json` tells you why the review landed there

## Interpreting Common Outcomes

Some common reading patterns:

* `decision_counts_by_status.accepted > 0`
  The pack found milestone decisions that are ready to stand as written.
* `decision_counts_by_status.deferred > 0`
  The campaign artifacts are usable, but follow-up work or missing evidence
  still matters before final promotion or signoff.
* `related_artifacts.review_promotion_gates` missing
  The review stage may have completed without a persisted promotion-gate
  payload, so the milestone pack will usually surface an open question or a
  deferred review decision instead of an approved promotion outcome.
* `report_markdown_path` is `null`
  The canonical JSON pack still exists, but standalone Markdown output was
  intentionally disabled.

## Generating From An Existing Campaign

To build a milestone pack from a saved campaign directory:

```powershell
python -m src.cli.generate_milestone_report `
  --campaign-artifact-path docs/examples/output/real_world_campaign_case_study/artifacts/research_campaigns/<campaign_run_id> `
  --milestone-name "Milestone 18" `
  --title "Milestone 18 Readiness Report"
```

You can also point `--campaign-artifact-path` directly at that campaign
directory's `summary.json`.

The CLI prints:

* the resolved campaign artifact path
* the resulting milestone `summary.json` path

## Real-World Example

For a committed campaign example using real `features_daily` data, see
[examples/real_world_campaign_case_study.md](examples/real_world_campaign_case_study.md).

That case study shows:

* the existing campaign flow that produces the upstream artifacts
* where the auto-generated milestone pack is written
* how to interpret the milestone summary and decision log
* how to use those artifacts in a practical milestone review pass

## Related Docs

* [../README.md](../README.md)
* [milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md)
* [milestone_17_resume_workflow.md](milestone_17_resume_workflow.md)
* [milestone_reporting_artifacts.md](milestone_reporting_artifacts.md)
* [research_campaign_configuration.md](research_campaign_configuration.md)
* [examples/real_world_campaign_case_study.md](examples/real_world_campaign_case_study.md)
