# Milestone 16 Research Campaign Workflow Example

This example shows how one checked-in campaign config can coordinate the
existing research stack through one top-level campaign artifact directory.

Use this as the compact companion to the main workflow guide:
[../milestone_16_campaign_workflow.md](../milestone_16_campaign_workflow.md).

## Config Location

Committed example campaign configs live under:

```text
docs/examples/data/milestone_16_campaign_configs/
```

Included examples:

* [data/milestone_16_campaign_configs/full_campaign.yml](data/milestone_16_campaign_configs/full_campaign.yml)
* [data/milestone_16_campaign_configs/registry_chained_campaign.yml](data/milestone_16_campaign_configs/registry_chained_campaign.yml)

## Run It

Full end-to-end example:

```powershell
python -m src.cli.run_research_campaign --config docs/examples/data/milestone_16_campaign_configs/full_campaign.yml
```

Registry-chained downstream example:

```powershell
python -m src.cli.run_research_campaign --config docs/examples/data/milestone_16_campaign_configs/registry_chained_campaign.yml
```

## What It Demonstrates

* one normalized campaign config driving several existing CLIs
* campaign preflight before research execution
* shared dataset, target, and output-path inheritance
* stitched campaign artifacts:
  * `campaign_config.json`
  * `preflight_summary.json`
  * `manifest.json`
  * `summary.json`
* downstream stage chaining through saved artifact directories and registries

## Why The Example Stays Config-Focused

This companion intentionally ships configs instead of a large committed output
tree. That keeps the example easy to review in git while still showing the
intended Milestone 16 orchestration shape.

Use the main guide for output interpretation and stage-selection guidance.
