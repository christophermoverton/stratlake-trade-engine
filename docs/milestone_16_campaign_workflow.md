# Milestone 16 Research Campaign Workflow

## Overview

Milestone 16 adds a campaign-orchestration layer above the existing alpha,
strategy, candidate-selection, portfolio, and review CLIs.

Its job is not to replace those lower-level entrypoints. Instead, it gives you
one deterministic config and one top-level run id for a multi-stage workflow
that should share:

* dataset and timeframe assumptions
* target names and config paths
* artifact roots and output destinations
* preflight validation before expensive work starts
* a stitched summary for automation, audit, and review

Use this doc as the main practical guide for Milestone 16. For the config
contract, see [research_campaign_configuration.md](research_campaign_configuration.md).
For committed example configs, see
[examples/milestone_16_campaign_workflow.md](examples/milestone_16_campaign_workflow.md).
For the resume/retry/reuse flow added afterward, see
[milestone_17_resume_workflow.md](milestone_17_resume_workflow.md).

## Where It Fits

The campaign runner sits above the milestone-level CLIs:

```text
campaign config
    ->
preflight
    ->
alpha and strategy execution
    ->
comparison
    ->
candidate selection
    ->
portfolio
    ->
candidate review
    ->
unified review
    ->
campaign summary
```

In practical repository terms, `python -m src.cli.run_research_campaign` can
coordinate these existing surfaces:

* `src.cli.run_alpha`
* `src.cli.run_strategy`
* `src.cli.compare_alpha`
* `src.cli.compare_strategies`
* `src.cli.run_candidate_selection`
* `src.cli.run_portfolio`
* `src.cli.review_candidate_selection`
* `src.cli.compare_research`

The campaign layer does not define a second artifact contract for those stages.
Each stage still writes its normal run-local outputs under its existing
artifact root. Milestone 16 adds one stitched campaign directory on top.

## What To Run

### Quick start

Run the default repository campaign config:

```powershell
python -m src.cli.run_research_campaign
```

Use an explicit config path when you want a named workflow:

```powershell
python -m src.cli.run_research_campaign --config docs/examples/data/milestone_16_campaign_configs/full_campaign.yml
```

That command:

* resolves repository defaults, config values, and inherited campaign settings
* writes `campaign_config.json` before execution starts
* runs campaign preflight
* exits early if preflight fails
* executes enabled stages in deterministic order
* writes one stitched `manifest.json` and `summary.json` for the campaign

## Stage Order

When enabled, the runner processes stages in this order:

1. `preflight`
2. `research`
3. `comparison`
4. `candidate_selection`
5. `portfolio`
6. `candidate_review`
7. `review`

Interpretation notes:

* `research` runs alpha targets first, then strategy targets
* `comparison` may run alpha comparison, strategy comparison, or both
* `candidate_review` runs only when candidate selection is enabled and
  `candidate_selection.execution.enable_review` is `true`
* `review` always uses the unified review CLI and writes a review pack, even if
  some upstream stages were intentionally skipped

## Shared Campaign Inputs

Milestone 16 is most useful when several stages should inherit the same
research context.

The common shared sections are:

* `dataset_selection`
  Shared dataset, timeframe, evaluation horizon, mapping name, and optional
  ticker file.
* `time_windows`
  Shared bounded single-run window plus optional train/predict windows for
  alpha workflows.
* `targets`
  Alpha, strategy, and portfolio names plus config paths.
* `outputs`
  Shared artifact roots for campaign, alpha, candidate-selection, portfolio,
  comparison, and review outputs.

The loader then fans those values back into lower-level stages. For example:

* candidate selection can inherit dataset, timeframe, evaluation horizon,
  mapping name, and a single alpha target
* portfolio can inherit a single portfolio target and shared timeframe
* unified review can inherit dataset, timeframe, and single-target filters

That inheritance keeps campaign files short while still preserving a fully
normalized `campaign_config.json` in the saved artifact directory.

## Example Workflow

One practical Milestone 16 workflow is:

1. run one or more alpha targets over a shared train/predict window
2. run one or more strategy targets over the same research window
3. compare saved alpha and strategy outputs
4. select governed alpha candidates
5. build one candidate-driven portfolio
6. generate candidate review outputs
7. create one registry-backed unified review pack
8. inspect the campaign summary instead of opening every stage directory first

Example invocation:

```powershell
python -m src.cli.run_research_campaign --config docs/examples/data/milestone_16_campaign_configs/full_campaign.yml
```

This is the right shape when you want one top-level research packet rather than
manually stitching together several CLI runs.

## Example Campaign Configs

Committed examples live under:

```text
docs/examples/data/milestone_16_campaign_configs/
```

### Full end-to-end campaign

Use
[examples/data/milestone_16_campaign_configs/full_campaign.yml](examples/data/milestone_16_campaign_configs/full_campaign.yml)
when you want the campaign runner to execute fresh research, comparison,
candidate selection, portfolio construction, candidate review, and unified
review in one pass.

This example demonstrates:

* shared `features_daily` dataset context
* shared time windows for alpha and strategy work
* fresh alpha and strategy execution
* enabled comparison stage
* enabled candidate-selection stage with review outputs
* portfolio chaining from the candidate-selection artifact
* unified review output under one review directory

### Registry-chained downstream campaign

Use
[examples/data/milestone_16_campaign_configs/registry_chained_campaign.yml](examples/data/milestone_16_campaign_configs/registry_chained_campaign.yml)
when upstream alpha work already exists and you want the campaign to resolve a
candidate-selection artifact from the registry, then continue with portfolio
construction and unified review.

This example demonstrates:

* disabled fresh candidate-selection execution
* explicit matching keys for candidate-selection registry resolution
* portfolio chaining from a resolved candidate-selection artifact
* review-only downstream orchestration without rerunning the whole stack

## Expected Artifacts

Every campaign writes one top-level directory:

```text
artifacts/research_campaigns/<campaign_run_id>/
```

Core files:

* `campaign_config.json`
* `checkpoint.json`
* `preflight_summary.json`
* `manifest.json`
* `summary.json`

### `checkpoint.json`

This is the canonical persisted stage-state contract for resumable campaign
execution.

It records all seven campaign stages in canonical order and persists one
normalized state per stage:

* `completed`: stage finished in this campaign run
* `failed`: stage ran and failed; resume can restart from this boundary
* `skipped`: stage was intentionally disabled or not applicable
* `reused`: stage inputs were resolved from an existing artifact or registry
  entry instead of being recomputed
* `partial`: stage emitted incomplete resumable state and has not finished yet
* `pending`: stage has not run yet or is blocked by an upstream failure

Each stage entry also carries deterministic `selected_run_ids`,
`key_metrics`, `output_paths`, `outcomes`, `details`, plus `terminal` and
`resumable` flags so orchestration tooling can decide whether to continue,
retry, or inspect reused artifacts.

Each stage entry now also persists:

* `fingerprint_inputs`: the canonical stage-defining inputs after campaign
  defaults, inheritance, and upstream chaining are resolved
* `input_fingerprint`: a deterministic SHA-256 hash of those canonical
  `fingerprint_inputs`

The checkpoint root mirrors those values in `stage_input_fingerprints` so
orchestration code can compare the current effective inputs with prior
campaign state without reparsing every stage payload.

### Fingerprint Rules

The fingerprints intentionally capture the inputs that define stage behavior,
not incidental bookkeeping like campaign artifact file names.

Current rules:

* `preflight` fingerprints the fully normalized campaign config because
  preflight validates the whole effective workflow contract.
* `research` fingerprints shared dataset/time-window context, alpha and
  strategy target selection, catalog paths, and the alpha artifact root used
  by downstream stages.
* `comparison` fingerprints the normalized comparison config plus the upstream
  `research` fingerprint.
* `candidate_selection` fingerprints the candidate-selection config after
  removing non-defining output and registration knobs, then chains in the
  upstream `research` fingerprint and any resolved reference metadata.
* `portfolio` fingerprints the normalized portfolio config plus the upstream
  `candidate_selection` fingerprint or resolved candidate-selection reference.
* `candidate_review` fingerprints the review-mode knobs for candidate review
  plus the upstream candidate-selection and portfolio fingerprints.
* `review` fingerprints the normalized unified-review config, the registry
  artifact roots it reads from, and the upstream comparison/candidate-review
  fingerprints.

That means orchestration can safely reuse a prior stage only when the stored
`input_fingerprint` exactly matches the fingerprint for the current effective
inputs for that stage.

### Reuse Policy Controls

Campaign configs can now override the default "reuse on exact fingerprint
match" behavior through `reuse_policy`:

* `enable_checkpoint_reuse: false` disables checkpoint reuse entirely
* `reuse_prior_stages` whitelists the stages that may restore matching
  checkpoints
* `force_rerun_stages` forces selected stages to execute again even when their
  fingerprints match
* `invalidate_downstream_after_stages` cascades a rerun into every later stage
  in the same campaign pass

Every stage persists the applied decision under `details.reuse_policy` so
operators can see whether a stage reused prior work, reran because of policy,
or reran because an upstream stage invalidated downstream reuse.

### `campaign_config.json`

This is the normalized effective config after default loading, inheritance, and
path normalization.

Use it to answer:

* which targets were selected
* which outputs were configured
* which dataset and time windows were actually used
* whether a stage inherited campaign-level values or set them explicitly

### `preflight_summary.json`

This is the fail-fast campaign gate.

It records:

* overall preflight status
* check counts
* passed, warning, and failed checks
* stage dependencies that could not be resolved

If preflight fails, this file is the first place to inspect. Milestone 16
still writes the campaign manifest and summary on preflight failure so
automation can read a structured status without scraping stderr.

### `summary.json`

This is the main machine-readable stitched output for the campaign.

It includes:

* `stage_statuses`
* `checkpoint`
* ordered `stages`
* `selected_run_ids`
* `key_metrics`
* `output_paths`
* `final_outcomes`

Use it to answer:

* which stage completed, failed, or was skipped
* which alpha, strategy, candidate-selection, portfolio, and review run ids
  were selected
* where each downstream artifact directory was written
* what the campaign-level review outcome was

### `manifest.json`

This is the campaign inventory file.

It records:

* the core campaign artifact set
* the relative checkpoint path
* stage statuses
* selected run ids
* campaign targets
* the relative summary path

Use it when you need a deterministic file-level entrypoint to the campaign
directory itself.

## Stage-Local Artifacts Still Matter

Milestone 16 adds a stitched layer, but it does not flatten or replace the
existing stage contracts.

You should still expect the normal stage-local outputs:

* alpha runs under `artifacts/alpha/<run_id>/`
* strategy runs under `artifacts/strategies/<run_id>/`
* comparisons under their comparison artifact roots
* candidate-selection runs under `artifacts/candidate_selection/<run_id>/`
* portfolio runs under `artifacts/portfolios/<run_id>/`
* candidate review outputs under `artifacts/reviews/candidate_selection_<run_id>/`
* unified review outputs under `artifacts/reviews/<review_id>/`

The campaign `summary.json` points back to those outputs through
`output_paths`, so a normal review flow is:

1. open campaign `summary.json`
2. find the stage or run id you care about
3. jump into that run-local directory
4. inspect that run's own `manifest.json`, metrics, QA, and promotion outputs

## Preflight And Dependency Resolution

Campaign preflight exists to catch orchestration mistakes before expensive work
starts.

Current preflight checks validate:

* target names and config-path resolution
* alpha, strategy, and portfolio catalog availability
* required feature dataset roots
* writability of configured artifact roots
* registry availability for registry-backed campaign stages
* downstream dependency resolution such as candidate-selection or portfolio
  chaining

Practical examples:

* if candidate selection is enabled and more than one alpha target is present,
  preflight can fail unless `candidate_selection.alpha_name` resolves to one
  concrete value
* if portfolio chaining expects a candidate-selection artifact from the
  registry, preflight checks that a unique matching candidate-selection run can
  actually be found

## When To Use Campaign Orchestration

Use `run_research_campaign` when:

* you want one reproducible end-to-end research packet with one top-level run id
* several stages should share the same dataset, timeframe, and output roots
* you want preflight validation before triggering multiple expensive stages
* automation or CI should consume one stitched `summary.json`
* you need an auditable handoff from upstream research through downstream
  review

Good Milestone 16 use cases:

* nightly or milestone-level research sweeps
* governed alpha-to-portfolio workflows
* repeatable review-packet generation
* environments where a failed downstream dependency should stop the whole run

## When To Use Lower-Level CLIs

Use the lower-level CLIs directly when:

* you are iterating on one alpha or strategy and do not want full orchestration
* you are debugging one stage in isolation
* you only need one comparison or one review refresh
* you want custom one-off CLI flags that are not naturally shared at the
  campaign layer
* upstream artifacts already exist and manual inspection matters more than one
  stitched campaign record

Common examples:

* use `src.cli.run_alpha` to tune one alpha model quickly
* use `src.cli.compare_strategies` to benchmark a few strategies without
  candidate selection or portfolio work
* use `src.cli.run_candidate_selection` when you are adjusting gates and
  redundancy thresholds repeatedly
* use `src.cli.compare_research` when you only need a new review pack from
  existing registries

The practical rule is:

* choose campaign orchestration for repeatable multi-stage workflows
* choose lower-level CLIs for tight local iteration and stage-specific debugging

## Reading A Campaign Result

One practical review pass is:

1. open `preflight_summary.json` and confirm the campaign passed preflight
2. open `summary.json` and check `stage_statuses`
3. inspect `selected_run_ids` to see which concrete runs were produced or
   resolved
4. use `output_paths` to jump into the relevant alpha, strategy,
   candidate-selection, portfolio, candidate-review, or unified-review
   artifact directory
5. inspect the downstream run-local manifests and metrics for deeper analysis

That keeps Milestone 16 review fast without losing access to the stage-level
detail that still matters.

## Related Docs

* [../README.md](../README.md)
* [research_campaign_configuration.md](research_campaign_configuration.md)
* [alpha_workflow.md](alpha_workflow.md)
* [alpha_evaluation_workflow.md](alpha_evaluation_workflow.md)
* [strategy_evaluation_workflow.md](strategy_evaluation_workflow.md)
* [milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md)
* [milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md)
* [milestone_15_candidate_selection_issue_1.md](milestone_15_candidate_selection_issue_1.md)
* [milestone_16_merge_readiness.md](milestone_16_merge_readiness.md)
* [milestone_17_resume_workflow.md](milestone_17_resume_workflow.md)
* [examples/milestone_16_campaign_workflow.md](examples/milestone_16_campaign_workflow.md)
* [examples/milestone_17_resume_workflow.md](examples/milestone_17_resume_workflow.md)
* [examples/real_world_resume_workflow_case_study.md](examples/real_world_resume_workflow_case_study.md)
