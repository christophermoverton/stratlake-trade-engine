# Research Campaign Configuration

For the practical end-to-end workflow guide, see
[milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md).

`src.config.research_campaign` defines one normalized contract for campaign-level
research settings that span alpha comparison, strategy comparison, candidate
selection, portfolio construction, and unified review.

## Contract

The campaign schema supports these sections:

* `dataset_selection`: shared dataset, timeframe, evaluation horizon,
  mapping name, and optional ticker file
* `time_windows`: shared outer, train, and predict windows
* `targets`: alpha, strategy, and portfolio names plus catalog/config paths
* `reuse_policy`: operator-facing checkpoint reuse controls for explicit
  stage reuse, forced reruns, global reuse disablement, and optional
  downstream invalidation after a rerun
* `comparison`: registry-backed comparison settings and ranking controls
* `candidate_selection`: governed-candidate filters, thresholds, allocation,
  execution toggles, and artifact destinations
* `portfolio`: portfolio execution target and high-level run settings
* `review`: unified research review filters, ranking, outputs, and promotion
  gate wiring through the existing `ReviewConfig` contract
* `outputs`: common artifact roots and shared output destinations, including
  the campaign artifact root used for persisted campaign manifests, summaries,
  and preflight reports

The repository example lives at `configs/research_campaign.yml`.

The default config now uses the canonical Milestone 16 field names such as
`min_mean_ic`, `max_pairwise_correlation`, `allocation_method`, and
`max_weight_per_candidate`. Backward-compatible aliases still resolve in code,
but docs and new configs should prefer the canonical names.

## Precedence

Use `resolve_research_campaign_config(...)` to merge layers deterministically:

```text
repository defaults < config sources < CLI overrides
```

This matches the repository's existing config precedence pattern used by review,
runtime, and execution settings.

## Normalization

The loader applies a few shared defaults so one campaign file can stay concise:

* `candidate_selection.dataset`, `timeframe`, `evaluation_horizon`, and
  `mapping_name` inherit from `dataset_selection` when omitted
* `candidate_selection.alpha_name` inherits from `targets.alpha_names` when
  exactly one alpha target is listed
* `portfolio.portfolio_name` inherits from `targets.portfolio_names` when
  exactly one portfolio target is listed
* `review.filters.dataset`, `timeframe`, and single-target alpha/strategy/
  portfolio filters inherit from the campaign-wide sections
* candidate-selection and review output paths inherit from `outputs` when their
  local sections omit explicit paths
* `candidate_selection.artifacts_root` inherits from
  `outputs.alpha_artifacts_root` when left at the default `artifacts/alpha`
* string lists are trimmed, deduplicated, and preserved in input order
* path-like strings are normalized to forward-slash form for stable manifests
* `reuse_policy` stage lists are normalized against the canonical campaign
  stage order: `preflight`, `research`, `comparison`, `candidate_selection`,
  `portfolio`, `candidate_review`, and `review`

## Reuse Policy

`reuse_policy` makes campaign resume behavior explicit instead of purely
implicit:

* `enable_checkpoint_reuse`: when `false`, every stage reruns even if a
  matching checkpoint exists
* `reuse_prior_stages`: whitelist of stages allowed to restore a matching
  checkpoint
* `force_rerun_stages`: stages that must rerun even when their checkpoint
  fingerprint matches
* `invalidate_downstream_after_stages`: when one of these stages reruns, all
  later stages rerun in the same campaign pass

Default behavior remains unchanged: checkpoint reuse stays enabled, every stage
is eligible for reuse, no stages are force-rerun, and downstream invalidation
is opt-in.

## Preflight

`python -m src.cli.run_research_campaign` now runs a campaign preflight stage
before research execution begins.

Preflight validates:

* target resolution and cross-stage config dependencies
* alpha, strategy, and portfolio catalog availability
* required feature datasets and their parquet-backed roots
* writable artifact roots for campaign, alpha, candidate-selection, portfolio,
  comparison, and review outputs
* registry availability for registry-backed campaign stages that require
  pre-existing inputs

Each campaign persists:

* `campaign_config.json`
* `checkpoint.json`
* `preflight_summary.json`
* `manifest.json`
* `summary.json`

under `outputs.campaign_artifacts_root/<campaign_run_id>/`.

If preflight fails, the runner exits before expensive execution starts and the
written `preflight_summary.json` records the failing checks. The campaign-level
`manifest.json` and `summary.json` are also still emitted so automation can
inspect the failed stage state without parsing exception text.

## Campaign Artifacts

The campaign directory now acts as the top-level stitched artifact surface for
the full workflow:

* `checkpoint.json`: canonical resumable campaign state with one persisted
  stage-state record for each of `preflight`, `research`, `comparison`,
  `candidate_selection`, `portfolio`, `candidate_review`, and `review`
* `manifest.json`: deterministic file inventory and stage/run index for the
  campaign artifact directory itself
* `summary.json`: machine-readable stitched campaign output with:
  * stage state for preflight, research, comparison, candidate selection,
    portfolio, candidate review, and unified review
  * selected run IDs and comparison/review IDs
  * key alpha, strategy, candidate-selection, portfolio, and review metrics
  * output file paths for downstream stage artifacts
  * final review and promotion outcomes when review promotion gates are present

The canonical checkpoint stage states are:

* `completed`
* `failed`
* `skipped`
* `reused`
* `partial`
* `pending`

`summary.json` is intended for automation, orchestration, and audit tooling,
while `manifest.json` is the stable inventory entry point for the campaign
directory and `checkpoint.json` is the resumable execution contract.

Each executed or reused stage also persists `details.reuse_policy`, which
records the deterministic reuse decision that was applied for that stage,
including whether a checkpoint matched, whether fingerprints matched, whether
the stage was invalidated by an upstream rerun, and the exact reason for the
final reuse vs rerun choice.

## Loading

```python
from src.config.research_campaign import resolve_research_campaign_config

config = resolve_research_campaign_config(
    {"research_campaign": {"comparison": {"enabled": True}}},
    cli_overrides={"portfolio": {"enabled": True}},
)
```

Use `load_research_campaign_config()` when you only want repository defaults or
one YAML/JSON file.
