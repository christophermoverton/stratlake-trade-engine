# Research Campaign Configuration

For the practical end-to-end workflow guide, see
[milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md).
For the operator-facing resume/retry/reuse flow, see
[milestone_17_resume_workflow.md](milestone_17_resume_workflow.md).

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
* `milestone_reporting`: campaign milestone artifact controls for decision
  categories, decision-log render formats, report section toggles, summary
  behavior, and optional `report.md` emission
* `outputs`: common artifact roots and shared output destinations, including
  the campaign artifact root used for persisted campaign manifests, summaries,
  and preflight reports
* `scenarios`: optional parameter-sweep and scenario-matrix expansion contract
  for generating multiple resolved campaign variants from one campaign spec

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
* milestone reporting defaults to the full artifact pack, but can selectively
  disable `report.md`, filter decision categories, and trim report sections
* string lists are trimmed, deduplicated, and preserved in input order
* path-like strings are normalized to forward-slash form for stable manifests
* `reuse_policy` stage lists are normalized against the canonical campaign
  stage order: `preflight`, `research`, `comparison`, `candidate_selection`,
  `portfolio`, `candidate_review`, and `review`
* scenario matrix values are expanded in declaration order, then explicit
  included scenarios are appended in declaration order

## Scenarios

`scenarios` adds an optional multi-scenario contract on top of the base
campaign config. The base campaign still uses the same sections and validation
rules; scenarios only describe how to derive additional resolved variants from
that shared baseline.

Supported fields:

* `enabled`: turns scenario expansion on or off
* `matrix`: ordered list of sweep axes, each with:
  * `name`: stable axis label used in deterministic scenario IDs
  * `path`: dotted config path to override, for example
    `dataset_selection.timeframe` or `comparison.top_k`
  * `values`: non-empty ordered list of scalar sweep values
* `include`: ordered list of explicit one-off scenarios, each with:
  * `scenario_id`: stable explicit scenario identifier
  * `description`: optional operator-facing description
  * `overrides`: normal campaign override mapping using the same root sections
    as the main campaign spec

Two additional rate-limiting fields are available to keep sweep campaigns
operationally safe:

* `max_scenarios` *(optional positive integer)*: maximum total number of
  scenarios that the expansion is allowed to produce, combining the matrix
  cartesian product and the explicit `include` list.  When omitted, the hard
  ceiling `_SCENARIOS_HARD_MAX` (1000) applies.  Set this to a project-specific
  threshold to prevent accidental combinatorial explosions.
* `max_values_per_axis` *(optional positive integer)*: maximum number of
  values allowed per individual matrix axis.  Validation is applied at
  config-resolution time; each axis that exceeds this limit is reported by name
  in the error message.

The effective scenario limit is always `min(hard_max, configured_max)`.  The
hard maximum cannot be raised via config — it is enforced in the resolver.

Example:

```yaml
research_campaign:
  dataset_selection:
    dataset: features_daily
    timeframe: 1D
    evaluation_horizon: 5
  targets:
    alpha_names: [ml_alpha_q1]
  scenarios:
    enabled: true
    max_scenarios: 20
    max_values_per_axis: 5
    matrix:
      - name: timeframe
        path: dataset_selection.timeframe
        values: [1D, 4H]
      - name: top_k
        path: comparison.top_k
        values: [5, 10]
    include:
      - scenario_id: sleeve_review
        description: Use sleeve comparison for the review-only pass.
        overrides:
          comparison:
            alpha_view: sleeve
          review:
            filters:
              run_types: [alpha_evaluation]
```

## Scenario Expansion

Use `expand_research_campaign_scenarios(config)` to materialize the scenario
set from one resolved campaign config when you want typed
`ResolvedResearchCampaignScenario` objects in Python.

Use `build_research_campaign_scenario_catalog(config)` when you want one
JSON-serializable deterministic payload containing:

* the base campaign fingerprint without the `scenarios` section
* the ordered concrete scenario list
* each scenario's `scenario_id`, `source`, `sweep_values`, fingerprint, and
  normalized effective config snapshot
* an `expansion` summary block (see below)

Use `compute_scenario_expansion_size(config)` when you want the lightweight
expansion-size dictionary without materializing any resolved scenario
configs.  This is useful in tooling and preflight checks.  The returned
dictionary has these keys:

| key | description |
|---|---|
| `matrix_axis_count` | number of declared matrix axes |
| `per_axis_value_counts` | list of value counts in axis declaration order |
| `matrix_combination_count` | cartesian product of per-axis value counts (0 for no matrix) |
| `include_count` | number of explicit `include` scenarios |
| `total_scenario_count` | `matrix_combination_count + include_count` |
| `effective_max_scenarios` | operative limit (min of hard cap and configured cap) |
| `hard_max_scenarios` | non-overridable hard ceiling (1000) |
| `configured_max_scenarios` | `scenarios.max_scenarios` value or `None` |
| `max_values_per_axis` | `scenarios.max_values_per_axis` value or `None` |
| `exceeds_limit` | `True` when `total_scenario_count > effective_max_scenarios` |

The same expansion summary block is embedded in the `scenario_catalog.json`
artifact under the key `expansion`.

Expansion rules:

* when `scenarios.enabled` is `false`, expansion returns one implicit scenario
  with `scenario_id=default`
* matrix expansion is the cartesian product of `matrix[*].values` using the
  declared axis order
* each matrix scenario starts from the fully resolved base campaign config, then
  applies one override value per declared matrix axis
* each `include[*].overrides` scenario also starts from the same resolved base
  campaign config, then applies the explicit override mapping
* every expanded scenario is re-resolved through the same campaign resolver, so
  inheritance and validation behave exactly like a standalone campaign config
* limit violations are raised as `ResearchCampaignConfigError` at
  config-resolution time, before any execution begins

## Scenario Identity

Scenario identity is deterministic:

* matrix scenarios receive IDs like
  `scenario_0003_timeframe_4h__top_k_10`
* explicit scenarios use their normalized `include[*].scenario_id`
* every expanded scenario also exposes a short deterministic fingerprint based
  on the resolved config payload without the `scenarios` section

This makes scenario identity stable across repeated loads, docs/examples, and
future orchestration code that needs durable per-scenario artifact partitioning.

## Scenario Validation

`scenarios` is validated explicitly in the config layer:

* `scenarios.enabled=true` requires at least one matrix axis or included
  scenario
* matrix axes must use unique `name` and `path` values
* matrix `values` must be non-empty and deduplicated
* matrix `path` must be a dotted override path and cannot target the
  `scenarios` section itself
* matrix expansion must still produce unique normalized scenario identifiers
* included scenarios must use unique normalized `scenario_id` values
* included `overrides` cannot override the `scenarios` section
* included `overrides` must be valid campaign override mappings using the same
  schema as the main campaign config
* `max_values_per_axis` is checked per axis at config-resolution time — the
  axis name is included in the error message
* `max_scenarios` is checked against the full expansion count at
  config-resolution time — the total, effective limit, hard cap, and
  configured value are all included in the error message
* the absolute hard ceiling (1000) is enforced even when `max_scenarios` is
  not set

## Sweep Expansion Preflight

When `scenarios.enabled` is `true`, the orchestration writes an
**`expansion_preflight.json`** file to the orchestration artifact directory
before any scenarios run.  This file records:

```json
{
  "status": "passed",
  "scenarios_enabled": true,
  "expansion": {
    "matrix_axis_count": 2,
    "per_axis_value_counts": [2, 2],
    "matrix_combination_count": 4,
    "include_count": 1,
    "total_scenario_count": 5,
    "per_axis_details": [
      {"name": "timeframe", "path": "dataset_selection.timeframe", "value_count": 2},
      {"name": "top_k", "path": "comparison.top_k", "value_count": 2}
    ]
  },
  "limits": {
    "hard_max_scenarios": 1000,
    "configured_max_scenarios": 20,
    "effective_max_scenarios": 20,
    "max_values_per_axis": 5,
    "exceeds_limit": false
  }
}
```

The orchestration also prints a one-line expansion summary to stdout before
the scenario loop starts:

```
[expansion preflight] scenarios=5 (matrix=4, include=1) | limit=20 (hard=1000, configured=20) | axes=2
```

The status is always `"passed"` at orchestration time because limit violations
are caught earlier in `_resolve_scenarios`.  The file is still useful for
automation that inspects the orchestration artifact directory to confirm the
intended expansion size.

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
* a milestone-report pack under `milestone_report/` when
  `milestone_reporting.enabled` stays `true`

under `outputs.campaign_artifacts_root/<campaign_run_id>/`.

Sweep-enabled campaigns additionally write `expansion_preflight.json` to the
orchestration artifact directory (see the **Sweep Expansion Preflight**
section above).

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

Milestone 17 also exposes stitched retry and resumability metadata directly in
the campaign artifacts:

* `summary.json.final_outcomes.retry_stage_names`
* `summary.json.final_outcomes.partial_stage_names`
* `summary.json.final_outcomes.resumable_stage_names`
* `summary.json.stage_execution`
* `manifest.json.retry_stage_names`
* `manifest.json.partial_stage_names`
* `manifest.json.resumable_stage_names`
* `manifest.json.stage_execution`

## Milestone Reporting

`milestone_reporting` controls the campaign-level milestone pack written after a
campaign summary and manifest exist.

Supported fields:

* `enabled`: turn milestone generation on or off for campaign runs
* `decision_categories`: optional allowlist of decision categories to keep, for
  example `campaign_execution` or `review_promotion`
* `output.include_markdown_report`: emit or suppress `milestone_report/report.md`
* `output.decision_log_render_formats`: rendered decision-log views to persist
  inside `decision_log.json`; supported values are `markdown` and `text`
* `sections.*`: booleans for the optional markdown sections
  `campaign_scope`, `selections`, `key_findings`, `key_metrics`,
  `gate_outcomes`, `risks`, `next_steps`, `open_questions`,
  `decision_snapshot`, and `related_artifacts`
* `summary.include_stage_counts`: include or suppress the tracked-stage count
  sentence in the top-level milestone summary text
* `summary.include_review_outcome`: include or suppress the review/promotion
  outcome sentence in the top-level milestone summary text

Example:

```yaml
research_campaign:
  milestone_reporting:
    decision_categories: [review_promotion]
    output:
      include_markdown_report: false
      decision_log_render_formats: [text]
    sections:
      related_artifacts: false
      decision_snapshot: false
    summary:
      include_stage_counts: false
```

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
