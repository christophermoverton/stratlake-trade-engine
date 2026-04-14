# Milestone 19 Scenario Sweeps and Comparison Outputs

## Overview

Milestone 19 documents the sweep-enabled campaign orchestration that now sits on
 top of Milestones 16, 17, and 18.

The core idea is straightforward:

- a single campaign spec can expand into a deterministic set of scenarios
- each scenario runs as a normal campaign with scenario-aware identity
- orchestration emits deterministic cross-scenario outputs for ranking and review

This guide explains the contract and how to interpret the resulting artifacts.

## What Is a Scenario Sweep

A scenario sweep is one campaign config with `scenarios.enabled: true` that
expands into multiple concrete scenario configs before execution.

Each expanded scenario is a fully resolved campaign payload with:

- deterministic `scenario_id`
- deterministic scenario fingerprint
- a concrete effective config snapshot

That scenario then runs through the same staged campaign flow:

1. `preflight`
2. `research`
3. `comparison`
4. `candidate_selection`
5. `portfolio`
6. `candidate_review`
7. `review`

When `scenarios.enabled` is `false`, the runner behaves like a single campaign
run and emits one implicit scenario (`scenario_id=default`) in expansion logic.

## Single Run vs Sweep

Single campaign run:

- one resolved campaign config
- one `campaign_run_id`
- one campaign artifact directory

Sweep orchestration:

- one resolved sweep config
- one deterministic orchestration run id
- multiple scenario campaigns under `scenarios/<scenario_id>/`
- one orchestration-level matrix and catalog for cross-scenario comparison

## Deterministic Guarantees

Milestone 19 keeps the deterministic behavior used across the repository:

- deterministic expansion ordering based on matrix axis declaration order
- deterministic matrix scenario ids (`scenario_<index>_<axis/value labels>`)
- deterministic include scenario ids (normalized `include[*].scenario_id`)
- deterministic scenario fingerprints from resolved config payloads without the
  `scenarios` section
- deterministic orchestration and campaign serialization (`json.dumps(...,
  sort_keys=True)`, newline-normalized writes)
- deterministic ranking order in `scenario_matrix.csv` and
  `scenario_matrix.json`

Scenario matrix tie-breaking is explicit in code:

`portfolio_sharpe_ratio desc, strategy_sharpe_ratio_max desc, alpha_ic_ir_max desc, portfolio_total_return desc, strategy_cumulative_return_max desc, alpha_mean_ic_max desc, candidate_selected_count desc, then scenario_id asc`

## Configuration Structure

Example sweep block:

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
        description: Review-focused pass using sleeve view.
        overrides:
          comparison:
            alpha_view: sleeve
          review:
            filters:
              run_types: [alpha_evaluation]
```

Important semantics:

- `matrix` is a cartesian product across all axis `values`
- each matrix scenario starts from the same resolved base config
- each include scenario starts from the same resolved base config and applies
  only `include[*].overrides`
- every scenario is re-resolved through the same campaign resolver

## Guardrails and Hard Limits

Sweep sizing is validated before execution starts.

Config-level controls:

- `scenarios.max_scenarios`
- `scenarios.max_values_per_axis`

Hard ceiling:

- `hard_max_scenarios=1000` (non-overridable)

Sizing helper output (`compute_scenario_expansion_size`) includes:

- `matrix_axis_count`
- `per_axis_value_counts`
- `matrix_combination_count`
- `include_count`
- `total_scenario_count`
- `effective_max_scenarios`
- `hard_max_scenarios`
- `configured_max_scenarios`
- `max_values_per_axis`
- `exceeds_limit`

If limits are exceeded, config resolution raises `ResearchCampaignConfigError`
before any scenario execution begins.

## Expansion Preflight

Sweep orchestration writes `expansion_preflight.json` before running scenarios.

Purpose:

- confirms intended expansion dimensions
- captures active limits
- provides one lightweight machine-readable planning artifact

Example shape:

```json
{
  "status": "passed",
  "scenarios_enabled": true,
  "expansion": {
    "matrix_axis_count": 3,
    "per_axis_value_counts": [2, 2, 2],
    "matrix_combination_count": 8,
    "include_count": 1,
    "total_scenario_count": 9,
    "per_axis_details": [
      {"name": "timeframe", "path": "dataset_selection.timeframe", "value_count": 2},
      {"name": "top_k", "path": "comparison.top_k", "value_count": 2},
      {"name": "max_candidates", "path": "candidate_selection.max_candidates", "value_count": 2}
    ]
  },
  "limits": {
    "hard_max_scenarios": 1000,
    "configured_max_scenarios": 16,
    "effective_max_scenarios": 16,
    "max_values_per_axis": 4,
    "exceeds_limit": false
  }
}
```

## Orchestration Flow and Artifact Layout

For sweeps, orchestration writes one root directory:

```text
artifacts/research_campaigns/<orchestration_run_id>/
  campaign_config.json
  expansion_preflight.json
  scenario_catalog.json
  scenario_matrix.csv
  scenario_matrix.json
  manifest.json
  summary.json
  scenarios/
    <scenario_id>/
      campaign_config.json
      checkpoint.json
      preflight_summary.json
      manifest.json
      summary.json
      milestone_report/...
```

Interpretation:

- orchestration files summarize cross-scenario behavior
- scenario directories keep per-scenario campaign state, checkpoint, and summary
- orchestration `manifest.json` and `summary.json` prefix scenario files under
  `scenarios/<scenario_id>/...` using relative POSIX paths

## Scenario Catalog Interpretation

`scenario_catalog.json` is the deterministic expansion contract.

Top-level keys:

- `scenario_count`
- `base_fingerprint`
- `scenarios_enabled`
- `expansion`
- `scenarios`

Each scenario entry includes:

- `scenario_id`
- `description`
- `source` (`matrix`, `include`, or `default`)
- `sweep_values`
- `fingerprint`
- `effective_config`

Use cases:

- trace an output scenario back to its exact effective config
- inspect which matrix/include source produced the scenario
- compare fingerprints across reruns

## Scenario Matrix Interpretation

Two files are emitted:

- `scenario_matrix.csv`: tabular matrix for spreadsheets and downstream tooling
- `scenario_matrix.json`: same leaderboard plus metadata and selection rule

Common row fields:

- identity: `rank`, `scenario_id`, `source`, `campaign_run_id`, `fingerprint`
- sweep dimensions: `sweep_<axis_name>` and `sweep_summary`
- stage status: `status`, `preflight_status`
- core metrics:
  - `portfolio_sharpe_ratio`, `portfolio_total_return`
  - `strategy_sharpe_ratio_max`, `strategy_cumulative_return_max`
  - `alpha_ic_ir_max`, `alpha_mean_ic_max`
  - `candidate_selected_count`
- review context: `review_entry_count`, `review_promotion_status`,
  `review_promotion_gate_status`

Annotated example row:

```json
{
  "rank": 1,
  "scenario_id": "scenario_0007_timeframe_4h__top_k_10__max_candidates_3",
  "sweep_timeframe": "4H",
  "sweep_top_k": 10,
  "sweep_max_candidates": 3,
  "ranking_metric": "portfolio_sharpe_ratio",
  "ranking_value": 1.56,
  "portfolio_sharpe_ratio": 1.56,
  "strategy_sharpe_ratio_max": 1.12,
  "alpha_ic_ir_max": 1.24,
  "campaign_summary_path": ".../scenarios/scenario_0007_timeframe_4h__top_k_10__max_candidates_3/summary.json"
}
```

Reading pattern:

1. start with `rank`, `ranking_metric`, and `ranking_value`
2. inspect sweep columns (`sweep_*`) to understand what changed
3. open `campaign_summary_path` for full per-scenario outcomes
4. cross-check scenario config in `scenario_catalog.json`

## Relationship Between Orchestration Summary, Per-Scenario Summaries, and Matrix

- orchestration `summary.json`:
  - scenario inventory
  - scenario status counts
  - output paths
  - embedded scenario-matrix summary payload
- per-scenario `summary.json`:
  - full stage-level campaign results for one scenario
  - stage execution metadata and reuse decisions
  - milestone report paths and review outcomes
- scenario matrix outputs:
  - normalized cross-scenario leaderboard derived from per-scenario key metrics

In practice:

- use orchestration `summary.json` to locate and audit all scenarios
- use `scenario_matrix.*` to rank and shortlist
- use per-scenario `summary.json` for root-cause and decision traceability

## Resume and Reuse in Multi-Scenario Sweeps

Reuse remains checkpoint-driven and deterministic per scenario campaign.

Key behavior:

- each scenario campaign run id includes scenario provenance:
  - `orchestration_run_id`
  - `scenario_id`
  - scenario `fingerprint`
- each scenario checkpoint stores the same scenario provenance payload
- checkpoint loading rejects mismatched scenario provenance
- stage reuse still depends on stage input fingerprints and reuse policy

Operationally this means:

- identical rerun of the same sweep can reuse completed stages per scenario
- changed scenario provenance does not silently reuse an incompatible checkpoint
- forced reruns and downstream invalidation still apply within each scenario
  campaign exactly as in Milestone 17

## Practical Guidance

When to use sweeps:

- comparing parameter tradeoffs across a bounded grid
- evaluating robustness of ranking outcomes across plausible assumptions
- preparing review packets with explicit scenario evidence

When to prefer a single run:

- narrow hypothesis with one clear config
- fast turnaround needed before broader sensitivity analysis
- expensive stages where exploration budget is limited

How to keep sweeps manageable:

- start with 2-3 axes and low per-axis value counts
- set `max_values_per_axis` and `max_scenarios` explicitly
- use include scenarios for targeted edge cases instead of expanding axis grids
- inspect `expansion_preflight.json` before trusting runtime duration

How to interpret tradeoffs:

- do not rely on one metric alone
- use ranking as triage, then inspect per-scenario summaries and review outcomes
- compare top-ranked scenarios against runner-up scenarios with different
  `sweep_*` values to understand sensitivity

Suggested iteration loop:

1. define bounded sweep
2. validate expansion preflight and limits
3. run orchestration
4. inspect matrix leaderboard
5. trace finalists to catalog effective configs
6. inspect per-scenario summaries and milestone reports
7. refine sweep axes for next pass

## Real-World Case Study

For a deterministic, runnable sweep example with matrix and include scenarios,
see:

- [examples/real_world_scenario_sweep_case_study.md](examples/real_world_scenario_sweep_case_study.md)
- [examples/real_world_scenario_sweep_case_study.py](examples/real_world_scenario_sweep_case_study.py)

For real repository data and existing checked-in alpha models, use one of these
two variants depending on the stage coverage you want:

- [examples/real_data_scenario_sweep_case_study.md](examples/real_data_scenario_sweep_case_study.md)
  Simpler real-data sweep. Portfolio is disabled, so the run focuses on alpha,
  comparison, candidate-selection, and review outputs without shared-root
  portfolio registry collisions.
- [examples/real_data_full_scenario_sweep_case_study.md](examples/real_data_full_scenario_sweep_case_study.md)
  Full-stage real-data sweep. Portfolio, candidate-review, and unified review
  stay enabled, with per-scenario localized stage-output roots under each
  scenario directory to keep registries and artifacts isolated.

## Related Docs

- [research_campaign_configuration.md](research_campaign_configuration.md)
- [milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md)
- [milestone_17_resume_workflow.md](milestone_17_resume_workflow.md)
- [milestone_18_milestone_review_workflow.md](milestone_18_milestone_review_workflow.md)
- [milestone_reporting_artifacts.md](milestone_reporting_artifacts.md)
