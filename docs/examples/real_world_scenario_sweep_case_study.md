# Real-World Scenario Sweep Case Study (Milestone 19)

## Objective

Demonstrate a practical, deterministic scenario sweep that expands one campaign
spec into multiple scenarios, writes orchestration-level comparison artifacts,
and supports checkpoint reuse on rerun.

This case study shows:

1. one sweep-enabled campaign spec expanding into matrix plus include scenarios
2. orchestration artifact layout and scenario partitioning
3. cross-scenario ranking in `scenario_matrix.csv` and `scenario_matrix.json`
4. traceability from leaderboard row to `scenario_catalog.json` effective config
5. second-pass checkpoint reuse across all scenarios

## Execute

```powershell
python docs/examples/real_world_scenario_sweep_case_study.py
```

Optional output override:

```powershell
python docs/examples/real_world_scenario_sweep_case_study.py --output-root docs/examples/output/real_world_scenario_sweep_case_study
```

## Sweep Specification

The script builds a deterministic campaign payload with three matrix axes and
one explicit include scenario.

Matrix axes:

- `dataset_selection.timeframe`: `1D`, `4H`
- `comparison.top_k`: `5`, `10`
- `candidate_selection.max_candidates`: `2`, `3`

Include scenario:

- `scenario_id=review_only_sleeve`
- overrides `comparison.alpha_view=sleeve`
- narrows `review.filters.run_types` to `alpha_evaluation`

Total expansion:

- matrix: $2 \times 2 \times 2 = 8$
- include: $1$
- total scenarios: $9$

## Workflow Surfaces Used

- `src.config.research_campaign.resolve_research_campaign_config`
- `src.cli.run_research_campaign.run_research_campaign`
- orchestration artifact writers in `src.cli.run_research_campaign`

The example uses deterministic stage stubs so it is fast and reproducible while
still exercising the real sweep-expansion, orchestration, matrix, catalog,
manifest, summary, and checkpoint logic.

## Output Location

The example writes to:

```text
docs/examples/output/real_world_scenario_sweep_case_study/
```

Primary case-study summary:

```text
docs/examples/output/real_world_scenario_sweep_case_study/summary.json
```

Native orchestration root:

```text
docs/examples/output/real_world_scenario_sweep_case_study/artifacts/research_campaigns/<orchestration_run_id>/
```

## Artifacts To Inspect

Orchestration-level:

- `campaign_config.json`
- `expansion_preflight.json`
- `scenario_catalog.json`
- `scenario_matrix.csv`
- `scenario_matrix.json`
- `manifest.json`
- `summary.json`

Per-scenario under `scenarios/<scenario_id>/`:

- `campaign_config.json`
- `checkpoint.json`
- `preflight_summary.json`
- `manifest.json`
- `summary.json`
- `milestone_report/summary.json`
- `milestone_report/decision_log.json`
- `milestone_report/manifest.json`
- `milestone_report/report.md`

## Interpreting `expansion_preflight.json`

Read this first to validate planned sweep size and limits before examining
results.

Example shape:

```json
{
  "status": "passed",
  "scenarios_enabled": true,
  "expansion": {
    "matrix_axis_count": 3,
    "matrix_combination_count": 8,
    "include_count": 1,
    "total_scenario_count": 9
  },
  "limits": {
    "effective_max_scenarios": 16,
    "hard_max_scenarios": 1000,
    "max_values_per_axis": 4,
    "exceeds_limit": false
  }
}
```

## Interpreting `scenario_matrix` Outputs

`scenario_matrix.csv` and `scenario_matrix.json` share the same leaderboard
rows.

Common fields:

- scenario identity: `scenario_id`, `source`, `campaign_run_id`, `fingerprint`
- sweep columns: `sweep_timeframe`, `sweep_top_k`, `sweep_max_candidates`
- ranking: `rank`, `ranking_metric`, `ranking_value`
- core metrics: `portfolio_sharpe_ratio`, `strategy_sharpe_ratio_max`,
  `alpha_ic_ir_max`, `candidate_selected_count`
- traceability: `campaign_summary_path`

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
  "strategy_sharpe_ratio_max": 0.78,
  "alpha_ic_ir_max": 1.58,
  "campaign_summary_path": ".../scenarios/scenario_0007_timeframe_4h__top_k_10__max_candidates_3/summary.json"
}
```

Reading workflow:

1. sort by `rank` and note `ranking_metric`
2. compare `sweep_*` columns between top rows
3. open `campaign_summary_path` for the winning scenario
4. locate the same `scenario_id` in `scenario_catalog.json`
5. inspect that scenario's `effective_config`

## Tracing Scenario Back to Config

`scenario_catalog.json` is the source of truth for expanded config snapshots.

Example catalog entry:

```json
{
  "scenario_id": "scenario_0007_timeframe_4h__top_k_10__max_candidates_3",
  "source": "matrix",
  "sweep_values": {
    "timeframe": "4H",
    "top_k": 10,
    "max_candidates": 3
  },
  "fingerprint": "4f56d4d2f8f9a2b1",
  "effective_config": {
    "dataset_selection": {"timeframe": "4H"},
    "comparison": {"top_k": 10},
    "candidate_selection": {"max_candidates": 3}
  }
}
```

This is the exact bridge from leaderboard result to concrete scenario config.

## Relationship of Summary Layers

- orchestration `summary.json`:
  - global scenario counts
  - status counts
  - embedded `scenario_matrix`
  - scenario inventory with per-scenario summary paths
- per-scenario `summary.json`:
  - full campaign stage states and key metrics for one scenario
- `scenario_matrix.*`:
  - normalized ranking view generated from per-scenario key metrics

## Resume and Reuse Behavior in the Case Study

The script runs orchestration twice with the same config and output root.

Expected behavior:

- first pass: scenarios execute and write checkpoints
- second pass: scenarios reuse prior checkpoints when fingerprints match

The case-study `summary.json` includes second-pass reuse counters under
`second_pass.scenario_preflight_state_counts` and references to the second-pass
orchestration summary for deeper inspection.

## Practical Usage Guidance

Use sweeps when:

- you need sensitivity analysis across bounded assumptions
- you want leaderboard-style scenario comparison with deterministic traceability
- you need one orchestration packet for review and handoff

Keep sweeps manageable by:

- limiting axis width with `max_values_per_axis`
- bounding expansion with `max_scenarios`
- using include scenarios for targeted edge cases
- checking `expansion_preflight.json` before deep analysis

Interpret tradeoffs by:

- starting from matrix rank
- comparing top scenarios with different sweep settings
- validating conclusions in per-scenario summaries and milestone outputs

## Related Docs

- [../milestone_19_scenario_sweeps.md](../milestone_19_scenario_sweeps.md)
- [../research_campaign_configuration.md](../research_campaign_configuration.md)
- [../milestone_17_resume_workflow.md](../milestone_17_resume_workflow.md)
- [../milestone_18_milestone_review_workflow.md](../milestone_18_milestone_review_workflow.md)
