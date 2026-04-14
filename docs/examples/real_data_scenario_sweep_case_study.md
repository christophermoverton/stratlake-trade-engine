# Real-Data Scenario Sweep Case Study (Milestone 19)

## Objective

Run a scenario-sweep research campaign on repository data and existing checked-in
alpha models, then inspect orchestration-level sweep outputs for ranking and
traceability.

This case study demonstrates:

1. campaign scenario expansion from matrix plus include definitions
2. execution against real repository feature data (`features_daily`)
3. use of real alpha catalog entries from `configs/alphas_2026_q1.yml`
4. orchestration outputs (`expansion_preflight`, `scenario_catalog`, and
   `scenario_matrix`)
5. optional second-pass checkpoint reuse across all scenarios
6. a sweep configuration that disables the portfolio stage to keep scenario
   runs registry-safe in a shared output root

## Execute

```powershell
python docs/examples/real_data_scenario_sweep_case_study.py
```

Checkpoint-reuse demo:

```powershell
python docs/examples/real_data_scenario_sweep_case_study.py --checkpoint-demo
```

Optional output override:

```powershell
python docs/examples/real_data_scenario_sweep_case_study.py --output-root docs/examples/output/real_data_scenario_sweep_case_study
```

## Base Config and Real Targets

The script starts from:

```text
docs/examples/data/milestone_16_campaign_configs/real_world_campaign.yml
```

Then it applies output-path overrides under this case-study output root and
injects a sweep block.

The case study keeps `portfolio.enabled: false` and runs review on
`alpha_evaluation` artifacts only.

Real alpha targets come from:

```text
configs/alphas_2026_q1.yml
```

Alpha set used by the base config:

- `ml_cross_sectional_xgb_2026_q1`
- `ml_cross_sectional_lgbm_2026_q1`
- `ml_cross_sectional_elastic_net_2026_q1`
- `rank_composite_momentum_2026_q1`

## Sweep Specification

Matrix axes:

- `comparison.top_k`: `3`, `4`
- `candidate_selection.max_candidates`: `2`, `3`
- `candidate_selection.redundancy.max_pairwise_correlation`: `0.70`, `0.80`

Include scenario:

- `scenario_id=sleeve_review_only`
- overrides `comparison.alpha_view=sleeve`
- narrows `review.filters.run_types` to `alpha_evaluation`

Total expansion:

- matrix: $2 \times 2 \times 2 = 8$
- include: $1$
- total scenarios: $9$

## Output Location

```text
docs/examples/output/real_data_scenario_sweep_case_study/
```

Primary case-study summary:

```text
docs/examples/output/real_data_scenario_sweep_case_study/summary.json
```

Native orchestration root:

```text
docs/examples/output/real_data_scenario_sweep_case_study/artifacts/research_campaigns/<orchestration_run_id>/
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

Note:

- Portfolio artifacts are not produced in this case study because portfolio
   execution is intentionally disabled.

## Reading Workflow

1. Start with `expansion_preflight.json` to confirm planned scenario count and
   limits.
2. Open `scenario_matrix.csv` or `scenario_matrix.json` and review top-ranked
   scenarios.
3. Use `campaign_summary_path` from matrix rows to open the full per-scenario
   summary.
4. Match `scenario_id` in `scenario_catalog.json` to inspect the exact
   `effective_config` that produced each row.

## Checkpoint-Reuse Demo

Run with `--checkpoint-demo` to execute the same sweep twice against the same
output root.

Expected behavior:

- first pass writes scenario checkpoints
- second pass reuses checkpoints when scenario provenance and stage fingerprints
  match

Reuse counters are recorded in:

- `summary.json` under `second_pass.scenario_preflight_state_counts`

## Related Docs

- [../milestone_19_scenario_sweeps.md](../milestone_19_scenario_sweeps.md)
- [real_world_campaign_case_study.md](real_world_campaign_case_study.md)
- [../research_campaign_configuration.md](../research_campaign_configuration.md)
- [../milestone_17_resume_workflow.md](../milestone_17_resume_workflow.md)