# Real-Data Full Scenario Sweep Case Study (Milestone 19)

## Objective

Run a full-stage scenario-sweep research campaign on repository data and
existing checked-in alpha models, including portfolio construction,
candidate-review, and unified research review.

This case study demonstrates:

1. execution against real repository feature data (`features_daily`)
2. use of real alpha catalog entries from `configs/alphas_2026_q1.yml`
3. scenario-localized output roots for comparison, candidate-selection,
   portfolio, candidate-review, and review stages
4. orchestration outputs (`expansion_preflight`, `scenario_catalog`, and
   `scenario_matrix`) for the full sweep
5. optional second-pass checkpoint reuse across all scenarios

## Execute

```powershell
python docs/examples/real_data_full_scenario_sweep_case_study.py
```

Checkpoint-reuse demo:

```powershell
python docs/examples/real_data_full_scenario_sweep_case_study.py --checkpoint-demo
```

## Base Config and Real Targets

The script starts from:

```text
docs/examples/data/milestone_16_campaign_configs/real_world_campaign.yml
```

It uses the same real alpha set defined in:

```text
configs/alphas_2026_q1.yml
```

Alpha set used:

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
- keeps review focused on the same real alpha and portfolio outputs for that
  scenario

Total expansion:

- matrix: $2 \times 2 \times 2 = 8$
- include: $1$
- total scenarios: $9$

## Why This Variant Exists

The simpler real-data sweep case study disables portfolio execution because
shared portfolio registry roots can collide across scenarios when the same
deterministic portfolio run id appears multiple times.

This full-stage variant keeps portfolio enabled by localizing stage output roots
per scenario at runtime under:

```text
artifacts/research_campaigns/<orchestration_run_id>/scenarios/<scenario_id>/stage_outputs/
```

That keeps portfolio registries, portfolio artifacts, candidate-review outputs,
review outputs, and comparison outputs isolated per scenario.

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
- `stage_outputs/campaign_comparisons/`
- `stage_outputs/candidate_selection/`
- `stage_outputs/portfolios/`
- `stage_outputs/reviews/candidate_review/`
- `stage_outputs/reviews/research_review/`
- `milestone_report/`

## Reading Workflow

1. Start with `expansion_preflight.json`.
2. Review `scenario_matrix.csv` to identify top scenarios.
3. Open each winning scenario's `summary.json`.
4. Inspect that scenario's localized `stage_outputs/` tree for portfolio and
   review evidence.
5. Use `scenario_catalog.json` to tie the result back to the effective config.

## Related Docs

- [../milestone_19_scenario_sweeps.md](../milestone_19_scenario_sweeps.md)
- [real_data_scenario_sweep_case_study.md](real_data_scenario_sweep_case_study.md)
- [real_world_campaign_case_study.md](real_world_campaign_case_study.md)
- [../milestone_17_resume_workflow.md](../milestone_17_resume_workflow.md)