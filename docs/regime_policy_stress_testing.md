# Regime Policy Stress Testing

## Purpose

Regime policy stress testing evaluates accepted or candidate adaptive regime policies under deterministic adverse and ambiguous regime conditions. It is a reusable stress layer for governance workflows and is designed for fixture-backed reproducibility.

It answers:

- whether adaptive policies remain acceptable when regime evidence becomes unstable or contradictory
- how adaptive behavior compares with a static baseline under deterministic stress transforms
- whether fallback usage and turnover stay within configured tolerances

## Relationship to Existing Workflows

This workflow is read-only with respect to benchmark packs, promotion gates, review packs, and candidate selection artifacts.

It does not:

- rerun benchmark packs
- rerun promotion gates
- rerun review packs
- rerun candidate selection
- alter taxonomy, calibration, GMM classifier, policy optimization semantics, or live execution paths

It consumes review-pack metadata as provenance and consumes file-backed policy metrics as the initial execution mode.

## Config

Checked-in example:

- `configs/regime_stress_tests/m26_adaptive_policy_stress.yml`

Key fields:

- `stress_test_name`
- `source_review_pack`
- `source_policy_candidates.policy_metrics_path`
- `source_policy_candidates.baseline_policy`
- `scenarios`
- `metrics`
- `stress_gates`
- `output_root`

## Scenario Types

Supported deterministic scenario types:

- `transition_shock`
- `regime_whipsaw`
- `high_vol_persistence`
- `classifier_uncertainty`
- `taxonomy_ml_disagreement`
- `confidence_collapse`

Scenario IDs are deterministic and derived from scenario name, type, and normalized parameters.

## Policy Metrics Inputs

Policy metrics input supports CSV or JSON (`{"rows": [...]}`).

Required per policy row:

- `policy_name`
- `policy_type`
- `is_baseline`
- `regime_source`
- `classifier_model`
- `calibration_profile`

Optional base metrics include:

- `base_return`
- `base_sharpe`
- `base_max_drawdown`
- `base_policy_turnover`
- `base_state_change_count`
- `base_fallback_activation_count`
- `base_confidence`
- `base_entropy`

Optional scenario-specific pass-through metrics can be supplied using `scenario_name` rows. If absent, deterministic derivations are applied.

## Derived Metric Behavior

When scenario-specific rows are unavailable, the runner applies simple deterministic transforms:

- `transition_shock`: increases drawdown, turnover, state changes around transition windows
- `regime_whipsaw`: increases turnover and state changes, reduces Sharpe
- `high_vol_persistence`: worsens drawdown and can reward defensive/fallback-style policies
- `classifier_uncertainty`: reduces confidence, raises entropy
- `taxonomy_ml_disagreement`: raises disagreement rate and fallback activation
- `confidence_collapse`: forces low-confidence behavior and elevated fallback activation

These transforms are fixture-friendly approximations and are not full market simulations.

## Gates

Per scenario-policy row gates:

- `max_policy_turnover`
- `max_stress_drawdown`
- `min_adaptive_vs_static_drawdown_delta`
- `max_fallback_activation_rate`
- `max_state_change_count`

Gate behavior:

- rows fail when any configured gate fails
- missing gate-required metrics fail the row and emit warning metadata
- `primary_failure_reason` records the first deterministic failure in gate order

## Artifacts

Outputs are written under:

```text
artifacts/regime_stress_tests/<stress_run_id>/
```

Expected artifacts:

- `stress_matrix.csv`
- `stress_matrix.json`
- `stress_leaderboard.csv`
- `scenario_summary.json`
- `policy_stress_summary.json`
- `scenario_catalog.json`
- `scenario_results.csv`
- `fallback_usage.csv`
- `policy_turnover_under_stress.csv`
- `adaptive_vs_static_stress_comparison.csv`
- `config.json`
- `manifest.json`
- `source_regime_review_pack.json`
- `policy_input_inventory.json`

## CLI

```powershell
python -m src.cli.run_regime_policy_stress_tests `
  --config configs/regime_stress_tests/m26_adaptive_policy_stress.yml
```

Optional overrides:

```powershell
--source-review-pack artifacts/regime_reviews/<review_run_id>
--policy-metrics-path path/to/policy_candidates.csv
--output-root artifacts/regime_stress_tests
```

CLI output includes:

- stress run id
- output directory
- scenario count
- policy count
- most resilient policy
- stress matrix path
- policy stress summary path

## Interpretation Guidance

Use `stress_matrix.csv` for row-level audit and `stress_leaderboard.csv` for policy ranking under stress. Inspect `policy_turnover_under_stress.csv` and `fallback_usage.csv` for operational stability concerns. Use `adaptive_vs_static_stress_comparison.csv` to assess whether adaptive behavior remains useful versus static baseline behavior.

## Limitations and Non-Goals

- deterministic transforms are not market simulators
- registry-backed policy sourcing is future work
- this issue does not build a full-year case study or final portfolio
- this issue does not rerun or mutate upstream benchmark/gate/review/candidate artifacts
