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
- `market_simulation_stress` (optional)
- `output_root`

Path behavior:

- config paths are interpreted from the current working directory, normally the repository root
- the checked-in fixture config assumes the CLI is run from the repository root
- when running from another location, provide explicit paths via CLI overrides:
  - `--source-review-pack`
  - `--policy-metrics-path`
  - `--output-root`

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

Derived stress metrics are deterministic, fixture-friendly approximations intended for governance and behavior testing, not empirical market simulation.

Real-data case studies should clearly separate observed benchmark results from synthetic stress results.

## Optional Market Simulation Stress Evidence

M26 stress tests can optionally stitch in existing M27 `simulation_metrics/` artifacts. This is an artifact-consumption bridge: it does not duplicate M27 simulation logic and it does not replace the Issue #299 deterministic regime shock, whipsaw, uncertainty, fallback, turnover, and adaptive-vs-static stress checks.

Disabled or absent config is a no-op:

```yaml
market_simulation_stress:
  enabled: false
```

Existing-artifact mode consumes the canonical M27 metrics directory:

```yaml
market_simulation_stress:
  enabled: true
  mode: existing_artifacts
  simulation_metrics_dir: docs/examples/output/m27_market_simulation_case_study/source_simulation_artifacts/<run_id>/market_simulations/simulation_metrics
  include_in_policy_stress_summary: true
  include_in_case_study_report: true
```

`run_config` mode is also available for fixture-backed configs that already use the M27 execution stack:

```yaml
market_simulation_stress:
  enabled: true
  mode: run_config
  config_path: configs/regime_stress_tests/m27_market_simulation_case_study.yml
```

The bridge validates these M27 files before writing stitched M26 artifacts:

- `simulation_path_metrics.csv`
- `simulation_summary.csv`
- `simulation_leaderboard.csv`
- `policy_failure_summary.json`
- `simulation_metric_config.json`
- `manifest.json`

When enabled, the stress run writes:

- `market_simulation_stress_summary.json`
- `market_simulation_stress_leaderboard.csv`

The summary records source run ID, simulation types, path/summary/leaderboard counts, aggregate policy failure rate, best/worst ranked simulation scenarios, source artifact paths, and the Monte Carlo limitation note.

M27 market simulation evidence is optional and complementary. Monte Carlo paths are regime-only unless return or policy replay artifacts are explicitly available. Simulation outputs are not forecasts or trading recommendations.

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

When `market_simulation_stress.enabled=true`, the output directory also includes:

- `market_simulation_stress_summary.json`
- `market_simulation_stress_leaderboard.csv`

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

`stress_leaderboard.csv` ranks all evaluated policies, including the static baseline.

`policy_stress_summary.json` may report `most_resilient_policy` as the static baseline when it has the strongest stress ranking.

For downstream interpretation, distinguish:

- best overall policy under stress
- best adaptive policy under stress
- adaptive-vs-static deltas

Summary metadata includes explicit interpretation helpers such as:

- `most_resilient_adaptive_policy`
- `baseline_rank`
- `adaptive_policy_count`
- `baseline_included_in_leaderboard`

`source_candidate_selection` is provenance-only in Issue 5. When supplied, it is recorded in summary/manifest provenance paths but does not change stress evaluation semantics.

## Limitations and Non-Goals

- deterministic transforms are not market simulators
- optional M27 market simulation stress evidence complements deterministic transforms but does not replace them
- regime-transition Monte Carlo paths are regime-only unless return or policy replay artifacts are explicitly available
- simulation outputs are not forecasts or trading recommendations
- registry-backed policy sourcing is future work
- this issue does not build a full-year case study or final portfolio
- this issue does not rerun or mutate upstream benchmark/gate/review/candidate artifacts
