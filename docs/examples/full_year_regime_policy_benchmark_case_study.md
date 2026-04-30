# Full-Year Regime Policy Benchmark Case Study (Milestone 26)

## Research Question
Do regime-aware adaptive policies improve performance, robustness, and governance quality relative to a static baseline after accounting for calibration, ML confidence, transition instability, promotion gates, candidate-selection roles, policy turnover, drawdown, fallback behavior, and deterministic stress resilience?

## Workflow Overview
This case study stitches Milestone 26 Issue 1 through Issue 5 evidence into one deterministic, reproducible artifact set.

1. Build deterministic fixture-backed full-year feature inputs.
2. Run regime benchmark-pack policy variants.
3. Run promotion gates on benchmark outputs.
4. Generate a review pack from benchmark and gate evidence.
5. Run regime-aware candidate selection from review evidence.
6. Run adaptive regime-policy stress tests.
7. Stitch benchmark, governance, candidate, and stress evidence into one case-study summary.

## Data / Fixture Mode And Evaluation Window
- Mode: fixture-backed deterministic case study
- Evaluation window: 2026-01-01 through 2026-12-31
- Timeframe: 1D
- Input behavior: this example does not require live market data and does not require full local real-data coverage

The benchmark layer is executed with deterministic fixture features that mirror the full-year workflow. You can later replace those fixture benchmark inputs with real full-year benchmark outputs while keeping the same downstream stitching pattern.

The current implementation generates deterministic fixture benchmark inputs in the case-study script, then runs downstream promotion, review, candidate-selection, and stress stages through the existing Milestone 26 execution APIs. Real full-year benchmark outputs can be substituted later by replacing the benchmark source/output stage while preserving promotion/review/candidate/stress stitching schemas. Stress evidence remains deterministic synthetic evidence unless explicitly replaced with observed stress or robustness evidence.

## Policy Variants Compared
The stitched summary normalizes policy evidence into these conceptual roles:

- static_baseline
- taxonomy_only_policy
- calibrated_taxonomy_policy
- gmm_confidence_policy
- hybrid_calibrated_gmm_policy

The generated policy_variant_comparison.csv includes mapping fields:
- policy_name
- policy_role
- regime_source
- calibration_profile
- classifier_model
- adaptive_policy_type context through benchmark/review/stress evidence
- source_artifact path

## Benchmark-Pack Evidence
Observed benchmark evidence comes from:
- benchmark_summary.json
- benchmark_matrix.csv/json
- stability and transition summaries
- policy-comparison and calibration/model comparison outputs

This evidence is consumed as read-only input for governance and stitching.

## Promotion-Gate Outcomes
Governance gate evidence comes from:
- promotion_gate_summary.json (stitched)
- gate_results.csv/json
- decision_summary.json
- failed_gates.csv
- warning_gates.csv

Case-study interpretation references gate decisions and primary gate reasons without mutating upstream gate outputs.

## Review-Pack Decisions
Review evidence comes from:
- review_summary.json (stitched)
- leaderboard.csv/json
- decision_log.json
- accepted/warning/needs-review/rejected filtered outputs
- evidence_index.json

The case study surfaces decision counts and top reviewed policy for downstream interpretation.

## Regime-Aware Candidate Selection
Candidate-selection evidence comes from:
- candidate_selection_summary.json (stitched)
- candidate_selection.csv/json
- category assignments
- transition resilience
- defensive fallback candidates
- redundancy and allocation hints

The case study reports selected counts and category composition as governance context, not portfolio deployment advice.

## Adaptive Policy Stress Testing
Deterministic synthetic stress evidence comes from:
- stress_summary.json (stitched)
- stress_matrix.csv/json
- stress_leaderboard.csv
- scenario_summary.json
- policy_stress_summary.json
- adaptive_vs_static_stress_comparison.csv

Important: these stress transforms are deterministic synthetic perturbations. They are robustness diagnostics, not empirical market simulation claims.

## Final Interpretation
The stitched interpretation in final_interpretation.md separates:
- observed benchmark evidence
- promotion/review governance evidence
- candidate-selection evidence
- deterministic synthetic stress evidence

It also summarizes the static baseline comparison and identifies the most resilient adaptive policy under stress.

## Limitations
- Fixture-backed benchmark inputs are used to preserve reproducibility across environments without requiring complete local real-data coverage.
- Stress scenarios are deterministic synthetic transforms and should not be interpreted as historical market simulation.
- Outputs support research governance and evidence stitching; they do not imply production or trading readiness.

## M27 Market Simulation Integration (Optional)

Milestone 27 adds an optional market simulation bridge that extends this case
study with synthetic stress-path evidence. When enabled, the bridge reads
pre-computed M27 simulation metrics and appends simulation leaderboard and
policy failure data to the stitched report.

### Enabling the bridge

Add a `market_simulation_stress` block to the case-study config, or pass
equivalent CLI overrides:

```yaml
market_simulation_stress:
  enabled: true
  mode: existing_artifacts
  simulation_metrics_dir: artifacts/market_simulation/<run_id>/simulation_metrics
  include_in_policy_stress_summary: true
  include_in_case_study_report: true
```

Run the M27 case study first to generate the required simulation metrics:

```bash
python docs/examples/m27_market_simulation_case_study.py
```

Then run this case study as normal. The bridge will load
`simulation_leaderboard.csv` and `policy_failure_summary.json` from the
specified `simulation_metrics_dir` and include them in:

- `policy_stress_summary.json` (under the `market_simulation_stress` key)
- `final_interpretation.md` (as an additional simulation-stress section)

### Default no-op behavior

If the `market_simulation_stress` block is absent or `enabled: false`, this
case study runs exactly as before with no M27 artifacts loaded.

### Strict mode

Pass `--require-market-simulation-stress` to require M27 evidence and fail
with a non-zero exit if it is missing.

### Relationship to M26 deterministic stress tests

M27 simulation evidence supplements the M26 deterministic stress results. Both
are included in the stitched report when the bridge is enabled. See
[docs/market_simulation_models_and_integrations.md](../market_simulation_models_and_integrations.md)
for the full M27 architecture and integration reference.

## Follow-Up Recommendations
1. Replace fixture-backed benchmark inputs with fully covered real full-year benchmark outputs when available.
2. Compare multiple calibration profiles and policy profiles under the same stitched workflow.
3. Add additional stress scenarios and promotion-gate sensitivity sweeps for policy stability diagnostics.
4. Track how candidate-category composition changes when review decisions change.

## Reproduction Command
```bash
python docs/examples/full_year_regime_policy_benchmark_case_study.py
```

## Output Artifacts
Running the script writes deterministic stitched outputs to:

- docs/examples/output/full_year_regime_policy_benchmark_case_study/summary.json
- docs/examples/output/full_year_regime_policy_benchmark_case_study/manifest.json
- docs/examples/output/full_year_regime_policy_benchmark_case_study/benchmark_summary.json
- docs/examples/output/full_year_regime_policy_benchmark_case_study/promotion_gate_summary.json
- docs/examples/output/full_year_regime_policy_benchmark_case_study/review_summary.json
- docs/examples/output/full_year_regime_policy_benchmark_case_study/candidate_selection_summary.json
- docs/examples/output/full_year_regime_policy_benchmark_case_study/stress_summary.json
- docs/examples/output/full_year_regime_policy_benchmark_case_study/policy_variant_comparison.csv
- docs/examples/output/full_year_regime_policy_benchmark_case_study/final_interpretation.md
- docs/examples/output/full_year_regime_policy_benchmark_case_study/evidence_index.json
- docs/examples/output/full_year_regime_policy_benchmark_case_study/workflow_outputs.json
