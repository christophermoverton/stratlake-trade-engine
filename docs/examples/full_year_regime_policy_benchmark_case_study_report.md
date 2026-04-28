# Full-Year Regime Policy Benchmark Case Study Findings Report

## Executive Summary

The full-year regime policy benchmark case study validates that the Milestone 26 workflow can stitch benchmark, promotion-gate, review, candidate-selection, and deterministic stress evidence into one reproducible research package.

The strongest governance result is the `gmm_confidence_policy`: it ranked first in the review leaderboard, passed 11 gates, had no failed gates, and showed very high classifier confidence (`mean_confidence=0.999770`). The static, taxonomy-only, calibrated taxonomy, and GMM confidence variants were accepted. The hybrid calibrated GMM policy was rejected because its adaptive return delta versus the static baseline was negative (`-0.108847`) even though its Sharpe delta was positive (`0.286653`) and its stress ranking was strong.

Stress evidence favored adaptive behavior, but that evidence is synthetic and diagnostic. Under the deterministic stress transforms, `policy_optimized` ranked first with a 100% stress pass rate, the highest mean stress Sharpe (`1.026667`), and the least severe mean stress max drawdown (`-0.129667`). The static baseline ranked third with an 83.33% stress pass rate.

## 1. Evidence Scope

The report is based on artifacts generated under:

```text
docs/examples/output/full_year_regime_policy_benchmark_case_study/
```

Primary evidence files used:

- `summary.json`
- `benchmark_summary.json`
- `promotion_gate_summary.json`
- `review_summary.json`
- `candidate_selection_summary.json`
- `stress_summary.json`
- `policy_variant_comparison.csv`
- `final_interpretation.md`
- `workflow_outputs/benchmark/.../benchmark_matrix.csv`
- `workflow_outputs/review/.../leaderboard.csv`
- `workflow_outputs/stress_tests/.../stress_leaderboard.csv`
- `workflow_outputs/candidate_selection/.../candidate_selection.csv`

The evaluation window is `2026-01-01` through `2026-12-31` at daily frequency. The run is fixture-backed and deterministic, so the findings are suitable for workflow validation and research governance review. They are not production trading evidence.

## 2. Benchmark Coverage

The benchmark layer compared five policy variants:

| Rank Surface | Policy Variant | Role | Regime Source | Calibration | Classifier |
|---:|---|---|---|---|---|
| 4 | `static_baseline` | static baseline | static | none | none |
| 3 | `taxonomy_only_policy` | taxonomy-only | taxonomy | none | none |
| 2 | `calibrated_taxonomy_policy` | calibrated taxonomy | taxonomy | baseline | none |
| 1 | `gmm_confidence_policy` | GMM confidence overlay | gmm | none | gaussian mixture model |
| 5 | `hybrid_calibrated_gmm_policy` | adaptive policy | policy | baseline | gaussian mixture model |

The benchmark run produced one warning: conditional-performance comparison was enabled, but the dedicated conditional-performance comparison remains reserved for follow-up work.

## 3. Governance Findings

Promotion gates accepted four of five variants:

| Variant | Decision | Passed Gates | Failed Gates | Warning Gates | Primary Reason |
|---|---|---:|---:|---:|---|
| `static_baseline` | accepted | 0 | 0 | 0 | No gate issues recorded |
| `taxonomy_only_policy` | accepted | 6 | 0 | 0 | Avg regime duration passed |
| `calibrated_taxonomy_policy` | accepted | 6 | 0 | 0 | Avg regime duration passed |
| `gmm_confidence_policy` | accepted | 11 | 0 | 0 | Mean confidence passed |
| `hybrid_calibrated_gmm_policy` | rejected | 16 | 1 | 1 | Adaptive return delta failed |

The notable governance tension is the hybrid policy. It passed most gates and improved Sharpe versus static, but the strict promotion policy rejected it because the return delta was negative:

```text
adaptive_vs_static_return_delta = -0.108847
adaptive_vs_static_sharpe_delta = 0.286653
adaptive_vs_static_max_drawdown_delta = -0.004566
```

This is a useful governance outcome. The workflow does not automatically promote a more complex adaptive policy just because it improves one metric or stress profile. The return shortfall blocks promotion under the configured policy.

## 4. Regime and Classifier Findings

The taxonomy and calibrated taxonomy variants both covered 365 observations. Calibration reduced regime churn:

| Variant | Regime Count | Avg Duration | Transition Count | Transition Rate | Dominant Regime Share |
|---|---:|---:|---:|---:|---:|
| `taxonomy_only_policy` | 37 | 9.864865 | 34 | 0.093407 | 0.317808 |
| `calibrated_taxonomy_policy` | 27 | 13.518519 | 25 | 0.068681 | 0.317808 |
| `gmm_confidence_policy` | 37 | 9.864865 | 34 | 0.093407 | 0.317808 |
| `hybrid_calibrated_gmm_policy` | 27 | 13.518519 | 25 | 0.068681 | 0.317808 |

The GMM confidence overlay was extremely confident on this deterministic fixture:

```text
mean_confidence = 0.999770
median_confidence = 1.000000
min_confidence = 0.963509
low_confidence_observation_count = 0
taxonomy_ml_disagreement_rate = 0.037791
```

That confidence result explains why the GMM variant led the review leaderboard. It also deserves caution: fixture-backed confidence can validate plumbing and governance rules, but should not be interpreted as a live classifier calibration claim.

## 5. Review Leaderboard Findings

The review pack ranked accepted variants above the rejected hybrid policy:

| Rank | Variant | Decision | Accepted | Failed Gates | Passed Gates |
|---:|---|---|---|---:|---:|
| 1 | `gmm_confidence_policy` | accepted | true | 0 | 11 |
| 2 | `calibrated_taxonomy_policy` | accepted | true | 0 | 6 |
| 3 | `taxonomy_only_policy` | accepted | true | 0 | 6 |
| 4 | `static_baseline` | accepted | true | 0 | 0 |
| 5 | `hybrid_calibrated_gmm_policy` | rejected | false | 1 | 16 |

The top reviewed policy was `gmm_confidence_policy`, with the primary reason:

```text
Metric mean_confidence passed: 0.99977 >= 0.55.
```

## 6. Candidate-Selection Findings

Candidate selection produced six selected candidates. Category counts overlap because a candidate can serve multiple roles:

| Category | Selected Count |
|---|---:|
| `transition_resilient` | 5 |
| `global_performer` | 4 |
| `regime_specialist` | 4 |
| `defensive_fallback` | 3 |

Selected candidates:

| Candidate | Artifact Type | Global Rank | Selection Category |
|---|---|---:|---|
| `Alpha Global Carry` | alpha | 1 | global performer, regime specialist, transition resilient |
| `Portfolio Transition Sleeve` | portfolio | 3 | defensive fallback, global performer, regime specialist, transition resilient |
| `Alpha Raw Metric Blend` | alpha | 4 | defensive fallback, global performer, transition resilient |
| `Alpha Trend Specialist` | alpha | 5 | global performer, regime specialist, transition resilient |
| `Portfolio Defensive Sleeve` | portfolio | 6 | defensive fallback |
| `Strategy Chop Specialist` | strategy | 7 | regime specialist, transition resilient |

`Alpha Redundant Carry` ranked second globally but was not selected because the configured selection and redundancy rules did not retain it. This gives the case study a realistic governance behavior: rank alone does not guarantee final selection.

## 7. Stress Findings

The deterministic stress layer evaluated three policies over six scenarios:

| Stress Rank | Policy | Type | Pass Rate | Mean Stress Return | Mean Stress Sharpe | Mean Stress Max Drawdown | Worst Scenario |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `policy_optimized` | adaptive | 1.000000 | 0.058167 | 1.026667 | -0.129667 | `confidence_collapse` |
| 2 | `gmm_calibrated_overlay` | adaptive | 1.000000 | 0.052500 | 0.950000 | -0.136833 | `high_vol_persistence` |
| 3 | `static_baseline` | static | 0.833333 | 0.033333 | 0.731667 | -0.155000 | `high_vol_persistence` |

The most resilient policy under stress was `policy_optimized`. It improved mean stress return by `0.024833`, mean stress Sharpe by `0.295000`, and mean drawdown by `0.025333` versus the static baseline in the stress comparison.

This stress result conflicts productively with the promotion-gate result. The adaptive policy looks best under synthetic stress, but governance still rejects the stitched hybrid policy because the observed benchmark return delta is negative.

## 8. Main Findings

1. `gmm_confidence_policy` is the cleanest accepted policy in the run. It ranks first in review, passes confidence gates, has no failed or warning gates, and has no low-confidence observations in the fixture-backed GMM overlay.

2. Calibration reduces transition churn. The calibrated taxonomy path lowers the transition rate from `0.093407` to `0.068681` and increases average regime duration from `9.864865` to `13.518519` observations while preserving the same dominant regime share.

3. The hybrid adaptive policy is not promotion-ready under the configured gates. It improves Sharpe but loses return versus static, which causes a strict rejection.

4. Synthetic stress favors adaptive policies. `policy_optimized` and `gmm_calibrated_overlay` both pass all six stress scenarios, while the static baseline passes five of six.

5. Candidate selection is role-diversified. The selected set spans alpha, portfolio, and strategy artifacts, with heavy representation in transition-resilient and regime-specialist roles.

6. The case study successfully separates evidence types. Observed fixture-backed benchmark evidence, governance review evidence, candidate-selection evidence, and deterministic synthetic stress evidence are stitched together without claiming that stress transforms are empirical market simulations.

## 9. Limitations

- The benchmark evidence is fixture-backed. It validates reproducibility and governance flow, not live market edge.
- Stress results are deterministic synthetic transforms, not historical market simulations.
- Two stress policies used fallback baseline sourcing because no base row was available for `gmm_calibrated_overlay` or `policy_optimized`.
- Registry-backed candidate sourcing is reserved for future expansion.
- Conditional-performance comparison is still a follow-up item.

## 10. Recommended Follow-Ups

1. Replace fixture-backed benchmark inputs with fully covered real full-year benchmark outputs.
2. Re-run the governance gates with sensitivity profiles that separately test return, Sharpe, and drawdown trade-offs.
3. Add dedicated conditional-performance comparisons to close the current benchmark warning.
4. Validate whether the high GMM confidence profile persists on real full-year data.
5. Expand candidate sourcing from registry-backed evidence once that workflow is available.

## Reproduction Command

```bash
python docs/examples/full_year_regime_policy_benchmark_case_study.py
```