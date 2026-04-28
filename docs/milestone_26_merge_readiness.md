# Milestone 26 Merge Readiness

## Milestone Summary

Milestone 26 finalizes a governed adaptive regime-policy research workflow. The
workflow starts with deterministic regime benchmark packs, evaluates configured
promotion gates, packages review evidence and decision logs, selects
regime-aware candidates, runs adaptive policy stress tests, and stitches the
result into a fixture-backed full-year benchmark case study.

This is a research-governance layer. It documents deterministic evidence and
review readiness, but it does not introduce live trading, production
deployment, or new research semantics.

## Branch / Target

* Source branch: `feature/m26-regime-policy-governance-benchmarking`
* Requested milestone branch name:
  `feature/m25-regime-calibration-adaptive-optimization`
* Target branch: `main`
* Branch note: local inspection found
  `feature/m26-regime-policy-governance-benchmarking` as the actual current
  Milestone 26 branch.

## Issues Included

* Issue 295: Regime Benchmark Pack Infrastructure
* Issue 296: Regime Promotion Gates & Decision Outcomes
* Issue 297: Regime Review Pack & Decision Log Artifacts
* Issue 298: Regime-Aware Candidate Selection
* Issue 299: Adaptive Policy Stress Testing
* Issue 300: Full-Year Regime Policy Benchmark Case Study
* Issue 301: Merge Readiness, Documentation Refresh, and Mainline Validation

## Major Capabilities Added

* Deterministic regime benchmark packs for static, taxonomy, calibrated, GMM,
  and policy-optimized variants.
* Configurable promotion gates and deterministic variant decision outcomes.
* Review packs with leaderboards, decision logs, evidence indexes, manifests,
  and optional Markdown reports.
* Regime-aware candidate selection for global performers, regime specialists,
  transition-resilient candidates, and defensive fallbacks.
* Adaptive policy stress testing under deterministic adverse and ambiguous
  regime conditions.
* A fixture-backed full-year case study that stitches benchmark, gate, review,
  candidate-selection, and stress evidence into one reproducible output set.

## New CLI Entrypoints / Scripts

```powershell
python -m src.cli.run_regime_benchmark_pack --config configs/regime_benchmark_packs/m26_regime_policy_benchmark.yml
python -m src.cli.run_regime_promotion_gates --benchmark-path artifacts/regime_benchmarks/<benchmark_run_id> --config configs/regime_promotion_gates/m26_regime_policy_gates.yml
python -m src.cli.generate_regime_review_pack --config configs/regime_reviews/m26_regime_review_pack.yml
python -m src.cli.run_regime_aware_candidate_selection --config configs/candidate_selection/m26_regime_aware_candidate_selection.yml
python -m src.cli.run_regime_policy_stress_tests --config configs/regime_stress_tests/m26_adaptive_policy_stress.yml
python docs/examples/full_year_regime_policy_benchmark_case_study.py
```

## New Configs

* `configs/regime_benchmark_packs/m26_regime_policy_benchmark.yml`
* `configs/regime_promotion_gates/m26_regime_policy_gates.yml`
* `configs/regime_reviews/m26_regime_review_pack.yml`
* `configs/candidate_selection/m26_regime_aware_candidate_selection.yml`
* `configs/regime_stress_tests/m26_adaptive_policy_stress.yml`

## New Artifact Roots

* `artifacts/regime_benchmarks/`
* `artifacts/regime_benchmarks/<benchmark_run_id>/promotion_gates/`
* `artifacts/regime_reviews/`
* `artifacts/candidate_selection/`
* `artifacts/regime_stress_tests/`
* `docs/examples/output/full_year_regime_policy_benchmark_case_study/`

## Documentation Updated

* `README.md`
* `docs/milestone_26_merge_readiness.md`

## Pre-Merge Validation Results

Pre-merge validation ran on
`feature/m26-regime-policy-governance-benchmarking` after `git fetch origin`.
The branch was clean before validation and was tracking
`origin/feature/m26-regime-policy-governance-benchmarking`.

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\ruff.exe check src docs tests configs` | Passed |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_benchmark_pack_config.py tests\test_regime_benchmark_pack.py` | Passed, 8 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_promotion_gates_config.py tests\test_regime_promotion_gates.py` | Passed, 17 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_review_pack_config.py tests\test_regime_review_pack.py` | Passed, 18 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_aware_candidate_selection_config.py tests\test_regime_aware_candidate_selection.py` | Passed, 17 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_policy_stress_tests_config.py tests\test_regime_policy_stress_tests.py` | Passed, 24 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_full_year_regime_policy_benchmark_case_study.py` | Passed, 6 tests |
| `.\.venv\Scripts\python.exe docs\examples\full_year_regime_policy_benchmark_case_study.py` | Passed; status `success (fixture_backed)` |
| `.\.venv\Scripts\python.exe -m pytest` | Passed, 1448 tests, 304 warnings |

Runtime-generated
`docs/examples/output/full_year_regime_policy_benchmark_case_study/` outputs
were removed after validation.

## Known Limitations

* The flagship full-year benchmark case study is fixture-backed for
  deterministic portability.
* Deterministic stress transforms are governance diagnostics, not empirical
  market simulations.
* Registry-backed expansion remains future work for selected file-backed
  candidate and policy sourcing workflows.
* No production or trading readiness claim is made.

## Merge Procedure

Actual merge commit:
`ebdf3be0173c5489c1b0086988e9268abc89c73b`

Commands used:

```powershell
git fetch origin
git checkout main
git pull origin main
git merge --no-ff feature/m26-regime-policy-governance-benchmarking
```

Conflict resolution: no conflicts.

## Post-Merge Validation Results

Post-merge validation ran on `main` after the no-fast-forward merge.

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\ruff.exe check src docs tests configs` | Passed |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_benchmark_pack_config.py tests\test_regime_benchmark_pack.py` | Passed, 8 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_promotion_gates_config.py tests\test_regime_promotion_gates.py` | Passed, 17 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_review_pack_config.py tests\test_regime_review_pack.py` | Passed, 18 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_aware_candidate_selection_config.py tests\test_regime_aware_candidate_selection.py` | Passed, 17 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_policy_stress_tests_config.py tests\test_regime_policy_stress_tests.py` | Passed, 24 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_full_year_regime_policy_benchmark_case_study.py` | Passed, 6 tests |
| `.\.venv\Scripts\python.exe docs\examples\full_year_regime_policy_benchmark_case_study.py` | Passed; status `success (fixture_backed)` |
| `.\.venv\Scripts\python.exe -m pytest` | Passed, 1448 tests, 304 warnings |

Tests not run: none. Full-suite validation was practical and completed.

Runtime-generated
`docs/examples/output/full_year_regime_policy_benchmark_case_study/` outputs
were removed after post-merge validation.

## Release Recommendation

Suggested tag:
`v0.26.0-regime-policy-benchmark-governance`

Suggested release title:
`Milestone 26 - Regime Policy Benchmark Governance`
