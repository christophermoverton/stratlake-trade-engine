# Milestone 27 Merge Readiness

## Milestone Summary

Milestone 27 adds the market simulation stress-testing stack and the M26
integration bridge. The milestone covers the scenario framework, historical
episode replay, shock overlays, regime-aware block bootstrap, regime-transition
Monte Carlo, simulation-aware stress metrics, policy failure leaderboards, the
M27 market simulation case study, and optional M26 adaptive policy stress
integration.

M27 is research-governance evidence. It does not introduce live data, broker,
real-time, forecast, or trading-recommendation behavior.

## Branch / Target

* Source branch: `feature/m27-market-simulation-stress-testing`
* Release-readiness branch if needed:
  `feature/m27-release-readiness-and-merge`
* Target branch: `main`

## Documentation Refreshed

* `README.md`
* `docs/market_simulation_stress_testing.md`
* `docs/market_simulation_models_and_integrations.md`
* `docs/regime_policy_stress_testing.md`
* `docs/examples/m27_market_simulation_case_study.md`
* `docs/examples/m27_market_simulation_case_study_report.md`
* `docs/examples/full_year_regime_policy_benchmark_case_study.md`

## Artifact Hygiene Checks

Expected checks before merge:

| Check | Expected result |
| --- | --- |
| `git status --short` | Clean except intentional release-readiness edits before commit |
| `git diff --check` | Passed |
| `git ls-files --eol README.md docs/market_simulation_stress_testing.md docs/regime_policy_stress_testing.md .gitignore` | No unexpected line-ending rewrites |
| M27/M26 requested docs and tracked outputs absolute-path scan | No local user-home or machine-specific absolute paths |
| `docs/examples/output/m27_market_simulation_case_study/source_simulation_artifacts/` | Ignored/untracked |

Tracked fixture-backed M27 outputs remain compact:

* `docs/examples/output/m27_market_simulation_case_study/simulation_summary.json`
* `docs/examples/output/m27_market_simulation_case_study/leaderboard.csv`
* `docs/examples/output/m27_market_simulation_case_study/case_study_report.md`
* `docs/examples/output/m27_market_simulation_case_study/manifest.json`

Canonical source metrics remain under the ignored source-artifact root:

* `simulation_path_metrics.csv`
* `simulation_summary.csv`
* `simulation_leaderboard.csv`
* `policy_failure_summary.json`
* `simulation_metric_config.json`
* `manifest.json`

M26 stitched integration artifacts:

* `market_simulation_stress_summary.json`
* `market_simulation_stress_leaderboard.csv`

## Pre-Merge Validation Checklist

Pre-merge validation ran on
`feature/m27-market-simulation-stress-testing`.

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\ruff.exe check src docs tests configs` | Passed |
| `.\.venv\Scripts\python.exe -m pytest tests\test_market_simulation_policy_stress_integration.py` | Passed, 7 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_m27_market_simulation_case_study.py` | Passed, 7 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_simulation_stress_metrics.py tests\test_market_simulation_artifacts.py` | Passed, 28 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_regime_policy_stress_tests.py` | Passed, 16 tests |
| `.\.venv\Scripts\python.exe -m pytest tests\test_full_year_regime_policy_benchmark_case_study.py` | Passed, 8 tests |
| `.\.venv\Scripts\python.exe docs\examples\m27_market_simulation_case_study.py` | Passed |
| `.\.venv\Scripts\python.exe docs\examples\full_year_regime_policy_benchmark_case_study.py` | Passed; status `success (fixture_backed)` |
| `git diff --check` | Passed |
| `.\.venv\Scripts\python.exe -m pytest` | Passed, 1570 tests, 304 warnings |

Notes:

* The sandbox denied direct launches of the project venv Python, so pytest and
  example-script validations were rerun with approved escalation.
* The first full-suite attempt exposed an absolute-path guard finding in this
  readiness note; the wording was corrected and the full suite was rerun.

## Post-Merge Validation Checklist

Record final results here after merging to `main`:

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\ruff.exe check src docs tests configs` | Pending |
| Focused M27/M26 pytest commands | Pending |
| M27 case-study example | Pending |
| Full-year M26 case-study example | Pending |
| `git diff --check` | Pending |
| GitHub Actions merge commit status | Pending if available |

## Release Recommendation

Suggested tag:
`v0.27.0-market-simulation-stress-testing`

Suggested release title:
`Milestone 27 - Market Simulation Stress Testing and Adaptive Policy Integration`

Release notes should mention:

* market simulation scenario framework
* historical episode replay
* market shock overlays
* regime-aware block bootstrap
* regime-transition Monte Carlo
* simulation-aware stress metrics and policy failure leaderboards
* quantile-aware tail-risk metrics
* M27 market simulation case study
* M26 adaptive policy stress integration bridge
* `existing_artifacts` and `run_config` modes
* optional/unavailable evidence behavior
* strict `--require-market-simulation-stress` behavior
* documentation for market simulation models and integrations
* known limitations

## Known Limitations

* Fixture-backed examples validate workflow and artifact contracts only.
* Deterministic M26 stress transforms are governance diagnostics, not empirical
  simulations.
* Regime-transition Monte Carlo remains regime-only unless return or policy
  replay artifacts exist.
* Market simulation outputs are not forecasts or trading recommendations.
* M27 evidence complements but does not replace Issue #299 deterministic stress
  tests.

## Post-Merge Checklist

* Confirm `main` contains the no-fast-forward merge commit.
* Push `main` to `origin/main`.
* Run post-merge focused validation on `main`.
* Check GitHub Actions status when available.
* Draft release notes using the recommendation above.
* Add the final closing comment to Issue #313 with validation and merge
  references.
