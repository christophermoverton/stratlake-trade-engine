# Full-Year 2025 Regime Calibration Case Study

## Research Question

How does the full regime-aware research stack behave across the full 2025
calendar year when we compare:

1. static baseline strategy and portfolio behavior
2. raw Milestone 24 regime attribution and transition analysis
3. Milestone 25 calibration profile evaluation
4. GMM-based confidence and uncertainty diagnostics
5. adaptive regime-aware portfolio policy behavior

## Full-Year 2025 Data Window

The case study targets the full 2025 calendar year:

- requested start date: `2025-01-01`
- requested end date: `2025-12-31`
- loader end-exclusive boundary: `2026-01-01`

The market input uses repository-local `features_daily` data under
`data/curated/features_daily`.

## Real Downloaded Data Requirement

This workflow is intentionally strict:

- it uses real repository data only
- it does not create mock, synthetic, random, or stub market data
- it does not silently fall back to fixture data
- it fails fast if full-year 2025 coverage is missing or incomplete

Tests may call the example with an explicit real-data fixture root only when
that path is marked as fixture coverage and `allow_real_data_fixture=True`.
The default production path requires the repository's downloaded 2025 data.

## Data Coverage Validation

Before any regime workflow runs, the script validates:

- `features_daily` exists
- `ts_utc` or `date` is present
- row count is non-zero
- symbol count is non-zero
- 2025 symbol partitions exist
- observed coverage reaches early January 2025 and late December 2025
- all 12 calendar months are present
- unique-date coverage is large enough for a real full-year study
- large date gaps are surfaced as diagnostics

Coverage is persisted to:

```text
docs/examples/output/full_year_regime_calibration_case_study/data_coverage_summary.json
```

That artifact records:

- requested start and end dates
- observed min and max timestamp/date
- symbol count
- row count
- coverage-gap diagnostics
- pass/fail status
- whether the data source was downloaded real data or an explicit real-data fixture

## Input Assumptions

The example uses:

- dataset: `features_daily`
- market basket: first deterministic set of repository symbols with `year=2025`
  partitions
- static strategies: `momentum_v1` and `mean_reversion_v1`
- static portfolio: equal-weight portfolio built from those two completed
  strategy runs
- downstream calibration profile: `baseline`

The basket is used for market-regime classification only. Strategy and
portfolio runs still execute against the repository's standard research inputs.

## Workflow Stages

1. validate real 2025 data coverage and write `data_coverage_summary.json`
2. load a deterministic 2025 market basket from `features_daily`
3. run the static baseline strategy set and build a static portfolio
4. classify market regimes with the Milestone 24 taxonomy
5. write raw regime classification artifacts
6. evaluate raw strategy and portfolio attribution and transition behavior
7. fit the Milestone 25 GMM confidence layer on canonical regime metrics
8. evaluate all built-in calibration profiles with GMM confidence input
9. rerun the portfolio regime interpretation using the downstream calibrated labels
10. apply regime policy decisions to the calibrated portfolio surface
11. compare adaptive versus static portfolio behavior
12. write `summary.json`, `final_interpretation.md`, and artifact inventory outputs

## Static Baseline Behavior

The static section is deliberately simple:

- two deterministic strategy runs cover the full 2025 window
- a static portfolio combines them without any regime-aware overlay
- the summary captures the raw strategy metrics, raw portfolio metrics, and
  regime-comparison context across the baseline strategies

This gives the adaptive layer a concrete baseline instead of a synthetic
reference point.

## M24 Taxonomy and Attribution Findings

Raw Milestone 24 outputs include:

- `regime_bundle/`
- `strategy_momentum_v1_bundle/`
- `strategy_mean_reversion_v1_bundle/`
- `static_portfolio_bundle/`

These bundles preserve the canonical taxonomy labels, raw attribution reports,
and transition summaries before calibration or adaptive policy is introduced.

## Transition Analysis Findings

Transition analysis remains part of the base workflow rather than a side note.
The case study writes transition event, window, and summary artifacts for both:

- static strategies
- static portfolio

That keeps the later calibration and policy interpretation grounded in the
actual regime changes observed across 2025.

## Calibration Findings

The script evaluates every built-in calibration profile:

- `baseline`
- `conservative`
- `reactive`
- `crisis_sensitive`

Artifacts are written under:

```text
docs/examples/output/full_year_regime_calibration_case_study/calibration_profiles/
```

The summary compares their:

- stability metrics
- warning counts
- profile flags

The downstream calibrated portfolio interpretation uses the `baseline` profile.

## GMM Confidence Findings

The GMM layer is used only as a confidence and uncertainty surface.

It writes:

- `regime_gmm_labels.csv`
- `regime_gmm_posteriors.csv`
- `regime_gmm_shift_events.csv`
- `regime_gmm_summary.json`
- `regime_gmm_manifest.json`

These artifacts help answer:

- where the regime surface was low confidence
- when posterior shifts looked material
- whether uncertainty clusters around the same periods later flagged by policy fallbacks

## Adaptive vs Static Comparison

Adaptive behavior is applied to the calibrated portfolio surface only.

The policy layer consumes:

- calibrated taxonomy labels
- calibration stability flags
- GMM confidence metadata
- a deterministic portfolio baseline frame

It writes:

```text
docs/examples/output/full_year_regime_calibration_case_study/adaptive_policy/
```

including:

- `regime_policy_decisions.csv`
- `regime_policy_summary.json`
- `adaptive_vs_static_comparison.csv`
- `adaptive_policy_manifest.json`

## Final Interpretation

`final_interpretation.md` summarizes:

- whether calibration changed the strongest or weakest interpreted regimes
- whether GMM confidence highlighted uncertain periods
- whether adaptive behavior improved or degraded the static baseline
- where the static baseline remained competitive
- what follow-up questions remain

## Limitations

- The market regime surface uses repository symbols, not a third-party benchmark index.
- The adaptive layer is a research diagnostic, not live execution logic.
- GMM confidence does not replace or rename the Milestone 24 taxonomy.
- The policy configuration is intentionally conservative and not tuned to force outperformance.

## Non-Goals

- no live trading behavior
- no external downloads during tests
- no new taxonomy labels
- no GMM cluster relabeling as canonical regimes
- no synthetic data fallback to make the example pass

## Execute

```powershell
python docs/examples/full_year_regime_calibration_case_study.py
```

Optional output-root override:

```powershell
python docs/examples/full_year_regime_calibration_case_study.py --output-root docs/examples/output/full_year_regime_calibration_case_study
```
