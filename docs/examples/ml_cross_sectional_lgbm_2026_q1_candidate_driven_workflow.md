# LightGBM Cross-Sectional Alpha Case Study (Q1 2026)

## Objective

Run `ml_cross_sectional_lgbm_2026_q1` end to end through the existing deterministic
research stack:

1. alpha evaluation and sleeve generation
2. registry-backed alpha comparison
3. candidate selection and review
4. candidate-driven portfolio construction

This workflow reuses current architecture and artifact contracts. No special-case
CLI behavior is required for LightGBM.

## Prerequisites

- Feature dataset and targets available through `features_daily`
- Python dependencies installed (including `lightgbm`)

Install dependencies:

```powershell
pip install -e .
```

## 1. Run LightGBM Alpha Evaluation + Sleeve

```powershell
python -m src.cli.run_alpha --config configs/alphas_2026_q1.yml --alpha-name ml_cross_sectional_lgbm_2026_q1
```

This writes deterministic alpha artifacts under `artifacts/alpha/<run_id>/`,
including prediction, evaluation, and sleeve files.

## 2. Compare Against Existing Alpha Runs

```powershell
python -m src.cli.compare_alpha --from-registry --artifacts-root artifacts/alpha --dataset features_daily --timeframe 1D
```

This writes comparison outputs under `artifacts/alpha_comparisons/` and ranks the
new LightGBM run with the same forecast and sleeve metrics used by other alphas.

## 3. Run Candidate Selection

```powershell
python -m src.cli.run_candidate_selection --config configs/candidate_selection.yml --dataset features_daily --mapping-name top_bottom_quantile_q20 --min-history-length 10 --min-ic -1 --min-ic-ir -1 --min-allocation-candidate-count 1 --max-weight-per-candidate 1.0
```

`configs/candidate_selection.yml` defaults are tuned for a different surface
(`bars_daily`, `rank_long_short`, longer history). The explicit overrides above
align candidate filtering with this case study (`features_daily`,
`top_bottom_quantile_q20`, current history length).

This applies eligibility, redundancy filtering, and allocation, then persists
selected candidates under `artifacts/candidate_selection/<run_id>/`.

If review artifacts are needed, run review after portfolio creation using:

```powershell
python -m src.cli.run_candidate_selection --candidate-selection-run-id <candidate_run_id> --enable-review --portfolio-run-id <portfolio_run_id>
```

## 4. Build Candidate-Driven Portfolio

Use the candidate-selection artifact directory from Step 3:

```powershell
python -m src.cli.run_portfolio --from-candidate-selection artifacts/candidate_selection/<candidate_run_id> --portfolio-name q1_2026_lgbm_candidate_driven --timeframe 1D
```

This constructs a governed candidate-driven portfolio from selected sleeves and
writes portfolio artifacts under `artifacts/portfolios/<portfolio_run_id>/`.

## 5. Unified Research Review (Optional)

```powershell
python -m src.cli.compare_research --from-registry --run-types alpha_evaluation portfolio --dataset features_daily --timeframe 1D --disable-plots
```

Use this to audit alpha and portfolio outcomes together with deterministic
registry-backed metadata.

## Observed Run Results (Validated 2026-04-07)

### Alpha Run

- alpha name: `ml_cross_sectional_lgbm_2026_q1`
- run id: `ml_cross_sectional_lgbm_2026_q1_alpha_eval_c561621a31d9`
- mean_ic: `0.248483`
- ic_ir: `1.589419`
- n_periods: `19`
- artifact dir: `artifacts/alpha/ml_cross_sectional_lgbm_2026_q1_alpha_eval_c561621a31d9`

### Alpha Comparison

- comparison id: `registry_alpha_forecast_760216bfcb7c`
- leaderboard:
	- 1: `ml_cross_sectional_xgb_2026_q1` (`ic_ir=1.919811`)
	- 2: `ml_cross_sectional_lgbm_2026_q1` (`ic_ir=1.589419`)
- artifacts:
	- `artifacts/alpha_comparisons/registry_alpha_forecast_760216bfcb7c/leaderboard.csv`
	- `artifacts/alpha_comparisons/registry_alpha_forecast_760216bfcb7c/leaderboard.json`

### Candidate Selection

- run id: `candidate_selection_fcf283ff3690`
- summary:
	- total candidates: `4`
	- eligible: `4`
	- rejected by redundancy: `1`
	- selected: `3`
- artifact dir: `artifacts/candidate_selection/candidate_selection_fcf283ff3690`

### Candidate-Driven Portfolio

- portfolio name: `q1_2026_lgbm_candidate_driven`
- run id: `q1_2026_lgbm_candidate_driven_portfolio_48e3fd4caa3f`
- total return: `1.37%`
- sharpe ratio: `8.49`
- max drawdown: `0.16%`
- artifact dir: `artifacts/portfolios/q1_2026_lgbm_candidate_driven_portfolio_48e3fd4caa3f`

### Unified Review

- review id: `registry_review_04b815ff1d5c`
- output artifacts:
	- `artifacts/reviews/registry_review_04b815ff1d5c/leaderboard.csv`
	- `artifacts/reviews/registry_review_04b815ff1d5c/review_summary.json`
