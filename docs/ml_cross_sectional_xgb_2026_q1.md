# ML Cross-Sectional XGBoost Alpha Case Study

`ml_cross_sectional_xgb_2026_q1` is a built-in alpha case study that trains an
XGBoost cross-sectional regressor on `features_daily` and predicts
`target_ret_5d`.

## Objective

Demonstrate that a trained ML alpha can move through the same first-class
StratLake pipeline as the deterministic alpha baselines:

* train on `2026-01-01 <= ts_utc < 2026-03-02`
* predict on `2026-03-02 <= ts_utc < 2026-04-03`
* emit canonical alpha predictions as `symbol, ts_utc, alpha_score`
* evaluate with IC and ICIR
* map predictions with `top_bottom_quantile`, `q=0.2`
* build a sleeve and make it loadable as `artifact_type: alpha_sleeve`
* compare the run via the alpha comparison CLI and unified review

## Model

Catalog entry:

* config file: `configs/alphas_2026_q1.yml`
* alpha name: `ml_cross_sectional_xgb_2026_q1`
* model type: `cross_sectional_xgboost`
* target: `target_ret_5d`
* features: all current `features_daily` feature columns declared in
  `configs/features.yml`

Determinism controls:

* fixed `random_state`
* sorted training and prediction inputs on `(symbol, ts_utc)`
* `subsample=1.0`
* `colsample_bytree=1.0`
* `colsample_bylevel=1.0`
* `colsample_bynode=1.0`
* `n_jobs=1`

Reasonable default hyperparameters are recorded directly in the alpha config and
copied into the persisted manifest/training summary. No tuning step is used.

## Run

```bash
python -m src.cli.run_alpha ^
  --config configs/alphas_2026_q1.yml ^
  --alpha-name ml_cross_sectional_xgb_2026_q1
```

Expected alpha artifacts:

* `predictions.parquet`
* `alpha_metrics.json`
* `ic_timeseries.csv`
* `signals.parquet`
* `sleeve_returns.csv`
* `sleeve_equity_curve.csv`
* `sleeve_metrics.json`
* `manifest.json`

The manifest now includes:

* model type
* feature columns
* hyperparameters

## Compare To Baseline

Suggested baseline:

* `rank_composite_momentum_2026_q1`

Run the baseline:

```bash
python -m src.cli.run_alpha ^
  --config configs/alphas_2026_q1.yml ^
  --alpha-name rank_composite_momentum_2026_q1
```

Then compare:

```bash
python -m src.cli.compare_alpha ^
  --artifacts-root artifacts/alpha ^
  --dataset features_daily ^
  --timeframe 1D
```

Or review the alpha beside downstream portfolio runs:

```bash
python -m src.cli.comparison_cli ^
  --run-types alpha_evaluation portfolio
```

## Results

Run executed on the repository's local `features_daily` dataset:

* ML alpha run id:
  `ml_cross_sectional_xgb_2026_q1_alpha_eval_d4c845bd3f5c`
* baseline run id:
  `rank_composite_momentum_2026_q1_alpha_eval_f3b800de1a88`
* linked portfolio run id:
  `ml_cross_sectional_xgb_2026_q1_equal_portfolio_ed7e55c30b30`

Observed forecast metrics:

| Alpha | Mean IC | ICIR | Mean Rank IC | Rank ICIR | Periods |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ml_cross_sectional_xgb_2026_q1` | 0.2546 | 1.9198 | 0.2252 | 3.0829 | 19 |
| `rank_composite_momentum_2026_q1` | -0.0079 | -0.0454 | -0.0019 | -0.0117 | 19 |

Observed sleeve metrics:

| Alpha | Total Return | Sharpe | Max Drawdown | Win Rate | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ml_cross_sectional_xgb_2026_q1` | 2.87% | 12.47 | 0.06% | 70.83% | 0.2583 |
| `rank_composite_momentum_2026_q1` | -1.12% | -2.83 | 1.36% | 45.83% | 0.1733 |

Registry-backed comparison output:

* comparison id: `registry_alpha_combined_425b7424a601`
* review id: `registry_review_f93a6729b9fd`

Use these artifacts for the comparison summary:

* `alpha_metrics.json` for IC and ICIR
* `sleeve_metrics.json` for sleeve performance
* `artifacts/alpha_comparisons/.../leaderboard.csv` for registry-backed ranking
* `artifacts/reviews/.../leaderboard.csv` for unified review output

## Notes

* The implementation uses only `feature_*` inputs.
* Targets remain forward-looking labels only in the training window.
* Prediction outputs remain standard pipeline artifacts and do not change the
  deterministic workflows used by existing built-in alphas.
* `configs/portfolios_alpha_2026_q1.yml` includes a working
  `ml_cross_sectional_xgb_2026_q1_equal` example that consumes the alpha sleeve
  with `artifact_type: alpha_sleeve`.
