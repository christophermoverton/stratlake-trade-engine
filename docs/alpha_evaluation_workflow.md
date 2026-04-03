# Alpha Evaluation Workflow

Milestone 12 adds a deterministic alpha-evaluation pipeline for answering one
question before downstream signal mapping or portfolio construction:

> Do this alpha model's predictions line up with future cross-sectional returns?

## Workflow

```text
Alpha -> Predict -> Align -> Validate -> Evaluate -> Aggregate -> Persist -> Register -> Compare
```

Milestone 11.5 covers deterministic alpha modeling, backtesting, and portfolio
construction. This Milestone 12 workflow is the evaluation layer that sits
between prediction and downstream signal mapping. Milestone 13 then adds the
registry-backed review layer that compares completed alpha-evaluation runs
alongside strategy and portfolio outputs in one shared review pack. See
[milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md).

## Quick Start

Run the committed end-to-end example:

```powershell
python docs/examples/alpha_evaluation_end_to_end.py
```

Run the CLI directly:

```powershell
python -m src.cli.run_alpha --alpha-name cs_linear_ret_1d --mode evaluate --start 2025-01-01 --end 2025-03-01
python -m src.cli.run_alpha --alpha-name rank_composite_momentum --start 2025-01-01 --end 2025-03-01
python -m src.cli.run_alpha --alpha-name cs_linear_ret_1d --start 2025-01-01 --end 2025-03-01 --signal-policy top_bottom_quantile --signal-quantile 0.2
python -m src.cli.run_alpha_evaluation --alpha-name cs_linear_ret_1d --start 2025-01-01 --end 2025-03-01
python -m src.cli.run_alpha_evaluation --alpha-model your_model --model-class path/to/model.py:YourModel --dataset features_daily --target-column target_ret_1d --price-column close
python -m src.cli.run_alpha_evaluation --alpha-model your_model --model-class path/to/model.py:YourModel --dataset features_daily --target-column target_ret_1d --price-column close --signal-policy rank_long_short
python -m src.cli.compare_alpha --from-registry
```

Built-in alpha definitions now live in `configs/alphas.yml`. Each named entry
declares the canonical dataset, target column, feature columns, model type,
model parameters, evaluation horizon, and optional defaults such as
`price_column` or `min_cross_section_size`. Use `--alpha-name` for those
registry-backed definitions, and keep `--model-class` for ad hoc external
models when you need a custom implementation. The registry now supports
`model_type: sklearn` for built-in scikit-learn regressors such as
`linear_regression` and `ridge`.

`python -m src.cli.run_alpha` is now the first-class built-in alpha runner.
It resolves one named config from `configs/alphas.yml`, runs it on
`features_daily`, prints a concise summary, and defaults to `full` mode. That
default currently persists the evaluation outputs plus an
`alpha_run_scaffold.json` file that marks sleeve generation as the next staged
step. Use `--mode evaluate` for evaluation-only runs with no scaffold file.

The example writes deterministic outputs under:

```text
docs/examples/output/alpha_evaluation_end_to_end/
```

## Stage By Stage

### Predict

`python -m src.cli.run_alpha_evaluation` trains a registered
`BaseAlphaModel`, then calls `predict_alpha_model(...)` to produce a canonical
prediction frame with `prediction_score`.

### Map Signals

Signal mapping is optional and explicit. When configured through
`signal_mapping` in YAML or the CLI flags `--signal-policy` plus
`--signal-quantile` where required, the workflow maps raw
`prediction_score` into a separate `signal` artifact.

Important boundary:

* alpha evaluation still scores raw `prediction_score` against `forward_return`
* mapped `signal` output is persisted for downstream backtesting or portfolio
  construction
* no score-to-signal conversion happens implicitly

### Align

`align_forward_returns(...)` joins predictions back to the loaded feature data
and creates `forward_return` for each `(symbol, ts_utc, timeframe)` row.

Supported alignment modes:

* `price_column`: computes forward return as `future_price / price - 1`
* `realized_return_column`: compounds future realized returns over the chosen
  horizon

### Validate

`validate_alpha_evaluation_input(...)` enforces the alpha-evaluation contract:

* required structural columns: `symbol`, `ts_utc`, `timeframe`
* numeric prediction and forward-return columns
* sorted, duplicate-free canonical rows
* enough cross-sectional breadth per timestamp
* cross-sectional variation in both predictions and forward returns

### Evaluate

`evaluate_alpha_predictions(...)` iterates by timestamp and computes:

* `ic`: Pearson correlation between `prediction_score` and `forward_return`
* `rank_ic`: Spearman-style rank correlation implemented by ranking both
  series, then applying Pearson correlation
* `sample_size`: number of non-null observations in that timestamp slice

The per-period output is written as `ic_timeseries.csv`.

### Aggregate

The evaluator summarizes the time series into deterministic metrics:

* `mean_ic`
* `std_ic`
* `ic_ir`
* `mean_rank_ic`
* `std_rank_ic`
* `rank_ic_ir`
* `n_periods`
* `ic_positive_rate`
* `valid_timestamps`

`ICIR` in the docs corresponds to the persisted `ic_ir` field. The rank-based
analogue is stored as `rank_ic_ir`.

### Persist

`write_alpha_evaluation_artifacts(...)` writes:

* `predictions.parquet`
* `signals.parquet` when explicit signal mapping is configured
* `signal_mapping.json` when explicit signal mapping is configured
* `training_summary.json`
* `coefficients.json`
* `cross_section_diagnostics.json`
* `ic_timeseries.csv`
* `alpha_metrics.json`
* `manifest.json`

Default location:

```text
artifacts/alpha/<run_id>/
```

### Register

`register_alpha_evaluation_run(...)` upserts one registry entry into:

```text
artifacts/alpha/registry.jsonl
```

Each entry links the run id, dataset, timeframe, evaluation horizon, summary
metrics, and concrete artifact paths.

### Compare

`python -m src.cli.compare_alpha --from-registry` loads alpha-evaluation runs
from the registry, ranks them by a selected metric, and writes a leaderboard.

Default comparison output:

```text
artifacts/alpha_comparisons/<comparison_id>/
```

Files:

* `leaderboard.csv`
* `leaderboard.json`

## Metrics

### IC

Information Coefficient is the same-timestamp cross-sectional Pearson
correlation between `prediction_score` and `forward_return`.

Interpretation:

* positive values mean higher scores tended to align with higher future returns
* negative values mean the alpha was directionally wrong
* values near zero suggest weak or unstable cross-sectional signal

### Rank IC

Rank IC is the same relationship computed on ranked values instead of raw
levels. It is more robust when the alpha mainly gets ordering right but not
magnitude.

### ICIR

ICIR is the mean IC divided by the sample standard deviation of IC across
timestamps. Higher values imply more stable signal quality over time.

In the implementation:

* ICIR is stored as `ic_ir`
* Rank ICIR is stored as `rank_ic_ir`

## How To Read Results

Use `alpha_metrics.json` for the headline summary and `ic_timeseries.csv` to
see whether the result is stable across time rather than driven by one period.

Common reading pattern:

* `mean_ic` tells you the average directional relationship
* `ic_ir` tells you whether that relationship is repeatable
* `mean_rank_ic` helps when ranking matters more than magnitude
* `n_periods` tells you how many timestamps were actually usable

## Artifacts And Registry Entries

Artifacts are the run-local files under `artifacts/alpha/<run_id>/`.
Registry entries are the cross-run index that point back to those files.

That relationship enables two workflows:

* audit one run by opening `training_summary.json`, `coefficients.json`,
  `cross_section_diagnostics.json`, `alpha_metrics.json`, `ic_timeseries.csv`,
  and `manifest.json`
* compare many runs by loading the registry and generating a leaderboard

## Start Here

* [README.md](../README.md)
* [docs/examples/alpha_evaluation_end_to_end.py](examples/alpha_evaluation_end_to_end.py)
* [docs/alpha_workflow.md](alpha_workflow.md)
* [docs/milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md)
