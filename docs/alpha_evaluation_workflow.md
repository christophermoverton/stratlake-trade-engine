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
between prediction and downstream signal mapping.

## Quick Start

Run the committed end-to-end example:

```powershell
python docs/examples/alpha_evaluation_end_to_end.py
```

Run the CLI directly:

```powershell
python -m src.cli.run_alpha_evaluation --alpha-model your_model --model-class path/to/model.py:YourModel --dataset features_daily --target-column target_ret_1d --price-column close
python -m src.cli.compare_alpha --from-registry
```

The example writes deterministic outputs under:

```text
docs/examples/output/alpha_evaluation_end_to_end/
```

## Stage By Stage

### Predict

`python -m src.cli.run_alpha_evaluation` trains a registered
`BaseAlphaModel`, then calls `predict_alpha_model(...)` to produce a canonical
prediction frame with `prediction_score`.

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

* audit one run by opening `alpha_metrics.json`, `ic_timeseries.csv`, and
  `manifest.json`
* compare many runs by loading the registry and generating a leaderboard

## Start Here

* [README.md](../README.md)
* [docs/examples/alpha_evaluation_end_to_end.py](examples/alpha_evaluation_end_to_end.py)
* [docs/alpha_workflow.md](alpha_workflow.md)
