# Alpha Workflow

## Overview

The alpha layer adds a deterministic modeling workflow on top of the canonical
research frame used elsewhere in StratLake.

It is designed for workflows where you want to:

* register an alpha model behind a stable interface
* fit the model on a bounded historical window
* generate deterministic out-of-sample predictions
* inspect same-timestamp cross-sections before signal mapping
* hand prediction-derived exposures into the backtest and portfolio layers

The alpha modules live under `src/research/alpha/`.

## End-To-End Flow

```text
validated features
    ->
alpha model registry
    ->
train_alpha_model(...)
    ->
predict_alpha_model(...)
    ->
cross-sectional review
    ->
signal mapping
    ->
run_backtest(...)
    ->
portfolio construction
    ->
artifacts
```

Alpha utilities do not write artifacts themselves in this milestone. They
prepare deterministic model state and prediction outputs that can feed the
existing research and portfolio layers.

## Alpha Model Interface

`src/research/alpha/base.py` defines `BaseAlphaModel`.

The contract is intentionally strict:

* inputs must be pandas DataFrames sorted by `(symbol, ts_utc)`
* `symbol` and `ts_utc` are required structural columns
* `fit(df)` must not mutate the caller's frame
* `predict(df)` must not mutate the caller's frame
* `predict(df)` must return a numeric pandas Series aligned exactly to
  `df.index`
* repeated calls to `predict(df)` on the same input must return identical
  values

The built-in `DummyAlphaModel` is a zero-signal placeholder for scaffolding and
tests.

## Registry

`src/research/alpha/registry.py` provides:

* `register_alpha_model(name, model_cls)`
* `get_alpha_model(name)`

The registry stores model classes under deterministic string names and
instantiates a fresh model when requested.

Typical pattern:

```python
from src.research.alpha import BaseAlphaModel, register_alpha_model


class MyAlphaModel(BaseAlphaModel):
    name = "my_alpha_model"

    def _fit(self, df):
        ...

    def _predict(self, df):
        ...


register_alpha_model(MyAlphaModel.name, MyAlphaModel)
```

## Training Workflow

`train_alpha_model(...)` fits one registered model on a half-open training
window `[train_start, train_end)`.

Inputs:

* canonical research DataFrame with `symbol` and `ts_utc`
* `model_name`
* `target_column`
* optional `feature_columns`
* optional `train_start`
* optional `train_end`

Current behavior:

* validates the full frame before slicing
* requires a non-empty target column with at least one non-null value
* derives features automatically from `feature_*` columns when
  `feature_columns` is omitted
* sorts the selected training frame by `(symbol, ts_utc)`
* returns `TrainedAlphaModel` metadata including the selected features and the
  effective train window

Window semantics are always half-open:

```text
[train_start, train_end)
```

## Prediction Workflow

`predict_alpha_model(...)` applies one `TrainedAlphaModel` on a half-open
prediction window `[predict_start, predict_end)`.

Inputs:

* `trained_model`
* canonical prediction frame
* optional `predict_start`
* optional `predict_end`

Current behavior:

* validates that the trained feature columns still exist
* keeps `symbol` and `ts_utc`
* preserves optional structural columns such as `timeframe`
* returns `AlphaPredictionResult.predictions` with:
  * `symbol`
  * `ts_utc`
  * optional `timeframe`
  * `prediction_score`

Predictions must be finite numeric values with exact index alignment.

## Time-Aware Split Utilities

`src/research/alpha/splits.py` provides deterministic train/predict split
helpers:

* `make_alpha_fixed_split(...)`
* `generate_alpha_rolling_splits(...)`

Supported modes:

* `fixed`
* `rolling`

Each split is represented by `AlphaTimeSplit`, which stores:

* `train_start`
* `train_end`
* `predict_start`
* `predict_end`
* `split_id`
* `metadata`

Important rules:

* all timestamps are normalized to UTC
* train and predict windows are half-open
* train and predict windows must not overlap
* rolling windows advance deterministically by `step`

## Cross-Sectional Utilities

`src/research/alpha/cross_section.py` helps inspect a same-timestamp universe
slice after prediction.

Available helpers:

* `list_cross_section_timestamps(df)`
* `get_cross_section(df, ts_utc, columns=None)`
* `iter_cross_sections(df, columns=None)`

These helpers require:

* non-empty input
* required `symbol` and `ts_utc` columns
* sorted `(symbol, ts_utc)` rows
* no duplicate `(symbol, ts_utc)` keys

Use them when you want to inspect relative scores, ranks, or mapped exposures
for one timestamp across many symbols.

## Relationship To Backtesting

Alpha predictions are not backtested automatically. The current workflow is:

1. train a registered alpha model
2. predict on an out-of-sample window
3. map `prediction_score` into a `signal`
4. run `run_backtest(...)`

The mapping step is intentionally explicit so research code decides how raw
scores become exposures.

Examples:

* sign mapping: `signal = np.sign(prediction_score)`
* continuous exposure mapping: `signal = prediction_score`
* rank or z-score mapping in downstream research code

`run_backtest(...)` accepts finite numeric `signal` values, so alpha workflows
can preserve continuous exposure magnitude when that is the intended behavior.

## Example Workflow

The repository ships an end-to-end example:

* script:
  [examples/milestone_11_5_alpha_portfolio_workflow.py](examples/milestone_11_5_alpha_portfolio_workflow.py)
* companion guide:
  [examples/milestone_11_5_alpha_portfolio_workflow.md](examples/milestone_11_5_alpha_portfolio_workflow.md)

Run it with:

```bash
python docs/examples/milestone_11_5_alpha_portfolio_workflow.py
```

It demonstrates:

* alpha model registration
* deterministic training and prediction
* fixed and rolling alpha splits
* cross-section inspection
* continuous-signal backtesting
* portfolio construction with and without volatility targeting

Outputs are written under:

```text
docs/examples/output/milestone_11_5_alpha_portfolio_workflow/
```

## Related Docs

* [../README.md](../README.md)
* [backtest_runner.md](backtest_runner.md)
* [strategy_evaluation_workflow.md](strategy_evaluation_workflow.md)
* [milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md)
* [examples/milestone_11_5_alpha_portfolio_workflow.md](examples/milestone_11_5_alpha_portfolio_workflow.md)
* [alpha_evaluation_workflow.md](alpha_evaluation_workflow.md)
