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

Alpha utilities feed the alpha-evaluation layer, which now persists
deterministic artifacts such as `predictions.parquet`,
`training_summary.json`, `coefficients.json`,
`cross_section_diagnostics.json`, `alpha_metrics.json`, and `manifest.json`.

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
* requires the configured target column to be present and non-empty
* derives features automatically from `feature_*` columns when
  `feature_columns` is omitted
* sorts the selected training frame by `(symbol, ts_utc)`
* returns `TrainedAlphaModel` metadata including the selected features and the
  effective train window

For built-in daily alpha configs, the canonical training targets live directly
in `features_daily`:

* `target_ret_1d`
* `target_ret_5d`

These are realized forward close-to-close returns aligned to the current row,
so `target_ret_5d` on date `t` equals `close[t+5] / close[t] - 1.0`. Missing
target columns now fail with a clear rebuild/reload error so stale daily
feature datasets do not silently drift from the alpha registry.

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

## Explicit Signal Mapping

`src/research/alpha/signals.py` provides one explicit, reusable layer for
turning `prediction_score` into deterministic exposures. Mapping always runs
cross-sectionally by `ts_utc` and never happens implicitly inside
`predict_alpha_model(...)`.

Available policies:

* `rank_long_short`: ranks one timestamp slice from highest score to lowest and
  maps those ordered names linearly onto `[-1, 1]`
* `zscore_continuous`: mean-centers one timestamp slice and divides by the
  population standard deviation; zero-dispersion slices map to `0.0`
* `top_bottom_quantile`: assigns `+1.0` to the top quantile, `-1.0` to the
  bottom quantile, and `0.0` elsewhere
* `long_only_top_quantile`: assigns `+1.0` to the top quantile and `0.0`
  elsewhere

Deterministic tie handling is explicit:

* cross-sections are evaluated by `ts_utc`
* rows remain canonically sorted by `(symbol, ts_utc[, timeframe])`
* ties break on `symbol` ascending after sorting by `prediction_score`

Quantile semantics are also explicit:

* `top_bottom_quantile` requires `quantile` in `(0, 0.5]`
* `long_only_top_quantile` requires `quantile` in `(0, 1.0]`
* quantile bucket sizes use `ceil(n * quantile)`
* long/short quantile selection is capped at `floor(n / 2)` so top and bottom
  buckets never overlap

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

Alpha predictions can now be converted into a deterministic downstream sleeve
using the same `run_backtest(...)` assumptions as the strategy layer. The
practical first-use default in `python -m src.cli.run_alpha --mode full` is a
market-neutral `rank_long_short` mapping.

The workflow is:

1. train a registered alpha model
2. predict on an out-of-sample window
3. map `prediction_score` into a `signal`
4. generate one sleeve return stream via `run_backtest(...)`
5. hand the sleeve into downstream portfolio construction when desired

The mapping step is intentionally explicit so research code decides how raw
scores become exposures.

Examples:

* sign mapping in downstream custom code
* continuous exposure mapping via `zscore_continuous`
* rank-based exposure mapping via `rank_long_short`
* quantile sleeves via `top_bottom_quantile` or `long_only_top_quantile`

`run_backtest(...)` accepts finite numeric `signal` values, so alpha workflows
can preserve continuous exposure magnitude when that is the intended behavior.

`python -m src.cli.run_alpha --mode full` now persists deterministic sleeve
artifacts alongside the evaluation outputs:

* `signals.parquet`
* `sleeve_returns.csv`
* `sleeve_equity_curve.csv`
* `sleeve_metrics.json`
* `alpha_run_scaffold.json`

## Example Workflow

The repository ships one primary built-in alpha workflow example:

* script:
  [examples/real_alpha_workflow.py](examples/real_alpha_workflow.py)
* companion guide:
  [examples/real_alpha_workflow.md](examples/real_alpha_workflow.md)

Run it with:

```bash
python docs/examples/real_alpha_workflow.py
```

It demonstrates:

* config-driven built-in alpha selection from `configs/alphas.yml`
* deterministic training, prediction, and alpha evaluation
* explicit alpha-to-signal mapping
* sleeve generation and persisted sleeve artifacts
* downstream portfolio consumption of an `alpha_sleeve`
* unified review artifact outputs for the alpha run and linked portfolio

Outputs are written under:

```text
docs/examples/output/real_alpha_workflow/
```

The older custom-model walkthrough remains available when you want lower-level
alpha registration details:

* [examples/milestone_11_5_alpha_portfolio_workflow.py](examples/milestone_11_5_alpha_portfolio_workflow.py)
* [examples/milestone_11_5_alpha_portfolio_workflow.md](examples/milestone_11_5_alpha_portfolio_workflow.md)

## Related Docs

* [../README.md](../README.md)
* [backtest_runner.md](backtest_runner.md)
* [strategy_evaluation_workflow.md](strategy_evaluation_workflow.md)
* [milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md)
* [examples/real_alpha_workflow.md](examples/real_alpha_workflow.md)
* [examples/milestone_11_5_alpha_portfolio_workflow.md](examples/milestone_11_5_alpha_portfolio_workflow.md)
* [alpha_evaluation_workflow.md](alpha_evaluation_workflow.md)
