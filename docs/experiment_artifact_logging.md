# Experiment Artifact Logging

## Overview

The experiment artifact logging utility persists the outputs of a research
strategy run so results can be reproduced, inspected, and compared later.

Current implementation:

* creates a unique run directory under `artifacts/strategies/`
* writes signal-engine outputs to parquet
* writes backtest equity-curve outputs to parquet
* writes metrics and strategy configuration to JSON
* writes split-level walk-forward artifacts when evaluation mode is used

This module is intentionally file-based and lightweight. It does not add
database tracking, dashboards, orchestration, or experiment metadata services.

---

## Location

```text
src/research/experiment_tracker.py
```

Primary entrypoint:

```python
save_experiment(
    strategy_name: str,
    results_df: pandas.DataFrame,
    metrics: dict,
    config: dict,
) -> pathlib.Path
```

Walk-forward entrypoint:

```python
save_walk_forward_experiment(
    strategy_name: str,
    split_results: list[dict],
    aggregate_summary: dict,
    config: dict,
) -> pathlib.Path
```

---

## Artifact Layout

Each call to `save_experiment()` creates a new directory using a timestamp-based
run identifier combined with the strategy name:

```text
artifacts/strategies/<run_id>/
```

Example contents:

```text
artifacts/strategies/20260318T153045123456Z_mean_reversion/
  signals.parquet
  equity_curve.parquet
  metrics.json
  config.json
```

Walk-forward runs add split-aware outputs inside the same root:

```text
artifacts/strategies/<run_id>/
  metrics.json
  config.json
  metrics_by_split.csv
  splits/<split_id>/signals.parquet
  splits/<split_id>/equity_curve.parquet
  splits/<split_id>/metrics.json
  splits/<split_id>/split.json
```

The timestamp component is generated in UTC and keeps runs unique while
remaining easy to sort chronologically.

---

## Input Contract

`save_experiment()` expects:

* `strategy_name`: a human-readable strategy identifier used in the run path
* `results_df`: a pandas DataFrame containing signal outputs and backtest results
* `metrics`: a JSON-serializable dictionary of computed performance metrics
* `config`: a JSON-serializable dictionary of strategy configuration values

The `results_df` input must include:

* `strategy_return`
* `equity_curve`

If either backtest column is missing, the function raises a `ValueError`.

---

## Saved Artifacts

### `signals.parquet`

Contains the signal-engine portion of the experiment DataFrame. In the current
implementation this includes all columns from `results_df` except
`strategy_return` and `equity_curve`.

This keeps the signal artifact aligned with the dataset used for strategy
evaluation while excluding the derived backtest-only outputs.

### `equity_curve.parquet`

Contains the backtest outputs needed to inspect realized strategy performance.

Current columns:

* `signal` when present in `results_df`
* `strategy_return`
* `equity_curve`

### `metrics.json`

Contains the summary performance metrics supplied to `save_experiment()`, such
as total return, annualized return, Sharpe ratio, drawdown, hit rate, profit
factor, turnover, or exposure.

### `config.json`

Contains the strategy configuration used for the experiment run, making the
artifact directory self-describing and reproducible.

### `metrics_by_split.csv`

Present for walk-forward runs. Contains one row per executed split with:

* split identifiers and train/test boundaries
* split, train, and test row counts
* the same metric columns used elsewhere in the research layer

### `splits/<split_id>/...`

Present for walk-forward runs. Each split directory stores:

* `signals.parquet` for test-window signal outputs with split metadata columns
* `equity_curve.parquet` for test-window backtest outputs
* `metrics.json` for split-level summary metrics
* `split.json` for the split definition itself

---

## Example

```python
from src.research.experiment_tracker import save_experiment
from src.research.metrics import compute_performance_metrics

metrics = compute_performance_metrics(backtest_df)

config = {
    "lookback": 20,
    "threshold": 0.75,
}

artifact_dir = save_experiment(
    strategy_name="mean_reversion",
    results_df=backtest_df,
    metrics=metrics,
    config=config,
)
```

Returned value:

```text
pathlib.Path("artifacts/strategies/<run_id>")
```

---

## Relationship To The Research Pipeline

The current research flow is:

```text
feature dataset
        ->
strategy.generate_signals(...)
        ->
signal_engine.generate_signals(...)
        ->
backtest_runner.run_backtest(...)
        ->
metrics.compute_performance_metrics(...)
        ->
experiment_tracker.save_experiment(...)
        ->
parquet + JSON experiment artifacts
```

This gives the research layer a reproducible file-based record of each strategy
run without introducing additional infrastructure.
