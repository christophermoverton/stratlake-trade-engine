# Backtest Runner

## Overview

The backtest runner is the minimal execution layer for the research pipeline.
It takes a feature dataset that already contains standardized strategy signals
and converts those signals into a deterministic strategy return series.

Current implementation:

* reads a single-asset return series from the input dataset
* applies the previous period's signal to the realized return
* interprets finite numeric `signal` values as literal exposures
* compounds the resulting strategy returns into an equity curve

This module is intentionally small and deterministic. It does not yet model
portfolio weights or multi-asset interactions, but it now supports the shared
execution configuration used by the strategy workflow.

---

## Location

```text
src/research/backtest_runner.py
```

Primary entrypoint:

```python
run_backtest(df: pandas.DataFrame) -> pandas.DataFrame
```

---

## Input Contract

`run_backtest()` expects a pandas DataFrame with:

* a `signal` column produced by the research signal engine
* a supported one-period asset return column

`signal` may be discrete or continuous. Any finite numeric value is accepted
and interpreted literally as exposure after lagged execution for direct/manual
backtest usage. Canonical strategy and alpha workflows now pass managed typed
signals and reject unmanaged legacy frames instead of inferring contracts.

Supported return column names are currently:

* `ret_1`
* `ret_1m`
* `ret_1d`
* `feature_ret_1m`
* `feature_ret_1d`
* `asset_return`
* `return`
* `returns`

If `signal` is missing, or none of the supported return columns are present,
the function raises a `ValueError`.

---

## Calculation Rules

The backtest runner avoids look-ahead bias by applying the prior row's signal to
the current row's realized return:

```python
strategy_return = signal.shift(1).fillna(0.0) * asset_return
```

This means:

* the first row always has zero strategy exposure
* a signal generated at time `t` affects returns at time `t+1`
* `signal=0.5` contributes half the underlying return at execution time
* the runner does not clip or normalize exposure values implicitly
* the output is deterministic for a fixed input DataFrame

The equity curve is then computed with cumulative compounding:

```python
equity_curve = (1.0 + strategy_return).cumprod()
```

---

## Output Columns

The returned DataFrame is a copy of the input with deterministic execution and
traceability columns, including:

* `executed_signal`
* `position`
* `delta_position`
* `turnover`
* `gross_strategy_return`
* `transaction_cost`
* `slippage_cost`
* `execution_friction`
* `net_strategy_return`
* `strategy_return`
* `equity_curve`

All original columns and the input index are preserved.

---

## Example

```python
import pandas as pd

from src.research.backtest_runner import run_backtest

df = pd.DataFrame(
    {
        "signal": [1.0, 0.5, -1.0, 0.0],
        "feature_ret_1d": [0.01, -0.02, 0.03, -0.01],
    }
)

result = run_backtest(df)

print(result[["signal", "feature_ret_1d", "strategy_return", "equity_curve"]])
```

Expected strategy returns:

```text
[0.00, -0.02, 0.015, 0.01]
```

Expected equity curve:

```text
[1.0, 0.98, 0.9947, 1.004647]
```

---

## Relationship To The Research Pipeline

The current research flow is:

```text
feature dataset
        ->
alpha prediction or strategy signal generation
        ->
mapped signal exposures
        ->
backtest_runner.run_backtest(...)
        ->
strategy return series and equity curve
```

This provides a simple and testable foundation for later additions such as
performance metrics, portfolio construction, and execution assumptions.

When consuming persisted canonical signal artifacts, reload
`signal_semantics.json` alongside `signals.parquet` so the typed contract
remains explicit.
