# Backtest Runner

## Overview

The backtest runner is the minimal execution layer for the research pipeline.
It takes a feature dataset that already contains standardized strategy signals
and converts those signals into a deterministic strategy return series.

Current implementation:

* reads a single-asset return series from the input dataset
* applies the previous period's signal to the realized return
* compounds the resulting strategy returns into an equity curve

This module is intentionally small and deterministic. It does not yet model
portfolio weights, transaction costs, slippage, or multi-asset interactions.

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

Supported return column names are currently:

* `ret_1`
* `ret_1d`
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
* the output is deterministic for a fixed input DataFrame

The equity curve is then computed with cumulative compounding:

```python
equity_curve = (1.0 + strategy_return).cumprod()
```

---

## Output Columns

The returned DataFrame is a copy of the input with two added columns:

* `strategy_return`: realized strategy return after applying the lagged signal
* `equity_curve`: cumulative compounded value of the strategy starting at `1.0`

All original columns and the input index are preserved.

---

## Example

```python
import pandas as pd

from src.research.backtest_runner import run_backtest

df = pd.DataFrame(
    {
        "signal": [1, 1, -1, 0],
        "feature_ret_1d": [0.01, -0.02, 0.03, -0.01],
    }
)

result = run_backtest(df)

print(result[["signal", "feature_ret_1d", "strategy_return", "equity_curve"]])
```

Expected strategy returns:

```text
[0.00, -0.02, 0.03, 0.01]
```

Expected equity curve:

```text
[1.0, 0.98, 1.0094, 1.019494]
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
strategy return series and equity curve
```

This provides a simple and testable foundation for later additions such as
performance metrics, portfolio construction, and execution assumptions.
