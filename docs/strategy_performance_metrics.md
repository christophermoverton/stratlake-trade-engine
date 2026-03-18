# Strategy Performance Metrics

## Overview

The strategy performance metrics module evaluates a backtest's
`strategy_return` series using a small set of standardized statistics.

Current implementation:

* computes total compounded return
* measures return volatility
* computes a simple Sharpe ratio with a zero risk-free rate
* derives maximum drawdown from compounded returns
* computes win rate as the share of positive-return periods

This module is intentionally narrow in scope. It does not log artifacts,
annualize metrics, model benchmark-relative performance, or apply portfolio
weighting assumptions.

---

## Location

```text
src/research/metrics.py
```

Primary functions:

```python
cumulative_return(strategy_return: pandas.Series) -> float
volatility(strategy_return: pandas.Series) -> float
sharpe_ratio(strategy_return: pandas.Series) -> float
max_drawdown(strategy_return: pandas.Series) -> float
win_rate(strategy_return: pandas.Series) -> float
```

---

## Input Contract

Each metric function accepts a pandas Series representing
`strategy_return`.

Behavioral rules:

* the input Series is not modified
* missing values are dropped before calculation
* calculations are deterministic for a fixed input series
* empty inputs return `0.0`

The functions are designed to operate directly on the return series produced by
`run_backtest()`.

---

## Metric Definitions

### Cumulative Return

Total compounded return across the full series:

```python
(1.0 + strategy_return).prod() - 1.0
```

### Volatility

Sample standard deviation of the return series:

```python
strategy_return.std()
```

This implementation does not annualize volatility.

### Sharpe Ratio

Mean return divided by sample return volatility:

```python
strategy_return.mean() / strategy_return.std()
```

Assumptions:

* risk-free rate is `0.0`
* the result is not annualized
* if volatility is zero, the function returns `0.0`

### Max Drawdown

Maximum peak-to-trough decline derived from the compounded equity curve:

```python
equity_curve = (1.0 + strategy_return).cumprod()
drawdown = 1.0 - (equity_curve / equity_curve.cummax())
```

The function returns the largest drawdown as a positive decimal fraction.

### Win Rate

Share of periods with strictly positive returns:

```python
(strategy_return > 0.0).mean()
```

Zero-return periods are not counted as wins.

---

## Example

```python
from src.research.backtest_runner import run_backtest
from src.research.metrics import (
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    volatility,
    win_rate,
)
from src.research.signal_engine import generate_signals

signals_df = generate_signals(features_df, strategy)
backtest_df = run_backtest(signals_df)

strategy_return = backtest_df["strategy_return"]

summary = {
    "cumulative_return": cumulative_return(strategy_return),
    "volatility": volatility(strategy_return),
    "sharpe_ratio": sharpe_ratio(strategy_return),
    "max_drawdown": max_drawdown(strategy_return),
    "win_rate": win_rate(strategy_return),
}
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
strategy_return + equity_curve
        ->
metrics.{cumulative_return, volatility, sharpe_ratio, max_drawdown, win_rate}
```

This provides a consistent evaluation layer for comparing strategy experiments
before later additions such as experiment tracking, allocation logic, and
transaction cost modeling.
