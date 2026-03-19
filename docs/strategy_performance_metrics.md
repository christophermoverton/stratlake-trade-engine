# Strategy Performance Metrics

## Overview

The strategy performance metrics module evaluates a backtest result frame using
deterministic return, risk, trade, and activity statistics.

Current implementation:

* keeps legacy-compatible `cumulative_return`, `volatility`, and `win_rate`
* adds `total_return`, `annualized_return`, `annualized_volatility`, and annualized `sharpe_ratio`
* derives `max_drawdown` from the compounded equity curve
* computes trade-level `hit_rate` and `profit_factor` from closed trades
* computes `turnover` and `exposure_pct` from executed position changes

The module stays serializable and is reused by single-run experiments,
baseline strategies, walk-forward split scoring, and aggregate walk-forward
summaries.

---

## Location

```text
src/research/metrics.py
```

Primary public helpers:

```python
compute_performance_metrics(results_df: pandas.DataFrame) -> dict[str, float | None]
cumulative_return(strategy_return: pandas.Series) -> float
total_return(strategy_return: pandas.Series) -> float
annualized_return(strategy_return: pandas.Series, *, periods_per_year: int = 252) -> float
volatility(strategy_return: pandas.Series) -> float
annualized_volatility(strategy_return: pandas.Series, *, periods_per_year: int = 252) -> float
sharpe_ratio(strategy_return: pandas.Series, *, periods_per_year: int = 252) -> float
max_drawdown(strategy_return: pandas.Series) -> float
win_rate(strategy_return: pandas.Series) -> float
hit_rate(trade_returns: pandas.Series) -> float
profit_factor(trade_returns: pandas.Series) -> float | None
turnover(position: pandas.Series) -> float
exposure_pct(position: pandas.Series) -> float
```

---

## Input Contract

`compute_performance_metrics()` expects a backtest result DataFrame containing:

* `strategy_return`
* usually `signal`
* optionally `timeframe` and/or `ts_utc` for annualization inference

Behavioral rules:

* the input frame is not modified
* missing return values are dropped before return-based calculations
* trade metrics use only closed trades
* empty inputs return deterministic values
* outputs remain JSON-serializable; undefined `profit_factor` is returned as `None`

The functions are designed to operate directly on the output of
`run_backtest()`.

---

## Annualization Assumptions

Annualization is deterministic and timeframe-aware:

* daily strategies: `252` trading periods per year
* minute strategies: `252 * 390 = 98,280` trading periods per year
* unknown timeframes: fall back to the daily assumption

Timeframe inference prefers the `timeframe` column. If it is unavailable, the
module falls back to known return-column names and then to `ts_utc` spacing.

---

## Metric Definitions

### `total_return`

Cumulative compounded return over the evaluation window:

```python
(1.0 + strategy_return).prod() - 1.0
```

`cumulative_return` is kept as a compatibility alias with the same value.

### `annualized_return`

Compounded annualized return using the observed number of return observations:

```python
(1.0 + total_return) ** (periods_per_year / observation_count) - 1.0
```

If the compounded growth path is less than or equal to zero, the function
returns `-1.0`.

### `annualized_volatility`

Sample return volatility scaled by the deterministic timeframe factor:

```python
strategy_return.std() * sqrt(periods_per_year)
```

If fewer than two return observations are present, the value is `0.0`.

### `sharpe_ratio`

Annualized mean excess return divided by annualized volatility, with a
zero risk-free rate:

```python
(strategy_return.mean() * periods_per_year) / annualized_volatility
```

If annualized volatility is zero, the function returns `0.0`.

### `max_drawdown`

Largest peak-to-trough decline from the compounded equity curve:

```python
equity_curve = (1.0 + strategy_return).cumprod()
drawdown = 1.0 - (equity_curve / equity_curve.cummax())
```

The metric is reported as a positive decimal fraction.

### `hit_rate`

Share of profitable closed trades:

```python
(closed_trade_returns > 0.0).mean()
```

A trade is one contiguous non-zero executed-position segment. Open terminal
trades are excluded.

### `profit_factor`

Gross profits divided by gross losses across closed trades:

```python
sum(positive_trade_returns) / abs(sum(negative_trade_returns))
```

Edge-case behavior:

* no closed trades -> `0.0`
* no losing closed trades but at least one winner -> `None`
* only losing closed trades -> `0.0`

### `turnover`

Average absolute executed-position change per observation:

```python
position_change = position.diff().fillna(position)
turnover = position_change.abs().mean()
```

This counts entries, exits, and direct flips. For example, a move from `1` to
`-1` contributes `2.0`.

### `exposure_pct`

Percentage of observations with non-zero executed position:

```python
(position != 0.0).mean() * 100.0
```

### Legacy Compatibility Metrics

The summary also retains:

* `volatility`: non-annualized sample return standard deviation
* `win_rate`: share of periods with strictly positive `strategy_return`

---

## Trade Extraction Rules

Trade-level metrics use the executed position, not the same-row signal:

```python
position = signal.shift(1).fillna(0.0)
```

Closed trades are built from contiguous non-zero position segments. Trade return
is compounded over that segment:

```python
(1.0 + trade_period_returns).prod() - 1.0
```

This matches the backtest’s lagged execution rule and keeps trade metrics
consistent across single-run and walk-forward evaluation.

---

## Example

```python
from src.research.backtest_runner import run_backtest
from src.research.metrics import compute_performance_metrics
from src.research.signal_engine import generate_signals

signals_df = generate_signals(features_df, strategy)
backtest_df = run_backtest(signals_df)

summary = compute_performance_metrics(backtest_df)
```

Typical keys in `summary`:

```python
{
    "cumulative_return": 0.12,
    "total_return": 0.12,
    "volatility": 0.018,
    "annualized_return": 0.31,
    "annualized_volatility": 0.29,
    "sharpe_ratio": 1.08,
    "max_drawdown": 0.07,
    "win_rate": 0.54,
    "hit_rate": 0.58,
    "profit_factor": 1.34,
    "turnover": 0.21,
    "exposure_pct": 63.5,
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
compute_performance_metrics(...)
```

This provides one deterministic metrics layer for standard experiments,
baselines, and walk-forward scoring without introducing a separate evaluation
path.
