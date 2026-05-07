# Strategy Performance Metrics

## Overview

The strategy performance metrics module evaluates a backtest result frame using
deterministic return, risk, trade, and activity statistics.

Current implementation:

* keeps legacy-compatible `cumulative_return`, `volatility`, and `win_rate`
* adds `total_return`, `annualized_return`, `annualized_volatility`, and annualized `sharpe_ratio`
* derives `max_drawdown` from the compounded equity curve
* reports return-stream serial-dependence diagnostics with `autocorr_lag1` and `effective_n`
* reports split-period consistency diagnostics with `split_mean_diff` and `split_mean_diff_p`
* reports rolling Sharpe stability diagnostics with `rolling_sharpe_mean`,
  `rolling_sharpe_sd`, and `sharpe_stability_ratio`
* computes trade-level `hit_rate`, `hit_rate_p_value`, and `profit_factor` from closed trades
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
compute_hit_rate_p_value(trade_returns: pandas.Series, *, null_probability: float = 0.5, alternative: str = "greater") -> float | None
compute_autocorr_lag1(strategy_return: pandas.Series) -> float | None
compute_effective_sample_size(strategy_return: pandas.Series) -> float | None
compute_split_period_diagnostics(strategy_return: pandas.Series) -> dict[str, float | None]
compute_rolling_sharpe_diagnostics(strategy_return: pandas.Series, *, window_size: int | None = None, min_periods: int | None = None, step_size: int | None = None, periods_per_year: int = 252) -> dict[str, float | None]
build_metrics_readiness_manifest(metrics: dict, *, run_id: str | None = None, source_metrics_artifact: str = "metrics.json", thresholds: dict | None = None) -> dict
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
* outputs remain JSON-serializable; undefined inference fields and undefined
  `profit_factor` values are returned as `None`

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

### `autocorr_lag1`

Lag-1 autocorrelation of finite period returns. Missing and non-finite returns
are excluded. The value is `None` when fewer than two finite observations are
available or when either lagged vector has near-zero variance.

### `effective_n`

Conservative AR(1)-style effective sample size estimate:

```python
n * (1.0 - autocorr_lag1) / (1.0 + autocorr_lag1)
```

Positive autocorrelation reduces effective sample size. Negative
autocorrelation is capped at the observed finite sample size for conservative
reporting. These diagnostics help interpret `t_stat`, `p_value`, and confidence
intervals, but they do not replace those fields with autocorrelation-adjusted
inference.

### `split_mean_diff` and `split_mean_diff_p`

Split-period consistency diagnostics compare finite period returns from the
first half of the stream with finite returns from the second half. Returns are
filtered for missing and non-finite values first, order is preserved, and the
split is by observation count: first half is the first `n // 2` observations and
second half is the remainder.

`split_mean_diff` is first-half mean return minus second-half mean return.
`split_mean_diff_p` is a two-sided SciPy Welch t-test p-value using
`scipy.stats.ttest_ind(..., equal_var=False, nan_policy="omit")`.

Both fields are `None` unless each half has at least two finite observations.
If SciPy returns a non-finite test result, such as in a degenerate variance
case, `split_mean_diff_p` is `None` while the signed mean difference remains
available when it is finite. These diagnostics indicate simple sub-period
consistency; they do not replace full walk-forward evaluation.

### `rolling_sharpe_mean`, `rolling_sharpe_sd`, and `sharpe_stability_ratio`

Rolling Sharpe stability diagnostics summarize valid window-level Sharpe
ratios across finite period returns. Returns are filtered for missing and
non-finite values first, order is preserved, and windows are deterministic.

Default windowing is sequential and non-overlapping:

* `252` observations when at least `252` finite returns are available
* otherwise `max(4, n // 3)` when at least `12` finite returns are available
* otherwise the diagnostics are `None`

By default, only full windows are used and `step_size` equals `window_size`.
Each window Sharpe is computed with `sharpe_ratio()` so annualization matches
the headline Sharpe ratio. `rolling_sharpe_mean` is the mean of valid window
Sharpes, and `rolling_sharpe_sd` is their sample standard deviation with
`ddof=1`. At least two valid window Sharpes are required; otherwise all rolling
Sharpe stability diagnostics are `None`.

`sharpe_stability_ratio` is `rolling_sharpe_mean / rolling_sharpe_sd` and is
`None` when the sample standard deviation is missing, non-finite, or near zero.
These fields are lightweight stability diagnostics and do not replace full
walk-forward evaluation.

### `hit_rate`

Share of profitable closed trades:

```python
(closed_trade_returns > 0.0).mean()
```

A trade is one contiguous non-zero executed-position segment. Open terminal
trades are excluded.

### `hit_rate_p_value`

One-sided SciPy binomial-test p-value for the trade-level hit rate:

```python
scipy.stats.binomtest(wins, total, p=0.5, alternative="greater").pvalue
```

The null hypothesis is a true win probability of `0.5`; the default
alternative is that the true win probability is greater than random chance.
The test uses finite closed trade returns only. Strictly positive trade
returns count as wins, while zero-return trades count as valid closed trades
but not wins. No closed trades, or only non-finite trade returns, produce
`None`. This diagnostic is trade-level and should not be interpreted as a
period `win_rate` test.

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

## Readiness Manifest

`build_metrics_readiness_manifest()` derives an additive
`metrics_readiness.json` payload from an existing metrics dictionary. It does
not replace or redefine `metrics.json`.

The manifest groups the existing diagnostics into:

* `return_inference`
* `hit_rate`
* `serial_dependence`
* `split_period`
* `rolling_stability`

It also records advisory readiness checks:

* `minimum_observations`: uses `effective_n` when present, otherwise a known
  observed period count such as `row_count`; the default advisory threshold is
  `30`
* `return_p_value_available`
* `hit_rate_p_value_available`
* `serial_dependence_available`
* `split_period_available`
* `rolling_stability_available`

Overall status is `FAIL` when any check fails, otherwise `WARN` when any check
warns, otherwise `PASS`. Missing or undefined diagnostics are represented as
`None`, and non-finite inputs are converted to `None` so the manifest can be
serialized with `allow_nan=False`.

These readiness checks are governance and review aids. They are not hard
promotion gates and do not replace full validation, walk-forward review, or
research judgment.

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
    "autocorr_lag1": 0.18,
    "effective_n": 69.49,
    "split_mean_diff": 0.0004,
    "split_mean_diff_p": 0.73,
    "rolling_sharpe_mean": 0.92,
    "rolling_sharpe_sd": 0.18,
    "sharpe_stability_ratio": 5.11,
    "max_drawdown": 0.07,
    "win_rate": 0.54,
    "hit_rate": 0.58,
    "hit_rate_p_value": 0.12,
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
