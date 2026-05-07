# CLI Strategy Runner

## Overview

The CLI strategy runner is the command-line entrypoint for executing a full
research experiment from an existing strategy configuration.

Current implementation:

* loads a strategy definition from `configs/strategies.yml`
* validates that the requested strategy exists
* loads the configured feature dataset with optional date filters
* generates standardized `signal` values
* runs the deterministic backtest pipeline
* computes summary performance metrics
* persists experiment artifacts under `artifacts/strategies/`
* prints a short console summary for the run

Canonical strategy runs now persist `signal_semantics.json` alongside the
standard signal/backtest artifacts and record the same contract in
`manifest.json`.

This command is intentionally scoped to a single research experiment. It does
not schedule runs, perform live trading, or manage portfolio optimization.

The repository also includes a comparison CLI for ranking multiple strategies
against the same metric without changing the execution or registry layers.
The dedicated comparison reference lives in
[docs/strategy_comparison_cli.md](strategy_comparison_cli.md).

---

## Location

```text
src/cli/run_strategy.py
```

Primary entrypoint:

```python
run_cli(argv: Sequence[str] | None = None) -> StrategyRunResult | WalkForwardRunResult
```

Module execution:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1
```

Comparison module execution:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1
```

---

## Required Configuration

The runner reads strategy definitions from:

```text
configs/strategies.yml
```

Current strategy entries follow this shape:

```yaml
momentum_v1:
  dataset: features_daily
  parameters:
    lookback_short: 5
    lookback_long: 20
```

Each strategy entry must define:

* `dataset`: feature dataset name passed to `load_features()`
* `parameters`: strategy-specific parameter dictionary

The CLI raises a `ValueError` if the requested strategy is not present in the
registry or if no implementation is registered for that strategy name.

Baseline benchmark entries use the same config shape and runner flow. Current
baseline names include:

* `buy_and_hold_v1`
* `sma_crossover_v1`
* `seeded_random_v1`

---

## Arguments

Required arguments:

* `--strategy` -> strategy name defined in `configs/strategies.yml`

Optional arguments:

* `--start` -> inclusive start date in `YYYY-MM-DD`
* `--end` -> exclusive end date in `YYYY-MM-DD`
* `--evaluation [PATH]` -> enable walk-forward evaluation using
  `configs/evaluation.yml` or a provided evaluation config path
* `--execution-delay` -> override lagged execution delay in bars
* `--transaction-cost-bps` -> override deterministic transaction cost
* `--slippage-bps` -> override deterministic slippage
* `--execution-enabled` -> force execution frictions on for this run
* `--disable-execution-model` -> force transaction costs and slippage off for
  this run
* `--strict` -> enable strict-mode enforcement for sanity and validation

Date filters are forwarded directly to `load_features()` and use the same
window semantics as the existing data loaders. `--start` and `--end` are for
single-run mode and cannot be combined with `--evaluation`.

---

## Research Flow

The CLI orchestrates the existing research modules in this order:

```text
configs/strategies.yml
        ->
load_features(dataset, start, end)
        ->
signal_engine.generate_signals(...)
        ->
backtest_runner.run_backtest(...)
        ->
metrics.compute_performance_metrics(...)
        ->
experiment_tracker.save_experiment(...)
```

When `--evaluation` is provided, the runner instead:

```text
configs/evaluation.yml
        ->
research.splits.generate_evaluation_splits(...)
        ->
load_features(dataset, min(train_start), max(test_end))
        ->
per split:
  slice [train_start, test_end)
        ->
  signal_engine.generate_signals(...)
        ->
  backtest_runner.run_backtest(...)
        ->
  score metrics on [test_start, test_end)
        ->
experiment_tracker.save_walk_forward_experiment(...)
```

The backtest continues to use lagged signal application:

```python
strategy_return = signal.shift(1).fillna(0.0) * asset_return
equity_curve = (1.0 + strategy_return).cumprod()
```

---

## Example Commands

Run a full experiment using all available rows in the configured feature
dataset:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1
```

Run the same strategy over a bounded research window:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --start 2025-01-01 --end 2025-04-01
```

Run the mean-reversion example strategy:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy mean_reversion_v1 --start 2025-01-01 --end 2025-06-01
```

Run the buy-and-hold baseline:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy buy_and_hold_v1
```

Run the seeded random baseline in walk-forward mode:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy seeded_random_v1 --evaluation
```

Run walk-forward evaluation using the default evaluation config:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

Run walk-forward evaluation with an explicit config path:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --evaluation configs/evaluation.yml
```

Compare multiple fresh single-run executions:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1
```

Compare walk-forward runs and rank by total return:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,sma_crossover_v1 --evaluation --metric total_return
```

Reuse prior registry runs instead of executing:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1 --from_registry
```

In registry mode, the comparison CLI selects the latest matching run per
strategy after filtering by evaluation mode. If `--evaluation [PATH]` is
provided, it also filters by the stored `evaluation_config_path`. "Latest"
means descending `timestamp`, then descending `run_id` as a deterministic
tie-breaker.

Leaderboard rows include:

* `strategy_name`
* `run_id`
* `evaluation_mode`
* `selected_metric_name`
* `selected_metric_value`
* core metrics such as `total_return`, `sharpe_ratio`, and `max_drawdown`

Leaderboard outputs are written to
`artifacts/comparisons/<comparison_id>/leaderboard.csv` and
`artifacts/comparisons/<comparison_id>/leaderboard.json` unless `--output_path`
overrides the CSV location or directory.

---

## Console Output

After a successful run, the CLI prints a concise summary:

Example deterministic single-run output now looks like:

```text
strategy: momentum_v1
run_id: momentum_v1_single_cf4a89987721
cumulative_return: -0.999855
sharpe_ratio: 0.032373
```

The `run_id` matches the directory name created under `artifacts/strategies/`.
Rerunning the same experiment on unchanged inputs reuses the same directory name
instead of creating a timestamp-only variant. Walk-forward runs also print
`split_count`.

---

## Artifact Outputs

Each CLI run writes a deterministic experiment directory:

```text
artifacts/strategies/<run_id>/
```

Current artifact contents:

* `config.json`
* `metrics.json`
* `signal_diagnostics.json`
* `qa_summary.json`
* `equity_curve.csv`
* `signals.parquet`
* `equity_curve.parquet`
* `trades.parquet` when closed trades are available
* `manifest.json`

Walk-forward runs keep the same run-directory pattern and add:

* `metrics_by_split.csv`
* `splits/<split_id>/signals.parquet`
* `splits/<split_id>/equity_curve.csv`
* `splits/<split_id>/equity_curve.parquet`
* `splits/<split_id>/metrics.json`
* `splits/<split_id>/split.json`

Single-run artifacts are produced by `save_experiment()`. Walk-forward artifacts
are produced by `save_walk_forward_experiment()`.

Metric payloads include legacy-compatible fields plus expanded evaluation
statistics:

* `cumulative_return` and `total_return`
* `annualized_return`
* `annualized_volatility`
* `sharpe_ratio`
* SciPy-backed period-return inference diagnostics: `t_stat`, `p_value`,
  `conf_int_lower`, and `conf_int_upper`
* Return-stream serial-dependence diagnostics: `autocorr_lag1` and
  `effective_n`
* Split-period consistency diagnostics: `split_mean_diff` and
  `split_mean_diff_p`
* Rolling Sharpe stability diagnostics: `rolling_sharpe_mean`,
  `rolling_sharpe_sd`, and `sharpe_stability_ratio`
* `max_drawdown`
* `win_rate`, trade-level `hit_rate`, and trade-level `hit_rate_p_value`
* `profit_factor`
* `turnover`
* `exposure_pct`

The inference diagnostics use Student-t calculations from SciPy for the mean
period return under an independent-observation assumption. `autocorr_lag1`
reports lag-1 autocorrelation of finite period returns, and `effective_n`
reports a conservative AR(1)-style effective sample size estimate. Positive
autocorrelation reduces `effective_n`; negative autocorrelation is capped at
the observed sample size. These diagnostics inform interpretation of the
Student-t fields, but autocorrelation-adjusted inference is intentionally
reserved for a later readiness milestone.
`split_mean_diff` is first-half mean period return minus second-half mean
period return after filtering missing and non-finite returns and splitting by
finite observation count. `split_mean_diff_p` is a two-sided SciPy Welch
t-test p-value comparing those halves. Undefined split tests return `None`;
the signed mean difference remains available when the halves meet the minimum
sample convention and the value is finite. The diagnostics flag simple
sub-period concentration but do not replace full walk-forward evaluation.
Rolling Sharpe stability diagnostics filter missing and non-finite returns,
preserve order, and use deterministic non-overlapping full windows by default:
`252` observations for streams with at least `252` finite returns, otherwise
`max(4, n // 3)` for streams with at least `12` finite returns. Each window
uses the standard annualized `sharpe_ratio()`. `rolling_sharpe_mean` is the
mean of valid window Sharpes, `rolling_sharpe_sd` is their sample standard
deviation, and `sharpe_stability_ratio` is the mean divided by that standard
deviation when the denominator is defined and not near zero. Undefined cases
return `None`; these diagnostics are lightweight and do not replace
walk-forward evaluation.
`hit_rate_p_value` uses SciPy's one-sided binomial test on closed trade returns
with null win probability `0.5` and `alternative="greater"`. Zero-return
closed trades count as valid non-wins, and the diagnostic is separate from
period-level `win_rate`.

---

## Related Modules

See the adjacent research-layer docs for implementation details:

* [docs/backtest_runner.md](backtest_runner.md)
* [docs/baseline_strategies.md](baseline_strategies.md)
* [docs/strategy_performance_metrics.md](strategy_performance_metrics.md)
* [docs/experiment_artifact_logging.md](experiment_artifact_logging.md)
* [docs/execution_model.md](execution_model.md)
* [docs/strict_mode.md](strict_mode.md)
* [docs/runtime_configuration.md](runtime_configuration.md)
* [docs/strategy_comparison_cli.md](strategy_comparison_cli.md)
* [docs/walk_forward_strategy_runner.md](walk_forward_strategy_runner.md)
* [configs/strategies.yml](../configs/strategies.yml)
