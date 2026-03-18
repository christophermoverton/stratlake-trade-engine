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

This command is intentionally scoped to a single research experiment. It does
not schedule runs, perform live trading, or manage portfolio optimization.

---

## Location

```text
src/cli/run_strategy.py
```

Primary entrypoint:

```python
run_cli(argv: Sequence[str] | None = None) -> StrategyRunResult
```

Module execution:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1
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

---

## Arguments

Required arguments:

* `--strategy` -> strategy name defined in `configs/strategies.yml`

Optional arguments:

* `--start` -> inclusive start date in `YYYY-MM-DD`
* `--end` -> exclusive end date in `YYYY-MM-DD`

Date filters are forwarded directly to `load_features()` and use the same
window semantics as the existing data loaders.

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
metrics.cumulative_return(...)
metrics.sharpe_ratio(...)
metrics.volatility(...)
metrics.max_drawdown(...)
metrics.win_rate(...)
        ->
experiment_tracker.save_experiment(...)
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

---

## Console Output

After a successful run, the CLI prints a concise summary:

```text
strategy: momentum_v1
run_id: 20260318T201530123456Z_momentum_v1
cumulative_return: 0.123456
sharpe_ratio: 1.234567
```

The `run_id` matches the directory name created under `artifacts/strategies/`.

---

## Artifact Outputs

Each CLI run writes a new experiment directory:

```text
artifacts/strategies/<run_id>/
```

Current artifact contents:

* `signals.parquet`
* `equity_curve.parquet`
* `metrics.json`
* `config.json`

These files are produced by `save_experiment()` and make the run reproducible.

---

## Related Modules

See the adjacent research-layer docs for implementation details:

* [docs/backtest_runner.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/backtest_runner.md)
* [docs/strategy_performance_metrics.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/strategy_performance_metrics.md)
* [docs/experiment_artifact_logging.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/experiment_artifact_logging.md)
* [configs/strategies.yml](/C:/Users/christophermoverton/stratlake-trade-engine/configs/strategies.yml)
