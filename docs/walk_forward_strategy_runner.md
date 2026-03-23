# Walk-Forward Strategy Runner

## Overview

The walk-forward strategy runner extends the existing research CLI so one
strategy can be executed across deterministic evaluation splits generated from
`configs/evaluation.yml`.

This mode reuses the same strategy, signal, backtest, metrics, and artifact
pipeline as the single-run flow. The main difference is that execution is
repeated per split and then summarized into one aggregate walk-forward result.

Implementation lives in:

```text
src/cli/run_strategy.py
src/research/walk_forward.py
src/research/splits.py
src/research/experiment_tracker.py
configs/evaluation.yml
```

---

## Invocation

Run with the repository default evaluation config:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

Run with an explicit evaluation config path:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --evaluation configs/evaluation.yml
```

Single-run mode remains available:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --start 2025-01-01 --end 2025-04-01
```

`--start` and `--end` are single-run filters and cannot be combined with
`--evaluation`.

---

## Execution Model

Walk-forward execution follows this pattern:

```text
load evaluation config
        ->
generate deterministic splits
        ->
load dataset once across [min(train_start), max(test_end))
        ->
for each split:
  slice [train_start, test_end)
        ->
  generate signals
        ->
  run backtest
        ->
  score metrics on [test_start, test_end)
        ->
write split artifacts
        ->
aggregate split test windows in split order
        ->
write aggregate artifacts
```

The split boundaries continue to use half-open semantics:

* `start` is inclusive
* `end` is exclusive

The scored output for each split is the test window only. Training rows are
loaded and processed so strategies still have the expected historical context.

---

## Per-Split Outputs

Each executed split records:

* `split_id`
* `mode`
* `train_start`
* `train_end`
* `test_start`
* `test_end`
* `split_rows`
* `train_rows`
* `test_rows`
* standard performance metrics

Split-level result rows also carry the same split metadata columns so later
benchmarking or leaderboard tooling can join results back to their source
window deterministically.

---

## Aggregate Summary

The aggregate walk-forward summary is deterministic.

Current method:

* concatenate split test-window result frames in split order
* compute the full strategy metric set on the concatenated test returns
* include split count and row-count context in the summary payload

The aggregate summary is written to `metrics.json` at the run root, while
`metrics_by_split.csv` provides the one-row-per-split table.

---

## Artifact Layout

Walk-forward runs use the same root pattern as normal strategy runs:

```text
artifacts/strategies/<run_id>/
```

Additional walk-forward artifacts:

```text
artifacts/strategies/<run_id>/
  metrics.json
  config.json
  equity_curve.csv
  equity_curve.parquet
  manifest.json
  metrics_by_split.csv
  splits/<split_id>/signals.parquet
  splits/<split_id>/equity_curve.csv
  splits/<split_id>/equity_curve.parquet
  splits/<split_id>/metrics.json
  splits/<split_id>/split.json
```

This keeps the artifact structure compatible with the existing experiment
logging conventions while adding the split-aware outputs needed for later
benchmarking work.

Per-split and aggregate metric payloads use the same names as single-run
experiments, including:

* `total_return` and `cumulative_return`
* `annualized_return`
* `annualized_volatility`
* `sharpe_ratio`
* `max_drawdown`
* `hit_rate`
* `profit_factor`
* `turnover`
* `exposure_pct`

---

## Failure Behavior

Walk-forward execution fails clearly when:

* the evaluation config is invalid
* split generation produces no splits
* the evaluation data window is empty
* a split has no training rows
* a split has no test rows

These failures are intentional so repeated runs remain deterministic and easy
to diagnose.

---

## Related Docs

* [docs/cli_strategy_runner.md](cli_strategy_runner.md)
* [docs/evaluation_split_configuration.md](evaluation_split_configuration.md)
* [docs/experiment_artifact_logging.md](experiment_artifact_logging.md)
