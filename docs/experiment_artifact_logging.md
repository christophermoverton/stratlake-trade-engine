# Experiment Artifact Logging

## Overview

The experiment artifact logging utility persists the outputs of a research
strategy run so results can be reproduced, inspected, and compared later.

Current implementation:

* creates a deterministic run directory under `artifacts/strategies/`
* upserts one registry row per deterministic run to `artifacts/strategies/registry.jsonl`
* writes signal-engine outputs to parquet
* writes standardized equity-curve outputs to CSV and preserves legacy parquet compatibility
* writes metrics and strategy configuration to JSON
* writes a manifest for fast inspection and reporting
* writes split-level walk-forward artifacts when evaluation mode is used

This module is intentionally file-based and lightweight. It does not add
database tracking, dashboards, orchestration, or experiment metadata services.

---

## Location

```text
src/research/experiment_tracker.py
src/research/registry.py
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

Each call to `save_experiment()` creates a deterministic directory using the
strategy name, evaluation mode, normalized config, and normalized results:

```text
artifacts/strategies/<run_id>/
```

Example contents:

```text
artifacts/strategies/mean_reversion_single_<digest>/
  config.json
  metrics.json
  metrics_readiness.json
  equity_curve.csv
  signals.parquet
  equity_curve.parquet
  trades.parquet
  manifest.json
```

Walk-forward runs add split-aware outputs inside the same root:

```text
artifacts/strategies/<run_id>/
  config.json
  metrics.json
  metrics_readiness.json
  equity_curve.csv
  signals.parquet
  equity_curve.parquet
  trades.parquet
  manifest.json
  metrics_by_split.csv
  splits/<split_id>/signals.parquet
  splits/<split_id>/equity_curve.csv
  splits/<split_id>/equity_curve.parquet
  splits/<split_id>/metrics.json
  splits/<split_id>/metrics_readiness.json
  splits/<split_id>/split.json
```

If the same experiment is rerun on unchanged inputs, the same `run_id` is
reused and the directory is rewritten in place with stable artifact contents.

In addition to the per-run directory, completed runs write one JSON object line
per `run_id` to:

```text
artifacts/strategies/registry.jsonl
```

The registry is intended for lightweight querying without scanning each artifact
directory. Repeated deterministic reruns update the existing row for the same
`run_id` instead of adding a timestamp-only duplicate.

---

## Registry Schema

Each registry entry is self-contained and records:

* `run_id`
* `timestamp` in ISO8601 UTC form
* `strategy_name`
* `dataset`
* `strategy_params`
* `evaluation_mode` (`single` or `walk_forward`)
* `evaluation_config`
* `evaluation_config_path`
* `data_range` with `start` and `end`
* `timeframe`
* `metrics_summary`
* `artifact_path`
* `split_count`

For walk-forward runs, `metrics_summary` contains only aggregate metrics. Raw
per-split outputs remain in the run directory and are not embedded in the
registry row.

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

Contains the signal-engine portion of the experiment DataFrame. It preserves
all non-backtest columns from `results_df` and standardizes the leading
inspection columns to include:

* `ts_utc`
* `date` when present
* `symbol` when present
* `signal`
* `position`

This keeps the signal artifact aligned with the dataset used for strategy
evaluation while excluding the derived backtest-only outputs.

### `equity_curve.csv`

Contains the standardized backtest timeline used for reporting and debugging.

Current columns:

* `ts_utc`
* `symbol` when present
* `equity`
* `strategy_return`
* `signal` when present
* `position`

Rows are sorted by time and use the same schema for single runs and split-level
walk-forward outputs.

### `equity_curve.parquet`

Legacy compatibility artifact that preserves the older parquet export.

Current columns:

* `signal` when present in `results_df`
* `strategy_return`
* `equity_curve`

### `metrics.json`

Contains the summary performance metrics supplied to `save_experiment()`, such
as total return, annualized return, Sharpe ratio, drawdown, hit rate, profit
factor, turnover, or exposure. Return-stream payloads also include SciPy-backed
Student-t inference diagnostics for mean period return: `t_stat`, two-sided
`p_value`, `conf_int_lower`, and `conf_int_upper`, plus serial-dependence
diagnostics: `autocorr_lag1` for lag-1 autocorrelation of finite period returns
and `effective_n` for a conservative AR(1)-style effective sample size
estimate. Positive autocorrelation reduces `effective_n`; negative
autocorrelation is capped at the observed sample size. These diagnostics inform
interpretation of the Student-t fields, but autocorrelation-adjusted inference
is handled by a later readiness milestone.

Return-stream payloads also include split-period consistency diagnostics:
`split_mean_diff`, first-half mean return minus second-half mean return, and
`split_mean_diff_p`, a two-sided SciPy Welch t-test p-value comparing the two
halves. Finite returns are split deterministically by observation count after
filtering missing and non-finite values. Undefined split p-value tests return
`None`; the signed mean difference remains available when both halves satisfy
the minimum sample convention and the value is finite. These diagnostics
highlight simple sub-period concentration but do not replace walk-forward
evaluation.

Return-stream payloads also include rolling Sharpe stability diagnostics.
`rolling_sharpe_mean` is the mean of valid sequential window Sharpe ratios,
`rolling_sharpe_sd` is their sample standard deviation, and
`sharpe_stability_ratio` is the mean divided by the standard deviation when the
denominator is defined and not near zero. Windows are order-preserving,
non-overlapping, and full-sized by default: `252` finite observations when
available, otherwise `max(4, n // 3)` for streams with at least `12` finite
observations. Undefined or degenerate diagnostics return `None`. These fields
are lightweight stability checks and do not replace walk-forward evaluation.

Strategy trade payloads also include `hit_rate_p_value`, a one-sided SciPy
binomial-test p-value for closed-trade hit rate with null win probability
`0.5` and `alternative="greater"`. Zero-return closed trades are counted as
valid non-wins. This is a trade-level diagnostic and is not a period
`win_rate` significance test.

### `metrics_readiness.json`

Contains an additive research-readiness summary derived from the adjacent
`metrics.json` file. It records:

* `schema_version`
* overall `status`
* `run_id`
* `source_metrics_artifact`
* grouped diagnostics for return inference, hit-rate significance,
  serial-dependence, split-period consistency, and rolling Sharpe stability
* advisory checks and summary counts

Readiness statuses use `PASS`, `WARN`, and `FAIL`. Overall status is `FAIL` if
any check fails, otherwise `WARN` if any check warns, otherwise `PASS`. The
default minimum effective sample size threshold is `30`. Missing diagnostics
and non-finite values are written as JSON `null`, and the artifact is safe to
serialize with `allow_nan=False`.

The readiness manifest supports campaign summaries, notebooks, and research
governance review. It does not replace `metrics.json`, promotion gates, or full
validation.

Both `metrics.json` and `metrics_readiness.json` are written with sorted keys,
stable indentation, and JSON-safe values. Undefined diagnostics are represented
as `null`; artifacts must not rely on `NaN`, `Infinity`, or platform-specific
path strings.

### `promotion_gates.json`

When `promotion_gates` are configured for a run, the shared promotion evaluator
writes `promotion_gates.json` beside the other run artifacts. This remains the
canonical promotion-policy artifact.

Legacy configs without per-gate `severity` preserve the original behavior:
`evaluation_status` is `pass` or `fail`, and `promotion_status` is selected
from `status_on_pass` or `status_on_fail`.

Severity-aware configs may set `severity: warn`, `review`, `reject`, or
`block` on individual gates. Failed or non-skipped missing severity gates map
deterministically to `promotion_status` values `warn`, `needs_review`,
`rejected`, or `blocked`, with `block > reject > review > warn` resolving mixed
failures. Existing human-readable gate reasons remain present, and M31 adds
stable `reason_codes` for machine-readable review.

M30 statistical diagnostics do not require a separate readiness policy file for
promotion. They can be referenced directly from `metrics.json` with
`source: metrics`, for example `effective_n`, `p_value`, `hit_rate_p_value`,
`split_mean_diff_p`, and `sharpe_stability_ratio`.

Downstream registry, review, campaign, and milestone flows should consume the
`promotion_gate_summary` generated from this artifact. They should not re-derive
severity outside `src/research/promotion.py`. Review metadata maps expanded
promotion outcomes as follows: `eligible -> candidate`, `warn -> needs_review`,
`needs_review -> needs_review`, `rejected -> rejected`, and
`blocked -> rejected`.

### `config.json`

Contains the strategy configuration used for the experiment run, making the
artifact directory self-describing and reproducible.

### `manifest.json`

Contains a compact run summary with:

* `run_id`
* `timestamp`
* `strategy_name`
* `evaluation_mode`
* `evaluation_config_path`
* `artifact_files`
* `split_count`
* `primary_metric`
* `metric_summary`

Use this as the first file to inspect when loading a run for debugging or
reporting.

### `registry.jsonl`

Contains one JSON object per deterministic strategy run. Each line is written
only after the run artifacts are written successfully.

### `metrics_by_split.csv`

Present for walk-forward runs. Contains one row per executed split with:

* split identifiers and train/test boundaries
* split, train, and test row counts
* the same metric columns used elsewhere in the research layer

The first columns are deterministic across runs:

* `split_id`
* `mode`
* `train_start`
* `train_end`
* `test_start`
* `test_end`
* `split_rows`
* `train_rows`
* `test_rows`

### `splits/<split_id>/...`

Present for walk-forward runs. Each split directory stores:

* `signals.parquet` for test-window signal outputs with split metadata columns
* `equity_curve.csv` for standardized test-window backtest outputs
* `equity_curve.parquet` for test-window backtest outputs
* `metrics.json` for split-level summary metrics
* `metrics_readiness.json` for split-level advisory readiness diagnostics
* `split.json` for the split definition itself

### Inspecting A Run

```python
from pathlib import Path

from src.research.reporting import load_run_artifacts, summarize_run

run_dir = Path("artifacts/strategies/<run_id>")
summary = summarize_run(run_dir)
artifacts = load_run_artifacts(run_dir)
```

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

## Lightweight Querying

`src/research/registry.py` includes small helpers for future comparison work:

```python
from pathlib import Path

from src.research.registry import (
    filter_by_metric_threshold,
    filter_by_strategy_name,
    load_registry,
)

entries = load_registry(Path("artifacts/strategies/registry.jsonl"))
momentum_runs = filter_by_strategy_name(entries, "momentum_v1")
strong_runs = filter_by_metric_threshold(entries, "sharpe_ratio", min_value=1.0)
```

These utilities intentionally stop short of leaderboard or ranking logic.
