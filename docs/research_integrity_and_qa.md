# Research Integrity And QA

## Overview

StratLake's research workflow is designed to make strategy results reviewable,
repeatable, and hard to misinterpret.

The full single-run flow is:

```text
features -> signals -> backtest -> metrics -> artifacts -> QA -> consistency -> comparison
```

Each stage adds one validation layer:

* `features -> signals`: input contract enforcement plus research-integrity checks
* `signals -> backtest`: lagged execution enforcement so returns use prior-bar signals
* `backtest -> metrics`: deterministic summary statistics plus benchmark-relative diagnostics
* `metrics -> artifacts`: stable, file-based outputs written under one deterministic `run_id`
* `artifacts -> QA`: concise strategy-level status and warning flags
* `QA -> consistency`: cross-artifact validation after files are written
* `consistency -> comparison`: fresh or registry-backed ranking built from validated runs

This document explains how those layers fit together and how to interpret the
artifacts they produce. For execution details, see
[strategy_evaluation_workflow.md](strategy_evaluation_workflow.md) and
[experiment_artifact_logging.md](experiment_artifact_logging.md).

## Validation Layers

### A. Research Integrity Layer

The research integrity layer protects temporal correctness before and during
backtesting.

It validates:

* temporal ordering by `(symbol, ts_utc)`
* no duplicate `(symbol, ts_utc)` rows
* monotonic time within each symbol
* exact signal-to-input index alignment
* signal values restricted to `-1`, `0`, and `1`
* no same-bar execution in the backtest

Backtests enforce lagged execution explicitly:

```text
position = signal.shift(1).fillna(0.0)
strategy_return = position * asset_return
```

This is the core anti-lookahead guard. The integrity layer also emits a
warm-up leakage heuristic warning when rolling-style feature columns appear
fully populated in the earliest rows of each symbol.

### B. Input Validation Layer

Before a strategy generates signals, StratLake validates that the input frame
matches the strategy contract.

Checks include:

* required columns are present
* the frame is not empty
* key columns `symbol`, `ts_utc`, and `timeframe` are non-null and parseable
* the configured timeframe matches the loaded data
* rows are sorted by `(symbol, ts_utc, timeframe)`
* at least one usable row remains after required-column validation

Hard input violations raise `StrategyInputError` immediately. This is
intentional fail-fast behavior: invalid inputs do not produce partial artifacts.

Low sample size is tracked separately as `low_data`; it is a warning condition,
not a hard failure.

### C. Signal Diagnostics Layer

After signals are attached, StratLake computes deterministic diagnostics to
make behavior visible before users rely on headline returns.

`signal_diagnostics.json` includes:

* direction mix: `pct_long`, `pct_short`, `pct_flat`
* activity: `total_trades`, `turnover`
* persistence: `avg_holding_period`
* market participation: `exposure_pct`
* degenerate-behavior flags such as `always_flat`, `always_long`, `always_short`, `no_trades`, and `high_turnover`

These diagnostics do not change the strategy output. They make silent failure
modes easier to spot, such as strategies that never trade or remain fully long
for the whole sample.

### D. Strategy QA Layer

The QA layer converts diagnostics, execution state, input metadata, and
benchmark-relative warnings into one compact summary artifact:

* `qa_summary.json`

The summary contains:

* run identity: `run_id`, `strategy_name`, `dataset`, `timeframe`
* coverage: `row_count`, `symbols_present`, `date_range`
* input metadata: `input_validation`
* signal summary: direction mix, turnover, trade count
* execution checks: `valid_returns`, `equity_curve_present`
* key metrics: `total_return`, `sharpe`, `max_drawdown`
* benchmark-relative metrics: benchmark return, excess return, correlation, relative drawdown
* flags: warning and failure indicators
* `overall_status`: `pass`, `warn`, or `fail`

Status rules are intentionally simple:

* `fail`: no usable output, missing execution outputs, or integrity failure
* `warn`: output exists, but one or more caution flags are active
* `pass`: no failure conditions and no active warning flags

### E. Consistency Validation Layer

After artifacts are written, StratLake validates that the saved files agree
with each other.

Consistency checks cover:

* `manifest.json` versus files actually present on disk
* `metrics.json` versus values recomputed from `equity_curve.csv`
* `qa_summary.json` versus `metrics.json`, `signal_diagnostics.json`, and row counts
* registry entry presence and metric agreement
* walk-forward split inventory and `metrics_by_split.csv` agreement

Consistency errors raise `ConsistencyError` and treat the run as invalid.
Consistency warnings are reserved for softer checks, such as diagnostics that
look numerically implausible but are still parseable.

### F. Deterministic Reproducibility

Reproducibility is enforced through deterministic execution and deterministic
artifact naming.

Current guarantees include:

* stable `run_id` generation from strategy name, evaluation mode, normalized config, and normalized results
* stable artifact ordering and timestamp formatting
* deterministic signal diagnostics and QA summaries
* reruns of unchanged experiments rewriting the same run directory
* registry deduplication through upsert-by-`run_id`

In practice, identical inputs and identical code produce the same logical run
record rather than a new timestamp-only variant.

### G. Benchmark-Relative Evaluation

StratLake does not stop at absolute returns. It also computes relative context
against a deterministic buy-and-hold benchmark on the same dataset.

The benchmark-relative layer adds:

* `benchmark_total_return`
* `excess_return`
* `benchmark_correlation`
* `relative_drawdown`
* warning-only plausibility flags

Those plausibility flags currently identify:

* high benchmark correlation
* low excess return
* high turnover with little edge
* beta-dominated behavior

These are interpretation aids, not execution blockers. A strategy can still
finish with `warn` and persist complete artifacts.

## Artifacts

### `metrics.json`

Purpose: summary metric payload for the completed run.

Key fields:

* absolute metrics such as `total_return`, `sharpe_ratio`, `max_drawdown`, `turnover`, `exposure_pct`
* benchmark-relative metrics such as `benchmark_total_return`, `excess_return`, `benchmark_correlation`, and `relative_drawdown`
* `plausibility_flags` for warning-only relative diagnostics

Interpretation: start here for headline performance, then confirm the result is
supported by diagnostics and QA rather than reading it in isolation.

### `equity_curve.csv`

Purpose: standardized timeline used for review, reporting, and recomputation.

Key fields:

* `ts_utc`
* `symbol` when present
* `equity`
* `strategy_return`
* `signal`
* `position`

Interpretation: this is the easiest artifact for checking when returns were
earned and whether executed positions match expected signal behavior.

### `signals.parquet`

Purpose: signal-engine output aligned to the feature dataset.

Key fields:

* canonical identifiers such as `ts_utc`, `date`, `symbol`
* `signal`
* `position`
* original feature columns used during evaluation

Interpretation: use this to inspect what the strategy decided before reviewing
backtest results.

### `signal_diagnostics.json`

Purpose: compact behavioral diagnostics for the generated signals.

Key fields:

* `pct_long`, `pct_short`, `pct_flat`
* `total_trades`
* `turnover`
* `avg_holding_period`
* `exposure_pct`
* `flags`

Interpretation: this answers "what kind of strategy behavior produced the
returns?" rather than "how high were the returns?"

### `qa_summary.json`

Purpose: one-file health summary for a completed run.

Key fields:

* `overall_status`
* `input_validation`
* `signal`
* `execution`
* `metrics`
* `relative`
* `flags`

Interpretation: use this as the primary trust readout. It compresses the most
important validation and warning signals into a single status payload.

### `manifest.json`

Purpose: compact run inventory and artifact index.

Key fields:

* `run_id`
* `strategy_name`
* `evaluation_mode`
* `artifact_files`
* `split_count`
* `primary_metric`
* `metric_summary`

Interpretation: inspect this first when loading a saved run. It tells you what
files should exist and what the run is supposed to represent.

## Failure And Warning Semantics

### Fail

`fail` means the run is not trustworthy enough to treat as a valid completed
research result.

Typical fail cases:

* invalid input structure
* missing required columns
* empty input data
* no usable rows after validation
* missing or unusable return data
* integrity violations such as misaligned signals or same-bar execution
* missing required artifacts
* cross-artifact mismatches detected by consistency validation

Failing runs exit the CLI with status code `1`. Error messages are printed as:

```text
Run failed: <reason>
```

### Warn

`warn` means the run completed and artifacts were written, but interpretation
should be cautious.

Current warning conditions include:

* low data
* no trades
* degenerate signal behavior
* high turnover
* low excess return versus benchmark
* high benchmark correlation
* high turnover with low edge
* beta-dominated benchmark behavior

Warnings appear in two places:

* `qa_summary.json` via `flags` plus `overall_status: "warn"`
* CLI output under a `Warnings:` section when the runner can summarize them

Example CLI warning lines:

```text
Warnings:
- insufficient data for a high-confidence analysis
- no trades were generated
- strategy is highly correlated with the benchmark (0.94)
```

## Examples

### QA Summary Snippet

```json
{
  "overall_status": "warn",
  "row_count": 84,
  "signal": {
    "pct_long": 0.57,
    "pct_short": 0.00,
    "pct_flat": 0.43,
    "turnover": 0.12,
    "total_trades": 10
  },
  "flags": {
    "low_data": true,
    "no_trades": false,
    "high_benchmark_correlation": true,
    "low_excess_return": false
  }
}
```

Interpretation: the run completed, but the sample is small and returns may be
too benchmark-driven to trust without deeper review.

### Signal Diagnostics Snippet

```json
{
  "pct_long": 0.61,
  "pct_short": 0.09,
  "pct_flat": 0.30,
  "total_trades": 18,
  "turnover": 0.21,
  "avg_holding_period": 4.8,
  "exposure_pct": 0.70,
  "flags": {
    "always_flat": false,
    "no_trades": false,
    "high_turnover": false
  }
}
```

Interpretation: the strategy is active and directional, with moderate turnover
and multi-bar holding periods.

### Benchmark-Relative Output Snippet

```json
{
  "benchmark_total_return": 0.18,
  "excess_return": 0.01,
  "benchmark_correlation": 0.93,
  "relative_drawdown": 0.02,
  "plausibility_flags": {
    "high_benchmark_correlation": true,
    "low_excess_return": true,
    "high_turnover_low_edge": false,
    "beta_dominated_strategy": false
  }
}
```

Interpretation: the strategy slightly outperformed buy-and-hold, but behaved so
similarly to the benchmark that the incremental edge may not be meaningful.

## Reading Order

For a fast review, inspect artifacts in this order:

1. `manifest.json`
2. `qa_summary.json`
3. `metrics.json`
4. `signal_diagnostics.json`
5. `equity_curve.csv`
6. `signals.parquet`

That sequence moves from run identity, to trust status, to headline metrics, to
behavioral explanation, to full timeline detail.
