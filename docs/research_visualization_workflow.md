# Research Visualization And Reporting Workflow

## Overview

StratLake's visualization and reporting layers extend the existing research
artifact workflow after signals, backtests, and metrics have already been
computed.

They do not replace the core research outputs. Instead, they consume saved run
artifacts and produce deterministic inspection assets that make a run easier to
review, compare, and share.

At a high level, the workflow is:

```text
feature dataset
    ->
signals
    ->
backtest
    ->
metrics
    ->
run artifacts
    ->
plots
    ->
report.md
```

The visualization layer lives under `src/research/visualization/`.
The reporting layer lives under `src/research/reporting/`.

Use this guide alongside:

* [strategy evaluation workflow](strategy_evaluation_workflow.md)
* [experiment artifact logging](experiment_artifact_logging.md)
* [strategy performance metrics](strategy_performance_metrics.md)
* [CLI strategy runner](cli_strategy_runner.md)

## Where Visualization Fits In The Research Workflow

The research runner still centers on deterministic strategy execution:

1. load a feature dataset
2. generate signals
3. run the backtest
4. compute summary metrics
5. persist run artifacts under `artifacts/strategies/<run_id>/`

Visualization starts after those artifacts exist.

The current implementation treats plotting as artifact-driven analysis:

* strategy artifacts remain the source of truth
* metrics remain the primary numeric summary
* plots provide visual inspection of those saved outputs
* reports assemble metrics and plot artifacts into a deterministic Markdown
  document

This keeps the workflow file-based and reproducible. A run can be visualized or
reported later without rerunning the strategy logic, as long as the required
artifacts are available in the run directory.

## Visualization Modules And Responsibilities

The visualization package groups plotting helpers by analysis domain.

### `src/research/visualization/equity.py`

Purpose:

* normalize time-indexed strategy inputs
* compound returns into cumulative equity when needed
* render strategy performance charts

Current responsibilities:

* `plot_equity_curve()`: plot a strategy equity curve, optionally with a
  benchmark overlay
* `plot_cumulative_returns()`: plot cumulative performance from returns or
  equity inputs

These functions accept pandas `Series` or single-column `DataFrame` inputs and
either return a matplotlib figure or save a deterministic PNG artifact when
`output_path` is provided.

### `src/research/visualization/diagnostics.py`

Purpose:

* provide diagnostic views derived from returns, drawdowns, signals, exposures,
  positions, and trade-level outputs

Current responsibilities:

* drawdown and underwater analysis via `plot_drawdown()` and
  `plot_underwater_curve()`
* rolling metric analysis via `compute_rolling_sharpe()`,
  `compute_rolling_volatility()`, `plot_rolling_metric()`, and
  `plot_rolling_sharpe()`
* signal diagnostics via `plot_signal_distribution()` and
  `plot_signal_diagnostics()`
* exposure diagnostics via `plot_exposure_over_time()`
* long/short count diagnostics via `compute_long_short_counts()` and
  `plot_long_short_counts()`
* trade-level analysis via `compute_trade_statistics()`,
  `plot_trade_return_distribution()`, and `plot_win_loss_distribution()`

These helpers are deterministic functions over already-saved or already-loaded
research outputs. They do not introduce a separate stateful reporting system.

### `src/research/visualization/walk_forward.py`

Purpose:

* visualize evaluation split structure and fold-level results for walk-forward
  analysis

Current responsibilities:

* `plot_walk_forward_splits()`: visualize train and test windows by fold
* `plot_fold_level_metrics()`: plot one metric across folds
* `plot_walk_forward_results()`: summarize per-fold outcomes from fold result
  frames

These functions are useful once walk-forward split definitions,
`metrics_by_split.csv`, or split-level result frames already exist.

### `src/research/visualization/comparison.py`

Purpose:

* visualize multiple strategies or runs on shared axes

Current responsibilities:

* `plot_equity_comparison()`: compare aligned equity curves across runs
* `plot_strategy_overlays()`: overlay legacy strategy frames
* `plot_metric_comparison()`: compare one metric across strategies or runs
* `plot_strategy_metric_bars()`: compare multiple metrics across strategies

These helpers complement the comparison CLI and leaderboard artifacts by adding
artifact-driven visual inspection of strategy differences.

## Plot Artifact Conventions

Visualization helpers follow a simple convention:

* if `output_path` is omitted, the function returns a matplotlib `Figure`
* if `output_path` is provided, the function saves a PNG artifact and returns
  the saved `Path`

The plotting layer itself does not enforce one global output directory. The
caller chooses where artifacts are stored.

The reporting layer currently standardizes plot output under:

```text
artifacts/strategies/<run_id>/plots/
```

Current report-generated filenames are:

* `equity_curve.png`
* `drawdown.png`
* `rolling_sharpe.png`
* `trade_return_distribution.png`
* `win_loss_distribution.png`

Only plots supported by the available run artifacts are generated. For example:

* `rolling_sharpe.png` requires a usable `strategy_return` series with at least
  the current 20-period rolling window
* trade distribution plots require `trades.parquet` with a numeric `return`
  column

Walk-forward and comparison plots exist in the visualization layer, but they
are not automatically emitted by `generate_strategy_report()` in the current
implementation.

## Report Generation Workflow

The reporting entrypoint is:

```python
from src.research.reporting import generate_strategy_report
```

Implementation location:

```text
src/research/reporting/report_generator.py
```

Primary interface:

```python
generate_strategy_report(run_dir: Path, output_path: Path | None = None) -> Path
```

`generate_strategy_report()` expects a saved strategy run directory and writes a
deterministic Markdown report.

Default output location:

```text
artifacts/strategies/<run_id>/report.md
```

If `output_path` is omitted, the report is written to `run_dir / "report.md"`.

### Inputs From A Run Directory

Required artifact:

* `metrics.json`

Optional artifacts:

* `manifest.json`
* `config.json`
* `equity_curve.csv`
* `trades.parquet`
* `signals.parquet`

Current input usage:

* `metrics.json`: required for the performance summary table
* `manifest.json`: used for run metadata and report title when present
* `config.json`: used for strategy name fallback and parameter display
* `equity_curve.csv`: used to derive equity, drawdown, and rolling Sharpe plots
* `trades.parquet`: used to derive trade statistics and trade distribution plots
* `signals.parquet`: currently used only for optional signal-count notes when
  equity artifacts do not already expose signal rows

### Report Sections

The current `report.md` structure is:

* `# Strategy Report: <name>`
* `## Run Metadata`
* `## Performance Summary`
* `## Equity Curve` when an equity artifact is available
* `## Drawdown` when an equity artifact is available
* `## Rolling Metrics`
* `## Trade Analysis`
* `## Signal Summary` when signal-related artifacts are available
* `## Observations`

`Rolling Metrics` and `Trade Analysis` always appear, but they may fall back to
deterministic placeholder text when the required optional artifacts are absent.

### Markdown Image References

Report plot references are stored as relative paths from `report.md` to the
plot artifact. In the default layout, image links look like:

```markdown
![Equity Curve](plots/equity_curve.png)
```

This keeps the report portable within its run directory.

## CLI Entry Points

The repository now exposes the existing visualization and reporting workflow
through two thin CLI modules:

* `src/cli/plot_strategy_run.py`
* `src/cli/generate_report.py`

These commands operate on an existing saved run directory. They do not rerun a
strategy, recompute metrics, or introduce a new reporting stack.

### `plot_strategy_run`

Usage:

```powershell
.\.venv\Scripts\python.exe -m src.cli.plot_strategy_run --run-dir artifacts/strategies/<run_id>
```

Behavior:

* validates that `--run-dir` exists and contains the required core run artifact
  `metrics.json`
* generates the currently supported plot artifacts for the available inputs
* writes those artifacts under `artifacts/strategies/<run_id>/plots/`
* skips plots that require optional artifacts not present in the run directory
* raises a clear error if no supported plot inputs are available

### `generate_report`

Usage:

```powershell
.\.venv\Scripts\python.exe -m src.cli.generate_report --run-dir artifacts/strategies/<run_id>
```

Optional output override:

```powershell
.\.venv\Scripts\python.exe -m src.cli.generate_report --run-dir artifacts/strategies/<run_id> --output-path artifacts/strategies/<run_id>/custom_report.md
```

Behavior:

* validates that `--run-dir` exists
* requires `metrics.json`
* calls `generate_strategy_report(...)`
* reuses existing plot artifacts when present and generates missing ones through
  the reporting flow
* writes `report.md` to the run directory by default

## Typical Output Structure

For a single-run report, the current output shape is:

```text
artifacts/strategies/<run_id>/
  config.json
  metrics.json
  equity_curve.csv
  equity_curve.parquet
  signals.parquet
  trades.parquet
  manifest.json
  report.md
  plots/
    equity_curve.png
    drawdown.png
    rolling_sharpe.png
    trade_return_distribution.png
    win_loss_distribution.png
```

A run does not need every optional artifact to produce a report. The minimum
contract is `metrics.json`, with additional sections and plots included only
when the supporting files exist.

For walk-forward runs, the report generator still operates at the run-root
artifact level. It does not currently generate one separate Markdown report per
split.

## Determinism And Reproducibility

The visualization and reporting layers are designed to match the rest of the
research stack's deterministic behavior.

Current reproducibility properties:

* reports are generated from persisted run artifacts rather than live notebook
  state
* report plot names are fixed by the implementation
* plot generation reuses an existing artifact unless overwrite behavior is
  requested internally
* report sections are emitted in a fixed order
* Markdown image links are derived relative to the report location
* identical inputs produce the same `report.md` contents

In practice, this means the visualization/reporting workflow is best thought of
as a deterministic presentation layer on top of saved research outputs.

## Relationship Between Artifacts, Metrics, Plots, And Reports

The current relationship is:

* strategy artifacts capture the saved outputs of one run
* metrics summarize the run numerically
* plots visualize selected time-series and diagnostic artifacts
* the Markdown report assembles metrics, plots, and metadata into one reviewable
  deliverable

A simple mental model is:

```text
run directory = source artifacts
metrics.json = numeric summary
plots/ = visual summary
report.md = assembled research brief
```

## Current Limitations And Future Extensions

The current implementation is intentionally narrow and file-based.

Current limitations:

* no dedicated reporting CLI is documented or required
* no dashboard or interactive visualization surface exists in this repository
* `generate_strategy_report()` currently emits a fixed report structure rather
  than a customizable template system
* walk-forward, comparison, signal, and exposure plotting helpers exist as
  modules, but are not yet automatically attached to the generated Markdown
  report
* plot filenames are deterministic only for the artifacts currently generated by
  the reporting layer

These limits are deliberate. The implemented workflow focuses on deterministic,
inspectable research artifacts rather than a larger visualization platform.
