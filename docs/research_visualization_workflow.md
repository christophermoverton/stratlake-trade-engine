# Research Visualization And Reporting Workflow

## Overview

StratLake's visualization and reporting layers sit after research execution.
They extend the saved outputs of a strategy run; they do not replace the core
artifacts, metrics, or evaluation flow.

The purpose of these layers is to turn existing run artifacts into
deterministic review assets:

* plots for visual inspection of performance and diagnostics
* a Markdown report for a portable research summary

The implementation is artifact-driven and file-based. Once a run has been
saved, plots and reports can be generated later without rerunning the strategy
logic.

Use this guide alongside:

* [docs/strategy_evaluation_workflow.md](strategy_evaluation_workflow.md)
* [docs/experiment_artifact_logging.md](experiment_artifact_logging.md)
* [docs/strategy_performance_metrics.md](strategy_performance_metrics.md)
* [docs/visualization_reporting_audit.md](visualization_reporting_audit.md)

## Workflow Integration

Visualization and reporting extend StratLake's existing research workflow:

```text
data
    ->
features
    ->
strategy
    ->
evaluation
    ->
visualization
    ->
reporting
```

In practical repository terms, that flow becomes:

```text
curated market data
    ->
feature datasets
    ->
signals
    ->
backtest and metrics
    ->
saved run artifacts
    ->
plots
    ->
report.md
```

The saved run directory remains the source of truth. Visualization consumes
artifacts such as `metrics.json`, `equity_curve.csv`, and optional trade or
signal files. Reporting then assembles those artifacts and any supported plots
into one deterministic Markdown deliverable.

## How To Generate Plots

The plot generation CLI is implemented by
`src/cli/plot_strategy_run.py` and the underlying reporting helpers in
`src/research/reporting/report_generator.py`.

The run-scoped command is:

```text
plot_strategy_run --run-dir <path>
```

Repository module invocation:

```powershell
.\.venv\Scripts\python.exe -m src.cli.plot_strategy_run --run-dir artifacts/strategies/<run_id>
```

Behavior:

* validates that `--run-dir` exists and contains `metrics.json`
* writes supported plot artifacts under `<run_dir>/plots/`
* generates only plots supported by the artifacts present in the run directory
* raises an error if the run has no usable plot inputs

Plots currently generated from a supported single-run artifact set are:

* `equity_curve.png`
* `drawdown.png`
* `rolling_sharpe.png` when `equity_curve.csv` contains enough
  `strategy_return` history for the current 20-period window
* `trade_return_distribution.png` when `trades.parquet` has a numeric
  `return` column
* `win_loss_distribution.png` when `trades.parquet` has a numeric `return`
  column

Only supported plots are emitted. For example:

* if `equity_curve.csv` is absent, equity, drawdown, and rolling Sharpe plots
  are skipped
* if `trades.parquet` is absent, trade distribution plots are skipped

## How To Generate Reports

The report generation CLI is implemented by `src/cli/generate_report.py`.

The run-scoped command is:

```text
generate_report --run-dir <path>
```

Repository module invocation:

```powershell
.\.venv\Scripts\python.exe -m src.cli.generate_report --run-dir artifacts/strategies/<run_id>
```

Optional output override:

```text
generate_report --run-dir <path> --output-path <path>
```

Repository module invocation with output override:

```powershell
.\.venv\Scripts\python.exe -m src.cli.generate_report --run-dir artifacts/strategies/<run_id> --output-path artifacts/strategies/<run_id>/custom_report.md
```

Behavior:

* validates that `--run-dir` exists and contains `metrics.json`
* writes `<run_dir>/report.md` by default
* accepts `--output-path <path>` to write the report somewhere else
* reuses existing standardized plot artifacts when they already exist
* generates missing supported plots as part of report generation when needed
* renders a fixed Markdown section order so reports stay easy to scan across runs

The report generator always anchors plot artifacts under the standardized
`<run_dir>/plots/` directory, even when the report itself is written to a
custom output path.

Generated reports follow a lightweight, portable structure:

* title and run header with strategy, run id, mode, timeframe, and date range
* run configuration summary with dataset, parameters, and evaluation settings
* key metrics table rendered in deterministic Markdown
* visualizations grouped into performance, rolling diagnostics, and trade diagnostics
* a short interpretation section derived from saved metrics and optional trade artifacts
* artifact references linking back to `metrics.json`, `equity_curve.csv`,
  optional parquet artifacts, `metrics_by_split.csv` for walk-forward runs,
  and generated plots

## Artifact Structure

The standardized visualization and reporting artifact layout is:

* `<run_dir>/plots/`
* `<run_dir>/report.md` by default
* a custom report location when `--output-path` is provided

Example run layout:

```text
artifacts/strategies/<run_id>/
  manifest.json
  config.json
  metrics.json
  equity_curve.csv
  signals.parquet
  trades.parquet
  plots/
    equity_curve.png
    drawdown.png
    rolling_sharpe.png
    trade_return_distribution.png
    win_loss_distribution.png
  report.md
```

Not every run will contain every optional artifact. The minimum contract for
plot and report generation is `metrics.json`, with additional visual outputs
included only when the supporting artifacts exist.

## Visualization Coverage

The visualization package is organized by analysis domain under
`src/research/visualization/`.

Shared plot-level defaults live in `src/research/visualization/plot_utils.py`.
That module centralizes figure sizing, save settings, date-axis formatting,
label/grid application, and legend behavior so the existing plot helpers stay
consistent without changing the artifact-driven workflow.

### `equity.py`

Performance-oriented plotting for saved return and equity artifacts.

Current responsibilities include:

* equity curve visualization
* cumulative return visualization

### `diagnostics.py`

Diagnostic plotting and summary helpers for risk, rolling metrics, signals,
exposure, and trades.

Current responsibilities include:

* drawdown and underwater analysis
* rolling Sharpe and rolling volatility helpers
* signal distribution and signal diagnostics
* exposure and long/short count diagnostics
* trade return and win/loss distributions

### `walk_forward.py`

Validation-oriented plotting for walk-forward research outputs.

Current responsibilities include:

* train/test split visualization
* fold-level metric visualization
* aggregate walk-forward result plotting

### `comparison.py`

Multi-strategy comparison plots for shared review across runs.

Current responsibilities include:

* strategy-level equity comparison summaries across runs
* raw equity overlay diagnostics for dense multi-run inspection
* single-metric comparisons
* multi-metric comparison bars

Comparison equity behavior is intentionally split by intent:

* `equity_comparison.png` is the report-quality view
* it renders one summary line per strategy
* when multiple runs share a strategy label, the summary line uses the median
  equity path and a shaded interquartile band to show variability
* legends describe the rendered summary lines rather than every raw trace
* debug-oriented raw overlays can be emitted separately as
  `equity_comparison_debug.png`, where faint individual runs sit behind the
  highlighted strategy median

Metric comparison bar charts also sort strategies from best to worst for the
selected metric so leaderboard-style questions are easier to answer quickly.

These modules exist as reusable visualization helpers. The current report
generation flow automatically emits only the supported single-run plots derived
from saved run artifacts.

## Determinism And Reproducibility

Visualization and reporting follow the same reproducibility goals as the rest
of StratLake:

* outputs are generated from persisted artifacts rather than notebook state
* plot filenames and plot locations are standardized
* existing plot artifacts are reused unless regeneration is required
* report sections are emitted in a fixed order
* relative plot links keep `report.md` portable within the artifact layout

Given the same saved run artifacts, StratLake produces the same plot set and
the same Markdown report content, including section order, metric formatting,
and relative artifact links.

## Limitations And Scope

The current scope is intentionally narrow:

* no dashboards
* no interactive UI
* CLI commands and saved artifacts are the primary interface
* report generation uses a fixed Markdown structure rather than a templating
  system
* visualization helpers for walk-forward and comparison workflows exist, but
  reports remain artifact-driven and lightweight rather than becoming
  full narrative analysis documents

This design keeps the workflow deterministic, reviewable, and aligned with the
repository's artifact-first research model.
