# Visualization And Reporting Audit

## Purpose And Scope

This audit reviews the repository's current visualization and Markdown
reporting outputs as implemented in the existing artifact-driven workflow.
The goal is to document what is already working well, what appears best suited
for debug-oriented review, and what should be redesigned in later Milestone 7
issues without changing research logic, artifact schemas, or CLI behavior.

This document is preserved as a historical Milestone 7 audit snapshot. Some
artifact names and report details below describe the pre-polish state reviewed
during the audit rather than the latest implementation. For the current
workflow and filenames, use [research_visualization_workflow.md](research_visualization_workflow.md).

Sources reviewed:

* workflow docs in `README.md` and `docs/`
* CLI entrypoints in `src/cli/`
* plotting and reporting modules in `src/research/visualization/` and
  `src/research/reporting/`
* tests covering visualization and reporting behavior
* checked-in example artifacts under `artifacts/`

## Current Workflow Inventory

### Active CLI Entry Points

* `src/cli/plot_strategy_run.py`
  Generates deterministic single-run plot artifacts from an existing run
  directory.
* `src/cli/generate_report.py`
  Generates `report.md` from an existing run directory and reuses or creates
  supported plots under `<run_dir>/plots/`.
* `src/cli/strategy_comparison_example.py`
  Demonstrates comparison outputs and writes two comparison plots, but this is
  an example workflow rather than the main comparison CLI.
* `src/cli/compare_strategies.py`
  Produces leaderboard CSV/JSON outputs only. It does not currently generate
  comparison plots by itself.

### Plot-Producing Code Paths

Implemented helpers live under `src/research/visualization/`:

* `equity.py`
  `plot_equity_curve()`, `plot_cumulative_returns()`
* `diagnostics.py`
  `plot_drawdown()`, `plot_underwater_curve()`, `plot_rolling_sharpe()`,
  `plot_signal_distribution()`, `plot_signal_diagnostics()`,
  `plot_exposure_over_time()`, `plot_long_short_counts()`,
  `plot_trade_return_distribution()`, `plot_win_loss_distribution()`
* `comparison.py`
  `plot_equity_comparison()`, `plot_metric_comparison()`,
  `plot_strategy_metric_bars()`, `plot_strategy_overlays()`
* `walk_forward.py`
  `plot_walk_forward_splits()`, `plot_fold_level_metrics()`,
  `plot_walk_forward_results()`

At audit time, auto-generated plots in the single-run reporting flow were
limited to:

* `equity_curve.png`
* `drawdown.png`
* `rolling_sharpe.png` when at least 20 return observations exist
* `trade_return_distribution.png` when `trades.parquet` has numeric `return`
* `win_loss_distribution.png` when `trades.parquet` has numeric `return`

### Report-Producing Code Paths

* `src/research/reporting/report_generator.py`
  Core report assembly, plot reuse, relative-path handling, and run-artifact
  loading.
* `src/research/reporting/__init__.py`
  Convenience loading and quick-summary helpers around saved run artifacts.

At audit time, report section order was:

* `Run Metadata`
* `Performance Summary`
* `Equity Curve` when available
* `Drawdown` when available
* `Rolling Metrics`
* `Trade Analysis`
* `Signal Summary` when signal rows are available
* `Observations`

### Artifact Names And Locations

Single-run visualization/report artifacts observed during the audit:

* `<run_dir>/plots/equity_curve.png`
* `<run_dir>/plots/drawdown.png`
* `<run_dir>/plots/rolling_sharpe.png`
* `<run_dir>/plots/trade_return_distribution.png`
* `<run_dir>/plots/win_loss_distribution.png`
* `<run_dir>/report.md`

Walk-forward run artifacts currently available for review:

* `<run_dir>/equity_curve.csv`
* `<run_dir>/metrics.json`
* `<run_dir>/metrics_by_split.csv`
* `<run_dir>/splits/<split_id>/...`

Comparison artifacts currently available for review:

* `artifacts/comparisons/<comparison_id>/leaderboard.csv`
* `artifacts/comparisons/<comparison_id>/leaderboard.json`
* example-only comparison plots under
  `artifacts/comparisons/strategy_comparison_example/plots/`

## What Is Working Well

* The workflow is artifact-first and deterministic. Plotting/reporting consume
  saved outputs instead of recomputing strategy logic.
* Report image links are relative, which keeps `report.md` portable inside a
  copied run directory.
* Plot filenames and the `<run_dir>/plots/` layout are consistent and tested.
* The report gracefully skips unsupported optional sections instead of failing
  when trade or equity artifacts are absent.
* The current section order is easy to scan and suitable for quick run review.
* Comparison example artifacts are discoverable and provide a concrete
  repository reference for multi-strategy visuals.

## Problems And Risks

### Readability And Visual Density

* `plot_equity_comparison()` overlays every strategy on one axis with a simple
  legend. This is fine for the three-strategy example, but it will become hard
  to read as strategy count grows.
* `plot_strategy_metric_bars()` can become crowded quickly because all selected
  metrics share one grouped bar chart without any normalization or label
  rotation support.
* `plot_walk_forward_splits()` uses one horizontal band per fold. It is clear
  for a few folds, but dense rolling evaluations will likely become hard to
  scan.

### Legend And Label Clarity

* Axis titles are technically present across the plotting helpers, but labels
  stay generic. For example, `Date`, `Equity`, `Drawdown`, and raw metric names
  do not distinguish daily vs minute data or single-run vs walk-forward
  context.
* `rolling_sharpe.png` uses a fixed 20-period window. The title includes
  `20-period`, but the plot does not clarify timeframe or annualization, which
  weakens interpretability across datasets.
* `trade_return_distribution.png` puts mean, median, and standard deviation in
  the legend label. That is compact, but the legend becomes more like a stats
  caption than a true series label.

### Report Structure And Portability

* The report is portable because of relative image links, and that should stay
  unchanged.
* The report is intentionally generic, but the `Observations` section is only a
  placeholder. This makes checked-in example reports look unfinished.
* `Trade Count`, `Winning Trades`, and `Losing Trades` are rendered with six
  decimal places because report tables format all values as floats. That hurts
  polish for report-facing output.
* `Signal Summary` currently reports row counts only. It does not summarize the
  actual signal mix, exposure behavior, or any diagnostic distribution even
  though helper plots exist for those areas.

### Coverage Gaps Between Existing Helpers And Delivered Outputs

* There are implemented helpers for signal diagnostics, exposure, long/short
  counts, walk-forward splits, fold metrics, and comparison plots, but the main
  plot/report CLI flow does not surface them.
* Walk-forward runs already save `metrics_by_split.csv` and split directories,
  but `generate_strategy_report()` does not currently add split-aware sections
  or walk-forward charts.
* The main comparison CLI writes leaderboard artifacts only. Comparison plots
  are currently reachable through the example workflow instead of the primary
  comparison workflow.

### Consistency Between Code And Docs

* `docs/research_visualization_workflow.md` correctly explains the current
  single-run plot/report flow.
* The same doc also lists broader helper responsibilities in
  `comparison.py` and `walk_forward.py`, which can read as if they are first-
  class generated outputs even though they are mostly reusable helpers today.
* The checked-in example comparison workflow helps close that gap, but the
  distinction between "helper exists" and "main CLI emits this artifact" should
  stay explicit in follow-on documentation.

## Classification

### Report-Quality

These outputs are suitable to keep as part of the main report/review workflow
with refinement rather than replacement:

| Output | Current location | Why it fits |
| --- | --- | --- |
| Equity curve plot | `<run_dir>/plots/equity_curve.png` | Core performance view, easy to read, already consistent with artifact-driven workflow |
| Drawdown plot | `<run_dir>/plots/drawdown.png` | High-value risk context and pairs well with equity curve |
| Performance summary table | `report.md` | Compact, deterministic, and portable |
| Run metadata section | `report.md` | Useful for reproducibility and run identification |
| Relative-path Markdown report | `<run_dir>/report.md` | Portable and aligned with repository expectations |
| Comparison equity overlay for small examples | example comparison workflow | Useful when strategy count stays low |
| Leaderboard CSV/JSON | `artifacts/comparisons/<comparison_id>/...` | Strong machine-readable review artifacts |

### Debug-Only

These outputs are helpful during inspection, but they should not be treated as
polished report defaults without additional design work:

| Output | Current location | Why it is debug-oriented |
| --- | --- | --- |
| Rolling Sharpe plot | `<run_dir>/plots/rolling_sharpe.png` | Useful for diagnostics, but context-light and sensitive to arbitrary window choice |
| Trade return histogram | `<run_dir>/plots/trade_return_distribution.png` | Helpful for debugging trade distribution shape, less essential for a concise report |
| Win/loss count chart | `<run_dir>/plots/win_loss_distribution.png` | Diagnostic summary, but coarse and lower-signal than core performance/risk views |
| Signal summary row count | `report.md` | Confirms artifact presence more than strategy behavior |
| Walk-forward split timeline | helper output | Good for validation/debugging split construction |
| Fold-level metric chart | helper output | Good for debugging split instability and outliers |
| Exposure, signal-distribution, and long/short count plots | helper outputs | Useful diagnostics, but not yet polished enough for default report inclusion |

### Needs Redesign

These outputs or workflow areas need follow-on design work before they should
be considered report-ready defaults:

| Output or area | Current state | Redesign need |
| --- | --- | --- |
| Comparison plots as a primary workflow output | Only generated through `strategy_comparison_example.py` | Needs a clearer home in the main comparison workflow |
| Multi-metric grouped comparison bars | Helper exists only | Likely to overplot and mix incomparable scales |
| Walk-forward reporting | Split artifacts exist but report integration is absent | Needs split-aware sections and plot selection rules |
| Placeholder observations section | Static `_Add analysis notes here._` text | Needs either meaningful autogenerated content or removal from default report |
| Trade statistics formatting | Counts shown as floats | Needs report-friendly formatting rules |
| Signal/exposure diagnostics in reports | Helpers exist but no report structure | Needs a better summary design before default inclusion |

## Prioritized Follow-On Recommendations

1. Keep the default single-run report focused on equity, drawdown, metrics, and
   reproducibility metadata. Treat rolling and trade-distribution views as
   optional diagnostics unless a later issue promotes them deliberately.
2. Add walk-forward-specific reporting in a separate follow-on issue rather than
   stretching the current single-run report structure. `metrics_by_split.csv`
   and split artifacts already provide the source material.
3. Decide whether comparison plots belong in `compare_strategies` or remain an
   example-only layer. The current split between leaderboard generation and plot
   generation is functional but not obvious.
4. Improve report polish before adding new content. The highest-value small
   fixes are removing the placeholder observations text and formatting count
   fields as integers.
5. Reserve signal, exposure, and long/short diagnostics for a later issue with
   clear selection criteria. The helpers are already present, but the current
   report does not have a strong structure for them yet.
6. When future redesign work happens, preserve the current deterministic naming,
   relative-path Markdown behavior, and run-scoped plot directory contract.

## Audit Summary

The current repository already has a solid deterministic foundation for
artifact-driven plotting and portable Markdown reporting. The strongest
report-ready pieces are the run metadata, performance summary, equity curve,
drawdown chart, and stable artifact layout. Most remaining work is not about
new plotting primitives; it is about choosing which existing visuals deserve to
be primary outputs, which should stay diagnostic, and how walk-forward and
comparison workflows should be presented more intentionally.
