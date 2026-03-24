# Strategy Comparison Example

## Overview

This example runs three existing daily strategies over one bounded date range,
generates the standard per-run plot artifacts, produces a Markdown report for a
representative run, and writes deterministic comparison artifacts in one place.

Selected strategies:

* `momentum_v1`
* `mean_reversion_v1`
* `buy_and_hold_v1`

Example data scope:

* dataset: `features_daily`
* start: `2025-01-01`
* end: `2025-03-01`

## End-To-End Command

Run the complete example workflow:

```powershell
.\.venv\Scripts\python.exe -m src.cli.strategy_comparison_example --start 2025-01-01 --end 2025-03-01 --output-dir artifacts/comparisons/strategy_comparison_example
```

This command:

* executes the selected strategies with bounded dates
* writes the leaderboard to `artifacts/comparisons/strategy_comparison_example/`
* generates per-run plots under each `<run_dir>/plots/`
* generates `report.md` for `momentum_v1`
* writes comparison plots under `artifacts/comparisons/strategy_comparison_example/plots/`
* writes `example_summary.json` with the resolved artifact paths

## Equivalent Manual Commands

Run the strategies directly:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --start 2025-01-01 --end 2025-03-01
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy mean_reversion_v1 --start 2025-01-01 --end 2025-03-01
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy buy_and_hold_v1 --start 2025-01-01 --end 2025-03-01
```

Generate plots for one run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.plot_strategy_run --run-dir artifacts/strategies/momentum_v1_single_5accf9137182
```

Generate the report for one run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.generate_report --run-dir artifacts/strategies/momentum_v1_single_5accf9137182
```

Generate the bounded comparison leaderboard directly:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1 mean_reversion_v1 buy_and_hold_v1 --start 2025-01-01 --end 2025-03-01 --output_path artifacts/comparisons/strategy_comparison_example
```

## Example Artifact Paths

Validated run directories for the bounded example:

* `artifacts/strategies/momentum_v1_single_5accf9137182/`
* `artifacts/strategies/mean_reversion_v1_single_8bc603e60db6/`
* `artifacts/strategies/buy_and_hold_v1_single_5de12151d6b1/`

Representative per-run artifacts:

* `artifacts/strategies/momentum_v1_single_5accf9137182/plots/equity_curve.png`
* `artifacts/strategies/momentum_v1_single_5accf9137182/plots/drawdown.png`
* `artifacts/strategies/momentum_v1_single_5accf9137182/plots/rolling_sharpe_debug.png`
* `artifacts/strategies/momentum_v1_single_5accf9137182/plots/trade_return_distribution_debug.png`
* `artifacts/strategies/momentum_v1_single_5accf9137182/plots/win_loss_distribution_debug.png`
* `artifacts/strategies/momentum_v1_single_5accf9137182/report.md`

Comparison artifacts:

* `artifacts/comparisons/strategy_comparison_example/leaderboard.csv`
* `artifacts/comparisons/strategy_comparison_example/leaderboard.json`
* `artifacts/comparisons/strategy_comparison_example/plots/equity_comparison.png`
* `artifacts/comparisons/strategy_comparison_example/plots/metric_comparison_sharpe_ratio.png`
* `artifacts/comparisons/strategy_comparison_example/example_summary.json`

## Notes

This workflow stays reproducible because the selected strategies, parameters,
dataset, and date range are fixed. The resulting single-run artifact directories
are deterministic for the same saved inputs, and the report reuses plots from
the standardized `<run_dir>/plots/` directory when they already exist. The
single-run report embeds only the report-quality plots, while the `_debug`
artifacts remain available in the same `plots/` directory for deeper
inspection.
