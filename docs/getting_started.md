# Getting Started

## Overview

StratLake Trade Engine is a research workflow for running rule-based strategies on curated market data. It does not ingest raw data itself. Instead, it reads validated datasets from a separate curated data source, builds feature datasets, runs strategies, computes metrics, and saves reproducible artifacts.

This guide walks you through the Milestone 5 workflow:

1. Set up the repository and Python environment
2. Point the project at curated data
3. Build or verify the `features_daily` dataset
4. Run a single strategy
5. Compare multiple strategies
6. Inspect the resulting artifacts

If you want deeper detail after this guide, start with [strategy_evaluation_workflow.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/strategy_evaluation_workflow.md).

## Prerequisites

Before you run anything, make sure you have:

- Python 3.10+ available in your shell
- access to the curated data produced by `fintech-market-ingestion`
- permission to create a virtual environment and install package dependencies
- a local `.env` file with the correct repository paths

Important environment settings:

- `MARKETLAKE_ROOT`: required, points to the curated data root from the ingestion repo
- `FEATURES_ROOT`: where StratLake reads and writes feature datasets
- `ARTIFACTS_ROOT`: where StratLake writes run outputs
- `DUCKDB_PATH`: optional, defaults to `:memory:`

In the current repository, the practical defaults are:

```text
MARKETLAKE_ROOT=C:/path/to/fintech-market-ingestion/data/curated
FEATURES_ROOT=data
ARTIFACTS_ROOT=artifacts
DUCKDB_PATH=:memory:
```

With `FEATURES_ROOT=data`, the strategy runner will look for `features_daily` under `data/curated/features_daily`, which matches the current feature writer behavior.

## Repository Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd stratlake-trade-engine
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e .
pip install -e ".[dev]"
```

### 4. Create your `.env`

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

Then edit `.env` so `MARKETLAKE_ROOT` points to your curated data directory.

Example:

```text
MARKETLAKE_ROOT=C:/Users/yourname/dev/fintech-market-ingestion/data/curated
FEATURES_ROOT=data
ARTIFACTS_ROOT=artifacts
DUCKDB_PATH=:memory:
LOG_LEVEL=INFO
DEFAULT_TIMEZONE=UTC
```

## Data and Feature Preparation

Strategies in Milestone 5 run on feature datasets, not raw bars. The current example strategies in `configs/strategies.yml` all use:

```text
features_daily
```

That means `python -m src.cli.run_strategy --strategy momentum_v1` expects the `features_daily` dataset to already exist and be readable from `FEATURES_ROOT`.

### Option A: Use an existing `features_daily` dataset

If your workspace already contains:

```text
data/curated/features_daily/
```

you can usually move on to running strategies.

Current project convention for daily features:

```text
data/curated/features_daily/
  symbol=<SYMBOL>/
    year=<YYYY>/
      part-0.parquet
```

### Option B: Build `features_daily`

If `features_daily` is missing or stale, build it from curated daily bars:

```powershell
python -m cli.build_features --timeframe 1D --start 2022-01-01 --end 2024-01-01 --tickers configs/tickers_50.txt
```

What this command does:

- loads curated daily bars from `MARKETLAKE_ROOT`
- computes the daily feature set
- writes parquet outputs under `data/curated/features_daily`
- writes a feature-run summary under `artifacts/feature_runs/<run_id>/summary.json`

Notes:

- `--start` is inclusive
- `--end` is exclusive
- the ticker file should contain one symbol per line
- daily features need enough history for rolling calculations, so choose a date range wide enough for lookback windows

### Configuration expectations

The repository currently relies on `.env` for path configuration. `configs/paths.yml` is optional and is not present in this workspace, so a missing or incorrect `.env` is the most common setup problem.

## Run a Single Strategy

Start with a simple strategy run:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

You can also run a bounded date window in single-run mode:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --start 2022-01-01 --end 2023-01-01
```

What this does:

- loads the `momentum_v1` config from `configs/strategies.yml`
- loads the `features_daily` dataset
- generates trading signals
- runs the backtest
- computes summary metrics
- saves a new experiment directory under `artifacts/strategies/<run_id>/`

The CLI prints a short summary including:

- `strategy`
- `run_id`
- `cumulative_return`
- `sharpe_ratio`

## Run a Strategy Comparison

Compare the three Milestone 5 example strategies:

```powershell
python -m src.cli.compare_strategies --strategies momentum_v1 mean_reversion_v1 buy_and_hold_v1
```

What this does:

- executes each listed strategy with the current pipeline
- ranks them by `sharpe_ratio` by default
- writes a comparison leaderboard

Current default output location:

```text
artifacts/comparisons/<comparison_id>/
  leaderboard.csv
  leaderboard.json
```

`leaderboard.csv` is useful for quick spreadsheet review. `leaderboard.json` keeps the same results in a machine-readable format, along with the selected metric and comparison mode.

You can also rank by a different metric:

```powershell
python -m src.cli.compare_strategies --strategies momentum_v1 mean_reversion_v1 buy_and_hold_v1 --metric total_return
```

## Understand the Outputs

### Individual strategy runs

Each successful single strategy run creates:

```text
artifacts/strategies/<run_id>/
```

Common files:

- `config.json`: the strategy name, dataset, parameters, and run window
- `metrics.json`: summary metrics such as `total_return`, `sharpe_ratio`, and `max_drawdown`
- `signals.parquet`: the strategy input rows plus generated signals
- `equity_curve.csv`: a simple time series of equity and strategy return values
- `equity_curve.parquet`: backtest-oriented output for deeper inspection
- `trades.parquet`: closed-trade summaries when the run produced trades
- `manifest.json`: a compact run inventory and summary

### Comparison runs

Each comparison run creates:

```text
artifacts/comparisons/<comparison_id>/
```

Common files:

- `leaderboard.csv`: one row per strategy, ranked by the selected metric
- `leaderboard.json`: the same leaderboard plus comparison metadata

The leaderboard includes fields such as:

- `rank`
- `strategy_name`
- `evaluation_mode`
- `selected_metric_name`
- `selected_metric_value`
- `cumulative_return`
- `total_return`
- `sharpe_ratio`
- `max_drawdown`
- `annualized_return`
- `annualized_volatility`
- `hit_rate`
- `profit_factor`
- `turnover`
- `exposure_pct`

## Reproducibility

Milestone 5 strategy examples are designed to be reproducible when the inputs are unchanged.

What that means in practice:

- repeated runs of `python -m src.cli.run_strategy --strategy momentum_v1` produce the same metrics and the same saved artifact contents
- repeated runs of `python -m src.cli.compare_strategies --strategies momentum_v1 mean_reversion_v1 buy_and_hold_v1` produce the same leaderboard ordering and the same `leaderboard.csv` and `leaderboard.json` contents
- saved strategy run directories use a deterministic `run_id` derived from the run inputs and normalized results instead of a wall-clock timestamp

How reproducibility is enforced:

- feature datasets are loaded with explicit ordering and normalized again before persistence
- strategy groupby-based calculations operate on deterministic row order
- comparison outputs exclude volatile fields such as per-run `run_id` values
- metrics are serialized with stable JSON key ordering
- seeded randomness is only reproducible when a strategy provides an explicit fixed seed
- registry entries are updated by deterministic `run_id`, so rerunning the same experiment refreshes the same logical record instead of appending a new timestamp-only variant

Assumptions:

- the curated source data and feature parquet files are unchanged between runs
- the same strategy config, date bounds, and evaluation config are used
- the same Python environment and dependency versions are used

Known limitations:

- if the curated input data changes, the deterministic `run_id` and saved artifacts may also change because the normalized results change
- registry-backed selection still depends on the set of saved runs available in `artifacts/strategies/registry.jsonl`
- reproducibility guarantees cover deterministic execution and persisted contents, not cross-platform floating-point identity across different Python, pandas, or parquet-engine versions

## Read the Example Reports

After your first runs, these example reports are good references:

- [momentum_v1.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/examples/momentum_v1.md)
- [mean_reversion_v1.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/examples/mean_reversion_v1.md)
- [comparison_case_study.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/examples/comparison_case_study.md)

They show how to interpret strategy behavior, metrics, and comparison results using repository artifacts.

## Troubleshooting

### `MARKETLAKE_ROOT must be set`

Cause: `.env` is missing, not loaded, or points to the wrong place.

Fix:

- create `.env` from `.env.example`
- set `MARKETLAKE_ROOT` to the curated data directory from `fintech-market-ingestion`
- restart your shell after editing the file if needed

Expected path shape:

```text
<marketlake_root>/
  bars_daily/
  bars_1m/
```

### `features_daily` is missing

Cause: strategy execution depends on engineered features, and they have not been built yet.

Fix:

```powershell
python -m cli.build_features --timeframe 1D --start 2022-01-01 --end 2024-01-01 --tickers configs/tickers_50.txt
```

Then confirm that this directory exists:

```text
data/curated/features_daily/
```

### The feature path is wrong

Cause: `FEATURES_ROOT` does not match where features were written.

Fix:

- if you use the repository defaults, keep `FEATURES_ROOT=data`
- make sure the feature writer output and loader input agree
- for the current codebase, `FEATURES_ROOT=data` means `features_daily` is read from `data/curated/features_daily`

### Comparison results look outdated or inconsistent

Cause: you rebuilt features or changed data inputs, but you are comparing against older run artifacts or registry entries.

Fix:

- rerun the individual strategies after rebuilding features
- rerun the comparison so all strategies are scored from the same dataset state
- if you intentionally want old results, use the registry-backed workflow documented in [strategy_comparison_cli.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/strategy_comparison_cli.md)

### A comparison fails after data changes

Cause: comparison executes fresh strategy runs by default, so it depends on the current `features_daily` dataset being valid for every listed strategy.

Fix:

- rebuild `features_daily`
- rerun a single strategy first to confirm the dataset is healthy
- then rerun the multi-strategy comparison

## Next Steps

Once you can run the commands in this guide successfully, the next useful docs are:

- [cli_strategy_runner.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/cli_strategy_runner.md)
- [strategy_comparison_cli.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/strategy_comparison_cli.md)
- [strategy_evaluation_workflow.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/strategy_evaluation_workflow.md)
