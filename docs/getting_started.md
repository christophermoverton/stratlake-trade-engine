# Getting Started

## Overview

StratLake Trade Engine is a deterministic strategy and portfolio research
workflow built on curated market data. It does not ingest raw data. Instead,
it reads validated datasets, builds features, runs strategies, applies
execution and validation rules, and writes reproducible artifacts.

This guide walks through the current repository workflow:

1. set up the environment
2. point the repo at curated data
3. build or verify features
4. run a strategy
5. run the Milestone 11.5 alpha-to-portfolio example
6. run the Milestone 12 alpha-evaluation example
7. run the Milestone 13 review-and-promotion example
8. run a strict strategy
9. build a portfolio from saved runs

For deeper detail, continue with:

* [alpha_workflow.md](alpha_workflow.md)
* [strategy_evaluation_workflow.md](strategy_evaluation_workflow.md)
* [portfolio_construction_workflow.md](portfolio_construction_workflow.md)
* [research_validity_framework.md](research_validity_framework.md)

## Prerequisites

Before running the CLI workflows, make sure you have:

* Python 3.10 or newer
* curated data produced by the upstream ingestion repository
* permission to create a virtual environment and install dependencies
* a local `.env` file with the correct paths

Important environment settings:

* `MARKETLAKE_ROOT`
* `FEATURES_ROOT`
* `ARTIFACTS_ROOT`
* `DUCKDB_PATH`

Typical local values:

```text
MARKETLAKE_ROOT=/path/to/fintech-market-ingestion/data/curated
FEATURES_ROOT=data
ARTIFACTS_ROOT=artifacts
DUCKDB_PATH=:memory:
```

## Repository Setup

### 1. Create and activate a virtual environment

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

### 2. Install dependencies

```powershell
pip install -e .
pip install -e ".[dev]"
```

### 3. Create `.env`

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

Edit `.env` so `MARKETLAKE_ROOT` points to the curated market-data root.

## Build or Verify Features

The example strategies in [../configs/strategies.yml](../configs/strategies.yml)
use `features_daily`, so that dataset must already exist.

If features are missing or stale, build them:

```powershell
python -m cli.build_features --timeframe 1D --start 2022-01-01 --end 2024-01-01 --tickers configs/tickers_50.txt
```

Notes:

* `--start` is inclusive
* `--end` is exclusive
* daily features need enough history for rolling lookbacks

## Run a Strategy

Basic run:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

Bounded single-run window:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --start 2022-01-01 --end 2023-01-01
```

Walk-forward run:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

The strategy CLI prints:

* `strategy`
* `run_id`
* `cumulative_return`
* `sharpe_ratio`

It also writes artifacts under `artifacts/strategies/<run_id>/`.

## Run The Milestone 11.5 Alpha Example Workflow

The repository includes an end-to-end alpha example that demonstrates model
registration, deterministic train/predict helpers, cross-sectional inspection,
continuous-signal backtesting, and portfolio construction.

```powershell
python docs/examples/milestone_11_5_alpha_portfolio_workflow.py
```

See:

* [alpha_workflow.md](alpha_workflow.md)
* [examples/milestone_11_5_alpha_portfolio_workflow.md](examples/milestone_11_5_alpha_portfolio_workflow.md)
* [milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md)

## Run The Milestone 12 Alpha-Evaluation Example

Milestone 12 adds deterministic alpha evaluation before signal mapping,
including forward-return alignment, IC and Rank IC scoring, artifact
persistence, registry-backed tracking, and leaderboard comparison.

```powershell
python docs/examples/alpha_evaluation_end_to_end.py
```

See:

* [alpha_evaluation_workflow.md](alpha_evaluation_workflow.md)

## Run The Milestone 13 Review-And-Promotion Example

Milestone 13 adds a small committed end-to-end review example that loads
completed alpha, strategy, and portfolio artifacts from checked-in registries,
then writes one deterministic review pack plus a review-level promotion
decision.

```powershell
python docs/examples/milestone_13_review_promotion_workflow.py
```

See:

* [milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md)
* [examples/milestone_13_review_promotion_workflow.md](examples/milestone_13_review_promotion_workflow.md)
* [review_configuration.md](review_configuration.md)

## Run With Strict Mode And Execution Frictions

Example strict run with deterministic execution realism:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --strict --execution-enabled --transaction-cost-bps 5 --slippage-bps 2
```

This run:

* keeps lagged execution
* applies deterministic costs and slippage
* turns flagged sanity issues into blocking failures
* persists effective runtime settings with the run when it succeeds

See:

* [strict_mode.md](strict_mode.md)
* [execution_model.md](execution_model.md)
* [runtime_configuration.md](runtime_configuration.md)

## Run a Portfolio

The portfolio runner consumes completed strategy artifacts.

Registry-backed example:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_equal --from-registry --timeframe 1D
```

Strict walk-forward example:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name strict_valid_builtin_pair --from-registry --evaluation configs/evaluation.yml --timeframe 1D --strict
```

Portfolio artifacts are written under `artifacts/portfolios/<run_id>/`.

To exercise operational volatility targeting from the shipped config:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_targeted --from-registry --timeframe 1D
```

## What Gets Written

### Strategy runs

Common files:

* `config.json`
* `metrics.json`
* `signal_diagnostics.json`
* `qa_summary.json`
* `equity_curve.csv`
* `signals.parquet`
* `manifest.json`

### Portfolio runs

Common files:

* `config.json`
* `components.json`
* `weights.csv`
* `portfolio_returns.csv`
* `portfolio_equity_curve.csv`
* `metrics.json`
* `qa_summary.json`
* `manifest.json`

## Troubleshooting

### `MARKETLAKE_ROOT` is missing or wrong

Fix:

* copy `.env.example` to `.env`
* set `MARKETLAKE_ROOT` to the curated data directory
* restart the shell if needed

### `features_daily` is missing

Fix:

```powershell
python -m cli.build_features --timeframe 1D --start 2022-01-01 --end 2024-01-01 --tickers configs/tickers_50.txt
```

### Strategy run fails under strict mode

That usually means a sanity or validation issue that non-strict mode would have
recorded as a warning. Review:

* [strict_mode.md](strict_mode.md)
* [research_integrity_and_qa.md](research_integrity_and_qa.md)

## Next Docs

* [cli_strategy_runner.md](cli_strategy_runner.md)
* [strategy_comparison_cli.md](strategy_comparison_cli.md)
* [portfolio_construction_workflow.md](portfolio_construction_workflow.md)
* [research_validity_framework.md](research_validity_framework.md)
