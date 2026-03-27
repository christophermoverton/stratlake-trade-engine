# StratLake Trade Engine

StratLake Trade Engine is a deterministic research platform for running
systematic strategy and portfolio experiments on curated market data.

It is built for research review, not ingestion or live trading. The engine
consumes validated feature datasets, runs backtests with explicit execution
assumptions, applies layered validation, and writes auditable artifacts for
later comparison, portfolio construction, and registry-backed reuse.

## Milestone 11 Summary

Milestone 11 extends StratLake from a strategy backtesting repository into a
full deterministic research and portfolio system. The repository now supports:

* centralized portfolio optimization with `equal_weight`, `max_sharpe`, and
  `risk_parity`
* centralized portfolio risk summaries including rolling volatility, volatility
  targeting diagnostics, drawdown, historical VaR, and historical CVaR
* deterministic return simulation with bootstrap and normal Monte Carlo paths
* deterministic robustness analysis for strategy parameter sweeps
* portfolio-level execution realism with transaction costs, fixed fees, and
  configurable slippage models
* artifact manifests and registry rows that capture optimizer, risk,
  simulation, and execution-friction metadata
* CLI flows for optimizer-aware, risk-aware, simulation-enabled, and
  execution-aware portfolio workflows

The main Milestone 11 usage guide is
[docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md).

## Overview

StratLake helps answer three practical research questions:

* Did the strategy or portfolio respect temporal integrity and deterministic
  execution assumptions?
* How do optimizer choices, execution frictions, and risk diagnostics change
  the result?
* Can the output be trusted across metrics, QA summaries, manifests, and
  registry entries?

The repository currently supports:

* deterministic feature-dataset driven strategy research
* single-run and walk-forward strategy evaluation
* deterministic robustness analysis for strategy parameter sweeps
* execution realism through lag, transaction costs, fixed fees, and slippage
* portfolio construction from completed strategy artifacts
* centralized portfolio optimization and validation
* centralized portfolio risk summaries and diagnostics
* deterministic return simulation for strategy or portfolio outputs
* strict-mode enforcement across strategy and portfolio CLIs
* unified runtime configuration with auditable persisted settings
* deterministic artifacts, manifests, and registry-backed reuse

## Architecture

At a high level, StratLake is an artifact-driven research pipeline:

```text
validated features
    ->
signals
    ->
backtest with execution assumptions
    ->
metrics + QA + sanity
    ->
strategy artifacts + registry
    ->
portfolio loading and alignment
    ->
allocator / optimizer
    ->
portfolio returns + execution friction
    ->
portfolio risk summaries + optional simulation
    ->
portfolio artifacts + registry
```

This design keeps strategy and portfolio workflows deterministic, auditable,
and easy to review from saved files rather than only in-memory results.

## Research Validity, Runtime, And Strict Mode

StratLake uses shared runtime configuration and strict-mode enforcement across
strategy and portfolio workflows.

### Research-validity layers

* Temporal integrity checks validate ordering, uniqueness, signal alignment,
  and lagged execution assumptions before trusting results.
* Consistency validation checks saved artifacts against each other after
  persistence.
* Sanity checks flag suspicious return paths, implausible smoothness, extreme
  annualized metrics, and other outliers.
* Portfolio validation enforces weight-sum, exposure, leverage, sleeve-weight,
  and compounding constraints.

### Execution realism

* `execution_delay` controls how many bars signals are lagged before execution.
* `transaction_cost_bps`, `fixed_fee`, and `slippage_bps` apply deterministic
  execution drag to strategy and portfolio runs.
* Slippage models currently include `constant`, `turnover_scaled`, and
  `volatility_scaled`.
* Strategy and portfolio runs persist gross returns, net returns, turnover, and
  total execution friction in their artifacts.

### Strict mode

* `--strict` promotes flagged validation and sanity issues into fail-fast CLI
  errors.
* In strict mode, runs that fail pre-persistence validation do not write
  artifacts or registry entries.
* Non-strict mode still records warnings in metrics and QA artifacts so review
  remains auditable.

### Runtime configuration

* One normalized `runtime` contract resolves execution, sanity,
  portfolio-validation, risk, and strict-mode settings.
* Precedence is deterministic: repository defaults < config < CLI.
* Effective runtime settings are persisted with completed runs for auditability.

See:

* [docs/research_validity_framework.md](docs/research_validity_framework.md)
* [docs/execution_model.md](docs/execution_model.md)
* [docs/runtime_configuration.md](docs/runtime_configuration.md)
* [docs/strict_mode.md](docs/strict_mode.md)
* [docs/research_integrity_and_qa.md](docs/research_integrity_and_qa.md)

## Quick Start

### 1. Set up the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `MARKETLAKE_ROOT` in `.env` to the curated data root produced by the
upstream ingestion repository.

### 2. Build or verify features

```powershell
python -m cli.build_features --timeframe 1D --start 2022-01-01 --end 2024-01-01 --tickers configs/tickers_50.txt
```

### 3. Run a strategy

Single run:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

Walk-forward evaluation:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

Robustness analysis:

```powershell
python -m src.cli.run_strategy --robustness
```

Simulation-enabled single run:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --simulation path/to/simulation.yml
```

### 4. Run a portfolio

Baseline registry-backed portfolio:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_equal --from-registry --timeframe 1D
```

Optimizer-aware portfolio:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_equal --from-registry --timeframe 1D --optimizer-method max_sharpe
```

Risk-aware and execution-aware portfolio:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_equal --from-registry --timeframe 1D --risk-target-volatility 0.12 --risk-volatility-window 20 --execution-enabled --transaction-cost-bps 5 --fixed-fee 0.001 --slippage-bps 2 --slippage-model turnover_scaled
```

Walk-forward portfolio evaluation:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name strict_valid_builtin_pair --from-registry --evaluation configs/evaluation.yml --timeframe 1D --strict
```

The end-to-end Milestone 11 guide, including config snippets and output
interpretation, lives in
[docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md).

## Milestone 11 Portfolio Workflow

Portfolio research now includes:

* component selection from explicit run ids or the shared strategy registry
* aligned return-matrix construction with `intersection` alignment
* deterministic static allocation from `equal_weight`, `max_sharpe`, or
  `risk_parity`
* execution-friction accounting on turnover-driven rebalances
* portfolio metrics plus centralized risk summaries
* optional simulation artifacts for single-run portfolios
* walk-forward portfolio evaluation across deterministic splits
* manifest and registry rows that expose optimizer, execution, risk, and
  simulation metadata

Start with:

* [docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md)
* [docs/portfolio_configuration.md](docs/portfolio_configuration.md)
* [docs/portfolio_artifact_logging.md](docs/portfolio_artifact_logging.md)

## Artifact Overview

### Strategy artifacts

Successful strategy runs write under `artifacts/strategies/<run_id>/`.

Core files:

* `config.json`
* `metrics.json`
* `signal_diagnostics.json`
* `qa_summary.json`
* `equity_curve.csv`
* `signals.parquet`
* `manifest.json`

Optional Milestone 11 additions:

* `simulation/` for single-run simulation artifacts
* robustness runs under `artifacts/strategies/robustness/<run_id>/`

Walk-forward runs also include:

* `metrics_by_split.csv`
* `splits/<split_id>/...`

### Portfolio artifacts

Successful portfolio runs write under `artifacts/portfolios/<run_id>/`.

Core files:

* `config.json`
* `components.json`
* `weights.csv`
* `portfolio_returns.csv`
* `portfolio_equity_curve.csv`
* `metrics.json`
* `qa_summary.json`
* `manifest.json`

Optional Milestone 11 additions:

* manifest metadata for optimizer, risk, and execution-friction summaries
* `simulation/` for single-run simulation artifacts

Walk-forward portfolio runs also include:

* `aggregate_metrics.json`
* `metrics_by_split.csv`
* `splits/<split_id>/...`

See:

* [docs/experiment_artifact_logging.md](docs/experiment_artifact_logging.md)
* [docs/portfolio_artifact_logging.md](docs/portfolio_artifact_logging.md)

## Documentation Map

Start here:

* [docs/getting_started.md](docs/getting_started.md)
* [docs/strategy_evaluation_workflow.md](docs/strategy_evaluation_workflow.md)
* [docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md)

Portfolio references:

* [docs/portfolio_construction_workflow.md](docs/portfolio_construction_workflow.md)
* [docs/portfolio_configuration.md](docs/portfolio_configuration.md)
* [docs/portfolio_artifact_logging.md](docs/portfolio_artifact_logging.md)

Research integrity and execution references:

* [docs/research_validity_framework.md](docs/research_validity_framework.md)
* [docs/execution_model.md](docs/execution_model.md)
* [docs/runtime_configuration.md](docs/runtime_configuration.md)
* [docs/strict_mode.md](docs/strict_mode.md)
* [docs/research_integrity_and_qa.md](docs/research_integrity_and_qa.md)

Merge-readiness notes:

* [docs/milestone_10_merge_readiness.md](docs/milestone_10_merge_readiness.md)
* [docs/milestone_11_merge_readiness.md](docs/milestone_11_merge_readiness.md)

## Repository Layout

```text
src/
  cli/        command-line entrypoints
  config/     execution, runtime, evaluation, robustness, and simulation config
  portfolio/  construction, optimization, risk, validation, QA, and artifacts
  research/   strategy execution, integrity checks, robustness, simulation, and reporting

configs/
  evaluation.yml
  execution.yml
  portfolios.yml
  robustness.yml
  sanity.yml
  strategies.yml

artifacts/
  strategies/
  portfolios/
```
