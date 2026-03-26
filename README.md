# StratLake Trade Engine

StratLake Trade Engine is a deterministic research platform for running
systematic strategy and portfolio experiments on curated market data.

It is built for research review, not ingestion or live trading. The engine
consumes validated feature datasets, runs backtests with explicit execution
assumptions, applies layered validation, and writes auditable artifacts for
later comparison and portfolio construction.

## Overview

StratLake helps answer three practical research questions:

* Did the strategy respect temporal integrity and avoid same-bar execution?
* How do execution delay, transaction costs, slippage, and turnover affect the
  result?
* Can the output be trusted across metrics, QA summaries, manifests, and
  registry entries?

The repository currently supports:

* feature-dataset driven strategy research
* deterministic single-run and walk-forward evaluation
* execution realism through lag, transaction costs, and slippage
* turnover and execution-friction attribution
* portfolio construction from completed strategy artifacts
* portfolio-level validation and sanity checks
* strict-mode enforcement across strategy and portfolio CLIs
* unified runtime configuration with auditable persisted settings
* deterministic artifacts, manifests, and registry-backed reuse

## Research Validity & Execution Realism (Milestone 10)

Milestone 10 adds a repository-wide framework for making research outputs more
realistic, reviewable, and harder to misinterpret.

### Research-validity layers

* Temporal integrity checks validate ordering, uniqueness, signal alignment,
  and lagged execution assumptions before trusting results.
* Consistency validation checks saved artifacts against each other after
  persistence.
* Sanity checks flag suspicious return paths, implausible smoothness, extreme
  annualized metrics, and other outliers.
* Portfolio validation enforces exposure, leverage, sleeve-weight, and
  compounding constraints.

### Execution realism

* `execution_delay` controls how many bars signals are lagged before execution.
* `transaction_cost_bps` and `slippage_bps` apply deterministic friction to
  turnover-driven trades.
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
  portfolio-validation, and strict-mode settings.
* Precedence is deterministic: repository defaults < config < CLI.
* Effective runtime settings are persisted with completed runs for auditability.

See:

* [docs/research_validity_framework.md](docs/research_validity_framework.md)
* [docs/execution_model.md](docs/execution_model.md)
* [docs/strict_mode.md](docs/strict_mode.md)
* [docs/runtime_configuration.md](docs/runtime_configuration.md)

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

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

Single-run window:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --start 2022-01-01 --end 2023-01-01
```

Walk-forward evaluation:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

Strict mode with execution realism enabled:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --strict --execution-enabled --transaction-cost-bps 5 --slippage-bps 2
```

### 4. Run a portfolio

Registry-backed portfolio:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_equal --from-registry --timeframe 1D
```

Strict walk-forward portfolio evaluation:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name strict_valid_builtin_pair --from-registry --evaluation configs/evaluation.yml --timeframe 1D --strict
```

## Strict Mode

`--strict` enables unified strict-mode enforcement for both strategy and
portfolio workflows.

In practice:

* validation and sanity warnings that would otherwise be recorded can become
  blocking failures
* CLI commands exit with status code `1` on strict-mode failures
* flagged runs fail before persistence instead of producing review artifacts

Without `--strict`, the same run can still complete and persist artifacts with
`sanity_status: "warn"` or warning details in QA summaries.

See [docs/strict_mode.md](docs/strict_mode.md).

## Runtime Configuration

Runtime settings come from three layers:

1. repository defaults
2. config-provided runtime values
3. CLI overrides

Current defaults come from:

* [configs/execution.yml](configs/execution.yml)
* [configs/sanity.yml](configs/sanity.yml)
* code defaults for portfolio validation in `src/portfolio/contracts.py`

Supported runtime sections:

* `execution`
* `sanity`
* `portfolio_validation`
* `strict_mode`

Config may define those sections either at top level or under `runtime`.
Portfolio configs also support the legacy alias `validation`, which resolves to
the same portfolio-validation contract.

CLI flags override config values for execution settings and can force strict
mode:

* `--execution-delay`
* `--transaction-cost-bps`
* `--slippage-bps`
* `--execution-enabled`
* `--disable-execution-model`
* `--strict`

See [docs/runtime_configuration.md](docs/runtime_configuration.md).

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
* [docs/portfolio_construction_workflow.md](docs/portfolio_construction_workflow.md)

Milestone 10 references:

* [docs/research_validity_framework.md](docs/research_validity_framework.md)
* [docs/execution_model.md](docs/execution_model.md)
* [docs/strict_mode.md](docs/strict_mode.md)
* [docs/runtime_configuration.md](docs/runtime_configuration.md)

Supporting references:

* [docs/cli_strategy_runner.md](docs/cli_strategy_runner.md)
* [docs/research_integrity_and_qa.md](docs/research_integrity_and_qa.md)
* [docs/portfolio_configuration.md](docs/portfolio_configuration.md)
* [docs/strategy_comparison_cli.md](docs/strategy_comparison_cli.md)

## Repository Layout

```text
src/
  cli/        command-line entrypoints
  config/     execution, sanity, evaluation, and runtime config resolution
  portfolio/  portfolio construction, validation, QA, and artifacts
  research/   strategy execution, integrity checks, sanity, and reporting

configs/
  execution.yml
  sanity.yml
  strategies.yml
  portfolios.yml
  evaluation.yml

artifacts/
  strategies/
  portfolios/
```
