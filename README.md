# StratLake Trade Engine

StratLake Trade Engine is a deterministic research platform for running
systematic strategy and portfolio experiments on curated market data.

It is built for research review, not ingestion or live trading. The engine
consumes validated feature datasets, runs backtests with explicit execution
assumptions, applies layered validation, and writes auditable artifacts for
later comparison, portfolio construction, and registry-backed reuse.

## Milestone 11.5 Summary

Milestone 11.5 extends StratLake into a deterministic research stack that now
connects feature engineering, alpha modeling, backtesting, and portfolio
construction in one auditable workflow. The repository now supports:

* alpha model registration through a deterministic `BaseAlphaModel` interface
* deterministic alpha training and prediction helpers with explicit half-open
  time windows
* time-aware alpha split utilities for fixed and rolling train/predict windows
* cross-sectional helpers for same-timestamp alpha inspection
* continuous-signal backtesting where finite numeric exposures are interpreted
  literally after lagged execution
* centralized portfolio optimization with `equal_weight`, `max_sharpe`, and
  `risk_parity`
* operational volatility targeting in portfolio workflows, separate from
  diagnostic risk summaries
* deterministic return simulation, robustness analysis, artifact manifests,
  and registry-backed reuse

Start with:

* [docs/alpha_workflow.md](docs/alpha_workflow.md)
* [docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md)
* [docs/examples/milestone_12_alpha_portfolio_workflow.md](docs/examples/milestone_12_alpha_portfolio_workflow.md)

## Overview

StratLake helps answer three practical research questions:

* Did the strategy or portfolio respect temporal integrity and deterministic
  execution assumptions?
* How do optimizer choices, execution frictions, and risk diagnostics change
  the result?
* Can the output be trusted across metrics, QA summaries, manifests, and
  registry entries?

The repository currently supports:

* deterministic feature-dataset driven strategy and alpha research
* alpha model registration, training, prediction, and cross-sectional review
* single-run and walk-forward strategy evaluation
* continuous-signal or discrete-signal backtesting with lagged execution
* deterministic robustness analysis for strategy parameter sweeps
* execution realism through lag, transaction costs, fixed fees, and slippage
* portfolio construction from completed strategy artifacts or alpha-derived
  return sleeves
* centralized portfolio optimization, validation, risk summaries, and
  operational volatility targeting
* deterministic return simulation for strategy or portfolio outputs
* strict-mode enforcement across strategy and portfolio CLIs
* unified runtime configuration with auditable persisted settings
* deterministic artifacts, manifests, and registry-backed reuse

## Architecture

At a high level, StratLake is an artifact-driven research pipeline:

```text
features
    ->
alpha
    ->
train
    ->
predict
    ->
cross-section
    ->
signal
    ->
backtest
    ->
portfolio
    ->
risk
    ->
artifacts
```

In practice, the alpha and strategy layers can meet at different points in that
pipeline:

* traditional strategies can emit `signal` directly from a validated feature
  dataset
* alpha workflows can train on the same canonical frame, emit
  `prediction_score`, inspect cross-sections, then map those predictions into
  backtestable exposures
* completed strategy artifacts or alpha-derived sleeves can then flow into the
  portfolio layer for optimization, execution accounting, risk review, and
  persistence

This design keeps research and portfolio workflows deterministic, auditable,
and easy to review from saved files rather than only in-memory results.

## Alpha Modeling

The alpha layer lives under `src/research/alpha/` and provides a deterministic
interface for ML-style models that operate on canonical research frames sorted
by `(symbol, ts_utc)`.

### Alpha model interface

`BaseAlphaModel` enforces:

* stable `fit(df)` and `predict(df)` behavior
* no mutation of the caller's input frame
* prediction output aligned exactly to `df.index`
* numeric prediction scores with deterministic repeatability checks

Models are registered through `register_alpha_model(...)` and instantiated by
name through the alpha registry.

### Training workflow

`train_alpha_model(...)`:

* validates the canonical input contract
* resolves feature columns explicitly or from `feature_*` columns
* applies half-open training bounds `[train_start, train_end)`
* returns a `TrainedAlphaModel` with the fitted model plus metadata about the
  training slice

### Prediction workflow

`predict_alpha_model(...)`:

* validates the trained model contract
* applies half-open prediction bounds `[predict_start, predict_end)`
* preserves structural columns such as `symbol`, `ts_utc`, and optional
  `timeframe`
* returns a deterministic prediction frame with `prediction_score`

See [docs/alpha_workflow.md](docs/alpha_workflow.md) for the full workflow.

## Cross-Sectional Utilities

Alpha workflows often need to inspect one same-timestamp asset slice before
mapping predictions into signals or downstream portfolios.

The cross-sectional helpers in `src/research/alpha/cross_section.py` provide:

* `list_cross_section_timestamps(...)` for deterministic timestamp discovery
* `get_cross_section(...)` for one timestamp-specific multi-symbol slice
* `iter_cross_sections(...)` for ordered `(timestamp, frame)` iteration

These utilities validate sorted, duplicate-free `(symbol, ts_utc)` inputs so
cross-sectional review stays deterministic and auditable.

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

### 4. Run an alpha example workflow

```powershell
python docs/examples/milestone_12_alpha_portfolio_workflow.py
```

This example demonstrates alpha model registration, deterministic training and
prediction, fixed and rolling alpha splits, cross-sectional inspection,
continuous-signal backtesting, and portfolio construction with and without
volatility targeting.

### 5. Run a portfolio

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

Operational volatility targeting:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_targeted --from-registry --timeframe 1D
```

Walk-forward portfolio evaluation:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name strict_valid_builtin_pair --from-registry --evaluation configs/evaluation.yml --timeframe 1D --strict
```

The end-to-end Milestone 11 guide, including config snippets and output
interpretation, lives in
[docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md).

## Backtesting

The backtest layer accepts finite numeric `signal` values and interprets them
as literal lagged exposures.

That means:

* discrete signals such as `-1`, `0`, and `1` continue to work
* continuous signals such as alpha prediction scores are also supported
* return contribution scales proportionally with the executed exposure
* the runner does not clip or normalize exposure values implicitly

See [docs/backtest_runner.md](docs/backtest_runner.md).

## Portfolio Workflow

Portfolio research now includes:

* component selection from explicit run ids or the shared strategy registry
* aligned return-matrix construction with `intersection` alignment
* deterministic static allocation from `equal_weight`, `max_sharpe`, or
  `risk_parity`
* optional operational post-optimizer volatility targeting
* execution-friction accounting on turnover-driven rebalances
* portfolio metrics plus centralized risk summaries
* optional simulation artifacts for single-run portfolios
* walk-forward portfolio evaluation across deterministic splits
* manifest and registry rows that expose optimizer, execution, risk, and
  simulation metadata

### Volatility targeting

Portfolio config now supports:

```yaml
volatility_targeting:
  enabled: true
  target_volatility: 0.10
  lookback_periods: 20
  volatility_epsilon: 1e-8
```

Important distinction:

* `risk.target_volatility` is diagnostic and affects risk summaries
* top-level `volatility_targeting` is operational and scales base weights
  before execution accounting and portfolio evaluation

When enabled, the constructor computes a deterministic scaling factor from the
estimated pre-target portfolio volatility and applies it directly to the base
weights. When disabled, base optimizer weights flow through unchanged.

Start with:

* [docs/alpha_workflow.md](docs/alpha_workflow.md)
* [docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md)
* [docs/portfolio_configuration.md](docs/portfolio_configuration.md)
* [docs/portfolio_artifact_logging.md](docs/portfolio_artifact_logging.md)

## Example Workflow

The main end-to-end alpha example lives at
[docs/examples/milestone_12_alpha_portfolio_workflow.py](docs/examples/milestone_12_alpha_portfolio_workflow.py).

Run it with:

```powershell
python docs/examples/milestone_12_alpha_portfolio_workflow.py
```

It demonstrates:

* alpha model registration and deterministic train/predict helpers
* fixed and rolling alpha splits
* cross-sectional inspection
* sign-based and continuous-exposure signal interpretation
* single-symbol backtesting
* portfolio construction with and without volatility targeting
* artifact writing under `docs/examples/output/milestone_12_alpha_portfolio_workflow/`

See the companion guide
[docs/examples/milestone_12_alpha_portfolio_workflow.md](docs/examples/milestone_12_alpha_portfolio_workflow.md).

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
* [docs/alpha_workflow.md](docs/alpha_workflow.md)
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

Examples:

* [docs/examples/milestone_12_alpha_portfolio_workflow.md](docs/examples/milestone_12_alpha_portfolio_workflow.md)

Merge-readiness notes:

* [docs/milestone_10_merge_readiness.md](docs/milestone_10_merge_readiness.md)
* [docs/milestone_11_merge_readiness.md](docs/milestone_11_merge_readiness.md)

## Repository Layout

```text
src/
  cli/        command-line entrypoints
  config/     execution, runtime, evaluation, robustness, and simulation config
  portfolio/  construction, optimization, risk, validation, QA, and artifacts
  research/   alpha, strategy execution, integrity checks, robustness, simulation, and reporting

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
