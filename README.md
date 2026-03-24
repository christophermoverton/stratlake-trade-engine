# StratLake Trade Engine

StratLake is a systematic trade research engine designed to operate on
curated market data produced by the `fintech-market-ingestion` repository.

This repository intentionally does **not** perform ingestion.
It consumes validated, QA-enforced OHLCV data from an external marketlake.

---

## Milestone Summary

StratLake now spans multiple connected milestones across repository
foundation, feature engineering, research execution, evaluation, and
artifact-driven review workflows.

Milestone 7 is now complete in
`v0.7.0-visualization-reporting`: Visualization & Reporting Refinement.
This release standardizes plotting and report generation on top of existing
saved run artifacts, making research outputs easier to review without changing
core strategy logic or artifact schemas.

The repository now supports:

* standardized strategy interfaces
* YAML-backed strategy configuration registry
* signal generation over engineered feature datasets
* deterministic backtest execution with lagged signal application
* performance metrics for experiment evaluation, including risk-adjusted and trade-activity metrics
* file-based experiment artifact logging
* append-only experiment registry tracking under `artifacts/strategies/registry.jsonl`
* a CLI strategy runner for end-to-end research execution
* a comparison CLI for multi-strategy leaderboards from fresh runs or registry-backed results
* standardized plotting utilities for single-run and comparison visualizations
* deterministic Markdown report generation from saved research artifacts
* report-quality vs debug artifact separation for saved plots and reports
* deterministic test coverage for visualization and reporting artifacts

### Milestone 7: Visualization & Reporting Refinement

Milestone 7 adds a lightweight presentation layer to the existing
artifact-driven research workflow:

* standardized plotting utilities for consistent saved figure output
* improved strategy comparison and equity visualizations
* structured Markdown report generation from saved run artifacts
* clear separation between report-quality plots and debug-only artifacts
* deterministic tests covering visualization and reporting outputs

### Start Here

* [docs/strategy_evaluation_workflow.md](docs/strategy_evaluation_workflow.md)
* [docs/research_visualization_workflow.md](docs/research_visualization_workflow.md)
* [docs/examples/strategy_comparison_example.md](docs/examples/strategy_comparison_example.md)

---

## Architecture Principle

StratLake is a **consumer-layer research engine**.

```text
fintech-market-ingestion
|-- data/curated/
v
stratlake-trade-engine
```

The ingestion repository owns:
- Raw data
- Normalization
- Deduplication
- QA validation
- Partitioned Parquet outputs

StratLake owns:
- Feature engineering
- Research experimentation
- Backtesting logic
- Reproducible research artifacts

---

## Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -e .
pip install -e ".[dev]"
```

### 3. Configure `.env`

Create a `.env` file from `.env.example`.

Important variables:

```text
MARKETLAKE_ROOT=/path/to/fintech-market-ingestion/data/curated
DUCKDB_PATH=/path/to/stratlake.duckdb
FEATURES_ROOT=/path/to/output/features
LOG_LEVEL=INFO
DEFAULT_TIMEZONE=UTC
```

`MARKETLAKE_ROOT` should point to the curated output directory produced by the ingestion repository.

---

## Configuration Design

StratLake uses a layered configuration system:

1. Environment variables (`.env`)
2. YAML configuration files (`configs/`)
3. Code defaults

Environment variables override YAML.

Tests disable `.env` auto-loading to ensure deterministic behavior.

---

## Repository Structure

```text
src/
  config/        # Environment + YAML settings loader
  data/          # Data access layer + feature writer
  features/      # Feature engineering
  pipeline/      # Load -> compute -> write orchestration
  cli/           # Research CLI modules
  research/      # Strategy interfaces, signals, and backtest execution

cli/             # Command-line entrypoints

configs/
  strategies.yml # Strategy registry for research experiments
  evaluation.yml # Evaluation split definitions for walk-forward validation
  paths.yml
  universe.yml
  features.yml

tests/           # Unit tests

artifacts/
  strategies/    # Strategy experiment outputs (not versioned)
  feature_runs/  # Feature pipeline run summaries (not versioned)
data/            # Locally generated data (not versioned)
```

---

## Current Status

Current implementation includes:

* Configuration loading with environment overrides
* DuckDB-backed curated bar loaders
* Consumer-side contract validation
* Daily and 1-minute feature computation
* Feature parquet writing
* Pipeline orchestration entry points
* CLI entrypoint for feature builds and run summaries
* Research-layer signal generation, deterministic backtest execution, and performance metrics
* CLI strategy runner for full experiment execution and artifact persistence
* Walk-forward evaluation across deterministic train/test splits
* Baseline benchmark strategies for buy-and-hold, SMA crossover, and seeded random references
* Strategy comparison and leaderboard generation from fresh runs or registry-backed selection
* Artifact-driven plot and Markdown report generation from saved run artifacts
* Standardized report-quality and debug visualization artifacts for review workflows

---

## Research Layer

The completed Strategy Research Layer provides the core building blocks needed
to move from feature datasets to systematic signals, deterministic backtests,
standardized metrics, and reproducible experiment artifacts:

* `BaseStrategy` for standardized strategy interfaces
* `configs/strategies.yml` for deterministic strategy configuration
* `generate_signals()` for attaching validated `signal` outputs to a dataset
* `run_backtest()` for converting lagged signals into realized strategy returns
* performance metrics for evaluating the resulting `strategy_return` series, trade outcomes, and market exposure
* `save_experiment()` for persisting reproducible experiment artifacts to disk
* `load_registry()` plus lightweight registry filters for querying prior runs

The current research modules are intentionally minimal and reproducible. They
operate on feature datasets already produced by the analytics layer, use the
previous period's signal to avoid look-ahead bias, compound returns into an
`equity_curve`, and compute deterministic summary statistics from the resulting
strategy return series.

The research package now also includes artifact-driven visualization and
reporting helpers. These layers consume saved run outputs such as
`metrics.json`, `equity_curve.csv`, and optional trade artifacts to generate
deterministic plot files and a structured Markdown `report.md` without
changing the underlying research results.

The research layer also includes a CLI entrypoint that runs the full experiment
flow from a strategy name in `configs/strategies.yml`, including dataset load,
signal generation, backtest execution, metric calculation, artifact logging,
and console summary output.

The current evaluation layer now also includes deterministic baseline strategies that run
through the same interface and execution path as research strategies:

* `buy_and_hold_v1`
* `sma_crossover_v1`
* `seeded_random_v1`

The evaluation workflow includes deterministic split generation via
`configs/evaluation.yml` and `src/research/splits.py`. Split windows use
half-open date intervals: `start` is inclusive and `end` is exclusive.

### Strategy Evaluation

The current research workflow extends the single-run execution path into a full
evaluation and comparison workflow:

* single-run execution with optional date filters
* walk-forward execution from [configs/evaluation.yml](configs/evaluation.yml)
* deterministic evaluation splits in `fixed`, `rolling`, and `expanding` modes
* baseline benchmarks through the same runner and metric stack
* file-based artifacts plus append-only registry tracking
* multi-strategy comparison with leaderboard outputs
* deterministic plots and Markdown reports derived from saved run artifacts
* report-quality vs debug plot separation for saved review outputs

The end-to-end flow is:

```text
feature datasets -> signals -> backtest -> metrics -> artifacts -> registry -> comparison
```

Visualization and reporting extend that artifact flow rather than replacing it:

```text
feature datasets -> signals -> backtest -> metrics -> artifacts -> plots -> report.md
```

Start with the central workflow guide:

* [docs/strategy_evaluation_workflow.md](docs/strategy_evaluation_workflow.md)

Focused references:

* [docs/cli_strategy_runner.md](docs/cli_strategy_runner.md)
* [docs/walk_forward_strategy_runner.md](docs/walk_forward_strategy_runner.md)
* [docs/strategy_comparison_cli.md](docs/strategy_comparison_cli.md)
* [docs/strategy_performance_metrics.md](docs/strategy_performance_metrics.md)
* [docs/experiment_artifact_logging.md](docs/experiment_artifact_logging.md)
* [docs/evaluation_split_configuration.md](docs/evaluation_split_configuration.md)
* [docs/research_visualization_workflow.md](docs/research_visualization_workflow.md)
* [docs/examples/strategy_comparison_example.md](docs/examples/strategy_comparison_example.md)

### Research Flow

```text
feature datasets -> signals -> backtest -> metrics -> artifacts
```

Pipeline mapping in code:

```text
load_features(...)
        ->
strategy.generate_signals(...)
        ->
signal_engine.generate_signals(...)
        ->
backtest_runner.run_backtest(...)
        ->
metrics.{total_return, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, hit_rate, profit_factor, turnover, exposure_pct}
        ->
experiment_tracker.save_experiment(...)
        ->
artifacts/strategies/<run_id>/
        +
artifacts/strategies/registry.jsonl
```

### Backtest Formula

```python
strategy_return = signal.shift(1).fillna(0.0) * asset_return
equity_curve = (1.0 + strategy_return).cumprod()
```

### Example Usage

```python
from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import save_experiment
from src.research.metrics import compute_performance_metrics
from src.research.reporting import generate_strategy_report
from src.research.signal_engine import generate_signals

signals_df = generate_signals(features_df, strategy)
backtest_df = run_backtest(signals_df)

metrics = compute_performance_metrics(backtest_df)

artifact_dir = save_experiment(
    strategy_name=strategy.name,
    results_df=backtest_df,
    metrics=metrics,
    config={"lookback": 20},
)

report_path = generate_strategy_report(artifact_dir)
```

See [docs/backtest_runner.md](docs/backtest_runner.md)
for the input contract, supported return columns, and expected outputs.
See [docs/strategy_evaluation_workflow.md](docs/strategy_evaluation_workflow.md)
for the end-to-end evaluation and benchmarking workflow.
See [docs/strategy_performance_metrics.md](docs/strategy_performance_metrics.md)
for the available metrics, formulas, and annualization behavior.
See [docs/experiment_artifact_logging.md](docs/experiment_artifact_logging.md)
for the artifact layout, registry schema, and saved files.
See [docs/research_visualization_workflow.md](docs/research_visualization_workflow.md)
for how saved run artifacts become plots and a deterministic Markdown report.
See [docs/strategy_comparison_cli.md](docs/strategy_comparison_cli.md)
for comparison modes, leaderboard ordering, and output artifacts.

### Visualization And Reporting

Saved strategy artifacts can be turned into deterministic plots and a Markdown
report without rerunning the strategy itself. These layers extend the existing
research outputs; they do not replace metrics, artifacts, or evaluation.

The workflow is artifact-driven: saved run files remain the source of truth,
plot generation is standardized, and reports embed only report-quality plots
while leaving debug artifacts available in the same run directory for deeper
inspection.

High-level CLI commands:

```text
plot_strategy_run --run-dir artifacts/strategies/<run_id>
generate_report --run-dir artifacts/strategies/<run_id>
```

Repository module entrypoints:

```powershell
.\.venv\Scripts\python.exe -m src.cli.plot_strategy_run --run-dir artifacts/strategies/<run_id>
.\.venv\Scripts\python.exe -m src.cli.generate_report --run-dir artifacts/strategies/<run_id>
```

Standardized outputs:

* `artifacts/strategies/<run_id>/plots/`
* `artifacts/strategies/<run_id>/report.md`

`generate_report` also supports `--output-path <path>` for a custom Markdown
destination while still keeping plot artifacts under `<run_dir>/plots/`.

Current single-run plot artifacts distinguish report-quality and debug outputs:

* report-quality: `equity_curve.png`, `drawdown.png`
* debug-only: `rolling_sharpe_debug.png`,
  `trade_return_distribution_debug.png`,
  `win_loss_distribution_debug.png`

Generated reports use a consistent lightweight structure with a run header,
configuration summary, key metrics table, report-quality visualizations, a
trade summary, a short interpretation section, and relative links back to saved
artifacts.

Visualization and reporting behavior is covered by deterministic tests so plot
and report outputs remain stable as the workflow evolves.

See [docs/research_visualization_workflow.md](docs/research_visualization_workflow.md)
for the workflow, supported plots, artifact layout, and reporting behavior.
See [docs/examples/strategy_comparison_example.md](docs/examples/strategy_comparison_example.md)
for the bounded comparison workflow that layers comparison plots on top of
saved run artifacts.

### Strategy CLI Usage

Run a configured strategy experiment from the command line:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

Example with an explicit research window:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --start 2025-01-01 --end 2025-04-01
```

Required argument:

* `--strategy` -> strategy name from `configs/strategies.yml`

Optional arguments:

* `--start` -> inclusive date filter in `YYYY-MM-DD`
* `--end` -> exclusive date filter in `YYYY-MM-DD`
* `--evaluation [PATH]` -> run walk-forward evaluation using
  `configs/evaluation.yml` or a provided config path

Console summary output includes:

* strategy name
* experiment run identifier
* cumulative return
* Sharpe ratio
* split count when evaluation mode is enabled

Compare multiple strategies with a shared leaderboard:

```powershell
python -m src.cli.compare_strategies --strategies momentum_v1 mean_reversion_v1
```

Optional arguments:

* `--evaluation [PATH]` -> compare walk-forward runs using the default or provided evaluation config
* `--metric` -> ranking metric, defaults to `sharpe_ratio`
* `--top_k` -> keep only the top `N` leaderboard rows
* `--from_registry` -> use prior runs from `artifacts/strategies/registry.jsonl` instead of executing
* `--output_path` -> override the default `artifacts/strategies/leaderboard.csv` location

Registry mode selects the latest matching run per strategy after filtering by
evaluation mode. If `--evaluation` is provided, it also filters by the stored
evaluation config path. "Latest" is determined by descending `timestamp`, then
descending `run_id`.

Leaderboard ranking sorts by the selected metric in descending order, places
missing metric values last, and breaks ties by `strategy_name` then `run_id`.

Saved leaderboard artifacts:

* `artifacts/strategies/leaderboard.csv`
* `artifacts/strategies/leaderboard.json`

See [docs/strategy_comparison_cli.md](docs/strategy_comparison_cli.md)
for the full comparison CLI reference, selection rules, and examples.

Metrics artifacts and walk-forward summaries also include:

* `total_return` and legacy-compatible `cumulative_return`
* `annualized_return`
* `annualized_volatility`
* `max_drawdown`
* trade-level `hit_rate` and `profit_factor`
* activity metrics `turnover` and `exposure_pct`

Walk-forward example:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

Baseline examples:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy buy_and_hold_v1
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy sma_crossover_v1 --evaluation
```

See [docs/baseline_strategies.md](docs/baseline_strategies.md)
for the benchmark intent, parameter definitions, and config examples.

Current release outcome:

* StratLake can now execute single-run and walk-forward strategy evaluations,
  benchmark against deterministic baselines, persist registry-backed artifacts,
  compare strategies through a shared leaderboard workflow, and generate
  deterministic visual and Markdown summaries from saved run outputs.

---

## Feature Engineering Layer

StratLake's feature engineering layer transforms curated market data into
analytics-ready feature datasets for research and downstream modeling.

Current feature datasets:

* `features_daily`
* `features_1m`

These datasets follow the shared feature dataset contract documented in
[docs/feature_dataset_contract.md](docs/feature_dataset_contract.md).

### Architecture Overview

```text
Alpaca market data
        v
curated bars datasets
        v
feature engineering pipeline
        v
daily and minute feature datasets
        v
analytics / research workflows
```

### Feature Groups

The current engineered feature surface is organized around a few recurring
feature groups:

* momentum features: multi-horizon returns such as short, medium, and longer lookback returns
* volatility features: rolling and realized volatility features derived from recent price behavior
* trend features: moving averages and price-vs-trend deviation measures
* liquidity/activity features: normalized volume and trading activity metrics

The repository-level feature registry in
[configs/features.yml](configs/features.yml)
documents the currently implemented feature sets, their source datasets, and
their engineered columns.

### Feature Pipeline

The feature pipeline runs strictly downstream of curated bars datasets. In
practice, the flow is:

* load validated curated `bars_daily` or `bars_1m` datasets
* compute deterministic engineered features for each `(symbol, ts_utc, timeframe)` row
* write partitioned Parquet feature outputs
* emit QA summaries for observability and downstream checks

This keeps feature engineering separate from ingestion while ensuring analytics
consumers work from validated, reproducible inputs.

### Dataset Layout

Feature datasets are designed to be consumed from partitioned Parquet storage
using a canonical layout:

```text
data/features/<dataset>/symbol=<SYMBOL>/date=<YYYY-MM-DD>/*.parquet
```

The feature loaders and DuckDB view helpers support this contract layout via
`FEATURES_ROOT`. They also remain compatible with the repository's current
feature output conventions so existing pipelines continue to work while exposing
the same logical datasets.

### Analytics Access

Feature datasets can be accessed in two main ways:

* `load_features()` utilities for pandas-based analytics and research workflows
* DuckDB feature views for direct SQL access over partitioned Parquet datasets

See the related documentation for details:

* [docs/load_features.md](docs/load_features.md) for loader usage and DuckDB feature views
* [docs/dataset_qa_metrics.md](docs/dataset_qa_metrics.md) for QA summary outputs and checks
* [docs/feature_metadata_export.md](docs/feature_metadata_export.md) for the feature metadata JSON export, included fields, and generation behavior
* [docs/feature_dataset_contract.md](docs/feature_dataset_contract.md) for schema, naming, and behavioral guarantees
* [configs/features.yml](configs/features.yml) for the feature registry and source dataset mapping

---

# Data Access Layer (Issue 2)

The Trade Analysis Engine reads directly from the curated MarketLake parquet datasets without re-running ingestion.

This layer provides:

* A centralized dataset catalog
* DuckDB-backed views over partitioned parquet
* Symbol + date range loaders
* Environment-driven defaults for curated root and DuckDB path
* Canonical schema guarantees
* Safe handling of empty datasets

---

## Curated Dataset Layout

Curated data follows Hive-style partitioning:

```text
data/curated/
  bars_daily/
    symbol=XYZ/
      date=YYYY-MM-DD/
        *.parquet
  bars_1m/
    symbol=XYZ/
      date=YYYY-MM-DD/
        *.parquet
```

Partition columns:

* `symbol`
* `date`

---

## Canonical Schema

All loaders return a DataFrame with:

| Column    | Type                |
| --------- | ------------------- |
| symbol    | string              |
| ts_utc    | datetime64          |
| open      | float               |
| high      | float               |
| low       | float               |
| close     | float               |
| volume    | int                 |
| source    | string              |
| timeframe | string              |
| date      | string (YYYY-MM-DD) |

Column order is deterministic.

---

## DuckDB Views

On load, DuckDB views are created dynamically:

```sql
CREATE VIEW bars_1m AS
SELECT * FROM read_parquet(
  'data/curated/bars_1m/symbol=*/date=*/*.parquet'
);
```

If no parquet files exist:

* An empty view with canonical schema is created
* Loaders return an empty DataFrame (no crash)

This makes the system:

* Test-safe
* Backfill-safe
* Incremental ingestion safe

---

## Loaders

Location:

```text
src/data/catalog.py
src/data/loaders.py
```

### Load Daily Bars

```python
from src.data.loaders import load_bars_daily

df = load_bars_daily(
    symbols=["AAPL", "MSFT"],
    start_date="2025-11-01",
    end_date="2025-12-01",
)
```

### Load 1-Minute Bars

```python
from src.data.loaders import load_bars_1m

df = load_bars_1m(
    symbols=["AAPL"],
    start_date="2025-11-15",
    end_date="2025-11-16",
)
```

If `paths` is omitted, loaders default to `MARKETLAKE_ROOT`. If `con` is omitted, DuckDB connects using `DUCKDB_PATH` and falls back to `:memory:`.

---

## Date Semantics

* `start_date` -> inclusive
* `end_date` -> **exclusive**

This prevents overlap when chaining time windows:

```text
[2025-11-01, 2025-12-01)
```

---

## Testing

Tests validate:

* Canonical schema enforcement
* Symbol filtering
* Date filtering
* End-date exclusivity
* Empty dataset handling
* Environment-driven loader defaults
* Pipeline orchestration and writer invocation
* CLI dispatch, metadata emission, and summary artifact creation

Run:

```powershell
python -m pytest -q
```

---

## Design Philosophy

This layer is:

* Read-only (no ingestion logic)
* Partition-aware
* DuckDB-native
* Deterministic
* Production-safe

It decouples:

Ingestion -> Storage -> Query -> Strategy Engine

Which keeps the trade engine modular and testable.

---

## Project Structure

This project follows a modern `src/` layout:

```text
stratlake-trade-engine/
|
|-- src/
|   |-- data/
|   |-- config/
|   |-- features/
|   |-- pipeline/
|   `-- ...
|
|-- tests/
|-- pyproject.toml
`-- .venv/
```

The `src/` directory is a proper package root (`src/__init__.py` exists), allowing imports such as:

```python
from src.data.loaders import load_bars_daily
from src.pipeline.feature_pipeline import run_daily_feature_pipeline
```

---

## Contract Validation (Issue #3)

Consumer-side schema validation is enforced via:

```text
src/data/contract_validation.py
```

The `BarsContract` validator ensures:

* Required canonical columns exist
* `ts_utc` is timezone-aware and normalized to UTC
* No nulls exist in primary key columns (`symbol`, `ts_utc`, `timeframe`)
* Empty result sets are valid if schema is intact

Contract validation is automatically invoked in:

```text
load_bars_daily()
load_bars_1m()
```

Fail-fast behavior prevents downstream analytics from running on structurally invalid data.

---

# Daily Feature Set v1 (Issue #4)

## Overview

The Trade Analysis Engine computes a deterministic, symbol-segmented daily feature set derived from curated daily OHLCV bars.

Shared output expectations for feature datasets are defined in [docs/feature_dataset_contract.md](docs/feature_dataset_contract.md).
Feature dataset QA metrics and artifact outputs are documented in [docs/dataset_qa_metrics.md](docs/dataset_qa_metrics.md).
Feature dataset loader usage is documented in [docs/load_features.md](docs/load_features.md).
The repository-level feature registry lives at [configs/features.yml](configs/features.yml) and documents the currently implemented engineered feature sets and source datasets.

This layer operates strictly downstream of:

* Ingestion QA (data-quality enforcement)
* Consumer-side contract validation (Issue #3)

Feature computation assumes canonical, validated daily bars as input.

---

## Input Contract

`compute_daily_features_v1()` expects canonical daily bars with at least:

```text
symbol
ts_utc
close
timeframe
date
```

Input must already satisfy the BarsContract validation.

---

## Example Usage

```python
from src.data.loaders import load_bars_daily
from src.features.daily_features import compute_daily_features_v1

bars = load_bars_daily(
    symbols=["AAPL"],
    start_date="2025-01-01",
    end_date="2025-03-01",
)

features = compute_daily_features_v1(bars)
print(features.head())
```

### Daily Pipeline Usage

```python
from src.pipeline.feature_pipeline import run_daily_feature_pipeline

features = run_daily_feature_pipeline(
    symbols=["AAPL", "MSFT"],
    start_date="2025-01-01",
    end_date="2025-03-01",
)

print(features.head())
```

This runs:

* `load_bars_daily(...)`
* `compute_daily_features_v1(...)`
* `write_features(..., "1D")`

### Daily CLI Usage

```powershell
.\.venv\Scripts\python.exe -m cli.build_features --timeframe 1D --start 2025-11-03 --end 2025-11-04 --tickers .\configs\tickers_50.txt
```

For daily features that require lookback history, use a wider date range, for example:

```powershell
.\.venv\Scripts\python.exe -m cli.build_features --timeframe 1D --start 2025-09-01 --end 2025-11-04 --tickers .\configs\tickers_50.txt
```

---

# 1-Minute Feature Set v1 (Issue #5)

## Overview

The Trade Analysis Engine includes a short-horizon feature layer built on curated 1-minute OHLCV bars.

This module computes microstructure-level features suitable for intraday modeling and short-horizon signals.

---

## Input Contract

`compute_minute_features_v1()` expects canonical 1-minute bars with:

```text
symbol
ts_utc
close
volume
timeframe
date
```

Input must already pass consumer-side contract validation and canonical schema enforcement via loaders.

---

## Example Usage

```python
from src.data.loaders import load_bars_1m
from src.features.minute_features import compute_minute_features_v1

bars = load_bars_1m(
    symbols=["AAPL", "MSFT"],
    start_date="2025-01-01",
    end_date="2025-01-05",
)

features = compute_minute_features_v1(bars)
print(features.head())
```

### Minute Pipeline Usage

```python
from src.pipeline.feature_pipeline import run_minute_feature_pipeline

features = run_minute_feature_pipeline(
    symbols=["AAPL"],
    start_date="2025-01-02",
    end_date="2025-01-03",
)

print(features.head())
```

This runs:

* `load_bars_1m(...)`
* `compute_minute_features_v1(...)`
* `write_features(..., "1Min")`

---

# Feature Build CLI

## Overview

The repository now includes a CLI wrapper around the existing feature pipeline.

Use it to:

* Load curated daily or 1-minute bars from `MARKETLAKE_ROOT`
* Run the existing feature pipeline
* Preserve the current parquet feature writer behavior
* Emit a run summary artifact for traceability

No separate ingestion pipeline is introduced here. The CLI is a thin entrypoint over `src.pipeline.feature_pipeline`.

---

## Command

```powershell
.\.venv\Scripts\python.exe -m cli.build_features --timeframe 1Min --start 2025-11-03 --end 2025-11-04 --tickers .\configs\tickers_50.txt
```

Example daily run:

```powershell
.\.venv\Scripts\python.exe -m cli.build_features --timeframe 1D --start 2025-11-03 --end 2025-11-04 --tickers .\configs\tickers_50.txt
```

Arguments:

* `--timeframe` -> supported values: `1Min`, `1D`
* `--start` -> inclusive start date in `YYYY-MM-DD`
* `--end` -> exclusive end date in `YYYY-MM-DD`
* `--tickers` -> path to a text file with one ticker per line

Date window note:

* `--start 2025-11-03 --end 2025-11-04` targets the single trading date `2025-11-03`

Ticker file rules:

* One ticker per line
* Blank lines are ignored
* Surrounding whitespace is trimmed

---

## Outputs

Feature outputs are still written by the existing writer:

* Daily features -> `data/curated/features_daily/...`
* Minute features -> `data/curated/features_1m/...`

You can read engineered feature datasets back into analytics or research code with `load_features()`. See [docs/load_features.md](docs/load_features.md) for examples and path semantics.

Feature datasets can also be exposed as DuckDB views with `create_feature_views()` for direct SQL analytics over partitioned parquet, including `features_daily` and `features_1m`.

Each CLI run also writes:

```text
artifacts/feature_runs/<run_id>/summary.json
```

Feature QA summaries are also written to:

```text
artifacts/qa/features/
```

Feature metadata is also written to:

```text
artifacts/features/feature_metadata.json
```

See [docs/dataset_qa_metrics.md](docs/dataset_qa_metrics.md) for the QA export structure and included metrics.
See [docs/feature_metadata_export.md](docs/feature_metadata_export.md) for the feature metadata export structure and usage.

The summary includes:

* `run_id`
* `timeframe`
* `start`
* `end`
* `tickers_file`
* `symbols_requested`
* `symbols_processed`
* `feature_row_count`
* `marketlake_root`
* `input_partitions_used`
* `missingness_by_feature_column`

---

## Logging

The CLI logs:

* Resolved `MARKETLAKE_ROOT`
* Resolved timeframe
* Input partitions used
* Run ID

This makes each feature build easier to trace back to its input slice and artifact directory.
