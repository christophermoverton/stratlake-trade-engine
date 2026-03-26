# Portfolio Construction Workflow

## Overview

Milestone 9 adds a first-class portfolio layer on top of the existing
strategy research workflow.

The portfolio layer answers a different question than a single strategy run:

* Which completed strategy runs should be combined?
* How are those strategy return streams aligned and weighted?
* What portfolio-level return, equity, metric, QA, and artifact outputs should
  be persisted for reproducibility?

At a high level, the portfolio workflow sits downstream of strategy execution:

```text
features
    ->
signals
    ->
backtest
    ->
strategy metrics
    ->
portfolio construction
    ->
portfolio metrics
    ->
artifacts
    ->
registry
```

Use this document as the central guide, then drill into the focused docs for
details:

* [docs/portfolio_configuration.md](portfolio_configuration.md)
* [docs/portfolio_artifact_logging.md](portfolio_artifact_logging.md)
* [docs/strategy_evaluation_workflow.md](strategy_evaluation_workflow.md)
* [docs/experiment_artifact_logging.md](experiment_artifact_logging.md)
* [docs/evaluation_split_configuration.md](evaluation_split_configuration.md)

## Where The Portfolio Layer Fits

The existing strategy pipeline produces completed strategy runs under
`artifacts/strategies/<run_id>/`. Each run includes an `equity_curve.csv`
artifact with the timestamped `strategy_return` series needed by the portfolio
layer.

The portfolio layer consumes those saved strategy artifacts rather than
rerunning strategies inside the portfolio constructor. This keeps the workflow
artifact-driven and makes portfolio runs reproducible, deterministic, and easy
to audit.

The end-to-end flow is:

```text
completed strategy runs
        ->
portfolio loaders
        ->
aligned return matrix
        ->
allocator
        ->
portfolio constructor
        ->
portfolio metrics
        ->
portfolio artifacts + QA
        ->
portfolio registry entry
```

## Component Map

### Loaders

Location:

```text
src/portfolio/loaders.py
```

The loaders read completed strategy run directories and normalize them into
portfolio-ready return inputs.

Current behavior:

* reads root-level `equity_curve.csv` from each component strategy run
* requires `ts_utc` and `strategy_return`
* resolves `strategy_name` from `config.json` or `manifest.json`
* compounds same-timestamp rows into one strategy-level return when a strategy artifact contains multiple symbol rows for the same timestamp
* builds one long-form return table across component strategies
* aligns returns with `intersection` semantics only

`intersection` alignment means the portfolio keeps only timestamps present for
every component strategy. If no shared timestamps remain, the run fails
explicitly rather than silently filling missing returns.

### Allocator

Location:

```text
src/portfolio/allocators.py
```

The allocator converts the aligned return matrix into a weight matrix.

Milestone 9 currently supports:

* `equal_weight`

`EqualWeightAllocator` assigns the same weight to every component strategy at
every timestamp. The resulting weight rows must sum to `1.0`.

### Constructor

Location:

```text
src/portfolio/constructor.py
```

The constructor combines aligned component returns and weights into the
in-memory portfolio output.

It produces:

* `strategy_return__<strategy>` traceability columns
* `weight__<strategy>` traceability columns
* `portfolio_return`
* `portfolio_equity_curve`

Current construction logic is:

```python
portfolio_return = (aligned_returns * weights).sum(axis=1)
portfolio_equity_curve = initial_capital * (1.0 + portfolio_return).cumprod()
```

### Metrics

Location:

```text
src/portfolio/metrics.py
```

Portfolio metrics reuse the existing research metric primitives on the
portfolio return stream.

Current summary metrics:

* `cumulative_return`
* `total_return`
* `volatility`
* `annualized_return`
* `annualized_volatility`
* `sharpe_ratio`
* `max_drawdown`
* `win_rate`
* `hit_rate`
* `profit_factor`
* `turnover`
* `exposure_pct`

Weight-based metrics are computed from `weight__<strategy>` traceability
columns:

* `turnover` is the mean absolute component-weight change by timestamp
* `exposure_pct` is the percentage of timestamps with non-zero gross exposure

### Artifacts

Location:

```text
src/portfolio/artifacts.py
src/portfolio/qa.py
```

Portfolio runs write deterministic CSV and JSON artifacts under
`artifacts/portfolios/<run_id>/`.

The artifact layer persists:

* normalized config
* resolved components
* weights
* component returns plus traceability columns
* portfolio equity curve
* metrics
* QA summary
* manifest

Walk-forward portfolio runs additionally write:

* `aggregate_metrics.json`
* `metrics_by_split.csv`
* per-split files under `splits/<split_id>/`

See [docs/portfolio_artifact_logging.md](portfolio_artifact_logging.md) for the
full artifact contract.

### Registry

Location:

```text
src/research/registry.py
```

Portfolio runs are registered alongside strategy runs in a shared JSONL
registry model, but with `run_type: "portfolio"`.

Portfolio registry entries capture:

* `run_id`
* `portfolio_name`
* `allocator_name`
* `component_run_ids`
* `component_strategy_names`
* `timeframe`
* `start_ts`
* `end_ts`
* `artifact_path`
* `metrics`
* `evaluation_config_path`
* `split_count`
* `config`
* `components`
* `metadata`

This makes portfolio runs queryable without reopening every artifact directory.

### CLI

Location:

```text
src/cli/run_portfolio.py
```

The portfolio CLI is the entrypoint for end-to-end portfolio construction from
saved strategy runs.

Supported inputs:

* explicit `--run-ids`
* `--portfolio-config` plus explicit component `run_id` values
* `--portfolio-config --from-registry` to resolve the latest matching strategy
  run per configured `strategy_name`

Supported modes:

* single-run portfolio construction
* walk-forward portfolio evaluation with `--evaluation`

## Step-By-Step Workflow

The portfolio workflow maps to the current implementation as follows:

1. `select component strategy runs`
   Choose completed strategy run ids directly with `--run-ids`, declare them
   in a portfolio config, or resolve them from the strategy registry with
   `--from-registry`.
2. `load and align returns`
   `load_strategy_runs_returns()` reads each component run's
   `equity_curve.csv`, then `build_aligned_return_matrix()` keeps only shared
   timestamps under `intersection` alignment.
3. `allocate weights`
   The allocator creates a deterministic weight matrix from the aligned return
   matrix. Milestone 9 currently uses `equal_weight`.
4. `compute portfolio returns`
   `compute_portfolio_returns()` writes one weighted return stream plus
   component traceability columns.
5. `compute equity curve`
   `compute_portfolio_equity_curve()` compounds `portfolio_return` from the
   configured `initial_capital`.
6. `compute metrics`
   `compute_portfolio_metrics()` calculates return, risk, and weight-derived
   activity metrics.
7. `write artifacts`
   `write_portfolio_artifacts()` writes normalized files and runs deterministic
   portfolio QA before finalizing the manifest.
8. `register run`
   `register_portfolio_run()` appends one portfolio entry to
   `artifacts/portfolios/registry.jsonl`.

## CLI Usage

Module execution:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_portfolio --portfolio-name core_portfolio --run-ids run-a run-b --timeframe 1D
```

Required arguments:

* exactly one of `--portfolio-config` or `--run-ids`
* `--timeframe`

Important optional arguments:

* `--portfolio-name` -> required when using `--run-ids`; also used to select a
  named portfolio from config when multiple definitions exist
* `--from-registry` -> resolve the latest matching strategy run per configured
  `strategy_name`
* `--evaluation [PATH]` -> run walk-forward portfolio evaluation
* `--output-dir` -> override the default `artifacts/portfolios/` root

Validation rules:

* `--run-ids` cannot be combined with `--from-registry`
* `--from-registry` requires `--portfolio-config`
* supported timeframes are `1D` and `1Min`

Console summary output includes:

* portfolio name
* run id
* allocator
* component count
* timeframe
* single-run totals or walk-forward aggregate statistics

## Reproducible Example

This section shows a minimal end-to-end portfolio run using the current CLI and
the repository's existing strategy artifact flow.

### 1. Ensure component strategy runs exist

The portfolio runner consumes completed strategy artifacts. You can either use
existing runs already recorded in `artifacts/strategies/registry.jsonl` or
create fresh strategy runs first with:

```bash
python -m src.cli.run_strategy --strategy momentum_v1
python -m src.cli.run_strategy --strategy mean_reversion_v1
```

Those commands write strategy artifacts under `artifacts/strategies/<run_id>/`
and register them for later selection.

### 2. Create `configs/portfolios.yml`

The portfolio config can pin explicit strategy `run_id` values:

```yaml
portfolios:
  momentum_meanrev_equal:
    allocator: equal_weight
    initial_capital: 1.0
    components:
      - strategy_name: momentum_v1
        run_id: <run_id>
      - strategy_name: mean_reversion_v1
        run_id: <run_id>
    alignment_policy: intersection
```

If you want a zero-edit registry-backed demo instead, use:

```yaml
portfolios:
  momentum_meanrev_equal:
    allocator: equal_weight
    initial_capital: 1.0
    components:
      - strategy_name: momentum_v1
      - strategy_name: mean_reversion_v1
    alignment_policy: intersection
```

### 3. Run a single portfolio

Explicit run ids:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --timeframe 1D
```

Registry-backed selection:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D
```

The CLI prints the portfolio name, run id, allocator, component count,
timeframe, total return, Sharpe ratio, and max drawdown.

### 4. Optional walk-forward portfolio evaluation

If the component strategy runs cover the same evaluation period and timeframe,
you can construct the portfolio independently on each evaluation split:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --evaluation configs/evaluation.yml \
  --timeframe 1D
```

In walk-forward mode the CLI prints:

* split count
* mean total return across splits
* mean Sharpe ratio across splits
* worst max drawdown across splits

### 5. Inspect the outputs

Single-run portfolio artifacts are written to:

```text
artifacts/portfolios/<run_id>/
```

Start with:

* `manifest.json` for a compact run summary
* `metrics.json` for the portfolio metric payload
* `portfolio_returns.csv` for traceable component contributions
* `portfolio_equity_curve.csv` for the compounded portfolio path
* `qa_summary.json` for deterministic validation results

Walk-forward portfolio runs additionally include:

* `aggregate_metrics.json`
* `metrics_by_split.csv`
* `splits/<split_id>/`

### Interpreting Results

When reviewing a portfolio run:

* use `portfolio_returns.csv` to confirm how component returns and weights map
  into each `portfolio_return`
* use `portfolio_equity_curve.csv` to inspect compounding behavior over time
* use `metrics.json` or `aggregate_metrics.json` to compare return, risk, and
  activity statistics
* use `qa_summary.json` to confirm deterministic validation passed and no
  no-lookahead or traceability invariants were broken

## Walk-Forward Semantics

Walk-forward portfolio evaluation uses the same split configuration model as
the strategy runner.

Current behavior:

* loads the evaluation config from `configs/evaluation.yml` or a supplied path
* generates deterministic evaluation splits
* slices each component strategy return stream to the split test window only
* constructs and scores one independent portfolio per split
* aggregates split-level portfolio metrics into descriptive statistics

As with the broader evaluation layer:

* split windows use half-open intervals
* `start` is inclusive
* `end` is exclusive

See [docs/evaluation_split_configuration.md](evaluation_split_configuration.md)
for split definitions and validation details.
