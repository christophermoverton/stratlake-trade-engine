# Portfolio Construction Workflow

## Overview

The portfolio layer adds deterministic portfolio research on top of completed
strategy artifacts.

The portfolio workflow answers questions that a single strategy run cannot:

* Which completed strategy runs should be combined?
* How are those return streams aligned and weighted?
* How do optimizer choice, risk diagnostics, simulation, and execution
  friction affect the review?
* What portfolio-level artifacts should be persisted for reproducibility?

For the main Milestone 11 usage guide, start with
[milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md).

## Where The Portfolio Layer Fits

The portfolio layer consumes completed strategy runs under
`artifacts/strategies/<run_id>/` rather than rerunning strategies inside the
portfolio constructor.

```text
completed strategy runs
        ->
portfolio loaders
        ->
aligned return matrix
        ->
allocator / optimizer
        ->
optional volatility targeting
        ->
portfolio constructor
        ->
portfolio metrics + risk summaries
        ->
optional simulation
        ->
portfolio artifacts + QA
        ->
portfolio registry entry
```

This keeps the workflow deterministic, artifact-driven, and easy to audit.

## Component Map

### Loaders

Location:

```text
src/portfolio/loaders.py
```

Current behavior:

* reads root-level `equity_curve.csv` from each component strategy run
* requires `ts_utc` and `strategy_return`
* resolves `strategy_name` from saved strategy artifacts
* compounds same-timestamp rows into one strategy-level return when needed
* builds one long-form return table across component strategies
* aligns returns with `intersection` semantics only

### Allocator And Optimizer

Location:

```text
src/portfolio/allocators.py
src/portfolio/optimizer.py
```

Current supported methods:

* `equal_weight`
* `max_sharpe`
* `risk_parity`

All current methods produce one deterministic static weight vector. For
single-run portfolios that vector is estimated from the aligned return matrix
used for the run. For walk-forward portfolios it is estimated from the train
window and applied to the test window for each split.

Current optimizer limits:

* long-only allocation only
* no dynamic intra-window re-optimization

### Constructor

Location:

```text
src/portfolio/constructor.py
src/portfolio/execution.py
```

The constructor combines aligned component returns, weights, and execution
frictions into the in-memory portfolio output.

When top-level `volatility_targeting` is enabled, the constructor keeps the
optimizer-produced base weights for auditability, applies a deterministic
post-optimizer scaling factor to create the final executable weights, and then
passes those targeted weights into execution accounting and portfolio
evaluation.

It produces:

* `strategy_return__<strategy>` traceability columns
* `weight__<strategy>` traceability columns
* `gross_portfolio_return`
* turnover and rebalance diagnostics
* transaction-cost, fixed-fee, and slippage columns
* `net_portfolio_return`
* `portfolio_return`
* `portfolio_equity_curve`

Execution drag is deterministic and portfolio-level. Current slippage models
are `constant`, `turnover_scaled`, and `volatility_scaled`.

### Metrics And Risk

Location:

```text
src/portfolio/metrics.py
src/portfolio/risk.py
```

Portfolio metrics reuse the shared research metric primitives on the net
portfolio return stream and add centralized risk summaries.

Current headline metrics include:

* `total_return`
* `gross_total_return`
* `execution_drag_total_return`
* `annualized_return`
* `annualized_volatility`
* `sharpe_ratio`
* `turnover`
* `rebalance_count`
* `total_execution_friction`

Current risk metrics include:

* `realized_volatility`
* `rolling_volatility_latest`
* `rolling_volatility_mean`
* `max_drawdown`
* `current_drawdown`
* `value_at_risk`
* `conditional_value_at_risk`
* `volatility_targeting_enabled`
* `estimated_pre_target_volatility`
* `estimated_post_target_volatility`
* `volatility_scaling_factor`
* `volatility_target_scale`

Important current behavior:

* `risk.target_volatility` remains a diagnostic setting
* top-level `volatility_targeting` activates executable post-optimizer scaling
* the operational scaling factor is literal and uncapped in the current
  milestone
* downstream execution, metrics, QA, manifests, and registry metadata all use
  the final targeted weights when targeting is enabled

### Simulation

Location:

```text
src/research/simulation.py
```

Single-run portfolios can optionally write deterministic simulation artifacts
using bootstrap or normal Monte Carlo return paths.

Current limits:

* simulation is not available for walk-forward portfolios
* simulation summarizes synthetic outcome distributions; it does not change the
  realized portfolio results

### Artifacts And Registry

Location:

```text
src/portfolio/artifacts.py
src/portfolio/walk_forward.py
src/research/registry.py
```

Portfolio runs write deterministic CSV and JSON artifacts under
`artifacts/portfolios/<run_id>/` and append one `run_type: "portfolio"` row to
`artifacts/portfolios/registry.jsonl`.

Milestone 11 adds manifest and registry metadata for:

* optimizer settings and diagnostics
* risk summaries
* execution-friction summaries
* simulation summaries for single-run portfolios

See [portfolio_artifact_logging.md](portfolio_artifact_logging.md) for the full
artifact contract.

## Step-By-Step Workflow

1. Select component strategy runs.
   Use explicit `--run-ids`, define components in config, or resolve the latest
   matching runs from the strategy registry with `--from-registry`.
2. Load and align returns.
   `load_strategy_runs_returns()` reads component artifacts and
   `build_aligned_return_matrix()` keeps only shared timestamps.
3. Allocate weights.
   The allocator or optimizer produces a deterministic static weight matrix.
4. Optionally apply volatility targeting.
   A deterministic post-optimizer scaling factor can transform the base weights
   toward a configured target volatility before downstream evaluation.
5. Apply execution-friction accounting.
   The constructor computes turnover, costs, fees, slippage, and net portfolio
   returns.
6. Compute equity and metrics.
   `compute_portfolio_metrics()` calculates return, execution, exposure, and
   risk summary metrics.
7. Optionally run simulation.
   Single-run portfolios can write bootstrap or Monte Carlo simulation outputs.
8. Write artifacts.
   `write_portfolio_artifacts()` persists normalized files and validates
   consistency.
9. Register the run.
   `register_portfolio_run()` appends one deterministic portfolio row to the
   shared registry model.

## CLI Usage

Baseline single-run portfolio:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D
```

Optimizer-aware portfolio:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D \
  --optimizer-method risk_parity
```

Risk-aware and execution-aware portfolio:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D \
  --risk-target-volatility 0.12 \
  --risk-volatility-window 20 \
  --execution-enabled \
  --transaction-cost-bps 5 \
  --fixed-fee 0.001 \
  --slippage-bps 2 \
  --slippage-model volatility_scaled \
  --slippage-volatility-scale 1.5
```

Single-run simulation-enabled portfolio:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D \
  --simulation path/to/simulation.yml
```

Walk-forward portfolio:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name strict_valid_builtin_pair \
  --from-registry \
  --evaluation configs/evaluation.yml \
  --timeframe 1D
```

## Interpreting Results

When reviewing a portfolio run:

* use `manifest.json` first for run identity and high-level metadata
* use `metrics.json` for return, execution, and risk summaries
* use `portfolio_returns.csv` to see how component returns and weights map into
  gross and net portfolio returns
* use `portfolio_equity_curve.csv` to inspect the compounded net portfolio path
* use `simulation/summary.json` when simulation is enabled
* use `qa_summary.json` to confirm deterministic validation passed

For walk-forward runs, also inspect:

* `aggregate_metrics.json`
* `metrics_by_split.csv`
* `splits/<split_id>/qa_summary.json`

## Related Docs

* [milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md)
* [portfolio_configuration.md](portfolio_configuration.md)
* [portfolio_artifact_logging.md](portfolio_artifact_logging.md)
* [strategy_evaluation_workflow.md](strategy_evaluation_workflow.md)
* [evaluation_split_configuration.md](evaluation_split_configuration.md)
