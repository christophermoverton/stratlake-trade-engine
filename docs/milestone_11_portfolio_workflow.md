# Milestone 11 Portfolio Workflow

## Overview

Milestone 11 extends the portfolio layer from equal-weight construction into a
deterministic workflow that can include:

* optimizer-driven static allocation
* centralized portfolio risk summaries
* deterministic return simulation
* execution-friction accounting with transaction costs, fixed fees, and
  slippage
* artifact and registry metadata that expose those settings for review

This document is the main practical guide for using those features end to end.
For repository-level context, see [../README.md](../README.md). For detailed
artifact and config references, see
[alpha_workflow.md](alpha_workflow.md),
[portfolio_artifact_logging.md](portfolio_artifact_logging.md),
[portfolio_configuration.md](portfolio_configuration.md), and
[research_integrity_and_qa.md](research_integrity_and_qa.md).

## Where Milestone 11 Fits

The Milestone 11 portfolio flow sits downstream of completed strategy runs:

```text
completed strategy artifacts
        ->
component selection
        ->
aligned return matrix
        ->
allocator / optimizer
        ->
optional volatility targeting
        ->
portfolio returns + execution friction
        ->
portfolio metrics + risk summaries
        ->
optional simulation
        ->
artifacts + manifest + registry row
```

The portfolio CLI does not rerun strategies. It loads saved strategy artifacts,
which keeps the workflow deterministic and auditable.

Alpha-derived sleeves can also feed the same constructor once they have been
backtested into aligned return streams. See
[examples/milestone_12_alpha_portfolio_workflow.md](examples/milestone_12_alpha_portfolio_workflow.md).

## What To Run First

Portfolio runs need completed strategy artifacts under
`artifacts/strategies/<run_id>/`.

Example strategy runs:

```bash
python -m src.cli.run_strategy --strategy momentum_v1
python -m src.cli.run_strategy --strategy mean_reversion_v1
```

For registry-backed portfolio demos, the shipped
[../configs/portfolios.yml](../configs/portfolios.yml) already references
`momentum_v1` and `mean_reversion_v1` by strategy name.

## Baseline Portfolio Run

The simplest portfolio flow uses the latest matching strategy runs from the
strategy registry:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D
```

What this does:

* loads the named portfolio definition from `configs/portfolios.yml`
* resolves the latest strategy run per component `strategy_name`
* aligns component returns with `intersection` semantics
* constructs a deterministic static portfolio
* writes artifacts under `artifacts/portfolios/<run_id>/`
* appends a `run_type: "portfolio"` row to `artifacts/portfolios/registry.jsonl`

Useful first files to inspect:

* `manifest.json`
* `metrics.json`
* `qa_summary.json`
* `portfolio_returns.csv`
* `portfolio_equity_curve.csv`

## Optimizer Workflow

### Supported methods

The centralized optimizer currently supports:

* `equal_weight`
* `max_sharpe`
* `risk_parity`

These methods all produce one static weight vector that is applied across the
application return window.

### Supported constraint types

At a high level, the optimizer supports:

* long-only allocation only
* target weight sum
* optional per-sleeve minimum and maximum weights
* optional leverage ceiling
* optional max single weight
* optional max turnover versus previous weights
* full-investment controls

Important current limits:

* `long_only=False` is not supported
* optimizer output is static, not dynamically re-optimized each period
* optimization uses historical mean and covariance from the estimation sample
* walk-forward portfolio runs estimate weights on each split's train window and
  apply them to that split's test window

### How to supply optimizer settings

You can configure the optimizer either in the portfolio config or from the CLI.

CLI example:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D \
  --optimizer-method max_sharpe
```

Example config snippet for a custom portfolio definition:

```yaml
portfolio_name: momentum_meanrev_max_sharpe
allocator: max_sharpe
initial_capital: 1.0
alignment_policy: intersection
components:
  - strategy_name: momentum_v1
    run_id: <run_id>
  - strategy_name: mean_reversion_v1
    run_id: <run_id>
optimizer:
  method: max_sharpe
  long_only: true
  target_weight_sum: 1.0
  leverage_ceiling: 1.0
  max_single_weight: 0.75
  covariance_ridge: 1e-8
  max_iterations: 500
  tolerance: 1e-8
```

Rules worth knowing:

* `allocator` must match `optimizer.method`
* optimizer-derived validation overrides are applied automatically
* the CLI `--optimizer-method` override changes both the effective allocator and
  optimizer method

### What to expect in outputs

Optimizer-aware runs surface their configuration in:

* `config.json`
* `manifest.json`
* the portfolio registry row

`manifest.json` also includes:

* `optimizer.method`
* `optimizer.constraint_summary`
* `optimizer.diagnostic_summary`

Those diagnostics summarize items such as convergence, exposure, objective
volatility, objective Sharpe, and observation count.

## Risk-Aware Workflow

### What the risk layer computes

The centralized portfolio risk layer summarizes:

* rolling volatility
* realized volatility
* volatility targeting diagnostics
* drawdown and drawdown duration
* historical VaR
* historical CVaR

The portfolio workflow can also apply optional operational volatility targeting
through a top-level `volatility_targeting` section or the CLI flags
`--enable-volatility-targeting`, `--volatility-target-volatility`, and
`--volatility-target-lookback`.

CLI example with risk overrides:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D \
  --risk-target-volatility 0.12 \
  --risk-volatility-window 20 \
  --risk-var-confidence-level 0.95 \
  --risk-cvar-confidence-level 0.95
```

### How risk settings are configured

Risk settings can be supplied under `risk` in portfolio config or overridden by
CLI flags.

Supported fields:

* `volatility_window`
* `target_volatility`
* `min_volatility_scale`
* `max_volatility_scale`
* `allow_scale_up`
* `var_confidence_level`
* `cvar_confidence_level`
* `volatility_epsilon`
* `periods_per_year_override`

Example config snippet:

```yaml
risk:
  volatility_window: 20
  target_volatility: 0.12
  min_volatility_scale: 0.0
  max_volatility_scale: 1.0
  allow_scale_up: false
  var_confidence_level: 0.95
  cvar_confidence_level: 0.95
```

Operational volatility targeting is configured separately:

```yaml
volatility_targeting:
  enabled: true
  target_volatility: 0.10
  lookback_periods: 20
  volatility_epsilon: 1e-8
```

CLI example:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_targeted \
  --from-registry \
  --timeframe 1D \
  --enable-volatility-targeting \
  --volatility-target-volatility 0.10 \
  --volatility-target-lookback 20
```

### How to interpret the risk outputs

Current behavior is important:

* `risk.target_volatility` still controls diagnostic risk summaries
* top-level `volatility_targeting` applies an operational post-optimizer
  scaling step before execution and portfolio evaluation
* the operational scaling factor is computed as
  `target_volatility / estimated_portfolio_volatility`
* `lookback_periods` controls the rolling volatility estimate used by the
  operational scaling step
* `volatility_epsilon` defines the effective zero-volatility cutoff for that
  operational targeting path
* scaling is literal and uncapped in this milestone; there is no hidden
  clipping or leverage cap
* when enabled, the estimated pre-target volatility uses the configured
  `lookback_periods` rolling volatility and falls back to realized volatility
  only when the rolling window has not warmed up yet
* if the estimated pre-target volatility is effectively zero, the run fails
  clearly instead of silently inventing leverage behavior

That means:

* base optimizer weights remain auditable
* `weight__<strategy>` columns in the realized portfolio output are the final
  targeted weights when targeting is enabled
* metrics and manifests now expose both pre-target and post-target visibility

Risk metrics appear in:

* `metrics.json`
* `manifest.json`
* the portfolio registry row

Key fields include:

* `volatility_targeting_enabled`
* `estimated_pre_target_volatility`
* `estimated_post_target_volatility`
* `volatility_scaling_factor`
* `realized_volatility`
* `rolling_volatility_latest`
* `rolling_volatility_mean`
* `max_drawdown`
* `current_drawdown`
* `value_at_risk`
* `conditional_value_at_risk`
* `volatility_target_scale`
* `volatility_target_scale_capped`

## Execution-Friction Workflow

### Supported execution-friction inputs

The portfolio execution model supports:

* proportional transaction cost via `transaction_cost_bps`
* fixed fees via `fixed_fee`
* slippage via `slippage_bps`

Supported slippage models:

* `constant`
* `turnover_scaled`
* `volatility_scaled`

Supported fixed-fee model:

* `per_rebalance`

### CLI example

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D \
  --execution-enabled \
  --transaction-cost-bps 5 \
  --fixed-fee 0.001 \
  --slippage-bps 2 \
  --slippage-model turnover_scaled \
  --slippage-turnover-scale 1.5
```

### How to interpret gross vs net results

Portfolio outputs keep both realized return views:

* `gross_portfolio_return`
* `net_portfolio_return`
* `portfolio_return`

Current behavior:

* `portfolio_return` is the net return series
* `gross_total_return` is computed from gross portfolio returns
* `total_return` and `net_total_return` reflect the net portfolio path
* `execution_drag_total_return` is `gross_total_return - total_return`

Execution drag appears in:

* CLI summary output
* `portfolio_returns.csv`
* `metrics.json`
* `manifest.json`

Key friction metrics include:

* `total_transaction_cost`
* `total_fixed_fee`
* `total_slippage_cost`
* `total_execution_friction`
* `average_execution_friction_per_trade`
* `rebalance_count`

Current limitations to document clearly:

* fixed fees currently support only `per_rebalance`
* slippage uses deterministic proxy models, not market microstructure replay
* execution friction is applied from the portfolio weight-change path; it is not
  a live execution simulator

## Simulation Workflow

### What simulation does

Simulation takes a realized return series and generates synthetic return paths
for distributional analysis.

Supported methods:

* `bootstrap`
* `monte_carlo`

Current Monte Carlo distribution:

* `normal`

This is available for:

* strategy single runs via `src.cli.run_strategy --simulation ...`
* portfolio single runs via `src.cli.run_portfolio --simulation ...`

It is not available with:

* strategy walk-forward runs
* strategy robustness runs
* portfolio walk-forward runs

### Portfolio CLI example

The repository does not currently ship a `configs/simulation.yml`, so create
your own config file and point the CLI at it.

Example simulation config:

```yaml
simulation:
  method: bootstrap
  num_paths: 500
  path_length: 252
  seed: 7
  drawdown_threshold: 0.20
```

Example run:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D \
  --simulation path/to/simulation.yml
```

### How to interpret simulation outputs

Simulation writes a `simulation/` subdirectory under the portfolio run:

```text
artifacts/portfolios/<run_id>/simulation/
```

Files written:

* `config.json`
* `assumptions.json`
* `path_metrics.csv`
* `simulated_paths.csv`
* `summary.json`
* `manifest.json`

What the outputs mean:

* realized backtest metrics describe the observed portfolio path
* simulation metrics describe a synthetic distribution built from that observed
  return series and the chosen simulation assumptions

Important current limits:

* bootstrap is IID resampling with replacement
* Monte Carlo uses a normal distribution
* no regime-switching, serial dependence model, or cross-asset path generator
* simulation is for distributional context, not proof of out-of-sample edge

## Robustness Workflow

Milestone 11 also includes deterministic robustness analysis for strategy
parameter sweeps. This is part of the broader research flow even though it runs
through `run_strategy`, not `run_portfolio`.

Use robustness when you want to answer:

* is the strategy sensitive to small parameter changes?
* do neighboring parameter choices behave similarly?
* does the preferred parameter set remain competitive across subperiods or
  walk-forward splits?

Shipped example:

```bash
python -m src.cli.run_strategy --robustness
```

The default [../configs/robustness.yml](../configs/robustness.yml) performs a
parameter sweep for `momentum_v1`.

High-level outputs include:

* `metrics_by_variant.csv`
* `stability_metrics.csv` when stability analysis is enabled
* `neighbor_metrics.csv` when neighboring variants can be compared
* `summary.json`

Keep the interpretation narrow:

* robustness compares realized backtest outcomes across parameter variants
* simulation estimates synthetic outcome distributions for one chosen realized
  return stream
* they answer different questions and should not be merged conceptually

## Walk-Forward Portfolio Workflow

Portfolio walk-forward evaluation is supported and stable enough to document.

Example:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name strict_valid_builtin_pair \
  --from-registry \
  --evaluation configs/evaluation.yml \
  --timeframe 1D
```

Current behavior:

* component strategy returns are sliced by split
* optimization uses each split's train window
* the resulting static weights are applied to that split's test window
* per-split metrics are aggregated into descriptive statistics

Walk-forward artifacts write to:

```text
artifacts/portfolios/<run_id>/
  aggregate_metrics.json
  metrics_by_split.csv
  splits/<split_id>/
```

Simulation is intentionally disabled in this mode.

## Where Artifacts And Registry Rows Are Written

Single-run portfolio outputs:

```text
artifacts/portfolios/<run_id>/
```

Portfolio registry:

```text
artifacts/portfolios/registry.jsonl
```

What the registry captures for Milestone 11:

* `portfolio_name`
* `allocator_name`
* `optimizer_method`
* `timeframe`
* `start_ts`
* `end_ts`
* `metrics_summary`
* `risk_summary`
* `simulation_enabled`
* `simulation_method`
* execution summary and manifest metadata

Use the registry when you want a lightweight summary surface without opening
every artifact directory.

## Reading The Main Outputs

Recommended inspection order for a single-run portfolio:

1. `manifest.json`
2. `qa_summary.json`
3. `metrics.json`
4. `portfolio_returns.csv`
5. `portfolio_equity_curve.csv`
6. `simulation/summary.json` when simulation is enabled

What each file is best for:

* `manifest.json`: run identity, artifact inventory, optimizer/risk/execution
  summaries
* `qa_summary.json`: validation status and review flags
* `metrics.json`: headline return, risk, turnover, and execution-friction
  metrics
* `portfolio_returns.csv`: per-timestamp traceability from component returns
  and weights into gross and net portfolio returns
* `portfolio_equity_curve.csv`: compounded net portfolio path
* `simulation/summary.json`: synthetic path distribution summaries

## Related Docs

* [../README.md](../README.md)
* [portfolio_configuration.md](portfolio_configuration.md)
* [portfolio_construction_workflow.md](portfolio_construction_workflow.md)
* [portfolio_artifact_logging.md](portfolio_artifact_logging.md)
* [research_integrity_and_qa.md](research_integrity_and_qa.md)
* [execution_model.md](execution_model.md)
* [runtime_configuration.md](runtime_configuration.md)
* [examples/milestone_12_alpha_portfolio_workflow.md](examples/milestone_12_alpha_portfolio_workflow.md)
