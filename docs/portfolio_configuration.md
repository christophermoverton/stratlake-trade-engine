# Portfolio Configuration

## Overview

Portfolio configs describe which completed strategy runs should be combined and
how the portfolio should be constructed. Components can now come from either
completed strategy runs or alpha-evaluation sleeve artifacts.

The portfolio CLI accepts config files in:

* YAML
* JSON

The config is consumed by `src.cli.run_portfolio` and normalized through the
portfolio contract layer. For the full Milestone 11 workflow, see
[milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md).

## Supported Top-Level Shapes

The CLI can resolve a portfolio definition from any of these shapes.

### Named mapping under `portfolios`

```yaml
portfolios:
  momentum_meanrev_equal:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
        run_id: <run_id>
      - strategy_name: mean_reversion_v1
        run_id: <run_id>
```

### List under `portfolios`

```yaml
portfolios:
  - portfolio_name: momentum_meanrev_equal
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
        run_id: <run_id>
      - strategy_name: mean_reversion_v1
        run_id: <run_id>
```

### Single top-level definition

```yaml
portfolio_name: momentum_meanrev_equal
allocator: equal_weight
components:
  - strategy_name: momentum_v1
    run_id: <run_id>
  - strategy_name: mean_reversion_v1
    run_id: <run_id>
```

If the file contains multiple portfolio definitions, pass `--portfolio-name`
to select one deterministically.

## Example Definition

```yaml
portfolios:
  momentum_meanrev_risk_parity:
    allocator: risk_parity
    initial_capital: 1.0
    alignment_policy: intersection
    components:
      - strategy_name: momentum_v1
      - strategy_name: mean_reversion_v1
    optimizer:
      method: risk_parity
      long_only: true
      target_weight_sum: 1.0
      leverage_ceiling: 1.0
      covariance_ridge: 1e-8
    risk:
      volatility_window: 20
      target_volatility: 0.12
      var_confidence_level: 0.95
      cvar_confidence_level: 0.95
    volatility_targeting:
      enabled: true
      target_volatility: 0.10
      lookback_periods: 20
      volatility_epsilon: 1e-8
    execution:
      enabled: true
      transaction_cost_bps: 5
      fixed_fee: 0.001
      slippage_bps: 2
      slippage_model: turnover_scaled
      slippage_turnover_scale: 1.5
```

## Field Reference

### `portfolio_name`

Human-readable portfolio identifier.

It appears in saved config and manifest files, contributes to the deterministic
`run_id`, is printed by the CLI, and is stored in registry entries.

### `allocator`

Allocator or optimizer method used to produce static portfolio weights.

Supported values:

* `equal_weight`
* `max_sharpe`
* `risk_parity`

If `optimizer` is provided, `allocator` must match `optimizer.method`.

### `components`

List of component strategy definitions used to build the portfolio.

General form:

```yaml
components:
  - strategy_name: momentum_v1
    run_id: <run_id>
    artifact_type: strategy
  - strategy_name: mean_reversion_v1
    run_id: <run_id>
    artifact_type: strategy
```

Alpha sleeve components use the same shape but set `artifact_type:
alpha_sleeve` and point `run_id` at an alpha-evaluation run directory:

```yaml
components:
  - strategy_name: alpha_sleeve_v1
    run_id: <alpha_evaluation_run_id>
    artifact_type: alpha_sleeve
  - strategy_name: momentum_v1
    run_id: <strategy_run_id>
    artifact_type: strategy
```

Required behavior depends on selection mode:

* without `--from-registry`, each component must include both
  `strategy_name` and `run_id`
* with `--from-registry`, each component must include `strategy_name`, must
  not include `run_id`, and currently supports only `artifact_type: strategy`

Rules:

* must be a non-empty list
* components are normalized into deterministic sorted order
* `artifact_type` is optional and defaults to `strategy`
* supported `artifact_type` values are `strategy` and `alpha_sleeve`
* components must remain unique by strategy after resolution

Alpha sleeve artifact contract:

* the component directory must contain `sleeve_returns.csv`
* `sleeve_returns.csv` must contain `ts_utc` and `sleeve_return`
* timestamps must already be timezone-aware UTC
* the portfolio config's `strategy_name` becomes the sleeve identifier used in
  portfolio traceability columns such as `strategy_return__<strategy_name>`
* provenance is preserved in resolved component metadata through
  `artifact_type` and `source_artifact_path`

### `initial_capital`

Starting portfolio equity used for `portfolio_equity_curve`.

Optional. Defaults to `1.0`.

### `alignment_policy`

Return-series alignment behavior before portfolio construction.

Current supported value:

* `intersection`

`intersection` keeps only timestamps that exist for every component strategy.

### `optimizer`

Optional optimizer configuration.

Supported fields include:

* `method`
* `long_only`
* `target_weight_sum`
* `min_weight`
* `max_weight`
* `leverage_ceiling`
* `full_investment`
* `max_single_weight`
* `max_turnover`
* `risk_free_rate`
* `covariance_ridge`
* `max_iterations`
* `tolerance`

Important current limit:

* only `long_only: true` is supported

### `execution`

Optional execution-friction settings shared with the repository runtime model.

Supported fields include:

* `enabled`
* `execution_delay`
* `transaction_cost_bps`
* `slippage_bps`
* `fixed_fee`
* `fixed_fee_model`
* `slippage_model`
* `slippage_turnover_scale`
* `slippage_volatility_scale`

Supported `slippage_model` values:

* `constant`
* `turnover_scaled`
* `volatility_scaled`

Supported `fixed_fee_model` values:

* `per_rebalance`

### `validation` or `portfolio_validation`

Optional portfolio-validation thresholds.

Supported fields include:

* `long_only`
* `target_weight_sum`
* `weight_sum_tolerance`
* `target_net_exposure`
* `net_exposure_tolerance`
* `max_gross_exposure`
* `max_leverage`
* `max_single_sleeve_weight`
* `min_single_sleeve_weight`
* `max_abs_period_return`
* `max_equity_multiple`
* `strict_sanity_checks`

`validation` and `portfolio_validation` are treated as the same section by the
runtime layer.

### `risk`

Optional portfolio risk-summary configuration.

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

Current behavior:

* these settings control risk diagnostics and summary outputs
* `target_volatility` under `risk` does not execute post-optimizer scaling by
  itself
* use `risk` when you want review-time volatility summaries and recommended
  scaling diagnostics without changing executable weights

### `volatility_targeting`

Optional operational post-optimizer portfolio scaling.

Supported fields:

* `enabled`
* `target_volatility`
* `lookback_periods`
* `volatility_epsilon`

Current behavior:

* this section is separate from `risk`
* when enabled, the constructor computes a deterministic scale factor as
  `target_volatility / estimated_portfolio_volatility`
* `lookback_periods` controls the rolling volatility estimate used for that
  operational scaling step
* `volatility_epsilon` defines the effective zero-volatility cutoff that blocks
  unsafe scaling
* scaling is applied directly to the optimizer-produced base weights before
  execution-cost modeling and portfolio evaluation
* scaling is literal and uncapped in the current milestone
* metrics and manifests surface the enabled flag plus pre-target, post-target,
  and applied-scale metadata

Operational versus diagnostic behavior:

* `risk.target_volatility` affects diagnostics only
* `volatility_targeting.enabled: true` activates executable scaling
* when `volatility_targeting.enabled: false`, the base optimizer weights flow
  through unchanged even if `risk.target_volatility` is set

### `sanity`

Optional sanity-check settings from the shared runtime system.

### `simulation`

Optional simulation config for single-run portfolios.

Supported fields are the same as `src.config.simulation.SimulationConfig`,
including:

* `method`
* `num_paths`
* `path_length`
* `seed`
* `monte_carlo_mean`
* `monte_carlo_volatility`
* `distribution`
* `min_samples`
* `drawdown_threshold`
* `metrics`

Current limit:

* simulation is not supported when `--evaluation` is used

## CLI Interaction

### Config plus registry-backed selection

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D
```

In this mode:

* each component is matched by `strategy_name`
* the latest matching strategy run is selected from the strategy registry
* selection is filtered by the requested portfolio timeframe

### Config plus CLI overrides

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D \
  --optimizer-method max_sharpe \
  --enable-volatility-targeting \
  --volatility-target-volatility 0.10 \
  --volatility-target-lookback 20 \
  --risk-target-volatility 0.12 \
  --execution-enabled \
  --transaction-cost-bps 5 \
  --fixed-fee 0.001 \
  --slippage-bps 2
```

CLI overrides are applied on top of config values and repository defaults.

### Explicit run ids without a config file

```bash
python -m src.cli.run_portfolio \
  --portfolio-name momentum_meanrev_equal \
  --run-ids run-alpha run-beta \
  --timeframe 1D
```

This builds an implicit config with:

* `allocator: equal_weight`
* `initial_capital: 1.0`
* `alignment_policy: intersection`

## Timeframe, Evaluation, And Simulation

`--timeframe` remains required because it controls:

* portfolio metric annualization behavior
* registry filtering when `--from-registry` is used
* walk-forward timeframe compatibility checks
* simulation annualization

Walk-forward example:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name strict_valid_builtin_pair \
  --from-registry \
  --evaluation configs/evaluation.yml \
  --timeframe 1D
```

In walk-forward mode:

* the portfolio config still defines components and allocator
* the evaluation config defines the splits
* the portfolio timeframe must match the evaluation timeframe
* simulation cannot be combined with `--evaluation`

## Validation Summary

Portfolio config validation enforces:

* required fields `portfolio_name`, `allocator`, and `components`
* supported optimizer methods
* `allocator == optimizer.method` when optimizer config is provided
* JSON-serializable normalized config payloads
* deterministic ordering before artifact writing and registry registration

These checks are designed to fail fast before portfolio construction starts.
