# Runtime Configuration

## Overview

StratLake resolves execution realism, sanity checks, portfolio validation, and
strict mode through one normalized runtime contract implemented in
`src/config/runtime.py`.

The resolved runtime object contains:

* `execution`
* `sanity`
* `portfolio_validation`
* `strict_mode`

## Precedence

Resolution is deterministic:

```text
repository defaults < config < CLI
```

More specifically:

1. execution defaults load from [../configs/execution.yml](../configs/execution.yml)
2. sanity defaults load from [../configs/sanity.yml](../configs/sanity.yml)
3. portfolio validation starts from code defaults in `src/portfolio/contracts.py`
4. runtime values from strategy or portfolio config are merged in
5. CLI overrides are applied last

`--strict` always forces strict mode on with `source: "cli"`.

## Supported Sections

### `execution`

Supported keys:

* `enabled`
* `execution_delay`
* `transaction_cost_bps`
* `slippage_bps`

### `sanity`

Supported keys:

* `max_abs_period_return`
* `max_annualized_return`
* `max_sharpe_ratio`
* `max_equity_multiple`
* `strict_sanity_checks`
* `min_annualized_volatility_floor`
* `min_volatility_trigger_sharpe`
* `min_volatility_trigger_annualized_return`
* `smoothness_min_sharpe`
* `smoothness_min_annualized_return`
* `smoothness_max_drawdown`
* `smoothness_min_positive_return_fraction`

### `portfolio_validation`

Supported keys:

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

### `strict_mode`

Supported keys:

* `enabled`
* `source`

`source` is informational when persisted. Users typically only set `enabled`.

## Accepted Shapes

Runtime sections can be supplied either at top level or under `runtime`.

Example top-level shape:

```yaml
execution:
  enabled: true
  transaction_cost_bps: 5.0

sanity:
  max_sharpe_ratio: 6.0

portfolio_validation:
  max_gross_exposure: 1.0

strict_mode:
  enabled: true
```

Example nested shape:

```yaml
runtime:
  execution:
    enabled: true
    transaction_cost_bps: 5.0
  sanity:
    max_sharpe_ratio: 6.0
  portfolio_validation:
    max_gross_exposure: 1.0
  strict_mode:
    enabled: true
```

Portfolio configs also accept the legacy alias `validation`. It resolves to the
same portfolio-validation section. `validation` and `portfolio_validation`
cannot be defined with different values in the same payload.

## CLI Overrides

### Strategy CLI

`src/cli/run_strategy.py` supports:

* `--execution-delay`
* `--transaction-cost-bps`
* `--slippage-bps`
* `--execution-enabled`
* `--disable-execution-model`
* `--strict`

### Portfolio CLI

`src/cli/run_portfolio.py` supports the same execution and strict-mode flags:

* `--execution-delay`
* `--transaction-cost-bps`
* `--slippage-bps`
* `--execution-enabled`
* `--disable-execution-model`
* `--strict`

## Persistence

Completed runs persist effective runtime settings for auditability.

Strategy artifacts include:

* top-level `execution`
* top-level `sanity`
* top-level `strict_mode`
* nested `runtime`

Portfolio artifacts include:

* top-level `execution`
* top-level `sanity`
* top-level `validation`
* top-level `strict_mode`
* nested `runtime`

For strategy runs, `portfolio_validation` is part of the nested `runtime`
payload even though strategy execution does not use portfolio construction.

## Strict-Mode Resolution

Strict mode can be enabled in three ways:

* `strict_mode.enabled: true`
* `sanity.strict_sanity_checks: true`
* `portfolio_validation.strict_sanity_checks: true`

If any of those are enabled in config, resolved strict mode is on with
`source: "config"`. If `--strict` is supplied, the resolved source becomes
`"cli"`.

## Related Docs

* [execution_model.md](execution_model.md)
* [strict_mode.md](strict_mode.md)
* [research_validity_framework.md](research_validity_framework.md)
