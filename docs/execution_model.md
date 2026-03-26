# Execution Model

## Overview

StratLake uses a deterministic execution model for both strategy and portfolio
research. The goal is to make lag, turnover, transaction costs, and slippage
explicit while keeping reruns stable.

Execution settings are resolved through the shared runtime configuration
contract described in [runtime_configuration.md](runtime_configuration.md).

## Strategy Execution

Strategy backtests are executed in `src/research/backtest_runner.py`.

Current behavior:

* signals are coerced to numeric values
* positions are lagged by `execution_delay`
* gross returns are computed from lagged positions
* transaction cost and slippage are charged from absolute position change
* `strategy_return` is the net return after execution friction

Formula:

```text
executed_signal = signal.shift(execution_delay).fillna(0.0)
gross_strategy_return = executed_signal * asset_return
transaction_cost = abs(delta_position) * transaction_cost_bps / 10000
slippage_cost = abs(delta_position) * slippage_bps / 10000
net_strategy_return = gross_strategy_return - transaction_cost - slippage_cost
strategy_return = net_strategy_return
equity_curve = (1.0 + strategy_return).cumprod()
```

Current strategy output columns include:

* `executed_signal`
* `position`
* `delta_position`
* `abs_delta_position`
* `turnover`
* `trade_event`
* `gross_strategy_return`
* `transaction_cost`
* `slippage_cost`
* `execution_friction`
* `net_strategy_return`
* `strategy_return`
* `equity_curve`

## Portfolio Execution

Portfolio execution is handled in `src/portfolio/constructor.py`.

Current behavior:

* strategy run returns are aligned into a wide matrix
* the allocator produces weights
* gross portfolio return is the weighted sum of sleeve returns
* transaction cost and slippage are charged from portfolio turnover
* `portfolio_return` is the net return after execution friction
* `portfolio_equity_curve` compounds from `initial_capital`

Formula:

```text
gross_portfolio_return = sum(strategy_return_i * weight_i)
portfolio_transaction_cost = portfolio_turnover * transaction_cost_bps / 10000
portfolio_slippage_cost = portfolio_turnover * slippage_bps / 10000
net_portfolio_return = gross_portfolio_return - portfolio_transaction_cost - portfolio_slippage_cost
portfolio_return = net_portfolio_return
portfolio_equity_curve = initial_capital * (1.0 + portfolio_return).cumprod()
```

Current portfolio output columns include:

* `strategy_return__<strategy>`
* `weight__<strategy>`
* `gross_portfolio_return`
* `portfolio_weight_change`
* `portfolio_abs_weight_change`
* `portfolio_turnover`
* `portfolio_rebalance_event`
* `portfolio_transaction_cost`
* `portfolio_slippage_cost`
* `portfolio_execution_friction`
* `net_portfolio_return`
* `portfolio_return`
* `portfolio_equity_curve`

## Configuration Fields

Execution settings currently support:

* `enabled`
* `execution_delay`
* `transaction_cost_bps`
* `slippage_bps`

Defaults come from [../configs/execution.yml](../configs/execution.yml):

```yaml
execution:
  enabled: false
  execution_delay: 1
  transaction_cost_bps: 0.0
  slippage_bps: 0.0
```

Important details:

* `execution_delay` must be an integer greater than or equal to `1`
* costs must be non-negative
* if `enabled` is omitted, non-zero costs imply execution is enabled

## Deterministic Behavior

The execution model is deterministic by design:

* no random slippage model is used
* no stochastic fill process is introduced
* reruns with unchanged inputs and config produce the same outputs

That makes execution realism explicit without sacrificing reproducibility.

## Related Docs

* [research_validity_framework.md](research_validity_framework.md)
* [strict_mode.md](strict_mode.md)
* [runtime_configuration.md](runtime_configuration.md)
