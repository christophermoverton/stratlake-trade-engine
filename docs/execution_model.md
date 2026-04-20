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
* when directional asymmetry is configured, separate long/short costs are computed
* `strategy_return` is the net return after execution friction

Formula (symmetric case):

```text
executed_signal = signal.shift(execution_delay).fillna(0.0)
gross_strategy_return = executed_signal * asset_return
transaction_cost = abs(delta_position) * transaction_cost_bps / 10000
slippage_cost = abs(delta_position) * slippage_bps / 10000
net_strategy_return = gross_strategy_return - transaction_cost - slippage_cost
strategy_return = net_strategy_return
equity_curve = (1.0 + strategy_return).cumprod()
```

Formula (directional case, when `has_directional_asymmetry = True`):

```text
long_delta = max(0, delta_position)
short_delta = abs(min(0, delta_position))

long_transaction_cost = long_delta * long_transaction_cost_bps / 10000
short_transaction_cost = short_delta * short_transaction_cost_bps / 10000
long_slippage_cost = long_delta * slippage_bps / 10000
short_slippage_cost = short_delta * (slippage_bps * short_slippage_multiplier) / 10000
short_borrow_cost = position.clip(upper=0).abs() * short_borrow_cost_bps / 10000

long_execution_cost = long_transaction_cost + long_slippage_cost
short_execution_cost = short_transaction_cost + short_slippage_cost + short_borrow_cost

gross_strategy_return = executed_signal * asset_return
net_strategy_return = gross_strategy_return - long_execution_cost - short_execution_cost
strategy_return = net_strategy_return
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

Side-specific columns (when directional asymmetry is configured):

* `long_execution_cost` - transaction + slippage costs for long trades
* `short_execution_cost` - transaction + slippage + borrow costs for short trades

## Portfolio Execution

Portfolio execution is handled in `src/portfolio/execution.py`.

Current behavior:

* strategy run returns are aligned into a wide matrix
* the allocator produces weights
* gross portfolio return is the weighted sum of sleeve returns
* transaction cost and slippage are charged from portfolio turnover
* when directional asymmetry is configured, separate long/short turnover and costs are computed
* `portfolio_return` is the net return after execution friction
* `portfolio_equity_curve` compounds from `initial_capital`

Formula (symmetric case):

```text
gross_portfolio_return = sum(strategy_return_i * weight_i)
portfolio_transaction_cost = portfolio_turnover * transaction_cost_bps / 10000
portfolio_slippage_cost = portfolio_turnover * slippage_bps / 10000
net_portfolio_return = gross_portfolio_return - portfolio_transaction_cost - portfolio_slippage_cost
portfolio_return = net_portfolio_return
portfolio_equity_curve = initial_capital * (1.0 + portfolio_return).cumprod()
```

Formula (directional case, when `has_directional_asymmetry = True`):

```text
long_turnover = sum of positive weight changes
short_turnover = sum of absolute negative weight changes

long_transaction_cost = long_turnover * long_transaction_cost_bps / 10000
short_transaction_cost = short_turnover * short_transaction_cost_bps / 10000
long_slippage_cost = long_turnover * slippage_bps / 10000
short_slippage_cost = short_turnover * (slippage_bps * short_slippage_multiplier) / 10000
short_borrow_cost = average_short_exposure * short_borrow_cost_bps / 10000

gross_portfolio_return = sum(strategy_return_i * weight_i)
net_portfolio_return = gross_portfolio_return - long_transaction_cost - short_transaction_cost
                                               - long_slippage_cost - short_slippage_cost
                                               - short_borrow_cost
portfolio_return = net_portfolio_return
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

Side-specific columns (when directional asymmetry is configured):

* `portfolio_long_turnover` - absolute value of positive weight changes
* `portfolio_short_turnover` - absolute value of negative weight changes
* `portfolio_short_exposure` - cumulative short weight (sum of negative weights)
* `portfolio_long_transaction_cost` - transaction cost for long rebalancing
* `portfolio_short_transaction_cost` - transaction cost for short rebalancing
* `portfolio_long_slippage_proxy` - volatility proxy for long slippage
* `portfolio_short_slippage_proxy` - volatility proxy for short slippage
* `portfolio_long_slippage_cost` - slippage cost for long positions
* `portfolio_short_slippage_cost` - slippage cost for short positions
* `portfolio_short_borrow_cost` - daily borrow cost for short positions

## Configuration Fields

Execution settings currently support:

### Basic Fields

* `enabled`
* `execution_delay`
* `transaction_cost_bps`
* `slippage_bps`
* `fixed_fee`
* `fixed_fee_model`
* `slippage_model`
* `slippage_turnover_scale`
* `slippage_volatility_scale`

### Side-Aware Cost Fields (M22.5)

When modeling asymmetric long/short conditions, use:

* `long_transaction_cost_bps` - transaction cost for long trades (falls back to symmetric `transaction_cost_bps` if not set)
* `short_transaction_cost_bps` - transaction cost for short trades (falls back to symmetric `transaction_cost_bps` if not set)
* `short_slippage_multiplier` - multiplier on `slippage_bps` for short trades (default: 1.0)
* `short_borrow_cost_bps` - daily borrow/locate cost for short positions (applied to position size, not turnover)

Example configuration with directional asymmetry:

```yaml
execution:
  enabled: true
  execution_delay: 1
  transaction_cost_bps: 5.0        # symmetric default
  slippage_bps: 2.0                # symmetric default
  long_transaction_cost_bps: 3.0   # cheaper long trades
  short_transaction_cost_bps: 8.0  # expensive short trades
  short_slippage_multiplier: 1.5   # 1.5x slippage for shorts
  short_borrow_cost_bps: 50.0      # 0.5% annual borrow cost
```

### Short-Capacity and Availability Fields (M22.5)

When constraining short-side exposure, use:

* `max_short_weight_sum` - maximum total short exposure (e.g., 0.40 = 40% of portfolio)
* `short_availability_limit` - maximum exposure for hard-to-borrow securities
* `short_availability_policy` - how to handle unavailable shorts: `exclude`, `cap`, or `penalty`

Example configuration with short constraints:

```yaml
execution:
  enabled: true
  execution_delay: 1
  transaction_cost_bps: 5.0
  slippage_bps: 2.0
  max_short_weight_sum: 0.40          # max 40% short exposure
  short_availability_limit: 0.20      # max 20% hard-to-borrow
  short_availability_policy: "exclude" # exclude unavailable shorts
```

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
* side-specific cost fields are optional and backward-compatible
* short-capacity fields are optional; when absent, no constraints apply

## Deterministic Behavior

The execution model is deterministic by design:

* no random slippage model is used
* no stochastic fill process is introduced
* reruns with unchanged inputs and config produce the same outputs

That makes execution realism explicit without sacrificing reproducibility.

---

# 🔍 **M22.5: Interpretation & Stress-Analysis Layer**

The execution model includes a deterministic interpretation layer that explains **what short-side constraints actually did** to the portfolio. This section is automatically included when short constraints are configured.

## 🔹 Constraint Events & Binding Frequency

When `max_short_weight_sum` or `short_availability_limit` are configured, the system tracks:

### Constraint Events

```json
"constraint_events": {
  "max_short_weight_hits": 3,
  "availability_caps_triggered": 1,
  "availability_exclusions": 0
}
```

**What these mean:**

* `max_short_weight_hits` - how many periods the portfolio exceeded `max_short_weight_sum`
* `availability_caps_triggered` - how many periods short availability limit was hit (policy=cap)
* `availability_exclusions` - how many periods securities were excluded from portfolio (policy=exclude)

### Constraint Utilization

```json
"constraint_utilization": {
  "avg_short_utilization": 0.78,
  "max_short_utilization": 0.95
}
```

**Interpretation:**

* `avg_short_utilization` - average usage as ratio of limit (0.78 = 78% of max allowed)
* `max_short_utilization` - maximum single period (0.95 = 95% of max allowed)

Low utilization means the constraint is not binding and could be relaxed. High utilization (>0.95) means the constraint is nearly always active.

---

## 🔹 Side-Aware Cost Attribution

Breaks down execution costs by portfolio side (long vs short):

```json
"side_cost_attribution": {
  "long_cost_pct_total": 35.2,
  "short_cost_pct_total": 64.8,
  "short_borrow_cost_drag_pct": 42.1
}
```

**Interpretation:**

* `long_cost_pct_total` - transaction + slippage costs for long trades as % of total costs
* `short_cost_pct_total` - transaction + slippage + borrow costs for short trades as % of total
* `short_borrow_cost_drag_pct` - borrow cost specifically as % of total costs

**Decision usefulness:**

If `short_cost_pct_total` is high (>60%), consider:
- Reducing short exposure
- Using different borrow venues
- Increasing short position holding periods to amortize borrow costs

If `short_borrow_cost_drag_pct` is dominant (>40%), focus on borrow cost reduction.

---

## 🔹 Capacity Impact Analysis

When constraints are configured, this section estimates the portfolio impact **of** the constraints:

```json
"capacity_impact": {
  "return_delta": -0.0015,
  "turnover_delta": 0.05,
  "short_exposure_delta": -0.08,
  "baseline_friction": 0.0045,
  "constrained_friction": 0.0060
}
```

**Interpretation:**

* `return_delta` - additional friction cost due to constraints (negative = worse performance)
* `turnover_delta` - change in portfolio turnover (positive = more rebalancing)
* `short_exposure_delta` - reduction in average short exposure (negative = less shorting)
* `baseline_friction` - estimated costs without constraints
* `constrained_friction` - actual costs with constraints

**Example:** If `return_delta = -0.0015` and `short_exposure_delta = -0.08`, the constraint reduces short exposure by 8% but costs 15 bps in additional friction. Evaluate if the reduced short exposure justifies the cost.

---

## 🔹 Side-Stress Analysis Summary

Top-level summary that integrates all constraint insights:

```json
"side_stress_analysis": {
  "short_cost_drag_pct": 52.3,
  "constraint_impact_on_return": -0.0015,
  "constraint_binding_frequency": 0.60,
  "constraint_binding_events": 3,
  "max_short_utilization": 0.95
}
```

**Interpretation Guide:**

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| `short_cost_drag_pct` | <30% | Long-side costs dominate; short strategy is cheap |
| | 30-60% | Balanced cost structure; both sides matter |
| | >60% | Short-side costs dominate; consider cost reduction |
| `constraint_impact_on_return` | -10 to -1 bps | Mild constraint impact; acceptable trade-off |
| | <-10 bps | Strong constraint impact; reconsider limits |
| `constraint_binding_frequency` | <0.25 | Constraints almost never bind; too permissive |
| | 0.25-0.75 | Constraints are active; meaningful risk control |
| | >0.75 | Constraints nearly always bind; consider relaxing |
| `max_short_utilization` | <0.50 | Plenty of short capacity headroom |
| | 0.50-0.90 | Moderate utilization; current setup is efficient |
| | >0.90 | Very tight utilization; small capacity reduction |

---

## Artifacts and Metrics

All execution configuration and results are persisted deterministically:

### config.json

The full execution configuration is serialized in `config.json`, including all side-specific fields:

* `long_transaction_cost_bps`
* `short_transaction_cost_bps`
* `short_slippage_multiplier`
* `short_borrow_cost_bps`
* `max_short_weight_sum`
* `short_availability_limit`
* `short_availability_policy`

### manifest.json

The `execution` section includes:

* `config` - full execution configuration
* `directional_summary` - side-specific return contributions and metrics
* `summary` - aggregated metrics including:
  * `total_long_transaction_cost`
  * `total_short_transaction_cost`
  * `total_long_slippage_cost`
  * `total_short_slippage_cost`
  * `total_short_borrow_cost`
  * `long_turnover`, `short_turnover`
* `constraint_events` - binding frequency and event counts (M22.5)
* `constraint_utilization` - capacity usage ratios (M22.5)
* `side_cost_attribution` - cost breakdown by side (M22.5)
* `capacity_impact` - constraint impact deltas (M22.5, when constraints configured)
* `side_stress_analysis` - comprehensive stress summary (M22.5, when constraints configured)

Example manifest excerpt with interpretation layer:

```json
{
  "execution": {
    "config": {
      "enabled": true,
      "execution_delay": 1,
      "transaction_cost_bps": 5.0,
      "short_borrow_cost_bps": 50.0,
      "max_short_weight_sum": 0.40
    },
    "directional_summary": {
      "long_total_return": 0.0245,
      "short_total_return": -0.0032,
      "total_short_borrow_cost": 0.0018
    },
    "constraint_events": {
      "max_short_weight_hits": 3,
      "availability_caps_triggered": 0,
      "availability_exclusions": 0
    },
    "constraint_utilization": {
      "avg_short_utilization": 0.78,
      "max_short_utilization": 0.95
    },
    "side_cost_attribution": {
      "long_cost_pct_total": 35.2,
      "short_cost_pct_total": 64.8,
      "short_borrow_cost_drag_pct": 42.1
    },
    "capacity_impact": {
      "return_delta": -0.0015,
      "turnover_delta": 0.05,
      "short_exposure_delta": -0.08
    },
    "side_stress_analysis": {
      "short_cost_drag_pct": 64.8,
      "constraint_impact_on_return": -0.0015,
      "constraint_binding_frequency": 0.60,
      "constraint_binding_events": 3,
      "max_short_utilization": 0.95
    },
    "summary": {
      "total_transaction_cost": 0.0042,
      "total_long_transaction_cost": 0.0030,
      "total_short_transaction_cost": 0.0012,
      "total_short_borrow_cost": 0.0018
    }
  }
}
```

### CSV Outputs

Result CSVs include all execution cost columns:

* `portfolio_returns.csv` - includes transaction, slippage, and borrow costs with side breakdown
* `weights.csv` - portfolio weights including side-specific exposures

## Related Docs

* [research_validity_framework.md](research_validity_framework.md)
* [strict_mode.md](strict_mode.md)
* [runtime_configuration.md](runtime_configuration.md)
