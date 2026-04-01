# Research Validity Framework

## Overview

StratLake's research-validity framework is the set of checks that make strategy
and portfolio results reviewable instead of merely executable.

The framework is layered:

```text
inputs
  -> temporal integrity
  -> execution realism
  -> sanity checks
  -> promotion gates
  -> artifact QA
  -> consistency validation
```

Use this document as the high-level map, then drill into:

* [execution_model.md](execution_model.md)
* [strict_mode.md](strict_mode.md)
* [runtime_configuration.md](runtime_configuration.md)
* [research_integrity_and_qa.md](research_integrity_and_qa.md)

## Validation Layers

### Temporal integrity

Research integrity checks protect against lookahead and row-identity drift.

Current checks include:

* sorted, unique `(symbol, ts_utc)` ordering
* exact signal alignment with the input frame
* signal-domain validation
* no same-bar execution
* cross-layer key preservation between features, signals, and backtest outputs

At the strategy layer, executed positions are always lagged by the configured
execution delay before returns are applied.

### Execution realism

Execution settings make the backtest more realistic without introducing
nondeterminism.

Current controls:

* `execution_delay`
* `transaction_cost_bps`
* `slippage_bps`
* execution enabled/disabled toggle

Strategy runs record fields such as:

* `executed_signal`
* `gross_strategy_return`
* `transaction_cost`
* `slippage_cost`
* `execution_friction`
* `net_strategy_return`

Portfolio runs record the portfolio equivalents plus weight-change and turnover
columns.

See [execution_model.md](execution_model.md).

### Sanity checks

Sanity checks are deterministic thresholds that flag suspicious but still
parseable output.

They currently evaluate:

* extreme single-period returns
* unusually high annualized return or Sharpe ratio
* extreme equity multiples
* low-volatility / high-performance combinations
* suspiciously smooth return paths

Sanity checks can either warn or fail depending on strictness.

### Portfolio validation

Portfolio validation adds structural constraints on top of strategy validity.

Current constraints include:

* target weight sum
* target net exposure
* gross exposure and leverage limits
* optional sleeve-weight bounds
* return-stream and equity-curve consistency
* traceability between sleeve returns, weights, and aggregate returns

### Promotion gates

Promotion gates are deterministic, auditable pass/fail checks applied to saved
alpha, strategy, and portfolio runs.

They complement sanity checks and QA rather than replacing them.

Current gate definitions support:

* metric thresholds such as minimum `ic_ir` or minimum `sharpe_ratio`
* upper bounds such as maximum `max_drawdown`
* sample-size requirements such as minimum `n_periods`
* sanity-derived limits such as zero issue-count requirements
* split-stability thresholds derived from per-split metrics

Configured runs persist a stable `promotion_gates.json` artifact and mirror a
compact promotion summary into the registry-backed unified review flow.

### Consistency validation

Consistency validation cross-checks saved artifacts after persistence.

Current checks include:

* manifest inventory versus files on disk
* metrics versus recomputed results
* QA summaries versus diagnostics and metrics
* registry entry alignment
* walk-forward split inventories and aggregate summaries

## Strict Mode

Strict mode is the policy layer on top of the checks above.

It does not add separate validation logic. Instead, it changes enforcement:

* flagged sanity issues can become blocking failures
* portfolio validation failures remain blocking
* strict-mode failures stop strategy and portfolio CLIs before persistence

See [strict_mode.md](strict_mode.md).

## Runtime Configuration

The framework is controlled through one normalized runtime contract with these
sections:

* `execution`
* `sanity`
* `portfolio_validation`
* `strict_mode`

Repository defaults, config values, and CLI overrides are merged
deterministically. Effective settings are persisted with completed runs for
auditability.

See [runtime_configuration.md](runtime_configuration.md).

## Review Workflow

For a practical review pass:

1. Start with `manifest.json`.
2. Check `qa_summary.json`.
3. Inspect `promotion_gates.json` when present.
4. Inspect `metrics.json`.
5. Confirm execution-friction and turnover fields.
6. Review `equity_curve.csv` or `portfolio_returns.csv`.

That sequence keeps trust and realism checks ahead of headline metrics.
