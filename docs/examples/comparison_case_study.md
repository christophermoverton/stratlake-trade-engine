# Strategy Comparison Case Study

## Overview

Comparing strategies is useful because strong standalone results do not always
mean a strategy is the best fit. A side-by-side view helps answer practical
questions:

* which strategy delivered the strongest risk-adjusted return
* which one was more directional or more selective
* whether a custom strategy is meaningfully better than a simple baseline

This case study uses real artifacts already present in the repository and calls
out one current reproducibility caveat in the workspace.

## Strategies Compared

The intended comparison set for Milestone 5 is:

* `momentum_v1`
* `mean_reversion_v1`
* `buy_and_hold_v1`

## Evaluation Setup

Requested comparison command:

```powershell
python -m src.cli.compare_strategies --strategies momentum_v1 mean_reversion_v1 buy_and_hold_v1
```

What is currently available in this workspace:

* a stored single-run leaderboard for `momentum_v1` and `mean_reversion_v1`
* a stored walk-forward artifact for `buy_and_hold_v1`

What is not currently available:

* a stored single-run `buy_and_hold_v1` artifact that matches the single-run
  leaderboard mode
* a fresh three-strategy single-run comparison artifact

Current reproducibility caveat:

Running the exact command above now fails on this workspace because the current
`features_daily` dataset no longer includes a supported return column for fresh
single-run execution. The existing stored leaderboard therefore remains the
latest valid single-run comparison artifact.

## Leaderboard Summary

Current single-run leaderboard artifact:
`artifacts/strategies/leaderboard.json`

| Rank | Strategy | `total_return` | `annualized_return` | `sharpe_ratio` | `max_drawdown` | `win_rate` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `momentum_v1` | `0.507750` | `0.220640` | `16.212535` | `0.004283` | `0.797688` |
| 2 | `mean_reversion_v1` | `0.005803` | `0.002813` | `0.997992` | `0.002411` | `0.117534` |

Baseline context from the latest stored `buy_and_hold_v1` artifact:

* source artifact:
  `artifacts/strategies/20260320T154611644147Z_buy_and_hold_v1/metrics.json`
* evaluation mode: `walk_forward`
* `total_return`: `0.019292`
* `annualized_return`: `2.332852`
* `sharpe_ratio`: `3.334314`
* `max_drawdown`: `0.020000`
* `win_rate`: `0.500000`

## Comparative Insights

`momentum_v1` is the clear leader in the available single-run leaderboard. It
combined the highest return with the highest Sharpe ratio while keeping
drawdown very small. In this stored window, the market behavior favored steady
trend capture over short-horizon reversal trades.

`mean_reversion_v1` behaved very differently. It traded less, stayed in the
market far less often, and delivered only a small positive return. That makes
it a more selective and defensive profile, but it also means it lagged badly in
an environment where directional persistence seemed stronger than reversion.

The latest available `buy_and_hold_v1` baseline artifact gives useful context
even though it is not part of the current single-run leaderboard. Its
walk-forward Sharpe ratio of `3.334314` is materially below the stored
single-run momentum result and above the stored single-run mean-reversion
result. That suggests momentum was the strongest of the three implemented
approaches in the best available evidence, but the baseline still sets an
important reference point for future apples-to-apples reruns.

## Key Takeaways

The main takeaway is that `momentum_v1` currently has the strongest stored
evidence in the repository. It produced the best balance of return and risk in
the available single-run comparison.

`mean_reversion_v1` looks more cautious and selective, but it did not convert
that selectivity into competitive returns in the stored evaluation window.

For a fully aligned Milestone 5 three-strategy case study, the next step is not
new code. It is restoring a `features_daily` dataset version that supports
fresh single-run execution so the exact comparison command above can regenerate
an apples-to-apples leaderboard including `buy_and_hold_v1`.
