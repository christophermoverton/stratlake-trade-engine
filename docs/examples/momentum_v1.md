# momentum_v1

## Overview

`momentum_v1` is a simple trend-following strategy on the `features_daily`
dataset. It compares a short rolling average of returns with a longer rolling
average and stays long when recent returns are stronger than the broader trend.
If the short trend weakens below the long trend, it flips short. If they are
equal, it stays flat.

The intuition is straightforward: assets that have been moving consistently in
one direction often keep moving that way for a while, especially during stable
trending periods.

## Signal Definition

Current implementation:

* short trend = rolling mean of returns over 5 days
* long trend = rolling mean of returns over 20 days
* signal = `1` when short trend > long trend
* signal = `-1` when short trend < long trend
* signal = `0` when they are equal

In plain language, the strategy asks whether recent returns are stronger or
weaker than the longer-run average and positions in that direction.

## Evaluation Setup

The cited run uses the repository's `features_daily` dataset and the latest
stored single-run artifact for this strategy:

* run id: `20260320T072236052407Z_momentum_v1`
* dataset: `features_daily`
* timeframe: `1D`
* artifact evaluation mode: `single`
* artifact data range: `2022-01-03` to `2023-12-29`

The repository also includes a default walk-forward configuration in
`configs/evaluation.yml`:

* mode: `rolling`
* timeframe: `1d`
* train window: `12M`
* test window: `3M`
* step: `3M`
* overall span: `2022-01-01` to `2024-01-01`

CLI command for the cited single-run workflow:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

Optional walk-forward command:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

## Performance Summary

Source artifact:
`artifacts/strategies/20260320T072236052407Z_momentum_v1/metrics.json`

| Metric | Value |
| --- | ---: |
| `total_return` | `0.507750` |
| `annualized_return` | `0.220640` |
| `sharpe_ratio` | `16.212535` |
| `max_drawdown` | `0.004283` |
| `win_rate` | `0.797688` |

## Key Insights

Momentum performed very well in this stored run. The strategy was invested most
of the time, with `98.65%` exposure, and captured a strong compounded return
with very small drawdowns. That combination suggests the evaluation window was
friendly to persistent trends and did not punish direction changes much.

This kind of profile is usually strongest when assets move in sustained,
orderly trends and weaker when price action chops sideways.

## Risks and Limitations

Momentum depends on trend persistence. It can struggle when markets reverse
quickly, oscillate in narrow ranges, or generate repeated false breaks.

Because the strategy uses rolling averages of returns, it also reacts with some
lag. That lag can help avoid noise, but it can also make entries late and exits
slow after trend exhaustion.

## How to Run

Single run:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

Walk-forward evaluation:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```
