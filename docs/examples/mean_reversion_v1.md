# mean_reversion_v1

## Overview

`mean_reversion_v1` is a counter-trend strategy on the `features_daily`
dataset. It looks for prices that have moved unusually far away from their
recent average and bets on a move back toward that average.

The intuition is that short-term overreactions can fade. When price gets too
far above normal, the strategy leans short. When price falls too far below
normal, it leans long.

## Signal Definition

Current implementation uses a rolling z-score on close prices:

* rolling mean = average close over the last 20 days
* rolling standard deviation = close-price volatility over the same 20 days
* z-score = `(close - rolling mean) / rolling std`
* signal = `1` when z-score is below `-1.0`
* signal = `-1` when z-score is above `1.0`
* signal = `0` otherwise

In plain language, the strategy buys statistically weak closes and shorts
statistically strong closes, expecting reversion back toward the recent mean.

## Evaluation Setup

The cited run uses the latest stored single-run artifact for this strategy:

* run id: `20260320T072236135299Z_mean_reversion_v1`
* dataset: `features_daily`
* timeframe: `1D`
* artifact evaluation mode: `single`
* artifact data range: `2022-01-03` to `2023-12-29`

The repository also defines a reusable walk-forward setup in
`configs/evaluation.yml`:

* mode: `rolling`
* timeframe: `1d`
* train window: `12M`
* test window: `3M`
* step: `3M`
* overall span: `2022-01-01` to `2024-01-01`

CLI command for the cited single-run workflow:

```powershell
python -m src.cli.run_strategy --strategy mean_reversion_v1
```

Optional walk-forward command:

```powershell
python -m src.cli.run_strategy --strategy mean_reversion_v1 --evaluation
```

## Performance Summary

Source artifact:
`artifacts/strategies/20260320T072236135299Z_mean_reversion_v1/metrics.json`

| Metric | Value |
| --- | ---: |
| `total_return` | `0.005803` |
| `annualized_return` | `0.002813` |
| `sharpe_ratio` | `0.997992` |
| `max_drawdown` | `0.002411` |
| `win_rate` | `0.117534` |

Additional context from the same artifact:

* exposure: `21.35%`
* hit rate: `0.611111`
* profit factor: `1.620963`

## Key Insights

This stored run was much more selective than momentum. The strategy spent only
about one-fifth of the period in the market and produced a small positive total
return with a modest Sharpe ratio.

That profile fits a mean-reversion system: it waits for larger dislocations,
trades less often, and can look quiet during strong directional markets where
price keeps trending instead of snapping back.

## Risks and Limitations

Mean reversion is sensitive to regime changes. It tends to struggle when strong
trends continue longer than expected, because stretched prices can stay
stretched.

It can also be noisy around the entry threshold. Small changes in volatility or
price level can move the z-score in and out of signal territory, which makes
results sensitive to threshold choice and data quality.

## How to Run

Single run:

```powershell
python -m src.cli.run_strategy --strategy mean_reversion_v1
```

Walk-forward evaluation:

```powershell
python -m src.cli.run_strategy --strategy mean_reversion_v1 --evaluation
```
