# Baseline Strategies

## Overview

Baseline strategies provide deterministic benchmark runs through the same
research pipeline used by the existing strategy implementations. They are meant
to answer a simple question: does a research strategy actually improve on a
basic reference?

All baselines conform to `BaseStrategy`, run through the standard signal,
backtest, metrics, artifact, and walk-forward flow, and require no special
handling in the CLI or metrics layer.

---

## Available Baselines

### `buy_and_hold_v1`

Purpose:

* reference for passive long exposure

Behavior:

* stays flat until the first row with a valid asset return
* switches to `1` on that row
* remains long for the rest of the dataset

Parameters:

* none

### `sma_crossover_v1`

Purpose:

* simple trend-following reference

Behavior:

* reconstructs a synthetic price path from the dataset's supported return column
* computes fast and slow simple moving averages
* emits `1` when `fast_sma > slow_sma`
* emits `0` otherwise

Parameters:

* `fast_window`
* `slow_window`

### `seeded_random_v1`

Purpose:

* reproducible naive benchmark for checking whether a strategy beats random long-or-flat positioning

Behavior:

* uses a local seeded RNG
* emits deterministic random `0` or `1` signals on rows with valid returns
* remains flat on rows without a usable return

Parameters:

* `seed`

---

## Example Config

```yaml
buy_and_hold_v1:
  dataset: features_daily
  parameters: {}

sma_crossover_v1:
  dataset: features_daily
  parameters:
    fast_window: 5
    slow_window: 20

seeded_random_v1:
  dataset: features_daily
  parameters:
    seed: 7
```

---

## CLI Usage

Single run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy buy_and_hold_v1
```

Walk-forward:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy sma_crossover_v1 --evaluation
```
