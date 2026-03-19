# Evaluation Split Configuration

## Overview

Milestone 4 introduces deterministic evaluation split generation for future
walk-forward strategy validation.

This layer does not run backtests across splits yet. It only converts
configuration into serializable train/test window definitions that can be used
later by the research runner and artifact logging.

Implementation lives in:

```text
configs/evaluation.yml
src/config/evaluation.py
src/research/splits.py
```

---

## Supported Modes

The evaluation split framework supports three modes:

* `fixed`
* `rolling`
* `expanding`

### Fixed

Produces exactly one split from explicit boundaries:

* `train_start`
* `train_end`
* `test_start`
* `test_end`

### Rolling

Uses an overall evaluation window and moves both train and test windows forward
by the configured `step`.

### Expanding

Keeps the initial `train_start` fixed, expands `train_end` by `step`, and rolls
the test window forward by `step`.

---

## Boundary Semantics

All generated windows use half-open date intervals:

* `start` is inclusive
* `end` is exclusive

Example:

```text
[2022-01-01, 2022-04-01)
```

This keeps adjacent train/test windows deterministic and avoids overlap when
splits are chained together.

---

## Config Shape

Typical rolling configuration:

```yaml
evaluation:
  mode: rolling
  timeframe: 1d
  start: "2022-01-01"
  end: "2024-01-01"
  train_window: 12M
  test_window: 3M
  step: 3M
```

Fixed mode uses explicit boundaries instead:

```yaml
evaluation:
  mode: fixed
  timeframe: 1d
  train_start: "2022-01-01"
  train_end: "2023-01-01"
  test_start: "2023-01-01"
  test_end: "2023-04-01"
```

Supported duration units:

* `D` for days
* `W` for weeks
* `M` for months
* `Y` for years

---

## Output Format

Each generated split is returned as structured metadata with:

* `split_id`
* `mode`
* `train_start`
* `train_end`
* `test_start`
* `test_end`

Example:

```python
{
    "split_id": "rolling_0001",
    "mode": "rolling",
    "train_start": "2022-04-01",
    "train_end": "2022-10-01",
    "test_start": "2022-10-01",
    "test_end": "2023-01-01",
}
```

---

## Validation Rules

Split generation fails clearly when configuration is invalid, including:

* missing required fields
* unsupported modes
* invalid dates
* zero or negative window sizes
* overlapping fixed windows
* evaluation ranges that cannot produce at least one split

This keeps split generation deterministic and safe for later runner
integration.
