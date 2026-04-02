# Feature Dataset Contract

## Overview

This document defines the shared contract for feature datasets produced by StratLake's feature engineering layer.

It applies to current feature outputs such as:

* `features_daily`
* `features_1m`

This contract is intended to guide future feature additions without changing existing feature computation logic or dataset naming.

---

## Purpose

The feature dataset contract exists to ensure:

* reproducibility across repeated runs on identical inputs
* consistency across feature datasets and timeframes
* reliable downstream use in analytics, research, and strategy development

---

## Required Schema

All feature datasets must include the following core columns:

* `symbol`
* `ts_utc`
* `timeframe`
* `date`
* one or more engineered columns prefixed with `feature_`

Expected semantics:

| Column | Description |
| --- | --- |
| `symbol` | Instrument identifier for the row. |
| `ts_utc` | Canonical event timestamp in UTC for the observation represented by the row. |
| `timeframe` | Timeframe identifier for the dataset row, for example `1D` or `1Min`. |
| `date` | Calendar date associated with the row, stored in `YYYY-MM-DD` form. |
| `feature_*` | Engineered feature columns derived from curated source data. |

Feature datasets may include additional non-feature metadata columns when needed, but engineered feature values must always use the `feature_` prefix.

---

## Primary Key

The logical primary key for every feature dataset is:

```text
(symbol, ts_utc, timeframe)
```

Contract expectations:

* each row must be uniquely identified by this key
* primary key columns must be present for every feature dataset
* downstream consumers may rely on this key for joins, validation, and reproducibility checks

---

## Naming Convention

All engineered feature columns must begin with:

```text
feature_
```

Examples:

* `feature_return_1d`
* `feature_sma_20`
* `feature_volume_zscore_30`

This prefix is the repository-level signal that a column is part of the engineered feature surface rather than source market data or operational metadata.

For lookback-style features, the canonical pattern is:

```text
feature_<name>_<window>
```

Examples:

* `feature_sma_20`
* `feature_sma_50`

Legacy aliases such as `feature_sma20` may still be accepted by config-facing tooling during migration, but new datasets, registry entries, and documentation should use the canonical underscore form.

---

## Row Alignment

Feature datasets must preserve row-level alignment with the source curated dataset used to compute them.

That means:

* feature generation must not reorder rows in a way that breaks correspondence with source observations
* each output row represents the same `(symbol, ts_utc, timeframe)` observation as the input curated row
* feature computation may append engineered columns, but it must not redefine the observation represented by the row

This alignment requirement allows downstream systems to join curated inputs and engineered outputs without ambiguity.

---

## Behavioral Rules

### Determinism

Feature transformations should be deterministic for identical inputs.

Given the same validated curated source data and the same feature logic, the resulting feature values should be identical across repeated runs.

### No Lookahead

Features must not use future observations relative to `ts_utc`.

For any row, feature values must be derived only from information available at or before that row's timestamp. Future bars, returns, volumes, or other post-`ts_utc` observations must not influence the row's engineered values.

### Rolling and Null Behavior

Rolling or lookback-based features should return `NaN` until sufficient history exists to compute the value correctly.

This rule is preferred over silently backfilling, truncating windows, or substituting default values that would obscure the true warm-up boundary.

---

## Dataset Coverage

This contract applies to all feature datasets produced by the feature engineering layer, including current datasets such as:

* `features_daily`
* `features_1m`

Future feature datasets should adopt the same schema, naming, key, and behavioral guarantees unless an explicitly documented exception is introduced at the repository level.

---

## Guidance for Future Feature Additions

When adding new engineered features:

* retain the core columns `symbol`, `ts_utc`, `timeframe`, and `date`
* preserve uniqueness of `(symbol, ts_utc, timeframe)`
* name every engineered column with the `feature_` prefix
* maintain row alignment with the curated source dataset
* avoid lookahead or any dependence on future observations
* emit `NaN` for rolling features before the required history window is available
* preserve deterministic behavior for identical inputs

This document is a dataset contract and specification. It does not change or redefine the current feature pipeline implementation.
