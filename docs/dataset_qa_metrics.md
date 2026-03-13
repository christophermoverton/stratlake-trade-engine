# Dataset QA Metrics

## Overview

StratLake emits deterministic QA summary artifacts for engineered feature datasets produced by the feature pipeline.

This QA layer currently applies to:

* `features_daily`
* `features_1m`

The goal is to provide lightweight structural and statistical health checks for feature outputs without changing feature computation logic.

---

## Output Location

Feature dataset QA artifacts are written to:

```text
artifacts/qa/features/
```

Generated files:

```text
qa_features_summary_by_symbol.csv
qa_features_summary_global.csv
```

When the CLI is used, the output root follows the configured `ARTIFACTS_ROOT`.

---

## QA Checks

The feature QA summaries include these checks.

### Structural checks

* duplicate full-row count
* duplicate primary-key count using `(symbol, ts_utc, timeframe)`
* total row count
* symbol coverage against requested symbols

### Data quality checks

* null percentage for every `feature_*` column
* infinite value count for every `feature_*` column
* minimum finite value for every `feature_*` column
* maximum finite value for every `feature_*` column

---

## Summary Fields

Both QA exports include core metadata such as:

* `dataset_name`
* `timeframe`
* `total_rows`
* `duplicate_row_count`
* `duplicate_key_count`
* `expected_symbol_count`
* `observed_symbol_count`
* `missing_symbol_count`
* `symbol_coverage_pct`
* `missing_symbols`
* `dataset_status`

For every engineered feature column, the summaries also include:

* `<feature_name>_null_pct`
* `<feature_name>_inf_count`
* `<feature_name>_min`
* `<feature_name>_max`

`qa_features_summary_by_symbol.csv` includes one row per `(dataset_name, timeframe, symbol)`.

`qa_features_summary_global.csv` includes one row per `(dataset_name, timeframe)`.

---

## Dataset Status

The QA status is intentionally simple:

* `PASS` -> no duplicates, no infinite values, and no feature nulls
* `WARN` -> no duplicates or infinite values, but at least one feature column contains nulls
* `FAIL` -> empty dataset, duplicate rows, duplicate keys, or infinite values detected

This status is meant to support quick observability and basic downstream gating.

---

## Determinism

The QA outputs are designed to be reproducible across runs on identical inputs.

Deterministic behavior is achieved by:

* stable dataset and symbol sorting
* fixed CSV filenames
* overwrite-by-key behavior for repeated runs of the same dataset/timeframe
* metric computation directly from the produced feature dataframe

---

## Pipeline Integration

Feature QA runs automatically after feature datasets are built in:

* `run_daily_feature_pipeline(...)`
* `run_minute_feature_pipeline(...)`

The pipeline flow is:

```text
load bars -> compute features -> write feature parquet -> write feature QA summaries
```

This keeps QA generation aligned with the feature outputs that were actually produced.
