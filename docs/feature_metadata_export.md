# Feature Metadata Export

## Overview

StratLake emits a deterministic metadata artifact describing the engineered
feature datasets produced by the feature pipeline.

This export currently covers:

* `features_daily`
* `features_1m`

The metadata artifact is intended to improve dataset introspection,
discoverability, and analytics-oriented observability without changing feature
computation logic or storage layout.

---

## Output Location

Feature metadata is written to:

```text
artifacts/features/feature_metadata.json
```

When the feature pipeline is run repeatedly on the same underlying feature
datasets, the file is regenerated deterministically.

---

## Included Metadata

The export includes one summary object per feature dataset.

### Dataset information

Each dataset summary includes:

* `dataset_name`
* `source_dataset`
* `feature_list`
* `feature_aliases`
* `feature_count`

### Schema information

Each dataset summary also includes:

* `schema.column_names`
* `schema.feature_columns`
* `schema.data_types`

### Dataset metrics

The metadata includes:

* `metrics.row_count`
* `metrics.symbol_coverage`
* `metrics.date_range.start`
* `metrics.date_range.end`

### Feature statistics

For each engineered feature column, the export includes:

* `null_pct`
* `min`
* `max`

Minimum and maximum values are computed only from finite numeric values. If no
finite values are present for a feature, `min` and `max` are written as `null`.

---

## Determinism

The metadata export is designed to be reproducible across repeated runs on
identical inputs.

Deterministic behavior is achieved by:

* fixed dataset ordering
* stable feature column sorting
* stable symbol and date aggregation
* sorted JSON keys in the written artifact
* metadata generation from the persisted feature datasets and registry

---

## Pipeline Integration

Feature metadata is generated automatically after feature datasets are built in:

* `run_daily_feature_pipeline(...)`
* `run_minute_feature_pipeline(...)`

The pipeline flow is:

```text
load bars -> compute features -> write feature parquet -> write QA summaries -> write feature metadata
```

This keeps the metadata aligned with the current persisted feature datasets.

---

## Registry Relationship

The export reads dataset-level metadata from the feature registry:

```text
configs/features.yml
```

The registry is used for:

* source dataset mapping
* declared feature list
* declared backward-compatible feature aliases

Observed schema and dataset statistics are loaded from the persisted parquet
feature datasets rather than inferred only from registry configuration.

---

## Example Usage

### Automatic generation via the pipeline

```python
from src.pipeline.feature_pipeline import run_daily_feature_pipeline

features = run_daily_feature_pipeline(
    symbols=["AAPL", "MSFT"],
    start_date="2025-01-01",
    end_date="2025-03-01",
)
```

After the pipeline completes, metadata is refreshed at:

```text
artifacts/features/feature_metadata.json
```

### Direct generation

```python
from src.data.feature_metadata import export_feature_metadata

path = export_feature_metadata()
print(path)
```

---

## Related Documentation

* [docs/feature_dataset_contract.md](feature_dataset_contract.md)
* [docs/load_features.md](load_features.md)
* [docs/dataset_qa_metrics.md](dataset_qa_metrics.md)
* [configs/features.yml](../configs/features.yml)
