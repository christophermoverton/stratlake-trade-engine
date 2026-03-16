# Feature Dataset Loader Usage

`src.data.load_features.load_features()` provides a reusable way to read engineered feature datasets from partitioned Parquet storage for analytics and research workflows.

## Supported Datasets

The loader supports:

* `features_daily`
* `features_1m`

## Supported Storage Layouts

The loader is compatible with both of these layouts:

```text
data/features/<dataset>/symbol=<SYMBOL>/date=<YYYY-MM-DD>/*.parquet
```

and:

```text
<FEATURES_ROOT>/<dataset>/symbol=<SYMBOL>/...
```

This lets it work with the issue target layout as well as the repository's current feature output conventions.

## Function Signature

```python
load_features(
    dataset: str,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
)
```

Arguments:

* `dataset` -> feature dataset name, either `features_daily` or `features_1m`
* `start` -> inclusive start date in `YYYY-MM-DD`
* `end` -> exclusive end date in `YYYY-MM-DD`
* `symbols` -> optional symbol filter list

The function returns a pandas `DataFrame`.

## Behavior

The loader:

* recursively scans partitioned Parquet files
* uses DuckDB with Hive partition discovery
* filters by `symbol` and `date` when requested
* sorts results deterministically by `(symbol, ts_utc)`
* returns an empty `DataFrame` when no matching files or rows exist

## Example: Load Daily Features

```python
from src.data.load_features import load_features

daily_features = load_features(
    dataset="features_daily",
    start="2025-01-01",
    end="2025-03-01",
    symbols=["AAPL", "MSFT"],
)

print(daily_features.head())
```

## Example: Load 1-Minute Features

```python
from src.data.load_features import load_features

minute_features = load_features(
    dataset="features_1m",
    start="2025-01-02",
    end="2025-01-03",
    symbols=["AAPL"],
)

print(minute_features.head())
```

## Path Resolution

By default, `FeaturePaths` resolves its root from `FEATURES_ROOT` and falls back to `data` when the environment variable is unset.

If you need explicit path control, pass a custom `FeaturePaths` instance:

```python
from pathlib import Path

from src.data.load_features import FeaturePaths, load_features

paths = FeaturePaths(root=Path("data"))
df = load_features("features_daily", paths=paths)
```

## Notes

* `end` is exclusive, matching the existing curated bar loaders
* symbol filters are normalized to uppercase
* `ts_utc` is normalized to UTC in the returned DataFrame
