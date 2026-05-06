# Catalog Query API and CLI

## Overview

`src/catalog/query.py` provides the M29 user-facing query layer for the unified
research catalog. It filters and summarizes in-memory `CatalogRecord` objects
produced by the read-only indexer, and it uses the lineage layer for related-run
queries.

The query layer is read-only. It does not write files, create a cache, update a
registry, repair artifacts, or execute research workflows.

## Python API

```python
from src.catalog import build_catalog, query_catalog, CatalogQuery

records = build_catalog("artifacts", repo_root=".")
strategies = query_catalog(
    records,
    CatalogQuery(
        run_types=("strategy",),
        statuses=("completed",),
        min_metric=("sharpe_ratio", 1.0),
    ),
)
```

Direct keyword filtering is also available:

```python
from src.catalog import filter_catalog_records, summarize_catalog

filtered = filter_catalog_records(
    records,
    strategy_name="momentum_v1",
    start_ts="2024-01-01",
    end_ts="2024-12-31",
)
summary = summarize_catalog(filtered)
```

Serialization helpers:

```python
from src.catalog import records_to_dicts, records_to_rows

json_ready = records_to_dicts(filtered)
table_ready = records_to_rows(filtered)
```

## CLI Usage

```bash
python -m src.cli.query_catalog --artifacts-root artifacts --format table
python -m src.cli.query_catalog --artifacts-root artifacts --format json
python -m src.cli.query_catalog --run-type strategy --status completed
python -m src.cli.query_catalog --strategy-name momentum_v1
python -m src.cli.query_catalog --portfolio-name risk_parity
python -m src.cli.query_catalog --min-metric sharpe_ratio 1.0
python -m src.cli.query_catalog --include-templates
python -m src.cli.query_catalog --summary
```

`--format json` prints deterministic JSON with sorted keys and two-space
indentation. `--format table` prints a stable tab-separated table with fixed
column order.

`--limit N` limits printed records. Summary output ignores table formatting and
prints summary JSON for the matching records.

## Filter Semantics

Supported filters include:

- `--run-type`
- `--status`
- `--strategy-name`
- `--portfolio-name`
- `--allocator-name`
- `--alpha-model-name`
- `--regime-method`
- `--campaign-id`
- `--scenario-id`
- `--min-metric NAME VALUE`
- `--max-metric NAME VALUE`
- `--metric-equals NAME VALUE`
- `--start-ts`
- `--end-ts`

Name, ID, run type, and status filters are exact string matches.

Metric filters read scalar values from `record.metrics_summary`. Records missing
the requested metric are excluded from metric-filtered results.

Time filters use lexicographic comparison of ISO-like strings. They are intended
for normalized values such as `YYYY-MM-DD` or ISO-8601 timestamps; mixed formats
may not sort chronologically.

## Template and Unknown Behavior

By default, `portfolio_template` records are excluded because they represent
metadata/template registry entries, not completed research runs.

Use `include_templates=True` in Python or `--include-templates` in the CLI to
include them.

Unknown records are included by default. Use `include_unknown=False` in Python
or `--exclude-unknown` in the CLI to exclude records where `run_type == "unknown"`
or `status == "unknown"`.

## Lineage Queries

Related-record helpers use `src.catalog.lineage.build_lineage_edges()` instead
of re-deriving relationships:

```python
from src.catalog import get_upstream_records, get_downstream_records

upstream = get_upstream_records(portfolio_record, records, repo_root=".")
downstream = get_downstream_records(strategy_record, records, repo_root=".")
```

CLI examples:

```bash
python -m src.cli.query_catalog --related portfolio_run_1 --direction upstream --format json
python -m src.cli.query_catalog --related strategy_run_1 --direction downstream --edge-type portfolio_component
```

`--related` accepts either a `run_id` or `catalog_id`. Upstream means records
feeding into the target record. Downstream means records that depend on the
target record.

## Read-Only Guarantees

The query API and CLI only call:

- `build_catalog()` to scan current artifact/catalog state
- `build_lineage_edges()` for relationship discovery
- in-memory filter, sort, summary, and serialization helpers

They do not write, append, delete, move, lock, register, cache, or execute
anything.

## Limitations and Non-Goals

- No persistent catalog cache, database, export file, or search backend.
- No dashboard or notebook examples.
- No new registry writer or execution wrapper.
- No new lineage derivation beyond calling the lineage layer.
- No artifact repair, schema migration, or marker/manifest mutation.
- Time filtering is lexicographic and assumes normalized ISO-like strings.
