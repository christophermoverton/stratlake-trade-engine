# Catalog Notebook Workflows

## Purpose

M29 makes the unified research catalog usable from notebooks and lightweight
example scripts. The catalog workflow is read-only and artifact-first: it reads
existing artifact roots, registries, manifests, marker files, metrics, lineage
metadata, and validation signals, then returns in-memory records for
interactive inspection.

This is complementary to M28 notebook execution. M28 notebooks call the
existing execution APIs and inspect their returned artifacts. M29 catalog
notebooks start from artifacts that already exist and use the catalog APIs to
discover, filter, relate, validate, and reuse those artifacts.

## Recommended Pattern

1. Build the catalog from the existing artifact root.
2. Query records by run type, lifecycle status, name, date range, or metrics.
3. Inspect upstream and downstream lineage relationships.
4. Validate catalog integrity in memory.
5. Load relative artifact paths for follow-up notebook analysis.

```python
from src.catalog import build_catalog, CatalogQuery, query_catalog, records_to_rows

records = build_catalog("artifacts", repo_root=".")
completed_strategies = query_catalog(
    records,
    CatalogQuery(run_types=("strategy",), statuses=("completed",)),
)
rows = records_to_rows(completed_strategies)
```

For table-oriented notebook display:

```python
import pandas as pd

df = pd.DataFrame(records_to_rows(completed_strategies))
```

Pandas is optional. The catalog helpers return plain Python lists and
dictionaries, so notebooks can display rows without adding a notebook runtime
dependency to CI.

## Lineage Inspection

Use the lineage-aware query helpers instead of rebuilding relationship logic in
notebook cells.

```python
from src.catalog import get_downstream_records, get_upstream_records

target = completed_strategies[0]
upstream = get_upstream_records(target, records, repo_root=".")
downstream = get_downstream_records(target, records, repo_root=".")

records_to_rows(upstream)
records_to_rows(downstream)
```

The helpers consume existing lineage metadata through the M29 lineage layer.
They do not write lineage files, repair references, or infer new workflow
contracts.

## Validation

Validation is also read-only. It returns an in-memory report that can be
displayed, filtered, or summarized in a notebook.

```python
from src.catalog import validate_catalog

report = validate_catalog(records, repo_root=".")
{
    "records": report.total_records,
    "artifacts": report.total_artifacts,
    "errors": report.error_count,
    "warnings": report.warning_count,
    "by_code": report.summary.get("by_code", {}),
}
```

Validation does not execute research workflows, create reports, write caches,
or modify manifests and marker files.

## Artifact Reuse

Use catalog rows to locate artifacts for follow-up analysis with relative paths.

```python
rows = records_to_rows(completed_strategies)
artifact_roots = [row["artifact_root"] for row in rows if row["artifact_root"]]
```

Follow-up notebook cells can read files under those artifact roots with normal
project-relative paths. Keep derived exports as local user actions and avoid
committing generated notebook outputs.

## Examples

The script companion is CI-testable and safe in empty repositories:

```powershell
python docs/examples/m29_catalog_driven_research_workflow.py
```

The notebook mirrors the same workflow:

```text
docs/examples/notebooks/m29_catalog_driven_research_workflow.ipynb
```

Both examples use:

- `build_catalog`
- `query_catalog`
- `CatalogQuery`
- `records_to_rows`
- `records_to_dicts`
- `get_upstream_records`
- `get_downstream_records`
- `validate_catalog`

## Relationship to M28

M28 established that StratLake has one execution system with multiple entry
points. Notebook execution examples are thin wrappers over `src.execution`
surfaces and canonical artifact contracts.

M29 preserves that boundary. Catalog-driven notebooks do not execute strategy,
alpha, portfolio, pipeline, campaign, benchmark-pack, or validation workflows.
They inspect the unified catalog view over prior outputs.

## Relationship to Query Docs

See `docs/catalog_query.md` for full query filters, CLI behavior, serialization
helpers, template handling, unknown-record behavior, and lineage query details.
Use this guide for the notebook research pattern and use the query guide for API
semantics.

## Limitations and Non-Goals

- No dashboard or database backend.
- No persistent catalog cache or committed export.
- No new registry writer, manifest writer, or execution wrapper.
- No artifact repair or validation CLI.
- No duplicated strategy, alpha, portfolio, pipeline, campaign, or benchmark
  logic.
- Empty artifact roots are valid for the example workflow and return empty
  tables plus a zero-record validation summary.
