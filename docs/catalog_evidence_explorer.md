# Catalog Evidence Explorer

The M35 catalog evidence explorer is a lightweight local review surface over
the existing catalog, query, and lineage APIs. It renders deterministic views of
catalog records, M35 evidence fields, and evidence lineage without creating a
server, database, persistent backend, search service, canonical cache, or second
catalog.

The explorer is read-only. It does not mutate source artifacts, replay
promotion gates, enforce governance decisions, or update promotion outcomes.

## CLI Usage

Render Markdown for all catalog records:

```powershell
python -m src.cli.explore_catalog_evidence --artifacts-root artifacts --format markdown
```

Render JSON for robustness evidence needing review:

```powershell
python -m src.cli.explore_catalog_evidence `
  --record-family robustness_bundle `
  --robustness-status needs_review `
  --format json
```

Render a governance-focused table:

```powershell
python -m src.cli.explore_catalog_evidence `
  --record-family governance_bundle `
  --governance-status pass `
  --format table
```

Render one run and its related evidence lineage:

```powershell
python -m src.cli.explore_catalog_evidence --run-id strategy_run_1 --format markdown
```

Write a derived local review file when explicitly requested:

```powershell
python -m src.cli.explore_catalog_evidence `
  --run-id strategy_run_1 `
  --output artifacts/local_reviews/strategy_run_1_evidence.md
```

Output written with `--output` is a derived review artifact, not a canonical
registry, catalog cache, or source-of-truth file.

## Python API

```python
from src.catalog import (
    CatalogQuery,
    build_catalog,
    build_evidence_explorer_view,
    render_evidence_markdown,
)

records = build_catalog("artifacts", repo_root=".")
view = build_evidence_explorer_view(
    records,
    query=CatalogQuery(record_family="robustness_bundle", robustness_status="needs_review"),
    repo_root=".",
)
markdown = render_evidence_markdown(view)
```

For selected-run review:

```python
view = build_evidence_explorer_view(
    records,
    selected_run_id="strategy_run_1",
    include_lineage=True,
    repo_root=".",
)
```

Notebook-friendly wrappers for these same calls are documented in
[`docs/catalog_notebook_ergonomics.md`](catalog_notebook_ergonomics.md).

## Output Formats

Markdown output uses stable sections:

1. Catalog Records
2. Evidence Status
3. Evidence Summary
4. Evidence Lineage

JSON output is sorted and deterministic. Table output is tab-separated with a
stable column order.

The explorer includes these catalog and evidence fields where present:

- `run_id`, `run_type`, `status`, `record_family`, `artifact_root`
- `review_status`, `promotion_status`
- `robustness_status`, `wfe_status`, `sample_size_status`,
  `trade_count_status`, `sensitivity_status`, `fragility_status`,
  `multiple_testing_status`, `temporal_validation_status`
- `governance_status`, `promotion_review_status`
- `validation_readiness_present`, `release_validation_present`

Lineage rows include `edge_type`, source/target run IDs, relationship source,
relationship path, and compact metadata derived from existing lineage edges.

## Empty And Sparse Behavior

Empty filters render explicit messages such as `No matching records.` and `No
evidence lineage found.` Sparse records with missing optional evidence fields
render blank cells instead of failing.

## Determinism And Portability

The explorer reuses deterministic catalog sorting, query filtering, and lineage
sorting. It does not add runtime timestamps. Paths in rendered output are the
portable catalog paths already derived by the indexer and lineage layers; local
absolute paths, Windows-only separators, and `file://` links are not introduced
by the explorer.

## Boundary

This is a local review tool, not a production dashboard. It does not add:

- a web service or dashboard server
- a graph database or graph cache
- persistent storage or a canonical cache
- a search backend
- a new registry or alternate catalog
- policy simulation, promotion mutation, or governance enforcement
