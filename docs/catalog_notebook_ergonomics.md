# Catalog Notebook Ergonomics

M35 notebook helpers make robustness-aware catalog exploration easier without
creating notebook-only execution paths. They are convenience wrappers over the
same catalog, query, lineage, and explorer APIs used by the CLI.

The helpers live in `src.catalog` and return plain Python objects suitable for
display in notebooks or conversion to data frames.

## Helpers

```python
from src.catalog import (
    build_catalog,
    evidence_for_run,
    evidence_lineage_rows,
    find_governance_evidence,
    find_release_evidence,
    find_robustness_evidence,
    find_validation_evidence,
    render_notebook_markdown,
)

records = build_catalog("artifacts", repo_root=".")
robustness_rows = find_robustness_evidence(records, robustness_status="needs_review")
governance_rows = find_governance_evidence(records, governance_status="pass")
validation_rows = find_validation_evidence(records)
release_rows = find_release_evidence(records)

run_view = evidence_for_run(records, "strategy_run_1", repo_root=".")
lineage_rows = evidence_lineage_rows(records, run_id="strategy_run_1", repo_root=".")
markdown = render_notebook_markdown(records, run_id="strategy_run_1", repo_root=".")
```

`build_notebook_evidence_view()` returns the same dictionary shape as
`build_evidence_explorer_view()`. `render_notebook_markdown()`,
`render_notebook_json()`, and `render_notebook_table()` call the shared explorer
renderers.

## CLI/API Parity

These calls use the same filters as:

```powershell
python -m src.cli.query_catalog --record-family robustness_bundle --robustness-status needs_review
python -m src.cli.explore_catalog_evidence --run-id strategy_run_1 --format markdown
```

Notebook helpers do not duplicate query, lineage, or explorer logic. They call
`CatalogQuery`, `query_catalog()`, `build_lineage_edges()`,
`build_evidence_explorer_view()`, and the explorer renderers.

## Example

Run the CI-safe notebook-style example:

```powershell
python docs/examples/catalog_evidence_notebook_workflow.py
```

The example creates synthetic artifacts in a temporary directory, builds a
catalog, finds robustness and governance evidence, renders selected-run evidence,
and prints JSON. It does not download data, require credentials, or write
repository artifacts.

## Determinism And Portability

Helper outputs are deterministic and JSON-friendly:

- table helpers return `list[dict[str, object]]`
- single-run helpers return `dict[str, object]`
- render helpers return strings
- returned paths are repository-relative/POSIX-style catalog paths
- no open file handles or `Path` objects are returned in JSON-like payloads

## Boundary

These helpers are a notebook/API ergonomics layer only. They do not add:

- notebook-only workflow logic
- duplicated CLI logic
- a server or dashboard
- a database, backend, search service, or canonical cache
- policy replay, promotion mutation, or governance enforcement
