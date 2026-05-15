# M35 Evidence Discovery Validation

This validation slice protects the M35 robustness-aware catalog surfaces added
across Issues #394 through #398. It is intentionally focused on deterministic,
read-only discovery behavior rather than new catalog features.

## What The Suite Protects

- Evidence catalog indexing for `robustness_bundle`, `governance_bundle`,
  `milestone_validation_bundle`, and `release_validation_artifact` records.
- Robustness, governance, validation, and release discovery filters through the
  shared catalog query API.
- Read-only evidence lineage edges derived from explicit metadata or source
  artifact references.
- Local evidence explorer Markdown, JSON, and table rendering.
- Notebook/API helper parity with the shared query, lineage, and explorer APIs.
- Empty, sparse, missing, and orphan evidence behavior.
- Portable path serialization with repository-relative, POSIX-style paths.
- Governance and promotion review boundaries remaining observable/read-only.

The tests use synthetic local artifacts, fixed payloads, explicit line endings,
and no external services, credentials, live market data, or current-time inputs.

## Targeted Commands

Run this pre-merge slice when touching M35 catalog discovery, lineage, explorer,
or notebook ergonomics:

```bash
python -m pytest tests/test_catalog_indexer.py tests/test_catalog_query.py tests/test_catalog_lineage.py tests/test_catalog_explorer.py tests/test_catalog_notebook_helpers.py tests/test_m35_evidence_discovery_validation.py -q
python docs/examples/catalog_evidence_notebook_workflow.py
python -m compileall src docs/examples -q
```

For a broader regression pass, run:

```bash
python -m pytest tests/ -k "catalog or robustness or governance or lineage or explorer or notebook" -q
```

On Windows with the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_indexer.py tests\test_catalog_query.py tests\test_catalog_lineage.py tests\test_catalog_explorer.py tests\test_catalog_notebook_helpers.py tests\test_m35_evidence_discovery_validation.py -q
.\.venv\Scripts\python.exe docs\examples\catalog_evidence_notebook_workflow.py
.\.venv\Scripts\python.exe -m compileall src docs\examples -q
.\.venv\Scripts\python.exe -m pytest tests\ -k "catalog or robustness or governance or lineage or explorer or notebook" -q
```

## Read-Only Boundary

The M35 discovery surfaces derive records, filters, lineage rows, explorer
views, and notebook helper outputs from existing artifacts. They do not write a
canonical catalog, registry, graph store, cache, search index, database, or
persistent backend. They also do not replay promotion policy, mutate promotion
decision artifacts, or enforce governance decisions.

Regression tests snapshot source artifact bytes before and after running the
catalog, query, lineage, explorer, and notebook helper surfaces. Governance
reports remain review context only; fields such as `governance_status` and
`promotion_review_status` are indexed for review without recomputing or
rewriting `promotion_status`.

## Path Portability

Catalog records, lineage metadata, explorer output, notebook helper output, and
the CI-safe example must avoid local absolute paths, `file://` links, and
Windows-only path separators. Paths exposed to users are repository-relative or
POSIX-style relative paths whenever source artifacts are under the repository or
test artifact root.

## Sparse And Orphan Evidence

Sparse evidence bundles with missing optional fields remain indexable and
renderable. Orphan source references are skipped deterministically by lineage
extraction. M35 lineage must not create edges from name similarity, aggregate
counts, or release identifiers alone.

## Scope

This document covers validation/regression for Issues #394 through #399. Final
release notes and milestone checklist integration live in
[`docs/m35_release_notes.md`](m35_release_notes.md) and
[`docs/m35_release_validation_checklist.md`](m35_release_validation_checklist.md).
