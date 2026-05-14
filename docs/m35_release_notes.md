# M35 Release Notes - Artifact Lineage, Discovery, and Robustness-Aware Research Catalog

Milestone title: `M35 - Artifact Lineage, Discovery, and Robustness-Aware Research Catalog`

## Milestone Principle

Research evidence should be discoverable, traceable, and reviewable without
creating a second source of truth.

## Summary

M35 extends the existing M29 catalog, query, lineage, and validation
architecture so M34 robustness evidence, M32 governance reporting, milestone
validation bundles, and release-validation outputs can be reviewed through one
read-only catalog surface.

The milestone makes evidence easier to find and explain without introducing a
new registry, database, persistent cache, search backend, graph database,
dashboard service, policy replay layer, or governance enforcement layer. Source
artifacts remain the source of truth; catalog records, query results, lineage
edges, explorer views, and notebook helpers are derived review surfaces.

## Platform Improvements

- Issue #394: evidence catalog record families for `robustness_bundle`,
  `governance_bundle`, `milestone_validation_bundle`, and
  `release_validation_artifact`.
- Issue #395: robustness, governance, validation, and release discovery filters
  for the Python query API and CLI.
- Issue #396: read-only evidence lineage edges from explicit metadata, manifest
  references, source artifact references, and supported evidence files.
- Issue #397: lightweight local evidence explorer with deterministic Markdown,
  JSON, and table rendering.
- Issue #398: notebook/API helper functions over the shared catalog, query,
  lineage, and explorer APIs.
- Issue #399: deterministic validation and regression coverage for discovery,
  lineage, explorer, notebook helpers, path portability, and read-only
  governance boundaries.

## Architecture Preserved

- One catalog system: M35 extends `src.catalog`; it does not create a second
  catalog or canonical registry.
- Artifact-derived evidence: all M35 records and views are read from existing
  artifacts.
- Deterministic output: catalog records, query results, lineage edges, explorer
  renders, notebook helper outputs, and examples keep stable ordering.
- Portable paths: user-visible paths remain repository-relative or POSIX-style
  relative paths, with no local absolute paths or `file://` links.
- Shared implementation: CLI, API, local explorer, and notebook helpers use the
  same catalog/query/lineage contracts.
- Governance boundary: governance and promotion-review fields remain observable
  review context and do not mutate promotion decisions.

## Validation

Focused M35 validation:

```bash
python -m pytest tests/test_catalog_indexer.py tests/test_catalog_query.py tests/test_catalog_lineage.py tests/test_catalog_explorer.py tests/test_catalog_notebook_helpers.py tests/test_m35_evidence_discovery_validation.py -q
```

Example and compile validation:

```bash
python docs/examples/catalog_evidence_notebook_workflow.py
python -m compileall src docs/examples -q
```

Broader regression validation:

```bash
python -m pytest tests/ -k "catalog or robustness or governance or lineage or explorer or notebook" -q
```

Observed local validation for this release-readiness pass:

- `ruff check src/catalog src/cli docs/examples tests`: passed.
- Focused M35 validation slice: passed.
- `docs/examples/catalog_evidence_notebook_workflow.py`: passed.
- `compileall src docs/examples`: passed.
- Broader catalog/robustness/governance/lineage/explorer/notebook slice:
  passed.
- Full pytest suite: passed.

## CI-Safe Example

`docs/examples/catalog_evidence_notebook_workflow.py` builds synthetic temporary
artifacts, indexes them through `build_catalog()`, runs notebook-friendly
evidence helpers, renders a selected-run view, and prints deterministic JSON.
It uses no external services, credentials, live market-data downloads, or
repository artifact mutation.

## Deferred And Non-Goals

M35 does not add:

- a production dashboard or web service
- a graph database or persistent graph cache
- a persistent search backend or canonical cache
- a new registry or alternate catalog
- live monitoring
- dataset contract migration or cache invalidation infrastructure
- promotion policy simulation, promotion mutation, or governance enforcement

## Further Reading

- `docs/m35_evidence_catalog_foundation.md`
- `docs/catalog_indexer.md`
- `docs/catalog_query.md`
- `docs/catalog_lineage.md`
- `docs/catalog_evidence_explorer.md`
- `docs/catalog_notebook_ergonomics.md`
- `docs/m35_evidence_discovery_validation.md`
- `docs/m35_release_validation_checklist.md`
- `docs/examples/catalog_evidence_notebook_workflow.py`
