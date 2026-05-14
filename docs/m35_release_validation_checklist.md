# M35 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 35. It does not
replace existing CI or release automation.

Milestone title: `M35 - Artifact Lineage, Discovery, and Robustness-Aware Research Catalog`

## Pre-Merge Validation

Run lint over the M35 catalog, CLI, example, and test surfaces:

```powershell
.\.venv\Scripts\ruff.exe check src\catalog src\cli docs\examples tests
```

Run the focused M35 pytest slice:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_indexer.py tests\test_catalog_query.py tests\test_catalog_lineage.py tests\test_catalog_explorer.py tests\test_catalog_notebook_helpers.py tests\test_m35_evidence_discovery_validation.py -q
```

Run the CI-safe notebook/API example:

```powershell
.\.venv\Scripts\python.exe docs\examples\catalog_evidence_notebook_workflow.py
```

Run compile validation:

```powershell
.\.venv\Scripts\python.exe -m compileall src docs\examples -q
```

Run the broader catalog and robustness-aware regression slice:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -k "catalog or robustness or governance or lineage or explorer or notebook" -q
```

When practical before merge, run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Known non-blocking warnings may appear from existing robustness fixtures, such
as low sample-size or degenerate signal warnings. A newly introduced warning
class should be investigated before release.

## Documentation And Path Portability

Confirm the M35 docs are present and linked:

- `README.md`
- `docs/m35_evidence_catalog_foundation.md`
- `docs/m35_evidence_discovery_validation.md`
- `docs/m35_release_notes.md`
- `docs/m35_release_validation_checklist.md`
- `docs/catalog_indexer.md`
- `docs/catalog_query.md`
- `docs/catalog_lineage.md`
- `docs/catalog_evidence_explorer.md`
- `docs/catalog_notebook_ergonomics.md`
- `docs/examples/catalog_evidence_notebook_workflow.py`

Check that M35 docs and example output avoid local absolute paths, `file://`
links, and Windows-only serialized paths. User-visible catalog paths should be
repository-relative or POSIX-style relative paths whenever they refer to source
artifacts under the repository or artifact root.

## CI-Safe Example Validation

`docs/examples/catalog_evidence_notebook_workflow.py` should:

- build only synthetic temporary artifacts
- require no external services, credentials, or live market-data downloads
- avoid repository artifact mutation
- print deterministic JSON with sorted keys
- exercise robustness, governance, milestone validation, release-validation,
  selected-run explorer, and evidence lineage surfaces

## Read-Only Governance Boundary

Confirm M35 review surfaces remain derived and read-only:

- catalog/indexer scans existing artifacts without writing canonical records
- query filters operate in memory
- lineage edges are derived in memory and not persisted as a graph store
- explorer and notebook helpers render derived review output only
- governance evidence remains review context
- promotion decisions are not replayed, recomputed, or mutated

## Post-Merge Validation On Main

After merge:

- verify primary CI is green
- rerun the focused M35 pytest slice from a clean checkout
- rerun the CI-safe notebook workflow example
- rerun compile validation
- confirm documentation links resolve in the merged tree
- confirm no generated machine-specific paths were committed

## Release Tag Readiness

Before creating a release tag:

- confirm the working tree is clean
- confirm focused and broader validation commands pass
- confirm `docs/m35_release_notes.md` reflects the merged feature set
- confirm `docs/m35_release_validation_checklist.md` reflects final commands
- confirm Issue #394 through Issue #400 summaries and audits are complete
- confirm no new registry, database, cache, search backend, server, graph store,
  policy replay layer, promotion mutation, or governance enforcement was added
- candidate tag name: `v0.35.0-artifact-lineage-research-catalog`

## Release Notes

See `docs/m35_release_notes.md`.

