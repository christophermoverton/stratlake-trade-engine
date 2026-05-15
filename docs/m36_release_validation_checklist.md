# M36 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 36. It does not
replace existing CI, milestone validation, or release automation.

Milestone title: `M36 - Scalable Evidence Interoperability and Release Hardening`

M36 branch:
`feature/m36-scalable-evidence-interoperability-release-hardening`

Candidate milestone release tag:
`v0.36.0-scalable-evidence-interoperability-release-hardening`

## Version And Release Semantics

Confirm the M36 version policy before merge:

- package version in `pyproject.toml`: Python distribution metadata for editable
  installs, installed metadata checks, and wheel/sdist validation
- milestone release tag: repository snapshot and GitHub Release evidence
- current policy: keep package version and milestone release tags separate
- M36 package version: preserve `0.1.0` unless package distribution semantics
  intentionally change
- M36 milestone tag candidate:
  `v0.36.0-scalable-evidence-interoperability-release-hardening`
- package publication to PyPI/TestPyPI remains out of scope

Future milestones should update milestone release notes, validation checklists,
and candidate tag names. They should preserve `pyproject.toml` unless package
metadata, install behavior, or distribution compatibility changes.

## Branch And Workflow Coverage

Confirm `.github/workflows/milestone_validation.yml` keeps:

- manual dispatch through `workflow_dispatch`
- legacy push branch support for `milestone/**`
- legacy push branch support for `m22/**`
- current and future milestone branch support through `feature/m*`
- pull request job coverage for source branches beginning with `feature/m`
- unchanged milestone validation job behavior beyond trigger coverage

The active M36 branch is:

```text
feature/m36-scalable-evidence-interoperability-release-hardening
```

Future milestone branches should use:

```text
feature/m<NUMBER>-<short-kebab-description>
```

## GitHub Actions Supply-Chain Hardening

Confirm every external `uses:` reference under `.github/workflows/` is pinned to
a 40-character commit SHA unless a documented exception is intentionally added.
The current workflow set has no local reusable actions and no unpinned external
exceptions.

Current reviewed pins:

- `actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5`
- `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065`
- `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02`
- `softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65`

When refreshing a pin:

- confirm the upstream tag and exact commit SHA before editing
- update the nearby tag-to-SHA workflow comment
- preserve workflow names, job names, matrices, inputs, and commands
- rerun YAML parsing, pinning tests, focused workflow tests, docs/path lint,
  package build validation, and full pytest when practical

## Pre-Merge Validation

Run Ruff over the changed workflow/release-adjacent Python tests:

```powershell
.\.venv\Scripts\ruff.exe check tests
```

Run YAML/workflow syntax sanity for CI, release, and milestone validation:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import yaml; [yaml.safe_load(path.read_text(encoding='utf-8')) for path in [Path('.github/workflows/ci.yml'), Path('.github/workflows/release.yml'), Path('.github/workflows/milestone_validation.yml')]]"
```

Run focused workflow and package metadata tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_github_actions_pinning.py tests\test_milestone_validation_workflow.py tests\test_release_workflow.py tests\test_packaging_readiness.py -q
```

Run the focused Issue #404 scale baseline validation:

```powershell
.\.venv\Scripts\ruff.exe check tests\test_catalog_scale_baselines.py tests\catalog_scale_fixtures.py
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_scale_baselines.py -q
```

Run the broader catalog/evidence regression slice:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_indexer.py tests\test_catalog_query.py tests\test_catalog_lineage.py tests\test_catalog_explorer.py tests\test_catalog_notebook_helpers.py tests\test_m35_evidence_discovery_validation.py tests\test_catalog_scale_baselines.py -q
```

Run the focused Issue #405 derived-index validation:

```powershell
.\.venv\Scripts\ruff.exe check src\catalog src\cli tests\test_catalog_derived_index.py tests\test_catalog_scale_baselines.py tests\catalog_scale_fixtures.py
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_derived_index.py -q
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_scale_baselines.py tests\test_catalog_indexer.py tests\test_catalog_query.py tests\test_catalog_lineage.py tests\test_catalog_explorer.py tests\test_catalog_notebook_helpers.py -q
```

Run the focused Issue #406 lineage-export validation:

```powershell
.\.venv\Scripts\ruff.exe check src\catalog src\cli tests\test_catalog_lineage_export.py
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_lineage_export.py -q
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_lineage.py tests\test_catalog_scale_baselines.py tests\test_catalog_derived_index.py tests\test_catalog_explorer.py tests\test_catalog_notebook_helpers.py -q
```

Run docs/path lint:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_docs_path_lint --output artifacts\qa\docs_path_lint_m36.json
```

Run package build validation:

```powershell
.\.venv\Scripts\python.exe -m build
```

When practical before merge, run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Release Workflow Validation

Before creating a milestone release tag:

- confirm the working tree is clean
- confirm M36 release notes and checklist use repository-relative paths only
- confirm external action references remain full-SHA pinned or explicitly
  justified in the M36 release notes
- confirm `.github/workflows/release.yml` remains tag-driven on `v*`
- confirm the Release workflow still performs constrained editable install,
  focused release tests, docs/path lint, and `python -m build`
- confirm package build artifacts remain workflow artifacts only
- confirm GitHub Release assets remain deterministic release notes and
  docs/path lint evidence
- confirm no package publication target was added

## Architecture Boundaries

Confirm M36 release-hardening issues did not change:

- research artifact contracts
- catalog indexing, query, lineage, explorer, or notebook behavior
- promotion or governance decisions
- dependency declarations or lock policy
- cross-platform reproducibility guarantees
- deterministic artifact provenance
- portable path serialization
- CI-safe example behavior
- direct artifact scanning remains the only catalog data source
- the optional Issue #405 derived index remains disposable, rebuildable, and
  never canonical
- Issue #406 lineage exports remain derived from explicit catalog lineage only
- exported lineage preserves original StratLake edge types and portable paths
- no remote metadata service, production search backend, graph store, or second
  registry has been introduced

## Post-Merge Validation On Main

After merge:

- verify primary CI is green
- verify milestone validation runs for the M36 branch or can be run manually
- verify CI, release, and milestone validation workflows still use only pinned
  external action references
- rerun the focused workflow/package metadata tests from a clean checkout
- rerun docs/path lint
- confirm documentation links resolve in the merged tree
- confirm no generated machine-specific paths were committed

## Release Notes

See `docs/m36_release_notes.md`.
