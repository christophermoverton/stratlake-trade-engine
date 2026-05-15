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

## Pre-Merge Validation

Run Ruff over the changed workflow/release-adjacent Python tests:

```powershell
.\.venv\Scripts\ruff.exe check tests\test_milestone_validation_workflow.py tests\test_release_workflow.py tests\test_packaging_readiness.py
```

Run YAML/workflow syntax sanity for CI, release, and milestone validation:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import yaml; [yaml.safe_load(path.read_text(encoding='utf-8')) for path in [Path('.github/workflows/ci.yml'), Path('.github/workflows/release.yml'), Path('.github/workflows/milestone_validation.yml')]]"
```

Run focused workflow and package metadata tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_milestone_validation_workflow.py tests\test_release_workflow.py tests\test_packaging_readiness.py -q
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
- confirm `.github/workflows/release.yml` remains tag-driven on `v*`
- confirm the Release workflow still performs constrained editable install,
  focused release tests, docs/path lint, and `python -m build`
- confirm package build artifacts remain workflow artifacts only
- confirm GitHub Release assets remain deterministic release notes and
  docs/path lint evidence
- confirm no package publication target was added

## Architecture Boundaries

Confirm Issue #402 did not change:

- research artifact contracts
- catalog indexing, query, lineage, explorer, or notebook behavior
- promotion or governance decisions
- dependency declarations or lock policy
- cross-platform reproducibility guarantees
- deterministic artifact provenance
- portable path serialization
- CI-safe example behavior

## Post-Merge Validation On Main

After merge:

- verify primary CI is green
- verify milestone validation runs for the M36 branch or can be run manually
- rerun the focused workflow/package metadata tests from a clean checkout
- rerun docs/path lint
- confirm documentation links resolve in the merged tree
- confirm no generated machine-specific paths were committed

## Release Notes

See `docs/m36_release_notes.md`.
