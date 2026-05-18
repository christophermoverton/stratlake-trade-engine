# M37 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 37. It does not
replace existing CI, milestone validation, or release automation.

Milestone title: `M37 - Artifact-First Evidence Hardening and Canonicality Contracts`

M37 branch:
`feature/m37-artifact-first-canonicality-contracts`

Candidate milestone release tag:
`v0.37.0-artifact-first-canonicality-contracts`

## Pre-Merge Validation

Run focused M37 canonicality-stack validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_canonicality_envelope.py tests\test_derived_namespace_and_load_source.py tests\test_canonical_resolver.py tests\test_m37_architecture_guardrails.py tests\test_m37_deterministic_validation.py -q
```

Run M36/catalog regression validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_m36_deterministic_validation.py tests\test_catalog_derived_index.py tests\test_catalog_lineage_export.py tests\test_catalog_cli_api_ergonomics.py -q
```

Run CLI query/explorer regression validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli_query_catalog.py tests\test_cli_catalog_explorer.py -q
```

Run docs/path lint:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_docs_path_lint --output artifacts\qa\docs_path_lint_m37.json
```

Run package build validation:

```powershell
.\.venv\Scripts\python.exe -m build --outdir artifacts\qa\m37_package_build
```

Run the full milestone validation bundle when practical:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_milestone_validation --bundle-dir artifacts\qa\milestone_validation_bundle --include-full-pytest
```

If source or example Python files changed, run Ruff:

```powershell
.\.venv\Scripts\ruff.exe check src tests docs\examples
```

## Architecture Checks

Confirm M37 keeps the artifact-first boundary intact:

- canonical artifacts remain the source of truth
- direct scan remains canonical and the default load mode
- newly generated derived outputs default beneath `artifacts/_derived/`
- derived indexes remain disposable, rebuildable read models
- lineage exports remain local JSON views only
- evidence, explorer, and workflow outputs remain non-authoritative
- resolver-first APIs reopen canonical files for consequential use
- artifact-root containment still rejects repo-relative non-artifact paths
- architecture guardrails still reject direct and public-facade imports of
  derived read-model helpers in decision-authority modules
- legacy no-envelope M36 derived payloads remain readable

## Release Tag Checklist

Before pushing
`v0.37.0-artifact-first-canonicality-contracts`:

- confirm the working tree is clean
- confirm the M37 docs use repository-relative links only
- confirm pre-merge validation is complete
- confirm the candidate tag name matches the release notes
- confirm `.github/workflows/release.yml` remains tag-driven on `v*`
- confirm no unsupported capability claims were added to docs
- do not claim hosted CI success until GitHub reports it

## Hosted GitHub Actions Follow-Up

After pushing the branch or opening the release pull request:

- verify primary CI is green
- verify Milestone Validation is green on the M37 branch
- inspect the uploaded milestone validation bundle if hosted validation fails
- confirm `pytest_full` remains green after the Issue `#419` query-output fix
- record hosted validation status before merge

## Post-Merge Validation On Main

After merge:

- rerun the focused M37 stack validation from a clean checkout
- rerun the M36/catalog and CLI regression slices
- rerun docs/path lint, package build, and milestone validation when practical
- confirm documentation links resolve in the merged tree
- confirm no generated machine-specific paths were committed
- confirm the release tag candidate is still appropriate before publishing it

## CI-Safe Example

Run the compact M37 example:

```powershell
.\.venv\Scripts\python.exe docs\examples\m37_artifact_first_evidence_contracts_example.py
```

The script builds a temporary synthetic artifact tree, writes a disposable
derived index under `artifacts/_derived/`, exposes `load_source` metadata,
renders a lineage export, and resolves canonical source files without touching
repository artifacts.

## Release Notes

See `docs/m37_release_notes.md`.
