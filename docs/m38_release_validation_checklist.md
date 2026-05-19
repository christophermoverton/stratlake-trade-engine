# M38 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 38. It does not
replace existing CI, milestone validation, or release automation.

Milestone title: `M38 - Static Evidence Review Packs and Catalog Health Diagnostics`

M38 branch:
`feature/m38-static-evidence-review-packs-catalog-health`

Candidate milestone release tag:
`v0.38.0-static-evidence-review-packs-catalog-health`

## Pre-Merge Validation

Run Ruff over the focused M38 source, tests, and example:

```powershell
.\.venv\Scripts\ruff.exe check src\catalog\review_pack.py src\cli\build_evidence_review.py src\catalog\__init__.py tests\test_m38_deterministic_validation.py tests\test_evidence_review_cli.py tests\test_evidence_review_pack_writer.py tests\test_evidence_review_builder.py tests\test_catalog_health_diagnostics.py tests\test_review_pack_contracts.py tests\test_m37_architecture_guardrails.py docs\examples\m38_static_evidence_review_pack_example.py
```

Run the focused M38 validation slice:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_m38_deterministic_validation.py tests\test_evidence_review_cli.py tests\test_evidence_review_pack_writer.py tests\test_evidence_review_builder.py tests\test_catalog_health_diagnostics.py tests\test_review_pack_contracts.py tests\test_canonical_resolver.py tests\test_catalog_lineage_export.py tests\test_catalog_derived_index.py tests\test_derived_namespace_and_load_source.py tests\test_m37_architecture_guardrails.py tests\test_portable_paths.py tests\test_docs_path_portability.py -q
```

Run the CI-safe example:

```powershell
.\.venv\Scripts\python.exe docs\examples\m38_static_evidence_review_pack_example.py
```

Run docs/path lint:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_docs_path_lint --output artifacts\qa\docs_path_lint_m38.json
```

Run package build validation:

```powershell
.\.venv\Scripts\python.exe -m build --outdir artifacts\qa\m38_package_build
```

Run the milestone validation bundle when practical:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_milestone_validation --bundle-dir artifacts\qa\m38_validation_bundle --include-full-pytest
```

## Merge-Readiness Checklist

Before merging the M38 branch:

- all M38 implementation issues are resolved or intentionally deferred
- focused M38 validation is green
- CI-safe example smoke is green
- docs/path lint is green
- package build validation is green
- milestone validation bundle is green when practical
- hosted GitHub Actions Milestone Validation is green
- no documentation contains machine-local absolute paths or `file://` links
- review packs remain under `artifacts/_derived/evidence_review/<review_id>/`
- review packs remain derived, disposable, rebuildable, non-authoritative, and
  write-back-forbidden
- generated packs do not mutate canonical artifacts
- deleting review packs does not change canonical catalog identity
- `_derived/evidence_review` is excluded from canonical scans
- decision-authority modules do not import review-pack builders, writers,
  validators, or CLI modules
- no dashboard, server, graph store, second registry, second catalog, remote
  metadata service, inferred lineage, governance mutation, or promotion mutation
  was introduced

## Post-Merge Validation On Main

After merge:

- checkout and update `main`
- verify the merge commit contains the M38 release notes, checklist, docs,
  example, CLI, and validation tests
- rerun the focused M38 validation slice from a clean checkout
- rerun the CI-safe example smoke
- rerun docs/path lint
- rerun package build validation
- run the milestone validation bundle or the full relevant regression suite when
  practical
- confirm GitHub Actions are green on `main`
- confirm no generated machine-specific paths were committed
- confirm the release tag candidate is still appropriate
- create the release tag:
  `v0.38.0-static-evidence-review-packs-catalog-health`
- prepare the GitHub Release using the draft release notes in
  `docs/m38_release_notes.md`

## Architecture Checks

Confirm M38 keeps the artifact-first boundary intact:

- canonical artifacts remain the source of truth
- direct scan remains canonical and default
- review packs live under `artifacts/_derived/evidence_review/<review_id>/`
- review packs are derived, disposable, rebuildable, non-authoritative, and
  write-back-forbidden
- resolver-first canonical reopening remains the decision-sensitive path
- diagnostics are advisory review context, not governance or promotion decisions
- static reports are review surfaces, not dashboards or sources of truth
- optional HTML remains static, escaped, self-contained, and server-free
- no raw resolver source content is embedded in review models or packs
- serialized review content uses repository-relative POSIX paths only

## Release Tag Checklist

Before pushing
`v0.38.0-static-evidence-review-packs-catalog-health`:

- confirm the working tree is clean
- confirm post-merge validation on `main` is complete
- confirm `.github/workflows/release.yml` remains tag-driven on `v*`
- confirm package build outputs remain workflow artifacts only
- confirm package publication remains out of scope
- confirm the GitHub Release body uses the M38 draft release notes
- do not claim hosted CI or release success until GitHub reports it

## Release Notes

See `docs/m38_release_notes.md`.
