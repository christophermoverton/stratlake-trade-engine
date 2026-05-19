# M39 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 39. It does not
replace existing CI, milestone validation, or release automation.

Milestone title: `M39 - Configuration Profiles and Environment Readiness`

M39 branch:
`feature/m39-configuration-profiles-environment-readiness`

Candidate milestone release tag:
`v0.39.0-configuration-profiles-environment-readiness`

## Milestone Principle

Configuration should make correct workflows easier to start, inspect, and
reproduce without hiding execution behavior or weakening artifact boundaries.

## Pre-Merge Validation

Run the focused M39 validation slice:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_m39_first_run_example.py tests\test_runtime_explain.py tests\test_environment_doctor.py tests\test_validate_config_cli.py tests\test_config_resolution.py tests\test_runtime_profiles.py tests\test_runtime_config.py tests\test_docs_path_portability.py -q
```

Run Ruff over the M39-facing source, tests, and examples:

```powershell
.\.venv\Scripts\ruff.exe check docs\examples src\config src\cli tests
```

Run the thin config validation CLI:

```powershell
.\.venv\Scripts\python.exe -m src.cli.validate_config --profile ci
```

Run the advisory environment doctor:

```powershell
.\.venv\Scripts\python.exe -m src.cli.stratlake_doctor --profile ci
```

Run the runtime explain helper:

```powershell
.\.venv\Scripts\python.exe -m src.cli.explain_config --profile ci --workflow strategy
```

Run the CI-safe first-run example:

```powershell
.\.venv\Scripts\python.exe docs\examples\m39_first_run_configuration_profile_example.py --output-root docs\examples\output\m39_first_run_configuration_profile_example
```

Run docs/path lint:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_docs_path_lint --output artifacts\qa\docs_path_lint_m39_pre_merge.json
```

Run package build validation:

```powershell
.\.venv\Scripts\python.exe -m build --outdir artifacts\qa\m39_package_build_pre_merge
```

Run the milestone validation bundle when practical:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_milestone_validation --bundle-dir artifacts\qa\m39_validation_bundle --include-full-pytest
```

## Validation Coverage

The focused M39 pytest slice covers:

- runtime profile contracts and starter profile parseability
- profile path portability and secret-like unknown-key rejection
- deterministic config precedence:
  `defaults < profile config < environment variables < CLI flags`
- per-field resolution provenance and stable serialization
- config validation CLI success and failure behavior
- environment doctor readiness checks, categories, and non-mutation guarantees
- runtime explain reports, workflow assumptions, provenance summaries, and
  evidence-review derived-root alignment
- CI-safe first-run onboarding output, determinism, portability, and generated
  output hygiene
- docs/path lint behavior, including generated `docs/examples/output/**`
  ignore policy

## Generated Output Hygiene

The M39 first-run example writes generated reports by default under:

```text
docs/examples/output/m39_first_run_configuration_profile_example/
```

Those reports are generated, disposable, advisory, and non-authoritative. They
are not canonical artifacts, not checked-in workflow configs, and not release
evidence by themselves. The directory is ignored by default because a local
first-run smoke should not create unintentional release diffs.

Before merging or tagging:

- remove or review any generated `docs/examples/output/...` files
- do not commit M39 first-run generated reports unless intentionally changing
  the repository's tracked example-output policy
- confirm docs/path lint remains green for guarded source docs and examples
- confirm generated reports contain no machine-local absolute paths, `file://`
  links, backslash paths, or parent traversal
- confirm generated validation, doctor, explain, and first-run reports remain
  advisory and non-authoritative

## Merge-Readiness Checklist

Before merging the M39 branch:

- all M39 implementation issues are resolved or intentionally deferred
- focused M39 pytest validation is green
- focused Ruff validation is green
- config validation, doctor, explain, and first-run smoke commands are green
- docs/path lint is green
- package build validation is green
- milestone validation bundle is green when practical
- hosted GitHub Actions Milestone Validation is green
- no generated M39 first-run output is unintentionally staged
- no documentation contains machine-local absolute paths or `file://` links
- starter profiles remain non-secret and repository-relative
- generated reports remain derived, disposable, advisory, and non-authoritative
- validation, doctor, explain, and first-run onboarding do not execute workflows
- no dashboard, server, graph store, second registry, second catalog, remote
  metadata service, deployment infrastructure, or secrets framework was
  introduced

## Post-Merge Validation On Main

After merge:

- checkout and update `main`
- verify the merge commit contains M39 release notes, checklist, docs, example,
  CLI/API tests, generated-output hygiene, and README links
- rerun the focused M39 validation slice from a clean checkout
- rerun config validation, doctor, explain, and first-run smoke commands
- rerun docs/path lint
- rerun package build validation
- run the milestone validation bundle or the full relevant regression suite when
  practical
- confirm GitHub Actions are green on `main`
- confirm no generated machine-specific paths were committed
- confirm the release tag candidate is still appropriate
- create the release tag:
  `v0.39.0-configuration-profiles-environment-readiness`
- prepare the GitHub Release using the draft release notes in
  `docs/m39_release_notes.md`

## Architecture Checks

Confirm M39 keeps the configuration and artifact boundaries intact:

- canonical artifacts remain the source of truth
- direct scan remains available, default, and canonical
- runtime profiles remain non-secret starter context, not a second source of
  truth
- `.env.example`, `Settings.load()`, and workflow config files remain existing
  config surfaces rather than being replaced by profiles
- validation, doctor, explain, and first-run reports remain advisory
- derived outputs remain disposable and non-authoritative
- evidence-review explain roots use `artifacts/_derived/evidence_review`
- no workflow execution is introduced by validation, doctor, explain, or the
  first-run onboarding example
- no live market data, credentials, network access, external services,
  deployment manifests, dashboards, servers, graph stores, or second registries
  are required

## Non-Goals Confirmed

M39 does not implement:

- deployment, Docker, Kubernetes, or hosted runtime behavior
- dashboards, servers, remote metadata services, graph stores, or second
  registries
- secrets management infrastructure
- live-data validation or credential inspection
- workflow execution rewrites
- a new orchestration layer
- authoritative generated configuration, readiness, explain, or first-run
  reports

## Expected Validation Artifacts

Generated validation artifacts, when requested, should remain under disposable
locations such as:

- `artifacts/qa/docs_path_lint_m39_pre_merge.json`
- `artifacts/qa/m39_package_build_pre_merge/`
- `artifacts/qa/m39_validation_bundle/`
- `docs/examples/output/m39_first_run_configuration_profile_example/`

The `docs/examples/output/m39_first_run_configuration_profile_example/`
directory is ignored by default. Treat all of these paths as generated outputs,
not canonical artifacts.

## Release Tag Checklist

Before pushing
`v0.39.0-configuration-profiles-environment-readiness`:

- confirm the working tree is clean
- confirm post-merge validation on `main` is complete
- confirm `.github/workflows/release.yml` remains tag-driven on `v*`
- confirm package build outputs remain workflow artifacts only
- confirm package publication remains out of scope
- confirm the GitHub Release body uses the M39 draft release notes
- do not claim hosted CI or release success until GitHub reports it

## Release Notes

See `docs/m39_release_notes.md`.
