# M39 Release Notes - Configuration Profiles and Environment Readiness

Milestone title: `M39 - Configuration Profiles and Environment Readiness`

M39 branch:
`feature/m39-configuration-profiles-environment-readiness`

Candidate milestone release tag:
`v0.39.0-configuration-profiles-environment-readiness`

## Milestone Principle

Configuration should make correct workflows easier to start, inspect, and
reproduce without hiding execution behavior or weakening artifact boundaries.

## Summary

M39 adds a deterministic configuration-profile and environment-readiness stack
for StratLake. Contributors can validate starter profiles, resolve effective
configuration with provenance, run an advisory environment doctor, explain
runtime assumptions before execution, and complete a CI-safe first-run
onboarding path without live market data, credentials, network access, or
external services.

The result is a clearer first mile for local development, clean CI, notebooks,
and pipeline preflight while preserving artifact-first boundaries:
canonical artifacts remain the source of truth, direct scan remains available
and canonical, and generated validation/readiness/explain reports remain
advisory and non-authoritative.

## Issue Summary

| Issue | Outcome |
| --- | --- |
| `#429` | Added runtime profile contracts, starter profiles under `configs/profiles/`, profile docs, and parseability/portability tests. |
| `#430` | Added deterministic config resolution with precedence and per-field provenance. |
| `#431` | Added a thin config validation CLI around the shared profile and resolver APIs. |
| `#432` | Added the environment doctor API and CLI with deterministic readiness-flow checks. |
| `#433` | Added runtime explain/dry-run helpers, workflow assumptions, provenance summaries, and derived evidence-review root alignment. |
| `#434` | Added the CI-safe first-run onboarding example, generated-output hygiene, and docs/path lint policy for generated example output. |
| `#435` | Added final deterministic validation and release-readiness documentation for M39. |

## Core Implementation Work

- `src/config/profiles.py` defines the runtime profile contract.
- `src/config/resolution.py` resolves effective configuration through:

  ```text
  defaults < profile config < environment variables < CLI flags
  ```

- `src/cli/validate_config.py` exposes deterministic config/profile validation.
- `src/config/doctor.py` and `src/cli/stratlake_doctor.py` expose advisory
  environment readiness checks.
- `src/config/explain.py` and `src/cli/explain_config.py` expose deterministic
  runtime explain reports and workflow assumptions.
- `configs/profiles/local.yml`, `ci.yml`, `notebook.yml`, and `pipeline.yml`
  provide non-secret starter profiles.

## Docs And Examples Work

- `docs/runtime_profiles.md` documents the profile contract, precedence,
  validation CLI, environment doctor, explain helpers, and first-run path.
- `docs/runtime_configuration.md` links existing runtime config behavior to the
  broader M39 profile/resolution model.
- `docs/getting_started.md` includes the CI-safe first-run command sequence.
- `docs/examples/m39_first_run_configuration_profile_example.py` writes
  deterministic validation, doctor, explain, synthetic probe, and summary
  reports for clean-checkout onboarding.
- `.gitignore` ignores
  `docs/examples/output/m39_first_run_configuration_profile_example/` so local
  first-run reports do not become accidental release diffs.

## Validation Work

The focused M39 validation slice covers:

- runtime profile contracts and starter profile portability
- config resolution precedence and provenance
- config validation CLI/API parity
- environment doctor pass/warning/fail/skipped behavior
- runtime explain report determinism and non-execution guarantees
- first-run example determinism, generated-output hygiene, and portability
- docs/path lint behavior for guarded docs and generated example output policy

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_m39_first_run_example.py tests\test_runtime_explain.py tests\test_environment_doctor.py tests\test_validate_config_cli.py tests\test_config_resolution.py tests\test_runtime_profiles.py tests\test_runtime_config.py tests\test_docs_path_portability.py -q
```

## Usage Notes

Clean-checkout first run:

```powershell
python -m src.cli.validate_config --profile ci
python -m src.cli.stratlake_doctor --profile ci
python -m src.cli.explain_config --profile ci --workflow strategy
python docs/examples/m39_first_run_configuration_profile_example.py
```

Notebook-friendly usage:

```python
from src.config.doctor import run_environment_doctor
from src.config.explain import build_runtime_explain_report
from src.config.resolution import resolve_runtime_profile_config

resolved = resolve_runtime_profile_config("ci").to_json_dict()
doctor = run_environment_doctor("ci").to_json_dict()
explain = build_runtime_explain_report("ci", workflow="strategy").to_json_dict()
```

Pipeline-friendly usage:

- run validation, doctor, and explain as preflight steps
- fail only on explicit validation or doctor failures
- archive reports as generated advisory evidence, not canonical artifacts
- keep generated first-run output out of release diffs unless intentionally
  changing tracked example-output policy

## Generated Output Policy

M39 generated reports are advisory, disposable, and non-authoritative.

Recommended generated-output locations:

- `artifacts/_derived/config_validation/`
- `artifacts/_derived/environment_readiness/`
- `artifacts/_derived/config_explain/`
- `docs/examples/output/m39_first_run_configuration_profile_example/`

The first-run example writes reports by default because onboarding should be
inspectable. The M39 first-run output directory is ignored by default, and
maintainers should not commit generated reports unless intentionally changing
the repository's tracked example-output policy.

## Architecture Guarantees Preserved

- canonical artifacts remain the source of truth
- direct scan remains available, default, and canonical
- starter profiles are non-secret context files, not a second source of truth
- workflow config files under `configs/` remain canonical workflow inputs
- derived config, readiness, explain, and first-run reports remain
  non-authoritative
- validation, doctor, explain, and first-run helpers do not execute workflows
- no live market data, credentials, network access, or external services are
  required
- evidence-review explain roots point to `artifacts/_derived/evidence_review`

## Non-Goals And Future Candidates

M39 does not implement:

- deployment, Docker, Kubernetes, or hosted runtime behavior
- dashboards, servers, remote metadata services, graph stores, or second
  registries
- secrets management infrastructure
- live-data validation or credential inspection
- workflow execution rewrites
- a new orchestration layer
- authoritative generated reports

Future M40+ candidates may add richer workflow-specific explain surfaces,
broader environment diagnostics, or release automation refinements, provided
they keep generated reports advisory and preserve artifact-first boundaries.

## Draft GitHub Release Notes

Title:
`M39 - Configuration Profiles and Environment Readiness`

Tag:
`v0.39.0-configuration-profiles-environment-readiness`

Branch:
`feature/m39-configuration-profiles-environment-readiness`

Summary:
M39 adds runtime profile contracts, deterministic config resolution with
provenance, a config validation CLI, an environment doctor, runtime explain
helpers, and a CI-safe first-run onboarding example. The stack helps users
inspect and reproduce configuration behavior before execution without requiring
live data, credentials, network access, or external services.

Highlights:

- Added non-secret runtime profile contracts and starter profiles.
- Added deterministic config resolution and per-field provenance.
- Added config validation, environment doctor, and runtime explain CLIs.
- Added CI-safe first-run onboarding example.
- Added generated-output hygiene and release-readiness documentation.
- Preserved canonical artifact, direct-scan, and derived-output boundaries.

Validation:

- Focused Ruff: `<record result>`
- Focused M39 pytest: `<record result>`
- Config validation CLI: `<record result>`
- Environment doctor CLI: `<record result>`
- Runtime explain CLI: `<record result>`
- First-run example smoke: `<record result>`
- Docs/path lint: `<record result>`
- Package build: `<record result>`
- Milestone validation: `<record result>`
- Hosted GitHub Actions: `<record result>`

Known boundaries:
M39 reports are advisory and non-authoritative. The milestone does not add
deployment behavior, dashboards, servers, remote services, graph stores, second
registries, secrets infrastructure, live-data validation, or workflow execution
changes.

## Further Reading

- `docs/runtime_profiles.md`
- `docs/runtime_configuration.md`
- `docs/getting_started.md`
- `docs/m39_release_validation_checklist.md`
- `docs/examples/m39_first_run_configuration_profile_example.py`
