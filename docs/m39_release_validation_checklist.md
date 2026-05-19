# M39 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 39 configuration
profiles, resolution provenance, validation, environment doctor, explain
helpers, and first-run onboarding. It does not replace existing CI, milestone
validation, or release automation.

## Pre-Merge Validation

Run the focused M39 validation slice:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_m39_first_run_example.py tests\test_runtime_explain.py tests\test_environment_doctor.py tests\test_validate_config_cli.py tests\test_config_resolution.py tests\test_runtime_profiles.py tests\test_runtime_config.py tests\test_docs_path_portability.py -q
```

Run Ruff over the M39-facing source, tests, and examples:

```powershell
.\.venv\Scripts\ruff.exe check docs\examples src\config src\cli tests
```

Run the CI-safe first-run example:

```powershell
.\.venv\Scripts\python.exe docs\examples\m39_first_run_configuration_profile_example.py --output-root docs\examples\output\m39_first_run_configuration_profile_example
```

Run docs/path lint:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_docs_path_lint --output artifacts\qa\docs_path_lint_m39.json
```

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
- confirm generated reports remain advisory and non-authoritative

## Architecture Checks

Confirm M39 keeps the configuration and artifact boundaries intact:

- canonical artifacts remain the source of truth
- direct scan remains available, default, and canonical
- runtime profiles remain non-secret starter context, not a second source of
  truth
- validation, doctor, explain, and first-run reports remain advisory
- derived outputs remain disposable and non-authoritative
- no workflow execution is introduced by validation, doctor, explain, or the
  first-run onboarding example
- no live market data, credentials, network access, external services,
  deployment manifests, dashboards, servers, graph stores, or second registries
  are required
