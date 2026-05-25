# M42 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 42. It does not
replace existing CI, milestone validation, or release automation.

Milestone title:
`M42 - Notebook Project Sessions and Filesystem Drive Persistence`

M42 branch:
`release/m42-version-tag-update`

Candidate milestone release tag:
`v0.42.0-notebook-project-sessions-drive-persistence`

Package/build version:
`0.42.0`

## M42 Scope Recap

M42 adds:

* notebook project-session contract
* session metadata files under `.stratlake/`
* `stratlake-init-session`
* session-aware path resolution helpers
* filesystem-only mounted-Drive import/export
* Colab project-session documentation
* deterministic validation and packaging checks

## Focused Validation

Run Ruff over the M42-facing source and tests:

```bash
ruff check src/session src/cli tests
```

Run the focused M42 pytest slice:

```bash
pytest tests/test_init_notebook_workspace.py
pytest tests/test_notebook_project_session.py
pytest tests/test_init_session_cli.py
pytest tests/test_session_path_resolution.py
pytest tests/test_drive_persistence_adapter.py
pytest tests/test_session_import_export.py
pytest tests/test_m42_release_validation.py
```

Run the opt-in wheel smoke when the release environment is allowed to build and
install a local wheel in a temporary virtual environment:

```bash
pytest tests/test_notebook_bootstrap_wheel_smoke.py
```

The wheel smoke may be skipped unless
`STRATLAKE_RUN_NOTEBOOK_BOOTSTRAP_WHEEL_SMOKE=1` is set.

## Session Contract Validation

Confirm validation covers:

* stable session schema version
* `.stratlake/session.json` creation
* `.stratlake/path_resolution.json` creation
* project-internal POSIX-style relative serialization
* external MarketLake root classification
* optional Drive root classification
* path source/provenance for resolved roots
* no mutation of `.env`, `os.environ`, package resources, or canonical artifacts

## Session Bootstrap CLI Validation

Confirm validation covers:

* notebook workspace initialization through `stratlake-init-session`
* delegated starter-template copy behavior
* copied `configs/session.yml`
* copied `docs/colab_project_sessions.md`
* no-overwrite-by-default behavior
* `--force` refresh behavior for known templates and session metadata
* clear failures for unsafe session metadata writes

## Path-Resolution Validation

Confirm validation covers:

* `find_session_root`
* `load_session`
* `resolve_session_paths`
* `write_path_resolution_report`
* deterministic precedence:
  explicit overrides, session metadata, environment-variable fallbacks, defaults
* environment-variable provenance when a fallback is used
* no hidden CWD mutation

## Drive Filesystem Persistence Validation

Confirm validation covers:

* export dry-run behavior
* export actual copy behavior
* import roundtrip behavior
* no-overwrite-by-default behavior
* `--force` overwrite behavior
* safe default excludes:
  `.env`, credentials, secrets, API keys, notebook checkpoints, caches,
  bytecode, and temporary files
* feature data gated by `--include-features`
* market data gated by `--include-market-data`
* deterministic manifest ordering
* non-authoritative manifest metadata

## Docs And Resource Validation

Run docs/path portability validation:

```bash
pytest tests/test_docs_path_portability.py
pytest tests/test_portable_paths.py
pytest tests/test_m28_notebook_integration_examples.py
```

If present in the checkout, also run:

```bash
pytest tests/test_docs_path_lint.py
```

Inspect these docs before tagging:

* [Notebook Workspace Bootstrap](notebook_workspace_bootstrap.md)
* [Notebook Integration](notebook_integration.md)
* [Colab Project Sessions](colab_project_sessions.md)
* [M42 Release Notes](m42_release_notes.md)

Confirm:

* Colab-specific `/content/...` paths are confined to Colab-scoped examples
* no local Windows absolute paths appear in docs
* no `file://` links appear in docs
* no docs describe Drive copies as canonical artifacts
* no docs imply Google API, OAuth, credentials, or network access
* copied resource docs match release-facing guidance

## Package Build Validation

Run package build validation:

```bash
python -m build
```

Confirm packaged resources include:

* `configs/session.yml`
* `docs/notebook_integration.md`
* `docs/colab_project_sessions.md`
* existing notebook workspace starter resources

Confirm installed entry points include:

* `stratlake-init-session`
* `stratlake-session-export`
* `stratlake-session-import`

Confirm package/build version metadata reports:

* `0.42.0`

If the build emits the existing setuptools license-table deprecation warning,
record it as existing and non-blocking unless this branch changes packaging
metadata.

## Full Test Validation

Run full pytest when practical:

```bash
pytest
```

No M42 test should require:

* network access
* Google Drive credentials
* OAuth
* a real mounted Drive
* live market data
* external services

## Generated Output Hygiene

Before merging:

* confirm `git status --short` contains only intentional source/doc changes
* confirm no generated `docs/examples/output/` directories are staged
* confirm no `dist/`, `build/`, or `*.egg-info/` outputs are staged
* confirm no local absolute paths appear in M42 docs
* confirm no credentials, tokens, API keys, or passwords are staged

## Known Non-Goals And Preserved Boundaries

M42 does not implement:

* Google API calls
* OAuth
* network access
* background sync
* daemon behavior
* remote registries
* remote metadata services
* credential workflows
* live market data dependency
* hidden CWD mutation
* `.env` mutation
* `os.environ` mutation
* canonical artifact schema changes beyond the diagnostic session metadata

Confirm M42 keeps the artifact-first boundary intact:

* session files are diagnostic notebook/session state
* Drive copies are persistence snapshots only
* Drive manifests are diagnostic and non-authoritative
* canonical artifacts remain authoritative
* no second source of truth is introduced

## Release Tag Checklist

Before creating `v0.42.0-notebook-project-sessions-drive-persistence`:

* focused M42 validation is green
* docs/path portability validation is green
* package build validation is green
* full pytest is green or any skipped tests are documented
* release notes are final
* generated output hygiene check is clean
* branch name and tag candidate are still accurate

Prepare GitHub Release notes from [M42 Release Notes](m42_release_notes.md).
