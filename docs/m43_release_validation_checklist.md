# M43 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 43. It does not
replace existing CI, milestone validation, or release automation.

Milestone title:
`M43 - Portable Notebook Session Archives`

Candidate milestone release tag:
`v0.43.0-portable-notebook-session-archives`

Package/build version:
`0.43.2`

## M43 Mini-Release Checklist (Issue #481)

Deferred release tag:
`v0.43.2-session-archive-restore-bootstrap`

Package/build version:
`0.43.2`

Scope:
M43 mini-release readiness for session archive restore-bootstrap commands.

Branch:
`feature/m43-session-archive-restore-bootstrap`

Primary issue:
`#481`

Related issue:
`#480`

Release tag status:
No release tag has been created or pushed. Do not create
`v0.43.2-session-archive-restore-bootstrap` until PR completion, review, merge,
and final post-merge release validation are complete.

### Required Validation Commands

```bash
python -m pytest tests/test_session_archive_bootstrap_cli.py
python -m pytest tests/test_session_archive_restore_bootstrap_cli.py
python -m pytest tests/test_session_archive_cli.py tests/test_session_archive_writer.py tests/test_session_archive_validation.py tests/test_session_archive_restore.py tests/test_session_archive_roundtrip_validation.py
python -m pytest tests/test_packaging_readiness.py
python -m pytest tests/test_release_workflow.py
python -m pytest tests/test_docs_path_portability.py
python -m ruff check src tests docs
python -m ruff format --check src tests docs README.md pyproject.toml
python -m build
git diff --check
```

Run `python -m pytest tests/test_packaging_workflow.py` only if that optional
workflow guard test file is present in the repository.

### Validation Results

Validation results for the `0.43.2` restore-bootstrap mini-release readiness
pass are recorded in [Final Local Validation Status](#final-local-validation-status).

* `python -m pytest tests/test_session_archive_bootstrap_cli.py`
  * `20 passed`
* `python -m pytest tests/test_session_archive_restore_bootstrap_cli.py`
  * `24 passed`
* branch-validation-equivalent M43 pytest slice including restore-bootstrap:
  * `172 passed, 1 skipped`
* `python -m pytest tests/test_packaging_readiness.py`
  * `8 passed`
* `python -m pytest tests/test_session_archive_cli.py tests/test_session_archive_writer.py tests/test_session_archive_validation.py tests/test_session_archive_restore.py tests/test_session_archive_roundtrip_validation.py`
  * `99 passed, 1 skipped`
* `python -m ruff check src tests docs`
  * `All checks passed!`
* package build:
  * built `stratlake_trade_engine-0.43.2.tar.gz` and
    `stratlake_trade_engine-0.43.2-py3-none-any.whl`

### Workflow Branch/Tag Coverage (Issue #481 Scope Update)

Branch validation coverage:

* `feature/m43-session-archive-restore-bootstrap`
* `feature/m*` (existing durable milestone pattern)

Tag validation coverage:

* `v0.43.*` in milestone branch validation workflow
* `v*` in release and TestPyPI workflows (existing durable release pattern)

No `v0.43.2-session-archive-restore-bootstrap` tag should exist before PR
completion and post-merge validation.

Workflow files reviewed:

* `.github/workflows/ci.yml`
* `.github/workflows/milestone_branch_validation.yml`
* `.github/workflows/milestone_validation.yml`
* `.github/workflows/release.yml`
* `.github/workflows/publish-testpypi.yml`

Workflow files changed:

* `.github/workflows/milestone_branch_validation.yml`

This scope update changes focused branch validation coverage only and does not
change workflow job logic. The branch validation workflow now includes the
restore-bootstrap CLI surface in its M43 pytest, Ruff check, and Python format
check steps.

Remote workflow runs for this update were not observed locally in this
checklist; this section records trigger coverage and local validation only.

No repository-specific YAML lint command found.

## M43 Scope Recap

M43 adds:

* portable session archive manifest contract
* deterministic archive writer and shard planner
* local restore APIs with dry-run planning
* validation and inspection APIs
* CLI commands for `pack`, `validate`, `inspect`, and `restore`
* notebook and Colab workflow documentation
* deterministic round-trip validation

## Focused M43 Validation

Run the focused M43 archive pytest slice:

```bash
python -m pytest tests/test_session_archive_manifest.py tests/test_session_archive_writer.py tests/test_session_archive_restore.py tests/test_session_archive_validation.py tests/test_session_archive_cli.py tests/test_session_archive_bootstrap_cli.py tests/test_session_archive_restore_bootstrap_cli.py tests/test_session_archive_roundtrip_validation.py
```

Run workflow and release guard tests:

```bash
python -m pytest tests/test_github_actions_pinning.py tests/test_release_workflow.py
```

Run workflow-pattern policy checks:

```bash
python -m pytest tests/test_m36_deterministic_validation.py::test_cli_api_parity_and_release_hardening_assumptions tests/test_milestone_validation_workflow.py::test_milestone_validation_covers_current_milestone_branch_pattern
```

Run docs/path portability validation:

```bash
python -m pytest tests/test_docs_path_portability.py
```

Run Ruff over M43-facing source, tests, and docs:

```bash
python -m ruff check src/session_archive src/cli/session_archive.py src/cli/session_archive_bootstrap.py src/cli/session_archive_restore_bootstrap.py tests/test_session_archive_manifest.py tests/test_session_archive_writer.py tests/test_session_archive_restore.py tests/test_session_archive_validation.py tests/test_session_archive_cli.py tests/test_session_archive_bootstrap_cli.py tests/test_session_archive_restore_bootstrap_cli.py tests/test_session_archive_roundtrip_validation.py docs/session_archives.md README.md docs/getting_started.md src/resources/notebook_workspace/docs/getting_started.md
```

Run format checks:

```bash
python -m ruff format --check src/session_archive src/cli/session_archive.py src/cli/session_archive_bootstrap.py src/cli/session_archive_restore_bootstrap.py tests/test_session_archive_manifest.py tests/test_session_archive_writer.py tests/test_session_archive_restore.py tests/test_session_archive_validation.py tests/test_session_archive_cli.py tests/test_session_archive_bootstrap_cli.py tests/test_session_archive_restore_bootstrap_cli.py tests/test_session_archive_roundtrip_validation.py
python -m ruff format --check --preview docs/session_archives.md README.md docs/getting_started.md src/resources/notebook_workspace/docs/getting_started.md
```

Run package build validation:

```bash
python -m build
```

If `twine` is available in the validation environment, run:

```bash
python -m twine check dist/*
```

Run the whitespace/path check:

```bash
git diff --check
```

Run full pytest when practical:

```bash
python -m pytest
```

If full pytest has unrelated failures, record them separately from M43 archive
validation results.

## Final Local Validation Status

As of the Issue #481 restore-bootstrap mini-release readiness pass:

* archive bootstrap CLI:
  `20 passed`
* restore-bootstrap CLI:
  `24 passed`
* branch-validation-equivalent M43 pytest slice including restore-bootstrap:
  `172 passed, 1 skipped`
* focused M43 archive suite:
  `99 passed, 1 skipped`
* release workflow guard:
  `8 passed`
* packaging workflow guard:
  not present in this repository; no `tests/test_packaging_workflow.py` file
  was found
* docs/path portability tests:
  `3 passed`
* packaging readiness:
  `8 passed`
* Ruff check:
  `All checks passed!`
* broad Ruff format check:
  requested repository-wide command was run and exposed pre-existing
  repo-wide formatting drift plus README Markdown parsing that requires
  `--preview`; targeted changed-file format checks passed
* targeted Python Ruff format check:
  `1 file already formatted`
* targeted docs Ruff format check:
  `6 files already formatted`
* package build validation:
  built `stratlake_trade_engine-0.43.2.tar.gz` and
  `stratlake_trade_engine-0.43.2-py3-none-any.whl`
* package artifact validation:
  not run; `twine` validation remains optional for environments that include
  it
* whitespace/path check:
  `git diff --check` passed
* changed-doc path scan:
  no machine-local absolute path patterns, real Colab Drive paths, or
  workstation-specific cloud storage names found in the M43-facing docs
* full pytest:
  `2475 passed, 6 skipped`
* release tag:
  no `v0.43.2-session-archive-restore-bootstrap` tag was created or found
  locally
* GitHub release:
  not published

Workflow trigger policy:

* `.github/workflows/milestone_branch_validation.yml` owns focused
  `feature/m*` branch-push validation, including the M43 session archive test
  slice
* `.github/workflows/milestone_validation.yml` remains the broader milestone
  validation bundle for manual dispatch, pull requests into `main`, and legacy
  milestone branch patterns
* full-pytest workflow-pattern checks enforce this split so focused branch-push
  validation is not confused with broader release-readiness validation

Package build emitted the existing setuptools deprecation warning for
`project.license` as a TOML table. The M43 build completed successfully; the
license metadata modernization is release hygiene for a later packaging pass.

## Package Build Checklist

Confirm package/build metadata reports:

* `0.43.2`

Confirm package build artifacts are generated under `dist/` and are not staged
unless release policy explicitly asks for them.

Confirm source distributions and wheels include the session archive package and
the CLI modules:

* `src/session_archive/`
* `src/cli/session_archive.py`
* `src/cli/session_archive_bootstrap.py`
* `src/cli/session_archive_restore_bootstrap.py`

## Release Tag Preparation

Create the restore-bootstrap mini-release tag only after PR completion, review,
merge to `main`, and post-merge validation:

```bash
git tag -a v0.43.2-session-archive-restore-bootstrap -m "M43.3 - Session Archive Restore Bootstrap"
git push origin v0.43.2-session-archive-restore-bootstrap
```

Tag-driven release workflow notes:

* pre-merge validation should pass on the feature branch
* merge to `main`
* run post-merge validation on `main`
* create the annotated tag from the validated merge commit
* confirm the tag-driven release workflow starts
* confirm package build artifacts and deterministic release notes are uploaded
* confirm the GitHub Release body uses the M43 release notes
* package publication to PyPI/TestPyPI remains out of scope unless a separate
  release process explicitly enables it

## Docs And Link Alignment

Review these docs before tagging:

* [Session Archives](session_archives.md)
* [Getting Started](getting_started.md)
* [Notebook Workspace Getting Started](../src/resources/notebook_workspace/docs/getting_started.md)
* [M43 Release Notes](m43_release_notes.md)

Confirm docs use repository-relative placeholders such as:

* `mounted_drive/`
* `restored_workspace/`
* `artifacts/`
* `data/`
* `configs/`

Confirm docs do not introduce machine-local absolute paths, credential names,
or real Google Drive mount paths.

## Architecture Boundary Review

Confirm release-facing docs preserve these boundaries:

* archive packs are derived
* archive packs are disposable
* archive packs are transport-only
* archive packs are non-authoritative
* canonical StratLake artifacts remain the source of truth
* direct repository-relative paths remain active workflow inputs
* archive shards are not active workflow inputs
* validation, inspection, and restore do not execute strategy, alpha,
  portfolio, feature, or research workflows
* Google Drive is optional mounted storage, not a dependency
* no Google Drive API, credentials, network access, live market data, dashboard,
  server, remote metadata service, or second registry is introduced

## Generated Output Hygiene

Before merging:

* confirm `git status --short` contains only intentional source/doc changes
* confirm no generated `docs/examples/output/` directories are staged
* confirm no `dist/`, `build/`, or `*.egg-info/` outputs are staged
* confirm no local absolute paths appear in M43 docs
* confirm no credentials, tokens, API keys, or passwords are staged

## Non-Goals

M43 does not implement:

* archive packs as canonical storage
* archive packs as canonical evidence
* an archive registry
* Google Drive API integration
* credential handling
* network access
* live market data access
* cloud APIs
* dashboards, servers, or remote metadata services
* running active workflows directly from archive shards

Prepare GitHub Release notes from [M43 Release Notes](m43_release_notes.md).
