# M43 Release Notes - Portable Notebook Session Archives

Milestone title:
`M43 - Portable Notebook Session Archives`

Candidate milestone release tag:
`v0.43.0-portable-notebook-session-archives`

Package/build version:
`0.43.0`

## M43 Mini-Release (Issue #481)

Deferred release tag:
`v0.43.2-session-archive-restore-bootstrap`

Package/build version:
`0.43.2`

Scope:
M43 mini-release readiness for session archive restore-bootstrap work.

Branch:
`feature/m43-session-archive-restore-bootstrap`

Primary issue:
`#481`

Related issue:
`#480`

### Mini-Release Summary

`v0.43.2-session-archive-restore-bootstrap` is the deferred tag for a
mini-release candidate under the already-published M43 archive line. It
prepares notebook-friendly session archive bootstrap commands for creating M43
portable archive packs, optionally copying them to mounted filesystem paths
such as Drive-mounted Colab folders, validating and inspecting before restore,
and restoring copied packs into explicit local target workspaces.

The command pair supports explicit copy and local archive collision policies,
whole-pack-safe skip-existing behavior, destination safety checks, optional
validation/inspection after copy, optional validation/inspection before restore,
deterministic bootstrap reporting, and structured JSON restore-bootstrap
failure summaries while preserving M43's derived, disposable, transport-only
archive boundary model.

### Mini-Release Highlights

* Added `stratlake-session-archive-bootstrap` and
  `python -m src.cli.session_archive_bootstrap` for thin M43 orchestration.
* Added `stratlake-session-archive-restore-bootstrap` and
  `python -m src.cli.session_archive_restore_bootstrap` for restore-side
  notebook bootstrap orchestration.
* Added optional mounted filesystem copy via `--drive-root`.
* Added explicit policy split:
  `--archive-collision-policy fail_if_exists|overwrite_allowed` for local
  archive generation and
  `--copy-policy fail_if_exists|skip_existing|overwrite_allowed` for mounted
  destination copy behavior.
* Hardened `skip_existing` to skip existing destination archive packs as a
  whole, avoiding mixed stale/new archive contents.
* Added destination safety checks that reject unsafe local/destination
  relationships before copy.
* Added optional post-copy validation and inspection through existing M43 APIs.
* Added deterministic derived bootstrap reporting under
  `artifacts/_derived/session_archives/<archive_id>/bootstrap_report.json`.
* Added optional pre-restore validation and inspection, dry-run restore
  planning, checksum verification, explicit restore target handling, and
  deterministic restore-bootstrap reporting through existing M43 restore APIs.
* Hardened restore-bootstrap failure output so JSON-mode notebook and
  automation callers receive deterministic structured failure payloads with
  boundary flags and best-effort archive IDs when `manifest.json` is readable.
* Updated release-readiness docs, validation commands, and local validation
  evidence for package/build version `0.43.2`.

### Mini-Release Boundaries

Session archive packs remain derived, disposable, transport-only snapshots.
They are not canonical storage, canonical evidence, or a registry.

This mini-release does not add Google API integration, OAuth, credentials,
cloud SDK behavior, network access, or background sync.

Workflow release-readiness coverage for this mini-release includes explicit
branch validation for
`feature/m43-session-archive-restore-bootstrap` and M43 patch-tag validation
path coverage through `v0.43.*` trigger matching in milestone branch
validation. Do not create or push
`v0.43.2-session-archive-restore-bootstrap` until PR completion, review, merge,
and post-merge release validation are complete.

## Milestone Principle

Portable session archives should make notebook and cloud workflows faster to
save, move, and restore without turning archive packs into canonical storage or
hiding execution behavior.

## Summary

M43 adds portable notebook session archives for StratLake workflows. Users can
pack selected repository-relative features, artifacts, configs, and optional
file-backed DuckDB snapshots into deterministic archive shards, validate and
inspect the archive pack before restore, restore into a normal local workspace,
and then run StratLake from the restored repository-relative paths.

Archive packs are derived, disposable, transport-only snapshots. They are not
canonical storage, canonical evidence, a registry, or active workflow inputs.

## Highlights

* Added the `src.session_archive.manifest` contract with deterministic
  serialization, portable path validation, shard metadata, restore expectations,
  runtime profile context, and explicit non-authoritative boundaries.
* Added a deterministic archive writer and shard planner with dry-run support,
  default excludes, collision policy handling, manifest generation,
  `archive_index.json`, `checksums.json`, `restore_plan.json`, and tar shards.
* Added local restore APIs with dry-run planning, checksum verification, safe
  regular-file extraction, overwrite policies, and deterministic restore
  reports.
* Added validation and inspection APIs that read archive packs without
  extraction, verify required sidecars and shards, stream checksum validation,
  reject unsafe tar entries, and write deterministic derived reports.
* Added CLI wrappers through `python -m src.cli.session_archive` for `pack`,
  `validate`, `inspect`, and `restore`.
* Added notebook and Colab workflow documentation describing mounted-storage
  archive persistence and local runtime restore.
* Added deterministic round-trip validation proving
  `pack -> validate -> inspect -> restore -> compare` on small synthetic
  StratLake-style data.

## Archive Pack Layout

M43 archive packs include:

```text
artifacts/_derived/session_archives/<archive_id>/
  manifest.json
  archive_index.json
  checksums.json
  restore_plan.json
  shards/
```

Validation treats all sidecars and required shards as pack metadata. Missing
sidecars, missing shards, checksum mismatches, malformed shard metadata, unsafe
archive entries, and unsafe restore paths fail clearly.

## DuckDB Snapshot Behavior

DuckDB snapshots are optional. `:memory:` DuckDB context is metadata-only.
File-backed snapshots can be included when useful and are restored back into
normal repository-relative paths. Optional DuckDB warnings are advisory unless
validation reports error-severity issues.

## Architecture Boundaries

M43 preserves these boundaries:

* archive packs are derived
* archive packs are disposable
* archive packs are transport-only
* archive packs are non-authoritative
* canonical StratLake artifacts remain the source of truth
* direct repository-relative paths remain active workflow inputs
* archive shards are not active workflow inputs
* validation, inspection, and restore do not execute strategy, alpha,
  portfolio, feature, or research workflows
* mounted storage is treated as local filesystem storage only
* no Google Drive API, credentials, network access, live market data, dashboard,
  server, remote metadata service, or second registry is introduced

## User-Facing Commands

```bash
python -m src.cli.session_archive pack \
  --repository-root . \
  --archive-id demo-session \
  --include-group features \
  --include-group artifacts \
  --include-group configs

python -m src.cli.session_archive validate \
  --archive-root artifacts/_derived/session_archives/demo-session \
  --output-root artifacts

python -m src.cli.session_archive inspect \
  --archive-root artifacts/_derived/session_archives/demo-session

python -m src.cli.session_archive restore \
  --archive-root mounted_drive/stratlake_archives/demo-session \
  --target-root restored_workspace \
  --dry-run

python -m src.cli.session_archive restore \
  --archive-root mounted_drive/stratlake_archives/demo-session \
  --target-root restored_workspace \
  --overwrite-policy fail_if_exists

stratlake-session-archive-bootstrap \
  --root /content/stratlake \
  --archive-id notebook-session-001 \
  --drive-root /content/drive/MyDrive/stratlake-colab/session_archives \
  --include-features \
  --include-artifacts \
  --include-configs \
  --validate-after-copy \
  --inspect-after-copy

stratlake-session-archive-restore-bootstrap \
  --archive-root /content/drive/MyDrive/stratlake-colab/session_archives/notebook-session-001 \
  --target-root /content/stratlake \
  --validate-before-restore \
  --inspect-before-restore \
  --dry-run

stratlake-session-archive-restore-bootstrap \
  --archive-root /content/drive/MyDrive/stratlake-colab/session_archives/notebook-session-001 \
  --target-root /content/stratlake \
  --validate-before-restore \
  --inspect-before-restore \
  --overwrite-policy fail_if_exists
```

## Validation Status

Final local validation for the `0.43.2` restore-bootstrap mini-release
readiness branch includes:

* `20 passed` for `tests/test_session_archive_bootstrap_cli.py`
* `24 passed` for `tests/test_session_archive_restore_bootstrap_cli.py`
* `99 passed, 1 skipped` for the focused session archive CLI, writer,
  restore, validation, and round-trip slice
* `8 passed` for packaging readiness
* `8 passed` for release workflow guards
* `3 passed` for docs/path portability
* Ruff check passed across `src`, `tests`, `docs`, `README.md`, and
  `pyproject.toml`
* targeted Ruff format checks passed for changed source/test/docs files
* package build produced `stratlake_trade_engine-0.43.2.tar.gz` and
  `stratlake_trade_engine-0.43.2-py3-none-any.whl`
* full pytest passed with `2475 passed, 6 skipped`
* `git diff --check` passed

The broader repository-wide Ruff format command still reports pre-existing
formatting drift outside this readiness change and README Markdown formatting
requires Ruff `--preview`; targeted changed-file format checks passed. The M43
release validation checklist records the final command set, results, and this
caveat.

## Non-Goals

M43 does not add Google Drive API integration, credential handling, live market
data access, cloud APIs, notebook widgets, dashboards, servers, package
publishing, production deployment, remote metadata services, or a second
canonical archive registry.

## Draft GitHub Release Notes

Title:
`M43 - Portable Notebook Session Archives`

Tag:
`v0.43.0-portable-notebook-session-archives`

Summary:
M43 adds deterministic portable notebook session archives for StratLake. Users
can create, validate, inspect, move, and locally restore selected session
contents while keeping active workflows on normal repository-relative feature,
artifact, and config paths.

Highlights:

* Manifest contract for portable session archive packs.
* Deterministic sharded archive writer with manifest, index, checksum, and
  restore-plan sidecars.
* Local restore APIs with dry-run planning, checksum verification, and safe
  tar extraction.
* Validation and inspection APIs with deterministic derived reports.
* CLI wrappers for pack, validate, inspect, and restore.
* Notebook/Colab workflow documentation.
* CI-safe deterministic round-trip validation.

Known boundaries:
Archive packs remain derived, disposable, transport-only snapshots. They are
not canonical storage, canonical evidence, a registry, or active workflow
inputs. M43 does not require Google Drive APIs, credentials, network access, or
live market data.
