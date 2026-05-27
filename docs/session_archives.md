# Portable Session Archives

## Overview

Portable session archives are derived transport snapshots for notebook and
cloud workflows. They are meant to make a selected StratLake session easier to
save, move, inspect, and restore in later M43 work.

An archive pack is not canonical storage. Canonical Parquet features,
strategy artifacts, alpha artifacts, portfolio artifacts, governance outputs,
evidence outputs, checked-in configs, and direct repository-relative scans
remain authoritative. A session archive manifest is only metadata about a
disposable archive pack.

Issue #465 added the manifest contract. Issue #466 adds the Python archive
writer and deterministic shard planner. The writer creates derived archive
packs only; it does not extract archives, restore files, call Google Drive, read
live market data, create CLI commands, or mutate canonical artifacts.

## Manifest Contract

The Python contract lives in `src/session_archive/manifest.py`.

The manifest includes:

* `schema_version`: current manifest schema version, currently `1`.
* `archive_id`: stable archive identifier chosen by the future writer.
* `session_id`: optional source notebook/session identifier.
* `created_at_utc`: optional timestamp supplied by the caller. The manifest
  helper does not generate wall-clock time implicitly.
* `source_runtime_profile`: optional runtime profile name such as `notebook`.
* `source_profile_path`: optional repository-relative profile path.
* `source_roots`: repository-relative roots represented in the archive.
* `included_groups`: selected logical groups, currently `features`,
  `artifacts`, `configs`, and optionally `duckdb_snapshot`.
* `shards`: deterministic shard metadata.
* `duckdb_snapshot`: optional metadata for a DuckDB snapshot. `:memory:` is
  allowed only for DuckDB source-path metadata.
* `restore`: future restore expectations, including target relative roots,
  overwrite policy metadata, and compatibility metadata.
* `boundaries`: explicit non-canonical archive boundary flags.
* `metadata`: small non-secret descriptive metadata.

Shard metadata includes:

* `shard_name`
* `shard_path`
* `logical_group`
* `shard_index`
* `file_count`
* `size_bytes`
* `checksum_algorithm`
* `checksum`
* `archive_format`
* `compression`

## Boundary Flags

Every valid manifest must preserve these values:

```json
{
  "derived": true,
  "disposable": true,
  "transport_only": true,
  "authoritative": false,
  "canonical_storage": false,
  "requires_network": false,
  "requires_credentials": false,
  "requires_live_market_data": false
}
```

These flags make the non-goal explicit: restoring from an archive may help
bootstrap a workspace later, but the archive does not replace canonical
repository artifacts or execution provenance.

## Path Portability

Manifest paths must be normalized repository-relative POSIX paths, for example:

```text
configs/profiles/notebook.yml
data/curated/features_daily
artifacts/strategies/example_run/manifest.json
archives/session-a/shards/features-000.tar.zst
```

The validator rejects:

* absolute POSIX paths
* Windows drive paths, including drive-relative forms such as `C:relative/path`
* home-directory shortcuts such as `~/data`
* `file://` URIs
* URL-like paths
* path traversal such as `../data`
* backslash-separated paths

The only special non-path value is `:memory:`, and it is allowed only for
DuckDB source-path metadata.

Shard names must be simple filename-like values. They must not be `.`, `..`,
contain path separators, or start with a Windows drive prefix.

## Deterministic JSON

`manifest_to_deterministic_json(...)` validates the manifest, sorts object
keys, sorts shard records by logical group, shard index, shard name, and shard
path, disables non-finite JSON numbers, and writes one trailing newline.

`write_session_archive_manifest(...)` writes the same deterministic bytes using
the repository's atomic text writer. Identical logical manifest inputs should
produce byte-stable JSON.

## Archive Writer

The writer API lives in `src/session_archive/writer.py`:

```python
from src.session_archive.writer import (
    SessionArchiveIncludePolicy,
    SessionArchiveWriteRequest,
    build_session_archive_plan,
    write_session_archive_pack,
)
```

`build_session_archive_plan(...)` performs a dry-run plan. It enumerates files,
applies excludes, assigns files to deterministic shards, builds manifest,
index, checksum, and restore-plan payloads, and writes nothing.

`write_session_archive_pack(...)` builds the same plan and writes a derived
archive pack under a repository-relative output root.

By default, writes use `collision_policy="fail_if_exists"`. If the target
archive root already exists and contains files, the writer raises instead of
silently replacing a previous archive pack. Empty existing archive roots are
allowed. Intentional replacement requires `collision_policy="overwrite_allowed"`;
that policy only replaces known generated archive children under the derived
archive root:

```text
manifest.json
archive_index.json
checksums.json
restore_plan.json
shards/
```

Collision handling applies only to derived archive outputs. It does not mutate
canonical feature, artifact, or config inputs. Dry-run planning remains
write-free and never creates, clears, or overwrites an archive root.

User metadata is allowed for non-secret descriptive fields. The writer reserves
`writer`, `artifact_role`, and `collision_policy` so generated manifests cannot
misrepresent writer identity, archive role, or collision behavior.

The default layout is:

```text
artifacts/_derived/session_archives/<archive_id>/
  manifest.json
  archive_index.json
  checksums.json
  restore_plan.json
  shards/
    features__000.tar
    artifacts__000.tar
    configs__000.tar
```

The writer supports these include groups:

* `features`
* `artifacts`
* `configs`
* `duckdb_snapshot`

By default, features point at `data/curated`, artifacts point at `artifacts`,
and configs point at `configs`. Callers can pass explicit repository-relative
include paths per group. File-backed DuckDB snapshot paths are included only
when supplied and when `duckdb_snapshot` is selected. `:memory:` DuckDB context
is represented as metadata only.

## Sharding And Excludes

Shard planning is deterministic:

* traversal uses sorted repository-relative paths
* shard names use `<logical_group>__NNN.tar`
* shard assignment respects max byte and max entry thresholds
* archive member names are repository-relative POSIX paths
* tar member metadata is normalized with fixed uid, gid, owner names, mode, and
  `mtime`
* generated JSON sidecars use sorted keys and one trailing newline

M43.2 writes standard-library `tar` archives with `compression=none`. Zstandard
or other compression backends are deferred until a later issue can validate
portable deterministic behavior without adding hidden dependencies.

The default exclude policy skips noisy or unsafe local paths, including:

* `.git`
* `.venv`
* `__pycache__`
* `.pytest_cache`
* `.ruff_cache`
* `.mypy_cache`
* `.ipynb_checkpoints`
* `.DS_Store`
* `*.tmp`
* `*.temp`
* `artifacts/_derived/session_archives`

The archive output directory is excluded so a pack does not recursively include
itself when archiving the `artifacts` tree.

## Writer Sidecars

`manifest.json` uses the manifest contract documented above and validates with
`validate_session_archive_manifest(...)`.

`archive_index.json` records the archive ID, logical groups, included source
paths, file inventory, shard assignments, file counts, sizes, and the same
derived/non-authoritative boundary flags.

`checksums.json` records SHA-256 checksums for shard files and included source
files. These checksums support transfer verification; they do not make archive
packs canonical storage.

`restore_plan.json` records target relative roots, overwrite policy metadata,
writer collision policy, compatibility metadata, and derived/non-authoritative
boundary flags. It is metadata for future restore work, not a restore
implementation.

The writer emits all three sidecars for every archive pack. M43 validation
treats `archive_index.json`, `checksums.json`, and `restore_plan.json` as
required pack metadata. If any are missing, validation fails so users can repair
or recreate the derived pack before restore.

## Local Restore

M43.3 adds a local filesystem restore API in `src/session_archive/restore.py`:

```python
from src.session_archive.restore import (
    SessionArchiveRestoreRequest,
    build_session_archive_restore_plan,
    restore_session_archive_pack,
)
```

`build_session_archive_restore_plan(...)` is the dry-run path. It reads and
validates `manifest.json`, reads sidecars when present, verifies required shard
presence, optionally checks shard checksums, inspects tar members without
extracting, applies overwrite-policy decisions, and writes nothing.

`restore_session_archive_pack(...)` builds that plan, extracts regular file
members into the selected local `target_root`, and optionally writes a derived
restore report.

The restore request supports:

* `archive_root`
* `target_root`
* `overwrite_policy`
* `verify_checksums`
* `write_report`
* `report_root`

Supported restore overwrite policies are:

* `fail_if_exists`: fail before extraction if any target file already exists.
* `skip_existing`: leave existing target files unchanged and record skipped
  entries.
* `replace_existing`: intentionally replace existing target files.

Restore supports the M43.2 archive format: standard-library `tar` shards with
`compression=none`. Other archive formats or compression modes are rejected.

Restore rejects unsafe member paths such as absolute paths, traversal paths,
home-directory shortcuts, file URIs, URL-like paths, Windows drive paths, and
backslash-separated paths. Restore also rejects symlinks, hardlinks, device
files, FIFO files, directories, and other non-regular tar members. Parent
directories are created implicitly from safe regular-file paths.

Restore rejects duplicate member paths inside a shard before extraction so
malformed archives cannot create ambiguous overwrite behavior within a single
archive.

When `verify_checksums=True`, restore verifies each shard checksum before any
extraction. If `checksums.json` includes file checksums, restored file content
is checked as it is written.

The default restore report path is:

```text
<target_root>/artifacts/_derived/session_archives/<archive_id>/restore_report.json
```

The report is deterministic derived JSON. It records archive ID, portable source
and target labels, overwrite policy, checksum status, restored entries, skipped
entries, warnings, manifest metadata, and non-authoritative boundaries. It is a
restore audit surface only; it is not canonical research evidence or a second
artifact registry.

Archive roots and target roots may be local mounted paths, including a locally
mounted cloud drive folder, but restore does not call Google Drive APIs, require
credentials, use network services, or access live market data.

## Validation And Inspection

M43.4 adds read-only validation and inspection APIs in
`src/session_archive/validation.py`:

```python
from src.session_archive.validation import (
    inspect_session_archive,
    validate_session_archive,
    write_session_archive_inspection_report,
    write_session_archive_validation_report,
)
```

`validate_session_archive(...)` checks an archive pack before restore without
extracting it. It verifies that `manifest.json` is present and valid, the
manifest schema and boundary fields are supported, required sidecars and shards
are present, shard metadata is well formed, shard checksums match when
requested, archive index counts and sizes are internally consistent, restore
roots are portable, and tar members are safe regular files with
repository-relative POSIX paths. Shard checksum verification uses streaming
reads so large notebook/cloud transfer shards do not need to be loaded into
memory all at once.

`inspect_session_archive(...)` summarizes the same pack without extraction. It
reports the archive ID, schema version, optional session/profile context,
included logical groups, missing optional groups, shard summary, estimated
restored file count and byte size, restore-root expectations, boundary status,
DuckDB snapshot status, portability status, warnings, and errors.

Both APIs return structured issues with stable codes and `warning` or `error`
severity. Errors mean the pack should not be restored until corrected. Warnings
describe non-fatal context, such as an optional logical group not being included
or `:memory:` DuckDB metadata that intentionally has no file-backed snapshot.

The warning/error taxonomy includes distinct codes for missing or malformed
manifests, unsupported schema versions, missing required fields, missing shard
index sidecars, missing checksum sidecars, missing restore-plan sidecars,
malformed shard indexes, missing shards, checksum mismatches, malformed shard
metadata, archive index inconsistencies, unsafe archive entries, unsafe restore
paths, missing optional groups, optional DuckDB metadata, empty packs, unknown
logical groups, non-portable paths, and report write failures.

The report writers emit deterministic derived JSON. The preferred report
destination is an explicit `output_root`, which writes outside the archive pack:

```text
<output_root>/_derived/session_archives/<archive_id>/validation_report.json
<output_root>/_derived/session_archives/<archive_id>/inspection_report.json
```

For example, passing `output_root="artifacts"` writes under
`artifacts/_derived/session_archives/<archive_id>/`.

Explicit `output_path` values are still supported for callers that need an
exact destination. Supplying both `output_path` and `output_root` is rejected to
avoid ambiguous report placement. If neither is supplied, the report writers
preserve the convenience behavior of writing `validation_report.json` or
`inspection_report.json` adjacent to the archive pack root. Adjacent reports are
derived convenience outputs only; they are not canonical archive contents,
canonical evidence, or a registry.

Report content uses archive-relative labels where possible and preserves the
same derived, disposable, transport-only, non-authoritative boundary flags.
Validation and inspection do not scan outside the archive pack, mutate restore
targets, execute workflows, read live market data, call cloud APIs, require
credentials, or make the archive canonical.

Users should validate and inspect archive packs before local restore,
especially when the pack was copied through mounted storage such as a local
cloud-drive folder. Mounted paths are treated as local filesystem paths only.

## Future M43 Use

Later M43 issues can build on this contract and writer for:

* notebook quickstart flows
* optional cloud transport integrations
* CLI entrypoints

CLI commands are deferred to M43.5.

Those future tools should treat the manifest as an inspectable description of
transport metadata. They should continue to reopen canonical manifests,
registries, checked-in configs, and repository-relative paths before making
decision-sensitive claims.
