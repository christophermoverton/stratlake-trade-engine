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
compatibility metadata, and derived/non-authoritative boundary flags. It is
metadata for future restore work, not a restore implementation.

## Future M43 Use

Later M43 issues can build on this contract and writer for:

* archive validation and inspection reports
* restore bootstrap workflows
* notebook quickstart flows
* optional cloud transport integrations
* CLI entrypoints

Restore/extraction is deferred to M43.3. Validation and inspection report APIs
are deferred to M43.4. CLI commands are deferred to M43.5.

Those future tools should treat the manifest as an inspectable description of
transport metadata. They should continue to reopen canonical manifests,
registries, checked-in configs, and repository-relative paths before making
decision-sensitive claims.
