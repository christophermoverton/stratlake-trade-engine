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

Issue #465 adds the manifest contract only. It does not write archive shards,
extract archive contents, restore files, call Google Drive, read live market
data, create CLI commands, or mutate canonical artifacts.

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

## Future M43 Use

Later M43 issues can build on this contract for:

* archive shard writing
* archive validation and inspection reports
* restore bootstrap workflows
* notebook quickstart flows
* optional cloud transport integrations
* CLI entrypoints

Those future tools should treat the manifest as an inspectable description of
transport metadata. They should continue to reopen canonical manifests,
registries, checked-in configs, and repository-relative paths before making
decision-sensitive claims.
