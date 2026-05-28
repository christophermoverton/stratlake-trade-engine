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

M43 adds the manifest contract, deterministic writer, local restore workflow,
validation and inspection APIs, and thin CLI wrappers. These surfaces create,
check, summarize, and restore derived archive packs only. They do not call
Google Drive, read live market data, require credentials, execute research
workflows, or mutate canonical artifacts outside the user-selected archive or
restore targets.

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

## CLI Usage

M43.5 adds thin CLI wrappers in `src/cli/session_archive.py`. The CLI delegates
to the shared Python APIs documented above; it does not implement a second
writer, restore path, validator, inspector, checksum routine, or tar extractor.

The commands are notebook-friendly and can be run from terminal sessions or
notebook shell cells:

```bash
python -m src.cli.session_archive_bootstrap \
  --root . \
  --archive-id session-a \
  --include-features \
  --include-artifacts \
  --include-configs

python -m src.cli.session_archive_bootstrap \
  --root . \
  --archive-id session-a \
  --drive-root /content/drive/MyDrive/stratlake-colab/session_archives \
  --include-features \
  --include-artifacts \
  --include-configs \
  --validate-after-copy \
  --inspect-after-copy

python -m src.cli.session_archive pack \
  --repository-root . \
  --archive-id session-a \
  --include-group features \
  --include-group artifacts \
  --include-group configs

python -m src.cli.session_archive validate \
  --archive-root artifacts/_derived/session_archives/session-a \
  --output-root artifacts

python -m src.cli.session_archive inspect \
  --archive-root artifacts/_derived/session_archives/session-a

python -m src.cli.session_archive restore \
  --archive-root artifacts/_derived/session_archives/session-a \
  --target-root restored-session \
  --dry-run

python -m src.cli.session_archive restore \
  --archive-root artifacts/_derived/session_archives/session-a \
  --target-root restored-session
```

For a mounted archive pack, restore from the copied archive root and run a
dry-run first:

```bash
python -m src.cli.session_archive restore \
  --archive-root /content/drive/MyDrive/stratlake-colab/session_archives/session-a \
  --target-root /content/restored-stratlake \
  --dry-run
```

The installed command `stratlake-session-archive-bootstrap` is a thin wrapper
over `python -m src.cli.session_archive_bootstrap`.

`session_archive_bootstrap` orchestrates existing M43 APIs only:

* pack creation through `write_session_archive_pack(...)`
* dry-run plan creation through `build_session_archive_plan(...)`
* optional post-copy validation through `validate_session_archive(...)`
* optional post-copy inspection through `inspect_session_archive(...)`

It does not implement a second writer, validator, inspector, tar format, or
restore path. Mounted Drive roots are treated as ordinary local filesystem
paths only. The command does not call Google APIs, does not require
credentials, does not require network access, and does not make archive packs
canonical storage.

When `--drive-root` is supplied, the copy destination is:

```text
<drive-root>/<archive-id>/
```

Supported copy policies are:

* `fail_if_exists`: fail when destination archive root already contains files.
* `skip_existing`: if destination archive root already contains files, skip the
  destination archive pack as a whole.
* `overwrite_allowed`: intentionally replace destination archive contents.

Use `--archive-collision-policy` to control local derived archive generation:

* `fail_if_exists` (default): fail if local archive output already exists.
* `overwrite_allowed`: intentionally regenerate local derived archive output.

Policy split for explicit reruns:

* `--archive-collision-policy` controls local derived archive creation.
* `--copy-policy` controls copied archive behavior under `--drive-root`.

Mounted destination safety: do not set `--drive-root` to the local archive
output directory or a child of the local archive pack.

Recommended explicit rerun pattern:

```bash
stratlake-session-archive-bootstrap \
  --root /content/stratlake \
  --archive-id notebook-session-001 \
  --archive-collision-policy overwrite_allowed \
  --drive-root /content/drive/MyDrive/stratlake-colab/session_archives \
  --copy-policy overwrite_allowed \
  --include-features \
  --include-artifacts \
  --include-configs \
  --validate-after-copy \
  --inspect-after-copy
```

For non-dry-run operations, bootstrap writes a deterministic derived report at:

```text
<root>/artifacts/_derived/session_archives/<archive_id>/bootstrap_report.json
```

Dry-run planning writes no archive pack, no destination copy, and no bootstrap
report.

The installed command `stratlake-session-archive-restore-bootstrap` is the
restore-side companion for notebook and mounted-storage workflows:

```bash
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

Restore bootstrap delegates to existing M43 APIs only:

* optional pre-restore validation through `validate_session_archive(...)`
* optional pre-restore inspection through `inspect_session_archive(...)`
* dry-run planning through `build_session_archive_restore_plan(...)`
* actual restore through `restore_session_archive_pack(...)`

The command exposes notebook-friendly overwrite policies:

* `fail_if_exists`
* `skip_existing`
* `overwrite_allowed`

`overwrite_allowed` maps to the lower-level restore API's intentional
replacement behavior. Dry-run writes no restored files and no restore-bootstrap
report. Non-dry-run restore writes a deterministic derived report at:

```text
<target_root>/artifacts/_derived/session_archives/<archive_id>/restore_bootstrap_report.json
```

When `--report-root` is supplied, the restore report and restore-bootstrap
report are written there under the same derived-report boundary. The restore
target remains explicit. The command does not call Google APIs, require
credentials, start background sync, execute research workflows, mutate
canonical sources outside `--target-root`, or make archive packs canonical.

`pack` creates an archive pack through `write_session_archive_pack(...)`.
`--dry-run` delegates to `build_session_archive_plan(...)` and writes no pack.
Use `--include-group` to select logical groups and `--include-path GROUP=PATH`
to override repository-relative include roots.

`validate` and `inspect` return exit code `0` for passed or warning-only
results and non-zero when error-severity issues are present. Both commands
support `--output-path` for an exact report path and `--output-root` for the
preferred derived location:

```text
<output-root>/_derived/session_archives/<archive_id>/
```

`restore` delegates to `build_session_archive_restore_plan(...)` for `--dry-run`
and `restore_session_archive_pack(...)` for extraction. Use `--dry-run` before
actual restore, especially when an archive pack was copied through mounted
storage. Supported overwrite policies are `fail_if_exists`, `skip_existing`,
and `replace_existing`.

All commands support local filesystem paths only. Mounted cloud-drive folders
are treated as ordinary local paths; the CLI does not call Google Drive APIs,
require credentials, access the network, or read live market data. Archive packs
and CLI reports remain derived, disposable, transport-only outputs, not
canonical storage, canonical evidence, or a registry.

## Notebook And Colab Workflow Overview

The intended notebook workflow is:

1. Do active StratLake work in a normal local repository workspace.
2. Pack selected repository-relative `features`, `artifacts`, `configs`, and
   optional `duckdb_snapshot` files into a derived archive pack.
3. Move or persist the archive pack through ordinary local or mounted storage.
4. Validate and inspect the copied pack before restore.
5. Restore the pack into a clean or intentionally selected local target root.
6. Run strategy, alpha, portfolio, feature, and research workflows from the
   restored repository-relative paths.

Archive shards are transport containers. Do not point active StratLake workflows
at tar shard internals, and do not treat archive packs as the active feature,
artifact, config, registry, or evidence store. Restored files return to the
normal StratLake layout before workflows run.

## What Session Archives Are

Portable session archives are useful when notebook environments are short-lived,
slow to copy many small files, or attached to mounted storage. A pack can carry
selected session context such as curated feature snapshots, research artifacts,
checked-in or local config files, and optional file-backed DuckDB snapshots.

The pack is deterministic and inspectable. Its manifest and sidecars describe
what was packed, which shards contain the files, which checksums should match,
which logical groups are included, and what restore paths are expected.

## What Session Archives Are Not

Portable session archives are not:

* canonical feature storage
* canonical research evidence
* a strategy, alpha, portfolio, governance, or artifact registry
* a hidden execution cache
* a way to run workflows directly from archive shards
* a replacement for standard repository-relative feature, artifact, and config
  paths

Canonical Parquet features, strategy outputs, alpha outputs, portfolio outputs,
governance outputs, evidence artifacts, runtime configs, and direct
repository-relative scans remain authoritative.

## Storage Boundary Model

Use a simple boundary model:

* **Local runtime workspace:** where active StratLake workflows run.
* **Derived archive output:** where `pack` writes disposable archive packs.
* **Mounted storage:** optional local filesystem storage used to copy or persist
  archive packs.
* **Restore target workspace:** where archive contents are restored back into
  normal StratLake paths.

Mounted storage such as Google Drive is optional. StratLake treats mounted
cloud folders as ordinary local paths and does not use Google Drive APIs,
credentials, network access, live market data, or external services for archive
pack, validate, inspect, or restore operations.

## Recommended Colab Or Mounted-Drive Pattern

In a notebook or Colab-style workflow, keep active work local to the runtime
filesystem when possible:

```text
workspace/
  data/curated/features_daily/
  artifacts/strategies/
  configs/
mounted_drive/
  stratlake_archives/
```

Pack from `workspace/`, copy the archive pack to
`mounted_drive/stratlake_archives/`, then later copy or reference that pack as
a local mounted path. Before restoring, validate and inspect it:

```bash
python -m src.cli.session_archive validate \
  --archive-root mounted_drive/stratlake_archives/demo-session \
  --output-root artifacts

python -m src.cli.session_archive inspect \
  --archive-root mounted_drive/stratlake_archives/demo-session
```

Restore into a normal workspace path, not into a tar shard:

```bash
python -m src.cli.session_archive restore \
  --archive-root mounted_drive/stratlake_archives/demo-session \
  --target-root restored_workspace \
  --dry-run

python -m src.cli.session_archive restore \
  --archive-root mounted_drive/stratlake_archives/demo-session \
  --target-root restored_workspace \
  --overwrite-policy fail_if_exists
```

After restore, run StratLake from `restored_workspace/` using normal local
repository-relative paths.

## First-Run Round Trip With Small Synthetic Data

This small documentation-only example uses tiny files in StratLake-style
directories. It does not require live data, credentials, network access, or
external services.

Create a small local workspace:

```bash
mkdir -p data/curated/features_daily artifacts/strategies/run_a configs/profiles
printf "feature-a\n" > data/curated/features_daily/AAPL.parquet
printf "{\"run_id\":\"run_a\"}\n" > artifacts/strategies/run_a/manifest.json
printf "schema_version: 1\nprofile: notebook\n" > configs/profiles/notebook.yml
```

Pack the selected groups:

```bash
python -m src.cli.session_archive pack \
  --repository-root . \
  --archive-id demo-session \
  --include-group features \
  --include-group artifacts \
  --include-group configs
```

Validate and inspect before restore:

```bash
python -m src.cli.session_archive validate \
  --archive-root artifacts/_derived/session_archives/demo-session \
  --output-root artifacts

python -m src.cli.session_archive inspect \
  --archive-root artifacts/_derived/session_archives/demo-session
```

Plan and then run restore into a clean target:

```bash
python -m src.cli.session_archive restore \
  --archive-root artifacts/_derived/session_archives/demo-session \
  --target-root restored_workspace \
  --dry-run

python -m src.cli.session_archive restore \
  --archive-root artifacts/_derived/session_archives/demo-session \
  --target-root restored_workspace
```

The restored files are back under normal paths such as:

```text
restored_workspace/data/curated/features_daily/AAPL.parquet
restored_workspace/artifacts/strategies/run_a/manifest.json
restored_workspace/configs/profiles/notebook.yml
```

Active workflows should now use the restored local workspace paths, not archive
shards.

## Python API Workflow

Notebook users can call the same APIs directly:

```python
from pathlib import Path

from src.session_archive import (
    SessionArchiveIncludePolicy,
    SessionArchiveLogicalGroup,
    SessionArchiveRestoreRequest,
    SessionArchiveWriteRequest,
    build_session_archive_restore_plan,
    inspect_session_archive,
    restore_session_archive_pack,
    validate_session_archive,
    write_session_archive_pack,
)

request = SessionArchiveWriteRequest(
    archive_id="demo-session",
    repository_root=Path("."),
    include_policy=SessionArchiveIncludePolicy(
        include_groups=(
            SessionArchiveLogicalGroup.FEATURES,
            SessionArchiveLogicalGroup.ARTIFACTS,
            SessionArchiveLogicalGroup.CONFIGS,
        ),
    ),
    source_runtime_profile="notebook",
    source_profile_path="configs/profiles/notebook.yml",
)

pack = write_session_archive_pack(request)
validation = validate_session_archive(pack.archive_root)
inspection = inspect_session_archive(pack.archive_root)

if validation.passed:
    restore_request = SessionArchiveRestoreRequest(
        archive_root=pack.archive_root,
        target_root=Path("restored_workspace"),
        overwrite_policy="fail_if_exists",
    )
    dry_run = build_session_archive_restore_plan(restore_request)
    result = restore_session_archive_pack(restore_request)
```

The Python APIs have the same boundary as the CLI: they create and restore local
filesystem snapshots but do not execute strategies, build features, call cloud
APIs, resolve live data, or make archives canonical.

## Runtime Profiles And Local Paths

Archive manifests can record non-secret runtime profile context when callers
supply `source_runtime_profile` or `source_profile_path`. This is descriptive
metadata only. A runtime profile does not make an archive canonical and does not
hide execution behavior.

After restore, point environment variables and runtime profiles at normal local
paths in the restored workspace. Do not point `FEATURES_ROOT`, `ARTIFACTS_ROOT`,
profile paths, strategy configs, or notebook code at tar shard internals.
Validation and inspection do not execute workflows, resolve live data, or prove
that a future strategy run is valid; they only verify and summarize the archive
pack.

## Include And Exclude Recommendations

Recommended logical groups:

* `configs`: profile and configuration context needed to rerun local workflows.
* `features`: curated feature snapshots that are expensive to recopy file by
  file.
* `artifacts`: research outputs, reports, and run artifacts useful for review.
* `duckdb_snapshot`: optional file-backed DuckDB state when it is useful and
  safe to move.

Avoid packing:

* secrets, credential files, API tokens, or private keys
* virtual environments such as `.venv/`
* `.git/`
* cache directories such as `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`,
  `.mypy_cache/`, and `.ipynb_checkpoints/`
* temporary notebook output and scratch files
* huge debug artifacts unless they are intentionally needed
* files that only make sense through machine-local absolute paths

The writer excludes common noisy local paths by default. Add extra
`--exclude-pattern` values when a notebook creates project-specific temporary
files.

## Large Feature Dataset Guidance

Archive packs are most useful when many small Parquet or artifact files are
slow to transfer through mounted storage. Shards group many files into fewer
portable tar files, reducing transfer overhead while keeping restore behavior
explicit.

Choose `--max-shard-size-bytes` and `--max-entries-per-shard` based on storage
limits, notebook runtime memory, and transfer behavior. Validate checksums after
copying packs through mounted storage. Restore locally before active workflows.
Do not read features directly from tar shards in strategy, alpha, portfolio, or
research code.

## Optional DuckDB Snapshot Guidance

DuckDB snapshot metadata is optional. `:memory:` DuckDB context is metadata
only and does not require or restore a snapshot file. File-backed snapshots may
be included when they are useful to move with the session.

Missing optional DuckDB metadata or `:memory:` warnings are not necessarily
failures. Treat error-severity validation issues as blockers; treat DuckDB
warnings as context to confirm whether the archive intentionally omitted a
file-backed snapshot. After restore, workflows should use normal local DuckDB
or config paths.

## Restore Safety And Overwrite Policies

Supported restore overwrite policies are:

* `fail_if_exists`: fail before extraction if any target file exists.
* `skip_existing`: preserve existing local files and restore non-conflicting
  files.
* `replace_existing`: intentionally replace existing target files.

Use `fail_if_exists` for first restore into a clean target. Always run
`--dry-run` before restoring into a non-empty workspace. Use `skip_existing`
only when preserving local files is intentional. Use `replace_existing` only
when replacing local targets is intentional and the dry-run plan is understood.

## Troubleshooting

Common validation or restore findings:

* `missing_manifest`: check that `--archive-root` points at the archive pack
  root, not the parent directory or a shard directory.
* `malformed_manifest_json`: recreate the archive or recopy it from mounted
  storage; avoid editing archive internals manually.
* `unsupported_schema_version`: use compatible StratLake code for the pack, or
  recreate the pack with the current repository version.
* `missing_shard_index`, `missing_checksums`, or `missing_restore_plan`:
  recreate or recopy the full archive pack; these sidecars are required.
* `missing_required_shard`: recopy the `shards/` directory or recreate the
  pack.
* `checksum_mismatch`: recopy the pack from mounted storage, then validate
  again before restore.
* `unsafe_archive_entry`: recreate the pack from safe repository-relative paths;
  archive entries must not be absolute paths, URL-like paths, home shortcuts, or
  parent traversal paths.
* `unsafe_restore_path`: check restore metadata and use a clean target root.
* `unknown_logical_group`: recreate the archive with supported logical groups.
* `non_portable_path`: remove absolute, machine-local, backslash, or traversal
  paths from the source selection and recreate the pack.
* optional DuckDB warnings: confirm whether `:memory:` or omitted file-backed
  snapshots are intentional.

When in doubt, recreate the archive from the original local workspace, copy the
entire archive pack, validate after transfer, inspect before restore, and
restore into a clean target root.

## Checklist Before Running Active Workflows

Before running strategy, alpha, portfolio, feature, or research workflows:

* The archive validated with no error-severity issues.
* The inspection summary matches the expected archive ID, groups, shard count,
  file count, and restore roots.
* A restore dry-run was reviewed for non-empty targets.
* Files were restored into normal repository-relative paths.
* Runtime profiles and environment variables point at restored local paths.
* No workflow is reading directly from archive shards.
* Archive packs and reports are treated as derived transport metadata, not
  canonical storage or canonical evidence.

## Deterministic Round-Trip Validation

M43 includes a focused CI-safe validation slice for the full archive lifecycle:

```text
pack -> validate -> inspect -> restore -> compare
```

The test uses tiny synthetic StratLake-style files under `data/curated/`,
`artifacts/`, `configs/`, and an optional file-backed DuckDB snapshot path. It
does not require Google Drive APIs, credentials, network access, live market
data, real mounted drives, production-sized datasets, or external services.

The round-trip validation proves that:

* a synthetic session tree can be archived into derived shards
* `manifest.json`, `archive_index.json`, `checksums.json`, `restore_plan.json`,
  and shard files are present
* archive validation and inspection succeed without error-severity issues
* validation and inspection reports are deterministic and avoid machine-local
  absolute paths
* restore dry-run plans the expected repository-relative files
* restore recreates normal local `data/`, `artifacts/`, and `configs/` paths
* restored file bytes match the original synthetic source files
* canonical source files remain unchanged after pack, validate, inspect, and
  restore
* archive and report boundary metadata remains derived and non-authoritative

The same validation slice also covers clear failure behavior for missing shards,
checksum mismatches, and unsafe archive member paths. These checks are meant to
prove transport integrity and restore safety. They do not run strategy, alpha,
portfolio, feature, or research workflows from archive shards, and they do not
make archive packs canonical storage or canonical evidence.

## Future M43 Use

Later M43 issues can build on this contract and writer for:

* notebook quickstart flows
* optional cloud transport integrations

CLI commands are available through `python -m src.cli.session_archive`.

Those future tools should treat the manifest as an inspectable description of
transport metadata. They should continue to reopen canonical manifests,
registries, checked-in configs, and repository-relative paths before making
decision-sensitive claims.
