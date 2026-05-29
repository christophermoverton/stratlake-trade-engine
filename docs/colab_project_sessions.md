# Colab Project Sessions

## Purpose

Use a StratLake project session in Colab or another cloud notebook when the
notebook current working directory, StratLake project root, MarketLake root,
and mounted Drive persistence root may all be different paths.

The session makes those roots explicit without changing StratLake's
artifact-first architecture. Session files and Drive copies are diagnostic
notebook/session state. Canonical research evidence remains in StratLake
artifact outputs such as manifests, metrics, summaries, inventories, and named
workflow outputs.

## Mental Model

Keep these roots separate:

- notebook CWD: where the notebook process happens to start
- StratLake project root: the selected workspace for `.stratlake/`, `configs/`,
  `artifacts/`, `docs/`, `contracts/`, and `notebooks/`
- configs root: usually `PROJECT_ROOT / "configs"`
- artifacts root: usually `PROJECT_ROOT / "artifacts"`
- features root: usually `PROJECT_ROOT / "data" / "curated"` unless configured
- external MarketLake root: a mounted or copied curated-data root outside the
  StratLake project
- mounted Drive persistence root: an optional local filesystem path used for
  explicit backup/import/export snapshots

Session-aware notebooks should resolve paths from the project session, not from
the accidental notebook CWD.

## Example Root Layout

This Colab layout is an example, not a hard requirement:

```text
/content/stratlake
  .stratlake/
  configs/
  artifacts/
  docs/
  notebooks/

/content/fintech/data/curated
  features_daily/
  features_1m/
  ...

/content/drive/MyDrive/stratlake-demo
  .stratlake/
  configs/
  artifacts/
  data/
```

## Session-First Notebook Setup

Install StratLake in the notebook environment. Use the package source that
matches the work you are doing:

```bash
!python -m pip install stratlake-trade-engine
```

If you are running from a checked-out repository instead:

```bash
!python -m pip install -e .
```

Optionally mount Google Drive in Colab:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Define a unified Colab session profile once, then reuse it in later cells:

```python
from pathlib import Path

FINTECH_ROOT = Path("/content/fintech-market-ingestion-demo").resolve()
STRATLAKE_ROOT = Path("/content/stratlake-workspace").resolve()
MARKETLAKE_ROOT = FINTECH_ROOT / "data" / "curated"
DRIVE_ROOT = Path("/content/drive/MyDrive/stratlake-fintech-colab").resolve()

START = "2024-10-01"
END = "2025-04-15"
UNIVERSE_CONFIG = STRATLAKE_ROOT / "configs" / "universe.yml"
PATHS_CONFIG = STRATLAKE_ROOT / "configs" / "paths.yml"

SESSION_ARCHIVES_ROOT = DRIVE_ROOT / "session_archives"
ARCHIVE_ID = "notebook-session-001"
ARCHIVE_ROOT = SESSION_ARCHIVES_ROOT / ARCHIVE_ID
ARTIFACTS_ROOT = STRATLAKE_ROOT / "artifacts"
FEATURES_ROOT = STRATLAKE_ROOT / "data" / "curated"
TICKERS_SAMPLE = STRATLAKE_ROOT / "configs" / "tickers_sample.txt"
```

Root meanings:

- `FINTECH_ROOT`: local Colab runtime checkout/workspace for fintech ingestion.
- `STRATLAKE_ROOT`: local Colab runtime workspace for StratLake files and runs.
- `MARKETLAKE_ROOT`: local curated-data root consumed by StratLake.
- `DRIVE_ROOT`: mounted Drive persistence/archive location, not canonical working data.
- `UNIVERSE_CONFIG`: generated or user-owned StratLake universe config.
- `PATHS_CONFIG`: generated or user-owned StratLake path config.

The profile is an explicit notebook convenience layer only. It does not mutate
`.env`, `os.environ`, Drive files, or canonical artifacts.

Initialize the project session:

```bash
!stratlake-init-session \
  --root "{STRATLAKE_ROOT}" \
  --project-name stratlake-demo \
  --marketlake-root "{MARKETLAKE_ROOT}" \
  --drive-root "{DRIVE_ROOT}" \
  --notebook-configs
```

Use `stratlake-init-session` when you want workspace starter files plus
`.stratlake/session.json` and `.stratlake/path_resolution.json`. Use
`stratlake-init-notebook` only when you want the workspace layout and starter
templates without session metadata.

`--notebook-configs` writes a deterministic notebook-ready config bundle under
`configs/`:

- `configs/paths.yml`
- `configs/universe.yml`
- `configs/tickers_sample.txt`

These starter configs are user-owned. Existing files are preserved by default.
Use `--force-notebook-configs` to overwrite only this bundle (without touching
unrelated files). `--force` remains the explicit refresh path for session
metadata and known notebook starter templates.

`configs/paths.yml` keeps project-owned paths relative where practical and keeps
external roots explicit when they are outside the project root, such as mounted
Drive paths or restored local MarketLake curated roots.

This pairing (`--notebook-configs` + profile values) ensures generated
`configs/paths.yml` and `configs/universe.yml` line up with the notebook's
explicit runtime roots.

## Validate Restored Curated Data Before Feature Builds

After restoring or refreshing curated fintech data, validate the handoff before
building features:

```bash
!stratlake-validate-marketlake-handoff \
  --root "{STRATLAKE_ROOT}" \
  --marketlake-root "{MARKETLAKE_ROOT}" \
  --universe "{UNIVERSE_CONFIG}" \
  --start "{START}" \
  --end "{END}" \
  --timeframe 1D \
  --json
```

This check is read-only. It verifies the curated root, notebook-session paths,
requested symbols, and the requested date window before any feature build
runs.

## Restore-First Colab Workflow (Fresh Runtime)

Use restore-first when a fresh Colab runtime starts and a previous session
archive already exists under mounted Drive storage. Restore local runtime state
into `STRATLAKE_ROOT` first, then continue research from the restored local
workspace.

Drive folders and archive packs are derived persistence and transport artifacts,
not canonical StratLake state.

Dry-run with validation and inspection before any restore writes:

```bash
!stratlake-session-archive-restore-bootstrap \
  --archive-root "{ARCHIVE_ROOT}" \
  --target-root "{STRATLAKE_ROOT}" \
  --validate-before-restore \
  --inspect-before-restore \
  --dry-run \
  --json
```

Dry-run validates archive structure and planned restore behavior, but does not
write restored files.

Run the restore intentionally after reviewing dry-run output:

```bash
!stratlake-session-archive-restore-bootstrap \
  --archive-root "{ARCHIVE_ROOT}" \
  --target-root "{STRATLAKE_ROOT}" \
  --validate-before-restore \
  --inspect-before-restore \
  --overwrite-policy fail_if_exists
```

`fail_if_exists` is the safest default for clean runtimes and accidental rerun
protection. Use `skip_existing` or `overwrite_allowed` only when that behavior
is intentionally required.

Post-restore local checklist:

```python
for path in [
    STRATLAKE_ROOT / ".stratlake" / "session.json",
    STRATLAKE_ROOT / ".stratlake" / "path_resolution.json",
    PATHS_CONFIG,
    UNIVERSE_CONFIG,
    STRATLAKE_ROOT / "configs" / "tickers_sample.txt",
    STRATLAKE_ROOT / "artifacts",
    STRATLAKE_ROOT / "data" / "curated",
]:
    print(path, "OK" if path.exists() else "MISSING")
```

After restore, run handoff validation before feature builds:

```bash
!stratlake-validate-marketlake-handoff \
  --root "{STRATLAKE_ROOT}" \
  --marketlake-root "{MARKETLAKE_ROOT}" \
  --universe "{UNIVERSE_CONFIG}" \
  --start "{START}" \
  --end "{END}" \
  --timeframe 1D \
  --json
```

Then continue with feature builds, strategy or research checks, and local
artifact inspection from `/content/stratlake-workspace`.

## Inspect Session Paths

Load the session and resolve the important roots before running workflow cells:

```python
from src.session import load_session, resolve_session_paths

session = load_session(STRATLAKE_ROOT)
paths = resolve_session_paths(session)

configs_root = paths["configs_root"].resolved_path
artifacts_root = paths["artifacts_root"].resolved_path
features_root = paths["features_root"].resolved_path
marketlake_root = paths["marketlake_root"].resolved_path
drive_root = paths["drive_root"].resolved_path
```

If you are using the profile cell above, load the session from `STRATLAKE_ROOT`:

```python
session = load_session(STRATLAKE_ROOT)
paths = resolve_session_paths(session)
```

Each resolved path includes the serialized path, resolved absolute path, path
kind, source/provenance, input value when relevant, and base path when relevant.
The helpers do not mutate CWD, `.env`, `os.environ`, Drive files, or canonical
artifacts.

Resolution precedence is deterministic:

1. explicit API/CLI overrides
2. session metadata
3. recorded environment-variable fallbacks
4. starter defaults

## Build Features With Explicit MarketLake Root

Create or upload a ticker file under the project root:

```python
(STRATLAKE_ROOT / "configs" / "tickers_demo.txt").write_text("AAPL\nMSFT\n", encoding="utf-8")
```

Run the feature builder with an explicit MarketLake root so the notebook CWD is
irrelevant:

```bash
!stratlake-build-features \
  --timeframe 1D \
  --start "{START}" \
  --end "{END}" \
  --tickers "{TICKERS_SAMPLE}" \
  --marketlake-root "{MARKETLAKE_ROOT}"
```

The feature-run summary records the effective MarketLake root and its source
under `config_resolution`.

Keep `marketlake_root` pointed to a local curated dataset root (for example,
restored fintech curated data under `/content/.../data/curated`). Do not treat
Drive archive-pack directories as canonical MarketLake roots.

## Run Workflows From Session Configs

Use explicit config paths from the project root. For example:

```bash
!stratlake-run-strategy \
  --strategies-config "{STRATLAKE_ROOT / 'configs' / 'strategies.yml'}" \
  --strategy momentum_v1 \
  --start "{START}" \
  --end "{END}" \
  --evaluation "{STRATLAKE_ROOT / 'configs' / 'evaluation.yml'}"
```

Use notebooks to run established StratLake APIs or CLI-equivalent commands and
inspect canonical outputs. Do not move strategy logic, validation decisions, or
artifact schemas into notebook cells.

## Export Snapshots To Mounted Drive

Drive persistence is optional. The Drive root is treated as a mounted
filesystem path. StratLake's persistence adapter uses no Google API, OAuth,
credentials, or network access.

Drive copies are explicit backup/import/export snapshots only. They are not
canonical artifact state, a remote registry, or a second source of truth.

Dry-run first:

```bash
!stratlake-session-export \
  --root "{STRATLAKE_ROOT}" \
  --drive-root "{DRIVE_ROOT}" \
  --include-configs \
  --include-artifacts \
  --dry-run
```

Export configs, artifacts, and feature data:

```bash
!stratlake-session-export \
  --root "{STRATLAKE_ROOT}" \
  --drive-root "{DRIVE_ROOT}" \
  --include-configs \
  --include-artifacts \
  --include-features
```

Feature data requires `--include-features`. Market data requires
`--include-market-data`. Neither is included by broad config, docs, or artifact
flags.

Use `--operation-id` when you want distinct historical manifests instead of
reusing `latest`:

```bash
!stratlake-session-export \
  --root "{STRATLAKE_ROOT}" \
  --drive-root "{DRIVE_ROOT}" \
  --include-configs \
  --include-artifacts \
  --operation-id colab-run-001
```

Non-dry-run operations write a diagnostic, non-authoritative manifest under:

```text
artifacts/_derived/notebook_sessions/<operation>_<operation_id>/drive_sync_manifest.json
```

For portable M43 archive packs, bootstrap a session archive directly from the
workspace:

```bash
!stratlake-session-archive-bootstrap \
  --root "{STRATLAKE_ROOT}" \
  --archive-id notebook-session-001 \
  --include-features \
  --include-artifacts \
  --include-configs
```

To persist the archive pack to mounted Drive-style storage and immediately
validate/inspect the copied pack:

```bash
!stratlake-session-archive-bootstrap \
  --root "{STRATLAKE_ROOT}" \
  --archive-id notebook-session-001 \
  --archive-collision-policy overwrite_allowed \
  --drive-root "{SESSION_ARCHIVES_ROOT}" \
  --copy-policy overwrite_allowed \
  --include-features \
  --include-artifacts \
  --include-configs \
  --validate-after-copy \
  --inspect-after-copy
```

The mounted Drive path is treated as local filesystem storage only. The archive
bootstrap command does not call Google APIs, does not require credentials, and
does not make archive packs canonical storage.

`--archive-collision-policy` controls local derived archive creation, while
`--copy-policy` controls copied archive behavior under `--drive-root`.
`skip_existing` skips an existing destination archive pack as a whole rather
than mixing old and new archive files.

Do not set `--drive-root` to the local archive output directory or a child of
the local archive pack.

Restore the copied archive into an explicit local workspace with the
restore-bootstrap companion:

```bash
!stratlake-session-archive-restore-bootstrap \
  --archive-root "{ARCHIVE_ROOT}" \
  --target-root "{STRATLAKE_ROOT}" \
  --validate-before-restore \
  --inspect-before-restore \
  --dry-run \
  --json

!stratlake-session-archive-restore-bootstrap \
  --archive-root "{ARCHIVE_ROOT}" \
  --target-root "{STRATLAKE_ROOT}" \
  --validate-before-restore \
  --inspect-before-restore \
  --overwrite-policy fail_if_exists
```

Restore bootstrap reuses the shared M43 validation, inspection, restore-plan,
and restore APIs. It treats Drive paths as mounted local filesystem paths only;
it does not call Google APIs, require OAuth or credentials, start background
sync, mutate `.env` or `os.environ`, execute research workflows, or make
archive packs canonical storage.

The restore-bootstrap JSON output is useful in notebooks because validation,
inspection, planning, restore, and runtime failure summaries share a stable
machine-readable shape. Active StratLake workflows should run only after files
are restored into the explicit local target workspace.

## Restore-First vs Import/Export

Use `stratlake-session-archive-bootstrap` and
`stratlake-session-archive-restore-bootstrap` for portable, restore-first
session continuity across Colab restarts.

Use `stratlake-session-export` and `stratlake-session-import` for lighter,
explicit transfer of selected configs, artifacts, and features when you are
already managing a local workspace.

Both patterns are explicit commands. Neither is hidden sync, and neither makes
Drive or archive packs canonical state.

## Safe Persistence Defaults

The filesystem adapter excludes sensitive and noisy files by default:

- `.env`
- credentials
- API keys and secrets
- notebook checkpoints
- caches
- Python bytecode
- temporary files

Do not use Drive persistence as a credential workflow. Keep secrets outside
session snapshots.

## Command Guide

- `stratlake-init-notebook`: workspace layout and starter templates only
- `stratlake-init-session`: workspace layout plus `.stratlake/` session metadata
- `stratlake-session-export`: explicit one-shot export to a mounted Drive path
- `stratlake-session-import`: explicit one-shot import from a mounted Drive path
- `stratlake-session-archive-bootstrap`: create an M43 archive pack and optionally
  copy it to a mounted Drive-style path
- `stratlake-session-archive-restore-bootstrap`: validate, inspect, plan, and
  restore an M43 archive pack into an explicit local target workspace

All commands should be given explicit roots in Colab-style notebooks. The
commands do not change notebook CWD, mutate `.env`, mutate `os.environ`, call
Google APIs, or start background sync.
