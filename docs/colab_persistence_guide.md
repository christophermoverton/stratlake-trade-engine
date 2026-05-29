# Unified Colab Persistence Guide

## Purpose

Use this guide to choose the right persistence pattern across fintech-market-ingestion
and StratLake Trade Engine when running Colab sessions.

The key rule is simple:

- run active workflows from local runtime roots under `/content/...`
- persist intentionally to mounted Drive roots
- restore locally first before new feature/research runs

This guide complements:

- [Colab project sessions](colab_project_sessions.md)
- [Notebook integration](notebook_integration.md)
- [Notebook workspace bootstrap](notebook_workspace_bootstrap.md)
- [Portable session archives](session_archives.md)

## Persistence Categories

### Local runtime state

Use local runtime workspaces for:

- active feature builds
- active strategy runs
- notebook-native `src.execution` API calls
- artifact-producing research workflows
- reading and writing canonical StratLake runtime artifacts

Typical roots:

- `/content/stratlake-workspace`
- `/content/fintech-market-ingestion-demo`

Why:

- fastest I/O in Colab sessions
- deterministic workflow behavior
- canonical runtime artifact contracts are produced here first

Constraint:

- Colab runtime disks are ephemeral, so persist intentionally before shutdown

### Lightweight session persistence (export/import)

Use lightweight session persistence for:

- configs and contracts
- small reports and selected artifacts
- continuity snapshots
- notebook session metadata
- small feature/report subsets when explicitly selected

StratLake commands:

- `stratlake-session-export`
- `stratlake-session-import`

Characteristics:

- explicit one-shot command execution only
- no background sync
- no restore-on-import semantics
- useful for small-to-medium continuity payloads
- persistence metadata remains non-canonical

### Archive and backup packs

Use archive or backup packs for:

- large curated OHLCV datasets
- feature data bundles
- larger artifact bundles
- fewer files for faster Drive transport
- restore-first workflows after clean runtime restarts

StratLake archive commands:

- `stratlake-session-archive-bootstrap`
- `stratlake-session-archive-restore-bootstrap`

Characteristics:

- archive packs are derived and non-canonical transport artifacts
- validate/inspect before restore
- restore into local runtime roots before active research
- choose collision and overwrite policies intentionally

### Mounted Drive paths

Use mounted Drive roots for:

- persistence and transport storage
- long-lived session archives
- fintech backup-pack storage
- StratLake archive-pack storage

Characteristics:

- mounted filesystem paths only (for example, `/content/drive/MyDrive/...`)
- no Google API/OAuth behavior is implied by StratLake commands
- not canonical working data
- do not point active MarketLake reads at archive-pack directories

## Decision Table

| Need | Prefer | Why |
| --- | --- | --- |
| Continue a small notebook session | `stratlake-session-export` and `stratlake-session-import` | Lightweight continuity for selected files |
| Move large curated fintech data | Fintech archive or backup pack flow | Fewer files and faster Drive transport |
| Restart a StratLake Colab session | `stratlake-session-archive-restore-bootstrap` | Restore local runtime state first |
| Run feature builds or strategy research | Local runtime workspace under `/content/...` | Fast I/O and canonical runtime artifacts |
| Inspect readiness before execution | `stratlake-notebook-doctor` and `stratlake-validate-marketlake-handoff` | Read-only diagnostics before active runs |

## StratLake Command References

Use these StratLake commands explicitly in Colab:

- `stratlake-init-session`
- `stratlake-session-export`
- `stratlake-session-import`
- `stratlake-session-archive-bootstrap`
- `stratlake-session-archive-restore-bootstrap`
- `stratlake-notebook-doctor`
- `stratlake-validate-marketlake-handoff`

Boundary reminder:

- setup, persistence, restore, doctor, and handoff validation stay CLI-first
- interactive research after readiness checks can use `src.execution` APIs

## Fintech Companion References

fintech-market-ingestion is the companion platform that produces curated data
consumed by StratLake.

Fintech responsibilities in this workflow:

- build and persist curated MarketLake-style datasets
- optionally package large curated data with archive or backup packs
- restore curated data locally before StratLake consumes it
- produce diagnostic handoff metadata where available

Companion command examples (fintech-side, naming may vary by fintech repo version):

- `fintech-session-archive-bootstrap`
- `fintech-session-archive-restore-bootstrap`
- `fintech-save-session`
- `fintech-restore-session`
- `fintech-notebook-doctor`

Treat those as companion-platform examples unless your fintech repository docs
confirm exact command names.

## Fresh Runtime Restore Sequence

For a clean Colab runtime restart:

1. Mount Drive in Colab at `/content/drive`.
2. Define unified roots and session profile variables:
   `FINTECH_ROOT`, `STRATLAKE_ROOT`, `MARKETLAKE_ROOT`, `DRIVE_ROOT`,
   `SESSION_ARCHIVES_ROOT`, `ARCHIVE_ID`, and `ARCHIVE_ROOT`.
3. Restore fintech curated data locally from a fintech archive or backup pack when needed.
4. Restore StratLake archive into local runtime state with dry-run first:
   `stratlake-session-archive-restore-bootstrap --dry-run --json`.
5. Validate and inspect before restore writes with
   `--validate-before-restore --inspect-before-restore`.
6. Use `--overwrite-policy fail_if_exists` for clean runtime safety.
7. Run `stratlake-notebook-doctor`.
8. Run `stratlake-validate-marketlake-handoff`.
9. Run active feature and research workflows locally under `/content/...`.

## End-of-Session Persistence Sequence

Before runtime shutdown:

1. Finish active local research in `/content/...` workspaces.
2. Persist refreshed fintech curated data with fintech archive or backup packs when relevant.
3. Persist StratLake session state with `stratlake-session-archive-bootstrap`.
4. Use collision and overwrite policies intentionally:
   `fail_if_exists` for safety, `skip_existing` to avoid replacement,
   `overwrite_allowed` only for intentional refresh.
5. Optionally use `stratlake-session-export` for smaller continuity snapshots.
6. Confirm archive or backup outputs exist under mounted Drive roots.
7. Assume runtime-local state is disposable after shutdown.

Safety notes:

- avoid mixing old and new archive files in the same destination pack
- avoid nesting archive output roots inside active target workspaces
- do not treat archive packs as canonical runtime state

## Canonical and Derived Boundaries

Keep these boundaries explicit:

- local runtime artifacts are canonical for active StratLake research runs
- Drive copies and archive packs are derived transport and persistence artifacts
- fintech handoff reports are diagnostic metadata only
- session metadata is notebook/session state, not a new research source of truth
- StratLake should read restored local curated roots, not archive-pack directories directly

Also explicit:

- no hidden sync
- no Google API calls or OAuth behavior in these CLI flows
- no automatic persistence or restore-on-import
