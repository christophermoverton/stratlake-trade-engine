# Notebook Workspace Bootstrap

## Purpose

`stratlake-init-notebook` initializes a local notebook workspace under an explicit root.
It copies curated starter config and guidance files while keeping mutable workspace state local.

This improves notebook-first and pip-installed workflows by removing repository-root assumptions.

## Install And Run

Normal wheel/pip installs are supported. Starter templates are bundled as package resources.

Editable install from a repository checkout:

```powershell
python -m pip install -e .
```

Optional dev extras:

```powershell
python -m pip install -e ".[dev]"
```

Bootstrap a workspace in the current directory:

```powershell
stratlake-init-notebook
```

Bootstrap a workspace at an explicit root:

```powershell
stratlake-init-notebook --root ./stratlake-notebooks
```

Bootstrap a session-first workspace and write `.stratlake/` metadata:

```powershell
stratlake-init-session --root ./stratlake-notebooks --project-name stratlake-demo
```

For Colab and mounted-Drive notebooks, use the session-first flow in
[`docs/colab_project_sessions.md`](colab_project_sessions.md).

Overwrite only copied starter templates:

```powershell
stratlake-init-notebook --root ./stratlake-notebooks --force
```

Build features from an installed package with an explicit run-local
MarketLake root:

```powershell
stratlake-build-features --timeframe 1D --start 2025-01-01 --end 2025-02-01 --tickers configs/tickers_50.txt --marketlake-root data/curated
```

The module command remains valid:

```powershell
python -m cli.build_features --timeframe 1D --start 2025-01-01 --end 2025-02-01 --tickers configs/tickers_50.txt
```

Feature builds resolve the curated root as `--marketlake-root` >
`MARKETLAKE_ROOT` > `configs/paths.yml`. The override is scoped to the
feature-build run and does not mutate `.env`, `configs/paths.yml`, canonical
artifacts, or `os.environ`. The feature-run summary records the effective root
and source under `config_resolution`.

## What Gets Created

Directory layout under the selected root:

- `notebooks/`
- `configs/`
- `docs/`
- `contracts/`
- `artifacts/`

Curated starter files are copied from bundled package resources into the local workspace.
Existing files are preserved by default.
The bundled package-resource templates are only the source for the initial copy. After copying,
the local workspace files are user-owned and mutable.

Use `stratlake-init-notebook` when you only want the workspace directories and
starter templates. Use `stratlake-init-session` when a notebook or Colab-style
environment also needs explicit session metadata that records the selected
project root and path provenance.

The starter config set includes `configs/session.yml`, a user-owned template
for notebook project-session metadata. The bootstrap command only copies this
template; it does not create session metadata or mutate canonical artifacts.

## Notebook Project Session Contract

M42.1 introduces a reusable session contract under `src.session` for notebooks
and Colab-style workflows that need explicit roots. A project session records:

- `schema_version`
- `project_name`
- notebook current working directory
- StratLake project root
- config, artifact, and feature roots
- optional external MarketLake root
- optional Drive persistence root
- path kind and source/provenance for each resolved path

When written, session metadata lives under the selected project root:

- `.stratlake/session.json`
- `.stratlake/path_resolution.json`

These files are diagnostic session metadata, not canonical artifact state. They
do not replace manifests, summaries, metrics, inventories, or named workflow
outputs.

Project-internal paths are serialized as portable POSIX-style relative paths
where practical, such as `configs` or `data/curated`. External absolute paths,
such as mounted Drive paths or external MarketLake roots, remain absolute and
are marked as external so they are not mistaken for project-owned files.

Minimal library usage:

```python
from src.session import create_notebook_project_session, write_session_files

session = create_notebook_project_session(
    project_name="stratlake-demo",
    project_root="./stratlake-notebooks",
    marketlake_root="../fintech/data/curated",
    drive_root="/content/drive/MyDrive/stratlake-demo",
)
write_session_files(session)
```

The installed `stratlake-init-session` command is a thin wrapper around the
same behavior:

```powershell
stratlake-init-session `
  --root ./stratlake-workspace `
  --project-name stratlake-demo `
  --marketlake-root ./fintech/data/curated
```

Optional Drive persistence metadata can be recorded for explicit import/export
workflows:

```powershell
stratlake-init-session `
  --root /content/stratlake `
  --project-name stratlake-colab `
  --marketlake-root /content/fintech/data/curated `
  --drive-root /content/drive/MyDrive/stratlake-colab `
  --enable-drive-persistence
```

`--enable-drive-persistence` records intent only. It does not sync, import,
export, copy, or back up Drive files. Use `stratlake-session-export` and
`stratlake-session-import` for explicit one-shot filesystem snapshots.

## Session-Aware Path Helpers

Notebook and CLI code can find and consume the session metadata without relying
on the active notebook CWD:

```python
from src.session import find_session_root, load_session, resolve_session_paths

root = find_session_root()
session = load_session(root)
paths = resolve_session_paths(session)

configs_root = paths["configs_root"].resolved_path
artifacts_root = paths["artifacts_root"].resolved_path
marketlake_root = paths["marketlake_root"].resolved_path
```

`find_session_root(start)` walks upward from `start` or `Path.cwd()` until it
finds `.stratlake/session.json`. `load_session(root)` accepts either a project
root or a path inside a project and validates the session schema before
returning a structured session object.

`resolve_session_paths(...)` uses deterministic precedence:

1. explicit `overrides={...}`
2. session metadata
3. environment-variable fallbacks such as `MARKETLAKE_ROOT`, `ARTIFACTS_ROOT`,
   and `FEATURES_ROOT`
4. starter defaults

Environment variables are only fallbacks and are recorded with
`environment_variable` provenance when used. The helpers do not mutate CWD,
`.env`, `os.environ`, package resources, Drive files, or canonical artifacts.

Override example:

```python
paths = resolve_session_paths(
    session,
    overrides={
        "marketlake_root": "/content/fintech/data/curated",
        "artifacts_root": "artifacts/session-demo",
    },
)
```

Use `write_path_resolution_report(session, overrides=...)` to refresh
`.stratlake/path_resolution.json` with the same deterministic path provenance.

## Filesystem Drive Persistence

M42.4 adds explicit filesystem import/export helpers for mounted Drive-style
paths. The mounted Drive path is treated as a normal local filesystem path:
there are no Google API calls, OAuth flows, credentials, network access,
background watchers, or automatic sync.

Exports and imports are persistence snapshots only. They are not canonical
artifact state, not a remote registry, and not a second source of truth.

Export selected configs and artifacts:

```powershell
stratlake-session-export `
  --root ./stratlake-workspace `
  --drive-root ./mounted-drive/stratlake-demo `
  --include-configs `
  --include-artifacts
```

Dry-run without copying files:

```powershell
stratlake-session-export `
  --root ./stratlake-workspace `
  --drive-root ./mounted-drive/stratlake-demo `
  --include-configs `
  --include-artifacts `
  --dry-run
```

Import selected files without overwriting existing files:

```powershell
stratlake-session-import `
  --root ./stratlake-workspace `
  --drive-root ./mounted-drive/stratlake-demo `
  --include-configs
```

Use `--force` to explicitly overwrite existing import destinations. Feature
data is copied only with `--include-features`; MarketLake data is copied only
with `--include-market-data`.

Session metadata is included by default. Other categories are opt-in:
`--include-configs`, `--include-contracts`, `--include-docs`,
`--include-artifacts`, `--include-derived-artifacts`, `--include-features`,
and `--include-market-data`.

Default excludes prevent copying `.env`, obvious credentials/secrets/API-key
files, notebook checkpoints, Python bytecode, caches, and temporary files.

Each non-dry-run operation writes a non-authoritative manifest under:

```text
artifacts/_derived/notebook_sessions/<operation>_<operation_id>/drive_sync_manifest.json
```

The manifest records the operation, roots, included categories, exclude rules,
source and destination paths, relative path, category, size, SHA-256 hash,
status, and skip reason when a destination is preserved.

Use `--operation-id` when you want distinct historical manifests instead of
reusing `latest`.

For notebook-friendly portable session archive packs, use the explicit archive
bootstrap command:

```powershell
stratlake-session-archive-bootstrap `
  --root /content/stratlake `
  --archive-id notebook-session-001 `
  --include-features `
  --include-artifacts `
  --include-configs
```

To persist the archive pack to a mounted Drive-style local filesystem path and
validate/inspect the copied pack:

```powershell
stratlake-session-archive-bootstrap `
  --root /content/stratlake `
  --archive-id notebook-session-001 `
  --drive-root /content/drive/MyDrive/stratlake-colab/session_archives `
  --include-features `
  --include-artifacts `
  --include-configs `
  --validate-after-copy `
  --inspect-after-copy
```

The mounted Drive path is treated as local filesystem storage only. The command
does not call Google APIs, does not require credentials, and does not make
archive packs canonical storage.

Release-facing validation for the full notebook-session stack is documented in
[`docs/m42_release_validation_checklist.md`](m42_release_validation_checklist.md).

## Installed Commands

The package now provides stable installed entry points for common workflows:

- `stratlake-init-notebook`
- `stratlake-init-session`
- `stratlake-session-export`
- `stratlake-session-import`
- `stratlake-session-archive-bootstrap`
- `stratlake-build-features`
- `stratlake-run-strategy`
- `stratlake-run-alpha`
- `stratlake-run-alpha-evaluation`
- `stratlake-run-portfolio`
- `stratlake-run-pipeline`
- `stratlake-run-research-campaign`
- `stratlake-run-benchmark-pack`
- `stratlake-run-candidate-selection`
- `stratlake-review-candidate-selection`
- `stratlake-compare-strategies`
- `stratlake-compare-alpha`
- `stratlake-validate-config`
- `stratlake-doctor`
- `stratlake-explain-config`
- `stratlake-catalog-index`
- `stratlake-query-catalog`
- `stratlake-explore-catalog-evidence`
- `stratlake-export-catalog-lineage`
- `stratlake-build-evidence-review`
- `stratlake-run-promotion-governance-report`

Python module invocations remain compatible, for example:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
python -m cli.build_features --timeframe 1D --start 2025-01-01 --end 2025-02-01 --tickers configs/tickers_50.txt
```

## Package Versus Workspace Boundaries

Package responsibilities:

- reusable command surfaces
- reusable library code
- deterministic execution and validation logic

Workspace responsibilities:

- mutable `configs/` copies
- mutable local `docs/` copies
- local `contracts/`
- local `notebooks/`
- generated local `artifacts/`

The bootstrap command does not create fake run outputs and does not mutate package files.
It never writes into site-packages.

## Troubleshooting

`stratlake-init-notebook` reports missing starter templates:

- Reinstall the package so bundled notebook workspace resources are present.
- Validate installation with `python -m pip show stratlake-trade-engine` and reinstall from wheel if needed.

Files were skipped unexpectedly:

- This is expected when destination files already exist.
- Re-run with `--force` to overwrite copied starter templates only.

Paths looked wrong:

- Always pass an explicit `--root` for notebooks and tutorials.
- The command refuses to write outside the selected workspace root.
