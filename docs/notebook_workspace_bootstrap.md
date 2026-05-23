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

Overwrite only copied starter templates:

```powershell
stratlake-init-notebook --root ./stratlake-notebooks --force
```

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

## Installed Commands

The package now provides stable installed entry points for common workflows:

- `stratlake-init-notebook`
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
