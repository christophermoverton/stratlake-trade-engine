# M41 Release Notes - Feature Builder Runtime Override

Milestone title: `M41 - Feature Builder CLI Ergonomics and MarketLake Root Override`

M41 branch:
`feature/m41-release-docs-version-metadata`

Candidate milestone release tag:
`v0.41.0-feature-builder-runtime-override`

Package version:
`0.41.0`

## Milestone Principle

Feature-building configuration should be explicit, inspectable, and convenient
from notebooks, shells, and installed package environments without creating a
second source of truth or mutating global configuration.

## Summary

M41 adds release-facing package metadata and documentation for the
notebook-friendly feature-builder ergonomics added in Issue #452. The feature
builder now has a packaged command, supports a run-local MarketLake root
override, logs the effective root and source, and records configuration
provenance in feature-run summaries.

The milestone preserves StratLake's artifact-first boundary. Canonical
artifacts remain the source of truth, CLI wrappers stay thin, notebook cells can
call the same CLI-equivalent path, and runtime configuration remains explicit
and deterministic.

## Scope Summary

M41 covers:

* `stratlake-build-features` project script documentation and package metadata
* `python -m cli.build_features` compatibility documentation
* `--marketlake-root` usage from shell and notebook contexts
* MarketLake root precedence documentation:
  `--marketlake-root` > `MARKETLAKE_ROOT` > `configs/paths.yml`
* local-only override behavior documentation
* `config_resolution` summary metadata documentation
* TestPyPI-ready package version metadata for `0.41.0`

M41 does not add new feature engineering logic, new runtime behavior, new
workflow logic, a new configuration framework, or changes to the TestPyPI
publication workflow.

## Usage Notes

Installed package CLI:

```bash
stratlake-build-features --timeframe 1D --start 2025-01-01 --end 2025-02-01 --tickers configs/tickers_50.txt --marketlake-root data/curated
```

Module invocation remains supported:

```bash
python -m cli.build_features --timeframe 1D --start 2025-01-01 --end 2025-02-01 --tickers configs/tickers_50.txt
```

Notebook cell with a direct run-local override:

```python
from cli.build_features import run_cli

summary_path = run_cli([
    "--timeframe", "1D",
    "--start", "2025-01-01",
    "--end", "2025-02-01",
    "--tickers", "configs/tickers_50.txt",
    "--marketlake-root", "data/curated",
])
```

Environment-driven notebook cell:

```python
import os

os.environ["MARKETLAKE_ROOT"] = "data/curated"

from cli.build_features import run_cli

summary_path = run_cli([
    "--timeframe", "1D",
    "--start", "2025-01-01",
    "--end", "2025-02-01",
    "--tickers", "configs/tickers_50.txt",
])
```

Feature-run summaries continue to include `marketlake_root` and now expose the
effective root provenance under `config_resolution`, for example:

```json
{
  "config_resolution": {
    "marketlake_root": {
      "value": "data/curated",
      "source": "cli"
    }
  }
}
```

The direct CLI override is scoped to the feature-build run. It does not edit
`.env`, write to `configs/paths.yml`, mutate canonical artifacts, or overwrite
`os.environ`.

## Documentation Updated

* [README](../README.md)
* [Getting Started](getting_started.md)
* [Notebook Integration](notebook_integration.md)
* [Notebook Workspace Bootstrap](notebook_workspace_bootstrap.md)

## Validation Commands

Focused feature-builder checks:

```bash
ruff check cli/build_features.py tests/test_build_features_cli.py pyproject.toml
pytest tests/test_build_features_cli.py
python -m cli.build_features --help
python -m pip install -e .
stratlake-build-features --help
python -m build
```

Docs/path lint, when available:

```bash
python -m src.cli.run_docs_path_lint
```

## Boundaries Preserved

* canonical artifacts remain the source of truth
* `python -m cli.build_features` remains valid
* feature-builder pipeline dispatch remains centralized
* runtime override behavior is explicit and local to the run
* `.env`, `configs/paths.yml`, canonical artifacts, and `os.environ` are not
  mutated by the override
* TestPyPI workflow logic is unchanged
* repository docs use relative paths

## Deferred Follow-Ups

* `--dry-run` or `--explain` mode for feature builds
* `--artifacts-root` runtime override
* broader project-script path-input surface audit
* more actionable missing-partition errors
* expanded configuration-resolution metadata beyond the feature-builder root

## Draft GitHub Release Notes

Title:
`M41 - Feature Builder Runtime Override`

Tag:
`v0.41.0-feature-builder-runtime-override`

Branch:
`feature/m41-release-docs-version-metadata`

Summary:
M41 updates package version metadata and release-facing documentation for the
feature-builder runtime override milestone. Users can run feature builds through
the installed `stratlake-build-features` command, keep using
`python -m cli.build_features`, and pass `--marketlake-root` for a visible
run-local curated-data root override from shells or notebooks.

Highlights:

* Added release docs for `stratlake-build-features`.
* Documented `--marketlake-root` precedence over `MARKETLAKE_ROOT` and
  `configs/paths.yml`.
* Documented local-only override behavior.
* Documented `config_resolution` feature-run summary metadata.
* Updated package version metadata to `0.41.0` for TestPyPI readiness.

Validation:

* Focused Ruff: `<record result>`
* Focused feature-builder pytest: `<record result>`
* Module help: `<record result>`
* Editable install: `<record result>`
* Console-script help: `<record result>`
* Docs/path lint: `<record result>`
* Package build: `<record result>`
* Built metadata version inspection: `<record result>`

Known boundaries:
M41 does not add `--dry-run`, `--artifacts-root`, broader project-script
path-input policy, a new configuration framework, dashboard/server behavior,
or TestPyPI workflow changes.
