# Runtime Profiles

## Overview

A StratLake runtime profile is a small, non-secret YAML contract that names the
intended runtime context for a workflow. Profiles make common entrypoints easier
to inspect and reproduce across local development, clean CI, notebooks, and
pipeline workflows.

Profiles are not a second source of truth. Canonical artifacts, checked-in
workflow configs, and the resolved runtime contracts remain authoritative.
Derived outputs stay disposable and non-authoritative.

Starter profiles live under [../configs/profiles](../configs/profiles).

## Supported Profiles

| Profile | Intended use |
| --- | --- |
| `local` | Local CLI development and repeatable research runs with repository-relative writable roots. |
| `ci` | Clean test and validation runs that do not depend on credentials, network access, live market data, or machine-local paths. |
| `notebook` | Interactive notebook inspection through importable execution APIs with disposable outputs. |
| `pipeline` | Orchestrated pipeline, benchmark-pack, and campaign entrypoints. |

## Precedence

M39 uses this intended precedence model:

```text
defaults < profile config < environment variables < CLI flags
```

The profile layer supplies explicit, inspectable defaults for a context. It does
not silently override environment variables or CLI flags. Environment variables
remain the existing bridge for `Settings.load()`, and CLI flags remain the final
per-run override layer.

## Profile Shape

Required top-level fields:

* `schema_version`: must be `1`
* `profile`: one of `local`, `ci`, `notebook`, or `pipeline`

Optional top-level fields:

* `description`
* `use_case`
* `settings`
* `workflow_configs`
* `runtime`
* `review`
* `boundaries`

Unknown top-level keys are invalid.

## Settings

`settings` may contain only non-secret values that correspond to existing
environment-driven settings:

* `marketlake_root`
* `features_root`
* `artifacts_root`
* `duckdb_path`
* `log_level`
* `default_timezone`

Path settings must be repository-relative POSIX-style paths such as
`data/curated`, `data`, or `artifacts/ci`. `duckdb_path` may be `:memory:` or a
repository-relative path. Profiles must not contain credentials, tokens,
machine-local absolute paths, home-directory shortcuts, or file URIs.

`.env.example` remains the contributor-facing list of environment variables.
Profiles describe a runtime context; `.env` and real environment variables still
provide machine-specific local values.

## Workflow Configs

`workflow_configs` names checked-in configuration files that a profile expects a
workflow to use. Supported keys are:

* `config_dir`
* `execution_config`
* `sanity_config`
* `review_config`
* `universe_config`
* `features_config`
* `strategies_config`
* `portfolios_config`
* `evaluation_config`
* `pipeline_config`
* `benchmark_pack_config`
* `research_campaign_config`

These paths are references to canonical checked-in config files or portable
repository locations. A profile must not generate or mutate those files.

## Runtime And Review Sections

`runtime` uses the same runtime section contract documented in
[runtime_configuration.md](runtime_configuration.md), including `execution`,
`sanity`, `portfolio_validation`, `risk`, and `strict_mode`.

`review` uses the review contract documented in
[review_configuration.md](review_configuration.md), including `filters`,
`ranking`, `output`, and `promotion_gates`.

The profile validator checks these sections for known keys and valid basic
values, but it does not run workflows, touch data, load credentials, or write
artifacts.

## Boundaries

Profiles must keep these artifact-first boundaries explicit:

* `direct_scan: true`
* `derived_outputs_authoritative: false`
* `mutates_canonical_artifacts: false`
* `requires_network: false`
* `requires_credentials: false`
* `requires_live_market_data: false`

Direct scan remains available, default, and canonical. Derived indexes,
summaries, profile outputs, and other generated files remain disposable views of
canonical artifacts.

## Defaults Versus Overrides

Omitted profile fields mean "use the existing repository default or later
precedence layer." Present profile fields are explicit profile overrides for the
named runtime context. This distinction is intentional: a profile should make
the chosen context visible without hiding execution behavior behind magic
defaults.

The starter examples use explicit `settings`, `workflow_configs`, `runtime`,
`review`, and `boundaries` values so contributors can see the contract without
reading implementation code.

## Relationship To Existing Configuration

`Settings.load()` continues to read `.env`, real environment variables, and
existing `configs/paths.yml` values when present. Runtime profiles define the
contract for a profile layer that can be merged before environment variables in
future M39 work.

Existing runtime modules remain the execution source of truth:

* `src/config/runtime.py` resolves execution, sanity, validation, risk, and
  strict mode for workflows.
* `src/config/execution.py` owns execution-assumption validation.
* `src/config/sanity.py` owns sanity-threshold validation.
* `src/config/review.py` owns review configuration validation.

Workflow config files under `configs/` remain canonical inputs for the workflows
that consume them. Profiles only point at those files or declare context-level
overrides; they do not become authoritative over artifacts or workflow configs.
