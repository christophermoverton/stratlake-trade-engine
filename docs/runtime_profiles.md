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

M39.2 adds a Python resolver for this model:

```python
from src.config.resolution import resolve_runtime_profile_config

result = resolve_runtime_profile_config(
    "ci",
    environment={"ARTIFACTS_ROOT": "artifacts/ci_run"},
    cli_overrides={
        "runtime": {"execution": {"transaction_cost_bps": 5.0}},
    },
)
```

The resolver accepts either a supported profile name, an explicit profile path,
or no profile. It does not call `load_dotenv()`, execute workflows, scan
artifacts, create outputs, or mutate canonical config files. To include real
environment variables, pass an explicit environment mapping from the calling
context.

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

## Resolution Provenance

`resolve_runtime_profile_config(...)` returns a `ConfigResolutionResult` with:

* `config`: the effective resolved settings, workflow config references,
  runtime config, review config, and boundaries.
* `provenance`: one `ConfigProvenanceEntry` per resolved field.
* `profile`: the selected profile name and path, when a profile was used.
* `precedence`: the ordered source layers.

Each provenance entry records the winning value, source, and source detail:

```json
{
  "settings.artifacts_root": {
    "value": "artifacts/ci",
    "source": "profile",
    "source_detail": "configs/profiles/ci.yml"
  }
}
```

Supported provenance sources are:

* `default`
* `profile`
* `environment`
* `cli_override`

Environment provenance records the environment variable name, such as
`ARTIFACTS_ROOT`. CLI-style provenance records the override section, such as
`cli_overrides.runtime`. Some normalized runtime values are derived by the
existing runtime resolver from explicit higher-precedence inputs; those entries
keep the winning source and include the resolver in `source_detail`.

The result supports deterministic serialization through `to_dict()`,
`to_json_dict()`, and `to_json()`. Serialized resolved config and provenance are
explanatory audit views only. They do not become authoritative over canonical
artifacts, checked-in workflow configs, or persisted run artifacts.

## Validation CLI

M39.3 adds a thin CLI wrapper around the same profile and resolution APIs:

```powershell
python -m src.cli.validate_config --profile ci
```

Other supported forms:

```powershell
python -m src.cli.validate_config --profile local
python -m src.cli.validate_config --profile notebook
python -m src.cli.validate_config --profile pipeline
python -m src.cli.validate_config --profile-path configs/profiles/ci.yml
python -m src.cli.validate_config --profile ci --output artifacts/_derived/config_validation/ci_validation.json
```

Exit codes:

* `0`: profile and resolved config are valid
* nonzero: profile loading, profile validation, or config resolution failed

The CLI prints deterministic JSON to stdout unless `--output` is provided. When
an output path is provided, it writes sorted, stable JSON there and prints a
short status summary to stderr. Validation reports are advisory,
non-authoritative, and disposable; keep them under `_derived` or another
generated-output location.

The CLI does not load `.env`, run workflows, check live data availability, read
market data, call external services, require credentials, or mutate canonical
artifacts.

## Environment Doctor

M39.4 adds an advisory readiness checker that layers environment-oriented checks
on top of profile validation and config resolution:

```powershell
python -m src.cli.stratlake_doctor --profile local
```

Supported forms:

```powershell
python -m src.cli.stratlake_doctor --profile ci
python -m src.cli.stratlake_doctor --profile notebook
python -m src.cli.stratlake_doctor --profile pipeline
python -m src.cli.stratlake_doctor --profile-path configs/profiles/ci.yml
python -m src.cli.stratlake_doctor --profile ci --output artifacts/_derived/environment_readiness/ci_doctor.json
```

The doctor checks:

* Python runtime version
* importability of core local M39 modules
* selected profile loading and config resolution
* artifact-first boundaries, including direct scan and non-authoritative derived outputs
* portable repository-relative path fields
* checked-in workflow config references
* optional roots such as `features_root` and `marketlake_root`
* whether the artifact root target or nearest existing parent appears writable
* whether an explicit output path follows the `_derived` recommendation

Checks are grouped in deterministic readiness-flow categories: `runtime`,
`imports`, `profile`, `boundaries`, `paths`, `workflow_configs`, `data_roots`,
and `outputs`. Reports preserve that order for human scanning and stable CI
diffs.

Finding statuses:

* `pass`: readiness check succeeded
* `warning`: something may need attention, but the doctor remains CI-safe
* `fail`: validation failed and the command exits nonzero
* `skipped`: check is not applicable, commonly because optional data does not exist

Warnings and skipped checks do not fail the command. Missing feature or
marketlake data is reported as skipped or warning unless the profile itself says
live data is required. The starter profiles explicitly do not require network
access, credentials, external services, or live market data.

The doctor prints deterministic JSON to stdout unless `--output` is provided.
With `--output`, it writes stable sorted JSON to the requested path and prints a
short status summary to stderr. Reports are advisory, disposable,
non-authoritative, and should be kept under
`artifacts/_derived/environment_readiness/` or another generated-output
location.

The doctor does not load `.env`, run backtests, build features, run portfolios,
run campaigns, inspect secrets, validate live data availability, call external
services, or mutate canonical artifacts.

## Dry-Run And Explain

M39.5 adds deterministic explain helpers for inspecting what StratLake would use
before a workflow executes:

```python
from src.config.explain import build_runtime_explain_report

report = build_runtime_explain_report("ci", workflow="strategy")
payload = report.to_json_dict()
```

The dedicated CLI is:

```powershell
python -m src.cli.explain_config --profile ci --workflow strategy
```

Supported forms:

```powershell
python -m src.cli.explain_config --profile local
python -m src.cli.explain_config --profile-path configs/profiles/ci.yml
python -m src.cli.explain_config --profile ci --workflow portfolio
python -m src.cli.explain_config --profile ci --workflow strategy --output artifacts/_derived/config_explain/ci_strategy_explain.json
```

Supported workflow subjects are `generic`, `strategy`, `alpha`, `portfolio`,
`pipeline`, `campaign`, and `evidence_review`. The selected subject only changes
the assumptions included in the report; it does not invoke that workflow.

Explain reports include:

* selected profile metadata and source
* resolved settings, workflow configs, runtime config, review config, and boundaries
* full provenance plus source-count summary
* highest-precedence source used
* key path summary and expected artifact-root locations
* workflow assumptions for the selected subject
* explicit safety flags showing no workflow execution and no canonical mutation

The CLI prints deterministic JSON to stdout unless `--output` is provided. With
`--output`, it writes stable sorted JSON to the requested path and prints a
short status summary to stderr. Reports are advisory, derived/disposable, and
non-authoritative; keep written reports under
`artifacts/_derived/config_explain/` or another generated-output location.

Explain helpers do not load `.env`, run backtests, train alpha models, build
portfolios, run pipeline stages, run research campaigns, generate evidence
review packs, require live data, require credentials, call external services, or
mutate canonical artifacts. They are notebook-friendly and pipeline-friendly
inspection tools for resolved configuration context.

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

The M39.2 resolver composes those existing contracts for inspection and
reproducibility. Workflow APIs continue to own execution behavior.
