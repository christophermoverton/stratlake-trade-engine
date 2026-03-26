# Strict Mode

## Overview

Strict mode is the enforcement policy for StratLake's research-validity
framework.

It is available in both primary CLIs:

* `python -m src.cli.run_strategy --strict`
* `python -m src.cli.run_portfolio --strict`

Strict mode does not introduce different metrics or alternative backtests. It
changes how validation and sanity findings are handled.

## What `--strict` Does

When strict mode is enabled:

* the resolved runtime policy is persisted as `{"enabled": true, "source": "cli"}`
* sanity warnings that are merely recorded in non-strict mode become blocking
  failures
* portfolio validation and strict-mode validation failures stop the CLI with
  exit code `1`

Strict mode is also enabled when config turns on either:

* `strict_mode.enabled`
* `sanity.strict_sanity_checks`
* `portfolio_validation.strict_sanity_checks`

In that case the persisted source is `"config"`.

## Strict vs Non-Strict

### Non-strict mode

Non-strict mode is useful when you want to inspect suspicious results without
blocking the run.

Current behavior:

* warning-level sanity issues are recorded in metrics and QA outputs
* the run can still write artifacts and registry entries
* `sanity_status` can be `"warn"`

### Strict mode

Strict mode is useful for release gating, regression checks, and auditable
validation-heavy workflows.

Current behavior:

* flagged sanity issues fail fast
* strategy and portfolio CLIs stop before persistence on those failures
* error messages state that artifacts and registry writes were prevented

## No-Write Guarantee

The strict-mode no-write guarantee is narrow and intentional:

* if a run fails strict-mode validation before persistence, artifact and
  registry writes are not attempted

This applies to pre-persistence strict failures in strategy, portfolio, and
walk-forward execution paths.

Post-write consistency validation is a separate step. Those checks still run
after artifacts are written, so strict mode should not be documented as a
blanket guarantee against every possible write on every failure path.

## Failure Messages

Strict-mode failures are surfaced as CLI errors of the form:

```text
Run failed: Strict mode failure [<validator>] (<scope>): <message> Artifacts and registry writes were prevented.
```

Non-strict validation failures use the same CLI exit pattern but are not
described as strict-mode failures unless strict enforcement was active.

## Where It Applies

### Strategy runs

Strict mode affects:

* single-run strategy execution
* walk-forward strategy splits
* walk-forward aggregate sanity checks

### Portfolio runs

Strict mode affects:

* single-run portfolio construction
* portfolio walk-forward splits
* portfolio walk-forward aggregate reporting through flagged split counts and
  sanity summaries

## Recommended Usage

Use strict mode when:

* preparing merge or release artifacts
* comparing research results that should be policy-compliant
* validating end-to-end portfolio workflows
* treating warnings as blockers in CI-style research runs

Use non-strict mode when:

* exploring ideas interactively
* inspecting borderline results
* intentionally reviewing warning-state outputs

## Related Docs

* [runtime_configuration.md](runtime_configuration.md)
* [research_validity_framework.md](research_validity_framework.md)
* [execution_model.md](execution_model.md)
