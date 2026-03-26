# Portfolio Configuration

## Overview

Portfolio configs describe which completed strategy runs should be combined and
how the portfolio should be constructed.

The portfolio CLI accepts config files in:

* YAML
* JSON

Location in the current workflow:

```text
portfolio config
        ->
component strategy selection
        ->
portfolio construction
        ->
portfolio artifacts
```

The config file is consumed by `src.cli.run_portfolio` and normalized by the
portfolio contract layer in `src/portfolio/contracts.py`.

## Supported Top-Level Shapes

The CLI can resolve a portfolio definition from any of these shapes:

### Named mapping under `portfolios`

```yaml
portfolios:
  momentum_meanrev_equal:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
        run_id: <run_id>
      - strategy_name: mean_reversion_v1
        run_id: <run_id>
```

### List under `portfolios`

```yaml
portfolios:
  - portfolio_name: momentum_meanrev_equal
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
        run_id: <run_id>
      - strategy_name: mean_reversion_v1
        run_id: <run_id>
```

### Single top-level definition

```yaml
portfolio_name: momentum_meanrev_equal
allocator: equal_weight
components:
  - strategy_name: momentum_v1
    run_id: <run_id>
  - strategy_name: mean_reversion_v1
    run_id: <run_id>
```

If the file contains multiple portfolio definitions, pass `--portfolio-name`
to select one deterministically.

## Example YAML

```yaml
portfolios:
  momentum_meanrev_equal:
    allocator: equal_weight
    initial_capital: 1.0
    components:
      - strategy_name: momentum_v1
        run_id: <run_id>
      - strategy_name: mean_reversion_v1
        run_id: <run_id>
    alignment_policy: intersection
```

## Field Reference

### `portfolio_name`

Human-readable portfolio identifier.

Where it comes from:

* the mapping key under `portfolios`, or
* an explicit `portfolio_name` field

Why it matters:

* appears in the saved config and manifest
* contributes to the deterministic portfolio `run_id`
* is printed in the CLI summary
* is stored in portfolio registry entries

Rules:

* must be a non-empty string
* required after normalization

### `allocator`

Allocator name used to convert aligned component returns into portfolio
weights.

Current supported value:

* `equal_weight`

Rules:

* must be a non-empty string
* unsupported allocator names fail fast

### `components`

List of component strategy definitions used to build the portfolio.

General form:

```yaml
components:
  - strategy_name: momentum_v1
    run_id: <run_id>
  - strategy_name: mean_reversion_v1
    run_id: <run_id>
```

Required behavior depends on selection mode:

* without `--from-registry`, each component must include both
  `strategy_name` and `run_id`
* with `--from-registry`, each component must include `strategy_name` and must
  not include `run_id`

Rules:

* must be a non-empty list
* each component must be unique by `(strategy_name, run_id)` after resolution
* components are normalized into deterministic sorted order before artifact
  writing and run-id generation

### `initial_capital`

Starting portfolio equity used when compounding `portfolio_equity_curve`.

Example:

```yaml
initial_capital: 1.0
```

Rules:

* optional
* defaults to `1.0`
* must be float-compatible

### `alignment_policy`

Return-series alignment behavior before weights and portfolio returns are
computed.

Current supported value:

* `intersection`

`intersection` keeps only timestamps that exist for every component strategy.
This avoids filling missing returns and helps preserve deterministic,
no-lookahead portfolio construction.

Rules:

* optional
* defaults to `intersection`
* must be a non-empty string

## CLI Interaction

The portfolio config interacts with the CLI in three main ways.

### Config + explicit run ids in the file

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --timeframe 1D
```

In this mode:

* the CLI loads the named portfolio definition
* component `run_id` values are resolved directly to
  `artifacts/strategies/<run_id>/`
* the config is validated before portfolio construction begins

### Config + registry-backed selection

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --from-registry \
  --timeframe 1D
```

In this mode:

* each component is matched by `strategy_name`
* the latest matching strategy run is selected from
  `artifacts/strategies/registry.jsonl`
* selection is filtered by the requested portfolio `timeframe`
* "latest" is determined by descending `timestamp`, then descending `run_id`

### Explicit run ids without a config file

```bash
python -m src.cli.run_portfolio \
  --portfolio-name momentum_meanrev_equal \
  --run-ids run-alpha run-beta \
  --timeframe 1D
```

In this mode the CLI builds an implicit config with:

* `allocator: equal_weight`
* `initial_capital: 1.0`
* `alignment_policy: intersection`

This is useful for fast ad hoc portfolio construction, but a config file is
preferred when you want a documented, repeatable portfolio definition.

## Timeframe And Evaluation

The portfolio config does not replace the CLI `--timeframe` argument.

`--timeframe` remains required because it controls:

* portfolio metric annualization behavior
* registry filtering when `--from-registry` is used
* walk-forward timeframe compatibility checks

Walk-forward example:

```bash
python -m src.cli.run_portfolio \
  --portfolio-config configs/portfolios.yml \
  --portfolio-name momentum_meanrev_equal \
  --evaluation configs/evaluation.yml \
  --timeframe 1D
```

In walk-forward mode:

* the portfolio config still defines the component strategies and allocator
* `configs/evaluation.yml` defines the evaluation splits
* the portfolio timeframe must match the evaluation timeframe

## Validation Summary

Current portfolio config validation enforces:

* required fields `portfolio_name`, `allocator`, and `components`
* non-empty string values for identifier fields
* non-empty component lists
* unique component `(strategy_name, run_id)` pairs
* JSON-serializable normalized config payloads
* deterministic ordering before artifact writing and registry registration

These checks are designed to fail fast before portfolio construction starts.
