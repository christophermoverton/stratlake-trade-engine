# Review Configuration

## Overview

Unified research review now resolves one normalized configuration contract from:

```text
repository defaults < review config file < CLI overrides
```

The contract lives in `src/config/review.py` and is consumed by
`src/cli/compare_research.py` and `src/research/review.py`.
For the end-to-end workflow that uses this contract, see
[milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md).

## Example Workflow

For one small committed end-to-end review example, run:

```powershell
python docs/examples/milestone_13_review_promotion_workflow.py
```

That example uses checked-in fixture registries and writes deterministic review
outputs under `docs/examples/output/milestone_13_review_promotion_workflow/`.

## Repository Defaults

Repository defaults live in [../configs/review.yml](../configs/review.yml).
They define the default run-type filters, ranking metrics, and output
preferences used when no explicit review config file or CLI override is
supplied.

## Supported Sections

### `filters`

Supported keys:

* `run_types`
* `timeframe`
* `dataset`
* `alpha_name`
* `strategy_name`
* `portfolio_name`
* `top_k_per_type`

### `ranking`

Supported keys:

* `alpha_evaluation_primary_metric`
* `alpha_evaluation_secondary_metric`
* `strategy_primary_metric`
* `strategy_secondary_metric`
* `portfolio_primary_metric`
* `portfolio_secondary_metric`

Primary metrics control displayed ranking. Secondary metrics remain the
deterministic tie-break within each run type before name and run-id ordering.

### `output`

Supported keys:

* `path`
* `emit_plots`

`path` can point to either a directory or a concrete CSV path. `emit_plots`
lets review runs disable plot generation without changing the leaderboard
schema.

### `promotion_gates`

Review-level promotion gates reuse the existing promotion gate contract from
`src/research/promotion.py`.

## Accepted Shapes

You can provide either a top-level review payload or a nested `review` block.

Example:

```yaml
review:
  filters:
    run_types: [strategy, portfolio]
    timeframe: 1D
    top_k_per_type: 3
  ranking:
    strategy_primary_metric: sharpe_ratio
    strategy_secondary_metric: total_return
    portfolio_primary_metric: total_return
    portfolio_secondary_metric: sharpe_ratio
  output:
    emit_plots: false
  promotion_gates:
    status_on_pass: review_ready
    status_on_fail: needs_work
    gates:
      - gate_id: minimum_rows
        source: metrics
        metric_path: entry_count
        comparator: gte
        threshold: 2
```

## CLI Overrides

`compare_research` supports the same filter flags as before plus explicit review
config overrides:

* `--review-config`
* `--alpha-metric`
* `--alpha-secondary-metric`
* `--strategy-metric`
* `--strategy-secondary-metric`
* `--portfolio-metric`
* `--portfolio-secondary-metric`
* `--promotion-gates`
* `--disable-plots`

Example:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_research `
  --from-registry `
  --review-config configs/review.yml `
  --run-types strategy portfolio `
  --strategy-metric sharpe_ratio `
  --portfolio-metric total_return `
  --promotion-gates configs/review_gates.yml `
  --disable-plots
```

## Persistence

Unified review artifacts now persist the effective resolved review config in:

* `review_summary.json`
* `manifest.json`

That persisted payload reflects the fully merged contract after repository
defaults, review-config file values, and CLI overrides are applied.

This config doc is intentionally focused on the contract itself. For practical
guidance on where the review layer fits in the pipeline, how to read
`leaderboard.csv`, and how to interpret review-level promotion results, see
[milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md).
