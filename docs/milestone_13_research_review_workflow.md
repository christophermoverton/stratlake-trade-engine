# Milestone 13 Research Review Workflow

## Overview

Milestone 13 adds the review layer that sits after alpha evaluation, strategy
runs, and portfolio runs. Its job is to pull completed artifacts back together
into one registry-backed review pack so you can rank candidates, inspect their
promotion status, and make one review-level promotion decision without
rerunning research.

Use this doc as the main practical guide for Milestone 13. For the small
committed example, see
[examples/milestone_13_review_promotion_workflow.md](examples/milestone_13_review_promotion_workflow.md).
For config details, see [review_configuration.md](review_configuration.md).

## Where It Fits

The review layer is downstream of the existing milestone workflows:

```text
feature dataset
    ->
alpha model
    ->
alpha evaluation or strategy backtest
    ->
portfolio construction
    ->
promotion gates on each saved run
    ->
registry-backed unified review
    ->
review-level promotion decision
```

In practical repository terms:

* alpha evaluation runs come from `artifacts/alpha/registry.jsonl`
* strategy runs come from `artifacts/strategies/registry.jsonl`
* portfolio runs come from `artifacts/portfolios/registry.jsonl`
* unified review writes a new artifact pack under `artifacts/reviews/<review_id>/`

The review layer does not retrain models, rerun backtests, or rebuild
portfolios. It reads saved registries and manifests, ranks the matching runs,
and writes one auditable review output set.

## What To Run

### Quick start

Run the unified review CLI directly:

```powershell
python -m src.cli.compare_research --from-registry
```

That command:

* requires `--from-registry`
* loads alpha-evaluation, strategy, and portfolio rows from their registries
* applies the default review config from `configs/review.yml`
* writes one deterministic review output directory under
  `artifacts/reviews/<review_id>/`

### Common filtered review

```powershell
python -m src.cli.compare_research `
  --from-registry `
  --run-types alpha_evaluation strategy portfolio `
  --timeframe 1D `
  --top-k 3
```

Use this when you want the top few candidates per run type in one review pack.

### Review with explicit config overrides

```powershell
python -m src.cli.compare_research `
  --from-registry `
  --review-config configs/review.yml `
  --run-types strategy portfolio `
  --strategy-metric sharpe_ratio `
  --portfolio-metric total_return `
  --promotion-gates configs/review_gates.yml `
  --disable-plots
```

This mirrors the implemented CLI surface in `src/cli/compare_research.py`:

* filters: `--run-types`, `--timeframe`, `--dataset`, `--alpha-name`,
  `--strategy-name`, `--portfolio-name`, `--top-k`
* output/config: `--output-path`, `--review-config`, `--disable-plots`
* ranking overrides: `--alpha-metric`, `--alpha-secondary-metric`,
  `--strategy-metric`, `--strategy-secondary-metric`, `--portfolio-metric`,
  `--portfolio-secondary-metric`
* review-level promotion: `--promotion-gates`

Legacy snake_case aliases still work, but the docs use kebab-case because that
is the preferred public shape.

## Review Config Contract

Review config is resolved deterministically as:

```text
repository defaults < review config file < CLI overrides
```

Repository defaults live in [../configs/review.yml](../configs/review.yml).
The normalized contract comes from `src/config/review.py`.

Supported top-level sections:

* `filters`
* `ranking`
* `output`
* `promotion_gates`

You can provide either:

* a top-level payload with those sections, or
* a nested `review:` block with the same sections

### `filters`

Supported keys:

* `run_types`
* `timeframe`
* `dataset`
* `alpha_name`
* `strategy_name`
* `portfolio_name`
* `top_k_per_type`

Notes:

* supported `run_types` are `alpha_evaluation`, `strategy`, and `portfolio`
* `dataset` applies to alpha-evaluation and strategy rows
* `top_k_per_type` is a per-run-type cap, not one global leaderboard limit

### `ranking`

Supported keys:

* `alpha_evaluation_primary_metric`
* `alpha_evaluation_secondary_metric`
* `strategy_primary_metric`
* `strategy_secondary_metric`
* `portfolio_primary_metric`
* `portfolio_secondary_metric`

The review layer ranks within each run type, not across all rows combined.
Primary metric sorts first, secondary metric breaks ties, then deterministic
name and run-id ordering finish the selection.

Default metrics from `configs/review.yml` are:

* alpha evaluation: `ic_ir`, then `mean_ic`
* strategy: `sharpe_ratio`, then `total_return`
* portfolio: `sharpe_ratio`, then `total_return`

### `output`

Supported keys:

* `path`
* `emit_plots`

Behavior:

* if `path` is a directory, the CLI writes `leaderboard.csv` inside it
* if `path` ends in `.csv`, that exact CSV path is used
* when `emit_plots` is `false`, review output still records skipped plot
  reasons in `review_summary.json` and `manifest.json`

### `promotion_gates`

Review-level promotion gates reuse the shared promotion-gate contract from
`src/research/promotion.py`.

The review CLI evaluates gates against review-specific sources such as:

* `metrics`
  Examples: `reviewed_entry_count`, `promoted_entry_count`,
  `blocked_entry_count`
* `metadata`
  Examples: `review_id`, selected run ids, selected metric names
* `aggregate_metrics`
  Review-level aggregates grouped by run type

Promotion gates are optional. When configured, the review pack writes
`promotion_gates.json` and mirrors a compact summary into `manifest.json`.

## What Gets Written

Successful unified review runs write under:

```text
artifacts/reviews/<review_id>/
```

Core files:

* `leaderboard.csv`
* `review_summary.json`
* `manifest.json`
* `promotion_gates.json` when review-level promotion gates are configured

Optional files:

* `plots/<run_type>/metric_comparison_<metric>.png` for review-sized groups

### `leaderboard.csv`

This is the fastest file to scan when you want to see what the review
selected.

Current columns are:

* `run_type`
* `rank_within_type`
* `entity_name`
* `run_id`
* `selected_metric_name`
* `selected_metric_value`
* `secondary_metric_name`
* `secondary_metric_value`
* `timeframe`
* `evaluation_mode`
* `promotion_status`
* `passed_gate_count`
* `gate_count`
* `mapping_name`
* `sleeve_metric_name`
* `sleeve_metric_value`
* `sleeve_secondary_metric_name`
* `sleeve_secondary_metric_value`
* `linked_portfolio_count`
* `linked_portfolio_names`
* `linked_portfolio_metric_name`
* `linked_portfolio_metric_value`
* `artifact_path`

Interpretation:

* `rank_within_type` is local to that run type
* `selected_metric_name` shows which metric actually drove ranking for that row
* alpha rows still rank only on `selected_metric_name` then
  `secondary_metric_name`; sleeve and linked-portfolio fields are downstream
  context and never change alpha ordering
* `mapping_name` is the persisted signal mapping label for alpha runs when one
  exists
* `sleeve_metric_*` fields summarize the alpha sleeve's own tradability, using
  `sharpe_ratio` then `total_return` when sleeve artifacts were written
* `linked_portfolio_*` fields summarize which selected portfolio review rows
  currently consume that alpha sleeve and the linked portfolio ranking metric
* `promotion_status` carries forward the saved run's promotion decision such as
  `eligible`, `promoted`, or `blocked`
* `passed_gate_count` and `gate_count` summarize the saved run's own promotion
  gate result, not the review-level decision
* `artifact_path` points back to the source run directory relative to that run
  type's artifact root

### `review_summary.json`

This is the canonical JSON summary for the unified review result.

It includes:

* `review_id`
* `filters`
* `review_config`
* `counts_by_run_type`
* `entry_count`
* `entries`
* `plot_paths`
* `run_ids`
* `skipped_plots`

Use it when you want one machine-readable view of:

* exactly which filters and ranking rules were used
* which run ids made the review
* how many rows were selected per run type
* whether plots were emitted or intentionally skipped

For alpha rows, `entries` now intentionally separates domains:

* `selected_metric_*` and `secondary_metric_*` remain the raw forecast-quality
  metrics used for alpha ranking
* `sleeve_metric_*` carries downstream sleeve tradability metrics when sleeve
  artifacts exist
* `linked_portfolio_*` carries downstream portfolio context when one of the
  selected portfolio review rows includes that alpha sleeve as a component

### `manifest.json`

`manifest.json` is the inventory and audit file for the review pack.

It includes:

* `artifact_files` and `artifact_groups`
* file-level metadata under `artifacts`
* `counts_by_run_type`
* `review_filters`
* `review_metrics`
* `selected_metrics`
* `plot_paths` and `skipped_plots`
* optional `promotion_gate_summary`

Use it to answer:

* what files were actually written
* how many rows landed in the leaderboard
* which ranking metric was used for each run type
* whether review-level promotion gates passed

### `promotion_gates.json`

When configured, this file contains the review-level promotion decision.

It includes:

* `evaluation_status`
* `promotion_status`
* `gate_count`
* `passed_gate_count`
* `failed_gate_count`
* `missing_gate_count`
* `definitions`
* `results`

Interpretation:

* `evaluation_status` is `pass` only when no gate failed and no gate is
  missing
* `promotion_status` is the configured status label, for example
  `review_ready` or `needs_work`
* each item in `results` explains the actual value, threshold, comparator, and
  pass/fail reason for one gate

## How To Read A Review Result

One practical review pass is:

1. Open `leaderboard.csv` to see which runs were selected per run type.
2. Check `promotion_status`, `passed_gate_count`, and `gate_count` to spot
   blocked or weak candidates quickly.
3. Open `review_summary.json` to confirm the exact filters, ranking metrics,
   and selected run ids.
4. Open the source run directories referenced by `artifact_path` and inspect
   each run's own `manifest.json`, `metrics.json`, `qa_summary.json`, and
   `promotion_gates.json`.
5. If present, inspect the review pack's `promotion_gates.json` to decide
   whether the overall review is ready for promotion.

The key idea is that Milestone 13 does not replace run-local review. It adds a
cross-run selection layer above it.

## Example Output Interpretation

The committed example at
[examples/milestone_13_review_promotion_workflow.md](examples/milestone_13_review_promotion_workflow.md)
shows one selected row from each run type:

* one alpha-evaluation row with `promotion_status: eligible`
* one strategy row with `promotion_status: promoted`
* one portfolio row with `promotion_status: blocked`

The example's review-level gates still pass because they only require:

* three reviewed rows
* at least one promoted candidate
* no more than one blocked row

That is an important Milestone 13 interpretation rule: a review can be
`review_ready` even when not every included run is promoted, as long as the
review-level gate policy says the overall pack passes.

## Practical Limits Intentionally Deferred

Current Milestone 13 scope is intentionally narrow:

* unified review is registry-backed only; it does not execute fresh research
  runs
* ranking happens within each run type, not on one cross-type normalized score
* plot generation is selective and may be skipped for large review sets
* review-level gates evaluate deterministic saved summaries, not manual analyst
  notes or discretionary approvals
* the review layer compares alpha-evaluation, strategy, and portfolio runs, but
  it does not yet assemble a richer narrative report the way run-scoped
  reporting does

## Related Docs

* [../README.md](../README.md)
* [getting_started.md](getting_started.md)
* [alpha_evaluation_workflow.md](alpha_evaluation_workflow.md)
* [strategy_evaluation_workflow.md](strategy_evaluation_workflow.md)
* [milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md)
* [strategy_comparison_cli.md](strategy_comparison_cli.md)
* [review_configuration.md](review_configuration.md)
* [examples/milestone_13_review_promotion_workflow.md](examples/milestone_13_review_promotion_workflow.md)
