# Portfolio Artifact Logging

## Overview

The portfolio artifact layer persists deterministic outputs for completed
portfolio runs so they can be inspected, validated, and queried later without
recomputing the in-memory result.

Current implementation:

* writes one deterministic run directory under `artifacts/portfolios/`
* writes normalized config, components, return, weight, equity, metric, QA,
  and manifest artifacts
* runs deterministic artifact-consistency QA before finalizing the manifest
* appends one `run_type: "portfolio"` entry to
  `artifacts/portfolios/registry.jsonl`
* writes split-level artifacts and aggregate summaries for walk-forward runs

This layer is intentionally file-based and aligned with the repository's
existing artifact-driven workflow.

## Location

```text
src/portfolio/artifacts.py
src/portfolio/qa.py
src/portfolio/walk_forward.py
src/research/registry.py
```

Primary single-run entrypoint:

```python
write_portfolio_artifacts(
    output_dir,
    portfolio_output,
    metrics,
    config,
    components,
)
```

Walk-forward entrypoint:

```python
run_portfolio_walk_forward(
    component_run_ids,
    evaluation_config_path,
    allocator,
    timeframe,
    output_dir,
    portfolio_name=...,
)
```

## Artifact Layout

### Single-Run Portfolio Artifacts

```text
artifacts/portfolios/<run_id>/
  config.json
  components.json
  weights.csv
  portfolio_returns.csv
  portfolio_equity_curve.csv
  metrics.json
  qa_summary.json
  manifest.json
```

### Walk-Forward Portfolio Artifacts

```text
artifacts/portfolios/<run_id>/
  config.json
  components.json
  metrics_by_split.csv
  aggregate_metrics.json
  manifest.json
  splits/
    <split_id>/
      split.json
      weights.csv
      portfolio_returns.csv
      portfolio_equity_curve.csv
      metrics.json
      qa_summary.json
```

The run directory name is the deterministic portfolio `run_id`.

## File-By-File Contract

### `config.json`

Purpose:

* stores the normalized portfolio configuration used for the run

Current fields typically include:

* `portfolio_name`
* `allocator`
* `initial_capital`
* `alignment_policy`
* `timeframe`
* `evaluation_config_path` when applicable

Relationship to in-memory outputs:

* documents the construction settings that produced the in-memory
  `portfolio_output`
* is used for traceability, QA interpretation, and registry metadata

### `components.json`

Purpose:

* stores the resolved component strategy set used for the run

Schema:

```json
{
  "components": [
    {
      "strategy_name": "momentum_v1",
      "run_id": "momentum_v1_single_...",
      "source_artifact_path": "artifacts/strategies/momentum_v1_single_..."
    }
  ]
}
```

Relationship to in-memory outputs:

* ties each component return stream in the portfolio output back to the exact
  strategy run artifact that supplied it

### `weights.csv`

Purpose:

* stores the portfolio weight matrix in inspection-friendly CSV form

Current columns:

* `ts_utc`
* one `weight__<strategy>` column per component strategy

Relationship to in-memory outputs:

* derived directly from the `weight__<strategy>` columns in the validated
  portfolio output

### `portfolio_returns.csv`

Purpose:

* stores the timestamped portfolio return stream plus traceability columns

Current columns:

* `ts_utc`
* `strategy_return__<strategy>` columns
* `weight__<strategy>` columns
* `portfolio_return`

Relationship to in-memory outputs:

* derived directly from the validated portfolio output
* preserves the exact weighted-return inputs needed to audit each row

### `portfolio_equity_curve.csv`

Purpose:

* stores the compounded portfolio equity path

Current columns:

* `ts_utc`
* `portfolio_equity_curve`

Relationship to in-memory outputs:

* derived directly from the validated portfolio output after compounding
  `portfolio_return` from the configured `initial_capital`

### `metrics.json`

Purpose:

* stores the portfolio metric payload for a single-run portfolio or a
  walk-forward split

Typical fields:

* `total_return`
* `sharpe_ratio`
* `max_drawdown`
* `annualized_return`
* `annualized_volatility`
* `turnover`
* `exposure_pct`

Relationship to in-memory outputs:

* computed from the in-memory portfolio output with
  `compute_portfolio_metrics()`

### `aggregate_metrics.json`

Purpose:

* walk-forward-only aggregate summary across split-level portfolio metrics

Current top-level fields include:

* `aggregation_method`
* `mode`
* `split_count`
* `timeframe`
* `first_train_start`
* `last_test_end`
* `split_ids`
* `metric_summary`
* `metric_statistics`

Relationship to in-memory outputs:

* summarizes the per-split metric outputs produced during portfolio
  walk-forward evaluation
* `metric_summary` stores the mean value for each metric across splits

### `metrics_by_split.csv`

Purpose:

* walk-forward-only tabular summary of one row per evaluation split

Current leading columns:

* `split_id`
* `mode`
* `train_start`
* `train_end`
* `test_start`
* `test_end`
* `start`
* `end`
* `row_count`

Followed by portfolio metric columns such as:

* `total_return`
* `sharpe_ratio`
* `max_drawdown`
* `turnover`
* `exposure_pct`

Relationship to in-memory outputs:

* each row corresponds to one split-level portfolio run

### `qa_summary.json`

Purpose:

* stores deterministic validation results for a single portfolio output

Current fields include:

* `run_id`
* `portfolio_name`
* `allocator`
* `timeframe`
* `row_count`
* `strategy_count`
* `date_range`
* `return_summary`
* `ending_equity`
* selected metric values
* `issues`
* `validation_status`

Relationship to in-memory outputs:

* summarizes whether the in-memory portfolio output and persisted artifacts
  passed return-consistency, equity-curve, weight-behavior, and artifact
  consistency checks

### `manifest.json`

Purpose:

* fast first-stop summary for inspecting a portfolio run

Single-run manifest fields include:

* `run_id`
* `timestamp`
* `portfolio_name`
* `allocator`
* `alignment_policy`
* `initial_capital`
* `component_count`
* `artifact_files`
* `artifacts`
* `row_counts`
* `metric_summary`
* `qa_summary_status`

Walk-forward manifest fields include:

* `run_id`
* `timestamp`
* `portfolio_name`
* `allocator`
* `evaluation_mode`
* `evaluation_config_path`
* `component_count`
* `split_count`
* `split_artifact_dirs`
* `artifact_files`
* `aggregate_metric_summary`
* `aggregate_metrics_path`
* `metrics_by_split_path`

Relationship to in-memory outputs:

* inventory and metadata only
* designed for fast inspection without reopening each CSV or JSON file

### `split.json`

Purpose:

* walk-forward-only persisted split metadata for one split directory

Current fields:

* `split_id`
* `mode`
* `train_start`
* `train_end`
* `test_start`
* `test_end`
* `start`
* `end`
* `row_count`

Relationship to in-memory outputs:

* documents the exact evaluation split used to construct that split-level
  portfolio

## Traceability Columns

Portfolio artifacts use two prefixed column families to preserve traceability:

* `strategy_return__<strategy>`
* `weight__<strategy>`

Why they matter:

* they show the exact component return stream used for each strategy at each
  timestamp
* they show the exact portfolio weight applied to each component strategy
* they let QA recompute `portfolio_return` deterministically from saved CSV
  artifacts

The portfolio contract requires these prefixed columns to stay consistent by
strategy suffix. A saved portfolio output cannot contain weights for one
strategy without the matching component return column, or vice versa.

## QA Summary And Deterministic Validation

Portfolio QA checks are designed to fail fast and preserve trust in persisted
artifacts.

Current checks include:

* weighted-return consistency
* equity-curve consistency
* weight finiteness and row-sum behavior
* allocator-specific checks such as constant equal-weight behavior
* artifact-vs-memory consistency for CSV and metric files

`qa_summary.json` records the result with:

* `validation_status: "pass"` when checks succeed
* `validation_status: "fail"` plus issue text when checks fail

This keeps the portfolio layer aligned with the repository's broader
deterministic, no lookahead, artifact-first QA model.

## Registry Relationship

Each completed portfolio run appends one JSON object line to:

```text
artifacts/portfolios/registry.jsonl
```

Portfolio registry entries include:

* `run_type: "portfolio"`
* `portfolio_name`
* `allocator_name`
* `component_run_ids`
* `component_strategy_names`
* `timeframe`
* `start_ts`
* `end_ts`
* `artifact_path`
* `metrics`
* `evaluation_config_path`
* `split_count`
* `config`
* `components`
* `metadata`

This provides a lightweight query surface for portfolio runs analogous to the
strategy registry workflow.

## Inspecting A Portfolio Run

Suggested inspection order:

1. open `manifest.json`
2. confirm `qa_summary.json` passed
3. inspect `metrics.json` or `aggregate_metrics.json`
4. inspect `portfolio_returns.csv` for component-level traceability
5. inspect `portfolio_equity_curve.csv` for the portfolio path

For walk-forward runs, also inspect:

* `metrics_by_split.csv`
* `splits/<split_id>/split.json`
* `splits/<split_id>/qa_summary.json`
