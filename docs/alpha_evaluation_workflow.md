# Alpha Evaluation Workflow

Milestone 12 adds a deterministic alpha-evaluation pipeline for answering one
question before downstream signal mapping or portfolio construction:

> Do this alpha model's predictions line up with future cross-sectional returns?

## Workflow

```text
Alpha -> Predict -> Align -> Validate -> Evaluate -> Aggregate -> Persist -> Register -> Compare
```

Milestone 11.5 covers deterministic alpha modeling, backtesting, and portfolio
construction. This Milestone 12 workflow is the evaluation layer that sits
between prediction and downstream signal mapping. Milestone 13 then adds the
registry-backed review layer that compares completed alpha-evaluation runs
alongside strategy and portfolio outputs in one shared review pack. See
[milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md).
In that unified review output, alpha rows keep forecast-quality ranking
separate from downstream outcomes: `selected_metric_*` and
`secondary_metric_*` stay forecast-only, while optional `sleeve_metric_*` and
`linked_portfolio_*` fields expose sleeve tradability and selected portfolio
consumption context when those downstream artifacts exist.

## Quick Start

Run the committed end-to-end example:

```powershell
python docs/examples/alpha_evaluation_end_to_end.py
```

Run the CLI directly:

```powershell
python -m src.cli.run_alpha --alpha-name cs_linear_ret_1d --mode evaluate --start 2025-01-01 --end 2025-03-01
python -m src.cli.run_alpha --alpha-name rank_composite_momentum --start 2025-01-01 --end 2025-03-01
python -m src.cli.run_alpha --alpha-name cs_linear_ret_1d --start 2025-01-01 --end 2025-03-01 --signal-policy top_bottom_quantile --signal-quantile 0.2
python -m src.cli.run_alpha_evaluation --alpha-name cs_linear_ret_1d --start 2025-01-01 --end 2025-03-01
python -m src.cli.run_alpha_evaluation --alpha-model your_model --model-class path/to/model.py:YourModel --dataset features_daily --target-column target_ret_1d --price-column close
python -m src.cli.run_alpha_evaluation --alpha-model your_model --model-class path/to/model.py:YourModel --dataset features_daily --target-column target_ret_1d --price-column close --signal-policy rank_long_short
python -m src.cli.compare_alpha --from-registry
python -m src.cli.compare_alpha --from-registry --view sleeve --sleeve-metric sharpe_ratio
python -m src.cli.compare_alpha --from-registry --view combined --metric ic_ir --sleeve-metric sharpe_ratio --dataset features_daily --horizon 5 --mapping-name top_bottom_quantile[q=0.2]
```

Built-in alpha definitions now live in `configs/alphas.yml`. Each named entry
declares the canonical dataset, target column, feature columns, model type,
model parameters, evaluation horizon, and optional defaults such as
`price_column` or `min_cross_section_size`. Use `--alpha-name` for those
registry-backed definitions, and keep `--model-class` for ad hoc external
models when you need a custom implementation. The registry now supports
`model_type: sklearn` for built-in scikit-learn regressors such as
`linear_regression` and `ridge`.

`python -m src.cli.run_alpha` is now the first-class built-in alpha runner.
It resolves one named config from `configs/alphas.yml`, runs it on
`features_daily`, prints a concise summary, and defaults to `full` mode. That
default now persists the evaluation outputs, maps signals with
`rank_long_short` when no explicit policy is supplied, generates a deterministic
sleeve return stream, and writes an `alpha_run_scaffold.json` file describing
the completed flow. Use `--mode evaluate` for evaluation-only runs with no
sleeve artifacts or scaffold file.

The example writes deterministic outputs under:

```text
docs/examples/output/alpha_evaluation_end_to_end/
```

## Stage By Stage

### Predict

`python -m src.cli.run_alpha_evaluation` trains a registered
`BaseAlphaModel`, then calls `predict_alpha_model(...)` to produce a canonical
prediction frame with `prediction_score`.

### Map Signals

Signal mapping is optional and explicit. When configured through
`signal_mapping` in YAML or the CLI flags `--signal-policy` plus
`--signal-quantile` where required, the workflow maps raw
`prediction_score` into a separate `signal` artifact.

Important boundary:

* alpha evaluation still scores raw `prediction_score` against `forward_return`
* mapped `signal` output is persisted for downstream backtesting or portfolio
  construction
* no score-to-signal conversion happens implicitly

### Align

`align_forward_returns(...)` joins predictions back to the loaded feature data
and creates `forward_return` for each `(symbol, ts_utc, timeframe)` row.

Supported alignment modes:

* `price_column`: computes forward return as `future_price / price - 1`
* `realized_return_column`: compounds future realized returns over the chosen
  horizon

### Validate

`validate_alpha_evaluation_input(...)` enforces the alpha-evaluation contract:

* required structural columns: `symbol`, `ts_utc`, `timeframe`
* numeric prediction and forward-return columns
* sorted, duplicate-free canonical rows
* enough cross-sectional breadth per timestamp
* cross-sectional variation in both predictions and forward returns

### Evaluate

`evaluate_alpha_predictions(...)` iterates by timestamp and computes:

* `ic`: Pearson correlation between `prediction_score` and `forward_return`
* `rank_ic`: Spearman-style rank correlation implemented by ranking both
  series, then applying Pearson correlation
* `sample_size`: number of non-null observations in that timestamp slice

The per-period output is written as `ic_timeseries.csv`.

### Aggregate

The evaluator summarizes the time series into deterministic metrics:

* `mean_ic`
* `std_ic`
* `ic_ir`
* `mean_rank_ic`
* `std_rank_ic`
* `rank_ic_ir`
* `n_periods`
* `ic_positive_rate`
* `valid_timestamps`

`ICIR` in the docs corresponds to the persisted `ic_ir` field. The rank-based
analogue is stored as `rank_ic_ir`.

### Persist

`write_alpha_evaluation_artifacts(...)` writes:

* `predictions.parquet`
* `signals.parquet` when signal mapping is configured
* `signal_mapping.json` when signal mapping is configured
* `signal_semantics.json` when signal mapping is configured
* `training_summary.json`
* `coefficients.json`
* `cross_section_diagnostics.json`
* `qa_summary.json`
* `ic_timeseries.csv`
* `alpha_metrics.json`
* `manifest.json`

`python -m src.cli.run_alpha --mode full` then adds:

* `sleeve_returns.csv`
* `sleeve_equity_curve.csv`
* `sleeve_metrics.json`
* `alpha_run_scaffold.json`

Default location:

```text
artifacts/alpha/<run_id>/
```

### Register

`register_alpha_evaluation_run(...)` upserts one registry entry into:

```text
artifacts/alpha/registry.jsonl
```

Each entry links the run id, dataset, timeframe, evaluation horizon, summary
metrics, explicit signal contract metadata, and concrete artifact paths.

### Compare

`python -m src.cli.compare_alpha --from-registry` loads alpha-evaluation runs
from the registry, ranks them by a selected view, and writes a leaderboard.

Supported views:

* `forecast`: rank on forecast quality only, using `--metric`
* `sleeve`: rank on downstream tradability only, using `--sleeve-metric`
* `combined`: rank forecast first, then sleeve as a deterministic tie-breaker

Useful filters:

* `--dataset`
* `--timeframe`
* `--evaluation-horizon` or `--horizon`
* `--mapping-name`

`--mapping-name` matches the persisted signal mapping name when one exists in
mapping metadata. Otherwise the CLI derives a stable fallback label from the
policy, such as `top_bottom_quantile[q=0.2]`.

Default comparison output:

```text
artifacts/alpha_comparisons/<comparison_id>/
```

Files:

* `leaderboard.csv`
* `leaderboard.json`

CLI output keeps the domains separate:

* forecast view prints forecast-quality columns such as `ic_ir` and `mean_ic`
* sleeve view prints tradability columns such as `sharpe_ratio`,
  `total_return`, and `turnover`
* combined view prints both with explicit `forecast_*` and `sleeve_*` labels

## Metrics

### IC

Information Coefficient is the same-timestamp cross-sectional Pearson
correlation between `prediction_score` and `forward_return`.

Interpretation:

* positive values mean higher scores tended to align with higher future returns
* negative values mean the alpha was directionally wrong
* values near zero suggest weak or unstable cross-sectional signal

### Rank IC

Rank IC is the same relationship computed on ranked values instead of raw
levels. It is more robust when the alpha mainly gets ordering right but not
magnitude.

### ICIR

ICIR is the mean IC divided by the sample standard deviation of IC across
timestamps. Higher values imply more stable signal quality over time.

In the implementation:

* ICIR is stored as `ic_ir`
* Rank ICIR is stored as `rank_ic_ir`

## How To Read Results

Use `alpha_metrics.json` for the headline summary and `ic_timeseries.csv` to
see whether the result is stable across time rather than driven by one period.
Use `qa_summary.json` when you want a quick answer to the practical follow-up
question: "is this alpha not just predictive, but usable?"

Common reading pattern:

* `mean_ic` tells you the average directional relationship
* `ic_ir` tells you whether that relationship is repeatable
* `mean_rank_ic` helps when ranking matters more than magnitude
* `n_periods` tells you how many timestamps were actually usable
* `qa_summary.json` adds practical checks for timestamp coverage, post-warmup
  nulls, and signal tradability metrics such as implied turnover,
  concentration, and net-exposure balance

## Alpha QA And Promotion Gates

Alpha runs now persist `qa_summary.json` alongside the existing evaluation
artifacts. The summary is intentionally practical rather than theoretical and
focuses on:

* minimum valid timestamps
* valid timestamp coverage rate
* minimum and mean valid cross-section size
* post-warmup prediction and forward-return null rates
* implied sleeve turnover from mapped signals
* single-name concentration
* net-exposure balance

When signal mapping is not configured, the QA summary still records the
forecast-quality checks and marks signal-tradability review as a warning rather
than a hard failure.

Promotion gates can now read those QA fields directly through the shared
`qa_summary` source in `src/research/promotion.py`.

Example:

```powershell
python -m src.cli.run_alpha `
  --alpha-name cs_linear_ret_1d `
  --start 2025-01-01 `
  --end 2025-03-01 `
  --promotion-gates configs/alpha_promotion_gates.yml
```

Example threshold file:

```yaml
promotion_gates:
  status_on_pass: eligible
  status_on_fail: blocked
  gates:
    - gate_id: min_valid_timestamps
      source: qa_summary
      metric_path: forecast.valid_timestamps
      comparator: gte
      threshold: 20
    - gate_id: min_mean_cross_section
      source: qa_summary
      metric_path: cross_section.mean_valid_cross_section_size
      comparator: gte
      threshold: 25
    - gate_id: post_warmup_prediction_nulls
      source: qa_summary
      metric_path: nulls.prediction_null_rate
      comparator: lte
      threshold: 0.02
    - gate_id: signal_turnover
      source: qa_summary
      metric_path: signals.mean_turnover
      comparator: lte
      threshold: 0.75
      missing_behavior: skip
    - gate_id: max_single_name_concentration
      source: qa_summary
      metric_path: signals.max_single_name_abs_share
      comparator: lte
      threshold: 0.35
      missing_behavior: skip
```

Those thresholds are examples, not universal truths. Daily broad-universe
alphas usually want larger valid timestamp counts and cross-sections, while
smaller research datasets may need looser thresholds during exploratory work.

## Artifacts And Registry Entries

Artifacts are the run-local files under `artifacts/alpha/<run_id>/`.
Registry entries are the cross-run index that point back to those files.

That relationship enables two workflows:

* audit one run by opening `training_summary.json`, `coefficients.json`,
  `cross_section_diagnostics.json`, `qa_summary.json`, `alpha_metrics.json`,
  `ic_timeseries.csv`, and `manifest.json`
* compare many runs by loading the registry and generating a leaderboard

## Start Here

* [README.md](../README.md)
* [docs/examples/alpha_evaluation_end_to_end.py](examples/alpha_evaluation_end_to_end.py)
* [docs/alpha_workflow.md](alpha_workflow.md)
* [docs/milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md)
