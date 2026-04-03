# Backfilled 2026 Q1 Alpha Workflow

## Overview

This document extends the existing 2026 Q1 real-data case study into the alpha
layer.

It shows how to:

* run a built-in alpha model on the real `features_daily` Q1 2026 surface
* evaluate the alpha with persisted forecast-quality artifacts
* map alpha scores into deterministic sleeve signals
* consume the resulting alpha sleeve in the portfolio layer
* review the alpha outputs alongside the already-existing Q1 2026 strategy and
  portfolio work

This workflow is intentionally concrete. The commands, run ids, and artifact
paths below were executed in the current repository state.

## Relationship To The Existing Q1 2026 Workflow

Read this document as the alpha-layer continuation of:

* [backfilled_2026_q1_research_workflow.md](backfilled_2026_q1_research_workflow.md)
* [alpha_workflow.md](alpha_workflow.md)
* [alpha_evaluation_workflow.md](alpha_evaluation_workflow.md)
* [milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md)
* [milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md)

The earlier Q1 2026 workflow already established:

* real daily feature coverage on `[2026-01-01, 2026-04-03)`
* nominated strategy:
  `mean_reversion_v1_safe_2026_q1`
* preferred downstream strategy-backed portfolio:
  `mean_reversion_safe_targeted_2026_q1`

This alpha case study reuses the same `features_daily` surface and the same
half-open date semantics:

```text
[start, end)
```

## Configs Added For This Case Study

### Dedicated alpha catalog

`configs/alphas_2026_q1.yml`

This catalog defines one built-in alpha case-study entry:

* alpha name: `rank_composite_momentum_2026_q1`
* dataset: `features_daily`
* training window: `[2026-01-01, 2026-03-02)`
* prediction window: `[2026-03-02, 2026-04-03)`
* target column: `target_ret_5d`
* signal mapping: `top_bottom_quantile` with `quantile: 0.2`
* stable mapping label: `top_bottom_quantile_q20`

### Dedicated alpha-sleeve portfolio config

`configs/portfolios_alpha_2026_q1.yml`

This config defines two single-sleeve portfolio views over the alpha run:

* `rank_composite_momentum_2026_q1_equal`
* `rank_composite_momentum_2026_q1_targeted`

Both portfolios consume the alpha artifact as:

```yaml
artifact_type: alpha_sleeve
```

The checked-in config is pinned to the concrete alpha run id produced in this
repo state:

* `rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15`

If you rerun the alpha workflow and want a fresh downstream portfolio, update
the `run_id` fields in `configs/portfolios_alpha_2026_q1.yml` to the new alpha
run id before calling `run_portfolio`.

## Practical Precondition: Normalize The Daily Feature Schema

The first alpha attempt on this repo state failed even after the Q1 2026 daily
features were rebuilt, because older yearly partitions in `features_daily` had
been written before canonical alpha targets were added.

That meant:

* the Q1 2026 parquet files contained `target_ret_1d` and `target_ret_5d`
* older yearly partitions did not
* the dataset-wide feature view therefore still resolved without the target
  columns needed for alpha training

The practical fix was to normalize the full daily feature history so every
yearly partition exposed the same schema.

Run:

```powershell
.\.venv\Scripts\python.exe -m cli.build_features `
  --timeframe 1D `
  --start 2023-01-01 `
  --end 2026-04-03 `
  --tickers configs/tickers_50.txt
```

Normalization artifact from this run:

* `artifacts/feature_runs/20260403T152351Z/summary.json`

The Q1-only refresh run is still useful as an operational boundary check:

```powershell
.\.venv\Scripts\python.exe -m cli.build_features `
  --timeframe 1D `
  --start 2026-01-01 `
  --end 2026-04-03 `
  --tickers configs/tickers_50.txt
```

Q1 refresh artifact from this run:

* `artifacts/feature_runs/20260403T152150Z/summary.json`

Observed Q1 refresh results:

* `50` symbols processed
* `3,150` rows written
* target columns now present in the 2026 yearly partitions:
  `target_ret_1d`, `target_ret_5d`

If your local `features_daily` surface is already fully normalized, you can
skip the full-history normalization step and go directly to the alpha run.

## Step 1. Run The Built-In Q1 2026 Alpha

Run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_alpha `
  --config configs/alphas_2026_q1.yml `
  --alpha-name rank_composite_momentum_2026_q1 `
  --promotion-gates configs/alpha_promotion_gates.yml
```

Observed output:

* alpha run id:
  `rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15`
* artifact directory:
  `artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15`
* `mean_ic`: about `-0.0079`
* `ic_ir`: about `-0.0454`
* `n_periods`: `19`

Why this command matters:

* it uses the built-in alpha registry path rather than an ad hoc custom model
* it trains and predicts on the real Q1 2026 `features_daily` surface
* it persists both forecast artifacts and sleeve artifacts in one run

## Step 2. Inspect The Alpha Artifacts

Primary files written under
`artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15/`:

* `predictions.parquet`
* `signals.parquet`
* `signal_mapping.json`
* `training_summary.json`
* `coefficients.json`
* `cross_section_diagnostics.json`
* `qa_summary.json`
* `alpha_metrics.json`
* `ic_timeseries.csv`
* `promotion_gates.json`
* `sleeve_returns.csv`
* `sleeve_equity_curve.csv`
* `sleeve_metrics.json`
* `manifest.json`
* `alpha_run_scaffold.json`

Useful interpretation points from the real run:

* training window:
  `[2026-01-01, 2026-03-02)`
* prediction/evaluation window:
  `[2026-03-02, 2026-04-03)`
* training rows: `1,950`
* prediction rows: `1,200`
* evaluation rows after alignment/null filtering: `950`
* valid timestamps: `19`
* valid cross-section size: `50` names per timestamp

Observed QA from `qa_summary.json`:

* overall QA status: `pass`
* valid timestamp rate: `1.0`
* post-warmup prediction null rate: `0.0`
* post-warmup forward-return null rate: `0.0`
* mean implied turnover: about `0.1633`
* max single-name absolute exposure share: `0.05`

Observed forecast-quality result from `alpha_metrics.json`:

* `mean_ic`: about `-0.0079`
* `ic_ir`: about `-0.0454`
* `mean_rank_ic`: about `-0.0019`
* `rank_ic_ir`: about `-0.0117`

Observed promotion result from `manifest.json`:

* QA summary passed
* promotion gates: `6 / 7` passed
* evaluation status: `fail`
* promotion status: `blocked`

Interpretation:

* the run is operationally valid and auditable
* the mapped sleeve is tradable by the configured QA checks
* the alpha itself was not forecast-strong enough on this Q1 2026 sample to
  clear the configured promotion threshold

That is still a useful case-study result because it demonstrates the full
real-data alpha workflow on the same surface as the rest of the Q1 2026
research pipeline.

## Step 3. Confirm Deterministic Alpha-To-Signal Mapping

This case study uses explicit score-to-signal mapping from
`configs/alphas_2026_q1.yml`:

```yaml
signal_mapping:
  policy: top_bottom_quantile
  quantile: 0.2
  metadata:
    name: top_bottom_quantile_q20
```

That means each timestamp cross-section is mapped as:

* top `20%` of names: `+1.0`
* bottom `20%` of names: `-1.0`
* middle names: `0.0`

Determinism is preserved because:

* mapping happens explicitly after prediction
* ties break by `symbol` after sorting by score
* the stable mapping name is persisted as `top_bottom_quantile_q20`

Artifact references:

* mapped signals:
  `artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15/signals.parquet`
* mapping metadata:
  `artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15/signal_mapping.json`

## Step 4. Generate And Consume The Alpha Sleeve

`python -m src.cli.run_alpha` in `full` mode already wrote the deterministic
alpha sleeve:

* `artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15/sleeve_returns.csv`
* `artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15/sleeve_equity_curve.csv`
* `artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15/sleeve_metrics.json`

Observed sleeve metrics from the real run:

* total return: about `-1.12%`
* Sharpe ratio: about `-2.83`
* annualized volatility: about `4.16%`
* max drawdown: about `1.36%`

Run the downstream targeted portfolio view:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_portfolio `
  --portfolio-config configs/portfolios_alpha_2026_q1.yml `
  --portfolio-name rank_composite_momentum_2026_q1_targeted `
  --timeframe 1D
```

Observed output:

* portfolio run id:
  `rank_composite_momentum_2026_q1_targeted_portfolio_aba629146c7e`
* artifact directory:
  `artifacts/portfolios/rank_composite_momentum_2026_q1_targeted_portfolio_aba629146c7e`

Observed portfolio metrics:

* total return: about `-2.68%`
* Sharpe ratio: about `-2.83`
* realized volatility: about `9.91%`
* max drawdown: about `3.23%`
* volatility targeting: enabled
* estimated pre-target volatility: about `4.20%`
* scaling factor: about `2.38`

This mirrors the existing strategy-backed Q1 2026 portfolio workflow:

* one saved upstream research artifact
* one deterministic downstream sleeve
* one portfolio config that consumes that sleeve
* one auditable portfolio artifact directory

## Step 5. Compare The Alpha Run On The Alpha Surface

Run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_alpha `
  --from-registry `
  --alpha-name rank_composite_momentum_2026_q1 `
  --view combined `
  --metric ic_ir `
  --sleeve-metric sharpe_ratio `
  --dataset features_daily `
  --timeframe 1D `
  --horizon 5 `
  --mapping-name top_bottom_quantile_q20
```

Observed comparison output:

* comparison id:
  `registry_alpha_combined_82eab5b654ca`
* leaderboard CSV:
  `artifacts/alpha_comparisons/registry_alpha_combined_82eab5b654ca/leaderboard.csv`
* leaderboard JSON:
  `artifacts/alpha_comparisons/registry_alpha_combined_82eab5b654ca/leaderboard.json`

Observed leaderboard row:

* alpha name: `rank_composite_momentum_2026_q1`
* run id: `rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15`
* mapping: `top_bottom_quantile_q20`
* forecast `ic_ir`: about `-0.0454`
* sleeve Sharpe ratio: about `-2.83`
* sleeve total return: about `-1.12%`

## Step 6. Review Alongside The Existing 2026 Q1 Strategy And Portfolio Work

To review the alpha against the already-existing nominated Q1 2026 strategy and
portfolio artifacts, run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_research `
  --from-registry `
  --run-types alpha_evaluation strategy portfolio `
  --timeframe 1D `
  --dataset features_daily `
  --alpha-name rank_composite_momentum_2026_q1 `
  --strategy-name mean_reversion_v1_safe_2026_q1 `
  --portfolio-name mean_reversion_safe_targeted_2026_q1 `
  --disable-plots
```

Observed review output:

* review id:
  `registry_review_aa6ed26ba065`
* leaderboard CSV:
  `artifacts/reviews/registry_review_aa6ed26ba065/leaderboard.csv`
* review summary JSON:
  `artifacts/reviews/registry_review_aa6ed26ba065/review_summary.json`

Observed rows:

* alpha:
  `rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15`
  with `ic_ir` about `-0.0454` and promotion status `rejected`
* strategy:
  `mean_reversion_v1_safe_2026_q1_single_4931682b006e`
  with Sharpe about `2.1423`
* portfolio:
  `mean_reversion_safe_targeted_2026_q1_portfolio_c364eb1a55a0`
  with Sharpe about `2.1423`

This is the cleanest cross-layer comparison for the current case study because
it puts:

* the real Q1 2026 alpha result
* the already-nominated Q1 2026 strategy result
* the already-preferred Q1 2026 strategy-backed portfolio result

onto one shared review surface.

## Optional: Review The Alpha Together With Its Own Sleeve-Backed Portfolio

If you want a review pack that includes the new alpha-backed portfolio instead
of the earlier strategy-backed portfolio, run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_research `
  --from-registry `
  --run-types alpha_evaluation strategy portfolio `
  --timeframe 1D `
  --dataset features_daily `
  --alpha-name rank_composite_momentum_2026_q1 `
  --strategy-name mean_reversion_v1_safe_2026_q1 `
  --portfolio-name rank_composite_momentum_2026_q1_targeted `
  --disable-plots
```

Observed output:

* review id:
  `registry_review_08b3be12d3e0`

That view is useful when you want to keep the alpha run and its downstream
portfolio consumption in one review bundle, while still showing the nominated
Q1 2026 strategy row for context.

## Expected Artifact Layout

For this case study, the main artifact roots are:

```text
artifacts/
  feature_runs/
    20260403T152150Z/
    20260403T152351Z/
  alpha/
    rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15/
    registry.jsonl
  alpha_comparisons/
    registry_alpha_combined_82eab5b654ca/
  portfolios/
    rank_composite_momentum_2026_q1_targeted_portfolio_aba629146c7e/
    registry.jsonl
  reviews/
    registry_review_aa6ed26ba065/
    registry_review_08b3be12d3e0/
```

## Practical Conclusions From This Run

This real-data Q1 2026 alpha case study shows that:

* the alpha layer now has a concrete workflow on the same backfilled and
  researched `features_daily` surface used by the rest of the Q1 2026 case
  study
* a built-in alpha can be trained, evaluated, mapped, sleeved, and reviewed
  through deterministic artifacts on that real surface
* older feature partitions may need schema normalization before alpha targets
  are visible to the dataset-wide loader
* operational QA can pass even when forecast quality is weak, so the alpha
  layer adds useful decision separation rather than collapsing everything into
  one backtest metric
* the resulting alpha run can be compared directly against the already-existing
  Q1 2026 strategy and portfolio artifacts through the shared review layer

## Recommended Next Step

Use this workflow as the template for future real-data alpha case studies:

1. normalize the relevant `features_daily` schema when target columns were
   introduced after earlier partitions were written
2. define one dedicated period-specific alpha catalog entry
3. run `python -m src.cli.run_alpha` on the bounded research window
4. inspect `alpha_metrics.json`, `qa_summary.json`, and sleeve artifacts before
   promoting the alpha
5. compare the alpha row against the already-selected strategy and portfolio
   rows in `compare_research`
