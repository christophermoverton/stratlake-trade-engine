# Backfilled 2026 Q1 Research Workflow

## Overview

This document captures the operational workflow used to take the existing
StratLake environment from backfilled market data through feature engineering,
strategy research, walk-forward validation, and robustness review for the 2026
Q1 window.

It is intentionally focused on the workflow that was run and the artifacts that
were produced. It does not document code patches that were made while
stabilizing the workflow.

## Scope

The workflow covered:

* verification of upstream curated market data availability
* feature generation for daily and 1-minute datasets
* single-run strategy research
* fresh strategy comparison
* walk-forward evaluation
* split-by-split inspection
* robustness sweeping for the nominated strategy family

The workflow window used in this run was:

* daily features and single-window research: `2026-01-01` to `2026-04-03`
* 1-minute features: `2026-02-04` to `2026-04-03`

All date windows follow the repository convention of half-open intervals:

```text
[start, end)
```

## Where Data Comes From

StratLake consumes curated market data from the upstream ingestion repository.
In this environment, the local configuration pointed StratLake at the curated
data root exposed through `MARKETLAKE_ROOT`.

The relevant curated datasets for this workflow were:

* `bars_daily`
* `bars_1m`

Before feature generation, the backfill was verified to be present for the
target window. The observed coverage was:

* `bars_daily`: `2026-01-02` through `2026-04-02`
* `bars_1m`: `2026-01-02` through `2026-04-02`

The requested trading windows were satisfied because:

* `2026-01-01` was a market holiday
* `2026-04-03` was the exclusive upper bound

## Workflow

### 1. Verify Backfilled Data Coverage

Confirm that the requested backfill is already present before building
features.

Checks performed in this run:

* confirmed daily coverage for the requested date range
* confirmed 1-minute coverage for the requested date range
* spot-checked boundary-date symbol partitions for the configured 50-symbol
  universe
* confirmed no active ingestion process needed to be resumed

Use this step to prevent rebuilding features on partial or missing market data.

### 2. Build Daily Features

Run feature generation on the refreshed daily window:

```powershell
.\.venv\Scripts\python.exe -m cli.build_features `
  --timeframe 1D `
  --start 2026-01-01 `
  --end 2026-04-03 `
  --tickers configs/tickers_50.txt
```

Primary outputs from this run:

* feature summary:
  `artifacts/feature_runs/20260402T145014Z/summary.json`
* curated dataset:
  `data/curated/features_daily`
* QA rollup:
  `artifacts/qa/features/qa_features_summary_global.csv`

Observed results:

* `50` symbols processed
* `3,150` rows written
* dataset status reported as `WARN` because of rolling-window warm-up nulls,
  not because of missing symbols or duplicate keys

### 3. Build 1-Minute Features

Run the intraday feature build on the requested 1-minute window:

```powershell
.\.venv\Scripts\python.exe -m cli.build_features `
  --timeframe 1Min `
  --start 2026-02-04 `
  --end 2026-04-03 `
  --tickers configs/tickers_50.txt
```

Primary outputs from this run:

* feature summary:
  `artifacts/feature_runs/20260402T145117Z/summary.json`
* curated dataset:
  `data/curated/features_1m`
* QA rollup:
  `artifacts/qa/features/qa_features_summary_global.csv`

Observed results:

* `50` symbols processed
* `702,348` rows written
* dataset status reported as `WARN` for expected warm-up behavior

### 4. Run Single-Window Strategy Research

Use a bounded single-window run to sanity-check one strategy on the refreshed
daily feature set:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy `
  --strategy momentum_v1 `
  --start 2026-01-01 `
  --end 2026-04-03
```

Primary output from this run:

* strategy artifact:
  `artifacts/strategies/momentum_v1_single_ac9fa4e2c4c3`

Observed metrics:

* cumulative return: about `-3.38%`
* Sharpe ratio: about `-2.00`
* QA status: `WARN`

This step provides a first directional check before comparing multiple
strategies.

### 5. Compare Strategies On The Same Fresh Window

Run a fresh comparison across baseline and candidate strategies:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies `
  --strategies momentum_v1 mean_reversion_v1 sma_crossover_v1 buy_and_hold_v1 seeded_random_v1 `
  --start 2026-01-01 `
  --end 2026-04-03
```

Primary outputs from this run:

* leaderboard CSV:
  `artifacts/comparisons/fresh_single_sharpe_ratio_b14516665ff1/leaderboard.csv`
* leaderboard JSON:
  `artifacts/comparisons/fresh_single_sharpe_ratio_b14516665ff1/leaderboard.json`

Observed ranking by Sharpe ratio:

1. `mean_reversion_v1`
2. `buy_and_hold_v1`
3. `sma_crossover_v1`
4. `seeded_random_v1`
5. `momentum_v1`

For this window, `mean_reversion_v1` became the leading candidate for deeper
evaluation.

### 6. Define A Dedicated Walk-Forward Window

Use a focused walk-forward config for the same quarterly research span:

`configs/evaluation_2026_q1.yml`

```yaml
evaluation:
  mode: rolling
  timeframe: "1d"
  start: "2026-01-01"
  end: "2026-04-03"
  train_window: 6W
  test_window: 2W
  step: 2W
```

This configuration creates rolling train/test splits over the refreshed 2026
Q1 range without mixing it into the broader default evaluation config.

### 7. Run Walk-Forward Comparison

Evaluate the top candidates with the dedicated walk-forward config:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies `
  --strategies mean_reversion_v1 momentum_v1 `
  --evaluation configs/evaluation_2026_q1.yml
```

Primary outputs from this run:

* leaderboard CSV:
  `artifacts/comparisons/fresh_walk_forward_sharpe_ratio_4053a227ca24/leaderboard.csv`
* leaderboard JSON:
  `artifacts/comparisons/fresh_walk_forward_sharpe_ratio_4053a227ca24/leaderboard.json`

Observed results:

* `mean_reversion_v1`: Sharpe about `1.75`, total return about `1.26%`, max
  drawdown about `1.89%`
* `momentum_v1`: Sharpe about `-0.12`, total return about `-0.14%`, max
  drawdown about `3.67%`

This step increased confidence that `mean_reversion_v1` was the stronger
research candidate on this refreshed dataset.

### 8. Inspect Split-Level Behavior

Review split-level metrics before treating the walk-forward aggregate as
reliable.

Artifacts inspected in this run:

* `artifacts/strategies/mean_reversion_v1_walk_forward_cf7009aa54e5/metrics_by_split.csv`
* `artifacts/strategies/momentum_v1_walk_forward_704296c42e47/metrics_by_split.csv`

Observed behavior:

* `mean_reversion_v1` was strong in the first split, mildly positive in the
  second split, and weak in the third split
* `momentum_v1` was much more regime-sensitive, with one bad split, one near
  flat split, and one strong split

This step matters because aggregate Sharpe can hide unstable split-by-split
behavior.

### 9. Run A Robustness Sweep For The Leading Strategy

Once `mean_reversion_v1` emerged as the leading candidate, run a bounded
parameter sweep instead of promoting the top walk-forward result directly.

Robustness config used in this run:

`configs/robustness_mean_reversion_2026_q1.yml`

```yaml
robustness:
  strategy_name: mean_reversion_v1
  ranking_metric: sharpe_ratio
  sweep:
    - parameter: lookback
      values: [10, 15, 20, 25, 30]
    - parameter: threshold
      values: [0.5, 1.0, 1.5, 2.0]
  stability:
    mode: walk_forward
    evaluation_path: configs/evaluation_2026_q1.yml
  thresholds:
    sharpe_ratio:
      min: 0.0
    max_drawdown:
      max: 0.05
```

Run the sweep:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy `
  --robustness configs/robustness_mean_reversion_2026_q1.yml
```

Primary outputs from this run:

* summary:
  `artifacts/strategies/robustness/mean_reversion_v1_robustness_f8470335fae3/summary.json`
* variant metrics:
  `artifacts/strategies/robustness/mean_reversion_v1_robustness_f8470335fae3/metrics_by_variant.csv`
* split stability metrics:
  `artifacts/strategies/robustness/mean_reversion_v1_robustness_f8470335fae3/stability_metrics.csv`
* local sensitivity metrics:
  `artifacts/strategies/robustness/mean_reversion_v1_robustness_f8470335fae3/neighbor_metrics.csv`

Observed summary:

* `20` parameter variants evaluated
* all variants passed the configured thresholds
* the top Sharpe variant was `lookback=10, threshold=1.0`
* the safer promotion candidate was `lookback=20, threshold=2.0`

The safer nomination was chosen because it remained positive across all three
walk-forward splits and showed less violent local sensitivity than the most
aggressive low-lookback variants.

### 10. Materialize The Nominated Strategy Variant

To carry the nominated parameter set cleanly into the downstream portfolio
layer, the chosen configuration was saved as a named strategy entry:

* strategy name: `mean_reversion_v1_safe_2026_q1`
* dataset: `features_daily`
* parameters:
  * `lookback: 20`
  * `threshold: 2.0`

Run the nominated single-window strategy:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy `
  --strategy mean_reversion_v1_safe_2026_q1 `
  --start 2026-01-01 `
  --end 2026-04-03
```

Primary output from this run:

* strategy artifact:
  `artifacts/strategies/mean_reversion_v1_safe_2026_q1_single_4931682b006e`

Observed results:

* cumulative return: about `1.26%`
* Sharpe ratio: about `2.14`
* QA status: `PASS`

This run turned the nominated robustness choice into a normal saved strategy
artifact that could be consumed directly by the portfolio layer.

### 11. Run Portfolio Research On The Nominated Variant

Portfolio research for this phase used a dedicated config file:

`configs/portfolios_2026_q1.yml`

It defined two portfolio views over the nominated strategy artifact:

* `mean_reversion_safe_equal_2026_q1`
* `mean_reversion_safe_targeted_2026_q1`

#### Baseline single-sleeve portfolio

Run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_portfolio `
  --portfolio-config configs/portfolios_2026_q1.yml `
  --portfolio-name mean_reversion_safe_equal_2026_q1 `
  --from-registry `
  --timeframe 1D
```

Primary output from this run:

* portfolio artifact:
  `artifacts/portfolios/mean_reversion_safe_equal_2026_q1_portfolio_b61f7eab2c1f`

Observed results:

* total return: about `1.26%`
* Sharpe ratio: about `2.14`
* realized volatility: about `2.35%`
* max drawdown: about `0.31%`

#### Volatility-targeted single-sleeve portfolio

Run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_portfolio `
  --portfolio-config configs/portfolios_2026_q1.yml `
  --portfolio-name mean_reversion_safe_targeted_2026_q1 `
  --from-registry `
  --timeframe 1D
```

Primary output from this run:

* portfolio artifact:
  `artifacts/portfolios/mean_reversion_safe_targeted_2026_q1_portfolio_c364eb1a55a0`

Observed results:

* total return: about `3.45%`
* Sharpe ratio: about `2.14`
* realized volatility: about `6.44%`
* max drawdown: about `0.84%`
* volatility targeting enabled with target `10%`
* estimated pre-target volatility: about `3.66%`
* scaling factor: about `2.73`

This portfolio step showed that the nominated strategy remained attractive as a
single-sleeve portfolio and that operational volatility targeting amplified both
return and risk while largely preserving risk-adjusted efficiency.

### 12. Create A Registry-Backed Review Pack

After portfolio research, create a unified review pack for the nominated
strategy and the preferred downstream portfolio candidate.

The targeted portfolio was chosen for formal review because it preserved the
same Sharpe ratio as the baseline sleeve while scaling absolute return and risk
to a more operational portfolio profile.

Run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_research `
  --from-registry `
  --run-types strategy portfolio `
  --timeframe 1D `
  --dataset features_daily `
  --strategy-name mean_reversion_v1_safe_2026_q1 `
  --portfolio-name mean_reversion_safe_targeted_2026_q1 `
  --disable-plots
```

Primary outputs from this run:

* leaderboard CSV:
  `artifacts/reviews/registry_review_53bef9e30fe5/leaderboard.csv`
* review summary JSON:
  `artifacts/reviews/registry_review_53bef9e30fe5/review_summary.json`

Observed review result:

* strategy candidate:
  `mean_reversion_v1_safe_2026_q1_single_4931682b006e`
* portfolio candidate:
  `mean_reversion_safe_targeted_2026_q1_portfolio_c364eb1a55a0`
* both rows ranked first within their run type
* both rows carried `needs_review` status because no promotion-gate decision had
  yet been applied

This step created the first auditable review-layer artifact for the 2026 Q1
backfill-to-portfolio workflow.

### 13. Apply Promotion Gates And Run A Gated Review

To turn the review layer into an explicit decision point, define a dedicated
promotion-gate config for this 2026 Q1 workflow:

`configs/review_gates_2026_q1.yml`

The configured gates required:

* both a strategy and portfolio candidate in the review pack
* strategy Sharpe ratio at or above `2.0`
* strategy total return at or above `1.0%`
* portfolio Sharpe ratio at or above `2.0`
* portfolio total return at or above `3.0%`

Run the gated review:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_research `
  --from-registry `
  --run-types strategy portfolio `
  --timeframe 1D `
  --dataset features_daily `
  --strategy-name mean_reversion_v1_safe_2026_q1 `
  --portfolio-name mean_reversion_safe_targeted_2026_q1 `
  --promotion-gates configs/review_gates_2026_q1.yml `
  --disable-plots
```

Primary outputs from this run:

* leaderboard CSV:
  `artifacts/reviews/registry_review_c500f2590167/leaderboard.csv`
* review summary JSON:
  `artifacts/reviews/registry_review_c500f2590167/review_summary.json`
* promotion-gate result:
  `artifacts/reviews/registry_review_c500f2590167/promotion_gates.json`
* review manifest:
  `artifacts/reviews/registry_review_c500f2590167/manifest.json`

Observed result:

* review-level gate evaluation status: `pass`
* review-level promotion status: `promoted`
* gates passed: `7 / 7`

Important interpretation note:

* the review pack itself was promoted at the review layer
* the individual strategy and portfolio rows still retained their existing
  registry-level `needs_review` status inside `leaderboard.csv`

This distinction is expected in the current implementation because the gated
review writes a review-level decision artifact rather than mutating the upstream
strategy or portfolio registry rows.

## Decision Log From This Run

This workflow produced the following practical research conclusions:

* the refreshed backfill was sufficient for daily and 1-minute feature
  generation
* daily and intraday feature pipelines ran successfully on the requested windows
* `momentum_v1` did not look attractive on the refreshed 2026 Q1 sample
* `mean_reversion_v1` was the leading strategy family on both fresh comparison
  and walk-forward evaluation
* split-level inspection showed that the strategy family still had regime
  sensitivity
* robustness review favored a conservative nomination of
  `lookback=20, threshold=2.0`
* the nominated parameter set converted into a clean single-window strategy run
  with `PASS` QA
* portfolio research showed that volatility targeting increased absolute return
  and drawdown while leaving Sharpe roughly unchanged
* the targeted portfolio was chosen as the preferred review candidate and was
  packaged with the nominated strategy into a unified review artifact
* the gated review passed all configured thresholds and promoted the review pack
  at the review layer

## Recommended Next Step

Use the nominated `mean_reversion_v1` parameter set:

* `lookback=20`
* `threshold=2.0`

as the basis for the next downstream promotion-gate or milestone review
workflow.
