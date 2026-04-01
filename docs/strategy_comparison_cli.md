# Unified Comparison Review CLI

## Overview

The repository now has one comparison review model across the existing CLI
entrypoints:

* `compare_strategies` compares configured strategies
* `compare_alpha` compares registered alpha-evaluation runs
* `compare_research` provides one registry-backed review surface across alpha,
  strategy, and portfolio runs

The comparison flow remains artifact-first. Each command reuses existing
research-layer comparison modules, writes deterministic comparison artifacts,
and prints a compact console leaderboard.

Fresh execution is still available only for strategy comparison. Alpha and
unified review remain registry-backed.

## Locations

```text
src/cli/comparison_cli.py
src/cli/compare_strategies.py
src/cli/compare_alpha.py
src/cli/compare_research.py
src/research/compare.py
src/research/alpha_eval/compare.py
src/research/review.py
```

## Shared CLI Conventions

Across the comparison CLIs:

* preferred flag style is kebab-case, for example `--from-registry`,
  `--output-path`, and `--top-k`
* legacy snake_case aliases remain supported for backward compatibility, for
  example `--from_registry`, `--output_path`, and `--top_k`
* repeated list arguments accept comma-separated and/or space-separated values
* console output follows one shared shape:
  identifier, key metadata, `rows`, leaderboard preview, `leaderboard_csv`,
  and `leaderboard_json`

Identifiers:

* strategy and alpha comparisons print `comparison_id`
* unified review prints `review_id`

## Strategy Comparison

Module execution:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1
```

Required argument:

* `--strategies` -> strategy names from `configs/strategies.yml`; accepts
  comma-separated and/or space-separated values

Optional arguments:

* `--evaluation [PATH]` -> run comparison in walk-forward mode using the
  default or provided evaluation config
* `--start` -> inclusive start date for fresh single-run comparisons
* `--end` -> exclusive end date for fresh single-run comparisons
* `--metric` -> metric used for ranking, defaults to `sharpe_ratio`
* `--top-k` / `--top_k` -> keep only the top `N` rows in the final leaderboard
* `--from-registry` / `--from_registry` -> use stored registry runs instead of
  executing new runs
* `--output-path` / `--output_path` -> override the default comparison output
  path

Without `--evaluation`, comparison runs in single-run mode. With
`--evaluation`, comparison runs in walk-forward mode and reuses the same split
logic as the single-strategy runner.

`--start` and `--end` are supported only for fresh single-run comparison mode.
They cannot be combined with `--evaluation` or `--from-registry`.

### Strategy Execution Modes

Fresh execution mode is the default behavior.

Per strategy:

```text
configs/strategies.yml
        ->
single-run or walk-forward runner
        ->
metrics from the executed result
        ->
leaderboard row
```

Registry mode is enabled with `--from-registry`.

Per strategy:

```text
artifacts/strategies/registry.jsonl
        ->
filter by strategy_name
        ->
filter by evaluation_mode
        ->
if walk-forward, filter by evaluation_config_path
        ->
select latest matching run
        ->
leaderboard row
```

Selection rule:

* latest matching run by descending `timestamp`
* tie-break by descending `run_id`

### Strategy Leaderboard Schema

Each in-memory leaderboard row includes:

* `rank`
* `strategy_name`
* `run_id`
* `evaluation_mode`
* `selected_metric_name`
* `selected_metric_value`
* `cumulative_return`
* `total_return`
* `sharpe_ratio`
* `max_drawdown`
* `annualized_return`
* `annualized_volatility`
* `volatility`
* `win_rate`
* `hit_rate`
* `profit_factor`
* `turnover`
* `exposure_pct`

Sorting rule:

* selected metric descending
* missing selected metric values last
* tie-break by `strategy_name`
* final tie-break by `evaluation_mode`

Persisted `leaderboard.csv` still omits `run_id` so repeated fresh executions
can keep saved comparison artifacts byte-stable. Persisted
`leaderboard.json` now also includes `comparison_id`, which makes the strategy
surface line up with the alpha comparison CLI.

Default outputs:

* `artifacts/comparisons/<comparison_id>/leaderboard.csv`
* `artifacts/comparisons/<comparison_id>/leaderboard.json`
* `artifacts/comparisons/<comparison_id>/plots/metric_comparison_<metric>.png`
* `artifacts/comparisons/<comparison_id>/plots/equity_comparison.png` when the comparison set stays intentionally small

`<comparison_id>` is deterministic for the same strategy list, metric,
evaluation mode, selection mode, evaluation path, and `top_k` inputs.

Comparison plots now come from the primary `compare_strategies` workflow rather
than the example-only flow. The default policy stays restrained:

* the metric bar chart is emitted only for leaderboards with `2` to `10` rows
* the equity overlay is emitted only for leaderboards with `2` to `6` rows and
  requires each selected run to have `equity_curve.csv`
* skipped plots are recorded in `leaderboard.json` so large review surfaces
  stay artifact-first without silently dropping context

## Alpha And Unified Review

Alpha comparison:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_alpha --from-registry --dataset features_daily --timeframe 1D
```

Unified research review:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_research --from-registry --run-types alpha_evaluation strategy portfolio --top-k 3
```

These commands remain registry-backed and reuse the existing alpha comparison
and research review modules. The CLI layer now shares the same argument and
summary conventions as strategy comparison instead of maintaining separate
parsing and print logic.

Unified review continues to cover:

* alpha runs via `run_type=alpha_evaluation`
* strategy runs via `run_type=strategy`
* portfolio runs via `run_type=portfolio`

Unified review artifact contract:

* `artifacts/reviews/<review_id>/leaderboard.csv`
* `artifacts/reviews/<review_id>/review_summary.json`
* `artifacts/reviews/<review_id>/manifest.json`
* `artifacts/reviews/<review_id>/plots/<run_type>/metric_comparison_<metric>.png`
  for run types with review-sized groups
* `artifacts/reviews/<review_id>/promotion_gates.json` when review-level
  promotion gates are configured

`review_summary.json` is the canonical JSON summary for unified review runs.
`manifest.json` inventories the written files, row counts, selected review
metrics, plot paths, skipped plot reasons, and optional promotion-gate summary
so review outputs are explicitly auditable across reruns.

Unified review now also resolves one explicit review configuration contract.
Defaults come from `configs/review.yml`, an optional `--review-config` file can
override those defaults, and CLI flags win last. The effective merged config is
persisted into `review_summary.json` and `manifest.json` so review filtering,
ranking metrics, plot preferences, and review-level promotion gates are
auditable after the run is written.

## Compatibility Notes

The main compatibility change is additive:

* kebab-case flags are now the preferred public shape
* legacy snake_case flags still work
* strategy comparison console output now includes `comparison_id` and `rows`
* strategy comparison JSON artifacts now include `comparison_id`

Existing command behavior, artifact paths, ranking rules, and fresh-vs-registry
execution semantics are otherwise preserved.

## Related Docs

* [docs/cli_strategy_runner.md](cli_strategy_runner.md)
* [docs/walk_forward_strategy_runner.md](walk_forward_strategy_runner.md)
* [docs/experiment_artifact_logging.md](experiment_artifact_logging.md)
* [docs/strategy_performance_metrics.md](strategy_performance_metrics.md)
* [docs/research_visualization_workflow.md](research_visualization_workflow.md)
* [docs/review_configuration.md](review_configuration.md)
* [docs/examples/strategy_comparison_example.md](examples/strategy_comparison_example.md)
