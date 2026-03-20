# Strategy Comparison CLI

## Overview

The strategy comparison CLI ranks multiple configured strategies using the
existing research execution and registry layers.

It supports two modes:

* fresh execution mode, which runs each strategy through the current pipeline
* registry mode, which reuses prior runs from `artifacts/strategies/registry.jsonl`

The comparison flow does not introduce new metrics or a separate artifact
system. It reuses the current strategy runner, walk-forward execution, and
registry schema.

---

## Location

```text
src/cli/compare_strategies.py
src/research/compare.py
```

Module execution:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1
```

---

## Arguments

Required arguments:

* `--strategies` -> comma-separated strategy names from `configs/strategies.yml`

Optional arguments:

* `--evaluation [PATH]` -> run comparison in walk-forward mode using the default or provided evaluation config
* `--metric` -> metric used for ranking, defaults to `sharpe_ratio`
* `--top_k` -> keep only the top `N` rows in the final leaderboard
* `--from_registry` -> use stored registry runs instead of executing new runs
* `--output_path` -> override the default comparison output path

Without `--evaluation`, comparison runs in single-run mode. With
`--evaluation`, comparison runs in walk-forward mode and reuses the same split
logic as the single-strategy runner.

---

## Execution Modes

### Fresh Execution Mode

Fresh execution is the default behavior.

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

This mode writes the normal per-run experiment artifacts first, then writes the
shared leaderboard outputs.

### Registry Mode

Registry mode is enabled with `--from_registry`.

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

This rule is deterministic for identical registry state.

---

## Leaderboard Schema

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

Persisted `leaderboard.csv` and `leaderboard.json` omit `run_id` on purpose so
repeated fresh executions can keep the saved comparison artifacts byte-stable
even when the CLI table still prints source run identifiers.

---

## Outputs

Default outputs:

* `artifacts/comparisons/<comparison_id>/leaderboard.csv`
* `artifacts/comparisons/<comparison_id>/leaderboard.json`

`<comparison_id>` is deterministic for the same strategy list, metric,
evaluation mode, selection mode, evaluation path, and `top_k` inputs.

The CLI also prints a compact console table that includes:

* rank
* strategy name
* run id
* evaluation mode
* selected metric value
* `total_return`
* `sharpe_ratio`
* `max_drawdown`

---

## Example Commands

Compare fresh single-run results:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1
```

Compare fresh walk-forward results:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,sma_crossover_v1 --evaluation
```

Rank by a different metric:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1 --metric total_return
```

Use stored registry runs:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1 --from_registry
```

Limit the final leaderboard:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1,buy_and_hold_v1 --top_k 2
```

Write the leaderboard to a custom location:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1 --output_path artifacts/comparisons/custom_leaderboard.csv
```

---

## Related Docs

* [docs/cli_strategy_runner.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/cli_strategy_runner.md)
* [docs/walk_forward_strategy_runner.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/walk_forward_strategy_runner.md)
* [docs/experiment_artifact_logging.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/experiment_artifact_logging.md)
* [docs/strategy_performance_metrics.md](/C:/Users/christophermoverton/stratlake-trade-engine/docs/strategy_performance_metrics.md)
