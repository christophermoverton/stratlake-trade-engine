from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def compare_strategies(
    strategies: Sequence[str],
    *,
    metric: str | None = None,
    evaluation_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    top_k: int | None = None,
    from_registry: bool = False,
    output_path: str | Path | None = None,
) -> ExecutionResult:
    """Run deterministic strategy comparison through the shared API."""

    from src.research.compare import DEFAULT_METRIC, compare_strategies as compare

    raw_result = compare(
        list(strategies),
        metric=metric or DEFAULT_METRIC,
        evaluation_path=None if evaluation_path is None else Path(evaluation_path),
        start=start,
        end=end,
        top_k=top_k,
        from_registry=from_registry,
        output_path=None if output_path is None else Path(output_path),
    )
    return summarize_execution_result(
        workflow="strategy_comparison",
        raw_result=raw_result,
        name="strategy_comparison",
        run_id=raw_result.comparison_id,
        artifact_dir=raw_result.csv_path.parent,
        output_paths={
            "leaderboard_csv": raw_result.csv_path,
            "summary_json": raw_result.json_path,
            **{f"plot_{index}": path for index, path in enumerate(raw_result.plot_paths, start=1)},
        },
        extra={
            "metric": raw_result.metric,
            "evaluation_mode": raw_result.evaluation_mode,
            "selection_mode": raw_result.selection_mode,
            "row_count": int(len(raw_result.leaderboard)),
        },
    )


def compare_strategies_from_cli_args(args) -> ExecutionResult:
    from src.cli import compare_strategies as cli
    from src.cli.comparison_cli import optional_output_path

    raw_result = cli.compare_strategies(
        cli.parse_strategy_names(args.strategies),
        metric=args.metric,
        evaluation_path=None if args.evaluation is None else Path(args.evaluation),
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        from_registry=args.from_registry,
        output_path=optional_output_path(args.output_path),
    )
    return summarize_execution_result(
        workflow="strategy_comparison",
        raw_result=raw_result,
        name="strategy_comparison",
        run_id=raw_result.comparison_id,
        artifact_dir=raw_result.csv_path.parent,
        output_paths={
            "leaderboard_csv": raw_result.csv_path,
            "summary_json": raw_result.json_path,
            **{f"plot_{index}": path for index, path in enumerate(raw_result.plot_paths, start=1)},
        },
        extra={
            "metric": raw_result.metric,
            "evaluation_mode": raw_result.evaluation_mode,
            "selection_mode": raw_result.selection_mode,
            "row_count": int(len(raw_result.leaderboard)),
        },
    )
