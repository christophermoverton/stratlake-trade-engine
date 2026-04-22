from __future__ import annotations

from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_portfolio(argv: Sequence[str] | None = None, **kwargs) -> ExecutionResult:
    """Run portfolio construction through the shared execution surface.

    The current portfolio runner has a broad CLI-era argument surface. This bridge
    preserves that behavior while giving notebooks a stable import path and a
    standardized result contract.
    """

    from src.cli.run_portfolio import _run_cli_impl

    raw_result = _run_cli_impl(argv, **kwargs)
    if isinstance(raw_result, dict):
        return ExecutionResult(
            workflow="portfolio",
            run_id=str(raw_result.get("identifier", "portfolio")),
            name=str(raw_result.get("name", "portfolio")),
            raw_result=raw_result,
        )
    return summarize_execution_result(
        workflow="portfolio",
        raw_result=raw_result,
        output_paths={
            "manifest_json": raw_result.experiment_dir / "manifest.json",
            "metrics_json": raw_result.experiment_dir / "metrics.json",
            "portfolio_returns_csv": raw_result.experiment_dir / "portfolio_returns.csv",
            "portfolio_equity_curve_csv": raw_result.experiment_dir / "portfolio_equity_curve.csv",
            "weights_csv": raw_result.experiment_dir / "weights.csv",
        },
        extra={"timeframe": raw_result.timeframe, "allocator_name": raw_result.allocator_name},
    )


def run_portfolio_from_argv(argv: Sequence[str] | None = None, **kwargs) -> ExecutionResult:
    return run_portfolio(argv, **kwargs)
