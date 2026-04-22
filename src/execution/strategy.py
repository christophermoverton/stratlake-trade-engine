from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_strategy(
    strategy_name: str,
    *,
    start: str | None = None,
    end: str | None = None,
    evaluation_path: str | Path | None = None,
    robustness_path: str | Path | None = None,
    execution_override: dict[str, Any] | None = None,
    strict: bool = False,
    simulation_path: str | Path | None = None,
    simulation_config: dict[str, Any] | None = None,
    promotion_gates_path: str | Path | None = None,
    strategies_config_path: str | Path | None = None,
) -> ExecutionResult:
    """Run one strategy workflow through the shared Python execution surface."""

    from src.cli import run_strategy as cli

    config_path = Path(strategies_config_path) if strategies_config_path is not None else cli.STRATEGIES_CONFIG
    robustness_config = None
    if robustness_path is not None:
        robustness_config = cli.load_robustness_config(Path(robustness_path))
        strategy_name = robustness_config.resolve_strategy_name(strategy_name)
    if not strategy_name:
        raise ValueError("A strategy must be provided.")

    config = cli.get_strategy_config(strategy_name, path=config_path)
    if promotion_gates_path is not None:
        config = dict(config)
        config["promotion_gates"] = cli.load_promotion_gate_config(str(promotion_gates_path))
    runtime_config = cli.resolve_runtime_config(
        config,
        cli_overrides=None if execution_override is None else {"execution": execution_override},
        cli_strict=strict,
    )
    if simulation_path is not None and robustness_config is not None:
        raise ValueError("simulation_path cannot be combined with robustness_path.")
    if robustness_config is not None:
        raw_result = cli.run_robustness_experiment(
            strategy_name,
            robustness_config=robustness_config,
            start=start,
            end=end,
            evaluation_path=None if evaluation_path is None else Path(evaluation_path),
            execution_config=runtime_config.execution,
            strict=strict,
            strategy_config_path=config_path,
        )
    elif evaluation_path is not None:
        if simulation_path is not None or simulation_config is not None:
            raise ValueError("Simulation cannot be combined with strategy walk-forward evaluation.")
        if start or end:
            raise ValueError("The --start and --end arguments cannot be combined with --evaluation.")
        strategy = cli.build_strategy(strategy_name, config)
        raw_result = cli.run_walk_forward_experiment(
            strategy_name,
            strategy,
            evaluation_path=Path(evaluation_path),
            strategy_config=config,
            execution_config=runtime_config.execution,
            strict=strict,
        )
    else:
        resolved_simulation_config = simulation_config
        if simulation_path is not None:
            resolved_simulation_config = cli.load_simulation_config(Path(simulation_path)).to_dict()
        raw_result = cli.run_strategy_experiment(
            strategy_name,
            start=start,
            end=end,
            execution_config=runtime_config.execution,
            strict=strict,
            simulation_config=resolved_simulation_config,
            strategies_config_path=config_path,
        )

    artifact_dir = getattr(raw_result, "experiment_dir", None)
    output_paths = _strategy_output_paths(Path(artifact_dir)) if artifact_dir is not None else {}
    return summarize_execution_result(
        workflow="strategy",
        raw_result=raw_result,
        output_paths=output_paths,
        extra={"result_type": raw_result.__class__.__name__},
    )


def run_strategy_from_cli_args(args: Any) -> ExecutionResult:
    """Run strategy execution from a parsed CLI namespace."""

    from src.cli import run_strategy as cli

    execution_override = cli._execution_override_from_args(args)
    return run_strategy(
        str(args.strategy) if args.strategy is not None else "",
        start=args.start,
        end=args.end,
        evaluation_path=None if args.evaluation is None else Path(args.evaluation),
        robustness_path=None if args.robustness is None else Path(args.robustness),
        execution_override=execution_override,
        strict=bool(args.strict),
        simulation_path=None if args.simulation is None else Path(args.simulation),
        promotion_gates_path=args.promotion_gates,
        strategies_config_path=Path(args.strategies_config),
    )


def run_strategy_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_strategy import parse_args

    return run_strategy_from_cli_args(parse_args(argv))


def _strategy_output_paths(artifact_dir: Path) -> dict[str, Path]:
    paths = {
        "manifest_json": artifact_dir / "manifest.json",
        "metrics_json": artifact_dir / "metrics.json",
        "qa_summary_json": artifact_dir / "qa_summary.json",
        "equity_curve_csv": artifact_dir / "equity_curve.csv",
    }
    ranked_configs = artifact_dir / "ranked_configs.csv"
    if ranked_configs.exists():
        paths["ranked_configs_csv"] = ranked_configs
    return paths
