from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from src.config.execution import ExecutionConfig
from src.config.evaluation import EVALUATION_CONFIG
from src.config.robustness import ROBUSTNESS_CONFIG, load_robustness_config  # noqa: F401
from src.config.simulation import load_simulation_config, resolve_simulation_config  # noqa: F401
from src.config.runtime import RuntimeConfig, resolve_runtime_config
from src.data.load_features import load_features
from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import save_experiment
from src.research.input_validation import StrategyInputError
from src.research.metrics import aggregate_strategy_returns, compute_benchmark_relative_metrics, infer_periods_per_year
from src.research.promotion import load_promotion_gate_config  # noqa: F401
from src.research.robustness import RobustnessRunResult, run_robustness_experiment  # noqa: F401
from src.research.simulation import SimulationRunResult, run_return_simulation, write_simulation_artifacts
from src.research.strict_mode import ResearchStrictModeError, raise_research_validation_error
from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.signal_engine import generate_signals
from src.research.sanity import SanityCheckError, validate_strategy_backtest_sanity
from src.research.strategies import build_strategy
from src.research.strategy_qa import generate_strategy_qa_summary
from src.research.walk_forward import WalkForwardRunResult, compute_metrics, run_walk_forward_experiment  # noqa: F401
from src.pipeline.cli_adapter import build_pipeline_cli_result

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_CONFIG = REPO_ROOT / "configs" / "strategies.yml"


@dataclass(frozen=True)
class StrategyRunResult:
    """Structured result returned from a strategy experiment run."""

    strategy_name: str
    run_id: str
    metrics: dict[str, Any]
    experiment_dir: Path
    results_df: pd.DataFrame
    signal_diagnostics: dict[str, Any] = field(default_factory=dict)
    qa_summary: dict[str, Any] = field(default_factory=dict)
    simulation_result: SimulationRunResult | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a strategy experiment run."""

    parser = argparse.ArgumentParser(
        description="Run a full strategy experiment using the StratLake research pipeline."
    )
    parser.add_argument(
        "--strategies-config",
        default=str(STRATEGIES_CONFIG),
        help="Strategy config registry path. Defaults to configs/strategies.yml.",
    )
    parser.add_argument("--strategy", help="Strategy name defined in configs/strategies.yml.")
    parser.add_argument("--start", dest="start", help="Inclusive experiment start date (YYYY-MM-DD).")
    parser.add_argument("--end", dest="end", help="Exclusive experiment end date (YYYY-MM-DD).")
    parser.add_argument(
        "--evaluation",
        nargs="?",
        const=str(EVALUATION_CONFIG),
        help="Enable walk-forward evaluation using configs/evaluation.yml or a provided path.",
    )
    parser.add_argument(
        "--robustness",
        nargs="?",
        const=str(ROBUSTNESS_CONFIG),
        help="Run deterministic parameter robustness analysis using configs/robustness.yml or a provided path.",
    )
    parser.add_argument(
        "--execution-delay",
        type=int,
        help="Override execution delay in bars.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        help="Override deterministic transaction cost in basis points.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        help="Override deterministic slippage in basis points.",
    )
    parser.add_argument(
        "--execution-enabled",
        action="store_true",
        help="Enable execution frictions even when config defaults are disabled.",
    )
    parser.add_argument(
        "--disable-execution-model",
        action="store_true",
        help="Disable transaction-cost and slippage frictions for this run.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict research-validity enforcement and block artifact or registry writes on flagged runs.",
    )
    parser.add_argument(
        "--simulation",
        help="Optional simulation config path for deterministic bootstrap/Monte Carlo return analysis.",
    )
    parser.add_argument(
        "--promotion-gates",
        help="Optional YAML/JSON promotion gate config override.",
    )
    return parser.parse_args(argv)


def load_strategies_config(path: Path = STRATEGIES_CONFIG) -> dict[str, dict[str, Any]]:
    """Load the strategy registry YAML file."""

    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}

    if not isinstance(payload, dict):
        raise ValueError("Strategy configuration must be a mapping of strategy names to config dictionaries.")

    return payload


def get_strategy_config(
    strategy_name: str,
    strategies: dict[str, dict[str, Any]] | None = None,
    *,
    path: Path = STRATEGIES_CONFIG,
) -> dict[str, Any]:
    """Return the requested strategy configuration or raise a clear validation error."""

    registry = strategies or load_strategies_config(path)
    if strategy_name not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Available strategies: {available}."
        )

    config = registry[strategy_name]
    if not isinstance(config, dict):
        raise ValueError(f"Strategy '{strategy_name}' configuration must be a dictionary.")

    return config


def run_strategy_experiment(
    strategy_name: str,
    *,
    start: str | None = None,
    end: str | None = None,
    runtime_config: RuntimeConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    strict: bool = False,
    simulation_config: dict[str, Any] | None = None,
    strategies_config_path: Path = STRATEGIES_CONFIG,
) -> StrategyRunResult:
    """Run the full strategy research pipeline and persist experiment artifacts."""

    config = get_strategy_config(strategy_name, path=strategies_config_path)
    strategy = build_strategy(strategy_name, config)
    dataset = load_features(strategy.dataset, start=start, end=end)
    signal_frame = generate_signals(dataset, strategy)
    resolved_runtime = runtime_config or resolve_runtime_config(
        config,
        cli_overrides=None if execution_config is None else {"execution": execution_config.to_dict()},
        cli_strict=strict,
    )
    results_df = run_backtest(
        signal_frame,
        resolved_runtime.execution,
        require_managed_signals=True,
    )
    results_df.attrs["dataset"] = strategy.dataset
    metrics = compute_metrics(results_df)
    metrics.update(_compute_benchmark_metrics(results_df, dataset, strategy.dataset, resolved_runtime.execution))
    try:
        sanity_report = validate_strategy_backtest_sanity(results_df, metrics, resolved_runtime.sanity)
    except SanityCheckError as exc:
        raise_research_validation_error(
            validator="sanity",
            scope=f"strategy:{strategy_name}",
            exc=exc,
            strict_mode=resolved_runtime.strict_mode.enabled,
        )
    metrics = sanity_report.apply_to_metrics(metrics)
    results_df.attrs["sanity_check"] = sanity_report.to_dict()
    results_df.attrs["runtime_config"] = resolved_runtime.to_dict()
    signal_diagnostics = compute_signal_diagnostics(results_df["signal"], results_df)
    resolved_simulation = resolve_simulation_config(simulation_config, base=config.get("simulation"))

    experiment_payload = {
        "strategy_name": strategy_name,
        "dataset": strategy.dataset,
        "parameters": dict(config.get("parameters", {})),
        "start": start,
        "end": end,
    }
    if config.get("promotion_gates") is not None:
        experiment_payload["promotion_gates"] = config.get("promotion_gates")
    experiment_config = resolved_runtime.apply_to_payload(
        experiment_payload,
        include_validation_section=False,
    )
    if resolved_simulation is not None:
        experiment_config["simulation"] = resolved_simulation.to_dict()
    experiment_dir = save_experiment(strategy_name, results_df, metrics, experiment_config)
    qa_summary = generate_strategy_qa_summary(
        results_df,
        results_df["signal"],
        signal_diagnostics,
        metrics,
        strategy_name=strategy_name,
        run_id=experiment_dir.name,
    )
    simulation_result = None
    if resolved_simulation is not None:
        aggregated_returns = aggregate_strategy_returns(results_df)["strategy_return"]
        simulation_result = run_return_simulation(
            aggregated_returns,
            config=resolved_simulation,
            periods_per_year=infer_periods_per_year(aggregate_strategy_returns(results_df)),
            owner=f"strategy {strategy_name} returns",
        )
        write_simulation_artifacts(
            experiment_dir / "simulation",
            simulation_result,
            parent_manifest_dir=experiment_dir,
        )

    return StrategyRunResult(
        strategy_name=strategy_name,
        run_id=experiment_dir.name,
        metrics=metrics,
        experiment_dir=experiment_dir,
        results_df=results_df,
        signal_diagnostics=signal_diagnostics,
        qa_summary=qa_summary,
        simulation_result=simulation_result,
    )


def print_summary(result: StrategyRunResult | WalkForwardRunResult | RobustnessRunResult) -> None:
    """Print a concise experiment summary for CLI users."""

    if isinstance(result, RobustnessRunResult):
        print(f"strategy: {result.strategy_name}")
        print(f"run_id: {result.run_id}")
        print(f"variant_count: {result.summary['variant_count']}")
        print(f"ranking_metric: {result.summary['ranking_metric']}")
        print(f"best_variant: {result.summary['best_variant_id']}")
        print(f"best_metric_value: {result.summary['best_metric_value']:.6f}")
        metric_spread = result.summary.get("metric_spread")
        if metric_spread is not None:
            print(f"metric_spread: {float(metric_spread):.6f}")
        split_count = result.summary.get("split_count")
        if split_count:
            print(f"split_count: {int(split_count)}")
        threshold_pass_rate = result.summary.get("threshold_pass_rate")
        if threshold_pass_rate is not None:
            print(f"threshold_pass_rate: {float(threshold_pass_rate):.2%}")
        statistical_validity = result.summary.get("statistical_validity")
        if isinstance(statistical_validity, dict):
            correction_method = statistical_validity.get("correction_method_used")
            if correction_method is not None:
                print(f"validity_correction: {correction_method}")
            validity_method = statistical_validity.get("validity_ranking_method_used")
            if validity_method is not None:
                print(f"validity_ranking: {validity_method}")
            unavailable_reason = statistical_validity.get("validity_ranking_unavailable_reason")
            if unavailable_reason:
                print(f"validity_note: {unavailable_reason}")
        return

    print(f"strategy: {result.strategy_name}")
    print(f"run_id: {result.run_id}")
    if isinstance(result, WalkForwardRunResult):
        print(f"split_count: {result.aggregate_summary['split_count']}")
    print(f"cumulative_return: {result.metrics['cumulative_return']:.6f}")
    print(f"sharpe_ratio: {result.metrics['sharpe_ratio']:.6f}")
    diagnostics = getattr(result, "signal_diagnostics", None)
    if isinstance(diagnostics, dict) and diagnostics:
        print("Signal diagnostics:")
        print(
            f"- long: {diagnostics['pct_long']:.0%} | "
            f"short: {diagnostics['pct_short']:.0%} | "
            f"flat: {diagnostics['pct_flat']:.0%}"
        )
        print(
            f"- trades: {diagnostics['total_trades']} | "
            f"turnover: {diagnostics['turnover']:.2f}"
        )
        print(f"- avg holding: {diagnostics['avg_holding_period']:.1f} bars")
    qa_summary = getattr(result, "qa_summary", None)
    if isinstance(qa_summary, dict) and qa_summary:
        print("QA Summary:")
        print(f"- status: {str(qa_summary['overall_status']).upper()}")
        print(
            f"- rows: {qa_summary['row_count']:,} | "
            f"symbols: {qa_summary['symbols_present']}"
        )
        print(
            f"- trades: {qa_summary['signal']['total_trades']} | "
            f"turnover: {qa_summary['signal']['turnover']:.2f}"
        )
        _print_benchmark_summary(result.metrics)
        warnings_list = summarize_qa_warnings(qa_summary)
        if warnings_list:
            print("Warnings:")
            for warning in warnings_list:
                print(f"- {warning}")
    simulation_result = getattr(result, "simulation_result", None)
    if simulation_result is not None:
        simulation_summary = simulation_result.summary
        stats = simulation_summary["metric_statistics"]["cumulative_return"]
        print("Simulation:")
        print(
            f"- method: {simulation_summary['method']} | "
            f"paths: {simulation_summary['num_paths']} | "
            f"loss_prob: {simulation_summary['probability_of_loss']:.0%}"
        )
        print(
            f"- mean cumulative return: {stats['mean']:.6f} | "
            f"median: {stats['median']:.6f} | "
            f"p05: {stats['p05']:.6f}"
        )


def summarize_qa_warnings(qa_summary: dict[str, Any]) -> list[str]:
    flags = qa_summary.get("flags")
    if not isinstance(flags, dict):
        return []

    warnings_list: list[str] = []
    if bool(flags.get("low_data")):
        warnings_list.append("insufficient data for a high-confidence analysis")
    if bool(flags.get("no_trades")):
        warnings_list.append("no trades were generated")
    if bool(flags.get("high_benchmark_correlation")):
        correlation = qa_summary.get("relative", {}).get("correlation")
        if correlation is None:
            warnings_list.append("strategy is highly correlated with the benchmark")
        else:
            warnings_list.append(f"strategy is highly correlated with the benchmark ({float(correlation):.2f})")
    if bool(flags.get("low_excess_return")):
        warnings_list.append("strategy delivered little excess return versus buy and hold")
    if bool(flags.get("high_turnover_low_edge")):
        warnings_list.append("strategy turnover is high relative to its excess return")
    if bool(flags.get("beta_dominated_strategy")):
        warnings_list.append("strategy returns appear largely benchmark-driven")
    sanity = qa_summary.get("sanity")
    if isinstance(sanity, dict):
        for issue in sanity.get("issues", []):
            if isinstance(issue, dict) and issue.get("message"):
                warnings_list.append(str(issue["message"]))
    return warnings_list


def _compute_benchmark_metrics(
    results_df: pd.DataFrame,
    dataset: pd.DataFrame,
    strategy_dataset: str,
    execution_config: ExecutionConfig,
) -> dict[str, float | dict[str, bool]]:
    benchmark_strategy = build_strategy("buy_and_hold_v1", {"dataset": strategy_dataset, "parameters": {}})
    benchmark_signal_frame = generate_signals(dataset, benchmark_strategy)
    benchmark_results = run_backtest(
        benchmark_signal_frame,
        execution_config,
        require_managed_signals=True,
    )
    return compute_benchmark_relative_metrics(results_df, benchmark_results)


def _print_benchmark_summary(metrics: dict[str, Any]) -> None:
    benchmark_return = metrics.get("benchmark_total_return")
    excess_return = metrics.get("excess_return")
    correlation = metrics.get("benchmark_correlation")
    if benchmark_return is None and excess_return is None and correlation is None:
        return

    print("Benchmark comparison:")
    print(f"- benchmark return: {_format_pct(benchmark_return)}")
    print(f"- excess return: {_format_signed_pct(excess_return)}")
    print(f"- correlation: {_format_decimal(correlation)}")


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0%}"


def _format_signed_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.0%}"


def _format_decimal(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    state: dict[str, Any] | None = None,
    pipeline_context: dict[str, Any] | None = None,
) -> StrategyRunResult | WalkForwardRunResult | RobustnessRunResult | dict[str, Any]:
    """Execute the strategy runner CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    from src.execution.strategy import run_strategy_from_cli_args

    execution_result = run_strategy_from_cli_args(args)
    result = execution_result.raw_result
    print_summary(result)
    if pipeline_context is not None:
        strategy_name = str(getattr(result, "strategy_name", args.strategy or "strategy"))
        artifact_dir = getattr(result, "experiment_dir", None)
        metrics = getattr(result, "metrics", None)
        output_paths = {}
        state_updates: dict[str, Any] = {}
        if artifact_dir is not None:
            output_paths = {
                "manifest_json": Path(artifact_dir) / "manifest.json",
                "metrics_json": Path(artifact_dir) / "metrics.json",
                "qa_summary_json": Path(artifact_dir) / "qa_summary.json",
                "equity_curve_csv": Path(artifact_dir) / "equity_curve.csv",
            }
            ranked_configs = Path(artifact_dir) / "ranked_configs.csv"
            if ranked_configs.exists():
                state_updates["sweep_artifact_dir"] = Path(artifact_dir).as_posix()
                state_updates["sweep_ranked_configs_csv"] = ranked_configs.as_posix()
        return build_pipeline_cli_result(
            identifier=str(getattr(result, "run_id", strategy_name)),
            name=str(getattr(result, "strategy_name", strategy_name)),
            artifact_dir=artifact_dir,
            manifest_path=(
                None
                if artifact_dir is None
                else Path(artifact_dir) / "manifest.json"
            ),
            output_paths=output_paths,
            metrics=(metrics if isinstance(metrics, dict) else None),
            extra={
                "strategy_name": str(getattr(result, "strategy_name", strategy_name)),
                "result_type": result.__class__.__name__,
            },
            state_updates=state_updates,
        )
    return result


def _execution_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    override: dict[str, Any] = {}
    if args.execution_delay is not None:
        override["execution_delay"] = args.execution_delay
    if args.transaction_cost_bps is not None:
        override["transaction_cost_bps"] = args.transaction_cost_bps
    if args.slippage_bps is not None:
        override["slippage_bps"] = args.slippage_bps
    if args.execution_enabled:
        override["enabled"] = True
    if args.disable_execution_model:
        override["enabled"] = False
    return override or None


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    try:
        run_cli()
    except (ResearchStrictModeError, StrategyInputError, ValueError) as exc:
        print(_format_run_failure(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def _format_run_failure(exc: Exception) -> str:
    message = str(exc).strip()
    if message.startswith("Run failed:"):
        return message
    return f"Run failed: {message}"


if __name__ == "__main__":
    main()
