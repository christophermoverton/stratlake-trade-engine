from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from src.config.execution import ExecutionConfig
from src.config.simulation import load_simulation_config, resolve_simulation_config
from src.config.runtime import resolve_runtime_config
from src.portfolio import (
    EqualWeightAllocator,
    OptimizerAllocator,
    SUPPORTED_PORTFOLIO_OPTIMIZERS,
    build_aligned_return_matrix,
    compute_portfolio_metrics,
    construct_portfolio,
    load_strategy_runs_returns,
    register_validated_portfolio_run,
    run_portfolio_walk_forward,
    validate_portfolio_config,
)
from src.research import experiment_tracker
from src.research.metrics import MINUTE_PERIODS_PER_YEAR, TRADING_DAYS_PER_YEAR
from src.research.registry import (
    STRATEGY_RUN_TYPE,
    default_registry_path,
    filter_by_run_type,
    generate_portfolio_run_id,
    load_registry,
)
from src.research.simulation import SimulationRunResult, run_return_simulation, write_simulation_artifacts
from src.research.strict_mode import ResearchStrictModeError, raise_research_validation_error
from src.research.sanity import validate_portfolio_output_sanity

DEFAULT_PORTFOLIO_ARTIFACTS_ROOT = Path("artifacts") / "portfolios"
SUPPORTED_TIMEFRAMES = ("1D", "1Min")
DEFAULT_ALLOCATOR = "equal_weight"
DEFAULT_ALIGNMENT_POLICY = "intersection"
DEFAULT_INITIAL_CAPITAL = 1.0

# Backward-compatible alias for tests and callers that monkeypatch the portfolio
# registry writer at the CLI module boundary.
register_portfolio_run = register_validated_portfolio_run


@dataclass(frozen=True)
class PortfolioRunResult:
    portfolio_name: str
    run_id: str
    allocator_name: str
    timeframe: str
    component_count: int
    metrics: dict[str, float | None]
    experiment_dir: Path
    portfolio_output: pd.DataFrame
    config: dict[str, Any]
    components: list[dict[str, Any]]
    simulation_result: SimulationRunResult | None = None


@dataclass(frozen=True)
class PortfolioWalkForwardRunResult:
    portfolio_name: str
    run_id: str
    allocator_name: str
    timeframe: str
    component_count: int
    split_count: int
    metrics: dict[str, float | None]
    aggregate_metrics: dict[str, Any]
    experiment_dir: Path
    config: dict[str, Any]
    components: list[dict[str, Any]]
    simulation_result: SimulationRunResult | None = None


@dataclass(frozen=True)
class PortfolioCliOverrides:
    optimizer: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a portfolio construction run."""

    parser = argparse.ArgumentParser(
        description="Build and evaluate a deterministic portfolio from completed strategy runs."
    )
    parser.add_argument(
        "--portfolio-config",
        help="Optional YAML/JSON file describing one or more portfolio definitions.",
    )
    parser.add_argument(
        "--portfolio-name",
        help="Portfolio name to resolve from config, or the required name when using --run-ids.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="Explicit strategy run ids. Accepts comma-separated and/or space-separated values.",
    )
    parser.add_argument(
        "--from-registry",
        action="store_true",
        help="Select the latest matching run per configured strategy from the strategy registry.",
    )
    parser.add_argument(
        "--evaluation",
        help="Optional evaluation config path. When provided, runs portfolio walk-forward evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional portfolio artifact root directory. "
            "The deterministic run id will be appended."
        ),
    )
    parser.add_argument(
        "--timeframe",
        required=True,
        help="Portfolio metrics timeframe. Supported values: 1D, 1Min.",
    )
    parser.add_argument(
        "--optimizer-method",
        choices=SUPPORTED_PORTFOLIO_OPTIMIZERS,
        help=(
            "Override the effective portfolio allocator/optimizer method for this run. "
            "Use equal_weight to keep deterministic equal-weight construction."
        ),
    )
    parser.add_argument(
        "--risk-target-volatility",
        type=float,
        help="Override the effective portfolio risk target volatility.",
    )
    parser.add_argument(
        "--risk-volatility-window",
        type=int,
        help="Override the rolling volatility window used by portfolio risk diagnostics.",
    )
    parser.add_argument(
        "--risk-var-confidence-level",
        type=float,
        help="Override the portfolio historical VaR confidence level.",
    )
    parser.add_argument(
        "--risk-cvar-confidence-level",
        type=float,
        help="Override the portfolio historical CVaR confidence level.",
    )
    parser.add_argument(
        "--risk-allow-scale-up",
        action="store_true",
        help="Allow portfolio volatility targeting diagnostics to recommend leverage scaling above 1x.",
    )
    parser.add_argument(
        "--risk-max-volatility-scale",
        type=float,
        help="Override the portfolio volatility scaling cap used by risk diagnostics.",
    )
    parser.add_argument(
        "--execution-delay",
        type=int,
        help="Override execution delay in bars for strategy/portfolio execution settings.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        help="Override deterministic execution transaction cost in basis points.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        help="Override deterministic execution slippage in basis points.",
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
    return parser.parse_args(argv)


def parse_run_ids(raw_values: Sequence[str] | None) -> list[str]:
    """Return normalized run ids from comma-separated and/or repeated CLI values."""

    if raw_values is None:
        return []

    run_ids: list[str] = []
    for raw_value in raw_values:
        for run_id in raw_value.split(","):
            normalized = run_id.strip()
            if normalized:
                run_ids.append(normalized)
    return run_ids


def run_cli(argv: Sequence[str] | None = None) -> PortfolioRunResult | PortfolioWalkForwardRunResult:
    """Execute the portfolio runner CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    timeframe = _normalize_timeframe(args.timeframe)
    execution_override = _execution_override_from_args(args)
    cli_overrides = _milestone_11_overrides_from_args(args)
    _validate_cli_args(args, timeframe=timeframe)

    resolved_config, run_dirs, components = _resolve_portfolio_inputs(
        portfolio_config_path=(
            None if args.portfolio_config is None else Path(args.portfolio_config)
        ),
        portfolio_name=args.portfolio_name,
        explicit_run_ids=parse_run_ids(args.run_ids),
        from_registry=bool(args.from_registry),
        timeframe=timeframe,
        evaluation_path=None if args.evaluation is None else Path(args.evaluation),
        execution_override=execution_override,
        cli_overrides=cli_overrides,
    )
    resolved_simulation = resolve_simulation_config(
        None if args.simulation is None else load_simulation_config(Path(args.simulation)).to_dict(),
        base=resolved_config.get("simulation"),
    )
    runtime_config = resolve_runtime_config(
        resolved_config,
        cli_strict=args.strict,
    )
    resolved_config = runtime_config.apply_to_payload(dict(resolved_config), validation_key="validation")
    if resolved_simulation is not None:
        resolved_config["simulation"] = resolved_simulation.to_dict()

    allocator = _build_allocator(
        resolved_config["allocator"],
        optimizer_config=resolved_config.get("optimizer"),
    )
    output_root = DEFAULT_PORTFOLIO_ARTIFACTS_ROOT if args.output_dir is None else Path(args.output_dir)

    if args.evaluation is not None:
        if resolved_simulation is not None:
            raise ValueError("The --simulation argument cannot be combined with --evaluation.")
        walk_forward_result = run_portfolio_walk_forward(
            component_run_ids=[str(component["run_id"]) for component in components],
            evaluation_config_path=Path(args.evaluation),
            allocator=allocator,
            timeframe=timeframe,
            output_dir=output_root,
            portfolio_name=str(resolved_config["portfolio_name"]),
            initial_capital=float(resolved_config["initial_capital"]),
            alignment_policy=str(resolved_config["alignment_policy"]),
            execution_config=runtime_config.execution,
            validation_config=runtime_config.portfolio_validation.to_dict(),
            risk_config=runtime_config.risk.to_dict(),
            sanity_config=runtime_config.sanity.to_dict(),
            strict_mode=runtime_config.strict_mode.enabled,
        )
        result = PortfolioWalkForwardRunResult(
            portfolio_name=str(walk_forward_result["portfolio_name"]),
            run_id=str(walk_forward_result["run_id"]),
            allocator_name=str(walk_forward_result["allocator_name"]),
            timeframe=str(walk_forward_result["timeframe"]),
            component_count=int(walk_forward_result["component_count"]),
            split_count=int(walk_forward_result["split_count"]),
            metrics=dict(walk_forward_result["metrics"]),
            aggregate_metrics=dict(walk_forward_result["aggregate_metrics"]),
            experiment_dir=Path(walk_forward_result["experiment_dir"]),
            config=dict(walk_forward_result["config"]),
            components=[dict(component) for component in walk_forward_result["components"]],
            simulation_result=None,
        )
    else:
        strategy_returns = load_strategy_runs_returns(run_dirs)
        aligned_returns = build_aligned_return_matrix(strategy_returns)
        try:
            portfolio_output = construct_portfolio(
                aligned_returns,
                allocator,
                initial_capital=float(resolved_config["initial_capital"]),
                execution_config=runtime_config.execution,
                validation_config=runtime_config.portfolio_validation,
                optimization_returns=aligned_returns,
            )
        except ValueError as exc:
            raise_research_validation_error(
                validator="portfolio_validation",
                scope=f"portfolio:{resolved_config['portfolio_name']}",
                exc=exc,
                strict_mode=runtime_config.strict_mode.enabled,
            )
        metrics = compute_portfolio_metrics(
            portfolio_output,
            timeframe,
            validation_config=runtime_config.portfolio_validation,
            risk_config=runtime_config.risk,
        )
        try:
            sanity_report = validate_portfolio_output_sanity(
                portfolio_output,
                metrics,
                runtime_config.sanity,
                initial_capital=float(resolved_config["initial_capital"]),
            )
        except ValueError as exc:
            raise_research_validation_error(
                validator="sanity",
                scope=f"portfolio:{resolved_config['portfolio_name']}",
                exc=exc,
                strict_mode=runtime_config.strict_mode.enabled,
            )
        metrics = sanity_report.apply_to_metrics(metrics)
        portfolio_output.attrs["sanity_check"] = sanity_report.to_dict()

        start_ts = _format_timestamp(aligned_returns.index.min())
        end_ts = _format_timestamp(aligned_returns.index.max())
        run_id = generate_portfolio_run_id(
            portfolio_name=str(resolved_config["portfolio_name"]),
            allocator_name=allocator.name,
            component_run_ids=[str(component["run_id"]) for component in components],
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            config=resolved_config,
            evaluation_config_path=None if args.evaluation is None else Path(args.evaluation),
        )
        experiment_dir = output_root / run_id

        from src.portfolio import write_portfolio_artifacts

        manifest = write_portfolio_artifacts(
            output_dir=experiment_dir,
            portfolio_output=portfolio_output,
            metrics=metrics,
            config=resolved_config,
            components=components,
        )
        simulation_result = None
        if resolved_simulation is not None:
            simulation_result = run_return_simulation(
                portfolio_output["portfolio_return"],
                config=resolved_simulation,
                periods_per_year=_periods_per_year_from_timeframe(timeframe),
                owner=f"portfolio {resolved_config['portfolio_name']} returns",
                var_confidence_level=float(runtime_config.risk.var_confidence_level),
                cvar_confidence_level=float(runtime_config.risk.cvar_confidence_level),
            )
            write_simulation_artifacts(
                experiment_dir / "simulation",
                simulation_result,
                parent_manifest_dir=experiment_dir,
            )
            manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))

        registry_path = default_registry_path(output_root)
        register_portfolio_run(
            registry_path=registry_path,
            run_id=run_id,
            config=resolved_config,
            components=components,
            metrics=metrics,
            artifact_path=experiment_dir.as_posix(),
            manifest=manifest,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        result = PortfolioRunResult(
            portfolio_name=str(resolved_config["portfolio_name"]),
            run_id=run_id,
            allocator_name=allocator.name,
            timeframe=timeframe,
            component_count=len(components),
            metrics=metrics,
            experiment_dir=experiment_dir,
            portfolio_output=portfolio_output,
            config=resolved_config,
            components=components,
            simulation_result=simulation_result,
        )
    print_summary(result)
    return result


def print_summary(result: PortfolioRunResult | PortfolioWalkForwardRunResult) -> None:
    """Print a concise deterministic portfolio run summary."""

    print(f"Portfolio: {result.portfolio_name}")
    print(f"Run ID: {result.run_id}")
    print(f"Artifact Dir: {result.experiment_dir.as_posix()}")
    print(f"Allocator: {result.allocator_name}")
    print(f"Optimizer Method: {_summary_optimizer_method(result.config, fallback=result.allocator_name)}")
    print(f"Components: {result.component_count} strategies")
    print(f"Timeframe: {result.timeframe}")
    if isinstance(result, PortfolioWalkForwardRunResult):
        stats = result.aggregate_metrics["metric_statistics"]
        print(f"Splits: {result.split_count}")
        print()
        print(f"Mean Total Return: {_format_pct(_nested_metric_stat(stats, 'total_return', 'mean'))}")
        print(f"Mean Sharpe Ratio: {_format_decimal(_nested_metric_stat(stats, 'sharpe_ratio', 'mean'))}")
        print(f"Mean Realized Volatility: {_format_pct(_nested_metric_stat(stats, 'realized_volatility', 'mean'))}")
        print(f"Worst Max Drawdown: {_format_pct(_nested_metric_stat(stats, 'max_drawdown', 'min'))}")
        print(f"Mean VaR: {_format_pct(_nested_metric_stat(stats, 'value_at_risk', 'mean'))}")
        print(f"Mean CVaR: {_format_pct(_nested_metric_stat(stats, 'conditional_value_at_risk', 'mean'))}")
        print("Simulation: disabled")
        return
    print()
    print(f"Total Return: {_format_pct(result.metrics.get('total_return'))}")
    print(f"Sharpe Ratio: {_format_decimal(result.metrics.get('sharpe_ratio'))}")
    print(f"Realized Volatility: {_format_pct(result.metrics.get('realized_volatility'))}")
    print(f"Max Drawdown: {_format_pct(result.metrics.get('max_drawdown'))}")
    print(f"VaR: {_format_pct(result.metrics.get('value_at_risk'))}")
    print(f"CVaR: {_format_pct(result.metrics.get('conditional_value_at_risk'))}")
    if result.simulation_result is not None:
        simulation_summary = result.simulation_result.summary
        stats = simulation_summary["metric_statistics"]["cumulative_return"]
        print()
        print(
            f"Simulation: {simulation_summary['method']} | "
            f"Paths: {simulation_summary['num_paths']} | "
            f"Loss Prob: {simulation_summary['probability_of_loss']:.2%}"
        )
        print(f"Mean Sim Return: {_format_pct(stats['mean'])}")
        print(f"Median Sim Return: {_format_pct(stats['median'])}")
        print(f"P05 Sim Return: {_format_pct(stats['p05'])}")
    else:
        print("Simulation: disabled")


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    try:
        run_cli()
    except (ResearchStrictModeError, ValueError) as exc:
        print(_format_run_failure(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def _validate_cli_args(args: argparse.Namespace, *, timeframe: str) -> None:
    has_config = args.portfolio_config is not None
    has_run_ids = bool(parse_run_ids(args.run_ids))
    if has_config == has_run_ids:
        raise ValueError("Provide exactly one of --portfolio-config or --run-ids.")
    if args.from_registry and has_run_ids:
        raise ValueError("The --run-ids argument cannot be combined with --from-registry.")
    if args.from_registry and not has_config:
        raise ValueError("The --from-registry flag requires --portfolio-config.")
    if args.portfolio_name and not has_config and not has_run_ids:
        raise ValueError("The --portfolio-name argument requires --portfolio-config or --run-ids.")
    if has_run_ids and not args.portfolio_name:
        raise ValueError("The --portfolio-name argument is required when using --run-ids.")
    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(SUPPORTED_TIMEFRAMES)
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Supported values: {supported}.")
    if args.execution_enabled and args.disable_execution_model:
        raise ValueError(
            "The --execution-enabled and --disable-execution-model arguments are mutually exclusive."
        )
    if args.simulation is not None and args.evaluation is not None:
        raise ValueError("The --simulation argument cannot be combined with --evaluation.")
    if args.risk_allow_scale_up and args.risk_max_volatility_scale is not None:
        if float(args.risk_max_volatility_scale) <= 1.0:
            raise ValueError(
                "The --risk-allow-scale-up flag requires --risk-max-volatility-scale > 1.0 when provided."
            )


def _resolve_portfolio_inputs(
    *,
    portfolio_config_path: Path | None,
    portfolio_name: str | None,
    explicit_run_ids: Sequence[str],
    from_registry: bool,
    timeframe: str,
    evaluation_path: Path | None,
    execution_override: Mapping[str, Any] | None,
    cli_overrides: PortfolioCliOverrides | None,
) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]]]:
    if portfolio_config_path is not None:
        raw_payload = load_portfolio_config(portfolio_config_path)
        raw_definition = resolve_portfolio_definition(raw_payload, portfolio_name=portfolio_name)
        base_definition = _normalize_portfolio_definition(
            raw_definition,
            portfolio_name=portfolio_name,
        )
        if from_registry:
            components = _resolve_registry_components(
                base_definition["components"],
                timeframe=timeframe,
            )
        else:
            components = _resolve_config_components(base_definition["components"])
    else:
        components = _resolve_explicit_run_id_components(explicit_run_ids)
        base_definition = {
            "portfolio_name": _normalize_required_string(
                portfolio_name,
                field_name="portfolio_name",
            ),
            "allocator": DEFAULT_ALLOCATOR,
            "components": components,
            "initial_capital": DEFAULT_INITIAL_CAPITAL,
            "alignment_policy": DEFAULT_ALIGNMENT_POLICY,
        }

    base_definition = _apply_cli_portfolio_overrides(base_definition, cli_overrides)
    runtime_override = _compose_runtime_override(
        execution_override=execution_override,
        milestone_11_overrides=cli_overrides,
    )
    runtime_config = resolve_runtime_config(base_definition, cli_overrides=runtime_override)
    resolved_config = runtime_config.apply_to_payload(
        {
        **base_definition,
        "portfolio_name": _normalize_required_string(
            base_definition.get("portfolio_name"),
            field_name="portfolio_name",
        ),
        "allocator": _normalize_required_string(
            base_definition.get("allocator", DEFAULT_ALLOCATOR),
            field_name="allocator",
        ),
        "components": components,
        "initial_capital": float(base_definition.get("initial_capital", DEFAULT_INITIAL_CAPITAL)),
        "alignment_policy": _normalize_required_string(
            base_definition.get("alignment_policy", DEFAULT_ALIGNMENT_POLICY),
            field_name="alignment_policy",
        ),
        "optimizer": base_definition.get("optimizer"),
        "timeframe": timeframe,
        "evaluation_config_path": None if evaluation_path is None else evaluation_path.as_posix(),
        },
        validation_key="validation",
    )
    validated_config = validate_portfolio_config(resolved_config)
    runtime_config = resolve_runtime_config(validated_config, cli_overrides=runtime_override)
    validated_config = runtime_config.apply_to_payload(validated_config, validation_key="validation")
    validated_config["timeframe"] = timeframe
    validated_config["evaluation_config_path"] = (
        None if evaluation_path is None else evaluation_path.as_posix()
    )
    if base_definition.get("simulation") is not None:
        validated_config["simulation"] = dict(base_definition["simulation"])
    run_dirs = [Path(str(component["source_artifact_path"])) for component in components]
    return validated_config, run_dirs, components


def load_portfolio_config(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON portfolio configuration file."""

    if not path.exists():
        raise ValueError(f"Portfolio config file does not exist: {path}.")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        else:
            raise ValueError(
                f"Unsupported portfolio config format {path.suffix!r}. Use JSON, YAML, or YML."
            )

    if not isinstance(payload, dict):
        raise ValueError("Portfolio config file must contain a JSON/YAML object at the top level.")
    return payload


def resolve_portfolio_definition(
    payload: Mapping[str, Any],
    *,
    portfolio_name: str | None,
) -> dict[str, Any]:
    """Resolve one portfolio definition from a config payload."""

    if _looks_like_portfolio_definition(payload):
        definition = dict(payload)
        if (
            portfolio_name is not None
            and definition.get("portfolio_name") not in {None, portfolio_name}
        ):
            raise ValueError(
                f"Portfolio config does not define portfolio {portfolio_name!r}."
            )
        if definition.get("portfolio_name") is None and portfolio_name is not None:
            definition["portfolio_name"] = portfolio_name
        return definition

    if "portfolios" in payload:
        raw_portfolios = payload["portfolios"]
        if isinstance(raw_portfolios, list):
            definitions = [dict(item) for item in raw_portfolios if isinstance(item, dict)]
            if len(definitions) != len(raw_portfolios):
                raise ValueError("Portfolio config field 'portfolios' must contain only objects.")
            return _select_portfolio_definition_from_list(
                definitions,
                portfolio_name=portfolio_name,
            )
        if isinstance(raw_portfolios, dict):
            return _select_portfolio_definition_from_mapping(
                raw_portfolios,
                portfolio_name=portfolio_name,
            )
        raise ValueError("Portfolio config field 'portfolios' must be a list or mapping.")

    return _select_portfolio_definition_from_mapping(payload, portfolio_name=portfolio_name)


def _select_portfolio_definition_from_mapping(
    definitions: Mapping[str, Any],
    *,
    portfolio_name: str | None,
) -> dict[str, Any]:
    eligible = {
        str(name): dict(value)
        for name, value in definitions.items()
        if isinstance(value, dict) and ("allocator" in value or "components" in value)
    }
    if not eligible:
        raise ValueError("Portfolio config does not contain any usable portfolio definitions.")

    selected_name = portfolio_name
    if selected_name is None:
        if len(eligible) != 1:
            available = ", ".join(sorted(eligible))
            raise ValueError(
                "Portfolio config contains multiple portfolio definitions. "
                f"Pass --portfolio-name. Available portfolios: {available}."
            )
        selected_name = next(iter(sorted(eligible)))

    if selected_name not in eligible:
        available = ", ".join(sorted(eligible))
        raise ValueError(
            f"Unknown portfolio '{selected_name}'. Available portfolios: {available}."
        )

    definition = dict(eligible[selected_name])
    definition.setdefault("portfolio_name", selected_name)
    return definition


def _select_portfolio_definition_from_list(
    definitions: Sequence[dict[str, Any]],
    *,
    portfolio_name: str | None,
) -> dict[str, Any]:
    named = {
        str(definition["portfolio_name"]): dict(definition)
        for definition in definitions
        if definition.get("portfolio_name") is not None
    }
    if portfolio_name is None:
        if len(definitions) != 1:
            available = ", ".join(sorted(named)) or "<unnamed>"
            raise ValueError(
                "Portfolio config contains multiple portfolio definitions. "
                f"Pass --portfolio-name. Available portfolios: {available}."
            )
        return dict(definitions[0])
    if portfolio_name not in named:
        available = ", ".join(sorted(named)) or "<none>"
        raise ValueError(
            f"Unknown portfolio '{portfolio_name}'. Available portfolios: {available}."
        )
    return named[portfolio_name]


def _normalize_portfolio_definition(
    definition: Mapping[str, Any],
    *,
    portfolio_name: str | None,
) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise ValueError("Portfolio definition must be a mapping.")

    components = definition.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Portfolio definition must include a non-empty 'components' list.")

    normalized_components: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(f"Portfolio component at index {index} must be a mapping.")
        normalized_components.append(dict(component))

    return {
        "portfolio_name": definition.get("portfolio_name", portfolio_name),
        "allocator": definition.get("allocator", DEFAULT_ALLOCATOR),
        "components": normalized_components,
        "initial_capital": definition.get("initial_capital", DEFAULT_INITIAL_CAPITAL),
        "alignment_policy": definition.get("alignment_policy", DEFAULT_ALIGNMENT_POLICY),
        "optimizer": definition.get("optimizer"),
        "execution": definition.get("execution"),
        "validation": definition.get("validation"),
        "risk": definition.get("risk"),
        "sanity": definition.get("sanity"),
        "simulation": definition.get("simulation"),
    }


def _resolve_config_components(raw_components: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for index, component in enumerate(raw_components):
        if "run_id" not in component:
            raise ValueError(
                "Portfolio config components must include 'run_id' "
                "when --from-registry is not used. "
                f"Missing at index {index}."
            )
        run_id = _normalize_required_string(
            component.get("run_id"),
            field_name=f"components[{index}].run_id",
        )
        resolved.append(
            _resolve_run_component(
                run_id,
                expected_strategy_name=component.get("strategy_name"),
            )
        )
    return _sorted_components(resolved)


def _resolve_explicit_run_id_components(run_ids: Sequence[str]) -> list[dict[str, Any]]:
    resolved = [
        _resolve_run_component(
            _normalize_required_string(run_id, field_name="run_id"),
            expected_strategy_name=None,
        )
        for run_id in run_ids
    ]
    return _sorted_components(resolved)


def _resolve_registry_components(
    raw_components: Sequence[Mapping[str, Any]],
    *,
    timeframe: str,
) -> list[dict[str, Any]]:
    entries = filter_by_run_type(
        load_registry(default_registry_path(experiment_tracker.ARTIFACTS_ROOT)),
        STRATEGY_RUN_TYPE,
    )
    resolved: list[dict[str, Any]] = []
    for index, component in enumerate(raw_components):
        if "run_id" in component:
            raise ValueError(
                "Portfolio config components cannot include explicit 'run_id' "
                "when --from-registry is used. "
                f"Found at index {index}."
            )
        strategy_name = _normalize_required_string(
            component.get("strategy_name"),
            field_name=f"components[{index}].strategy_name",
        )
        candidates = [
            entry
            for entry in entries
            if entry.get("strategy_name") == strategy_name and entry.get("timeframe") == timeframe
        ]
        if not candidates:
            raise ValueError(
                f"No registry runs found for strategy '{strategy_name}' "
                f"with timeframe '{timeframe}'."
            )
        selected = max(
            candidates,
            key=lambda entry: (str(entry.get("timestamp") or ""), str(entry.get("run_id") or "")),
        )
        resolved.append(
            _resolve_run_component(
                _normalize_required_string(selected.get("run_id"), field_name="run_id"),
                expected_strategy_name=strategy_name,
            )
        )
    return _sorted_components(resolved)


def _resolve_run_component(run_id: str, *, expected_strategy_name: object | None) -> dict[str, Any]:
    registry_entry = _find_registry_entry_for_run_id(run_id)
    run_dir = (
        Path(str(registry_entry["artifact_path"]))
        if registry_entry is not None and registry_entry.get("artifact_path") is not None
        else experiment_tracker.ARTIFACTS_ROOT / run_id
    )
    if not run_dir.exists():
        raise ValueError(f"Strategy run '{run_id}' could not be resolved to an artifact directory.")

    loader_frame = load_strategy_runs_returns([run_dir])
    strategy_names = sorted(loader_frame["strategy_name"].astype("string").unique().tolist())
    if len(strategy_names) != 1:
        raise ValueError(
            f"Strategy run '{run_id}' resolved to an invalid component strategy set: "
            f"{strategy_names}."
        )
    strategy_name = str(strategy_names[0])
    if expected_strategy_name is not None and str(expected_strategy_name).strip() != strategy_name:
        raise ValueError(
            f"Strategy run '{run_id}' resolved to strategy '{strategy_name}', "
            f"expected '{str(expected_strategy_name).strip()}'."
        )

    return {
        "strategy_name": strategy_name,
        "run_id": run_id,
        "source_artifact_path": run_dir.as_posix(),
    }


def _find_registry_entry_for_run_id(run_id: str) -> dict[str, Any] | None:
    registry_path = default_registry_path(experiment_tracker.ARTIFACTS_ROOT)
    if not registry_path.exists():
        return None
    entries = filter_by_run_type(load_registry(registry_path), STRATEGY_RUN_TYPE)
    for entry in entries:
        if entry.get("run_id") == run_id:
            return entry
    return None


def _sorted_components(components: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(component) for component in components]
    return sorted(
        normalized,
        key=lambda component: (str(component["strategy_name"]), str(component["run_id"])),
    )


def _build_allocator(
    name: str,
    *,
    optimizer_config: Mapping[str, Any] | None = None,
) -> EqualWeightAllocator | OptimizerAllocator:
    normalized = _normalize_required_string(name, field_name="allocator").lower()
    supported = set(SUPPORTED_PORTFOLIO_OPTIMIZERS)
    if normalized not in supported:
        raise ValueError(
            f"Unsupported portfolio allocator '{name}'. Supported allocators: {', '.join(sorted(supported))}."
        )
    if normalized == DEFAULT_ALLOCATOR and optimizer_config is None:
        return EqualWeightAllocator()
    return OptimizerAllocator(optimizer_config, fallback_method=normalized)


def _looks_like_portfolio_definition(payload: Mapping[str, Any]) -> bool:
    return "allocator" in payload and "components" in payload


def _normalize_timeframe(timeframe: str) -> str:
    normalized = _normalize_required_string(timeframe, field_name="timeframe").lower()
    if normalized in {"1d", "1day", "day", "daily"}:
        return "1D"
    if normalized in {"1m", "1min", "1minute", "minute", "minutes"}:
        return "1Min"
    return timeframe


def _format_timestamp(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _periods_per_year_from_timeframe(timeframe: str) -> int:
    normalized = _normalize_required_string(timeframe, field_name="timeframe").lower()
    if normalized in {"1d", "1day", "day", "daily"}:
        return TRADING_DAYS_PER_YEAR
    if normalized in {"1m", "1min", "1minute", "minute", "minutes"}:
        return MINUTE_PERIODS_PER_YEAR
    raise ValueError(f"Unsupported simulation timeframe: {timeframe!r}.")


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _format_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2%}"


def _format_decimal(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _format_run_failure(exc: Exception) -> str:
    message = str(exc).strip()
    if message.startswith("Run failed:"):
        return message
    return f"Run failed: {message}"


def _portfolio_execution_config(config: Mapping[str, Any]) -> ExecutionConfig:
    return resolve_runtime_config(config).execution


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


def _milestone_11_overrides_from_args(args: argparse.Namespace) -> PortfolioCliOverrides | None:
    optimizer_override: dict[str, Any] | None = None
    if args.optimizer_method is not None:
        optimizer_override = {"method": str(args.optimizer_method).strip().lower()}

    risk_override: dict[str, Any] = {}
    if args.risk_target_volatility is not None:
        risk_override["target_volatility"] = float(args.risk_target_volatility)
    if args.risk_volatility_window is not None:
        risk_override["volatility_window"] = int(args.risk_volatility_window)
    if args.risk_var_confidence_level is not None:
        risk_override["var_confidence_level"] = float(args.risk_var_confidence_level)
    if args.risk_cvar_confidence_level is not None:
        risk_override["cvar_confidence_level"] = float(args.risk_cvar_confidence_level)
    if args.risk_allow_scale_up:
        risk_override["allow_scale_up"] = True
    if args.risk_max_volatility_scale is not None:
        risk_override["max_volatility_scale"] = float(args.risk_max_volatility_scale)

    runtime_override = None if not risk_override else {"risk": risk_override}
    if optimizer_override is None and runtime_override is None:
        return None
    return PortfolioCliOverrides(optimizer=optimizer_override, runtime=runtime_override)


def _apply_cli_portfolio_overrides(
    base_definition: Mapping[str, Any],
    cli_overrides: PortfolioCliOverrides | None,
) -> dict[str, Any]:
    resolved = dict(base_definition)
    if cli_overrides is None or cli_overrides.optimizer is None:
        return resolved

    optimizer_override = dict(cli_overrides.optimizer)
    optimizer_method = _normalize_required_string(
        optimizer_override["method"],
        field_name="optimizer.method",
    ).lower()
    configured_optimizer = (
        dict(resolved["optimizer"])
        if isinstance(resolved.get("optimizer"), Mapping)
        else {}
    )
    configured_optimizer["method"] = optimizer_method
    resolved["optimizer"] = configured_optimizer
    resolved["allocator"] = optimizer_method
    return resolved


def _compose_runtime_override(
    *,
    execution_override: Mapping[str, Any] | None,
    milestone_11_overrides: PortfolioCliOverrides | None,
) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if execution_override is not None:
        merged["execution"] = dict(execution_override)
    if milestone_11_overrides is not None and milestone_11_overrides.runtime is not None:
        for key, value in milestone_11_overrides.runtime.items():
            merged[key] = dict(value) if isinstance(value, Mapping) else value
    return merged or None


def _summary_optimizer_method(config: Mapping[str, Any], *, fallback: str) -> str:
    optimizer = config.get("optimizer")
    if isinstance(optimizer, Mapping) and optimizer.get("method") is not None:
        return str(optimizer["method"])
    return fallback


def _nested_metric_stat(
    stats: Mapping[str, Any],
    metric_name: str,
    stat_name: str,
) -> object:
    metric_stats = stats.get(metric_name)
    if not isinstance(metric_stats, Mapping):
        return None
    return metric_stats.get(stat_name)


if __name__ == "__main__":
    main()
