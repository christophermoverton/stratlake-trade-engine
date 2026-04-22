from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_portfolio(
    *,
    portfolio_config_path: str | Path | None = None,
    portfolio_name: str | None = None,
    run_ids: Sequence[str] | None = None,
    from_registry: bool = False,
    from_sweep_top_ranked: bool = False,
    from_candidate_selection: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    timeframe: str = "1D",
    optimizer_method: str | None = None,
    enable_volatility_targeting: bool = False,
    volatility_target_volatility: float | None = None,
    volatility_target_lookback: int | None = None,
    risk_target_volatility: float | None = None,
    risk_volatility_window: int | None = None,
    risk_var_confidence_level: float | None = None,
    risk_cvar_confidence_level: float | None = None,
    risk_allow_scale_up: bool = False,
    risk_max_volatility_scale: float | None = None,
    execution_override: Mapping[str, Any] | None = None,
    strict: bool = False,
    simulation_path: str | Path | None = None,
    promotion_gates_path: str | Path | None = None,
    state: Mapping[str, Any] | None = None,
    pipeline_context: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Run portfolio construction through the notebook-facing execution surface."""

    from src.cli import run_portfolio as cli

    normalized_timeframe = cli._normalize_timeframe(timeframe)
    _validate_portfolio_api_args(
        portfolio_config_path=portfolio_config_path,
        portfolio_name=portfolio_name,
        run_ids=run_ids,
        from_registry=from_registry,
        from_sweep_top_ranked=from_sweep_top_ranked,
        from_candidate_selection=from_candidate_selection,
        timeframe=normalized_timeframe,
        execution_override=execution_override,
        simulation_path=simulation_path,
        evaluation_path=evaluation_path,
        risk_allow_scale_up=risk_allow_scale_up,
        risk_max_volatility_scale=risk_max_volatility_scale,
    )
    cli_overrides = _portfolio_cli_overrides(
        optimizer_method=optimizer_method,
        enable_volatility_targeting=enable_volatility_targeting,
        volatility_target_volatility=volatility_target_volatility,
        volatility_target_lookback=volatility_target_lookback,
        risk_target_volatility=risk_target_volatility,
        risk_volatility_window=risk_volatility_window,
        risk_var_confidence_level=risk_var_confidence_level,
        risk_cvar_confidence_level=risk_cvar_confidence_level,
        risk_allow_scale_up=risk_allow_scale_up,
        risk_max_volatility_scale=risk_max_volatility_scale,
    )
    raw_result = _run_portfolio_resolved(
        portfolio_config_path=None if portfolio_config_path is None else Path(portfolio_config_path),
        portfolio_name=portfolio_name,
        explicit_run_ids=list(run_ids or []),
        from_registry=from_registry,
        from_sweep_top_ranked=from_sweep_top_ranked,
        from_candidate_selection=(
            None if from_candidate_selection is None else Path(from_candidate_selection)
        ),
        evaluation_path=None if evaluation_path is None else Path(evaluation_path),
        output_dir=None if output_dir is None else Path(output_dir),
        timeframe=normalized_timeframe,
        execution_override=dict(execution_override) if execution_override is not None else None,
        cli_overrides=cli_overrides,
        strict=strict,
        simulation_path=None if simulation_path is None else Path(simulation_path),
        promotion_gates_path=None if promotion_gates_path is None else Path(promotion_gates_path),
        state=state,
        pipeline_context=pipeline_context,
    )
    return _summarize_portfolio_result(raw_result)


def run_portfolio_from_cli_args(
    args: Any,
    *,
    state: Mapping[str, Any] | None = None,
    pipeline_context: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Run portfolio construction from a parsed CLI namespace."""

    from src.cli import run_portfolio as cli

    timeframe = cli._normalize_timeframe(args.timeframe)
    cli._validate_cli_args(args, timeframe=timeframe)
    return _summarize_portfolio_result(
        _run_portfolio_resolved(
            portfolio_config_path=(
                None if args.portfolio_config is None else Path(args.portfolio_config)
            ),
            portfolio_name=args.portfolio_name,
            explicit_run_ids=cli.parse_run_ids(args.run_ids),
            from_registry=bool(args.from_registry),
            from_sweep_top_ranked=bool(args.from_sweep_top_ranked),
            from_candidate_selection=(
                None
                if getattr(args, "from_candidate_selection", None) is None
                else Path(getattr(args, "from_candidate_selection"))
            ),
            evaluation_path=None if args.evaluation is None else Path(args.evaluation),
            output_dir=None if args.output_dir is None else Path(args.output_dir),
            timeframe=timeframe,
            execution_override=cli._execution_override_from_args(args),
            cli_overrides=cli._milestone_11_overrides_from_args(args),
            strict=bool(args.strict),
            simulation_path=None if args.simulation is None else Path(args.simulation),
            promotion_gates_path=(
                None if args.promotion_gates is None else Path(args.promotion_gates)
            ),
            state=state,
            pipeline_context=pipeline_context,
        )
    )


def run_portfolio_from_argv(
    argv: Sequence[str] | None = None,
    *,
    state: Mapping[str, Any] | None = None,
    pipeline_context: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    from src.cli import run_portfolio as cli

    effective_argv = cli._pipeline_argv(argv, state=state)
    return run_portfolio_from_cli_args(
        cli.parse_args(effective_argv),
        state=state,
        pipeline_context=pipeline_context,
    )


def _run_portfolio_resolved(
    *,
    portfolio_config_path: Path | None,
    portfolio_name: str | None,
    explicit_run_ids: Sequence[str],
    from_registry: bool,
    from_sweep_top_ranked: bool,
    from_candidate_selection: Path | None,
    evaluation_path: Path | None,
    output_dir: Path | None,
    timeframe: str,
    execution_override: Mapping[str, Any] | None,
    cli_overrides: Any,
    strict: bool,
    simulation_path: Path | None,
    promotion_gates_path: Path | None,
    state: Mapping[str, Any] | None,
    pipeline_context: Mapping[str, Any] | None,
) -> Any:
    from src.cli import run_portfolio as cli

    resolved_config, run_dirs, components = cli._resolve_portfolio_inputs(
        portfolio_config_path=portfolio_config_path,
        portfolio_name=portfolio_name,
        explicit_run_ids=explicit_run_ids,
        from_registry=from_registry,
        from_sweep_top_ranked=from_sweep_top_ranked,
        from_candidate_selection=from_candidate_selection,
        timeframe=timeframe,
        evaluation_path=evaluation_path,
        execution_override=execution_override,
        cli_overrides=cli_overrides,
        state=state,
    )
    resolved_simulation = cli.resolve_simulation_config(
        None if simulation_path is None else cli.load_simulation_config(simulation_path).to_dict(),
        base=resolved_config.get("simulation"),
    )
    if promotion_gates_path is not None:
        resolved_config = dict(resolved_config)
        resolved_config["promotion_gates"] = cli.load_promotion_gate_config(promotion_gates_path)
    runtime_config = cli.resolve_runtime_config(
        resolved_config,
        cli_strict=strict,
    )
    resolved_config = runtime_config.apply_to_payload(
        dict(resolved_config),
        validation_key="validation",
    )
    if resolved_simulation is not None:
        resolved_config["simulation"] = resolved_simulation.to_dict()

    allocator = cli._build_allocator(
        resolved_config["allocator"],
        optimizer_config=resolved_config.get("optimizer"),
    )
    output_root = cli.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT if output_dir is None else output_dir

    if evaluation_path is not None:
        if resolved_simulation is not None:
            raise ValueError("The --simulation argument cannot be combined with --evaluation.")
        walk_forward_result = cli.run_portfolio_walk_forward(
            component_run_ids=[str(component["run_id"]) for component in components],
            evaluation_config_path=evaluation_path,
            allocator=allocator,
            timeframe=timeframe,
            output_dir=output_root,
            portfolio_name=str(resolved_config["portfolio_name"]),
            initial_capital=float(resolved_config["initial_capital"]),
            alignment_policy=str(resolved_config["alignment_policy"]),
            execution_config=runtime_config.execution,
            validation_config=runtime_config.portfolio_validation.to_dict(),
            risk_config=runtime_config.risk.to_dict(),
            volatility_targeting_config=resolved_config.get("volatility_targeting"),
            promotion_gates=resolved_config.get("promotion_gates"),
            sanity_config=runtime_config.sanity.to_dict(),
            strict_mode=runtime_config.strict_mode.enabled,
        )
        result = cli.PortfolioWalkForwardRunResult(
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
        strategy_returns = (
            cli.load_strategy_runs_returns(run_dirs)
            if all(str(component.get("artifact_type", "strategy")) == "strategy" for component in components)
            else cli.load_portfolio_component_runs_returns(components)
        )
        aligned_returns = cli.build_aligned_return_matrix(strategy_returns)
        try:
            portfolio_output = cli.construct_portfolio(
                aligned_returns,
                allocator,
                initial_capital=float(resolved_config["initial_capital"]),
                execution_config=runtime_config.execution,
                validation_config=runtime_config.portfolio_validation,
                optimization_returns=aligned_returns,
                risk_config=runtime_config.risk,
                volatility_targeting_config=resolved_config.get("volatility_targeting"),
                periods_per_year=cli._periods_per_year_from_timeframe(timeframe),
            )
        except ValueError as exc:
            cli.raise_research_validation_error(
                validator="portfolio_validation",
                scope=f"portfolio:{resolved_config['portfolio_name']}",
                exc=exc,
                strict_mode=runtime_config.strict_mode.enabled,
            )
        metrics = cli.compute_portfolio_metrics(
            portfolio_output,
            timeframe,
            validation_config=runtime_config.portfolio_validation,
            risk_config=runtime_config.risk,
        )
        try:
            sanity_report = cli.validate_portfolio_output_sanity(
                portfolio_output,
                metrics,
                runtime_config.sanity,
                initial_capital=float(resolved_config["initial_capital"]),
            )
        except ValueError as exc:
            cli.raise_research_validation_error(
                validator="sanity",
                scope=f"portfolio:{resolved_config['portfolio_name']}",
                exc=exc,
                strict_mode=runtime_config.strict_mode.enabled,
            )
        metrics = sanity_report.apply_to_metrics(metrics)
        portfolio_output.attrs["sanity_check"] = sanity_report.to_dict()

        start_ts = cli._format_timestamp(aligned_returns.index.min())
        end_ts = cli._format_timestamp(aligned_returns.index.max())
        run_id = cli.generate_portfolio_run_id(
            portfolio_name=str(resolved_config["portfolio_name"]),
            allocator_name=allocator.name,
            component_run_ids=[str(component["run_id"]) for component in components],
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            config=resolved_config,
            evaluation_config_path=evaluation_path,
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
            simulation_result = cli.run_return_simulation(
                portfolio_output["portfolio_return"],
                config=resolved_simulation,
                periods_per_year=cli._periods_per_year_from_timeframe(timeframe),
                owner=f"portfolio {resolved_config['portfolio_name']} returns",
                var_confidence_level=float(runtime_config.risk.var_confidence_level),
                cvar_confidence_level=float(runtime_config.risk.cvar_confidence_level),
            )
            cli.write_simulation_artifacts(
                experiment_dir / "simulation",
                simulation_result,
                parent_manifest_dir=experiment_dir,
            )
            import json

            manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))

        registry_path = cli.default_registry_path(output_root)
        cli.register_portfolio_run(
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

        result = cli.PortfolioRunResult(
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

    if pipeline_context is not None:
        artifact_dir = result.experiment_dir
        return cli.build_pipeline_cli_result(
            identifier=result.run_id,
            name=result.portfolio_name,
            artifact_dir=artifact_dir,
            manifest_path=artifact_dir / "manifest.json",
            output_paths=_portfolio_output_paths(result),
            metrics=dict(result.metrics),
            extra={
                "allocator_name": result.allocator_name,
                "component_count": int(result.component_count),
                "timeframe": result.timeframe,
            },
        )
    return result


def _summarize_portfolio_result(raw_result: Any) -> ExecutionResult:
    if isinstance(raw_result, dict):
        return ExecutionResult(
            workflow="portfolio",
            run_id=str(raw_result.get("identifier", "portfolio")),
            name=str(raw_result.get("name", "portfolio")),
            artifact_dir=_optional_path(raw_result.get("artifact_dir")),
            manifest_path=_optional_path(raw_result.get("manifest_path")),
            metrics=dict(raw_result.get("metrics") or {}),
            output_paths={
                str(key): Path(str(value))
                for key, value in dict(raw_result.get("output_paths") or {}).items()
                if value is not None
            },
            extra=dict(raw_result.get("extra") or {}),
            raw_result=raw_result,
        )
    return summarize_execution_result(
        workflow="portfolio",
        raw_result=raw_result,
        output_paths=_portfolio_output_paths(raw_result),
        extra=_portfolio_extra(raw_result),
    )


def _portfolio_output_paths(raw_result: Any) -> dict[str, Path]:
    artifact_dir = raw_result.experiment_dir
    paths = {
        "manifest_json": artifact_dir / "manifest.json",
        "qa_summary_json": artifact_dir / "qa_summary.json",
    }
    if hasattr(raw_result, "aggregate_metrics"):
        paths.update(
            {
                "aggregate_metrics_json": artifact_dir / "aggregate_metrics.json",
                "metrics_by_split_csv": artifact_dir / "metrics_by_split.csv",
            }
        )
    else:
        paths.update(
            {
                "metrics_json": artifact_dir / "metrics.json",
                "portfolio_returns_csv": artifact_dir / "portfolio_returns.csv",
                "portfolio_equity_curve_csv": artifact_dir / "portfolio_equity_curve.csv",
                "weights_csv": artifact_dir / "weights.csv",
            }
        )
    return paths


def _portfolio_extra(raw_result: Any) -> dict[str, Any]:
    extra = {
        "allocator_name": raw_result.allocator_name,
        "component_count": int(raw_result.component_count),
        "timeframe": raw_result.timeframe,
    }
    split_count = getattr(raw_result, "split_count", None)
    if split_count is not None:
        extra["split_count"] = int(split_count)
    return extra


def _portfolio_cli_overrides(
    *,
    optimizer_method: str | None,
    enable_volatility_targeting: bool,
    volatility_target_volatility: float | None,
    volatility_target_lookback: int | None,
    risk_target_volatility: float | None,
    risk_volatility_window: int | None,
    risk_var_confidence_level: float | None,
    risk_cvar_confidence_level: float | None,
    risk_allow_scale_up: bool,
    risk_max_volatility_scale: float | None,
) -> Any:
    from src.cli import run_portfolio as cli

    optimizer_override = None if optimizer_method is None else {"method": str(optimizer_method).strip().lower()}
    risk_override: dict[str, Any] = {}
    if risk_target_volatility is not None:
        risk_override["target_volatility"] = float(risk_target_volatility)
    if risk_volatility_window is not None:
        risk_override["volatility_window"] = int(risk_volatility_window)
    if risk_var_confidence_level is not None:
        risk_override["var_confidence_level"] = float(risk_var_confidence_level)
    if risk_cvar_confidence_level is not None:
        risk_override["cvar_confidence_level"] = float(risk_cvar_confidence_level)
    if risk_allow_scale_up:
        risk_override["allow_scale_up"] = True
    if risk_max_volatility_scale is not None:
        risk_override["max_volatility_scale"] = float(risk_max_volatility_scale)

    volatility_targeting_override: dict[str, Any] = {}
    if enable_volatility_targeting:
        volatility_targeting_override["enabled"] = True
    if volatility_target_volatility is not None:
        volatility_targeting_override["target_volatility"] = float(volatility_target_volatility)
    if volatility_target_lookback is not None:
        volatility_targeting_override["lookback_periods"] = int(volatility_target_lookback)

    runtime_override = None if not risk_override else {"risk": risk_override}
    if (
        optimizer_override is None
        and runtime_override is None
        and not volatility_targeting_override
    ):
        return None
    return cli.PortfolioCliOverrides(
        optimizer=optimizer_override,
        runtime=runtime_override,
        volatility_targeting=(None if not volatility_targeting_override else volatility_targeting_override),
    )


def _validate_portfolio_api_args(
    *,
    portfolio_config_path: str | Path | None,
    portfolio_name: str | None,
    run_ids: Sequence[str] | None,
    from_registry: bool,
    from_sweep_top_ranked: bool,
    from_candidate_selection: str | Path | None,
    timeframe: str,
    execution_override: Mapping[str, Any] | None,
    simulation_path: str | Path | None,
    evaluation_path: str | Path | None,
    risk_allow_scale_up: bool,
    risk_max_volatility_scale: float | None,
) -> None:
    from src.cli import run_portfolio as cli

    has_config = portfolio_config_path is not None
    has_run_ids = bool(run_ids)
    has_candidate_selection = from_candidate_selection is not None
    input_modes = [has_config, has_run_ids, has_candidate_selection]
    if sum(1 for mode in input_modes if mode) != 1:
        raise ValueError(
            "Provide exactly one input mode: portfolio_config_path, run_ids, or from_candidate_selection."
        )
    if from_registry and has_run_ids:
        raise ValueError("run_ids cannot be combined with from_registry.")
    if from_registry and not has_config:
        raise ValueError("from_registry requires portfolio_config_path.")
    if from_registry and has_candidate_selection:
        raise ValueError("from_registry cannot be combined with from_candidate_selection.")
    if from_sweep_top_ranked and not has_config:
        raise ValueError("from_sweep_top_ranked requires portfolio_config_path.")
    if from_sweep_top_ranked and from_registry:
        raise ValueError("from_sweep_top_ranked cannot be combined with from_registry.")
    if from_sweep_top_ranked and has_run_ids:
        raise ValueError("from_sweep_top_ranked cannot be combined with run_ids.")
    if from_sweep_top_ranked and has_candidate_selection:
        raise ValueError("from_sweep_top_ranked cannot be combined with from_candidate_selection.")
    if has_run_ids and not portfolio_name:
        raise ValueError("portfolio_name is required when using run_ids.")
    if has_candidate_selection and not portfolio_name:
        raise ValueError("portfolio_name is required when using from_candidate_selection.")
    if timeframe not in cli.SUPPORTED_TIMEFRAMES:
        supported = ", ".join(cli.SUPPORTED_TIMEFRAMES)
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Supported values: {supported}.")
    if execution_override is not None and execution_override.get("enabled") is True:
        if execution_override.get("disabled") is True:
            raise ValueError("Conflicting execution override flags were provided.")
    if simulation_path is not None and evaluation_path is not None:
        raise ValueError("simulation_path cannot be combined with evaluation_path.")
    if risk_allow_scale_up and risk_max_volatility_scale is not None:
        if float(risk_max_volatility_scale) <= 1.0:
            raise ValueError(
                "risk_allow_scale_up requires risk_max_volatility_scale > 1.0 when provided."
            )


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(str(value))
