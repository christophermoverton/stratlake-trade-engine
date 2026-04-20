from __future__ import annotations

import hashlib
import json
import shutil
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config.execution import ExecutionConfig
from src.config.robustness import EnsembleDefinitionConfig, RankingConfig, ResearchSweepConfig, RobustnessConfig
from src.config.runtime import RuntimeConfig, resolve_runtime_config
from src.data.load_features import load_features
from src.portfolio import compute_portfolio_equity_curve, compute_portfolio_metrics, compute_portfolio_returns
from src.portfolio.artifacts import write_portfolio_artifacts
from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import resolve_robustness_experiment_dir, save_experiment_outputs
from src.research.metrics import aggregate_strategy_returns, compute_benchmark_relative_metrics, compute_performance_metrics
from src.research.signal_engine import generate_signals
from src.research.signal_semantics import (
    Signal,
    attach_signal_metadata,
    ensure_signal_type_compatible,
    extract_signal_metadata,
    percentile_to_quantile_bucket,
    rank_to_percentile,
    resolve_signal_type_definition,
    score_to_binary_long_only,
    score_to_cross_section_rank,
    score_to_signed_zscore,
    score_to_ternary_long_short,
    spread_to_zscore,
    validate_directional_asymmetry_compatibility,
)
from src.research.statistical_validity import apply_statistical_validity_controls
from src.research.strategies import build_strategy
from src.research.strategies.validation import get_strategy_registry_entry, load_strategies_registry


def run_extended_robustness_experiment(
    strategy_name: str | None,
    *,
    robustness_config: RobustnessConfig,
    start: str | None = None,
    end: str | None = None,
    evaluation_path: Path | None = None,
    runtime_config: RuntimeConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    strict: bool = False,
    strategy_config_path: Path,
):
    from src.research.robustness import (
        RobustnessAnalysisError,
        RobustnessRunResult,
        StrategyVariant,
        _apply_sanity,
        _build_benchmark_results,
        _build_neighbor_metrics,
        _build_robustness_summary,
        _rank_stability_metrics,
        _resolve_metric_direction,
        _validate_robustness_outputs,
        load_strategy_config,
    )

    del evaluation_path

    registry = load_strategies_registry()
    research_sweep = robustness_config.research_sweep
    strategy_names = research_sweep.strategy.names or tuple(value for value in (strategy_name, robustness_config.strategy_name) if value)
    if not strategy_names:
        raise RobustnessAnalysisError("Extended robustness sweeps require at least one strategy.")

    valid_specs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    order_index = 0
    for current_strategy_name in strategy_names:
        base_config = load_strategy_config(current_strategy_name, path=strategy_config_path)
        registry_entry = get_strategy_registry_entry(current_strategy_name, registry=registry)
        strategy_overrides = _expand_keyed_space(dict(research_sweep.strategy.params), dict(research_sweep.strategy.options))
        signal_options = _expand_signal_space(research_sweep)
        constructor_options = _expand_layer_space(research_sweep.constructor)
        asymmetry_options = _expand_layer_space(research_sweep.asymmetry)
        for strategy_override in strategy_overrides:
            resolved_strategy_config = _merge_strategy_config(base_config, strategy_override)
            for signal_option, constructor_option, asymmetry_option in product(signal_options, constructor_options, asymmetry_options):
                record = _build_strategy_record(
                    strategy_name=current_strategy_name,
                    strategy_config=resolved_strategy_config,
                    signal_option=signal_option,
                    constructor_option=constructor_option,
                    asymmetry_option=asymmetry_option,
                    registry_entry=registry_entry,
                )
                error = _validate_strategy_record(record)
                if error is not None:
                    rejected.append({"config": record, "reason": error})
                    continue
                signature = json.dumps(record, sort_keys=True)
                if signature in seen_signatures:
                    rejected.append({"config": record, "reason": "duplicate configuration"})
                    continue
                seen_signatures.add(signature)
                flat = _flatten_record(record)
                valid_specs.append(
                    {
                        "run_kind": "strategy",
                        "record": record,
                        "runtime_base_config": resolved_strategy_config,
                        "variant": StrategyVariant(
                            variant_id=_variant_id(order_index, record),
                            variant_label=_label(flat),
                            parameter_values=flat,
                            strategy_config=record,
                            order_index=order_index,
                        ),
                    }
                )
                order_index += 1

    for ensemble_definition in research_sweep.ensemble.definitions:
        for signal_option, constructor_option, asymmetry_option in product(
            _expand_signal_space(research_sweep),
            _expand_layer_space(research_sweep.constructor),
            _expand_layer_space(research_sweep.asymmetry),
        ):
            record = _build_ensemble_record(
                ensemble_definition=ensemble_definition,
                signal_option=signal_option,
                constructor_option=constructor_option,
                asymmetry_option=asymmetry_option,
            )
            error = _validate_ensemble_record(record, registry)
            if error is not None:
                rejected.append({"config": record, "reason": error})
                continue
            signature = json.dumps(record, sort_keys=True)
            if signature in seen_signatures:
                rejected.append({"config": record, "reason": "duplicate configuration"})
                continue
            seen_signatures.add(signature)
            flat = _flatten_record(record)
            valid_specs.append(
                {
                    "run_kind": "ensemble",
                    "record": record,
                    "runtime_base_config": load_strategy_config(ensemble_definition.members[0], path=strategy_config_path),
                    "variant": StrategyVariant(
                        variant_id=_variant_id(order_index, record),
                        variant_label=_label(flat),
                        parameter_values=flat,
                        strategy_config=record,
                        order_index=order_index,
                    ),
                }
            )
            order_index += 1

    batching = research_sweep.batching
    if batching.batch_size is not None:
        start_index = batching.batch_index * batching.batch_size
        valid_specs = valid_specs[start_index:start_index + batching.batch_size]
    if not valid_specs:
        reasons = sorted({entry["reason"] for entry in rejected}) or ["no valid configurations"]
        raise RobustnessAnalysisError(f"Extended robustness sweep produced no valid configurations. Reasons: {', '.join(reasons)}.")

    resolved_runtime = runtime_config or resolve_runtime_config(
        valid_specs[0]["runtime_base_config"],
        cli_overrides=None if execution_config is None else {"execution": execution_config.to_dict()},
        cli_strict=strict,
    )
    dataset_cache: dict[tuple[str, str | None, str | None], pd.DataFrame] = {}
    benchmark_cache: dict[tuple[str, str | None, str | None, str], pd.DataFrame] = {}
    result_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    child_payloads: list[dict[str, Any]] = []

    for spec in valid_specs:
        cache_key = json.dumps(spec["record"], sort_keys=True)
        executed = result_cache.get(cache_key)
        if executed is None:
            if spec["run_kind"] == "strategy":
                executed = _execute_strategy_spec(
                    spec["record"],
                    runtime_config=resolved_runtime,
                    execution_config=execution_config,
                    dataset_cache=dataset_cache,
                    benchmark_cache=benchmark_cache,
                    start=start,
                    end=end,
                    apply_sanity_fn=_apply_sanity,
                    build_benchmark_fn=_build_benchmark_results,
                )
            else:
                executed = _execute_ensemble_spec(
                    spec["record"],
                    runtime_config=resolved_runtime,
                    execution_config=execution_config,
                    dataset_cache=dataset_cache,
                    start=start,
                    end=end,
                    result_cache=result_cache,
                    strategy_config_path=strategy_config_path,
                )
            result_cache[cache_key] = executed
        rows.append(
            {
                **spec["variant"].to_record(),
                "run_kind": spec["run_kind"],
                "strategy_name": spec["record"].get("strategy", {}).get("name"),
                "signal_type": spec["record"].get("signal", {}).get("type"),
                "constructor_id": spec["record"].get("constructor", {}).get("name"),
                "ensemble_name": spec["record"].get("ensemble", {}).get("name"),
                **executed["metrics"],
            }
        )
        child_payloads.append({"variant": spec["variant"], "record": spec["record"], "executed": executed})

    variant_metrics = pd.DataFrame(rows)
    higher_is_better = _resolve_metric_direction(robustness_config.ranking.primary_metric, robustness_config.ranking.higher_is_better)
    variant_metrics = _rank_with_tie_breakers(variant_metrics, robustness_config.ranking, higher_is_better=higher_is_better)
    stability_metrics = _build_extended_stability_metrics(
        child_payloads,
        robustness_config=robustness_config,
    )
    stability_metrics = _rank_stability_metrics(
        stability_metrics,
        robustness_config.ranking.primary_metric,
        higher_is_better,
    )
    neighbor_metrics = _build_extended_neighbor_metrics(
        [payload["variant"] for payload in child_payloads],
        variant_metrics,
        robustness_config.ranking.primary_metric,
        build_neighbor_metrics_fn=_build_neighbor_metrics,
    )
    summary = _build_robustness_summary(
        strategy_name=strategy_name or robustness_config.strategy_name or "research_sweep",
        robustness_config=robustness_config,
        variant_metrics=variant_metrics,
        stability_metrics=stability_metrics,
        neighbor_metrics=neighbor_metrics,
        higher_is_better=higher_is_better,
    )
    summary.update(
        {
            "config_count": int(len(variant_metrics)),
            "valid_config_count": int(len(variant_metrics)),
            "invalid_config_count": int(len(rejected)),
            "rejected_configurations": rejected,
            "group_by": list(research_sweep.group_by),
            "batching": research_sweep.batching.to_dict(),
            "multiple_testing_awareness": robustness_config.statistical_controls.multiple_testing_awareness,
            "overfitting_warning": len(variant_metrics) >= robustness_config.statistical_controls.overfitting_warning_search_space,
        }
    )
    validity = apply_statistical_validity_controls(
        variant_metrics=variant_metrics,
        stability_metrics=stability_metrics,
        neighbor_metrics=neighbor_metrics,
        child_payloads=child_payloads,
        robustness_config=robustness_config,
        higher_is_better=higher_is_better,
        summary=summary,
    )
    variant_metrics = validity.variant_metrics
    summary["statistical_validity"] = validity.summary
    _validate_robustness_outputs(variant_metrics, stability_metrics, neighbor_metrics, summary, robustness_config)
    owner_name = strategy_name or robustness_config.strategy_name or (strategy_names[0] if len(strategy_names) == 1 else "research_sweep")
    experiment_dir = resolve_robustness_experiment_dir(owner_name, config=robustness_config.to_dict(), variant_metrics=variant_metrics, stability_metrics=stability_metrics, summary=summary)
    _write_artifacts(experiment_dir=experiment_dir, robustness_config=robustness_config, variant_metrics=variant_metrics, stability_metrics=stability_metrics, neighbor_metrics=neighbor_metrics, summary=summary, child_payloads=child_payloads)
    return RobustnessRunResult(owner_name, experiment_dir.name, experiment_dir, summary, [payload["variant"] for payload in child_payloads], variant_metrics, stability_metrics, neighbor_metrics)


def _expand_keyed_space(params: Mapping[str, tuple[Any, ...]], options: Mapping[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    axes = [(f"params.{key}", params[key]) for key in sorted(params)] + [(f"options.{key}", options[key]) for key in sorted(options)]
    if not axes:
        return [{}]
    return [{axis_name: value for (axis_name, _), value in zip(axes, values, strict=True)} for values in product(*(axis_values for _, axis_values in axes))]


def _expand_signal_space(research_sweep: ResearchSweepConfig) -> list[dict[str, Any]]:
    signal = research_sweep.signal
    axes = [("type", signal.types or (None,))]
    axes.extend((f"params.{key}", signal.params[key]) for key in sorted(signal.params))
    axes.extend((f"options.{key}", signal.options[key]) for key in sorted(signal.options))
    return [{axis_name: value for (axis_name, _), value in zip(axes, values, strict=True) if value is not None} for values in product(*(axis_values for _, axis_values in axes))]


def _expand_layer_space(layer: Any) -> list[dict[str, Any]]:
    axes = [("name", layer.names or (None,))]
    axes.extend((f"params.{key}", layer.params[key]) for key in sorted(layer.params))
    axes.extend((f"options.{key}", layer.options[key]) for key in sorted(layer.options))
    return [{axis_name: value for (axis_name, _), value in zip(axes, values, strict=True) if value is not None} for values in product(*(axis_values for _, axis_values in axes))]


def _merge_strategy_config(base_config: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    parameters = dict(config.get("parameters", {}))
    for key, value in override.items():
        if key.startswith("params."):
            parameters[key.removeprefix("params.")] = value
        elif key.startswith("options."):
            config[key.removeprefix("options.")] = value
    config["parameters"] = parameters
    return config


def _build_strategy_record(
    *,
    strategy_name: str,
    strategy_config: Mapping[str, Any],
    signal_option: Mapping[str, Any],
    constructor_option: Mapping[str, Any],
    asymmetry_option: Mapping[str, Any],
    registry_entry: Mapping[str, Any],
) -> dict[str, Any]:
    position_constructor = strategy_config.get("position_constructor", {})
    constructor_name = constructor_option.get("name", position_constructor.get("name", "identity_weights"))
    base_constructor_params = (
        dict(position_constructor.get("params", {}))
        if constructor_name == position_constructor.get("name", "identity_weights")
        else {}
    )
    return {
        "strategy": {
            "name": strategy_name,
            "version": registry_entry.get("version"),
            "dataset": strategy_config.get("dataset"),
            "params": dict(strategy_config.get("parameters", {})),
            "source_signal_type": strategy_config.get("signal_type") or registry_entry.get("output_signal", {}).get("signal_type"),
        },
        "signal": {
            "type": signal_option.get("type", strategy_config.get("signal_type") or registry_entry.get("output_signal", {}).get("signal_type")),
            "params": {key.split(".", 1)[1]: signal_option[key] for key in sorted(signal_option) if key.startswith("params.") or key.startswith("options.")},
        },
        "constructor": {
            "name": constructor_name,
            "params": {
                **base_constructor_params,
                **{key.split(".", 1)[1]: constructor_option[key] for key in sorted(constructor_option) if key.startswith("params.") or key.startswith("options.")}
            },
        },
        "asymmetry": {key.split(".", 1)[1]: asymmetry_option[key] for key in sorted(asymmetry_option) if key.startswith("params.") or key.startswith("options.")},
    }


def _build_ensemble_record(
    *,
    ensemble_definition: EnsembleDefinitionConfig,
    signal_option: Mapping[str, Any],
    constructor_option: Mapping[str, Any],
    asymmetry_option: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "ensemble": ensemble_definition.to_dict(),
        "signal": {"type": signal_option.get("type"), "params": {key.split(".", 1)[1]: signal_option[key] for key in sorted(signal_option) if key.startswith("params.") or key.startswith("options.")}},
        "constructor": {"name": constructor_option.get("name"), "params": {key.split(".", 1)[1]: constructor_option[key] for key in sorted(constructor_option) if key.startswith("params.") or key.startswith("options.")}},
        "asymmetry": {key.split(".", 1)[1]: asymmetry_option[key] for key in sorted(asymmetry_option) if key.startswith("params.") or key.startswith("options.")},
    }


def _validate_strategy_record(record: Mapping[str, Any]) -> str | None:
    try:
        ensure_signal_type_compatible(record["signal"]["type"], position_constructor=record["constructor"]["name"])
        definition = resolve_signal_type_definition(record["signal"]["type"])
        validation = validate_directional_asymmetry_compatibility(definition, record["constructor"]["name"], asymmetry_config=dict(record.get("asymmetry", {})))
        if record.get("asymmetry") and not validation["valid"]:
            return validation["error_message"]
    except ValueError as exc:
        return str(exc)
    if not _supports_transform(record["strategy"]["source_signal_type"], record["signal"]["type"]):
        return f"strategy output signal '{record['strategy']['source_signal_type']}' is not compatible with requested signal semantics '{record['signal']['type']}'."
    return None


def _validate_ensemble_record(record: Mapping[str, Any], registry: list[dict[str, Any]]) -> str | None:
    members = record["ensemble"]["members"]
    if len(set(members)) != len(members):
        return "ensemble members must be unique"
    for member in members:
        try:
            get_strategy_registry_entry(str(member), registry=registry)
        except ValueError as exc:
            return str(exc)
    return None


def _supports_transform(source_signal_type: str, target_signal_type: str) -> bool:
    if source_signal_type == target_signal_type:
        return True
    return (source_signal_type, target_signal_type) in {
        ("prediction_score", "signed_zscore"),
        ("prediction_score", "cross_section_rank"),
        ("prediction_score", "cross_section_percentile"),
        ("prediction_score", "ternary_quantile"),
        ("prediction_score", "binary_signal"),
        ("prediction_score", "spread_zscore"),
        ("cross_section_rank", "cross_section_percentile"),
        ("cross_section_rank", "ternary_quantile"),
        ("cross_section_rank", "binary_signal"),
        ("cross_section_percentile", "ternary_quantile"),
        ("cross_section_percentile", "binary_signal"),
    }


def _execute_strategy_spec(
    record: Mapping[str, Any],
    *,
    runtime_config: RuntimeConfig,
    execution_config: ExecutionConfig | None,
    dataset_cache: dict[tuple[str, str | None, str | None], pd.DataFrame],
    benchmark_cache: dict[tuple[str, str | None, str | None, str], pd.DataFrame],
    start: str | None,
    end: str | None,
    apply_sanity_fn: Any,
    build_benchmark_fn: Any,
) -> dict[str, Any]:
    dataset_name = str(record["strategy"]["dataset"])
    dataset_key = (dataset_name, start, end)
    dataset = dataset_cache.get(dataset_key)
    if dataset is None:
        dataset = load_features(dataset_name, start=start, end=end)
        dataset_cache[dataset_key] = dataset
    strategy = build_strategy(record["strategy"]["name"], {"dataset": dataset_name, "parameters": dict(record["strategy"]["params"]), "signal_type": record["strategy"]["source_signal_type"], "signal_params": {}, "position_constructor": None})
    strategy.position_constructor_name = None
    strategy.position_constructor_params = {}
    base_frame = generate_signals(dataset, strategy)
    base_signal = Signal(record["strategy"]["source_signal_type"], "1.0.0", base_frame, "signal", dict(extract_signal_metadata(base_frame) or {}))
    transformed = _transform_signal(base_signal, record["signal"]["type"], dict(record["signal"].get("params", {})))
    managed = transformed.data.copy(deep=True)
    metadata = dict(extract_signal_metadata(managed) or {})
    metadata["constructor_id"] = record["constructor"]["name"]
    metadata["constructor_params"] = {**record["constructor"]["params"], **record.get("asymmetry", {})}
    attach_signal_metadata(managed, metadata)
    effective_execution = runtime_config.execution if execution_config is None else execution_config
    benchmark_key = (dataset_name, start, end, json.dumps(effective_execution.to_dict(), sort_keys=True))
    benchmark = benchmark_cache.get(benchmark_key)
    if benchmark is None:
        benchmark = build_benchmark_fn(dataset, dataset_name, effective_execution)
        benchmark_cache[benchmark_key] = benchmark
    results_df = run_backtest(
        managed,
        effective_execution,
        require_managed_signals=True,
    )
    metrics = compute_performance_metrics(results_df)
    metrics.update(compute_benchmark_relative_metrics(results_df, benchmark))
    metrics = apply_sanity_fn(metrics, results_df, runtime_config, scope=f"extended:{record['strategy']['name']}")
    return {
        "artifact_kind": "strategy",
        "results_df": results_df,
        "metrics": metrics,
        "child_config": {
            "strategy_name": record["strategy"]["name"],
            "dataset": dataset_name,
            "parameters": dict(record["strategy"]["params"]),
            "signal": dict(record["signal"]),
            "constructor": {"name": record["constructor"]["name"], "params": dict(metadata["constructor_params"])},
            "asymmetry": dict(record.get("asymmetry", {})),
            "execution": effective_execution.to_dict(),
        },
    }


def _execute_ensemble_spec(
    record: Mapping[str, Any],
    *,
    runtime_config: RuntimeConfig,
    execution_config: ExecutionConfig | None,
    dataset_cache: dict[tuple[str, str | None, str | None], pd.DataFrame],
    start: str | None,
    end: str | None,
    result_cache: dict[str, dict[str, Any]],
    strategy_config_path: Path,
) -> dict[str, Any]:
    from src.research.robustness import load_strategy_config

    member_results: list[dict[str, Any]] = []
    for member_name in record["ensemble"]["members"]:
        member_record = _build_strategy_record(
            strategy_name=str(member_name),
            strategy_config=load_strategy_config(str(member_name), path=strategy_config_path),
            signal_option={"type": record["signal"]["type"], **{f"params.{key}": value for key, value in record["signal"]["params"].items()}},
            constructor_option={"name": record["constructor"]["name"], **{f"params.{key}": value for key, value in record["constructor"]["params"].items()}},
            asymmetry_option={f"params.{key}": value for key, value in record["asymmetry"].items()},
            registry_entry=get_strategy_registry_entry(str(member_name), registry=load_strategies_registry()),
        )
        cache_key = json.dumps(member_record, sort_keys=True)
        if cache_key not in result_cache:
            result_cache[cache_key] = _execute_strategy_spec(
                member_record,
                runtime_config=runtime_config,
                execution_config=execution_config,
                dataset_cache=dataset_cache,
                benchmark_cache={},
                start=start,
                end=end,
                apply_sanity_fn=lambda metrics, results_df, runtime, scope: metrics,
                build_benchmark_fn=lambda dataset, dataset_name, execution: pd.DataFrame(),
            )
        member_results.append(result_cache[cache_key])
    merged = None
    member_names: list[str] = []
    for member_name, member_result in zip(record["ensemble"]["members"], member_results, strict=True):
        frame = aggregate_strategy_returns(member_result["results_df"]).loc[:, ["ts_utc", "strategy_return"]].rename(columns={"strategy_return": str(member_name)})
        merged = frame if merged is None else merged.merge(frame, on="ts_utc", how="inner", sort=False)
        member_names.append(str(member_name))
    assert merged is not None
    merged = merged.sort_values("ts_utc", kind="stable").reset_index(drop=True)
    returns_wide = merged.set_index("ts_utc")[member_names].astype("float64")
    weights_wide = _weight_frame(returns_wide.index, member_names, weighting_method=str(record["ensemble"]["weighting_method"]), weights=dict(record["ensemble"].get("weights", {})))
    portfolio_output = compute_portfolio_returns(returns_wide, weights_wide, execution_config=runtime_config.execution)
    portfolio_output = compute_portfolio_equity_curve(portfolio_output)
    metrics = compute_portfolio_metrics(portfolio_output, "1D")
    return {
        "artifact_kind": "portfolio",
        "portfolio_output": portfolio_output,
        "metrics": metrics,
        "child_config": {"portfolio_name": record["ensemble"]["name"], "allocator": record["ensemble"]["weighting_method"], "components": [{"strategy_name": name, "run_id": f"{name}_member"} for name in member_names], "timeframe": "1D", "initial_capital": 1.0, "alignment_policy": "intersection", "execution": runtime_config.execution.to_dict()},
        "components": [{"strategy_name": name, "run_id": f"{name}_member"} for name in member_names],
    }


def _transform_signal(signal: Signal, target_signal_type: str, signal_params: Mapping[str, Any]) -> Signal:
    if signal.signal_type == target_signal_type:
        return signal
    if signal.signal_type == "prediction_score":
        if target_signal_type == "signed_zscore":
            return score_to_signed_zscore(signal, clip=signal_params.get("clip"))
        if target_signal_type == "cross_section_rank":
            return score_to_cross_section_rank(signal)
        if target_signal_type == "cross_section_percentile":
            return rank_to_percentile(score_to_cross_section_rank(signal))
        if target_signal_type == "ternary_quantile":
            return score_to_ternary_long_short(signal, quantile=float(signal_params.get("quantile", 0.2)))
        if target_signal_type == "binary_signal":
            return score_to_binary_long_only(signal, quantile=float(signal_params.get("quantile", 0.2)))
        if target_signal_type == "spread_zscore":
            return spread_to_zscore(signal, clip=signal_params.get("clip"))
    if signal.signal_type == "cross_section_rank":
        percentile = rank_to_percentile(signal)
        if target_signal_type == "cross_section_percentile":
            return percentile
        return percentile_to_quantile_bucket(percentile, quantile=float(signal_params.get("quantile", 0.2)), long_only=target_signal_type == "binary_signal")
    if signal.signal_type == "cross_section_percentile":
        return percentile_to_quantile_bucket(signal, quantile=float(signal_params.get("quantile", 0.2)), long_only=target_signal_type == "binary_signal")
    raise ValueError(f"Unsupported signal transformation from '{signal.signal_type}' to '{target_signal_type}'.")


def _weight_frame(index: pd.Index, member_names: list[str], *, weighting_method: str, weights: Mapping[str, Any]) -> pd.DataFrame:
    if weighting_method == "equal_weight":
        weight = 1.0 / float(len(member_names))
        return pd.DataFrame({name: [weight] * len(index) for name in member_names}, index=index)
    total = sum(float(weights.get(name, 0.0)) for name in member_names)
    if total <= 0.0:
        raise ValueError("Configured ensemble weights must sum to a positive value.")
    return pd.DataFrame({name: [float(weights.get(name, 0.0)) / total] * len(index) for name in member_names}, index=index)


def _build_extended_neighbor_metrics(
    variants: list[Any],
    variant_metrics: pd.DataFrame,
    ranking_metric: str,
    *,
    build_neighbor_metrics_fn: Any,
) -> pd.DataFrame:
    groups: dict[tuple[str, ...], list[Any]] = {}
    for variant in variants:
        keyset = tuple(sorted(str(key) for key in variant.parameter_values))
        groups.setdefault(keyset, []).append(variant)

    frames: list[pd.DataFrame] = []
    for group_variants in groups.values():
        if len(group_variants) < 2:
            continue
        frame = build_neighbor_metrics_fn(group_variants, variant_metrics, ranking_metric)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _build_extended_stability_metrics(
    child_payloads: list[dict[str, Any]],
    *,
    robustness_config: RobustnessConfig,
) -> pd.DataFrame:
    if robustness_config.stability.mode == "disabled":
        return pd.DataFrame()
    if robustness_config.stability.mode != "subperiods":
        raise ValueError(
            "Extended robustness sweeps currently support statistical split evidence only with "
            "stability.mode='subperiods'."
        )

    rows: list[dict[str, Any]] = []
    for payload in child_payloads:
        variant = payload["variant"]
        executed = payload["executed"]
        if executed["artifact_kind"] == "strategy":
            results_df = executed["results_df"]
            for period in _generate_frame_periods(results_df, robustness_config):
                period_df = results_df.loc[period.mask].reset_index(drop=True)
                metrics = compute_performance_metrics(period_df)
                rows.append({**variant.to_record(), **period.to_record(), **metrics})
        else:
            portfolio_output = executed["portfolio_output"]
            timeframe = str(executed["child_config"].get("timeframe", "1D"))
            for period in _generate_frame_periods(portfolio_output, robustness_config):
                period_df = portfolio_output.loc[period.mask].reset_index(drop=True)
                metrics = compute_portfolio_metrics(period_df, timeframe)
                rows.append({**variant.to_record(), **period.to_record(), **metrics})
    return pd.DataFrame(rows)


def _generate_frame_periods(frame: pd.DataFrame, robustness_config: RobustnessConfig) -> list[Any]:
    key_column = "date" if "date" in frame.columns else "ts_utc" if "ts_utc" in frame.columns else None
    if key_column is None:
        raise ValueError("Extended robustness subperiod stability requires a 'date' or 'ts_utc' column.")
    time_values = frame[key_column].astype("string")
    unique_values = list(dict.fromkeys(time_values.tolist()))
    period_count = int(robustness_config.stability.periods or 0)
    if len(unique_values) < period_count:
        raise ValueError(
            "Extended robustness subperiod stability requires at least as many distinct timestamps as configured periods."
        )

    periods: list[Any] = []
    base_width, remainder = divmod(len(unique_values), period_count)
    start_index = 0
    for index in range(period_count):
        width = base_width + (1 if index < remainder else 0)
        end_index = start_index + width
        period_values = unique_values[start_index:end_index]
        periods.append(
            _ExtendedPeriodDefinition(
                period_id=f"period_{index:04d}",
                period_mode="subperiods",
                period_start=str(period_values[0]),
                period_end=str(period_values[-1]),
                mask=time_values.isin(period_values),
            )
        )
        start_index = end_index
    return periods


def _rank_with_tie_breakers(frame: pd.DataFrame, ranking: RankingConfig, *, higher_is_better: bool) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_primary"] = pd.to_numeric(ordered[ranking.primary_metric], errors="coerce")
    sort_columns = ["_primary"]
    ascending = [not higher_is_better]
    for tie_breaker in ranking.tie_breakers:
        if tie_breaker not in ordered.columns:
            continue
        helper = f"_tie_{tie_breaker}"
        ordered[helper] = pd.to_numeric(ordered[tie_breaker], errors="coerce")
        sort_columns.append(helper)
        ascending.append(False)
    sort_columns.extend(["variant_order", "variant_id"])
    ascending.extend([True, True])
    ordered = ordered.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ordered["rank"] = pd.Series(range(1, len(ordered) + 1), dtype="int64")
    ordered["is_best_variant"] = ordered["rank"] == 1
    return ordered.drop(columns=[column for column in ordered.columns if column.startswith("_primary") or column.startswith("_tie_")])


def _write_artifacts(*, experiment_dir: Path, robustness_config: RobustnessConfig, variant_metrics: pd.DataFrame, stability_metrics: pd.DataFrame, neighbor_metrics: pd.DataFrame, summary: dict[str, Any], child_payloads: list[dict[str, Any]]) -> None:
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / "config.json", robustness_config.to_dict())
    _write_json(experiment_dir / "summary.json", summary)
    _write_json(experiment_dir / "variants.json", {"variants": [payload["variant"].to_record() for payload in child_payloads]})
    variant_metrics.to_csv(experiment_dir / "metrics_by_variant.csv", index=False)
    variant_metrics.to_csv(experiment_dir / "metrics_by_config.csv", index=False)
    _write_json(
        experiment_dir / "aggregate_metrics.json",
        {
            "summary": summary,
            "ranking": robustness_config.ranking.to_dict(),
            "statistical_validity": summary.get("statistical_validity"),
        },
    )
    if not stability_metrics.empty:
        stability_metrics.to_csv(experiment_dir / "stability_metrics.csv", index=False)
    if not neighbor_metrics.empty:
        neighbor_metrics.to_csv(experiment_dir / "neighbor_metrics.csv", index=False)
    variant_metrics.to_csv(experiment_dir / "ranked_configs.csv", index=False)
    child_runs = []
    for payload in child_payloads:
        child_dir = runs_dir / payload["variant"].variant_id
        if payload["executed"]["artifact_kind"] == "strategy":
            child_dir.mkdir(parents=True, exist_ok=False)
            save_experiment_outputs(child_dir, payload["executed"]["results_df"], payload["executed"]["metrics"], payload["executed"]["child_config"], strategy_name=str(payload["record"].get("strategy", {}).get("name", "strategy")))
        else:
            write_portfolio_artifacts(child_dir, payload["executed"]["portfolio_output"], payload["executed"]["metrics"], payload["executed"]["child_config"], payload["executed"]["components"])
        child_runs.append({"variant_id": payload["variant"].variant_id, "artifact_dir": child_dir.relative_to(experiment_dir).as_posix()})
    _write_json(
        experiment_dir / "manifest.json",
        {
            "run_id": experiment_dir.name,
            "evaluation_mode": "robustness",
            "primary_metric": robustness_config.ranking.primary_metric,
            "artifact_files": sorted(
                path.relative_to(experiment_dir).as_posix()
                for path in experiment_dir.rglob("*")
                if path.is_file()
            ),
            "metric_summary": summary,
            "variant_count": int(summary.get("variant_count", len(variant_metrics))),
            "config_count": int(summary.get("config_count", len(variant_metrics))),
            "child_runs": child_runs,
            "statistical_validity": summary.get("statistical_validity"),
        },
    )


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if "strategy" in record:
        flat["strategy.name"] = record["strategy"]["name"]
        for key, value in sorted(record["strategy"]["params"].items()):
            flat[f"strategy.params.{key}"] = value
    if "signal" in record:
        flat["signal.type"] = record["signal"]["type"]
        for key, value in sorted(record["signal"]["params"].items()):
            flat[f"signal.params.{key}"] = value
    if "constructor" in record:
        flat["constructor.name"] = record["constructor"]["name"]
        for key, value in sorted(record["constructor"]["params"].items()):
            flat[f"constructor.params.{key}"] = value
    for key, value in sorted(record.get("asymmetry", {}).items()):
        flat[f"asymmetry.{key}"] = value
    if "ensemble" in record:
        flat["ensemble.name"] = record["ensemble"]["name"]
        flat["ensemble.weighting_method"] = record["ensemble"]["weighting_method"]
    return flat


def _slug(values: Mapping[str, Any]) -> str:
    return "__".join(f"{''.join(char if char.isalnum() else '_' for char in key).strip('_')}_{''.join(char if char.isalnum() else '_' for char in str(value)).strip('_')}" for key, value in values.items())


def _label(values: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _variant_id(order_index: int, record: Mapping[str, Any]) -> str:
    signature = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
    return f"variant_{order_index:04d}_{digest}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class _ExtendedPeriodDefinition:
    def __init__(
        self,
        *,
        period_id: str,
        period_mode: str,
        period_start: str,
        period_end: str,
        mask: pd.Series,
    ) -> None:
        self.period_id = period_id
        self.period_mode = period_mode
        self.period_start = period_start
        self.period_end = period_end
        self.mask = mask

    def to_record(self) -> dict[str, Any]:
        return {
            "period_id": self.period_id,
            "period_mode": self.period_mode,
            "period_start": self.period_start,
            "period_end": self.period_end,
        }
