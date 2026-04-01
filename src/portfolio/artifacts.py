from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.portfolio.contracts import PortfolioContractError, validate_portfolio_output
from src.portfolio.qa import run_portfolio_qa
from src.portfolio.risk import resolve_portfolio_risk_config, summarize_portfolio_risk
from src.research.consistency import validate_portfolio_artifact_payload_consistency
from src.research.promotion import (
    DEFAULT_PROMOTION_ARTIFACT_FILENAME,
    evaluate_promotion_gates,
    write_promotion_gate_artifact,
)
from src.research.registry import canonicalize_value, register_portfolio_run

_CONFIG_FILENAME = "config.json"
_COMPONENTS_FILENAME = "components.json"
_WEIGHTS_FILENAME = "weights.csv"
_PORTFOLIO_RETURNS_FILENAME = "portfolio_returns.csv"
_PORTFOLIO_EQUITY_FILENAME = "portfolio_equity_curve.csv"
_METRICS_FILENAME = "metrics.json"
_QA_SUMMARY_FILENAME = "qa_summary.json"
_MANIFEST_FILENAME = "manifest.json"


def write_portfolio_artifacts(
    output_dir: str | Path,
    portfolio_output: pd.DataFrame,
    metrics: dict[str, object],
    config: dict[str, object],
    components: list[dict[str, object]],
) -> dict[str, object]:
    """Persist deterministic first-class artifacts for one portfolio run."""

    resolved_output_dir = Path(output_dir)
    try:
        normalized_portfolio_output = validate_portfolio_output(portfolio_output)
    except PortfolioContractError as exc:
        raise ValueError(f"portfolio_output must be valid portfolio output: {exc}") from exc

    normalized_config = _normalize_portfolio_config(config, components=components)
    normalized_components = _normalize_components(components)
    normalized_metrics = _normalize_mapping(metrics, owner="metrics")
    run_portfolio_qa(
        normalized_portfolio_output,
        normalized_metrics,
        normalized_config,
        strict=True,
    )

    weights_frame = _weights_frame(normalized_portfolio_output)
    returns_frame = _portfolio_returns_frame(normalized_portfolio_output)
    equity_frame = _portfolio_equity_curve_frame(normalized_portfolio_output)
    qa_summary = run_portfolio_qa(
        normalized_portfolio_output,
        normalized_metrics,
        normalized_config,
        strict=True,
    )
    promotion_evaluation = evaluate_promotion_gates(
        run_type="portfolio",
        config=_promotion_gate_config(normalized_config),
        sources={
            "metrics": normalized_metrics,
            "qa_summary": qa_summary,
            "config": normalized_config,
        },
    )
    manifest = _build_manifest(
        output_dir=resolved_output_dir,
        portfolio_output=normalized_portfolio_output,
        config=normalized_config,
        components=normalized_components,
        metrics=normalized_metrics,
        qa_summary=qa_summary,
        weights_frame=weights_frame,
        returns_frame=returns_frame,
        equity_frame=equity_frame,
        promotion_evaluation=promotion_evaluation,
    )
    validate_portfolio_artifact_payload_consistency(
        portfolio_output=normalized_portfolio_output,
        weights_frame=weights_frame,
        returns_frame=returns_frame,
        equity_frame=equity_frame,
        metrics=normalized_metrics,
        qa_summary=qa_summary,
        config=normalized_config,
        components=normalized_components,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(resolved_output_dir / _CONFIG_FILENAME, normalized_config)
    _write_json(resolved_output_dir / _COMPONENTS_FILENAME, {"components": normalized_components})
    _write_csv(resolved_output_dir / _WEIGHTS_FILENAME, weights_frame)
    _write_csv(resolved_output_dir / _PORTFOLIO_RETURNS_FILENAME, returns_frame)
    _write_csv(resolved_output_dir / _PORTFOLIO_EQUITY_FILENAME, equity_frame)
    _write_json(resolved_output_dir / _METRICS_FILENAME, normalized_metrics)
    _write_json(resolved_output_dir / _QA_SUMMARY_FILENAME, qa_summary)
    write_promotion_gate_artifact(resolved_output_dir, promotion_evaluation)
    _write_json(resolved_output_dir / _MANIFEST_FILENAME, manifest)
    return manifest


def _normalize_portfolio_config(
    config: dict[str, object],
    *,
    components: list[dict[str, object]],
) -> dict[str, object]:
    if not isinstance(config, dict):
        raise TypeError("config must be provided as a dictionary.")

    merged = dict(config)
    merged["components"] = components

    required_fields = ("portfolio_name", "allocator")
    missing = [field for field in required_fields if field not in merged]
    if missing:
        formatted = ", ".join(repr(field) for field in missing)
        raise ValueError(f"config is missing required portfolio fields: {formatted}.")

    normalized = _normalize_mapping(merged, owner="config")
    if "components" in normalized:
        normalized.pop("components")
    return normalized


def _normalize_components(components: list[dict[str, object]]) -> list[dict[str, object]]:
    if not isinstance(components, list):
        raise TypeError("components must be provided as a list of dictionaries.")
    if not components:
        raise ValueError("components must contain at least one portfolio component.")

    normalized_components: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_strategy_names: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise TypeError(f"components[{index}] must be a dictionary.")

        missing = [field for field in ("strategy_name", "run_id") if field not in component]
        if missing:
            formatted = ", ".join(repr(field) for field in missing)
            raise ValueError(
                f"components[{index}] is missing required fields: {formatted}."
            )

        strategy_name = _normalize_required_string(
            component.get("strategy_name"),
            field_name=f"components[{index}].strategy_name",
        )
        run_id = _normalize_required_string(
            component.get("run_id"),
            field_name=f"components[{index}].run_id",
        )
        component_key = (strategy_name, run_id)
        if component_key in seen_keys:
            raise ValueError(
                "components must be unique by (strategy_name, run_id). "
                f"Duplicate component: ({strategy_name!r}, {run_id!r})."
            )
        seen_keys.add(component_key)
        if strategy_name in seen_strategy_names:
            raise ValueError(
                "components must be unique by strategy_name. "
                f"Duplicate strategy_name: {strategy_name!r}."
            )
        seen_strategy_names.add(strategy_name)

        normalized_component = _normalize_mapping(component, owner=f"components[{index}]")
        normalized_component["strategy_name"] = strategy_name
        normalized_component["run_id"] = run_id
        normalized_components.append(normalized_component)

    return sorted(
        normalized_components,
        key=lambda component: (
            str(component["strategy_name"]),
            str(component["run_id"]),
        ),
    )


def _normalize_mapping(mapping: dict[str, object], *, owner: str) -> dict[str, object]:
    if not isinstance(mapping, dict):
        raise TypeError(f"{owner} must be a dictionary.")

    normalized: dict[str, object] = {}
    for key in sorted(mapping):
        if not isinstance(key, str):
            raise TypeError(f"{owner} keys must be strings. Found {type(key).__name__}.")
        normalized[key] = _normalize_json_value(mapping[key], path=f"{owner}.{key}")
    return normalized


def _normalize_json_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{path} must not contain NaN or infinite floats.")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return _format_timestamp(value)
    if isinstance(value, dict):
        return _normalize_mapping(value, owner=path)
    if isinstance(value, list):
        return [_normalize_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return [_normalize_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, pd.Series):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value.tolist())
        ]
    if pd.isna(value):
        return None

    try:
        normalized = canonicalize_value(value)
    except Exception as exc:  # pragma: no cover - defensive normalization fallback
        raise TypeError(f"{path} is not JSON-serializable.") from exc

    try:
        json.dumps(normalized, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} is not JSON-serializable.") from exc
    return normalized


def _weights_frame(portfolio_output: pd.DataFrame) -> pd.DataFrame:
    weight_columns = [column for column in portfolio_output.columns if column.startswith("weight__")]
    if not weight_columns:
        raise ValueError("portfolio_output must include at least one weight__<strategy> column.")

    ordered_columns = ["ts_utc", *sorted(weight_columns)]
    return _csv_ready_frame(portfolio_output.loc[:, ordered_columns].copy())


def _portfolio_returns_frame(portfolio_output: pd.DataFrame) -> pd.DataFrame:
    return_columns = [column for column in portfolio_output.columns if column.startswith("strategy_return__")]
    weight_columns = [column for column in portfolio_output.columns if column.startswith("weight__")]
    if not return_columns:
        raise ValueError("portfolio_output must include at least one strategy_return__<strategy> column.")
    if not weight_columns:
        raise ValueError("portfolio_output must include at least one weight__<strategy> column.")

    execution_columns = [
        column
        for column in (
            "gross_portfolio_return",
            "portfolio_weight_change",
            "portfolio_abs_weight_change",
            "portfolio_turnover",
            "portfolio_rebalance_event",
            "portfolio_changed_sleeve_count",
            "portfolio_transaction_cost",
            "portfolio_fixed_fee",
            "portfolio_slippage_proxy",
            "portfolio_slippage_cost",
            "portfolio_execution_friction",
            "net_portfolio_return",
        )
        if column in portfolio_output.columns
    ]
    ordered_columns = [
        "ts_utc",
        *sorted(return_columns),
        *sorted(weight_columns),
        *execution_columns,
        "portfolio_return",
    ]
    return _csv_ready_frame(portfolio_output.loc[:, ordered_columns].copy())


def _portfolio_equity_curve_frame(portfolio_output: pd.DataFrame) -> pd.DataFrame:
    if "portfolio_equity_curve" not in portfolio_output.columns:
        raise ValueError(
            "portfolio_output must include required column 'portfolio_equity_curve' "
            "to write portfolio_equity_curve.csv."
        )

    return _csv_ready_frame(
        portfolio_output.loc[:, ["ts_utc", "portfolio_equity_curve"]].copy()
    )


def _csv_ready_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="raise").dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    frame = frame.sort_values("ts_utc", kind="stable").reset_index(drop=True)
    return frame


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_manifest(
    *,
    output_dir: Path,
    portfolio_output: pd.DataFrame,
    config: dict[str, object],
    components: list[dict[str, object]],
    metrics: dict[str, object],
    qa_summary: dict[str, object],
    weights_frame: pd.DataFrame,
    returns_frame: pd.DataFrame,
    equity_frame: pd.DataFrame,
    promotion_evaluation: object | None,
) -> dict[str, object]:
    run_id = output_dir.name
    optimizer_metadata = _optimizer_manifest_metadata(
        portfolio_output=portfolio_output,
        config=config,
        metrics=metrics,
    )
    risk_metadata = _risk_manifest_metadata(
        portfolio_output=portfolio_output,
        config=config,
        metrics=metrics,
    )
    execution_metadata = _execution_manifest_metadata(
        portfolio_output=portfolio_output,
        config=config,
        metrics=metrics,
    )
    artifact_inventory = {
        _COMPONENTS_FILENAME: {"path": _COMPONENTS_FILENAME, "rows": len(components)},
        _CONFIG_FILENAME: {"path": _CONFIG_FILENAME},
        _MANIFEST_FILENAME: {"path": _MANIFEST_FILENAME},
        _METRICS_FILENAME: {"path": _METRICS_FILENAME},
        _QA_SUMMARY_FILENAME: {"path": _QA_SUMMARY_FILENAME},
        **(
            {DEFAULT_PROMOTION_ARTIFACT_FILENAME: {"path": DEFAULT_PROMOTION_ARTIFACT_FILENAME}}
            if promotion_evaluation is not None
            else {}
        ),
        _PORTFOLIO_EQUITY_FILENAME: {
            "columns": equity_frame.columns.tolist(),
            "path": _PORTFOLIO_EQUITY_FILENAME,
            "rows": int(len(equity_frame)),
        },
        _PORTFOLIO_RETURNS_FILENAME: {
            "columns": returns_frame.columns.tolist(),
            "path": _PORTFOLIO_RETURNS_FILENAME,
            "rows": int(len(returns_frame)),
        },
        _WEIGHTS_FILENAME: {
            "columns": weights_frame.columns.tolist(),
            "path": _WEIGHTS_FILENAME,
            "rows": int(len(weights_frame)),
        },
    }

    return {
        "alignment_policy": config.get("alignment_policy"),
        "allocator": config.get("allocator"),
        "artifact_files": sorted(artifact_inventory),
        "artifacts": artifact_inventory,
        "artifact_groups": {
            "core": sorted(
                [
                    _COMPONENTS_FILENAME,
                    _CONFIG_FILENAME,
                    _MANIFEST_FILENAME,
                    _METRICS_FILENAME,
                    _PORTFOLIO_EQUITY_FILENAME,
                    _PORTFOLIO_RETURNS_FILENAME,
                    _QA_SUMMARY_FILENAME,
                    _WEIGHTS_FILENAME,
                    *(
                        [DEFAULT_PROMOTION_ARTIFACT_FILENAME]
                        if promotion_evaluation is not None
                        else []
                    ),
                ]
            ),
            "metrics": [_METRICS_FILENAME],
            "qa": sorted(
                [
                    _QA_SUMMARY_FILENAME,
                    *(
                        [DEFAULT_PROMOTION_ARTIFACT_FILENAME]
                        if promotion_evaluation is not None
                        else []
                    ),
                ]
            ),
            "risk": [],
            "simulation": [],
            "walk_forward": [],
        },
        "component_count": len(components),
        "components_path": _COMPONENTS_FILENAME,
        "config_path": _CONFIG_FILENAME,
        "config_snapshot": {
            "execution": config.get("execution"),
            "risk": config.get("risk"),
            "runtime": config.get("runtime"),
            "sanity": config.get("sanity"),
            "strict_mode": config.get("strict_mode"),
            "validation": config.get("validation"),
        },
        "evaluation_config_path": config.get("evaluation_config_path"),
        "evaluation_mode": "single",
        "files_written": len(artifact_inventory),
        "initial_capital": config.get("initial_capital"),
        "metric_summary": metrics,
        "metrics_path": _METRICS_FILENAME,
        "promotion_gate_summary": None if promotion_evaluation is None else promotion_evaluation.summary(),
        "execution": execution_metadata,
        "optimizer": optimizer_metadata,
        "optimizer_method": optimizer_metadata.get("method"),
        "qa_summary_status": qa_summary.get("validation_status"),
        "qa_summary_path": _QA_SUMMARY_FILENAME,
        "portfolio_name": config.get("portfolio_name"),
        "risk": risk_metadata,
        "risk_summary": {
            "target_volatility": risk_metadata["summary"].get("target_volatility"),
            "volatility_targeting_enabled": risk_metadata["summary"].get("volatility_targeting_enabled"),
            "estimated_pre_target_volatility": risk_metadata["summary"].get("estimated_pre_target_volatility"),
            "estimated_post_target_volatility": risk_metadata["summary"].get("estimated_post_target_volatility"),
            "volatility_scaling_factor": risk_metadata["summary"].get("volatility_scaling_factor"),
            "realized_volatility": risk_metadata["summary"].get("realized_volatility"),
            "value_at_risk": risk_metadata["summary"].get("value_at_risk"),
            "conditional_value_at_risk": risk_metadata["summary"].get("conditional_value_at_risk"),
            "max_drawdown": risk_metadata["summary"].get("max_drawdown"),
        },
        "simulation": {
            "artifact_path": None,
            "enabled": False,
            "method": None,
            "num_paths": None,
            "path_length": None,
            "probability_of_loss": None,
            "seed": None,
            "summary_path": None,
        },
        "strict_mode": config.get("strict_mode"),
        "row_counts": {
            "components": len(components),
            "portfolio_equity_curve": int(len(equity_frame)),
            "portfolio_returns": int(len(returns_frame)),
            "weights": int(len(weights_frame)),
        },
        "run_id": run_id,
        "timestamp": _utc_timestamp_from_run_id(run_id),
    }


def _optimizer_manifest_metadata(
    *,
    portfolio_output: pd.DataFrame,
    config: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    constructor_payload = portfolio_output.attrs.get("portfolio_constructor", {})
    optimizer_payload = (
        constructor_payload.get("optimizer")
        if isinstance(constructor_payload, dict)
        else None
    )
    configured_optimizer = config.get("optimizer") if isinstance(config.get("optimizer"), dict) else {}
    optimizer_config = {}
    if isinstance(optimizer_payload, dict) and isinstance(optimizer_payload.get("config"), dict):
        optimizer_config = _normalize_mapping(dict(optimizer_payload["config"]), owner="optimizer")
    elif isinstance(configured_optimizer, dict):
        optimizer_config = _normalize_mapping(dict(configured_optimizer), owner="optimizer")

    diagnostics = {}
    if isinstance(optimizer_payload, dict) and isinstance(optimizer_payload.get("diagnostics"), dict):
        diagnostics = _normalize_mapping(
            dict(optimizer_payload["diagnostics"]),
            owner="optimizer.diagnostics",
        )

    method = optimizer_config.get("method")
    return {
        "config": optimizer_config or None,
        "constraint_summary": {
            "full_investment": optimizer_config.get("full_investment"),
            "leverage_ceiling": optimizer_config.get("leverage_ceiling"),
            "long_only": optimizer_config.get("long_only"),
            "max_single_weight": optimizer_config.get("max_single_weight"),
            "max_turnover": optimizer_config.get("max_turnover"),
            "max_weight": optimizer_config.get("max_weight"),
            "min_weight": optimizer_config.get("min_weight"),
            "target_weight_sum": optimizer_config.get("target_weight_sum"),
        },
        "diagnostics": diagnostics or None,
        "diagnostic_summary": {
            "converged": diagnostics.get("converged"),
            "gross_exposure": diagnostics.get("gross_exposure", metrics.get("max_leverage")),
            "iterations": diagnostics.get("iterations"),
            "max_single_weight": diagnostics.get("max_single_weight", metrics.get("max_single_weight")),
            "net_exposure": diagnostics.get("net_exposure"),
            "objective_expected_return": diagnostics.get("objective_expected_return"),
            "objective_sharpe_ratio": diagnostics.get("objective_sharpe_ratio"),
            "objective_volatility": diagnostics.get("objective_volatility"),
            "observation_count": diagnostics.get("observation_count"),
            "strategy_count": diagnostics.get("strategy_count"),
            "turnover_vs_previous": diagnostics.get("turnover_vs_previous"),
            "weight_sum": diagnostics.get("weight_sum", optimizer_config.get("target_weight_sum")),
        },
        "method": method,
    }


def _risk_manifest_metadata(
    *,
    portfolio_output: pd.DataFrame,
    config: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    resolved_risk = resolve_portfolio_risk_config(
        config.get("risk") if isinstance(config.get("risk"), dict) else None
    )
    equity_curve = (
        portfolio_output["portfolio_equity_curve"]
        if "portfolio_equity_curve" in portfolio_output.columns
        else None
    )
    weight_diagnostics = metrics.get("max_leverage")
    risk_summary = summarize_portfolio_risk(
        portfolio_output["portfolio_return"],
        equity_curve=equity_curve,
        config=resolved_risk,
        periods_per_year=_periods_per_year_from_timeframe(str(config.get("timeframe"))),
        leverage_ceiling=None if weight_diagnostics is None else float(weight_diagnostics),
    )
    return {
        "config": _normalize_mapping(resolved_risk.to_dict(), owner="risk.config"),
        "rolling_volatility": risk_summary["rolling_volatility"],
        "summary": {
            "conditional_value_at_risk": metrics.get("conditional_value_at_risk"),
            "conditional_value_at_risk_confidence_level": metrics.get("conditional_value_at_risk_confidence_level"),
            "current_drawdown": metrics.get("current_drawdown"),
            "current_drawdown_duration": metrics.get("current_drawdown_duration"),
            "estimated_post_target_volatility": metrics.get("estimated_post_target_volatility"),
            "estimated_pre_target_volatility": metrics.get("estimated_pre_target_volatility"),
            "latest_rolling_volatility": metrics.get("latest_rolling_volatility"),
            "max_drawdown": metrics.get("max_drawdown"),
            "max_drawdown_duration": metrics.get("max_drawdown_duration"),
            "periods_per_year": risk_summary.get("periods_per_year"),
            "realized_volatility": metrics.get("realized_volatility"),
            "rolling_volatility_latest": metrics.get("rolling_volatility_latest"),
            "rolling_volatility_max": metrics.get("rolling_volatility_max"),
            "rolling_volatility_mean": metrics.get("rolling_volatility_mean"),
            "rolling_volatility_window": metrics.get("rolling_volatility_window"),
            "target_volatility": metrics.get("target_volatility"),
            "value_at_risk": metrics.get("value_at_risk"),
            "value_at_risk_confidence_level": metrics.get("value_at_risk_confidence_level"),
            "volatility_scaling_factor": metrics.get("volatility_scaling_factor"),
            "volatility_target_scale": metrics.get("volatility_target_scale"),
            "volatility_target_scale_capped": metrics.get("volatility_target_scale_capped"),
            "volatility_targeting_enabled": metrics.get("volatility_targeting_enabled"),
        },
        "tail_risk": risk_summary["tail_risk"],
        "volatility_targeting": risk_summary["volatility_targeting"],
        "operational_volatility_targeting": _normalize_mapping(
            {
                "enabled": metrics.get("volatility_targeting_enabled"),
                "estimated_pre_target_volatility": metrics.get("estimated_pre_target_volatility"),
                "estimated_post_target_volatility": metrics.get("estimated_post_target_volatility"),
                "target_volatility": metrics.get("target_volatility"),
                "volatility_scaling_factor": metrics.get("volatility_scaling_factor"),
            },
            owner="risk.operational_volatility_targeting",
        ),
    }


def _execution_manifest_metadata(
    *,
    portfolio_output: pd.DataFrame,
    config: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    constructor_payload = portfolio_output.attrs.get("portfolio_constructor", {})
    execution_summary = (
        constructor_payload.get("execution_summary")
        if isinstance(constructor_payload, dict) and isinstance(constructor_payload.get("execution_summary"), dict)
        else portfolio_output.attrs.get("portfolio_execution", {})
    )
    normalized_execution_summary = (
        _normalize_mapping(dict(execution_summary), owner="execution.summary")
        if isinstance(execution_summary, dict)
        else {}
    )
    execution_config = config.get("execution") if isinstance(config.get("execution"), dict) else {}
    return {
        "config": _normalize_mapping(dict(execution_config), owner="execution.config") if execution_config else {},
        "summary": {
            "gross_total_return": metrics.get("gross_total_return"),
            "net_total_return": metrics.get("net_total_return", metrics.get("total_return")),
            "execution_drag_total_return": metrics.get("execution_drag_total_return"),
            "total_transaction_cost": metrics.get("total_transaction_cost"),
            "total_fixed_fee": metrics.get("total_fixed_fee"),
            "total_slippage_cost": metrics.get("total_slippage_cost"),
            "total_execution_friction": metrics.get("total_execution_friction"),
            "turnover": metrics.get("turnover"),
            "rebalance_count": metrics.get("rebalance_count"),
        },
        "model_summary": normalized_execution_summary or None,
    }


def _periods_per_year_from_timeframe(timeframe: str) -> int:
    normalized = timeframe.strip().lower()
    if normalized in {"1d", "1day", "day", "daily"}:
        return 252
    if normalized in {"1m", "1min", "1minute", "minute", "minutes"}:
        return 98_280
    raise ValueError(f"Unsupported portfolio artifact timeframe: {timeframe!r}.")


def build_portfolio_registry_metadata(
    *,
    config: dict[str, object],
    metrics: dict[str, object],
    manifest: dict[str, object],
    start_ts: str,
    end_ts: str,
    split_count: int | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    simulation = manifest.get("simulation")
    if not isinstance(simulation, dict):
        simulation = {}
    optimizer = manifest.get("optimizer")
    if not isinstance(optimizer, dict):
        optimizer = {}
    payload = {
        "portfolio_name": config.get("portfolio_name"),
        "allocator_name": config.get("allocator"),
        "optimizer_method": optimizer.get("method"),
        "timeframe": config.get("timeframe"),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "evaluation_config_path": config.get("evaluation_config_path"),
        "split_count": split_count,
        "simulation": {
            "enabled": bool(simulation.get("enabled")),
            "method": simulation.get("method"),
            "num_paths": simulation.get("num_paths"),
            "path_length": simulation.get("path_length"),
            "probability_of_loss": simulation.get("probability_of_loss"),
            "summary_path": simulation.get("summary_path"),
        },
        "manifest": {
            "evaluation_mode": manifest.get("evaluation_mode"),
            "qa_summary_status": manifest.get("qa_summary_status"),
            "artifact_path": manifest.get("simulation", {}).get("artifact_path")
            if isinstance(manifest.get("simulation"), dict)
            else None,
        },
        "risk_summary": manifest.get("risk_summary"),
        "metrics_summary": {
            "total_return": metrics.get("total_return"),
            "gross_total_return": metrics.get("gross_total_return"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "volatility_targeting_enabled": metrics.get("volatility_targeting_enabled"),
            "target_volatility": metrics.get("target_volatility"),
            "estimated_pre_target_volatility": metrics.get("estimated_pre_target_volatility"),
            "estimated_post_target_volatility": metrics.get("estimated_post_target_volatility"),
            "volatility_scaling_factor": metrics.get("volatility_scaling_factor"),
            "realized_volatility": metrics.get("realized_volatility"),
            "value_at_risk": metrics.get("value_at_risk"),
            "conditional_value_at_risk": metrics.get("conditional_value_at_risk"),
            "total_execution_friction": metrics.get("total_execution_friction"),
            "execution_drag_total_return": metrics.get("execution_drag_total_return"),
        },
        "execution_summary": manifest.get("execution", {}).get("summary")
        if isinstance(manifest.get("execution"), dict)
        else None,
        "promotion_gate_summary": manifest.get("promotion_gate_summary"),
    }
    if isinstance(extra_metadata, dict):
        for key, value in sorted(extra_metadata.items()):
            payload[key] = value
    return payload


def register_validated_portfolio_run(
    *,
    registry_path: str | Path,
    run_id: str,
    config: dict[str, object],
    components: list[dict[str, object]],
    metrics: dict[str, object],
    artifact_path: str,
    manifest: dict[str, object],
    start_ts: str,
    end_ts: str,
    split_count: int | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> None:
    register_portfolio_run(
        registry_path=registry_path,
        run_id=run_id,
        config=dict(config),
        components=[dict(component) for component in components],
        metrics=dict(metrics),
        artifact_path=artifact_path,
        metadata=build_portfolio_registry_metadata(
            config=config,
            metrics=metrics,
            manifest=manifest,
            start_ts=start_ts,
            end_ts=end_ts,
            split_count=split_count,
            extra_metadata=extra_metadata,
        ),
    )


def _utc_timestamp_from_run_id(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    year = 2000 + int(digest[0:2], 16) % 25
    month = (int(digest[2:4], 16) % 12) + 1
    day = (int(digest[4:6], 16) % 28) + 1
    hour = int(digest[6:8], 16) % 24
    minute = int(digest[8:10], 16) % 60
    second = int(digest[10:12], 16) % 60
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _format_timestamp(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _promotion_gate_config(config: dict[str, object]) -> dict[str, object] | None:
    payload = config.get("promotion_gates")
    return dict(payload) if isinstance(payload, dict) else None


__all__ = [
    "build_portfolio_registry_metadata",
    "register_validated_portfolio_run",
    "write_portfolio_artifacts",
]
