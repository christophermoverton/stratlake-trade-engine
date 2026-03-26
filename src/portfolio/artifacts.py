from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.portfolio.contracts import (
    PortfolioContractError,
    validate_portfolio_output,
)
from src.portfolio.qa import run_portfolio_qa
from src.research.registry import canonicalize_value

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

    weights_frame = _weights_frame(normalized_portfolio_output)
    returns_frame = _portfolio_returns_frame(normalized_portfolio_output)
    equity_frame = _portfolio_equity_curve_frame(normalized_portfolio_output)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(resolved_output_dir / _CONFIG_FILENAME, normalized_config)
    _write_json(resolved_output_dir / _COMPONENTS_FILENAME, {"components": normalized_components})
    _write_csv(resolved_output_dir / _WEIGHTS_FILENAME, weights_frame)
    _write_csv(resolved_output_dir / _PORTFOLIO_RETURNS_FILENAME, returns_frame)
    _write_csv(resolved_output_dir / _PORTFOLIO_EQUITY_FILENAME, equity_frame)
    _write_json(resolved_output_dir / _METRICS_FILENAME, normalized_metrics)
    qa_summary = run_portfolio_qa(
        normalized_portfolio_output,
        normalized_metrics,
        normalized_config,
        artifacts_dir=resolved_output_dir,
        run_id=resolved_output_dir.name,
        strict=True,
    )

    manifest = _build_manifest(
        output_dir=resolved_output_dir,
        config=normalized_config,
        components=normalized_components,
        metrics=normalized_metrics,
        qa_summary=qa_summary,
        weights_frame=weights_frame,
        returns_frame=returns_frame,
        equity_frame=equity_frame,
    )
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

    ordered_columns = [
        "ts_utc",
        *sorted(return_columns),
        *sorted(weight_columns),
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
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_manifest(
    *,
    output_dir: Path,
    config: dict[str, object],
    components: list[dict[str, object]],
    metrics: dict[str, object],
    qa_summary: dict[str, object],
    weights_frame: pd.DataFrame,
    returns_frame: pd.DataFrame,
    equity_frame: pd.DataFrame,
) -> dict[str, object]:
    run_id = output_dir.name
    artifact_inventory = {
        _COMPONENTS_FILENAME: {"path": _COMPONENTS_FILENAME, "rows": len(components)},
        _CONFIG_FILENAME: {"path": _CONFIG_FILENAME},
        _MANIFEST_FILENAME: {"path": _MANIFEST_FILENAME},
        _METRICS_FILENAME: {"path": _METRICS_FILENAME},
        _QA_SUMMARY_FILENAME: {"path": _QA_SUMMARY_FILENAME},
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
        "component_count": len(components),
        "files_written": len(artifact_inventory),
        "initial_capital": config.get("initial_capital"),
        "metric_summary": metrics,
        "qa_summary_status": qa_summary.get("validation_status"),
        "portfolio_name": config.get("portfolio_name"),
        "row_counts": {
            "components": len(components),
            "portfolio_equity_curve": int(len(equity_frame)),
            "portfolio_returns": int(len(returns_frame)),
            "weights": int(len(weights_frame)),
        },
        "run_id": run_id,
        "timestamp": _utc_timestamp_from_run_id(run_id),
    }


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


__all__ = ["write_portfolio_artifacts"]
