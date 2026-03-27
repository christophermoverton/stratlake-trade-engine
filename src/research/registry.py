from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

REGISTRY_FILENAME = "registry.jsonl"
LOCK_SUFFIX = ".lock"
LOCK_TIMEOUT_SECONDS = 5.0
LOCK_POLL_INTERVAL_SECONDS = 0.05
STRATEGY_RUN_TYPE = "strategy"
PORTFOLIO_RUN_TYPE = "portfolio"
_VALID_RUN_TYPES = frozenset({STRATEGY_RUN_TYPE, PORTFOLIO_RUN_TYPE})


class RegistryError(RuntimeError):
    """Raised when the experiment registry cannot be read or updated safely."""


def default_registry_path(artifacts_root: Path) -> Path:
    """Return the JSONL registry path for a strategy artifact root."""

    return artifacts_root / REGISTRY_FILENAME


def canonicalize_value(value: Any) -> Any:
    """Return a recursively normalized value with deterministic mapping key order."""

    if isinstance(value, Mapping):
        return {key: canonicalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [canonicalize_value(item) for item in value]
    if isinstance(value, list):
        return [canonicalize_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise RegistryError("Registry values must not include NaN or infinite floats.")
        return value
    return value


def serialize_canonical_json(value: Any) -> str:
    """Serialize a value to a canonical JSON string for deterministic snapshots."""

    return json.dumps(canonicalize_value(value), sort_keys=True, separators=(",", ":"))


def load_registry(path: Path) -> list[dict[str, Any]]:
    """Load all registry entries from a JSONL registry file."""

    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            payload = line.strip()
            if not payload:
                continue

            try:
                entry = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RegistryError(
                    f"Registry file '{path}' contains invalid JSON on line {line_number}."
                ) from exc

            if not isinstance(entry, dict):
                raise RegistryError(
                    f"Registry file '{path}' contains a non-object entry on line {line_number}."
                )

            if "run_type" not in entry:
                entry["run_type"] = STRATEGY_RUN_TYPE
            entries.append(entry)
    return entries


def filter_by_run_type(
    entries: Iterable[dict[str, Any]],
    run_type: str,
) -> list[dict[str, Any]]:
    """Return registry entries for one run type, defaulting legacy rows to strategy."""

    _validate_run_type(run_type)
    return [
        entry
        for entry in entries
        if _entry_run_type(entry) == run_type
    ]


def filter_by_strategy_name(
    entries: Iterable[dict[str, Any]], strategy_name: str
) -> list[dict[str, Any]]:
    """Return registry entries for one strategy name."""

    return [entry for entry in entries if entry.get("strategy_name") == strategy_name]


def filter_by_portfolio_name(
    entries: Iterable[dict[str, Any]],
    portfolio_name: str,
) -> list[dict[str, Any]]:
    """Return portfolio registry entries for one portfolio name."""

    return [
        entry
        for entry in entries
        if _entry_run_type(entry) == PORTFOLIO_RUN_TYPE and entry.get("portfolio_name") == portfolio_name
    ]


def filter_by_allocator_name(
    entries: Iterable[dict[str, Any]],
    allocator_name: str,
) -> list[dict[str, Any]]:
    """Return portfolio registry entries for one allocator name."""

    return [
        entry
        for entry in entries
        if _entry_run_type(entry) == PORTFOLIO_RUN_TYPE and entry.get("allocator_name") == allocator_name
    ]


def filter_by_metric_threshold(
    entries: Iterable[dict[str, Any]],
    metric_name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[dict[str, Any]]:
    """Return registry entries whose summary metric lies within the requested bounds."""

    filtered: list[dict[str, Any]] = []
    for entry in entries:
        metrics_summary = _entry_metrics(entry)
        if not isinstance(metrics_summary, dict):
            continue

        value = metrics_summary.get(metric_name)
        if not isinstance(value, int | float):
            continue
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        filtered.append(entry)
    return filtered


def generate_portfolio_run_id(
    *,
    portfolio_name: str,
    allocator_name: str,
    component_run_ids: Iterable[str],
    timeframe: str,
    start_ts: str,
    end_ts: str,
    config: Mapping[str, Any],
    evaluation_config_path: str | Path | None = None,
) -> str:
    """Return a deterministic run id for a logical portfolio run."""

    normalized_portfolio_name = _normalize_required_string(portfolio_name, field_name="portfolio_name")
    normalized_allocator_name = _normalize_required_string(allocator_name, field_name="allocator_name")
    normalized_timeframe = _normalize_required_string(timeframe, field_name="timeframe")
    normalized_start_ts = _normalize_required_string(start_ts, field_name="start_ts")
    normalized_end_ts = _normalize_required_string(end_ts, field_name="end_ts")
    normalized_component_run_ids = _normalize_component_run_ids(component_run_ids)

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping.")

    payload = {
        "run_type": PORTFOLIO_RUN_TYPE,
        "portfolio_name": normalized_portfolio_name,
        "allocator_name": normalized_allocator_name,
        "component_run_ids": normalized_component_run_ids,
        "timeframe": normalized_timeframe,
        "start_ts": normalized_start_ts,
        "end_ts": normalized_end_ts,
        "config": canonicalize_value(dict(config)),
        "evaluation_config_path": _normalize_optional_path(evaluation_config_path),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize_name_component(normalized_portfolio_name)}_portfolio_{digest}"


def register_portfolio_run(
    registry_path: str | Path,
    run_id: str,
    config: dict[str, Any],
    components: list[dict[str, Any]],
    metrics: dict[str, Any],
    artifact_path: str,
    metadata: dict[str, Any],
) -> None:
    """Append one deterministic portfolio run if it is not already registered."""

    resolved_registry_path = Path(registry_path)
    normalized_run_id = _normalize_required_string(run_id, field_name="run_id")
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary.")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")
    normalized_artifact_path = _normalize_required_string(artifact_path, field_name="artifact_path")

    entry = _build_portfolio_registry_entry(
        run_id=normalized_run_id,
        config=config,
        components=components,
        metrics=metrics,
        artifact_path=normalized_artifact_path,
        metadata=metadata,
    )

    resolved_registry_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(canonicalize_value(entry), separators=(",", ":"), sort_keys=False) + "\n"

    with _registry_lock(resolved_registry_path):
        existing_entries = load_registry(resolved_registry_path)
        for existing in existing_entries:
            if existing.get("run_id") != normalized_run_id:
                continue
            if canonicalize_value(existing) != canonicalize_value(entry):
                raise RegistryError(
                    f"Registry already contains conflicting entry for run_id '{normalized_run_id}'."
                )
            return

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        try:
            descriptor = os.open(str(resolved_registry_path), flags)
        except OSError as exc:
            raise IOError(f"Failed to open registry path '{resolved_registry_path}' for append.") from exc
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        except OSError as exc:
            raise IOError(f"Failed to append portfolio registry entry to '{resolved_registry_path}'.") from exc
        finally:
            os.close(descriptor)


def append_registry_entry(path: Path, entry: Mapping[str, Any]) -> None:
    """Append one registry entry after validating run-id uniqueness."""

    canonical_entry = canonicalize_value(entry)
    if not isinstance(canonical_entry, dict):
        raise RegistryError("Registry entries must serialize to a JSON object.")

    run_id = canonical_entry.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RegistryError("Registry entries must include a non-empty string run_id.")

    line = json.dumps(canonical_entry, separators=(",", ":"), sort_keys=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    with _registry_lock(path):
        existing_ids = {existing.get("run_id") for existing in load_registry(path)}
        if run_id in existing_ids:
            raise RegistryError(f"Registry already contains run_id '{run_id}'.")

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        descriptor = os.open(str(path), flags)
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        except OSError as exc:
            raise RegistryError(f"Failed to append registry entry to '{path}'.") from exc
        finally:
            os.close(descriptor)


def upsert_registry_entry(path: Path, entry: Mapping[str, Any]) -> None:
    """Insert or replace one registry entry by deterministic run-id."""

    canonical_entry = canonicalize_value(entry)
    if not isinstance(canonical_entry, dict):
        raise RegistryError("Registry entries must serialize to a JSON object.")

    run_id = canonical_entry.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RegistryError("Registry entries must include a non-empty string run_id.")

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(canonical_entry, separators=(",", ":"), sort_keys=False) + "\n"

    with _registry_lock(path):
        existing_entries = [
            existing
            for existing in load_registry(path)
            if existing.get("run_id") != run_id
        ]
        serialized = [
            json.dumps(canonicalize_value(existing), separators=(",", ":"), sort_keys=False)
            for existing in existing_entries
        ]
        serialized.append(line.rstrip("\n"))
        payload = "\n".join(serialized)
        if payload:
            payload += "\n"

        try:
            path.write_text(payload, encoding="utf-8")
        except OSError as exc:
            raise RegistryError(f"Failed to upsert registry entry to '{path}'.") from exc


@contextmanager
def _registry_lock(path: Path) -> Iterable[None]:
    lock_path = path.with_name(f"{path.name}{LOCK_SUFFIX}")
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS

    while True:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise RegistryError(
                    f"Timed out waiting for registry lock '{lock_path.name}'."
                ) from exc
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)

    try:
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _entry_run_type(entry: Mapping[str, Any]) -> str:
    run_type = entry.get("run_type", STRATEGY_RUN_TYPE)
    if not isinstance(run_type, str):
        return STRATEGY_RUN_TYPE
    return run_type


def _entry_metrics(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    metrics_summary = entry.get("metrics_summary")
    if isinstance(metrics_summary, dict):
        return metrics_summary

    metrics = entry.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return None


def _validate_run_type(run_type: str) -> None:
    if run_type not in _VALID_RUN_TYPES:
        formatted = ", ".join(sorted(_VALID_RUN_TYPES))
        raise ValueError(f"run_type must be one of: {formatted}.")


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return _normalize_required_string(value, field_name="evaluation_config_path")


def _normalize_component_run_ids(component_run_ids: Iterable[str]) -> list[str]:
    normalized = sorted(
        _normalize_required_string(run_id, field_name="component_run_ids")
        for run_id in component_run_ids
    )
    if not normalized:
        raise ValueError("component_run_ids must contain at least one run id.")
    return normalized


def _normalize_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(components, list):
        raise ValueError("components must be provided as a list of dictionaries.")
    if not components:
        raise ValueError("components must contain at least one portfolio component.")

    normalized: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(f"components[{index}] must be a dictionary.")

        run_id = _normalize_required_string(
            component.get("run_id"),
            field_name=f"components[{index}].run_id",
        )
        strategy_name = _normalize_required_string(
            component.get("strategy_name"),
            field_name=f"components[{index}].strategy_name",
        )
        if run_id in seen_run_ids:
            raise ValueError(f"components must have unique run_id values. Duplicate: {run_id!r}.")
        seen_run_ids.add(run_id)

        normalized_component = canonicalize_value(component)
        if not isinstance(normalized_component, dict):
            raise ValueError(f"components[{index}] must serialize to an object.")
        normalized_component["run_id"] = run_id
        normalized_component["strategy_name"] = strategy_name
        normalized.append(normalized_component)

    return sorted(
        normalized,
        key=lambda component: (str(component["run_id"]), str(component["strategy_name"])),
    )


def _build_portfolio_registry_entry(
    *,
    run_id: str,
    config: dict[str, Any],
    components: list[dict[str, Any]],
    metrics: dict[str, Any],
    artifact_path: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    normalized_components = _normalize_components(components)
    component_run_ids = [str(component["run_id"]) for component in normalized_components]
    component_strategy_names = sorted({str(component["strategy_name"]) for component in normalized_components})

    portfolio_name = _normalize_required_string(
        metadata.get("portfolio_name", config.get("portfolio_name")),
        field_name="portfolio_name",
    )
    allocator_name = _normalize_required_string(
        metadata.get("allocator_name", config.get("allocator_name", config.get("allocator"))),
        field_name="allocator_name",
    )
    timeframe = _normalize_required_string(
        metadata.get("timeframe", config.get("timeframe")),
        field_name="timeframe",
    )
    start_ts = _normalize_required_string(
        metadata.get("start_ts", metadata.get("start")),
        field_name="start_ts",
    )
    end_ts = _normalize_required_string(
        metadata.get("end_ts", metadata.get("end")),
        field_name="end_ts",
    )
    evaluation_config_path = _normalize_optional_path(
        metadata.get("evaluation_config_path", config.get("evaluation_config_path"))
    )
    split_count = metadata.get("split_count")
    if split_count is not None and not isinstance(split_count, int):
        raise ValueError("split_count must be an integer when provided.")

    timestamp = metadata.get("timestamp")
    if timestamp is None:
        timestamp = _stable_timestamp_from_run_id(run_id)
    else:
        timestamp = _normalize_required_string(timestamp, field_name="timestamp")

    extra_metadata = {
        key: canonicalize_value(value)
        for key, value in metadata.items()
        if key not in {
            "timestamp",
            "portfolio_name",
            "allocator_name",
            "timeframe",
            "start_ts",
            "start",
            "end_ts",
            "end",
            "evaluation_config_path",
            "split_count",
        }
    }
    metrics_summary = _portfolio_metrics_summary(metrics)
    optimizer_summary = _portfolio_optimizer_summary(config, extra_metadata)
    risk_summary = _portfolio_risk_summary(metrics)
    simulation_summary = _portfolio_simulation_summary(config, extra_metadata)

    return {
        "run_id": run_id,
        "run_type": PORTFOLIO_RUN_TYPE,
        "timestamp": timestamp,
        "portfolio_name": portfolio_name,
        "allocator_name": allocator_name,
        "component_run_ids": component_run_ids,
        "component_strategy_names": component_strategy_names,
        "timeframe": timeframe,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "artifact_path": artifact_path,
        "metrics_summary": metrics_summary,
        "metrics": canonicalize_value(metrics),
        "optimizer_method": optimizer_summary.get("method"),
        "optimizer_constraint_summary": optimizer_summary.get("constraints"),
        "weight_sum_target": optimizer_summary.get("target_weight_sum"),
        "long_only": optimizer_summary.get("long_only"),
        "leverage_ceiling": optimizer_summary.get("leverage_ceiling"),
        "risk_summary": risk_summary,
        "simulation_enabled": simulation_summary.get("enabled"),
        "simulation_method": simulation_summary.get("method"),
        "simulation_summary": simulation_summary,
        "evaluation_config_path": evaluation_config_path,
        "split_count": split_count,
        "config": canonicalize_value(config),
        "components": normalized_components,
        "metadata": extra_metadata,
    }


def _portfolio_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    summary_keys = (
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "realized_volatility",
        "target_volatility",
        "value_at_risk",
        "conditional_value_at_risk",
        "turnover",
        "trade_count",
        "total_execution_friction",
        "exposure_pct",
        "sanity_issue_count",
        "sanity_warning_count",
        "flagged_split_count",
    )
    return {
        key: canonicalize_value(metrics[key])
        for key in summary_keys
        if key in metrics
    }


def _portfolio_optimizer_summary(
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, dict):
        optimizer = {}
    return {
        "constraints": {
            "full_investment": optimizer.get("full_investment"),
            "leverage_ceiling": optimizer.get("leverage_ceiling"),
            "long_only": optimizer.get("long_only"),
            "max_single_weight": optimizer.get("max_single_weight"),
            "max_turnover": optimizer.get("max_turnover"),
            "max_weight": optimizer.get("max_weight"),
            "min_weight": optimizer.get("min_weight"),
            "target_weight_sum": optimizer.get("target_weight_sum"),
        },
        "leverage_ceiling": optimizer.get("leverage_ceiling"),
        "long_only": optimizer.get("long_only"),
        "method": metadata.get("optimizer_method", optimizer.get("method")),
        "target_weight_sum": optimizer.get("target_weight_sum"),
    }


def _portfolio_risk_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "conditional_value_at_risk": metrics.get("conditional_value_at_risk"),
        "conditional_value_at_risk_confidence_level": metrics.get(
            "conditional_value_at_risk_confidence_level"
        ),
        "max_drawdown": metrics.get("max_drawdown"),
        "realized_volatility": metrics.get("realized_volatility"),
        "target_volatility": metrics.get("target_volatility"),
        "value_at_risk": metrics.get("value_at_risk"),
        "value_at_risk_confidence_level": metrics.get("value_at_risk_confidence_level"),
    }


def _portfolio_simulation_summary(
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    simulation_config = config.get("simulation")
    simulation_metadata = metadata.get("simulation")
    if not isinstance(simulation_config, dict):
        simulation_config = {}
    if not isinstance(simulation_metadata, dict):
        simulation_metadata = {}
    enabled = bool(simulation_config) or bool(simulation_metadata.get("enabled"))
    return {
        "enabled": enabled,
        "method": simulation_metadata.get("method", simulation_config.get("method")),
        "num_paths": simulation_metadata.get("num_paths", simulation_config.get("num_paths")),
        "path_length": simulation_metadata.get("path_length", simulation_config.get("path_length")),
        "probability_of_loss": simulation_metadata.get("probability_of_loss"),
        "summary_path": simulation_metadata.get("summary_path"),
    }


def _sanitize_name_component(name: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in name.strip().lower())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    return normalized or "portfolio"


def _stable_timestamp_from_run_id(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    year = 2000 + int(digest[0:2], 16) % 25
    month = (int(digest[2:4], 16) % 12) + 1
    day = (int(digest[4:6], 16) % 28) + 1
    hour = int(digest[6:8], 16) % 24
    minute = int(digest[8:10], 16) % 60
    second = int(digest[10:12], 16) % 60
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
