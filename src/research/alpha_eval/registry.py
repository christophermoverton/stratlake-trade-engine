from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.alpha_eval.artifacts import DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT
from src.research.alpha_eval.evaluator import AlphaEvaluationResult
from src.research.registry import (
    ALPHA_EVALUATION_RUN_TYPE,
    canonicalize_value,
    default_registry_path,
    load_registry,
    stable_timestamp_from_run_id,
    upsert_registry_entry,
)

_SUMMARY_KEYS = (
    "mean_ic",
    "ic_ir",
    "mean_rank_ic",
    "rank_ic_ir",
    "n_periods",
)


def alpha_evaluation_registry_path(
    artifacts_root: str | Path = DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT,
) -> Path:
    """Return the persistent JSONL registry path for alpha evaluation runs."""

    return default_registry_path(Path(artifacts_root))


def build_alpha_evaluation_registry_entry(
    *,
    run_id: str,
    alpha_name: str,
    effective_config: Mapping[str, Any],
    evaluation_result: AlphaEvaluationResult,
    artifact_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic alpha-evaluation registry entry."""

    normalized_run_id = _normalize_required_string(run_id, field_name="run_id")
    normalized_alpha_name = _normalize_required_string(alpha_name, field_name="alpha_name")
    if not isinstance(effective_config, Mapping):
        raise ValueError("effective_config must be a mapping.")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping.")

    resolved_artifact_dir = Path(artifact_dir)
    summary_metrics = {
        key: evaluation_result.summary.get(key)
        for key in _SUMMARY_KEYS
    }
    metadata = dict(evaluation_result.metadata)
    manifest_payload = dict(manifest)

    return {
        "run_id": normalized_run_id,
        "run_type": ALPHA_EVALUATION_RUN_TYPE,
        "timestamp": stable_timestamp_from_run_id(normalized_run_id),
        "alpha_name": normalized_alpha_name,
        "dataset": effective_config.get("dataset"),
        "timeframe": _coerce_optional_string(
            manifest_payload.get("timeframe", metadata.get("timeframe"))
        ),
        "evaluation_horizon": _coerce_optional_int(effective_config.get("alpha_horizon")),
        "prediction_column": evaluation_result.prediction_column,
        "forward_return_column": evaluation_result.forward_return_column,
        "min_cross_section_size": evaluation_result.min_cross_section_size,
        "data_range": {
            "start": _format_optional_timestamp(metadata.get("ts_utc_start")),
            "end": _format_optional_timestamp(metadata.get("ts_utc_end")),
        },
        "artifact_path": resolved_artifact_dir.as_posix(),
        "ic_timeseries_path": resolved_artifact_dir.joinpath(
            str(manifest_payload.get("timeseries_path", "ic_timeseries.csv"))
        ).as_posix(),
        "metrics_path": resolved_artifact_dir.joinpath(
            str(manifest_payload.get("metrics_path", "alpha_metrics.json"))
        ).as_posix(),
        "manifest_path": resolved_artifact_dir.joinpath("manifest.json").as_posix(),
        "artifact_files": list(manifest_payload.get("artifact_files", [])),
        "metrics_summary": summary_metrics,
        "promotion_status": (
            manifest_payload.get("promotion_gate_summary", {}).get("promotion_status")
            if isinstance(manifest_payload.get("promotion_gate_summary"), dict)
            else None
        ),
        "promotion_gate_summary": (
            canonicalize_value(dict(manifest_payload["promotion_gate_summary"]))
            if isinstance(manifest_payload.get("promotion_gate_summary"), dict)
            else None
        ),
        "row_count": evaluation_result.row_count,
        "timestamp_count": evaluation_result.timestamp_count,
        "symbol_count": evaluation_result.symbol_count,
        "config": canonicalize_value(dict(effective_config)),
        "manifest": canonicalize_value(manifest_payload),
        "metadata": canonicalize_value(metadata),
    }


def register_alpha_evaluation_run(
    *,
    run_id: str,
    alpha_name: str,
    effective_config: Mapping[str, Any],
    evaluation_result: AlphaEvaluationResult,
    artifact_dir: str | Path,
    manifest: Mapping[str, Any],
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Upsert one completed alpha-evaluation run into the persistent registry."""

    entry = build_alpha_evaluation_registry_entry(
        run_id=run_id,
        alpha_name=alpha_name,
        effective_config=effective_config,
        evaluation_result=evaluation_result,
        artifact_dir=artifact_dir,
        manifest=manifest,
    )
    resolved_registry_path = (
        alpha_evaluation_registry_path(Path(artifact_dir).parent)
        if registry_path is None
        else Path(registry_path)
    )
    upsert_registry_entry(resolved_registry_path, entry)
    return entry


def load_alpha_evaluation_registry(
    artifacts_root: str | Path = DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT,
) -> list[dict[str, Any]]:
    """Load persisted alpha-evaluation registry entries."""

    return [
        entry
        for entry in load_registry(alpha_evaluation_registry_path(artifacts_root))
        if entry.get("run_type") == ALPHA_EVALUATION_RUN_TYPE
    ]


def filter_by_alpha_name(
    entries: Iterable[Mapping[str, Any]],
    alpha_name: str,
) -> list[dict[str, Any]]:
    """Return alpha-evaluation registry entries matching one alpha name."""

    normalized_alpha_name = _normalize_required_string(alpha_name, field_name="alpha_name")
    return [
        dict(entry)
        for entry in entries
        if entry.get("run_type") == ALPHA_EVALUATION_RUN_TYPE
        and entry.get("alpha_name") == normalized_alpha_name
    ]


def filter_by_timeframe(
    entries: Iterable[Mapping[str, Any]],
    timeframe: str,
) -> list[dict[str, Any]]:
    """Return alpha-evaluation registry entries matching one timeframe."""

    normalized_timeframe = _normalize_required_string(timeframe, field_name="timeframe")
    return [
        dict(entry)
        for entry in entries
        if entry.get("run_type") == ALPHA_EVALUATION_RUN_TYPE
        and entry.get("timeframe") == normalized_timeframe
    ]


def get_alpha_evaluation_run(
    entries: Iterable[Mapping[str, Any]],
    run_id: str,
) -> dict[str, Any] | None:
    """Return one alpha-evaluation registry entry by run id when present."""

    normalized_run_id = _normalize_required_string(run_id, field_name="run_id")
    for entry in entries:
        if entry.get("run_type") == ALPHA_EVALUATION_RUN_TYPE and entry.get("run_id") == normalized_run_id:
            return dict(entry)
    return None


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _format_optional_timestamp(value: object) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


__all__ = [
    "alpha_evaluation_registry_path",
    "build_alpha_evaluation_registry_entry",
    "filter_by_alpha_name",
    "filter_by_timeframe",
    "get_alpha_evaluation_run",
    "load_alpha_evaluation_registry",
    "register_alpha_evaluation_run",
]
