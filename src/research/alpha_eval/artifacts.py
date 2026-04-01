from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.alpha_eval.evaluator import AlphaEvaluationResult

DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT = Path("artifacts") / "alpha"
_IC_TIMESERIES_FILENAME = "ic_timeseries.csv"
_ALPHA_METRICS_FILENAME = "alpha_metrics.json"
_MANIFEST_FILENAME = "manifest.json"


def resolve_alpha_evaluation_artifact_dir(
    run_id: str,
    *,
    artifacts_root: str | Path = DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT,
) -> Path:
    """Return the default deterministic output directory for one alpha-evaluation run."""

    return Path(artifacts_root) / str(run_id)


def write_alpha_evaluation_artifacts(
    output_dir: str | Path,
    result: AlphaEvaluationResult,
    *,
    parent_manifest_dir: str | Path | None = None,
    run_id: str | None = None,
    alpha_name: str | None = None,
) -> dict[str, Any]:
    """Persist deterministic alpha-evaluation artifacts and return a manifest payload."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    ic_timeseries = _artifact_ic_timeseries_frame(result.ic_timeseries)
    metrics_payload = _alpha_metrics_payload(result, run_id=run_id, alpha_name=alpha_name)

    _write_csv(resolved_output_dir / _IC_TIMESERIES_FILENAME, ic_timeseries)
    _write_json(resolved_output_dir / _ALPHA_METRICS_FILENAME, metrics_payload)

    manifest = _build_manifest(
        output_dir=resolved_output_dir,
        result=result,
        metrics_payload=metrics_payload,
        run_id=run_id,
        alpha_name=alpha_name,
    )
    _write_json(resolved_output_dir / _MANIFEST_FILENAME, manifest)

    if parent_manifest_dir is not None:
        _augment_parent_manifest(Path(parent_manifest_dir), resolved_output_dir.name, manifest, metrics_payload)

    return manifest


def _artifact_ic_timeseries_frame(ic_timeseries: pd.DataFrame) -> pd.DataFrame:
    frame = ic_timeseries.copy(deep=True)
    frame.attrs = {}

    if "sample_size" in frame.columns and "n_obs" not in frame.columns:
        insert_at = frame.columns.get_loc("sample_size")
        frame.insert(insert_at, "n_obs", pd.to_numeric(frame["sample_size"], errors="coerce"))

    if "ts_utc" in frame.columns:
        timestamps = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
        frame["ts_utc"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        frame = frame.assign(_sort_ts_utc=timestamps).sort_values(
            by=["_sort_ts_utc"],
            kind="stable",
            na_position="last",
        )
        frame = frame.drop(columns="_sort_ts_utc")

    preferred_columns = [
        column
        for column in ("ts_utc", "ic", "rank_ic", "n_obs", "sample_size")
        if column in frame.columns
    ]
    remaining_columns = [column for column in frame.columns if column not in preferred_columns]
    return frame.loc[:, [*preferred_columns, *remaining_columns]].reset_index(drop=True)


def _alpha_metrics_payload(
    result: AlphaEvaluationResult,
    *,
    run_id: str | None,
    alpha_name: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **dict(result.summary),
        "forward_return_column": result.forward_return_column,
        "min_cross_section_size": result.min_cross_section_size,
        "prediction_column": result.prediction_column,
        "row_count": result.row_count,
        "symbol_count": result.symbol_count,
        "timestamp_count": result.timestamp_count,
    }
    metadata = dict(result.metadata)
    if alpha_name is not None:
        metadata["alpha_name"] = alpha_name
    if run_id is not None:
        metadata["run_id"] = run_id
    payload["metadata"] = metadata
    return _normalize_mapping(payload)


def _build_manifest(
    *,
    output_dir: Path,
    result: AlphaEvaluationResult,
    metrics_payload: dict[str, Any],
    run_id: str | None,
    alpha_name: str | None,
) -> dict[str, Any]:
    artifact_files = sorted(
        set([*(path.name for path in output_dir.iterdir() if path.is_file()), _MANIFEST_FILENAME])
    )
    metadata = metrics_payload.get("metadata", {})
    timeframe = metadata.get("timeframe") if isinstance(metadata, dict) else None
    return _normalize_mapping({
        "alpha_name": alpha_name,
        "artifact_files": artifact_files,
        "artifact_groups": {
            "alpha_evaluation": sorted(
                [_ALPHA_METRICS_FILENAME, _IC_TIMESERIES_FILENAME, _MANIFEST_FILENAME]
            )
        },
        "evaluation_mode": "alpha",
        "files_written": len(artifact_files),
        "metrics_path": _ALPHA_METRICS_FILENAME,
        "metric_summary": dict(result.summary),
        "row_count": result.row_count,
        "run_id": run_id if run_id is not None else output_dir.name,
        "timeframe": timeframe,
        "timeseries_columns": _artifact_ic_timeseries_frame(result.ic_timeseries).columns.tolist(),
        "timeseries_path": _IC_TIMESERIES_FILENAME,
    })


def _augment_parent_manifest(
    parent_dir: Path,
    artifact_dir_name: str,
    manifest: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> None:
    manifest_path = parent_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_files = payload.get("artifact_files", [])
    if not isinstance(artifact_files, list):
        artifact_files = []

    alpha_files = sorted(
        str(Path(artifact_dir_name, filename).as_posix())
        for filename in (_ALPHA_METRICS_FILENAME, _IC_TIMESERIES_FILENAME, _MANIFEST_FILENAME)
    )
    payload["artifact_files"] = sorted(set([*artifact_files, *alpha_files]))
    payload["alpha_evaluation"] = {
        "artifact_files": alpha_files,
        "artifact_path": artifact_dir_name,
        "enabled": True,
        "manifest_path": Path(artifact_dir_name, _MANIFEST_FILENAME).as_posix(),
        "metrics_path": Path(artifact_dir_name, _ALPHA_METRICS_FILENAME).as_posix(),
        "summary": {
            key: metrics_payload.get(key)
            for key in (
                "mean_ic",
                "std_ic",
                "ic_ir",
                "mean_rank_ic",
                "std_rank_ic",
                "rank_ic_ir",
                "n_periods",
            )
        },
        "timeframe": (
            metrics_payload.get("metadata", {}).get("timeframe")
            if isinstance(metrics_payload.get("metadata"), dict)
            else None
        ),
        "timeseries_path": Path(artifact_dir_name, _IC_TIMESERIES_FILENAME).as_posix(),
    }
    artifact_groups = payload.get("artifact_groups")
    if not isinstance(artifact_groups, dict):
        artifact_groups = {}
    artifact_groups["alpha_evaluation"] = alpha_files
    payload["artifact_groups"] = {
        key: sorted(value) if isinstance(value, list) else value
        for key, value in sorted(artifact_groups.items())
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_json_value(mapping[key]) for key in sorted(mapping)}


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        timestamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return timestamp.isoformat().replace("+00:00", "Z")
    if isinstance(value, pd.Series):
        return [_normalize_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return _normalize_mapping(value)
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT",
    "resolve_alpha_evaluation_artifact_dir",
    "write_alpha_evaluation_artifacts",
]
