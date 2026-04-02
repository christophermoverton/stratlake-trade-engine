from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.alpha.predictor import AlphaPredictionResult
from src.research.alpha.trainer import TrainedAlphaModel
from src.research.alpha_eval.evaluator import AlphaEvaluationResult
from src.research.promotion import (
    DEFAULT_PROMOTION_ARTIFACT_FILENAME,
    evaluate_promotion_gates,
    write_promotion_gate_artifact,
)

DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT = Path("artifacts") / "alpha"
_IC_TIMESERIES_FILENAME = "ic_timeseries.csv"
_ALPHA_METRICS_FILENAME = "alpha_metrics.json"
_PREDICTIONS_FILENAME = "predictions.parquet"
_TRAINING_SUMMARY_FILENAME = "training_summary.json"
_COEFFICIENTS_FILENAME = "coefficients.json"
_CROSS_SECTION_DIAGNOSTICS_FILENAME = "cross_section_diagnostics.json"
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
    trained_model: TrainedAlphaModel | None = None,
    prediction_result: AlphaPredictionResult | None = None,
    aligned_frame: pd.DataFrame | None = None,
    parent_manifest_dir: str | Path | None = None,
    run_id: str | None = None,
    alpha_name: str | None = None,
    promotion_gate_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist deterministic alpha-evaluation artifacts and return a manifest payload."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    ic_timeseries = _artifact_ic_timeseries_frame(result.ic_timeseries)
    metrics_payload = _alpha_metrics_payload(result, run_id=run_id, alpha_name=alpha_name)
    predictions_frame = _artifact_predictions_frame(
        prediction_result.predictions if prediction_result is not None else None
    )
    training_summary_payload = _training_summary_payload(
        trained_model=trained_model,
        prediction_result=prediction_result,
        run_id=run_id,
        alpha_name=alpha_name,
    )
    coefficients_payload = _coefficients_payload(
        trained_model=trained_model,
        alpha_name=alpha_name,
        run_id=run_id,
    )
    cross_section_diagnostics_payload = _cross_section_diagnostics_payload(
        result=result,
        aligned_frame=aligned_frame,
    )
    promotion_evaluation = evaluate_promotion_gates(
        run_type="alpha_evaluation",
        config=promotion_gate_config,
        sources={
            "metrics": dict(result.summary),
            "metadata": dict(result.metadata),
            "timeseries": ic_timeseries,
        },
    )

    _write_csv(resolved_output_dir / _IC_TIMESERIES_FILENAME, ic_timeseries)
    _write_json(resolved_output_dir / _ALPHA_METRICS_FILENAME, metrics_payload)
    _write_parquet(resolved_output_dir / _PREDICTIONS_FILENAME, predictions_frame)
    _write_json(resolved_output_dir / _TRAINING_SUMMARY_FILENAME, training_summary_payload)
    _write_json(resolved_output_dir / _COEFFICIENTS_FILENAME, coefficients_payload)
    _write_json(
        resolved_output_dir / _CROSS_SECTION_DIAGNOSTICS_FILENAME,
        cross_section_diagnostics_payload,
    )
    write_promotion_gate_artifact(resolved_output_dir, promotion_evaluation)

    manifest = _build_manifest(
        output_dir=resolved_output_dir,
        result=result,
        metrics_payload=metrics_payload,
        training_summary_payload=training_summary_payload,
        coefficients_payload=coefficients_payload,
        cross_section_diagnostics_payload=cross_section_diagnostics_payload,
        run_id=run_id,
        alpha_name=alpha_name,
        promotion_evaluation=promotion_evaluation,
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


def _artifact_predictions_frame(predictions: pd.DataFrame | None) -> pd.DataFrame:
    if predictions is None:
        return pd.DataFrame(columns=["symbol", "ts_utc", "timeframe", "prediction_score"])

    frame = predictions.copy(deep=True)
    frame.attrs = {}
    if "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    sort_columns = [column for column in ("symbol", "ts_utc", "timeframe") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return frame


def _training_summary_payload(
    *,
    trained_model: TrainedAlphaModel | None,
    prediction_result: AlphaPredictionResult | None,
    run_id: str | None,
    alpha_name: str | None,
) -> dict[str, Any]:
    model = None if trained_model is None else trained_model.model
    model_training_metadata = getattr(model, "training_metadata", None)
    return _normalize_mapping(
        {
            "alpha_name": alpha_name,
            "feature_columns": [] if trained_model is None else list(trained_model.feature_columns),
            "model_class": None if model is None else type(model).__name__,
            "model_name": None if trained_model is None else trained_model.model_name,
            "prediction": {
                "columns": (
                    []
                    if prediction_result is None
                    else list(prediction_result.metadata.get("prediction_output_columns", []))
                ),
                "predict_end": None if prediction_result is None else prediction_result.predict_end,
                "predict_row_count": None if prediction_result is None else prediction_result.row_count,
                "predict_start": None if prediction_result is None else prediction_result.predict_start,
                "predict_symbol_count": None if prediction_result is None else prediction_result.symbol_count,
                "window_semantics": (
                    None
                    if prediction_result is None
                    else prediction_result.metadata.get("window_semantics")
                ),
            },
            "run_id": run_id,
            "target_column": None if trained_model is None else trained_model.target_column,
            "training": {
                "fit_columns": [] if trained_model is None else list(trained_model.metadata.get("fit_columns", [])),
                "model_training_metadata": (
                    {}
                    if not isinstance(model_training_metadata, dict)
                    else dict(model_training_metadata)
                ),
                "train_end": None if trained_model is None else trained_model.train_end,
                "train_row_count": None if trained_model is None else trained_model.row_count,
                "train_start": None if trained_model is None else trained_model.train_start,
                "train_symbol_count": None if trained_model is None else trained_model.symbol_count,
                "window_semantics": (
                    None if trained_model is None else trained_model.metadata.get("window_semantics")
                ),
            },
        }
    )


def _coefficients_payload(
    *,
    trained_model: TrainedAlphaModel | None,
    alpha_name: str | None,
    run_id: str | None,
) -> dict[str, Any]:
    model = None if trained_model is None else trained_model.model
    intercept = getattr(model, "intercept", None)
    coefficients = _extract_named_numeric_mapping(
        model,
        candidate_attributes=(
            "coefficient_by_feature",
            "feature_weight_by_name",
            "feature_ic_by_name",
            "feature_means",
        ),
    )
    representation = "unavailable" if not coefficients else _coefficient_representation(model)
    return _normalize_mapping(
        {
            "alpha_name": alpha_name,
            "intercept": intercept if isinstance(intercept, int | float) else None,
            "model_name": None if trained_model is None else trained_model.model_name,
            "representation": representation,
            "run_id": run_id,
            "values": coefficients,
        }
    )


def _cross_section_diagnostics_payload(
    *,
    result: AlphaEvaluationResult,
    aligned_frame: pd.DataFrame | None,
) -> dict[str, Any]:
    ic_timeseries = _artifact_ic_timeseries_frame(result.ic_timeseries)
    valid_ic = ic_timeseries.loc[ic_timeseries["ic"].notna(), ["ts_utc", "ic"]]
    valid_rank_ic = ic_timeseries.loc[ic_timeseries["rank_ic"].notna(), ["ts_utc", "rank_ic"]]
    sample_sizes = pd.to_numeric(ic_timeseries.get("sample_size"), errors="coerce")
    return _normalize_mapping(
        {
            "cross_section_size": {
                "max": None if sample_sizes.isna().all() else int(sample_sizes.max()),
                "mean": None if sample_sizes.isna().all() else float(sample_sizes.mean()),
                "min": None if sample_sizes.isna().all() else int(sample_sizes.min()),
            },
            "forward_return_column": result.forward_return_column,
            "ic_extrema": {
                "best_ic": _timestamp_metric_snapshot(valid_ic, value_column="ic", ascending=False),
                "best_rank_ic": _timestamp_metric_snapshot(
                    valid_rank_ic,
                    value_column="rank_ic",
                    ascending=False,
                ),
                "worst_ic": _timestamp_metric_snapshot(valid_ic, value_column="ic", ascending=True),
                "worst_rank_ic": _timestamp_metric_snapshot(
                    valid_rank_ic,
                    value_column="rank_ic",
                    ascending=True,
                ),
            },
            "min_cross_section_size": result.min_cross_section_size,
            "prediction_column": result.prediction_column,
            "prediction_score": (
                {}
                if aligned_frame is None or result.prediction_column not in aligned_frame.columns
                else _numeric_column_summary(aligned_frame[result.prediction_column])
            ),
            "row_count": result.row_count,
            "symbol_count": result.symbol_count,
            "timestamp_count": result.timestamp_count,
            "valid_periods": {
                "ic": int(valid_ic.shape[0]),
                "rank_ic": int(valid_rank_ic.shape[0]),
            },
        }
    )


def _build_manifest(
    *,
    output_dir: Path,
    result: AlphaEvaluationResult,
    metrics_payload: dict[str, Any],
    training_summary_payload: dict[str, Any],
    coefficients_payload: dict[str, Any],
    cross_section_diagnostics_payload: dict[str, Any],
    run_id: str | None,
    alpha_name: str | None,
    promotion_evaluation: Any,
) -> dict[str, Any]:
    artifact_files = sorted(
        set([*(path.name for path in output_dir.iterdir() if path.is_file()), _MANIFEST_FILENAME])
    )
    metadata = metrics_payload.get("metadata", {})
    timeframe = metadata.get("timeframe") if isinstance(metadata, dict) else None
    artifact_group = sorted(
        [
            _ALPHA_METRICS_FILENAME,
            _COEFFICIENTS_FILENAME,
            _CROSS_SECTION_DIAGNOSTICS_FILENAME,
            _IC_TIMESERIES_FILENAME,
            _MANIFEST_FILENAME,
            _PREDICTIONS_FILENAME,
            _TRAINING_SUMMARY_FILENAME,
            *(
                [DEFAULT_PROMOTION_ARTIFACT_FILENAME]
                if promotion_evaluation is not None
                else []
            ),
        ]
    )
    return _normalize_mapping({
        "alpha_name": alpha_name,
        "artifact_files": artifact_files,
        "artifact_paths": {
            "coefficients": _COEFFICIENTS_FILENAME,
            "cross_section_diagnostics": _CROSS_SECTION_DIAGNOSTICS_FILENAME,
            "manifest": _MANIFEST_FILENAME,
            "metrics": _ALPHA_METRICS_FILENAME,
            "predictions": _PREDICTIONS_FILENAME,
            "timeseries": _IC_TIMESERIES_FILENAME,
            "training_summary": _TRAINING_SUMMARY_FILENAME,
        },
        "artifact_groups": {
            "alpha_evaluation": artifact_group
        },
        "coefficient_representation": coefficients_payload.get("representation"),
        "cross_section_diagnostics_path": _CROSS_SECTION_DIAGNOSTICS_FILENAME,
        "evaluation_mode": "alpha",
        "files_written": len(artifact_files),
        "manifest_path": _MANIFEST_FILENAME,
        "metrics_path": _ALPHA_METRICS_FILENAME,
        "metric_summary": dict(result.summary),
        "prediction_row_count": training_summary_payload.get("prediction", {}).get("predict_row_count"),
        "predictions_path": _PREDICTIONS_FILENAME,
        "promotion_gate_summary": None if promotion_evaluation is None else promotion_evaluation.summary(),
        "row_count": result.row_count,
        "run_id": run_id if run_id is not None else output_dir.name,
        "timeframe": timeframe,
        "training_summary_path": _TRAINING_SUMMARY_FILENAME,
        "training_window": training_summary_payload.get("training", {}),
        "valid_periods": cross_section_diagnostics_payload.get("valid_periods", {}),
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
        for filename in (
            _ALPHA_METRICS_FILENAME,
            _COEFFICIENTS_FILENAME,
            _CROSS_SECTION_DIAGNOSTICS_FILENAME,
            _IC_TIMESERIES_FILENAME,
            _MANIFEST_FILENAME,
            _PREDICTIONS_FILENAME,
            _TRAINING_SUMMARY_FILENAME,
        )
    )
    payload["artifact_files"] = sorted(set([*artifact_files, *alpha_files]))
    payload["alpha_evaluation"] = {
        "artifact_files": alpha_files,
        "artifact_path": artifact_dir_name,
        "artifact_paths": {
            key: Path(artifact_dir_name, str(value)).as_posix()
            for key, value in dict(manifest.get("artifact_paths", {})).items()
        },
        "cross_section_diagnostics_path": Path(
            artifact_dir_name,
            _CROSS_SECTION_DIAGNOSTICS_FILENAME,
        ).as_posix(),
        "enabled": True,
        "manifest_path": Path(artifact_dir_name, _MANIFEST_FILENAME).as_posix(),
        "metrics_path": Path(artifact_dir_name, _ALPHA_METRICS_FILENAME).as_posix(),
        "predictions_path": Path(artifact_dir_name, _PREDICTIONS_FILENAME).as_posix(),
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
        "training_summary_path": Path(artifact_dir_name, _TRAINING_SUMMARY_FILENAME).as_posix(),
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


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


def _extract_named_numeric_mapping(
    model: Any,
    *,
    candidate_attributes: tuple[str, ...],
) -> dict[str, float]:
    if model is None:
        return {}
    for attribute_name in candidate_attributes:
        raw_value = getattr(model, attribute_name, None)
        if not isinstance(raw_value, dict):
            continue
        numeric_values: dict[str, float] = {}
        for key, value in sorted(raw_value.items()):
            if not isinstance(key, str):
                continue
            try:
                normalized = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(normalized) or math.isinf(normalized):
                continue
            numeric_values[key] = normalized
        if numeric_values:
            return numeric_values
    return {}


def _coefficient_representation(model: Any) -> str:
    if model is None:
        return "unavailable"
    for attribute_name in (
        "coefficient_by_feature",
        "feature_weight_by_name",
        "feature_ic_by_name",
        "feature_means",
    ):
        if isinstance(getattr(model, attribute_name, None), dict):
            return attribute_name
    return "unavailable"


def _timestamp_metric_snapshot(
    frame: pd.DataFrame,
    *,
    value_column: str,
    ascending: bool,
) -> dict[str, Any] | None:
    if frame.empty:
        return None
    ordered = frame.sort_values([value_column, "ts_utc"], ascending=[ascending, True], kind="stable")
    row = ordered.iloc[0]
    return {
        "ts_utc": row["ts_utc"],
        value_column: row[value_column],
    }


def _numeric_column_summary(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().all():
        return {"max": None, "mean": None, "min": None}
    return {
        "max": float(values.max()),
        "mean": float(values.mean()),
        "min": float(values.min()),
    }


__all__ = [
    "DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT",
    "resolve_alpha_evaluation_artifact_dir",
    "write_alpha_evaluation_artifacts",
]
