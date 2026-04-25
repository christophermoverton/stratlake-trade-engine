from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.regime_ml.artifacts import write_regime_ml_artifacts
from src.research.regime_ml.base import RegimeMLError
from src.research.regime_ml.calibration import PlattCalibrator, multiclass_brier_score
from src.research.regime_ml.clustering import compute_cluster_diagnostics
from src.research.regime_ml.models import LogisticRegressionRegimeModel
from src.research.regimes.taxonomy import TAXONOMY_VERSION
from src.research.registry import canonicalize_value

DEFAULT_CONFIDENCE_THRESHOLDS = {"high": 0.70, "medium": 0.40}


@dataclass(frozen=True)
class RegimeMLConfig:
    random_seed: int = 42
    label_column: str = "regime_label"
    time_column: str = "ts_utc"
    symbol_column: str = "symbol"
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    calibration_method: str = "platt"
    confidence_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CONFIDENCE_THRESHOLDS))
    ambiguous_margin_threshold: float = 0.10
    class_imbalance_warning_threshold: float = 0.10
    weak_alignment_purity_threshold: float = 0.60
    n_clusters: int | None = None
    model_version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeMLResult:
    regime_confidence: pd.DataFrame
    diagnostics: dict[str, Any]
    manifest: dict[str, Any]
    cluster_map: pd.DataFrame
    cluster_diagnostics: dict[str, Any]
    label_mapping: dict[str, Any]
    class_probabilities: pd.DataFrame
    output_manifest: dict[str, Any] | None = None


def run_regime_ml_pipeline(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...],
    config: RegimeMLConfig | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> RegimeMLResult:
    resolved_config = resolve_regime_ml_config(config)
    resolved_feature_columns = _resolve_feature_columns(feature_columns)
    frame, preprocessing_metadata = _normalize_dataset(
        dataset,
        feature_columns=resolved_feature_columns,
        config=resolved_config,
    )
    diagnostics = _base_diagnostics(
        frame=frame,
        feature_columns=resolved_feature_columns,
        preprocessing_metadata=preprocessing_metadata,
        config=resolved_config,
    )

    split_index = _build_temporal_splits(frame[resolved_config.time_column], config=resolved_config)
    diagnostics["leakage_risks"] = {
        "train_validation_overlap": bool(set(split_index["train_timestamps"]) & set(split_index["validation_timestamps"])),
        "train_test_overlap": bool(set(split_index["train_timestamps"]) & set(split_index["test_timestamps"])),
        "validation_test_overlap": bool(set(split_index["validation_timestamps"]) & set(split_index["test_timestamps"])),
    }

    train_mask = frame[resolved_config.time_column].isin(split_index["train_timestamps"])
    validation_mask = frame[resolved_config.time_column].isin(split_index["validation_timestamps"])
    test_mask = frame[resolved_config.time_column].isin(split_index["test_timestamps"])

    train_frame = frame.loc[train_mask].copy(deep=True)
    validation_frame = frame.loc[validation_mask].copy(deep=True)
    test_frame = frame.loc[test_mask].copy(deep=True)
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise RegimeMLError("Regime ML requires non-empty train, validation, and test windows.")

    training_labels = train_frame[resolved_config.label_column].astype("string")
    if int(training_labels.nunique()) < 2:
        raise RegimeMLError("Regime ML training requires at least two distinct labels in the train window.")

    train_distribution = _distribution(training_labels)
    imbalance_warnings = [
        label for label, share in train_distribution.items() if float(share) < resolved_config.class_imbalance_warning_threshold
    ]
    diagnostics["class_imbalance_warnings"] = imbalance_warnings
    diagnostics["insufficient_class_coverage"] = bool(int(training_labels.nunique()) < 2)
    diagnostics["unseen_labels"] = sorted(
        set(frame[resolved_config.label_column].astype("string")) - set(training_labels.astype("string"))
    )

    model = LogisticRegressionRegimeModel(random_seed=resolved_config.random_seed)
    model.fit(
        train_frame.loc[:, resolved_feature_columns],
        train_frame[resolved_config.label_column],
        metadata={
            "label_column": resolved_config.label_column,
            "taxonomy_version": TAXONOMY_VERSION,
            "feature_columns": list(resolved_feature_columns),
        },
    )

    validation_labels = validation_frame[resolved_config.label_column].astype("string")
    unseen_validation_labels = sorted(set(validation_labels.tolist()) - set(model.classes_))
    if unseen_validation_labels:
        raise RegimeMLError(
            "Validation labels contain classes not seen during training and cannot be used for Platt calibration. "
            f"Unseen labels: {unseen_validation_labels}."
        )

    validation_raw = model.predict_proba(validation_frame.loc[:, resolved_feature_columns])
    calibrator = PlattCalibrator.fit(validation_raw, validation_labels)
    validation_calibrated = calibrator.transform(validation_raw)
    pre_brier = multiclass_brier_score(validation_raw, validation_labels)
    post_brier = multiclass_brier_score(validation_calibrated, validation_labels)

    full_raw = model.predict_proba(frame.loc[:, resolved_feature_columns])
    full_probabilities = calibrator.transform(full_raw)
    predicted_labels = pd.Series(
        full_probabilities.idxmax(axis=1),
        index=frame.index,
        dtype="string",
        name="predicted_label",
    )

    confidence_frame = _build_confidence_frame(
        frame=frame,
        probabilities=full_probabilities,
        predicted_labels=predicted_labels,
        model_classes=model.classes_,
        config=resolved_config,
    )
    diagnostics["fallback_counts"] = {
        "fallback_flagged": int(confidence_frame["fallback_flag"].astype("bool").sum()),
        "low_confidence": int(confidence_frame["fallback_reason"].eq("low_confidence").sum()),
        "ambiguous": int(confidence_frame["fallback_reason"].eq("ambiguous").sum()),
        "unsupported": int(confidence_frame["fallback_reason"].eq("unsupported").sum()),
    }
    diagnostics["calibration"] = {
        "calibration_method": resolved_config.calibration_method,
        "calibration_window": {
            "start": validation_frame[resolved_config.time_column].min().isoformat().replace("+00:00", "Z"),
            "end": validation_frame[resolved_config.time_column].max().isoformat().replace("+00:00", "Z"),
            "row_count": int(len(validation_frame)),
        },
        "pre_calibration_brier": pre_brier,
        "post_calibration_brier": post_brier,
        "brier_improved": bool(post_brier <= pre_brier),
    }

    cluster_result = compute_cluster_diagnostics(
        frame.loc[:, resolved_feature_columns],
        frame[resolved_config.label_column],
        random_seed=resolved_config.random_seed,
        n_clusters=resolved_config.n_clusters or int(frame[resolved_config.label_column].astype("string").nunique()),
        weak_alignment_purity_threshold=resolved_config.weak_alignment_purity_threshold,
    )

    manifest = canonicalize_value(
        {
            "model_family": "logistic_regression",
            "model_version": resolved_config.model_version,
            "random_seed": resolved_config.random_seed,
            "train_window": {
                "start": train_frame[resolved_config.time_column].min().isoformat().replace("+00:00", "Z"),
                "end": train_frame[resolved_config.time_column].max().isoformat().replace("+00:00", "Z"),
            },
            "test_window": {
                "start": test_frame[resolved_config.time_column].min().isoformat().replace("+00:00", "Z"),
                "end": test_frame[resolved_config.time_column].max().isoformat().replace("+00:00", "Z"),
            },
            "feature_columns": list(resolved_feature_columns),
            "label_column": resolved_config.label_column,
            "n_classes": int(len(model.classes_)),
            "class_distribution": {label: int((training_labels == label).sum()) for label in sorted(train_distribution)},
            "taxonomy_version": TAXONOMY_VERSION,
            "window_semantics": "contiguous chronological split",
            "feature_imputation": preprocessing_metadata["feature_imputation"],
            "metadata": dict(sorted(resolved_config.metadata.items())),
        }
    )

    label_mapping = canonicalize_value(
        {
            "taxonomy_version": TAXONOMY_VERSION,
            "label_to_index": {label: index for index, label in enumerate(model.classes_)},
            "index_to_label": {str(index): label for index, label in enumerate(model.classes_)},
            "training_labels": model.classes_,
            "unseen_evaluation_labels": diagnostics["unseen_labels"],
        }
    )

    output_manifest = None
    if output_dir is not None:
        output_manifest = write_regime_ml_artifacts(
            output_dir,
            regime_confidence=confidence_frame,
            diagnostics=diagnostics,
            manifest=manifest,
            cluster_map=cluster_result.cluster_map,
            cluster_diagnostics=cluster_result.diagnostics,
            label_mapping=label_mapping,
        )

    return RegimeMLResult(
        regime_confidence=confidence_frame,
        diagnostics=canonicalize_value(diagnostics),
        manifest=manifest,
        cluster_map=cluster_result.cluster_map,
        cluster_diagnostics=cluster_result.diagnostics,
        label_mapping=label_mapping,
        class_probabilities=full_probabilities,
        output_manifest=output_manifest,
    )


def resolve_regime_ml_config(payload: RegimeMLConfig | dict[str, Any] | None) -> RegimeMLConfig:
    if payload is None:
        resolved = RegimeMLConfig()
        _validate_regime_ml_config(resolved)
        return resolved
    if isinstance(payload, RegimeMLConfig):
        _validate_regime_ml_config(payload)
        return payload
    if not isinstance(payload, dict):
        raise TypeError("Regime ML config must be a RegimeMLConfig or dictionary.")
    allowed = set(RegimeMLConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RegimeMLError(f"Regime ML config contains unsupported fields: {unknown}.")
    resolved = RegimeMLConfig(**payload)
    _validate_regime_ml_config(resolved)
    return resolved


def _validate_regime_ml_config(config: RegimeMLConfig) -> None:
    if not isinstance(config.random_seed, int):
        raise RegimeMLError("Regime ML config field 'random_seed' must be an integer.")
    for field_name in ("label_column", "time_column", "symbol_column", "model_version"):
        value = getattr(config, field_name)
        if not isinstance(value, str) or not value.strip():
            raise RegimeMLError(f"Regime ML config field '{field_name}' must be a non-empty string.")
    for field_name in ("validation_fraction", "test_fraction"):
        value = getattr(config, field_name)
        if not isinstance(value, int | float) or not (0.0 < float(value) < 1.0):
            raise RegimeMLError(f"Regime ML config field '{field_name}' must be between 0 and 1.")
    if float(config.validation_fraction) + float(config.test_fraction) >= 1.0:
        raise RegimeMLError("Regime ML config requires validation_fraction + test_fraction < 1.")
    for field_name in (
        "ambiguous_margin_threshold",
        "class_imbalance_warning_threshold",
        "weak_alignment_purity_threshold",
    ):
        value = getattr(config, field_name)
        if not isinstance(value, int | float) or not (0.0 <= float(value) <= 1.0):
            raise RegimeMLError(f"Regime ML config field '{field_name}' must be between 0 and 1.")
    if config.calibration_method != "platt":
        raise RegimeMLError("Regime ML currently supports only platt calibration.")
    thresholds = config.confidence_thresholds
    if not isinstance(thresholds, dict):
        raise RegimeMLError("Regime ML config field 'confidence_thresholds' must be a mapping.")
    missing = sorted({"high", "medium"} - set(thresholds))
    if missing:
        raise RegimeMLError(f"Regime ML config field 'confidence_thresholds' is missing required keys: {missing}.")
    medium = thresholds["medium"]
    high = thresholds["high"]
    for name, value in (("medium", medium), ("high", high)):
        if not isinstance(value, int | float) or not (0.0 <= float(value) <= 1.0):
            raise RegimeMLError(
                f"Regime ML confidence threshold '{name}' must be between 0 and 1."
            )
    if float(medium) > float(high):
        raise RegimeMLError("Regime ML confidence thresholds require medium <= high.")
    if config.n_clusters is not None and (not isinstance(config.n_clusters, int) or config.n_clusters <= 0):
        raise RegimeMLError("Regime ML config field 'n_clusters' must be a positive integer when provided.")


def _resolve_feature_columns(feature_columns: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(feature_columns, (list, tuple)):
        raise TypeError("Regime ML feature_columns must be a list or tuple.")
    if not feature_columns:
        raise RegimeMLError("Regime ML feature_columns must contain at least one feature.")
    normalized: list[str] = []
    seen: set[str] = set()
    for column in feature_columns:
        if not isinstance(column, str) or not column.strip():
            raise RegimeMLError("Regime ML feature_columns entries must be non-empty strings.")
        stripped = column.strip()
        if stripped in seen:
            raise RegimeMLError(f"Regime ML feature_columns must not contain duplicates. Found duplicate: {stripped!r}.")
        normalized.append(stripped)
        seen.add(stripped)
    return sorted(normalized)


def _normalize_dataset(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: RegimeMLConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Regime ML dataset must be a pandas DataFrame.")
    if dataset.empty:
        raise RegimeMLError("Regime ML dataset must not be empty.")
    required = {config.label_column, config.symbol_column, config.time_column, *feature_columns}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise RegimeMLError(f"Regime ML dataset is missing required columns: {missing}.")

    frame = dataset.copy(deep=True)
    frame.attrs = {}
    frame[config.time_column] = pd.to_datetime(frame[config.time_column], utc=True, errors="coerce")
    if frame[config.time_column].isna().any():
        raise RegimeMLError(f"Regime ML dataset contains invalid timestamps in {config.time_column!r}.")
    frame[config.symbol_column] = frame[config.symbol_column].astype("string")
    frame[config.label_column] = frame[config.label_column].astype("string")
    numeric_features = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    missing_count_by_feature = {
        column: int(numeric_features[column].isna().sum())
        for column in feature_columns
    }
    frame["_has_missing_features"] = numeric_features.isna().any(axis=1)
    medians = numeric_features.median(numeric_only=True)
    imputation_values: dict[str, float] = {}
    for column in feature_columns:
        median = medians.get(column)
        if pd.isna(median):
            raise RegimeMLError(f"Feature column {column!r} cannot be entirely missing.")
        imputation_values[column] = float(median)
        frame[column] = numeric_features[column].fillna(float(median))
    frame = frame.sort_values([config.time_column, config.symbol_column], kind="stable").reset_index(drop=True)
    preprocessing_metadata = canonicalize_value(
        {
            "feature_imputation": {
                "method": "median",
                "values": imputation_values,
                "missing_count_by_feature": missing_count_by_feature,
                "rows_with_any_missing_feature": int(frame["_has_missing_features"].sum()),
            }
        }
    )
    return frame, preprocessing_metadata


def _base_diagnostics(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    preprocessing_metadata: dict[str, Any],
    config: RegimeMLConfig,
) -> dict[str, Any]:
    missing_feature_rows = frame.loc[frame["_has_missing_features"], [config.symbol_column, config.time_column]]
    return canonicalize_value(
        {
            "taxonomy_version": TAXONOMY_VERSION,
            "row_count": int(len(frame)),
            "feature_columns": feature_columns,
            **preprocessing_metadata,
            "missing_features": {
                "row_count": int(len(missing_feature_rows)),
                "rows": [
                    {
                        "symbol": str(row[config.symbol_column]),
                        "ts_utc": row[config.time_column].isoformat().replace("+00:00", "Z"),
                    }
                    for _, row in missing_feature_rows.head(10).iterrows()
                ],
            },
        }
    )


def _build_temporal_splits(timestamps: pd.Series, *, config: RegimeMLConfig) -> dict[str, list[pd.Timestamp]]:
    unique_timestamps = sorted(pd.Index(pd.to_datetime(timestamps, utc=True, errors="raise")).unique().tolist())
    if len(unique_timestamps) < 5:
        raise RegimeMLError("Regime ML requires at least five distinct timestamps for train/validation/test splits.")
    validation_count = max(1, int(math.floor(len(unique_timestamps) * config.validation_fraction)))
    test_count = max(1, int(math.floor(len(unique_timestamps) * config.test_fraction)))
    train_count = len(unique_timestamps) - validation_count - test_count
    if train_count < 2:
        raise RegimeMLError("Regime ML split configuration leaves too few timestamps in the train window.")
    return {
        "train_timestamps": unique_timestamps[:train_count],
        "validation_timestamps": unique_timestamps[train_count:train_count + validation_count],
        "test_timestamps": unique_timestamps[train_count + validation_count:],
    }


def _distribution(labels: pd.Series) -> dict[str, float]:
    counts = labels.astype("string").value_counts(normalize=True, dropna=False)
    return {str(label): float(counts[label]) for label in sorted(counts.index.astype("string").tolist())}


def _build_confidence_frame(
    *,
    frame: pd.DataFrame,
    probabilities: pd.DataFrame,
    predicted_labels: pd.Series,
    model_classes: list[str],
    config: RegimeMLConfig,
) -> pd.DataFrame:
    ordered = probabilities.loc[:, list(model_classes)].copy(deep=True)
    top1 = ordered.max(axis=1)
    top2 = ordered.apply(lambda row: _top_two_gap(row.to_numpy(dtype="float64", copy=True))[1], axis=1)
    entropy = ordered.apply(lambda row: _entropy(row.to_numpy(dtype="float64", copy=True)), axis=1)

    confidence_scores = []
    buckets = []
    reasons = []
    fallback_flags = []
    for index, assigned_label in frame[config.label_column].astype("string").items():
        assigned_probability = float(ordered.loc[index, assigned_label]) if assigned_label in ordered.columns else 0.0
        confidence_scores.append(assigned_probability)
        bucket = _bucket_confidence(assigned_probability, thresholds=config.confidence_thresholds)
        buckets.append(bucket)
        reason = None
        if bool(frame.loc[index, "_has_missing_features"]) or assigned_label not in ordered.columns:
            reason = "unsupported"
        elif assigned_probability < float(config.confidence_thresholds["medium"]):
            reason = "low_confidence"
        elif float(top1.loc[index] - top2.loc[index]) < config.ambiguous_margin_threshold:
            reason = "ambiguous"
        reasons.append(reason)
        fallback_flags.append(bool(reason is not None))

    return pd.DataFrame(
        {
            "symbol": frame[config.symbol_column].astype("string"),
            "ts_utc": frame[config.time_column].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "regime_label": frame[config.label_column].astype("string"),
            "predicted_label": predicted_labels.astype("string"),
            "confidence_score": pd.Series(confidence_scores, index=frame.index, dtype="float64"),
            "confidence_bucket": pd.Series(buckets, index=frame.index, dtype="string"),
            "entropy": entropy.astype("float64"),
            "margin": (top1 - top2).astype("float64"),
            "fallback_flag": pd.Series(fallback_flags, index=frame.index, dtype="bool"),
            "fallback_reason": pd.Series(reasons, index=frame.index, dtype="string"),
        }
    )


def _bucket_confidence(value: float, *, thresholds: dict[str, float]) -> str:
    if value >= float(thresholds["high"]):
        return "high"
    if value >= float(thresholds["medium"]):
        return "medium"
    return "low"


def _top_two_gap(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(values)[::-1]
    if len(ordered) == 1:
        return float(ordered[0]), 0.0
    return float(ordered[0]), float(ordered[1])


def _entropy(values: np.ndarray) -> float:
    clipped = np.clip(values.astype("float64"), 1.0e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))
