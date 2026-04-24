from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency environments
    GaussianMixture = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

from src.research.regimes.taxonomy import REGIME_AUDIT_COLUMNS, TAXONOMY_VERSION
from src.research.registry import canonicalize_value

DEFAULT_GMM_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "high": 0.75,
    "medium": 0.50,
}

GMM_LABEL_COLUMNS: tuple[str, ...] = (
    "ts_utc",
    "gmm_cluster_label",
    "gmm_previous_cluster_label",
    "gmm_shift_flag",
    "gmm_shift_probability_delta",
    "gmm_posterior_probability",
    "gmm_confidence_score",
    "gmm_posterior_entropy",
    "gmm_entropy_confidence",
    "gmm_is_low_confidence",
    "confidence_score",
    "confidence_bucket",
    "fallback_flag",
    "fallback_reason",
    "predicted_label",
)

GMM_SHIFT_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_order",
    "ts_utc",
    "previous_cluster_label",
    "current_cluster_label",
    "shift_probability_delta",
    "confidence_score",
    "is_low_confidence",
    "taxonomy_version",
)


class RegimeGMMClassifierError(ValueError):
    """Raised when deterministic GMM regime classification cannot be completed."""


@dataclass(frozen=True)
class RegimeGMMClassifierConfig:
    """Deterministic Gaussian-mixture config for regime-shift detection."""

    n_components: int = 3
    random_state: int = 42
    covariance_type: str = "full"
    max_iter: int = 200
    n_init: int = 1
    feature_columns: tuple[str, ...] = REGIME_AUDIT_COLUMNS
    time_column: str = "ts_utc"
    low_confidence_threshold: float = 0.55
    confidence_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GMM_CONFIDENCE_THRESHOLDS)
    )
    ambiguous_margin_threshold: float = 0.10
    min_shift_probability: float = 0.55
    shift_probability_delta_threshold: float = 0.15
    shift_requires_not_low_confidence: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "ambiguous_margin_threshold": self.ambiguous_margin_threshold,
                "confidence_thresholds": dict(self.confidence_thresholds),
                "covariance_type": self.covariance_type,
                "feature_columns": list(self.feature_columns),
                "low_confidence_threshold": self.low_confidence_threshold,
                "max_iter": self.max_iter,
                "metadata": dict(sorted(self.metadata.items())),
                "min_shift_probability": self.min_shift_probability,
                "n_components": self.n_components,
                "n_init": self.n_init,
                "random_state": self.random_state,
                "shift_probability_delta_threshold": self.shift_probability_delta_threshold,
                "shift_requires_not_low_confidence": self.shift_requires_not_low_confidence,
                "time_column": self.time_column,
            }
        )


@dataclass(frozen=True)
class RegimeGMMClassifierResult:
    """Structured output for deterministic GMM regime clustering."""

    labels: pd.DataFrame
    posterior_probabilities: pd.DataFrame
    shift_events: pd.DataFrame
    summary: dict[str, Any]
    config: RegimeGMMClassifierConfig
    taxonomy_version: str = TAXONOMY_VERSION


def resolve_regime_gmm_classifier_config(
    payload: RegimeGMMClassifierConfig | dict[str, Any] | None,
) -> RegimeGMMClassifierConfig:
    if payload is None:
        resolved = RegimeGMMClassifierConfig()
        _validate_config(resolved)
        return resolved
    if isinstance(payload, RegimeGMMClassifierConfig):
        _validate_config(payload)
        return payload
    if not isinstance(payload, dict):
        raise TypeError("Regime GMM config must be a RegimeGMMClassifierConfig or dictionary.")
    allowed = set(RegimeGMMClassifierConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RegimeGMMClassifierError(f"Regime GMM config contains unsupported fields: {unknown}.")
    resolved = RegimeGMMClassifierConfig(**payload)
    _validate_config(resolved)
    return resolved


def classify_regime_shifts_with_gmm(
    features: pd.DataFrame,
    *,
    config: RegimeGMMClassifierConfig | dict[str, Any] | None = None,
) -> RegimeGMMClassifierResult:
    """Fit a deterministic Gaussian Mixture Model and detect cluster shifts."""

    resolved_config = resolve_regime_gmm_classifier_config(config)
    _require_sklearn()
    normalized = _normalize_features(features, config=resolved_config)

    estimator = GaussianMixture(
        covariance_type=resolved_config.covariance_type,
        max_iter=resolved_config.max_iter,
        n_components=resolved_config.n_components,
        n_init=resolved_config.n_init,
        random_state=resolved_config.random_state,
    )
    estimator.fit(normalized.loc[:, list(resolved_config.feature_columns)])
    probabilities = estimator.predict_proba(normalized.loc[:, list(resolved_config.feature_columns)])

    component_order = _stable_component_order(estimator.means_)
    ordered_probabilities = probabilities[:, component_order]
    posterior_columns = [f"posterior_cluster_{index}" for index in range(len(component_order))]
    posterior_frame = pd.DataFrame(ordered_probabilities, columns=posterior_columns, index=normalized.index)

    max_probability = posterior_frame.max(axis=1).astype("float64")
    cluster_codes = posterior_frame.to_numpy(dtype="float64", copy=False).argmax(axis=1)
    cluster_series = pd.Series(cluster_codes, index=normalized.index, dtype="int64")
    previous_cluster = cluster_series.shift(1)
    previous_probabilities = posterior_frame.shift(1)

    shift_probability_delta = pd.Series(np.nan, index=normalized.index, dtype="float64")
    for row_index, cluster_code in cluster_series.items():
        previous_probability = previous_probabilities.loc[row_index, f"posterior_cluster_{int(cluster_code)}"]
        if pd.notna(previous_probability):
            shift_probability_delta.loc[row_index] = abs(float(max_probability.loc[row_index]) - float(previous_probability))

    entropy = posterior_frame.apply(
        lambda row: _entropy(row.to_numpy(dtype="float64", copy=True)),
        axis=1,
    ).astype("float64")
    entropy_confidence = entropy.map(
        lambda value: _entropy_confidence(value, class_count=len(component_order))
    ).astype("float64")

    low_confidence_mask = max_probability.lt(resolved_config.low_confidence_threshold).astype("bool")
    cluster_changed = previous_cluster.notna() & cluster_series.ne(previous_cluster.astype("Int64"))
    shift_flag = (
        cluster_changed
        & max_probability.ge(resolved_config.min_shift_probability)
        & shift_probability_delta.ge(resolved_config.shift_probability_delta_threshold)
    )
    if resolved_config.shift_requires_not_low_confidence:
        shift_flag &= ~low_confidence_mask
    shift_flag = shift_flag.astype("bool")

    top2_margin = posterior_frame.apply(_top_two_margin, axis=1).astype("float64")
    confidence_bucket = max_probability.map(
        lambda value: _confidence_bucket(value, thresholds=resolved_config.confidence_thresholds)
    ).astype("string")
    fallback_reason = pd.Series([None] * len(normalized), index=normalized.index, dtype="object")
    fallback_reason.loc[low_confidence_mask] = "low_confidence"
    ambiguous_mask = (
        top2_margin.lt(resolved_config.ambiguous_margin_threshold)
        & ~low_confidence_mask
    )
    fallback_reason.loc[ambiguous_mask] = "ambiguous"
    fallback_flag = fallback_reason.notna().astype("bool")

    labels = pd.DataFrame(
        {
            "ts_utc": normalized[resolved_config.time_column].to_numpy(copy=True),
            "gmm_cluster_label": cluster_series.to_numpy(copy=True),
            "gmm_previous_cluster_label": previous_cluster.astype("Int64").astype("object"),
            "gmm_shift_flag": shift_flag.to_numpy(copy=True),
            "gmm_shift_probability_delta": shift_probability_delta.to_numpy(copy=True),
            "gmm_posterior_probability": max_probability.to_numpy(copy=True),
            "gmm_confidence_score": max_probability.to_numpy(copy=True),
            "gmm_posterior_entropy": entropy.to_numpy(copy=True),
            "gmm_entropy_confidence": entropy_confidence.to_numpy(copy=True),
            "gmm_is_low_confidence": low_confidence_mask.to_numpy(copy=True),
            "confidence_score": max_probability.to_numpy(copy=True),
            "confidence_bucket": confidence_bucket.to_numpy(copy=True),
            "fallback_flag": fallback_flag.to_numpy(copy=True),
            "fallback_reason": fallback_reason.to_numpy(copy=True),
            "predicted_label": cluster_series.map(lambda value: f"gmm_cluster_{int(value)}").astype("string").to_numpy(copy=True),
        }
    )
    labels = labels.loc[:, list(GMM_LABEL_COLUMNS)]

    posterior_output = pd.concat(
        [
            labels.loc[:, ["ts_utc"]],
            posterior_frame,
        ],
        axis=1,
    )

    events = _build_shift_events(labels)
    summary = _build_summary(
        labels=labels,
        posterior=posterior_frame,
        config=resolved_config,
        component_order=component_order,
    )

    return RegimeGMMClassifierResult(
        labels=labels,
        posterior_probabilities=posterior_output,
        shift_events=events,
        summary=summary,
        config=resolved_config,
    )


def _validate_config(config: RegimeGMMClassifierConfig) -> None:
    if not isinstance(config.n_components, int) or config.n_components < 2:
        raise RegimeGMMClassifierError("Regime GMM config field 'n_components' must be an integer >= 2.")
    if not isinstance(config.random_state, int):
        raise RegimeGMMClassifierError("Regime GMM config field 'random_state' must be an integer.")
    if not isinstance(config.max_iter, int) or config.max_iter < 1:
        raise RegimeGMMClassifierError("Regime GMM config field 'max_iter' must be a positive integer.")
    if not isinstance(config.n_init, int) or config.n_init < 1:
        raise RegimeGMMClassifierError("Regime GMM config field 'n_init' must be a positive integer.")
    if config.covariance_type not in {"full", "tied", "diag", "spherical"}:
        raise RegimeGMMClassifierError(
            "Regime GMM config field 'covariance_type' must be one of: full, tied, diag, spherical."
        )
    for field_name in (
        "low_confidence_threshold",
        "ambiguous_margin_threshold",
        "min_shift_probability",
        "shift_probability_delta_threshold",
    ):
        value = getattr(config, field_name)
        if not isinstance(value, int | float):
            raise RegimeGMMClassifierError(f"Regime GMM config field {field_name!r} must be numeric.")
        if field_name != "shift_probability_delta_threshold" and not (0.0 <= float(value) <= 1.0):
            raise RegimeGMMClassifierError(
                f"Regime GMM config field {field_name!r} must be between 0 and 1."
            )
        if field_name == "shift_probability_delta_threshold" and float(value) < 0.0:
            raise RegimeGMMClassifierError(
                "Regime GMM config field 'shift_probability_delta_threshold' must be non-negative."
            )
    if not isinstance(config.feature_columns, tuple | list) or len(config.feature_columns) == 0:
        raise RegimeGMMClassifierError("Regime GMM config field 'feature_columns' must contain at least one column.")
    normalized_columns = [str(column).strip() for column in config.feature_columns]
    if any(not column for column in normalized_columns):
        raise RegimeGMMClassifierError("Regime GMM feature columns must be non-empty strings.")
    if len(set(normalized_columns)) != len(normalized_columns):
        raise RegimeGMMClassifierError("Regime GMM feature columns must not contain duplicates.")
    thresholds = config.confidence_thresholds
    if not isinstance(thresholds, dict):
        raise RegimeGMMClassifierError("Regime GMM config field 'confidence_thresholds' must be a dictionary.")
    missing = sorted({"high", "medium"} - set(thresholds))
    if missing:
        raise RegimeGMMClassifierError(
            f"Regime GMM confidence_thresholds missing required keys: {missing}."
        )
    medium = thresholds["medium"]
    high = thresholds["high"]
    for name, value in (("medium", medium), ("high", high)):
        if not isinstance(value, int | float) or not (0.0 <= float(value) <= 1.0):
            raise RegimeGMMClassifierError(
                f"Regime GMM confidence threshold {name!r} must be between 0 and 1."
            )
    if float(medium) > float(high):
        raise RegimeGMMClassifierError("Regime GMM confidence thresholds require medium <= high.")


def _normalize_features(frame: pd.DataFrame, *, config: RegimeGMMClassifierConfig) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Regime GMM features must be provided as a pandas DataFrame.")
    if frame.empty:
        raise RegimeGMMClassifierError("Regime GMM features must not be empty.")

    missing = [
        column
        for column in (config.time_column, *config.feature_columns)
        if column not in frame.columns
    ]
    if missing:
        raise RegimeGMMClassifierError(f"Regime GMM input is missing required columns: {missing}.")

    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    normalized[config.time_column] = pd.to_datetime(normalized[config.time_column], utc=True, errors="coerce")
    if normalized[config.time_column].isna().any():
        raise RegimeGMMClassifierError(
            f"Regime GMM input contains invalid UTC timestamps in {config.time_column!r}."
        )
    if normalized[config.time_column].duplicated().any():
        raise RegimeGMMClassifierError("Regime GMM input contains duplicate timestamps.")

    normalized = normalized.sort_values(config.time_column, kind="stable").reset_index(drop=True)
    for column in config.feature_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")
        if normalized[column].isna().any():
            raise RegimeGMMClassifierError(
                f"Regime GMM feature column {column!r} must not contain missing values."
            )
        finite_mask = normalized[column].map(math.isfinite)
        if not finite_mask.all():
            raise RegimeGMMClassifierError(
                f"Regime GMM feature column {column!r} must contain only finite values."
            )
    return normalized.loc[:, [config.time_column, *list(config.feature_columns)]]


def _require_sklearn() -> None:
    if GaussianMixture is None:
        raise RegimeGMMClassifierError(
            "scikit-learn is required for regime GMM classification. Install project ML dependencies first."
        ) from _SKLEARN_IMPORT_ERROR


def _stable_component_order(means: np.ndarray) -> list[int]:
    if means.ndim != 2:
        raise RegimeGMMClassifierError("Gaussian Mixture means must be a 2D matrix.")
    keys = [
        (index, tuple(float(value) for value in means[index].tolist()))
        for index in range(means.shape[0])
    ]
    keys.sort(key=lambda item: item[1])
    return [index for index, _ in keys]


def _entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _entropy_confidence(entropy: float, *, class_count: int) -> float:
    if class_count <= 1:
        return 1.0
    normalizer = math.log(float(class_count))
    if normalizer <= 0.0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - (entropy / normalizer))))


def _top_two_margin(row: pd.Series) -> float:
    values = np.sort(row.to_numpy(dtype="float64", copy=True))
    if len(values) < 2:
        return 1.0
    return float(values[-1] - values[-2])


def _confidence_bucket(score: float, *, thresholds: dict[str, float]) -> str:
    high = float(thresholds["high"])
    medium = float(thresholds["medium"])
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def _build_shift_events(labels: pd.DataFrame) -> pd.DataFrame:
    shift_rows = labels.loc[labels["gmm_shift_flag"].astype("bool")].copy(deep=True)
    if shift_rows.empty:
        return pd.DataFrame(columns=list(GMM_SHIFT_EVENT_COLUMNS))
    rows: list[dict[str, Any]] = []
    for event_order, row in enumerate(shift_rows.itertuples(index=False), start=1):
        rows.append(
            {
                "event_id": f"gmm_shift:{event_order:04d}",
                "event_order": event_order,
                "ts_utc": row.ts_utc,
                "previous_cluster_label": None
                if row.gmm_previous_cluster_label is None
                else int(row.gmm_previous_cluster_label),
                "current_cluster_label": int(row.gmm_cluster_label),
                "shift_probability_delta": None
                if pd.isna(row.gmm_shift_probability_delta)
                else float(row.gmm_shift_probability_delta),
                "confidence_score": float(row.gmm_confidence_score),
                "is_low_confidence": bool(row.gmm_is_low_confidence),
                "taxonomy_version": TAXONOMY_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=list(GMM_SHIFT_EVENT_COLUMNS))


def _build_summary(
    *,
    labels: pd.DataFrame,
    posterior: pd.DataFrame,
    config: RegimeGMMClassifierConfig,
    component_order: list[int],
) -> dict[str, Any]:
    cluster_distribution = (
        labels["gmm_cluster_label"].value_counts(sort=False).sort_index().astype("int64")
    )
    confidence_distribution = labels["confidence_bucket"].value_counts(sort=False).astype("int64")
    return canonicalize_value(
        {
            "artifact_type": "regime_gmm_classifier",
            "taxonomy_version": TAXONOMY_VERSION,
            "row_count": int(len(labels)),
            "cluster_count": int(len(component_order)),
            "cluster_order": [int(value) for value in component_order],
            "feature_columns": list(config.feature_columns),
            "cluster_distribution": {
                f"cluster_{int(index)}": int(cluster_distribution.get(index, 0))
                for index in range(len(component_order))
            },
            "confidence_bucket_distribution": {
                bucket: int(confidence_distribution.get(bucket, 0))
                for bucket in ("high", "medium", "low")
            },
            "low_confidence_share": float(labels["gmm_is_low_confidence"].astype("float64").mean()),
            "shift_event_count": int(labels["gmm_shift_flag"].astype("int64").sum()),
            "mean_entropy": float(labels["gmm_posterior_entropy"].astype("float64").mean()),
            "posterior_shape": [int(len(posterior)), int(posterior.shape[1])],
            "config": config.to_dict(),
        }
    )


__all__ = [
    "DEFAULT_GMM_CONFIDENCE_THRESHOLDS",
    "GMM_LABEL_COLUMNS",
    "GMM_SHIFT_EVENT_COLUMNS",
    "RegimeGMMClassifierConfig",
    "RegimeGMMClassifierError",
    "RegimeGMMClassifierResult",
    "classify_regime_shifts_with_gmm",
    "resolve_regime_gmm_classifier_config",
]