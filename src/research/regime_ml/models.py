from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency environments
    LogisticRegression = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

from src.research.regime_ml.base import BaseRegimeModel, RegimeMLError


@dataclass
class LogisticRegressionRegimeModel(BaseRegimeModel):
    """Deterministic multinomial logistic-regression baseline."""

    random_seed: int = 42
    max_iter: int = 1000
    C: float = 1.0
    class_weight: str | dict[str, float] | None = None
    model: LogisticRegression | None = field(default=None, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    classes_: list[str] = field(default_factory=list, init=False)
    fit_metadata_: dict[str, Any] = field(default_factory=dict, init=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: dict[str, Any] | None = None,
    ) -> "LogisticRegressionRegimeModel":
        _require_sklearn()
        features = _validate_feature_frame(X)
        labels = _validate_target(y, expected_index=features.index)
        class_count = int(labels.astype("string").nunique())
        if class_count < 2:
            raise RegimeMLError("Regime classifier training requires at least two distinct regime labels.")

        estimator = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_seed,
            solver="lbfgs",
        )
        estimator.fit(features, labels.astype("string"))
        self.model = estimator
        self.feature_columns_ = list(features.columns)
        self.classes_ = [str(value) for value in estimator.classes_]
        self.fit_metadata_ = dict(sorted((metadata or {}).items()))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        estimator = _require_fit(self.model)
        features = _validate_prediction_frame(X, expected_columns=self.feature_columns_)
        return pd.Series(estimator.predict(features), index=features.index, dtype="string", name="predicted_label")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        estimator = _require_fit(self.model)
        features = _validate_prediction_frame(X, expected_columns=self.feature_columns_)
        probabilities = estimator.predict_proba(features)
        frame = pd.DataFrame(probabilities, index=features.index, columns=[str(label) for label in estimator.classes_])
        return frame.astype("float64")


def _require_fit(model: LogisticRegression | None) -> LogisticRegression:
    _require_sklearn()
    if model is None:
        raise RegimeMLError("Regime model must be fitted before prediction.")
    return model


def _require_sklearn() -> None:
    if LogisticRegression is None:
        raise RegimeMLError(
            "scikit-learn is required for regime ML. Install project ML dependencies before running this pipeline."
        ) from _SKLEARN_IMPORT_ERROR


def _validate_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Regime model features must be provided as a pandas DataFrame.")
    if X.empty:
        raise RegimeMLError("Regime model features must not be empty.")
    normalized = X.copy(deep=True)
    normalized.attrs = {}
    for column in normalized.columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")
    if normalized.isna().any().any():
        raise RegimeMLError("Regime model features must be finite after preprocessing.")
    return normalized


def _validate_target(y: pd.Series, *, expected_index: pd.Index) -> pd.Series:
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=expected_index)
    labels = y.astype("string")
    if not labels.index.equals(expected_index):
        labels = labels.reindex(expected_index)
    if labels.isna().any():
        raise RegimeMLError("Regime model labels must not contain missing values.")
    return labels


def _validate_prediction_frame(X: pd.DataFrame, *, expected_columns: list[str]) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Regime model features must be provided as a pandas DataFrame.")
    missing = [column for column in expected_columns if column not in X.columns]
    if missing:
        raise RegimeMLError(f"Prediction features are missing required columns: {missing}.")
    normalized = X.loc[:, expected_columns].copy(deep=True)
    normalized.attrs = {}
    for column in normalized.columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")
    if normalized.isna().any().any():
        raise RegimeMLError("Prediction features must be finite after preprocessing.")
    return normalized
