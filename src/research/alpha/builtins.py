from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.research.alpha.base import BaseAlphaModel

STRUCTURAL_COLUMNS = frozenset({"symbol", "ts_utc", "timeframe"})


@dataclass(frozen=True)
class LinearModelSpec:
    ridge_penalty: float = 0.0
    fit_intercept: bool = True


class LinearAlphaModel(BaseAlphaModel):
    """Deterministic closed-form linear predictor with optional ridge penalty."""

    name = "linear_alpha_model"

    def __init__(self, *, spec: LinearModelSpec | None = None) -> None:
        resolved_spec = spec or LinearModelSpec()
        if resolved_spec.ridge_penalty < 0.0:
            raise ValueError("ridge_penalty must be greater than or equal to zero.")
        self.spec = resolved_spec
        self.feature_columns: list[str] = []
        self.coefficients: np.ndarray | None = None
        self.intercept: float = 0.0
        self.target_column: str | None = None

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = _resolve_feature_columns(df)
        self.target_column = _resolve_target_column(df)
        design_matrix = _to_numeric_matrix(df, self.feature_columns)
        target = pd.to_numeric(df[self.target_column], errors="coerce").to_numpy(dtype="float64")
        if np.isnan(target).any():
            raise ValueError(f"Target column '{self.target_column}' must not contain NaN values in the training slice.")

        augmented = _augment_design_matrix(design_matrix, fit_intercept=self.spec.fit_intercept)
        regularization = np.eye(augmented.shape[1], dtype="float64") * float(self.spec.ridge_penalty)
        if self.spec.fit_intercept and regularization.size:
            regularization[0, 0] = 0.0

        lhs = augmented.T @ augmented + regularization
        rhs = augmented.T @ target
        solution = np.linalg.pinv(lhs) @ rhs
        if self.spec.fit_intercept:
            self.intercept = float(solution[0])
            self.coefficients = solution[1:]
        else:
            self.intercept = 0.0
            self.coefficients = solution

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before predict.")
        design_matrix = _to_numeric_matrix(df, self.feature_columns)
        prediction = design_matrix @ self.coefficients + self.intercept
        return pd.Series(prediction, index=df.index, dtype="float64", name="prediction")


class RankCompositeMomentumAlphaModel(BaseAlphaModel):
    """Cross-sectional rank composite of configured feature columns."""

    name = "rank_composite_momentum"

    def __init__(self, *, ascending: bool = False, normalize: bool = True) -> None:
        self.ascending = bool(ascending)
        self.normalize = bool(normalize)
        self.feature_columns: list[str] = []

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = _resolve_feature_columns(df)

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.feature_columns:
            raise RuntimeError("Model must be fit before predict.")

        scores = pd.Series(0.0, index=df.index, dtype="float64")
        group_key = df["ts_utc"] if "ts_utc" in df.columns else pd.Series("__all__", index=df.index, dtype="string")
        for column in self.feature_columns:
            values = pd.to_numeric(df[column], errors="coerce").astype("float64")
            ranks = values.groupby(group_key, sort=False).rank(
                method="average",
                ascending=self.ascending,
                na_option="keep",
            )
            if self.normalize:
                counts = values.groupby(group_key, sort=False).transform("count").astype("float64")
                centered = ranks - (counts + 1.0) / 2.0
                scale = (counts - 1.0) / 2.0
                normalized = centered.div(scale.replace(0.0, np.nan)).fillna(0.0)
                scores = scores + normalized.astype("float64")
            else:
                scores = scores + ranks.fillna(0.0).astype("float64")

        if self.feature_columns:
            scores = scores / float(len(self.feature_columns))
        return scores.rename("prediction")


class SklearnRegressorAlphaModel(BaseAlphaModel):
    """Deterministic wrapper for supported scikit-learn regression estimators."""

    name = "sklearn_regressor_alpha_model"

    def __init__(self, *, estimator_type: str, estimator_params: dict[str, object] | None = None) -> None:
        self.estimator_type = str(estimator_type).strip().lower()
        self.estimator_params = {} if estimator_params is None else dict(estimator_params)
        self.feature_columns: list[str] = []
        self.target_column: str | None = None
        self.estimator: Any | None = None

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = _resolve_feature_columns(df)
        self.target_column = _resolve_target_column(df)
        design_matrix = _to_numeric_matrix(df, self.feature_columns)
        target = pd.to_numeric(df[self.target_column], errors="coerce").to_numpy(dtype="float64")
        if np.isnan(target).any():
            raise ValueError(f"Target column '{self.target_column}' must not contain NaN values in the training slice.")

        estimator = _build_supported_sklearn_estimator(
            estimator_type=self.estimator_type,
            estimator_params=self.estimator_params,
        )
        estimator.fit(design_matrix, target)
        self.estimator = estimator

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if self.estimator is None:
            raise RuntimeError("Model must be fit before predict.")
        design_matrix = _to_numeric_matrix(df, self.feature_columns)
        prediction = self.estimator.predict(design_matrix)
        return pd.Series(prediction, index=df.index, dtype="float64", name="prediction")


def _resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    target_column = _resolve_target_column(df)
    return [
        column
        for column in df.columns
        if column not in STRUCTURAL_COLUMNS and column != target_column
    ]


def _resolve_target_column(df: pd.DataFrame) -> str:
    target_candidates = [column for column in df.columns if column.startswith("target_")]
    if len(target_candidates) != 1:
        raise ValueError(
            "Built-in alpha models expect exactly one target_* column in the training frame."
        )
    return target_candidates[0]


def _to_numeric_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if not feature_columns:
        raise ValueError("Built-in alpha models require at least one feature column.")
    numeric = df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        missing = [
            column
            for column in feature_columns
            if numeric[column].isna().any()
        ]
        formatted = ", ".join(missing)
        raise ValueError(f"Built-in alpha model features must be numeric and non-null. Offending columns: {formatted}.")
    return numeric.to_numpy(dtype="float64")


def _augment_design_matrix(matrix: np.ndarray, *, fit_intercept: bool) -> np.ndarray:
    if not fit_intercept:
        return matrix
    intercept = np.ones((matrix.shape[0], 1), dtype="float64")
    return np.hstack([intercept, matrix])


def _build_supported_sklearn_estimator(
    *,
    estimator_type: str,
    estimator_params: dict[str, object],
) -> Any:
    try:
        from sklearn.linear_model import LinearRegression, Ridge
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for sklearn-backed alpha models. Install project dependencies to use model_type='sklearn'."
        ) from exc

    normalized_type = estimator_type.strip().lower()
    if normalized_type in {"linear_regression", "linearregression"}:
        return LinearRegression(**estimator_params)
    if normalized_type == "ridge":
        return Ridge(**estimator_params)
    raise ValueError(
        "Unsupported scikit-learn estimator_type. Supported values: linear_regression, ridge."
    )
