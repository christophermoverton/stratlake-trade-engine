from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research.alpha.base import BaseAlphaModel

STRUCTURAL_COLUMNS = frozenset({"symbol", "ts_utc", "timeframe"})


@dataclass(frozen=True)
class LinearModelSpec:
    ridge_penalty: float = 0.0
    fit_intercept: bool = True
    min_cross_section_size: int = 2


@dataclass(frozen=True)
class XGBoostModelSpec:
    random_state: int = 20260302
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: float = 5.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    n_jobs: int = 1
    tree_method: str = "hist"
    objective: str = "reg:squarederror"
    importance_type: str = "gain"


@dataclass(frozen=True)
class LightGBMModelSpec:
    random_state: int = 20260302
    n_estimators: int = 200
    max_depth: int = -1
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    n_jobs: int = 1
    objective: str = "regression"
    boosting_type: str = "gbdt"


@dataclass(frozen=True)
class ElasticNetModelSpec:
    alpha: float = 0.1
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    max_iter: int = 2000
    tol: float = 1e-4
    selection: str = "cyclic"
    min_cross_section_size: int = 2


class CrossSectionalLinearAlphaModel(BaseAlphaModel):
    """Pooled cross-sectional linear alpha with inspectable coefficients."""

    name = "cross_sectional_linear_alpha_model"

    def __init__(self, *, spec: LinearModelSpec | None = None) -> None:
        resolved_spec = spec or LinearModelSpec()
        if resolved_spec.ridge_penalty < 0.0:
            raise ValueError("ridge_penalty must be greater than or equal to zero.")
        if resolved_spec.min_cross_section_size < 2:
            raise ValueError("min_cross_section_size must be greater than or equal to two.")

        self.spec = resolved_spec
        self.feature_columns: list[str] = []
        self.target_column: str | None = None
        self.coefficients: np.ndarray | None = None
        self.intercept: float = 0.0
        self.coefficient_by_feature: dict[str, float] = {}
        self.training_metadata: dict[str, float | int] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = _resolve_feature_columns(df)
        self.target_column = _resolve_target_column(df)
        design_matrix = _to_numeric_frame(df, self.feature_columns)
        target = _to_numeric_series(df, self.target_column)
        training_matrix, training_target = _build_cross_sectional_training_arrays(
            design_matrix,
            target,
            ts_utc=df["ts_utc"],
            min_cross_section_size=self.spec.min_cross_section_size,
        )

        augmented = _augment_design_matrix(training_matrix, fit_intercept=self.spec.fit_intercept)
        regularization = np.eye(augmented.shape[1], dtype="float64") * float(self.spec.ridge_penalty)
        if self.spec.fit_intercept and regularization.size:
            regularization[0, 0] = 0.0

        lhs = augmented.T @ augmented + regularization
        rhs = augmented.T @ training_target
        solution = np.linalg.pinv(lhs) @ rhs
        if self.spec.fit_intercept:
            self.intercept = float(solution[0])
            self.coefficients = solution[1:]
        else:
            self.intercept = 0.0
            self.coefficients = solution

        self.coefficient_by_feature = {
            column: float(value)
            for column, value in zip(self.feature_columns, self.coefficients, strict=False)
        }
        self.training_metadata = {
            "n_rows": int(len(df)),
            "n_training_samples": int(training_matrix.shape[0]),
            "n_cross_sections": int(pd.to_datetime(df["ts_utc"], utc=True).nunique()),
            "ridge_penalty": float(self.spec.ridge_penalty),
            "fit_intercept": int(self.spec.fit_intercept),
            "min_cross_section_size": int(self.spec.min_cross_section_size),
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before predict.")

        design_matrix = _to_numeric_frame(df, self.feature_columns)
        transformed = _demean_by_cross_section(
            design_matrix,
            ts_utc=df["ts_utc"],
            min_cross_section_size=self.spec.min_cross_section_size,
        )
        prediction = transformed.to_numpy(dtype="float64") @ self.coefficients + self.intercept
        return pd.Series(prediction, index=df.index, dtype="float64", name="prediction")


class RidgeLinearAlphaModel(CrossSectionalLinearAlphaModel):
    """Cross-sectional ridge baseline with inspectable shrinkage coefficients."""

    name = "ridge_linear_alpha_model"


class CrossSectionalXGBoostAlphaModel(BaseAlphaModel):
    """Deterministic pooled cross-sectional XGBoost regressor."""

    name = "cross_sectional_xgboost_alpha_model"

    def __init__(self, *, spec: XGBoostModelSpec | None = None) -> None:
        resolved_spec = spec or XGBoostModelSpec()
        _validate_xgboost_spec(resolved_spec)

        self.spec = resolved_spec
        self.feature_columns: list[str] = []
        self.target_column: str | None = None
        self.model = None
        self.feature_importance_by_name: dict[str, float] = {}
        self.model_params: dict[str, float | int | str] = _xgboost_params_from_spec(resolved_spec)
        self.training_metadata: dict[str, float | int | str] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = _resolve_feature_columns(df)
        self.target_column = _resolve_target_column(df)

        design_matrix = _to_numeric_frame_allowing_missing(df, self.feature_columns)
        target = _to_numeric_series_allowing_missing(df, self.target_column)
        valid_rows = ~target.isna()
        if not bool(valid_rows.any()):
            raise ValueError("Cross-sectional XGBoost alpha requires at least one non-null target row.")

        design_matrix = design_matrix.loc[valid_rows]
        target = target.loc[valid_rows]
        if len(target) < 2:
            raise ValueError("Cross-sectional XGBoost alpha requires at least two training rows.")

        xgb_regressor_cls = _resolve_xgb_regressor_cls()
        self.model = xgb_regressor_cls(**self.model_params)
        self.model.fit(design_matrix, target)

        raw_importance = getattr(self.model, "feature_importances_", None)
        if raw_importance is None:
            self.feature_importance_by_name = {}
        else:
            self.feature_importance_by_name = {
                column: float(value)
                for column, value in zip(self.feature_columns, raw_importance, strict=False)
            }

        self.training_metadata = {
            "n_rows": int(len(df)),
            "n_training_rows": int(len(target)),
            "n_cross_sections": int(pd.to_datetime(df["ts_utc"], utc=True).nunique()),
            "target_non_null_rows": int(valid_rows.sum()),
            **self.model_params,
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model must be fit before predict.")

        design_matrix = _to_numeric_frame_allowing_missing(df, self.feature_columns)
        prediction = self.model.predict(design_matrix)
        return pd.Series(prediction, index=df.index, dtype="float64", name="prediction")


class CrossSectionalLightGBMAlphaModel(BaseAlphaModel):
    """Deterministic pooled cross-sectional LightGBM regressor."""

    name = "cross_sectional_lightgbm_alpha_model"

    def __init__(self, *, spec: LightGBMModelSpec | None = None) -> None:
        resolved_spec = spec or LightGBMModelSpec()
        _validate_lightgbm_spec(resolved_spec)

        self.spec = resolved_spec
        self.feature_columns: list[str] = []
        self.target_column: str | None = None
        self.model = None
        self.feature_importance_by_name: dict[str, float] = {}
        self.model_params: dict[str, float | int | str] = _lightgbm_params_from_spec(resolved_spec)
        self.training_metadata: dict[str, float | int | str] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = _resolve_feature_columns(df)
        self.target_column = _resolve_target_column(df)

        design_matrix = _to_numeric_frame_allowing_missing(df, self.feature_columns)
        target = _to_numeric_series_allowing_missing(df, self.target_column)
        valid_rows = ~target.isna()
        if not bool(valid_rows.any()):
            raise ValueError("Cross-sectional LightGBM alpha requires at least one non-null target row.")

        design_matrix = design_matrix.loc[valid_rows]
        target = target.loc[valid_rows]
        if len(target) < 2:
            raise ValueError("Cross-sectional LightGBM alpha requires at least two training rows.")

        lgbm_regressor_cls = _resolve_lgbm_regressor_cls()
        self.model = lgbm_regressor_cls(**self.model_params)
        self.model.fit(design_matrix, target)

        raw_importance = getattr(self.model, "feature_importances_", None)
        if raw_importance is None:
            self.feature_importance_by_name = {}
        else:
            self.feature_importance_by_name = {
                column: float(value)
                for column, value in zip(self.feature_columns, raw_importance, strict=False)
            }

        self.training_metadata = {
            "n_rows": int(len(df)),
            "n_training_rows": int(len(target)),
            "n_cross_sections": int(pd.to_datetime(df["ts_utc"], utc=True).nunique()),
            "target_non_null_rows": int(valid_rows.sum()),
            **self.model_params,
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model must be fit before predict.")

        design_matrix = _to_numeric_frame_allowing_missing(df, self.feature_columns)
        prediction = self.model.predict(design_matrix)
        return pd.Series(prediction, index=df.index, dtype="float64", name="prediction")


class CrossSectionalElasticNetAlphaModel(BaseAlphaModel):
    """Deterministic pooled cross-sectional Elastic Net regressor."""

    name = "cross_sectional_elastic_net_alpha_model"

    def __init__(self, *, spec: ElasticNetModelSpec | None = None) -> None:
        resolved_spec = spec or ElasticNetModelSpec()
        _validate_elastic_net_spec(resolved_spec)

        self.spec = resolved_spec
        self.feature_columns: list[str] = []
        self.target_column: str | None = None
        self.model = None
        self.coefficient_by_feature: dict[str, float] = {}
        self.model_params: dict[str, float | int | str | bool] = _elastic_net_params_from_spec(resolved_spec)
        self.training_metadata: dict[str, float | int | str | bool] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = _resolve_feature_columns(df)
        self.target_column = _resolve_target_column(df)
        design_matrix = _to_numeric_frame(df, self.feature_columns)
        target = _to_numeric_series(df, self.target_column)
        training_matrix, training_target = _build_cross_sectional_training_arrays(
            design_matrix,
            target,
            ts_utc=df["ts_utc"],
            min_cross_section_size=self.spec.min_cross_section_size,
        )

        elastic_net_cls = _resolve_elastic_net_regressor_cls()
        self.model = elastic_net_cls(**self.model_params)
        self.model.fit(training_matrix, training_target)

        coefficients = getattr(self.model, "coef_", None)
        if coefficients is None:
            self.coefficient_by_feature = {}
        else:
            self.coefficient_by_feature = {
                column: float(value)
                for column, value in zip(self.feature_columns, coefficients, strict=False)
            }

        self.training_metadata = {
            "n_rows": int(len(df)),
            "n_training_samples": int(training_matrix.shape[0]),
            "n_cross_sections": int(pd.to_datetime(df["ts_utc"], utc=True).nunique()),
            **self.model_params,
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model must be fit before predict.")

        design_matrix = _to_numeric_frame(df, self.feature_columns)
        transformed = _demean_by_cross_section(
            design_matrix,
            ts_utc=df["ts_utc"],
            min_cross_section_size=self.spec.min_cross_section_size,
        )
        prediction = self.model.predict(transformed.to_numpy(dtype="float64"))
        return pd.Series(prediction, index=df.index, dtype="float64", name="prediction")


class RankCompositeAlphaModel(BaseAlphaModel):
    """Rank-composite alpha using deterministic training-time IC weights."""

    name = "rank_composite_alpha_model"

    def __init__(
        self,
        *,
        min_cross_section_size: int = 2,
        normalize: bool = True,
        use_ic_weights: bool = True,
    ) -> None:
        if min_cross_section_size < 2:
            raise ValueError("min_cross_section_size must be greater than or equal to two.")
        self.min_cross_section_size = int(min_cross_section_size)
        self.normalize = bool(normalize)
        self.use_ic_weights = bool(use_ic_weights)
        self.feature_columns: list[str] = []
        self.target_column: str | None = None
        self.feature_weight_by_name: dict[str, float] = {}
        self.feature_ic_by_name: dict[str, float] = {}
        self.training_metadata: dict[str, float | int] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = _resolve_feature_columns(df)
        self.target_column = _resolve_target_column(df)
        target = _to_numeric_series(df, self.target_column)
        target_ranks = _rank_by_cross_section(
            target,
            ts_utc=df["ts_utc"],
            min_cross_section_size=self.min_cross_section_size,
            normalize=self.normalize,
        )

        ic_by_feature: dict[str, float] = {}
        for column in self.feature_columns:
            feature_values = _to_numeric_series(df, column)
            feature_ranks = _rank_by_cross_section(
                feature_values,
                ts_utc=df["ts_utc"],
                min_cross_section_size=self.min_cross_section_size,
                normalize=self.normalize,
            )
            ic_by_feature[column] = _safe_correlation(feature_ranks, target_ranks)

        self.feature_ic_by_name = ic_by_feature
        self.feature_weight_by_name = _normalize_weights(ic_by_feature, enabled=self.use_ic_weights)
        self.training_metadata = {
            "n_rows": int(len(df)),
            "n_cross_sections": int(pd.to_datetime(df["ts_utc"], utc=True).nunique()),
            "min_cross_section_size": int(self.min_cross_section_size),
            "normalize": int(self.normalize),
            "use_ic_weights": int(self.use_ic_weights),
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.feature_columns:
            raise RuntimeError("Model must be fit before predict.")

        scores = pd.Series(0.0, index=df.index, dtype="float64")
        for column in self.feature_columns:
            values = _to_numeric_series(df, column)
            ranks = _rank_by_cross_section(
                values,
                ts_utc=df["ts_utc"],
                min_cross_section_size=self.min_cross_section_size,
                normalize=self.normalize,
            )
            scores = scores + ranks * float(self.feature_weight_by_name[column])
        return scores.rename("prediction")


def _resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    target_column = _resolve_target_column(df)
    feature_columns = [
        column
        for column in df.columns
        if column not in STRUCTURAL_COLUMNS and column != target_column
    ]
    if not feature_columns:
        raise ValueError("Built-in alpha models require at least one feature column.")
    return feature_columns


def _resolve_target_column(df: pd.DataFrame) -> str:
    target_candidates = [column for column in df.columns if column.startswith("target_")]
    if len(target_candidates) != 1:
        raise ValueError(
            "Built-in alpha models expect exactly one target_* column in the training frame."
        )
    return target_candidates[0]


def _to_numeric_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    numeric = df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        missing = [column for column in feature_columns if numeric[column].isna().any()]
        formatted = ", ".join(missing)
        raise ValueError(f"Built-in alpha model features must be numeric and non-null. Offending columns: {formatted}.")
    return numeric.astype("float64")


def _to_numeric_frame_allowing_missing(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    numeric = df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    return numeric.astype("float64")


def _to_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce").astype("float64")
    if values.isna().any():
        raise ValueError(f"Column '{column}' must be numeric and non-null for built-in alpha models.")
    return pd.Series(values, index=df.index, dtype="float64", name=column)


def _to_numeric_series_allowing_missing(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return pd.Series(values, index=df.index, dtype="float64", name=column)


def _build_cross_sectional_training_arrays(
    design_matrix: pd.DataFrame,
    target: pd.Series,
    *,
    ts_utc: pd.Series,
    min_cross_section_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    transformed_features = _demean_by_cross_section(
        design_matrix,
        ts_utc=ts_utc,
        min_cross_section_size=min_cross_section_size,
    )
    transformed_target = _demean_series_by_cross_section(
        target,
        ts_utc=ts_utc,
        min_cross_section_size=min_cross_section_size,
    )
    return (
        transformed_features.to_numpy(dtype="float64"),
        transformed_target.to_numpy(dtype="float64"),
    )


def _demean_by_cross_section(
    values: pd.DataFrame,
    *,
    ts_utc: pd.Series,
    min_cross_section_size: int,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    counts = timestamps.groupby(timestamps, sort=False).transform("size")
    group_means = values.groupby(timestamps, sort=False).transform("mean")
    demeaned = values - group_means
    valid_rows = counts >= int(min_cross_section_size)
    if not bool(valid_rows.all()):
        demeaned = demeaned.where(valid_rows.to_numpy()[:, None], other=0.0)
    return demeaned.astype("float64")


def _demean_series_by_cross_section(
    values: pd.Series,
    *,
    ts_utc: pd.Series,
    min_cross_section_size: int,
) -> pd.Series:
    timestamps = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    counts = timestamps.groupby(timestamps, sort=False).transform("size")
    demeaned = values - values.groupby(timestamps, sort=False).transform("mean")
    valid_rows = counts >= int(min_cross_section_size)
    return demeaned.where(valid_rows, other=0.0).astype("float64")


def _rank_by_cross_section(
    values: pd.Series,
    *,
    ts_utc: pd.Series,
    min_cross_section_size: int,
    normalize: bool,
) -> pd.Series:
    timestamps = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    counts = values.groupby(timestamps, sort=False).transform("count").astype("float64")
    valid_rows = counts >= float(min_cross_section_size)
    ranks = values.groupby(timestamps, sort=False).rank(
        method="average",
        ascending=False,
        na_option="keep",
    )
    if normalize:
        centered = (counts + 1.0) / 2.0 - ranks
        scale = (counts - 1.0) / 2.0
        ranks = centered.div(scale.replace(0.0, np.nan)).fillna(0.0)
    else:
        ranks = ranks.fillna(0.0)
    return ranks.where(valid_rows, other=0.0).astype("float64")


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) == 0:
        return 0.0
    left_values = left.to_numpy(dtype="float64")
    right_values = right.to_numpy(dtype="float64")
    if np.allclose(left_values, left_values[0]) or np.allclose(right_values, right_values[0]):
        return 0.0
    correlation = float(np.corrcoef(left_values, right_values)[0, 1])
    if np.isnan(correlation):
        return 0.0
    return correlation


def _normalize_weights(weights: dict[str, float], *, enabled: bool) -> dict[str, float]:
    if not weights:
        return {}
    if not enabled:
        equal_weight = 1.0 / float(len(weights))
        return {name: equal_weight for name in weights}

    scale = sum(abs(weight) for weight in weights.values())
    if scale <= 0.0:
        equal_weight = 1.0 / float(len(weights))
        return {name: equal_weight for name in weights}
    return {name: float(weight) / float(scale) for name, weight in weights.items()}


def _augment_design_matrix(matrix: np.ndarray, *, fit_intercept: bool) -> np.ndarray:
    if not fit_intercept:
        return matrix
    intercept = np.ones((matrix.shape[0], 1), dtype="float64")
    return np.hstack([intercept, matrix])


def _validate_xgboost_spec(spec: XGBoostModelSpec) -> None:
    if spec.n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero.")
    if spec.max_depth <= 0:
        raise ValueError("max_depth must be greater than zero.")
    if spec.learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than zero.")
    if spec.min_child_weight < 0.0:
        raise ValueError("min_child_weight must be greater than or equal to zero.")
    for field_name in ("subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"):
        value = float(getattr(spec, field_name))
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"{field_name} must be in the interval (0, 1].")
    if spec.reg_alpha < 0.0:
        raise ValueError("reg_alpha must be greater than or equal to zero.")
    if spec.reg_lambda < 0.0:
        raise ValueError("reg_lambda must be greater than or equal to zero.")
    if spec.gamma < 0.0:
        raise ValueError("gamma must be greater than or equal to zero.")
    if spec.n_jobs <= 0:
        raise ValueError("n_jobs must be greater than zero.")
    if not spec.tree_method.strip():
        raise ValueError("tree_method must be a non-empty string.")
    if not spec.objective.strip():
        raise ValueError("objective must be a non-empty string.")
    if not spec.importance_type.strip():
        raise ValueError("importance_type must be a non-empty string.")


def _xgboost_params_from_spec(spec: XGBoostModelSpec) -> dict[str, float | int | str]:
    return {
        "colsample_bylevel": float(spec.colsample_bylevel),
        "colsample_bynode": float(spec.colsample_bynode),
        "colsample_bytree": float(spec.colsample_bytree),
        "gamma": float(spec.gamma),
        "importance_type": str(spec.importance_type),
        "learning_rate": float(spec.learning_rate),
        "max_depth": int(spec.max_depth),
        "min_child_weight": float(spec.min_child_weight),
        "n_estimators": int(spec.n_estimators),
        "n_jobs": int(spec.n_jobs),
        "objective": str(spec.objective),
        "random_state": int(spec.random_state),
        "reg_alpha": float(spec.reg_alpha),
        "reg_lambda": float(spec.reg_lambda),
        "subsample": float(spec.subsample),
        "tree_method": str(spec.tree_method),
        "verbosity": 0,
    }


def _validate_lightgbm_spec(spec: LightGBMModelSpec) -> None:
    if spec.n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero.")
    if spec.max_depth == 0 or spec.max_depth < -1:
        raise ValueError("max_depth must be -1 (unbounded) or greater than zero.")
    if spec.learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than zero.")
    if spec.num_leaves <= 1:
        raise ValueError("num_leaves must be greater than one.")
    if spec.min_child_samples < 1:
        raise ValueError("min_child_samples must be greater than or equal to one.")
    for field_name in ("subsample", "colsample_bytree"):
        value = float(getattr(spec, field_name))
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"{field_name} must be in the interval (0, 1].")
    if spec.reg_alpha < 0.0:
        raise ValueError("reg_alpha must be greater than or equal to zero.")
    if spec.reg_lambda < 0.0:
        raise ValueError("reg_lambda must be greater than or equal to zero.")
    if spec.n_jobs <= 0:
        raise ValueError("n_jobs must be greater than zero.")
    if not spec.objective.strip():
        raise ValueError("objective must be a non-empty string.")
    if not spec.boosting_type.strip():
        raise ValueError("boosting_type must be a non-empty string.")


def _lightgbm_params_from_spec(spec: LightGBMModelSpec) -> dict[str, float | int | str]:
    return {
        "boosting_type": str(spec.boosting_type),
        "colsample_bytree": float(spec.colsample_bytree),
        "learning_rate": float(spec.learning_rate),
        "max_depth": int(spec.max_depth),
        "min_child_samples": int(spec.min_child_samples),
        "n_estimators": int(spec.n_estimators),
        "n_jobs": int(spec.n_jobs),
        "num_leaves": int(spec.num_leaves),
        "objective": str(spec.objective),
        "random_state": int(spec.random_state),
        "reg_alpha": float(spec.reg_alpha),
        "reg_lambda": float(spec.reg_lambda),
        "subsample": float(spec.subsample),
        "verbosity": -1,
    }


def _validate_elastic_net_spec(spec: ElasticNetModelSpec) -> None:
    if spec.alpha < 0.0:
        raise ValueError("alpha must be greater than or equal to zero.")
    if not (0.0 <= spec.l1_ratio <= 1.0):
        raise ValueError("l1_ratio must be in the interval [0, 1].")
    if spec.max_iter <= 0:
        raise ValueError("max_iter must be greater than zero.")
    if spec.tol <= 0.0:
        raise ValueError("tol must be greater than zero.")
    if spec.selection.strip().lower() not in {"cyclic", "random"}:
        raise ValueError("selection must be either 'cyclic' or 'random'.")
    if spec.min_cross_section_size < 2:
        raise ValueError("min_cross_section_size must be greater than or equal to two.")


def _elastic_net_params_from_spec(spec: ElasticNetModelSpec) -> dict[str, float | int | str | bool]:
    return {
        "alpha": float(spec.alpha),
        "fit_intercept": bool(spec.fit_intercept),
        "l1_ratio": float(spec.l1_ratio),
        "max_iter": int(spec.max_iter),
        "random_state": 20260302,
        "selection": str(spec.selection).strip().lower(),
        "tol": float(spec.tol),
    }


def _resolve_xgb_regressor_cls():
    try:
        module = importlib.import_module("xgboost")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "xgboost is required for model_type 'cross_sectional_xgboost'. "
            "Install the 'xgboost' package to run this alpha."
        ) from exc

    xgb_regressor_cls = getattr(module, "XGBRegressor", None)
    if xgb_regressor_cls is None:
        raise RuntimeError("xgboost.XGBRegressor is not available in the installed xgboost package.")
    return xgb_regressor_cls


def _resolve_lgbm_regressor_cls():
    try:
        module = importlib.import_module("lightgbm")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "lightgbm is required for model_type 'cross_sectional_lightgbm'. "
            "Install the 'lightgbm' package to run this alpha."
        ) from exc

    lgbm_regressor_cls = getattr(module, "LGBMRegressor", None)
    if lgbm_regressor_cls is None:
        raise RuntimeError("lightgbm.LGBMRegressor is not available in the installed lightgbm package.")
    return lgbm_regressor_cls


def _resolve_elastic_net_regressor_cls():
    try:
        module = importlib.import_module("sklearn.linear_model")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for model_type 'cross_sectional_elastic_net'. "
            "Install the 'scikit-learn' package to run this alpha."
        ) from exc

    elastic_net_cls = getattr(module, "ElasticNet", None)
    if elastic_net_cls is None:
        raise RuntimeError(
            "sklearn.linear_model.ElasticNet is not available in the installed scikit-learn package."
        )
    return elastic_net_cls
