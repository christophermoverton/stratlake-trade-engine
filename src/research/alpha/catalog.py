from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from src.research.alpha.builtins import (
    CrossSectionalElasticNetAlphaModel,
    CrossSectionalLinearAlphaModel,
    CrossSectionalLightGBMAlphaModel,
    CrossSectionalXGBoostAlphaModel,
    ElasticNetModelSpec,
    LightGBMModelSpec,
    LinearModelSpec,
    RankCompositeAlphaModel,
    RidgeLinearAlphaModel,
    XGBoostModelSpec,
)
from src.research.alpha.registry import _ALPHA_MODEL_REGISTRY, register_alpha_factory

REPO_ROOT = Path(__file__).resolve().parents[3]
ALPHAS_CONFIG = REPO_ROOT / "configs" / "alphas.yml"

REQUIRED_ALPHA_FIELDS = (
    "alpha_name",
    "dataset",
    "target_column",
    "feature_columns",
    "model_type",
    "model_params",
    "alpha_horizon",
)


def load_alphas_config(path: Path = ALPHAS_CONFIG) -> dict[str, dict[str, Any]]:
    """Load the alpha registry YAML file."""

    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}

    if not isinstance(payload, dict):
        raise ValueError("Alpha configuration must be a mapping of alpha names to config dictionaries.")
    return {str(name): _normalize_alpha_entry(str(name), config) for name, config in payload.items()}


def get_alpha_config(
    alpha_name: str,
    alphas: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the requested alpha configuration or raise a clear validation error."""

    registry = dict(alphas) if alphas is not None else load_alphas_config()
    if alpha_name not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"Unknown alpha '{alpha_name}'. Available alphas: {available}.")
    config = registry[alpha_name]
    if not isinstance(config, dict):
        raise ValueError(f"Alpha '{alpha_name}' configuration must be a dictionary.")
    return dict(config)


def register_builtin_alpha_catalog(path: Path = ALPHAS_CONFIG) -> None:
    """Register seeded alpha definitions as runtime factories."""

    for alpha_name, config in load_alphas_config(path).items():
        if alpha_name in _ALPHA_MODEL_REGISTRY:
            continue
        register_alpha_factory(alpha_name, _build_alpha_factory(alpha_name, config))


def resolve_alpha_config(
    payload: Mapping[str, Any] | None,
    *,
    alpha_name: str | None = None,
    path: Path = ALPHAS_CONFIG,
) -> dict[str, Any]:
    """Resolve one CLI/config payload against the canonical alpha registry."""

    base_payload = {} if payload is None else dict(payload)
    resolved_alpha_name = _coerce_optional_string(alpha_name) or _coerce_optional_string(base_payload.get("alpha_name"))
    if resolved_alpha_name is None:
        return dict(base_payload)

    resolved = get_alpha_config(resolved_alpha_name, load_alphas_config(path))
    resolved.update(base_payload)
    resolved["alpha_name"] = resolved_alpha_name
    resolved.setdefault("alpha_model", resolved_alpha_name)
    return resolved


def _normalize_alpha_entry(alpha_name: str, config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError(f"Alpha '{alpha_name}' must map to a dictionary.")

    normalized = dict(config)
    normalized.setdefault("alpha_name", alpha_name)
    defaults = normalized.pop("defaults", {})
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError(f"Alpha '{alpha_name}' defaults must be a dictionary when provided.")
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            normalized.setdefault(key, value)

    missing = [field for field in REQUIRED_ALPHA_FIELDS if field not in normalized]
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(f"Alpha '{alpha_name}' is missing required fields: {formatted}.")

    if normalized["alpha_name"] != alpha_name:
        raise ValueError(f"Alpha '{alpha_name}' must define alpha_name: {alpha_name!r}.")
    if not isinstance(normalized["dataset"], str) or not normalized["dataset"].strip():
        raise ValueError(f"Alpha '{alpha_name}' must define a non-empty dataset.")
    if not isinstance(normalized["target_column"], str) or not normalized["target_column"].strip():
        raise ValueError(f"Alpha '{alpha_name}' must define a non-empty target_column.")
    if not isinstance(normalized["feature_columns"], list) or not normalized["feature_columns"]:
        raise ValueError(f"Alpha '{alpha_name}' must define a non-empty feature_columns list.")
    if not all(isinstance(column, str) and column.strip() for column in normalized["feature_columns"]):
        raise ValueError(f"Alpha '{alpha_name}' feature_columns entries must be non-empty strings.")
    if not isinstance(normalized["model_type"], str) or not normalized["model_type"].strip():
        raise ValueError(f"Alpha '{alpha_name}' must define a non-empty model_type.")
    if not isinstance(normalized["model_params"], dict):
        raise ValueError(f"Alpha '{alpha_name}' model_params must be a dictionary.")
    normalized["alpha_horizon"] = int(normalized["alpha_horizon"])
    if normalized["alpha_horizon"] <= 0:
        raise ValueError(f"Alpha '{alpha_name}' alpha_horizon must be greater than zero.")

    normalized["feature_columns"] = [str(column) for column in normalized["feature_columns"]]
    normalized["alpha_model"] = alpha_name
    return normalized


def _build_alpha_factory(alpha_name: str, config: Mapping[str, Any]):
    model_type = str(config["model_type"]).strip().lower()
    params = dict(config.get("model_params", {}))
    if model_type == "linear":
        spec = LinearModelSpec(
            ridge_penalty=float(params.get("ridge_penalty", 0.0)),
            fit_intercept=bool(params.get("fit_intercept", True)),
            min_cross_section_size=int(params.get("min_cross_section_size", config.get("min_cross_section_size", 2))),
        )
        return partial(CrossSectionalLinearAlphaModel, spec=spec)
    if model_type == "cross_sectional_linear":
        spec = LinearModelSpec(
            ridge_penalty=float(params.get("ridge_penalty", 0.0)),
            fit_intercept=bool(params.get("fit_intercept", True)),
            min_cross_section_size=int(params.get("min_cross_section_size", config.get("min_cross_section_size", 2))),
        )
        return partial(CrossSectionalLinearAlphaModel, spec=spec)
    if model_type in {"ridge", "ridge_linear"}:
        spec = LinearModelSpec(
            ridge_penalty=float(params.get("ridge_penalty", params.get("alpha", 1.0))),
            fit_intercept=bool(params.get("fit_intercept", True)),
            min_cross_section_size=int(params.get("min_cross_section_size", config.get("min_cross_section_size", 2))),
        )
        return partial(RidgeLinearAlphaModel, spec=spec)
    if model_type in {"cross_sectional_xgboost", "xgboost_cross_sectional", "xgboost"}:
        spec = XGBoostModelSpec(
            random_state=int(params.get("random_state", 20260302)),
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            min_child_weight=float(params.get("min_child_weight", 5.0)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            colsample_bylevel=float(params.get("colsample_bylevel", 1.0)),
            colsample_bynode=float(params.get("colsample_bynode", 1.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            gamma=float(params.get("gamma", 0.0)),
            n_jobs=int(params.get("n_jobs", 1)),
            tree_method=str(params.get("tree_method", "hist")),
            objective=str(params.get("objective", "reg:squarederror")),
            importance_type=str(params.get("importance_type", "gain")),
        )
        return partial(CrossSectionalXGBoostAlphaModel, spec=spec)
    if model_type in {"cross_sectional_lightgbm", "lightgbm_cross_sectional", "lightgbm"}:
        spec = LightGBMModelSpec(
            random_state=int(params.get("random_state", 20260302)),
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", -1)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
            n_jobs=int(params.get("n_jobs", 1)),
            objective=str(params.get("objective", "regression")),
            boosting_type=str(params.get("boosting_type", "gbdt")),
        )
        return partial(CrossSectionalLightGBMAlphaModel, spec=spec)
    if model_type in {"cross_sectional_elastic_net", "elastic_net", "elasticnet"}:
        spec = ElasticNetModelSpec(
            alpha=float(params.get("alpha", 0.1)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            fit_intercept=bool(params.get("fit_intercept", True)),
            max_iter=int(params.get("max_iter", 2000)),
            tol=float(params.get("tol", 1e-4)),
            selection=str(params.get("selection", "cyclic")),
            min_cross_section_size=int(params.get("min_cross_section_size", config.get("min_cross_section_size", 2))),
        )
        return partial(CrossSectionalElasticNetAlphaModel, spec=spec)
    if model_type in {"rank_composite_momentum", "rank_composite"}:
        return partial(
            RankCompositeAlphaModel,
            min_cross_section_size=int(params.get("min_cross_section_size", config.get("min_cross_section_size", 2))),
            normalize=bool(params.get("normalize", True)),
            use_ic_weights=bool(params.get("use_ic_weights", True)),
        )
    raise ValueError(f"Alpha '{alpha_name}' uses unsupported model_type '{config['model_type']}'.")


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
