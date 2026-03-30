from __future__ import annotations

from typing import Type

from src.research.alpha.base import BaseAlphaModel, DummyAlphaModel

_ALPHA_MODEL_REGISTRY: dict[str, Type[BaseAlphaModel]] = {}


def register_alpha_model(name: str, model_cls: Type[BaseAlphaModel]) -> None:
    """Register one alpha model class under a deterministic string name."""

    normalized_name = _normalize_name(name)
    if normalized_name in _ALPHA_MODEL_REGISTRY:
        raise ValueError(f"Alpha model '{normalized_name}' is already registered.")
    if not issubclass(model_cls, BaseAlphaModel):
        raise TypeError("model_cls must inherit from BaseAlphaModel.")
    _ALPHA_MODEL_REGISTRY[normalized_name] = model_cls


def get_alpha_model(name: str) -> BaseAlphaModel:
    """Instantiate a registered alpha model by name."""

    normalized_name = _normalize_name(name)
    try:
        model_cls = _ALPHA_MODEL_REGISTRY[normalized_name]
    except KeyError as exc:
        raise ValueError(f"No alpha model implementation is registered for '{normalized_name}'.") from exc
    return model_cls()


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Alpha model name must be a string.")
    normalized = name.strip()
    if not normalized:
        raise ValueError("Alpha model name must be a non-empty string.")
    return normalized


register_alpha_model(DummyAlphaModel.name, DummyAlphaModel)
