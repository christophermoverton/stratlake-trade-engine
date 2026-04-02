from __future__ import annotations

from collections.abc import Callable
from typing import Type

from src.research.alpha.base import BaseAlphaModel, DummyAlphaModel

AlphaModelFactory = Callable[[], BaseAlphaModel]

_ALPHA_MODEL_REGISTRY: dict[str, AlphaModelFactory] = {}


def register_alpha_model(name: str, model_cls: Type[BaseAlphaModel]) -> None:
    """Register one alpha model class under a deterministic string name."""

    if not issubclass(model_cls, BaseAlphaModel):
        raise TypeError("model_cls must inherit from BaseAlphaModel.")
    register_alpha_factory(name, model_cls)


def register_alpha_factory(name: str, factory: AlphaModelFactory) -> None:
    """Register one alpha model factory under a deterministic string name."""

    normalized_name = _normalize_name(name)
    if normalized_name in _ALPHA_MODEL_REGISTRY:
        raise ValueError(f"Alpha model '{normalized_name}' is already registered.")
    if not callable(factory):
        raise TypeError("factory must be callable.")
    _ALPHA_MODEL_REGISTRY[normalized_name] = factory


def get_alpha_model(name: str) -> BaseAlphaModel:
    """Instantiate a registered alpha model by name."""

    normalized_name = _normalize_name(name)
    try:
        factory = _ALPHA_MODEL_REGISTRY[normalized_name]
    except KeyError as exc:
        raise ValueError(f"No alpha model implementation is registered for '{normalized_name}'.") from exc
    model = factory()
    if not isinstance(model, BaseAlphaModel):
        raise TypeError(f"Alpha model factory '{normalized_name}' must return a BaseAlphaModel instance.")
    return model


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Alpha model name must be a string.")
    normalized = name.strip()
    if not normalized:
        raise ValueError("Alpha model name must be a non-empty string.")
    return normalized


register_alpha_model(DummyAlphaModel.name, DummyAlphaModel)
