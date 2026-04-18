from __future__ import annotations

from typing import Any, Mapping

from src.research.position_constructors.base import (
    PositionConstructor,
    PositionConstructorError,
    finalize_positions,
    signal_values,
)
from src.research.signal_semantics import Signal


class IdentityWeightsPositionConstructor(PositionConstructor):
    """Consume the signal values literally as target positions."""

    def __init__(self, *, params: Mapping[str, Any] | None = None, version: str = "1.0.0") -> None:
        normalized_params = {} if params is None else dict(params)
        if normalized_params:
            raise PositionConstructorError("identity_weights does not accept any parameters.")
        super().__init__(constructor_id="identity_weights", params=normalized_params, version=version)

    def construct(self, signal: Signal):
        return finalize_positions(signal, signal_values(signal))


__all__ = ["IdentityWeightsPositionConstructor"]
