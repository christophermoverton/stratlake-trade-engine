from __future__ import annotations

from typing import Any, Mapping

from src.research.position_constructors.base import (
    PositionConstructor,
    apply_directional_position_controls,
    PositionConstructorError,
    finalize_positions,
    signal_values,
    validate_asymmetry_parameters,
)
from src.research.signal_semantics import Signal


class IdentityWeightsPositionConstructor(PositionConstructor):
    """Consume the signal values literally as target positions."""

    def __init__(self, *, params: Mapping[str, Any] | None = None, version: str = "1.0.0") -> None:
        normalized_params = {} if params is None else dict(params)
        validation = validate_asymmetry_parameters(normalized_params, "identity_weights")
        if not validation["valid"]:
            raise PositionConstructorError(validation["error_message"])
        if set(normalized_params) - set(validation["asymmetry_params"]):
            raise PositionConstructorError("identity_weights only accepts directional asymmetry parameters.")
        super().__init__(constructor_id="identity_weights", params=normalized_params, version=version)

    def construct(self, signal: Signal):
        positions, diagnostics = apply_directional_position_controls(
            signal,
            signal_values(signal),
            self.params,
            constructor_id=self.constructor_id,
        )
        frame = finalize_positions(signal, positions)
        if diagnostics["enabled"]:
            frame.attrs["directional_controls"] = diagnostics
        return frame


__all__ = ["IdentityWeightsPositionConstructor"]
