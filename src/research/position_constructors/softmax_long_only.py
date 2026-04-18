from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.research.position_constructors.base import (
    PositionConstructor,
    apply_directional_position_controls,
    finalize_positions,
    require_float,
    signal_values,
    validate_asymmetry_parameters,
)
from src.research.signal_semantics import Signal


class SoftmaxLongOnlyPositionConstructor(PositionConstructor):
    """Convert cross-sectional scores into probabilistic long-only weights."""

    def __init__(self, *, params: Mapping[str, Any] | None = None, version: str = "1.0.0") -> None:
        normalized_params = {} if params is None else dict(params)
        self._gross_exposure = require_float(normalized_params, name="gross_exposure", non_negative=True)
        self._temperature = require_float(normalized_params, name="temperature", positive=True)
        validation = validate_asymmetry_parameters(normalized_params, "softmax_long_only")
        if not validation["valid"]:
            raise ValueError(validation["error_message"])
        super().__init__(constructor_id="softmax_long_only", params=normalized_params, version=version)

    def construct(self, signal: Signal) -> pd.DataFrame:
        values = signal_values(signal)
        positions = pd.Series(0.0, index=signal.data.index, dtype="float64")
        for _, group in signal.data.groupby("ts_utc", sort=False):
            group_values = values.loc[group.index]
            logits = (group_values / self._temperature).astype("float64")
            max_logit = float(logits.max())
            weights = np.exp(logits - max_logit)
            denominator = float(weights.sum())
            if denominator == 0.0:
                continue
            positions.loc[group.index] = (weights / denominator) * self._gross_exposure
        positions, diagnostics = apply_directional_position_controls(
            signal,
            positions,
            self.params,
            constructor_id=self.constructor_id,
        )
        frame = finalize_positions(signal, positions)
        if diagnostics["enabled"]:
            frame.attrs["directional_controls"] = diagnostics
        return frame


__all__ = ["SoftmaxLongOnlyPositionConstructor"]
