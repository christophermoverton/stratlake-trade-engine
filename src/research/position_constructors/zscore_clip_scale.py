from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.research.position_constructors.base import (
    PositionConstructor,
    finalize_positions,
    require_float,
    signal_values,
)
from src.research.signal_semantics import Signal


class ZScoreClipScalePositionConstructor(PositionConstructor):
    """Clip signed scores and scale the resulting gross absolute exposure."""

    def __init__(self, *, params: Mapping[str, Any] | None = None, version: str = "1.0.0") -> None:
        normalized_params = {} if params is None else dict(params)
        self._clip = require_float(normalized_params, name="clip", positive=True)
        self._gross_exposure = require_float(normalized_params, name="gross_exposure", non_negative=True)
        super().__init__(constructor_id="zscore_clip_scale", params=normalized_params, version=version)

    def construct(self, signal: Signal) -> pd.DataFrame:
        values = signal_values(signal)
        positions = pd.Series(0.0, index=signal.data.index, dtype="float64")
        for _, group in signal.data.groupby("ts_utc", sort=False):
            group_values = values.loc[group.index].clip(lower=-self._clip, upper=self._clip)
            denominator = float(group_values.abs().sum())
            if denominator == 0.0:
                continue
            positions.loc[group.index] = (group_values / denominator) * self._gross_exposure
        return finalize_positions(signal, positions)


__all__ = ["ZScoreClipScalePositionConstructor"]
