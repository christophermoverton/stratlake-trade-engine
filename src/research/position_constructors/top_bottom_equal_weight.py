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


class TopBottomEqualWeightPositionConstructor(PositionConstructor):
    """Allocate equal long and short weights from ternary bucketed signals."""

    def __init__(self, *, params: Mapping[str, Any] | None = None, version: str = "1.0.0") -> None:
        normalized_params = {} if params is None else dict(params)
        self._gross_long = require_float(normalized_params, name="gross_long", non_negative=True)
        self._gross_short = require_float(normalized_params, name="gross_short", non_negative=True)
        super().__init__(constructor_id="top_bottom_equal_weight", params=normalized_params, version=version)

    def construct(self, signal: Signal) -> pd.DataFrame:
        values = signal_values(signal)
        positions = pd.Series(0.0, index=signal.data.index, dtype="float64")
        for _, group in signal.data.groupby("ts_utc", sort=False):
            group_values = values.loc[group.index]
            long_index = group_values.loc[group_values > 0.0].index
            short_index = group_values.loc[group_values < 0.0].index
            if len(long_index) > 0:
                positions.loc[long_index] = self._gross_long / float(len(long_index))
            if len(short_index) > 0:
                positions.loc[short_index] = -(self._gross_short / float(len(short_index)))
        return finalize_positions(signal, positions)


__all__ = ["TopBottomEqualWeightPositionConstructor"]
