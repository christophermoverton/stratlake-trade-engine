from __future__ import annotations

import pandas as pd
import pytest

from src.research.strategy_base import BaseStrategy


class DummyStrategy(BaseStrategy):
    name = "dummy_strategy"
    dataset = "features_daily"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index, name="signal", dtype="int64")


class IncompleteStrategy(BaseStrategy):
    name = "incomplete_strategy"
    dataset = "features_daily"


def test_base_strategy_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseStrategy()


def test_dummy_strategy_subclass_can_be_instantiated() -> None:
    strategy = DummyStrategy()

    assert strategy.name == "dummy_strategy"
    assert strategy.dataset == "features_daily"


def test_subclass_must_implement_generate_signals() -> None:
    with pytest.raises(TypeError):
        IncompleteStrategy()


def test_generate_signals_returns_series_aligned_with_input_index() -> None:
    df = pd.DataFrame(
        {"feature_alpha": [0.1, -0.2, 0.3]},
        index=pd.Index(["row_a", "row_b", "row_c"], name="row_id"),
    )

    signals = DummyStrategy().generate_signals(df)

    assert isinstance(signals, pd.Series)
    assert signals.index.equals(df.index)
    assert signals.tolist() == [1, 1, 1]
