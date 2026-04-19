from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.research.alpha.registry as alpha_registry
from src.config.execution import ExecutionConfig
from src.research.alpha import predict_alpha_model, register_alpha_model, train_alpha_model
from src.research.alpha.base import BaseAlphaModel
from src.research.backtest_runner import run_backtest
from src.research.signal_semantics import attach_signal_metadata


def _backtest_frame() -> pd.DataFrame:
    index = pd.Index(["row_a", "row_b", "row_c", "row_d"], name="row_id")
    ts_utc = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL", "AAPL", "AAPL"], index=index, dtype="string"),
            "ts_utc": pd.Series(ts_utc, index=index),
            "signal": [1, 1, -1, 0],
            "feature_ret_1d": [0.01, -0.02, 0.03, -0.01],
            "feature_alpha": [0.5, 0.6, 0.4, 0.7],
        },
        index=index,
    )


class WeightedFeatureAlphaModel(BaseAlphaModel):
    name = "backtest_runner_weighted_feature_alpha_model"

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.feature_means: dict[str, float] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = [
            column
            for column in df.columns
            if column not in {"symbol", "ts_utc", "target_ret_1d"}
        ]
        self.feature_means = {
            column: float(pd.to_numeric(df[column], errors="coerce").mean())
            for column in self.feature_columns
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index, dtype="float64")
        for idx, column in enumerate(self.feature_columns, start=1):
            centered = pd.to_numeric(df[column], errors="coerce").astype("float64") - self.feature_means[column]
            score = score + (centered * float(idx))
        return score.rename("prediction")


@pytest.fixture(autouse=True)
def reset_alpha_registry() -> None:
    original_registry = dict(alpha_registry._ALPHA_MODEL_REGISTRY)
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)
    yield
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)


def test_run_backtest_computes_strategy_returns_from_shifted_signals() -> None:
    df = _backtest_frame()

    result = run_backtest(df)

    assert result["executed_signal"].tolist() == [0.0, 1.0, 1.0, -1.0]
    assert result["strategy_return"].tolist() == [0.0, -0.02, 0.03, 0.01]


def test_run_backtest_computes_deterministic_equity_curve() -> None:
    df = _backtest_frame()

    result = run_backtest(df)

    expected = [1.0, 0.98, 1.0094, 1.019494]
    assert result["equity_curve"].tolist() == pytest.approx(expected)


def test_run_backtest_preserves_input_columns() -> None:
    df = _backtest_frame()

    result = run_backtest(df)

    assert result.index.equals(df.index)
    assert result["feature_alpha"].tolist() == df["feature_alpha"].tolist()


def test_run_backtest_applies_execution_costs_and_slippage_on_position_changes() -> None:
    df = _backtest_frame()
    config = ExecutionConfig(
        enabled=True,
        execution_delay=1,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    result = run_backtest(df, config)

    assert result["delta_position"].tolist() == [0.0, 1.0, 0.0, -2.0]
    assert result["abs_delta_position"].tolist() == [0.0, 1.0, 0.0, 2.0]
    assert result["turnover"].tolist() == [0.0, 1.0, 0.0, 2.0]
    assert result["trade_event"].tolist() == [False, True, False, True]
    assert result["transaction_cost"].tolist() == pytest.approx([0.0, 0.001, 0.0, 0.002])
    assert result["slippage_cost"].tolist() == pytest.approx([0.0, 0.0005, 0.0, 0.001])
    assert result["execution_friction"].tolist() == pytest.approx([0.0, 0.0015, 0.0, 0.003])
    assert result["gross_strategy_return"].tolist() == pytest.approx([0.0, -0.02, 0.03, 0.01])
    assert result["net_strategy_return"].tolist() == pytest.approx([0.0, -0.0215, 0.03, 0.007])
    assert result["strategy_return"].tolist() == pytest.approx(result["net_strategy_return"].tolist())


def test_run_backtest_turnover_tracks_entries_exits_and_flips_from_executed_positions() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "ts_utc": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
            "signal": [1, -1, -1, 0, 1],
            "feature_ret_1d": [0.0, 0.01, -0.02, 0.03, 0.04],
        }
    )

    result = run_backtest(df, ExecutionConfig(enabled=False, execution_delay=1, transaction_cost_bps=0.0, slippage_bps=0.0))

    assert result["executed_signal"].tolist() == [0.0, 1.0, -1.0, -1.0, 0.0]
    assert result["delta_position"].tolist() == [0.0, 1.0, -2.0, 0.0, 1.0]
    assert result["turnover"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]
    assert result["trade_event"].tolist() == [False, True, True, False, True]


def test_run_backtest_supports_longer_execution_delay_deterministically() -> None:
    df = _backtest_frame()
    config = ExecutionConfig(
        enabled=False,
        execution_delay=2,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    result = run_backtest(df, config)

    assert result["executed_signal"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert result["strategy_return"].tolist() == pytest.approx([0.0, 0.0, 0.03, -0.01])


def test_run_backtest_supports_continuous_positive_signals() -> None:
    df = _backtest_frame()
    df["signal"] = [0.5, 1.0, 0.25, 0.0]

    result = run_backtest(df)

    assert result["executed_signal"].tolist() == pytest.approx([0.0, 0.5, 1.0, 0.25])
    assert result["strategy_return"].tolist() == pytest.approx([0.0, -0.01, 0.03, -0.0025])
    assert result["equity_curve"].tolist() == pytest.approx([1.0, 0.99, 1.0197, 1.01715075])


def test_run_backtest_supports_continuous_negative_signals() -> None:
    df = _backtest_frame()
    df["signal"] = [-0.5, -1.0, -0.25, 0.0]

    result = run_backtest(df)

    assert result["executed_signal"].tolist() == pytest.approx([0.0, -0.5, -1.0, -0.25])
    assert result["strategy_return"].tolist() == pytest.approx([0.0, 0.01, -0.03, 0.0025])
    assert result["equity_curve"].tolist() == pytest.approx([1.0, 1.01, 0.9797, 0.98214925])


def test_run_backtest_supports_mixed_continuous_exposures() -> None:
    df = _backtest_frame()
    df["signal"] = [1.5, -0.5, 0.0, 0.25]

    result = run_backtest(df)

    assert result["executed_signal"].tolist() == pytest.approx([0.0, 1.5, -0.5, 0.0])
    assert result["strategy_return"].tolist() == pytest.approx([0.0, -0.03, -0.015, 0.0])
    assert result["equity_curve"].tolist() == pytest.approx([1.0, 0.97, 0.95545, 0.95545])


def test_run_backtest_raises_when_signal_column_is_missing() -> None:
    df = _backtest_frame().drop(columns=["signal"])

    with pytest.raises(ValueError, match="must include a 'signal' column"):
        run_backtest(df)


def test_run_backtest_raises_when_supported_return_column_is_missing() -> None:
    df = _backtest_frame().drop(columns=["feature_ret_1d"])

    with pytest.raises(ValueError, match="Run failed: missing returns"):
        run_backtest(df)


def test_run_backtest_raises_when_return_column_has_no_usable_values() -> None:
    df = _backtest_frame()
    df["feature_ret_1d"] = pd.Series([pd.NA] * len(df), dtype="Float64")

    with pytest.raises(ValueError, match="contains no usable values"):
        run_backtest(df)


def test_run_backtest_treats_partial_missing_returns_as_zero_for_warmup_rows() -> None:
    df = _backtest_frame()
    df.loc["row_a", "feature_ret_1d"] = pd.NA

    result = run_backtest(df)

    assert result["feature_ret_1d"].tolist() == pytest.approx([0.0, -0.02, 0.03, -0.01])
    assert result["strategy_return"].tolist() == pytest.approx([0.0, -0.02, 0.03, 0.01])
    assert result["equity_curve"].tolist() == pytest.approx([1.0, 0.98, 1.0094, 1.019494])


def test_run_backtest_rejects_same_bar_execution_if_positions_are_not_shifted() -> None:
    df = _backtest_frame()

    from src.research.integrity import validate_research_integrity

    with pytest.raises(ValueError, match="same_bar_execution"):
        validate_research_integrity(df, df["signal"], positions=df["signal"].astype("float64"))


def test_run_backtest_rejects_duplicate_signal_keys() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], utc=True),
            "signal": [1, 0],
            "feature_ret_1d": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        run_backtest(df)


def test_run_backtest_rejects_non_numeric_signal_values() -> None:
    df = _backtest_frame()
    df["signal"] = ["long", "flat", "short", "flat"]

    with pytest.raises(ValueError, match="finite numeric exposures"):
        run_backtest(df)


def test_run_backtest_rejects_nan_signal_values() -> None:
    df = _backtest_frame()
    df.loc["row_c", "signal"] = pd.NA

    with pytest.raises(ValueError, match="finite numeric exposures"):
        run_backtest(df)


def test_run_backtest_rejects_non_finite_signal_values() -> None:
    df = _backtest_frame()
    df["signal"] = df["signal"].astype("float64")
    df.loc["row_b", "signal"] = float("inf")

    with pytest.raises(ValueError, match="finite numeric exposures"):
        run_backtest(df)


def test_run_backtest_lags_signals_within_each_symbol_group() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "signal": [1.0, 0.5, -1.0, -0.5],
            "feature_ret_1d": [0.01, 0.02, 0.03, 0.04],
        }
    )

    result = run_backtest(df, ExecutionConfig(enabled=False, execution_delay=1, transaction_cost_bps=0.0, slippage_bps=0.0))

    assert result["executed_signal"].tolist() == pytest.approx([0.0, 1.0, 0.0, -1.0])
    assert result["delta_position"].tolist() == pytest.approx([0.0, 1.0, 0.0, -1.0])
    assert result["strategy_return"].tolist() == pytest.approx([0.0, 0.02, 0.0, -0.04])


def test_run_backtest_rejects_out_of_order_rows_before_execution() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "signal": [1.0, -1.0, 0.5, -0.5],
            "feature_ret_1d": [0.01, 0.02, 0.03, 0.04],
        }
    )

    with pytest.raises(ValueError, match="must be sorted by \\(symbol, ts_utc\\)"):
        run_backtest(df)


def test_run_backtest_preserves_legacy_discrete_behavior() -> None:
    df = _backtest_frame()
    config = ExecutionConfig(enabled=False, execution_delay=1, transaction_cost_bps=0.0, slippage_bps=0.0)

    result = run_backtest(df, config)

    assert result["executed_signal"].tolist() == pytest.approx([0.0, 1.0, 1.0, -1.0])
    assert result["strategy_return"].tolist() == pytest.approx([0.0, -0.02, 0.03, 0.01])
    assert result["equity_curve"].tolist() == pytest.approx([1.0, 0.98, 1.0094, 1.019494])


def test_run_backtest_rejects_unmanaged_legacy_signal_frames_when_managed_required() -> None:
    df = _backtest_frame()

    with pytest.raises(ValueError, match="Canonical workflows do not accept unmanaged legacy signal frames"):
        run_backtest(df, require_managed_signals=True)


def test_run_backtest_rejects_legacy_compatibility_metadata_when_managed_required() -> None:
    df = _backtest_frame()
    attach_signal_metadata(
        df,
        {
            "signal_type": "ternary_quantile",
            "version": "1.0.0",
            "value_column": "signal",
            "compatibility_mode": "legacy_inferred",
            "constructor_id": "backtest_numeric_exposure",
            "constructor_params": {},
            "parameters": {},
            "source": {"layer": "legacy_backtest_input"},
            "timestamp_normalization": "UTC",
            "transformation_history": [],
        },
    )

    with pytest.raises(ValueError, match="must declare managed typed signal metadata"):
        run_backtest(df, require_managed_signals=True)


def test_run_backtest_accepts_prediction_derived_continuous_signals() -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    research_frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "target_ret_1d": [0.01, 0.02, 0.03, 0.05, 0.04, 0.06],
            "feature_alpha": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "feature_beta": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            "feature_ret_1d": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03],
        },
        index=pd.Index([f"row_{idx}" for idx in range(6)], name="row_id"),
    )
    trained = train_alpha_model(
        research_frame,
        model_name=WeightedFeatureAlphaModel.name,
        target_column="target_ret_1d",
    )
    prediction_result = predict_alpha_model(trained, research_frame)
    backtest_input = research_frame.copy()
    backtest_input["signal"] = prediction_result.predictions["prediction_score"]

    result = run_backtest(backtest_input, ExecutionConfig(enabled=False, execution_delay=1, transaction_cost_bps=0.0, slippage_bps=0.0))

    expected_signal = prediction_result.predictions["prediction_score"].astype("float64")
    expected_executed_signal = (
        expected_signal.groupby(prediction_result.predictions["symbol"].astype("string"), sort=False, dropna=False)
        .shift(1)
        .fillna(0.0)
        .astype("float64")
    )
    expected_strategy_return = (expected_executed_signal * research_frame["feature_ret_1d"].astype("float64")).astype("float64")

    assert result["signal"].tolist() == pytest.approx(expected_signal.tolist())
    assert result["executed_signal"].tolist() == pytest.approx(expected_executed_signal.tolist())
    assert result["strategy_return"].tolist() == pytest.approx(expected_strategy_return.tolist())
