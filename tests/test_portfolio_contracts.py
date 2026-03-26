from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio.contracts import (
    PortfolioContractError,
    validate_aligned_returns,
    validate_portfolio_config,
    validate_portfolio_output,
    validate_strategy_returns,
    validate_weights,
)


def _strategy_returns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.Series(
                [
                    pd.Timestamp("2025-01-02 00:00:00+00:00"),
                    pd.Timestamp("2025-01-01 00:00:00+00:00"),
                    pd.Timestamp("2025-01-01 00:00:00+00:00"),
                ],
                dtype="datetime64[ns, UTC]",
            ),
            "strategy_name": pd.Series(["beta", "beta", "alpha"], dtype="string"),
            "strategy_return": pd.Series([0.03, 0.02, 0.01], dtype="float64"),
        }
    )


def _aligned_returns_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-02 00:00:00+00:00"),
            pd.Timestamp("2025-01-01 00:00:00+00:00"),
        ],
        name="ts_utc",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "beta": [0.03, 0.02],
            "alpha": [0.01, 0.00],
        },
        index=index,
        dtype="float64",
    )


def _weights_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-02 00:00:00+00:00"),
            pd.Timestamp("2025-01-01 00:00:00+00:00"),
        ],
        name="ts_utc",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "beta": [0.4, 0.5],
            "alpha": [0.6, 0.5],
        },
        index=index,
        dtype="float64",
    )


def _portfolio_output_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "weight__beta": [0.4, 0.5],
            "strategy_return__beta": [0.03, 0.02],
            "portfolio_return": [0.018, 0.01],
            "ts_utc": pd.Series(
                [
                    pd.Timestamp("2025-01-02 00:00:00+00:00"),
                    pd.Timestamp("2025-01-01 00:00:00+00:00"),
                ],
                dtype="datetime64[ns, UTC]",
            ),
            "weight__alpha": [0.6, 0.5],
            "strategy_return__alpha": [0.01, 0.00],
            "portfolio_equity_curve": [1.018, 1.0],
        }
    )


def _portfolio_config() -> dict[str, object]:
    return {
        "portfolio_name": " Core Portfolio ",
        "allocator": " equal_weight ",
        "components": [
            {"strategy_name": "beta", "run_id": "run-b"},
            {"strategy_name": "alpha", "run_id": "run-a"},
        ],
    }


def test_validate_strategy_returns_accepts_and_sorts_valid_input() -> None:
    normalized = validate_strategy_returns(_strategy_returns_frame())

    assert normalized["strategy_name"].tolist() == ["alpha", "beta", "beta"]
    assert normalized["ts_utc"].tolist() == [
        pd.Timestamp("2025-01-01 00:00:00+00:00"),
        pd.Timestamp("2025-01-01 00:00:00+00:00"),
        pd.Timestamp("2025-01-02 00:00:00+00:00"),
    ]
    assert normalized.attrs["portfolio_contract"]["strategy_identifier_column"] == "strategy_name"


def test_validate_strategy_returns_rejects_missing_columns() -> None:
    df = _strategy_returns_frame().drop(columns=["strategy_return"])

    with pytest.raises(PortfolioContractError, match="missing required columns"):
        validate_strategy_returns(df)


def test_validate_strategy_returns_rejects_tz_naive_timestamps() -> None:
    df = _strategy_returns_frame()
    df["ts_utc"] = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])

    with pytest.raises(PortfolioContractError, match="timezone-aware UTC"):
        validate_strategy_returns(df)


def test_validate_strategy_returns_rejects_duplicate_keys() -> None:
    df = pd.concat(
        [
            _strategy_returns_frame(),
            pd.DataFrame(
                {
                    "ts_utc": pd.Series(
                        [pd.Timestamp("2025-01-01 00:00:00+00:00")],
                        dtype="datetime64[ns, UTC]",
                    ),
                    "strategy_name": pd.Series(["alpha"], dtype="string"),
                    "strategy_return": pd.Series([0.9], dtype="float64"),
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(PortfolioContractError, match="duplicate \\(ts_utc, strategy\\) rows"):
        validate_strategy_returns(df)


def test_validate_aligned_returns_accepts_and_sorts_matrix() -> None:
    normalized = validate_aligned_returns(_aligned_returns_frame())

    assert list(normalized.columns) == ["alpha", "beta"]
    assert list(normalized.index) == [
        pd.Timestamp("2025-01-01 00:00:00+00:00"),
        pd.Timestamp("2025-01-02 00:00:00+00:00"),
    ]


def test_validate_aligned_returns_rejects_duplicate_timestamps() -> None:
    ts = pd.Timestamp("2025-01-01 00:00:00+00:00")
    df = pd.DataFrame(
        {"alpha": [0.0, 0.1], "beta": [0.2, 0.3]},
        index=pd.DatetimeIndex([ts, ts], name="ts_utc", tz="UTC"),
    )

    with pytest.raises(PortfolioContractError, match="duplicate timestamps"):
        validate_aligned_returns(df)


def test_validate_aligned_returns_rejects_non_utc_index() -> None:
    df = _aligned_returns_frame()
    df.index = df.index.tz_convert("US/Eastern")

    with pytest.raises(PortfolioContractError, match="must use timezone UTC"):
        validate_aligned_returns(df)


def test_validate_weights_accepts_and_sorts_valid_input() -> None:
    normalized = validate_weights(_weights_frame())

    assert list(normalized.columns) == ["alpha", "beta"]
    assert list(normalized.index) == [
        pd.Timestamp("2025-01-01 00:00:00+00:00"),
        pd.Timestamp("2025-01-02 00:00:00+00:00"),
    ]


def test_validate_weights_rejects_rows_not_summing_to_one() -> None:
    df = _weights_frame()
    df.iloc[0, 0] = 0.9

    with pytest.raises(PortfolioContractError, match="must sum to 1.0"):
        validate_weights(df)


def test_validate_weights_rejects_nan_values() -> None:
    df = _weights_frame()
    df.iloc[0, 0] = float("nan")

    with pytest.raises(PortfolioContractError, match="contains NaN values"):
        validate_weights(df)


def test_validate_portfolio_output_accepts_and_sorts_traceability_columns() -> None:
    normalized = validate_portfolio_output(_portfolio_output_frame())

    assert list(normalized.columns) == [
        "ts_utc",
        "strategy_return__alpha",
        "strategy_return__beta",
        "weight__alpha",
        "weight__beta",
        "portfolio_return",
        "portfolio_equity_curve",
    ]
    assert normalized["ts_utc"].tolist() == [
        pd.Timestamp("2025-01-01 00:00:00+00:00"),
        pd.Timestamp("2025-01-02 00:00:00+00:00"),
    ]


def test_validate_portfolio_output_rejects_missing_required_columns() -> None:
    df = _portfolio_output_frame().drop(columns=["portfolio_return"])

    with pytest.raises(PortfolioContractError, match="missing required columns"):
        validate_portfolio_output(df)


def test_validate_portfolio_output_rejects_inconsistent_traceability_pairs() -> None:
    df = _portfolio_output_frame().drop(columns=["weight__alpha"])

    with pytest.raises(PortfolioContractError, match="traceability columns are inconsistent"):
        validate_portfolio_output(df)


def test_validate_portfolio_config_accepts_and_normalizes_defaults() -> None:
    normalized = validate_portfolio_config(_portfolio_config())

    assert normalized == {
        "portfolio_name": "Core Portfolio",
        "allocator": "equal_weight",
        "components": [
            {"strategy_name": "alpha", "run_id": "run-a"},
            {"strategy_name": "beta", "run_id": "run-b"},
        ],
        "initial_capital": 1.0,
        "alignment_policy": "intersection",
        "validation": {
            "target_weight_sum": 1.0,
            "weight_sum_tolerance": 1e-08,
            "target_net_exposure": 1.0,
            "net_exposure_tolerance": 1e-08,
            "max_gross_exposure": 1.0,
            "max_leverage": 1.0,
            "max_single_sleeve_weight": None,
            "min_single_sleeve_weight": None,
            "max_abs_period_return": 1.0,
            "max_equity_multiple": 1000000.0,
            "strict_sanity_checks": True,
        },
    }


def test_validate_portfolio_config_rejects_empty_components() -> None:
    config = _portfolio_config()
    config["components"] = []

    with pytest.raises(PortfolioContractError, match="must be a non-empty list"):
        validate_portfolio_config(config)


def test_validate_portfolio_config_rejects_duplicate_components() -> None:
    config = _portfolio_config()
    config["components"] = [
        {"strategy_name": "alpha", "run_id": "run-a"},
        {"strategy_name": "alpha", "run_id": "run-a"},
    ]

    with pytest.raises(PortfolioContractError, match="must be unique by \\(strategy_name, run_id\\)"):
        validate_portfolio_config(config)


def test_validate_portfolio_config_rejects_duplicate_strategy_names() -> None:
    config = _portfolio_config()
    config["components"] = [
        {"strategy_name": "alpha", "run_id": "run-a"},
        {"strategy_name": "alpha", "run_id": "run-b"},
    ]

    with pytest.raises(PortfolioContractError, match="must be unique by strategy_name"):
        validate_portfolio_config(config)


def test_validate_portfolio_config_rejects_non_string_allocator() -> None:
    config = _portfolio_config()
    config["allocator"] = 123

    with pytest.raises(PortfolioContractError, match="allocator"):
        validate_portfolio_config(config)
