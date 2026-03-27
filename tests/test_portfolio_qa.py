from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import (
    PortfolioQAError,
    compute_portfolio_metrics,
    generate_portfolio_qa_summary,
    validate_equity_curve,
    validate_portfolio_artifact_consistency,
    validate_portfolio_return_consistency,
    validate_weights_behavior,
    write_portfolio_artifacts,
)


def _portfolio_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "strategy_return__alpha": [0.01, 0.02, -0.01],
            "strategy_return__beta": [0.03, -0.01, 0.00],
            "weight__alpha": [0.5, 0.5, 0.5],
            "weight__beta": [0.5, 0.5, 0.5],
            "portfolio_weight_change": [1.0, 0.0, 0.0],
            "portfolio_abs_weight_change": [1.0, 0.0, 0.0],
            "portfolio_turnover": [1.0, 0.0, 0.0],
            "portfolio_rebalance_event": [1, 0, 0],
            "portfolio_return": [0.02, 0.005, -0.005],
            "portfolio_equity_curve": [102.0, 102.51, 101.99745],
        }
    )


def _metrics() -> dict[str, float | None]:
    return compute_portfolio_metrics(_portfolio_output(), "1D")


def _config() -> dict[str, object]:
    return {
        "portfolio_name": "core_portfolio",
        "allocator": "equal_weight",
        "initial_capital": 100.0,
        "alignment_policy": "intersection",
        "timeframe": "1D",
    }


def _components() -> list[dict[str, object]]:
    return [
        {"strategy_name": "alpha", "run_id": "run-alpha"},
        {"strategy_name": "beta", "run_id": "run-beta"},
    ]


def test_portfolio_qa_accepts_valid_output() -> None:
    portfolio_output = _portfolio_output()

    validated_returns = validate_portfolio_return_consistency(portfolio_output)
    validated_equity = validate_equity_curve(portfolio_output, initial_capital=100.0)
    validated_weights = validate_weights_behavior(portfolio_output, allocator_name="equal_weight")
    summary = generate_portfolio_qa_summary(
        portfolio_output,
        _metrics(),
        portfolio_name="core_portfolio",
        allocator_name="equal_weight",
        timeframe="1D",
        run_id="portfolio_run_123",
    )

    assert len(validated_returns) == 3
    assert len(validated_equity) == 3
    assert len(validated_weights) == 3
    assert summary["validation_status"] == "pass"
    assert summary["issues"] == []
    assert summary["row_count"] == 3
    assert summary["strategy_count"] == 2
    assert summary["metrics"]["trade_count"] == pytest.approx(1.0)


def test_portfolio_qa_fails_for_incorrect_weighted_return() -> None:
    portfolio_output = _portfolio_output()
    portfolio_output.loc[1, "portfolio_return"] = 0.5

    with pytest.raises(PortfolioQAError, match="weighted sum of component returns"):
        validate_portfolio_return_consistency(portfolio_output)


def test_portfolio_qa_fails_for_incorrect_equity_curve() -> None:
    portfolio_output = _portfolio_output()
    portfolio_output.loc[2, "portfolio_equity_curve"] = 999.0

    with pytest.raises(PortfolioQAError, match="does not match compounded portfolio_return"):
        validate_equity_curve(portfolio_output, initial_capital=100.0)


def test_portfolio_qa_fails_for_equal_weight_drift() -> None:
    portfolio_output = _portfolio_output()
    portfolio_output.loc[2, "weight__alpha"] = 0.6
    portfolio_output.loc[2, "weight__beta"] = 0.4

    with pytest.raises(PortfolioQAError, match="equal_weight allocator should produce constant weights"):
        validate_weights_behavior(portfolio_output, allocator_name="equal_weight")


def test_portfolio_qa_fails_for_corrupted_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "portfolio_run_123"
    write_portfolio_artifacts(
        output_dir=output_dir,
        portfolio_output=_portfolio_output(),
        metrics=_metrics(),
        config=_config(),
        components=_components(),
    )

    weights = pd.read_csv(output_dir / "weights.csv")
    weights.loc[0, "weight__alpha"] = 0.75
    weights.to_csv(output_dir / "weights.csv", index=False)

    with pytest.raises(PortfolioQAError, match="weights.csv column 'weight__alpha' differs"):
        validate_portfolio_artifact_consistency(
            output_dir,
            portfolio_output=_portfolio_output(),
            metrics=_metrics(),
            config=_config(),
        )


def test_write_portfolio_artifacts_writes_qa_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "portfolio_run_123"

    write_portfolio_artifacts(
        output_dir=output_dir,
        portfolio_output=_portfolio_output(),
        metrics=_metrics(),
        config=_config(),
        components=_components(),
    )

    qa_summary = json.loads((output_dir / "qa_summary.json").read_text(encoding="utf-8"))
    assert qa_summary["validation_status"] == "pass"
    assert qa_summary["portfolio_name"] == "core_portfolio"
    assert qa_summary["allocator"] == "equal_weight"
