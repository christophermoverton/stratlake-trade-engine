from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import compute_portfolio_metrics, write_portfolio_artifacts


def _portfolio_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "weight__beta": [0.5, 0.5],
            "strategy_return__beta": [0.03, 0.02],
            "gross_portfolio_return": [0.02, 0.01],
            "portfolio_weight_change": [1.0, 0.0],
            "portfolio_abs_weight_change": [1.0, 0.0],
            "portfolio_turnover": [1.0, 0.0],
            "portfolio_rebalance_event": [1, 0],
            "portfolio_transaction_cost": [0.0, 0.0],
            "portfolio_slippage_cost": [0.0, 0.0],
            "portfolio_execution_friction": [0.0, 0.0],
            "net_portfolio_return": [0.02, 0.01],
            "portfolio_return": [0.02, 0.01],
            "ts_utc": pd.Series(
                [
                    pd.Timestamp("2025-01-02 00:00:00+00:00"),
                    pd.Timestamp("2025-01-01 00:00:00+00:00"),
                ],
                dtype="datetime64[ns, UTC]",
            ),
            "weight__alpha": [0.5, 0.5],
            "strategy_return__alpha": [0.01, 0.00],
            "portfolio_equity_curve": [103.02, 101.0],
        }
    )


def _config() -> dict[str, object]:
    return {
        "portfolio_name": "Core Portfolio",
        "allocator": "equal_weight",
        "initial_capital": 100.0,
        "alignment_policy": "intersection",
        "timeframe": "1D",
        "settings": {
            "rebalance": "daily",
            "long_only": True,
        },
        "execution": {
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 10.0,
            "slippage_bps": 5.0,
        },
    }


def _components() -> list[dict[str, object]]:
    return [
        {
            "strategy_name": "beta",
            "run_id": "run-b",
            "source_artifact_path": Path("artifacts/strategies/run-b"),
        },
        {
            "strategy_name": "alpha",
            "run_id": "run-a",
            "source_artifact_path": Path("artifacts/strategies/run-a"),
        },
    ]


def _metrics() -> dict[str, object]:
    return compute_portfolio_metrics(_portfolio_output().sort_values("ts_utc").reset_index(drop=True), "1D")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_write_portfolio_artifacts_creates_expected_files_and_schemas(tmp_path: Path) -> None:
    output_dir = tmp_path / "portfolio_equal_weight_abcd1234"

    manifest = write_portfolio_artifacts(
        output_dir=output_dir,
        portfolio_output=_portfolio_output(),
        metrics=_metrics(),
        config=_config(),
        components=_components(),
    )

    expected_files = {
        "components.json",
        "config.json",
        "manifest.json",
        "metrics.json",
        "qa_summary.json",
        "portfolio_equity_curve.csv",
        "portfolio_returns.csv",
        "weights.csv",
    }
    assert {path.name for path in output_dir.iterdir() if path.is_file()} == expected_files

    config_payload = _load_json(output_dir / "config.json")
    assert config_payload == {
        "alignment_policy": "intersection",
        "allocator": "equal_weight",
        "execution": {
            "enabled": True,
            "execution_delay": 1,
            "slippage_bps": 5.0,
            "transaction_cost_bps": 10.0,
        },
        "initial_capital": 100.0,
        "portfolio_name": "Core Portfolio",
        "settings": {"long_only": True, "rebalance": "daily"},
        "timeframe": "1D",
    }

    components_payload = _load_json(output_dir / "components.json")
    assert components_payload == {
        "components": [
            {
                "run_id": "run-a",
                "source_artifact_path": "artifacts/strategies/run-a",
                "strategy_name": "alpha",
            },
            {
                "run_id": "run-b",
                "source_artifact_path": "artifacts/strategies/run-b",
                "strategy_name": "beta",
            },
        ]
    }

    weights_frame = pd.read_csv(output_dir / "weights.csv")
    assert weights_frame.columns.tolist() == ["ts_utc", "weight__alpha", "weight__beta"]
    assert weights_frame["ts_utc"].tolist() == [
        "2025-01-01T00:00:00Z",
        "2025-01-02T00:00:00Z",
    ]

    returns_frame = pd.read_csv(output_dir / "portfolio_returns.csv")
    assert returns_frame.columns.tolist() == [
        "ts_utc",
        "strategy_return__alpha",
        "strategy_return__beta",
        "weight__alpha",
        "weight__beta",
        "gross_portfolio_return",
        "portfolio_weight_change",
        "portfolio_abs_weight_change",
        "portfolio_turnover",
        "portfolio_rebalance_event",
        "portfolio_transaction_cost",
        "portfolio_slippage_cost",
        "portfolio_execution_friction",
        "net_portfolio_return",
        "portfolio_return",
    ]
    assert len(returns_frame) == 2

    equity_frame = pd.read_csv(output_dir / "portfolio_equity_curve.csv")
    assert equity_frame.columns.tolist() == ["ts_utc", "portfolio_equity_curve"]
    assert len(equity_frame) == 2

    metrics_payload = _load_json(output_dir / "metrics.json")
    assert metrics_payload["total_return"] == pytest.approx(0.0302)
    assert metrics_payload["sharpe_ratio"] == pytest.approx(_metrics()["sharpe_ratio"])

    qa_summary = _load_json(output_dir / "qa_summary.json")
    assert qa_summary["validation_status"] == "pass"
    assert qa_summary["strategy_count"] == 2

    manifest_payload = _load_json(output_dir / "manifest.json")
    assert manifest == manifest_payload
    assert manifest_payload["artifact_files"] == sorted(expected_files)
    assert manifest_payload["component_count"] == 2
    assert manifest_payload["qa_summary_status"] == "pass"
    assert manifest_payload["row_counts"] == {
        "components": 2,
        "portfolio_equity_curve": 2,
        "portfolio_returns": 2,
        "weights": 2,
    }
    assert manifest_payload["artifacts"]["portfolio_returns.csv"]["columns"] == [
        "ts_utc",
        "strategy_return__alpha",
        "strategy_return__beta",
        "weight__alpha",
        "weight__beta",
        "gross_portfolio_return",
        "portfolio_weight_change",
        "portfolio_abs_weight_change",
        "portfolio_turnover",
        "portfolio_rebalance_event",
        "portfolio_transaction_cost",
        "portfolio_slippage_cost",
        "portfolio_execution_friction",
        "net_portfolio_return",
        "portfolio_return",
    ]


def test_write_portfolio_artifacts_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    first_dir = tmp_path / "first" / "portfolio_equal_weight_abcd1234"
    second_dir = tmp_path / "second" / "portfolio_equal_weight_abcd1234"

    write_portfolio_artifacts(first_dir, _portfolio_output(), _metrics(), _config(), _components())
    write_portfolio_artifacts(second_dir, _portfolio_output(), _metrics(), _config(), _components())

    for filename in (
        "components.json",
        "config.json",
        "manifest.json",
        "metrics.json",
        "qa_summary.json",
        "portfolio_equity_curve.csv",
        "portfolio_returns.csv",
        "weights.csv",
    ):
        assert (first_dir / filename).read_text(encoding="utf-8") == (
            second_dir / filename
        ).read_text(encoding="utf-8")

    pdt.assert_frame_equal(
        pd.read_csv(first_dir / "portfolio_returns.csv"),
        pd.read_csv(second_dir / "portfolio_returns.csv"),
    )


def test_write_portfolio_artifacts_rejects_missing_equity_curve(tmp_path: Path) -> None:
    output = _portfolio_output().drop(columns=["portfolio_equity_curve"])

    with pytest.raises(ValueError, match="portfolio_equity_curve"):
        write_portfolio_artifacts(
            tmp_path / "portfolio_equal_weight_abcd1234",
            output,
            _metrics(),
            _config(),
            _components(),
        )


def test_write_portfolio_artifacts_rejects_non_serializable_metric_values(tmp_path: Path) -> None:
    metrics = _metrics()
    metrics["sharpe_ratio"] = float("nan")

    with pytest.raises(ValueError, match="must not contain NaN or infinite floats"):
        write_portfolio_artifacts(
            tmp_path / "portfolio_equal_weight_abcd1234",
            _portfolio_output(),
            metrics,
            _config(),
            _components(),
        )


def test_write_portfolio_artifacts_rejects_duplicate_components(tmp_path: Path) -> None:
    components = [
        {"strategy_name": "alpha", "run_id": "run-a"},
        {"strategy_name": "alpha", "run_id": "run-a"},
    ]

    with pytest.raises(ValueError, match="unique by \\(strategy_name, run_id\\)"):
        write_portfolio_artifacts(
            tmp_path / "portfolio_equal_weight_abcd1234",
            _portfolio_output(),
            _metrics(),
            _config(),
            components,
        )


def test_write_portfolio_artifacts_does_not_persist_invalid_portfolio(tmp_path: Path) -> None:
    output_dir = tmp_path / "portfolio_invalid_abcd1234"
    invalid_output = _portfolio_output().sort_values("ts_utc").reset_index(drop=True)
    invalid_output.loc[1, "weight__alpha"] = 0.9
    invalid_output.loc[1, "weight__beta"] = 0.1
    recomputed_return = (
        invalid_output.loc[1, "strategy_return__alpha"] * invalid_output.loc[1, "weight__alpha"]
        + invalid_output.loc[1, "strategy_return__beta"] * invalid_output.loc[1, "weight__beta"]
    )
    invalid_output.loc[1, "gross_portfolio_return"] = recomputed_return
    invalid_output.loc[1, "net_portfolio_return"] = recomputed_return
    invalid_output.loc[1, "portfolio_return"] = recomputed_return
    invalid_output.loc[1, "portfolio_equity_curve"] = invalid_output.loc[0, "portfolio_equity_curve"] * (
        1.0 + recomputed_return
    )

    with pytest.raises(ValueError, match="equal_weight allocator should produce constant weights"):
        write_portfolio_artifacts(
            output_dir,
            invalid_output,
            _metrics(),
            _config(),
            _components(),
        )

    assert not output_dir.exists()


def test_write_portfolio_artifacts_blocks_write_on_metrics_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "portfolio_invalid_metrics_abcd1234"
    metrics = _metrics()
    metrics["total_return"] = 99.0

    with pytest.raises(ValueError, match="portfolio.metrics.total_return mismatch"):
        write_portfolio_artifacts(
            output_dir,
            _portfolio_output(),
            metrics,
            _config(),
            _components(),
        )

    assert not output_dir.exists()
