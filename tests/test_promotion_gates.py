from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import compute_portfolio_metrics, write_portfolio_artifacts
from src.research import experiment_tracker
from src.research.alpha_eval import evaluate_alpha_predictions, write_alpha_evaluation_artifacts
from src.research.experiment_tracker import save_experiment, save_walk_forward_experiment
from src.research.metrics import compute_performance_metrics
from src.research.promotion import evaluate_promotion_gates
from src.research.registry import load_registry


def test_evaluate_promotion_gates_handles_pass_fail_borderline_missing_and_split_statistics() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "gates": [
                {
                    "gate_id": "borderline_sharpe",
                    "source": "metrics",
                    "metric": "sharpe_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                },
                {
                    "gate_id": "drawdown_limit",
                    "source": "metrics",
                    "metric": "max_drawdown",
                    "comparator": "lte",
                    "threshold": 0.10,
                },
                {
                    "gate_id": "split_stability",
                    "source": "split_metrics",
                    "metric": "sharpe_ratio",
                    "statistic": "min",
                    "comparator": "gte",
                    "threshold": 0.50,
                },
                {
                    "gate_id": "missing_ic",
                    "source": "metrics",
                    "metric": "ic_ir",
                    "comparator": "gte",
                    "threshold": 0.20,
                },
            ]
        },
        sources={
            "metrics": {"sharpe_ratio": 1.0, "max_drawdown": 0.12},
            "split_metrics": [{"sharpe_ratio": 0.50}, {"sharpe_ratio": 0.80}],
        },
    )

    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    assert evaluation.passed_gate_count == 2
    assert evaluation.failed_gate_count == 1
    assert evaluation.missing_gate_count == 1
    assert [result.status for result in evaluation.results] == ["pass", "fail", "pass", "missing"]


def test_write_alpha_evaluation_artifacts_persists_promotion_gate_artifact(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "timeframe": ["1D", "1D", "1D", "1D", "1D", "1D"],
            "prediction_score": [0.1, 0.6, 0.2, 0.5, 0.3, 0.4],
            "forward_return": [0.2, 0.3, 0.1, 0.4, 0.0, 0.2],
        }
    )
    result = evaluate_alpha_predictions(frame)

    manifest = write_alpha_evaluation_artifacts(
        tmp_path / "alpha" / "run-1",
        result,
        run_id="run-1",
        alpha_name="demo_alpha",
        aligned_frame=frame,
        promotion_gate_config={
            "gates": [
                {
                    "gate_id": "min_valid_timestamps",
                    "source": "qa_summary",
                    "metric": "forecast.valid_timestamps",
                    "comparator": "gte",
                    "threshold": float(result.summary["n_periods"]),
                },
                {
                    "gate_id": "nulls_clean",
                    "source": "qa_summary",
                    "metric": "nulls.prediction_null_rate",
                    "comparator": "lte",
                    "threshold": 0.0,
                },
            ]
        },
    )

    assert "promotion_gates.json" in manifest["artifact_files"]
    assert manifest["promotion_gate_summary"]["evaluation_status"] == "pass"
    payload = json.loads((tmp_path / "alpha" / "run-1" / "promotion_gates.json").read_text(encoding="utf-8"))
    assert payload["promotion_status"] == "eligible"
    assert payload["passed_gate_count"] == 2


def test_save_experiment_persists_strategy_promotion_gate_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    results_df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "signal": [1.0, 1.0, 0.0, 0.0],
            "executed_signal": [0.0, 1.0, 1.0, 0.0],
            "position": [0.0, 1.0, 1.0, 0.0],
            "delta_position": [0.0, 1.0, 0.0, -1.0],
            "abs_delta_position": [0.0, 1.0, 0.0, 1.0],
            "turnover": [0.0, 1.0, 0.0, 1.0],
            "trade_event": [False, True, False, True],
            "gross_strategy_return": [0.0, 0.02, -0.01, 0.0],
            "transaction_cost": [0.0, 0.0, 0.0, 0.0],
            "slippage_cost": [0.0, 0.0, 0.0, 0.0],
            "execution_friction": [0.0, 0.0, 0.0, 0.0],
            "strategy_return": [0.0, 0.02, -0.01, 0.0],
            "equity_curve": [1.0, 1.02, 1.0098, 1.0098],
        }
    )
    metrics = compute_performance_metrics(results_df)
    metrics["sanity_issue_count"] = 0.0
    metrics["sanity_warning_count"] = 0.0
    experiment_dir = save_experiment(
        "demo_strategy",
        results_df,
        metrics,
        {
            "strategy_name": "demo_strategy",
            "promotion_gates": {
                "gates": [
                    {
                        "gate_id": "min_sharpe",
                        "source": "metrics",
                        "metric": "sharpe_ratio",
                        "comparator": "gte",
                        "threshold": float(metrics["sharpe_ratio"]),
                    },
                    {
                        "gate_id": "sanity_clean",
                        "source": "qa_summary",
                        "metric": "sanity.issue_count",
                        "comparator": "lte",
                        "threshold": 0.0,
                    },
                ]
            },
        },
    )

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "promotion_gates.json" in manifest["artifact_files"]
    assert manifest["promotion_gate_summary"]["promotion_status"] == "eligible"
    registry_entries = load_registry(experiment_tracker.ARTIFACTS_ROOT / "registry.jsonl")
    assert registry_entries[0]["promotion_status"] == "eligible"


def test_save_walk_forward_experiment_persists_split_stability_promotion_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")

    def _split_frame(returns: list[float]) -> pd.DataFrame:
        equity = []
        level = 1.0
        for value in returns:
            level *= 1.0 + value
            equity.append(level)
        return pd.DataFrame(
            {
                "ts_utc": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
                "symbol": ["SPY", "SPY"],
                "signal": [1.0, 1.0],
                "executed_signal": [1.0, 1.0],
                "position": [1.0, 1.0],
                "delta_position": [1.0, 0.0],
                "abs_delta_position": [1.0, 0.0],
                "turnover": [1.0, 0.0],
                "trade_event": [True, False],
                "gross_strategy_return": returns,
                "transaction_cost": [0.0, 0.0],
                "slippage_cost": [0.0, 0.0],
                "execution_friction": [0.0, 0.0],
                "strategy_return": returns,
                "equity_curve": equity,
            }
        )

    split_a = _split_frame([0.01, 0.02])
    split_b = _split_frame([0.01, 0.01])
    split_b["ts_utc"] = ["2025-01-03T00:00:00Z", "2025-01-04T00:00:00Z"]
    split_results = [
        {
            "split_id": "split_000",
            "split_metadata": {
                "split_id": "split_000",
                "mode": "rolling",
                "train_start": "2024-01-01",
                "train_end": "2024-12-31",
                "test_start": "2025-01-01",
                "test_end": "2025-01-03",
            },
            "split_rows": 4,
            "train_rows": 2,
            "test_rows": 2,
            "metrics": compute_performance_metrics(split_a),
            "results_df": split_a,
        },
        {
            "split_id": "split_001",
            "split_metadata": {
                "split_id": "split_001",
                "mode": "rolling",
                "train_start": "2024-01-02",
                "train_end": "2025-01-01",
                "test_start": "2025-01-03",
                "test_end": "2025-01-05",
            },
            "split_rows": 4,
            "train_rows": 2,
            "test_rows": 2,
            "metrics": compute_performance_metrics(split_b),
            "results_df": split_b,
        },
    ]
    aggregate = compute_performance_metrics(pd.concat([split_a, split_b], ignore_index=True))
    aggregate["split_count"] = 2

    experiment_dir = save_walk_forward_experiment(
        "demo_strategy",
        split_results,
        aggregate,
        {
            "strategy_name": "demo_strategy",
            "evaluation_config_path": "configs/evaluation.yml",
            "evaluation": {"mode": "rolling", "timeframe": "1D"},
            "promotion_gates": {
                "gates": [
                    {
                        "gate_id": "stable_sharpe_floor",
                        "source": "split_metrics",
                        "metric": "sharpe_ratio",
                        "statistic": "min",
                        "comparator": "gte",
                        "threshold": min(
                            float(split_results[0]["metrics"]["sharpe_ratio"]),
                            float(split_results[1]["metrics"]["sharpe_ratio"]),
                        ),
                    }
                ]
            },
        },
    )

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["promotion_gate_summary"]["evaluation_status"] == "pass"
    assert (experiment_dir / "promotion_gates.json").exists()


def test_write_portfolio_artifacts_persists_promotion_gate_artifact(tmp_path: Path) -> None:
    portfolio_output = pd.DataFrame(
        {
            "weight__alpha": [0.5, 0.5],
            "weight__beta": [0.5, 0.5],
            "strategy_return__alpha": [0.00, 0.01],
            "strategy_return__beta": [0.02, 0.03],
            "gross_portfolio_return": [0.01, 0.02],
            "portfolio_weight_change": [1.0, 0.0],
            "portfolio_abs_weight_change": [1.0, 0.0],
            "portfolio_turnover": [1.0, 0.0],
            "portfolio_rebalance_event": [1, 0],
            "portfolio_changed_sleeve_count": [2, 0],
            "portfolio_transaction_cost": [0.0, 0.0],
            "portfolio_fixed_fee": [0.0, 0.0],
            "portfolio_slippage_proxy": [0.01, 0.0],
            "portfolio_slippage_cost": [0.0, 0.0],
            "portfolio_execution_friction": [0.0, 0.0],
            "net_portfolio_return": [0.01, 0.02],
            "portfolio_return": [0.01, 0.02],
            "portfolio_equity_curve": [101.0, 103.02],
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
        }
    )
    metrics = compute_portfolio_metrics(portfolio_output, "1D")
    manifest = write_portfolio_artifacts(
        tmp_path / "portfolio-run",
        portfolio_output,
        metrics,
        {
            "portfolio_name": "core_portfolio",
            "allocator": "equal_weight",
            "timeframe": "1D",
            "initial_capital": 100.0,
            "promotion_gates": {
                "gates": [
                    {
                        "gate_id": "max_drawdown",
                        "source": "metrics",
                        "metric": "max_drawdown",
                        "comparator": "lte",
                        "threshold": float(metrics["max_drawdown"]),
                    }
                ]
            },
        },
        [
            {"strategy_name": "alpha", "run_id": "run-a"},
            {"strategy_name": "beta", "run_id": "run-b"},
        ],
    )

    assert "promotion_gates.json" in manifest["artifact_files"]
    assert manifest["promotion_gate_summary"]["promotion_status"] == "eligible"
