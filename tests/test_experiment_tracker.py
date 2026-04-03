from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research import experiment_tracker
from src.research.experiment_tracker import save_experiment, save_walk_forward_experiment
from src.research.metrics import compute_performance_metrics


def _metrics() -> dict[str, float | None]:
    return {
        "cumulative_return": 0.0098,
        "total_return": 0.0098,
        "volatility": 0.012,
        "annualized_return": 0.19,
        "annualized_volatility": 0.21,
        "sharpe_ratio": 0.41,
        "max_drawdown": 0.01,
        "win_rate": 0.5,
        "hit_rate": 0.5,
        "profit_factor": 1.1,
        "turnover": 0.33,
        "total_turnover": 1.32,
        "average_turnover": 0.33,
        "trade_count": 1.0,
        "rebalance_count": 1.0,
        "percent_periods_traded": 25.0,
        "average_trade_size": 1.32,
        "total_transaction_cost": 0.0,
        "total_slippage_cost": 0.0,
        "total_execution_friction": 0.0,
        "average_execution_friction_per_trade": 0.0,
        "exposure_pct": 66.7,
    }


def _experiment_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "feature_alpha": [0.15, -0.10, 0.25, 0.05],
            "signal": [1, 1, 0, 0],
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
        },
        index=pd.Index(["row_a", "row_b", "row_c", "row_d"], name="row_id"),
    )


def test_save_experiment_writes_standardized_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config = {"strategy_name": "mean_reversion", "lookback": 20, "threshold": 0.75}
    results_df = _experiment_results()

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_experiment("mean_reversion", results_df, _metrics(), config)

    assert experiment_dir.exists()
    assert experiment_dir.parent == artifact_root

    expected_files = {
        "config.json",
        "metrics.json",
        "signal_diagnostics.json",
        "qa_summary.json",
        "equity_curve.csv",
        "equity_curve.parquet",
        "signals.parquet",
        "trades.parquet",
        "manifest.json",
    }
    assert expected_files.issubset({path.name for path in experiment_dir.iterdir() if path.is_file()})

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == experiment_dir.name
    assert manifest["strategy_name"] == "mean_reversion"
    assert manifest["evaluation_mode"] == "single"
    assert manifest["evaluation_config_path"] is None
    assert manifest["split_count"] is None
    assert manifest["primary_metric"] == "sharpe_ratio"
    assert "config.json" in manifest["artifact_files"]
    assert "equity_curve.csv" in manifest["artifact_files"]
    assert "signal_diagnostics.json" in manifest["artifact_files"]
    assert "qa_summary.json" in manifest["artifact_files"]
    assert "trades.parquet" in manifest["artifact_files"]

    assert json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8")) == _metrics()
    assert json.loads((experiment_dir / "config.json").read_text(encoding="utf-8")) == config
    assert "runtime" not in json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    assert json.loads((experiment_dir / "signal_diagnostics.json").read_text(encoding="utf-8")) == {
        "total_rows": 4,
        "pct_long": 0.5,
        "pct_short": 0.0,
        "pct_flat": 0.5,
        "total_trades": 1,
        "turnover": 0.25,
        "avg_holding_period": 2.0,
        "exposure_pct": 0.5,
        "flags": {
            "always_flat": False,
            "always_long": False,
            "always_short": False,
            "no_trades": False,
            "high_turnover": False,
        },
    }
    assert json.loads((experiment_dir / "qa_summary.json").read_text(encoding="utf-8")) == {
        "run_id": experiment_dir.name,
        "strategy_name": "mean_reversion",
        "dataset": None,
        "timeframe": None,
        "row_count": 4,
        "symbols_present": 1,
        "date_range": ["2022-01-01T00:00:00Z", "2022-01-04T00:00:00Z"],
        "input_validation": {},
        "signal": {
            "pct_long": 0.5,
            "pct_short": 0.0,
            "pct_flat": 0.5,
            "turnover": 0.25,
            "total_trades": 1,
        },
        "execution": {
            "valid_returns": True,
            "equity_curve_present": True,
        },
        "metrics": {
            "total_return": 0.0098,
            "sharpe": 0.41,
            "max_drawdown": 0.01,
        },
        "relative": {
            "benchmark_return": None,
            "excess_return": None,
            "correlation": None,
            "relative_drawdown": None,
        },
        "sanity": {
            "issue_count": 0,
            "issues": [],
            "status": "pass",
            "strict_sanity_checks": False,
            "warning_count": 0,
        },
        "flags": {
            "no_data": False,
            "degenerate_signal": False,
            "no_trades": False,
            "high_turnover": False,
            "low_data": True,
            "high_benchmark_correlation": False,
            "low_excess_return": False,
            "high_turnover_low_edge": False,
            "beta_dominated_strategy": False,
            "sanity_warning": False,
        },
        "overall_status": "warn",
    }

    signals_df = pd.read_parquet(experiment_dir / "signals.parquet")
    equity_curve_df = pd.read_csv(experiment_dir / "equity_curve.csv")
    legacy_equity_curve_df = pd.read_parquet(experiment_dir / "equity_curve.parquet")
    trades_df = pd.read_parquet(experiment_dir / "trades.parquet")

    assert list(signals_df.columns[:9]) == [
        "ts_utc",
        "date",
        "symbol",
        "signal",
        "executed_signal",
        "position",
        "delta_position",
        "abs_delta_position",
        "turnover",
    ]
    assert signals_df["ts_utc"].tolist() == [
        "2022-01-01T00:00:00Z",
        "2022-01-02T00:00:00Z",
        "2022-01-03T00:00:00Z",
        "2022-01-04T00:00:00Z",
    ]
    assert signals_df["position"].tolist() == [0.0, 1.0, 1.0, 0.0]

    assert list(equity_curve_df.columns) == ["ts_utc", "symbol", "equity", "strategy_return", "signal", "position"]
    assert equity_curve_df["ts_utc"].tolist() == signals_df["ts_utc"].tolist()
    assert equity_curve_df["equity"].tolist() == [1.0, 1.02, 1.0098, 1.0098]
    assert equity_curve_df["position"].tolist() == [0.0, 1.0, 1.0, 0.0]

    assert list(legacy_equity_curve_df.columns) == ["signal", "strategy_return", "equity_curve"]
    assert trades_df.to_dict(orient="records") == [
        {
            "entry_ts_utc": "2022-01-02T00:00:00Z",
            "exit_ts_utc": "2022-01-03T00:00:00Z",
            "symbol": "SPY",
            "direction": "long",
            "return": pytest.approx(0.0098),
        }
    ]


def test_save_experiment_handles_empty_results_without_optional_trades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    empty_results = pd.DataFrame(
        columns=["ts_utc", "symbol", "signal", "strategy_return", "equity_curve"]
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_experiment(
        "empty_strategy",
        empty_results,
        _metrics(),
        {"strategy_name": "empty_strategy"},
    )

    assert (experiment_dir / "signals.parquet").exists()
    assert (experiment_dir / "equity_curve.csv").exists()
    assert (experiment_dir / "manifest.json").exists()
    assert (experiment_dir / "signal_diagnostics.json").exists()
    assert (experiment_dir / "qa_summary.json").exists()
    assert not (experiment_dir / "trades.parquet").exists()

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "signal_diagnostics.json" in manifest["artifact_files"]
    assert "qa_summary.json" in manifest["artifact_files"]
    assert "trades.parquet" not in manifest["artifact_files"]


def test_save_walk_forward_experiment_supports_empty_split_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config = {
        "strategy_name": "empty_walk_forward",
        "evaluation_config_path": "configs/evaluation.yml",
        "evaluation": {"mode": "rolling"},
    }

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_walk_forward_experiment(
        "empty_walk_forward",
        [],
        {"split_count": 0, **_metrics()},
        config,
    )

    metrics_by_split = pd.read_csv(experiment_dir / "metrics_by_split.csv")
    assert metrics_by_split.empty
    assert list(metrics_by_split.columns[:9]) == [
        "split_id",
        "mode",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "split_rows",
        "train_rows",
        "test_rows",
    ]

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_mode"] == "walk_forward"
    assert manifest["split_count"] == 0
    assert "metrics_by_split.csv" in manifest["artifact_files"]


def test_save_experiment_is_reproducible_across_repeated_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config = {"strategy_name": "mean_reversion", "lookback": 20, "threshold": 0.75}
    results_df = _experiment_results()

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    first_dir = save_experiment("mean_reversion", results_df, _metrics(), config)
    first_snapshot = {
        path.relative_to(first_dir).as_posix(): path.read_bytes()
        for path in sorted(first_dir.rglob("*"))
        if path.is_file()
    }
    first_registry = (artifact_root / "registry.jsonl").read_bytes()

    second_dir = save_experiment("mean_reversion", results_df, _metrics(), config)
    second_snapshot = {
        path.relative_to(second_dir).as_posix(): path.read_bytes()
        for path in sorted(second_dir.rglob("*"))
        if path.is_file()
    }
    second_registry = (artifact_root / "registry.jsonl").read_bytes()

    assert first_dir == second_dir
    assert first_snapshot == second_snapshot
    assert first_registry == second_registry


def test_save_experiment_persists_effective_runtime_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config = {
        "strategy_name": "mean_reversion",
        "execution": {
            "enabled": True,
            "execution_delay": 2,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 1.0,
        },
        "sanity": {
            "max_abs_period_return": 0.9,
            "strict_sanity_checks": True,
        },
        "strict_mode": {
            "enabled": True,
            "source": "config",
        },
        "runtime": {
            "execution": {
                "enabled": True,
                "execution_delay": 2,
                "transaction_cost_bps": 5.0,
                "slippage_bps": 1.0,
            },
            "sanity": {
                "max_abs_period_return": 0.9,
                "max_annualized_return": 25.0,
                "max_sharpe_ratio": 10.0,
                "max_equity_multiple": 1000000.0,
                "strict_sanity_checks": True,
                "min_annualized_volatility_floor": 0.02,
                "min_volatility_trigger_sharpe": 4.0,
                "min_volatility_trigger_annualized_return": 1.0,
                "smoothness_min_sharpe": 3.0,
                "smoothness_min_annualized_return": 0.75,
                "smoothness_max_drawdown": 0.02,
                "smoothness_min_positive_return_fraction": 0.95,
            },
            "portfolio_validation": {
                "long_only": False,
                "target_weight_sum": 1.0,
                "weight_sum_tolerance": 1e-8,
                "target_net_exposure": 1.0,
                "net_exposure_tolerance": 1e-8,
                "max_gross_exposure": 1.0,
                "max_leverage": 1.0,
                "max_single_sleeve_weight": None,
                "min_single_sleeve_weight": None,
                "max_abs_period_return": 1.0,
                "max_equity_multiple": 1000000.0,
                "strict_sanity_checks": True,
            },
            "risk": {
                "volatility_window": 20,
                "target_volatility": None,
                "min_volatility_scale": 0.0,
                "max_volatility_scale": 1.0,
                "allow_scale_up": False,
                "var_confidence_level": 0.95,
                "cvar_confidence_level": 0.95,
                "volatility_epsilon": 1e-12,
                "periods_per_year_override": None,
            },
            "strict_mode": {
                "enabled": True,
                "source": "config",
            },
        },
    }

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_experiment("mean_reversion", _experiment_results(), _metrics(), config)

    persisted = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    assert persisted["runtime"]["execution"]["execution_delay"] == 2
    assert persisted["runtime"]["execution"]["transaction_cost_bps"] == pytest.approx(5.0)
    assert persisted["runtime"]["sanity"]["strict_sanity_checks"] is True
    assert persisted["runtime"]["risk"]["volatility_window"] == 20
    assert persisted["runtime"]["strict_mode"] == {"enabled": True, "source": "config"}


def test_save_experiment_writes_aggregated_strategy_equity_for_multi_symbol_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    results_df = pd.DataFrame(
        {
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "signal": [1.0, 1.0, 1.0, 1.0],
            "position": [1.0, 1.0, 1.0, 1.0],
            "strategy_return": [0.04, 0.00, 0.02, 0.02],
            "equity_curve": [1.04, 1.0608, 1.0608, 1.082016],
        }
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_experiment(
        "multi_symbol_mean",
        results_df,
        compute_performance_metrics(results_df),
        {"strategy_name": "multi_symbol_mean"},
    )

    equity_curve_df = pd.read_csv(experiment_dir / "equity_curve.csv")

    assert equity_curve_df["strategy_return"].tolist() == pytest.approx([0.03, 0.01, 0.03, 0.01])
    assert equity_curve_df["equity"].tolist() == pytest.approx([1.03, 1.0403, 1.03, 1.0403])
