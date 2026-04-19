from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.run_strategy import (
    RobustnessRunResult,
    StrategyRunResult,
    WalkForwardRunResult,
    get_strategy_config,
    parse_args,
    run_cli,
    run_strategy_experiment,
)
from src.research.strategies import build_strategy


def _default_execution_config_dict() -> dict[str, float | bool | str]:
    return {
        "enabled": False,
        "execution_delay": 1,
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
        "fixed_fee": 0.0,
        "fixed_fee_model": "per_rebalance",
        "slippage_model": "constant",
        "slippage_turnover_scale": 1.0,
        "slippage_volatility_scale": 1.0,
    }


def _results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [0, 1],
            "strategy_return": [0.0, 0.02],
            "equity_curve": [1.0, 1.02],
        }
    )


def _write_daily_features_dataset(root: Path, symbol: str = "AAPL", periods: int = 30) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    closes = pd.Series([100.0 + float(index) for index in range(periods)], dtype="float64")
    feature_df = pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * periods, dtype="string"),
            "ts_utc": ts,
            "timeframe": pd.Series(["1D"] * periods, dtype="string"),
            "date": pd.Series(ts.strftime("%Y-%m-%d"), dtype="string"),
            "close": closes,
            "feature_ret_1d": closes.div(closes.shift(1)).sub(1.0),
        }
    )
    dataset_path = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
    dataset_path.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(dataset_path / "part-0.parquet", index=False)
    return feature_df


def test_parse_args_accepts_strategy_and_date_filters() -> None:
    args = parse_args(["--strategy", "momentum_v1", "--start", "2025-01-01", "--end", "2025-02-01"])

    assert args.strategy == "momentum_v1"
    assert args.start == "2025-01-01"
    assert args.end == "2025-02-01"
    assert args.evaluation is None
    assert args.strict is False


def test_parse_args_accepts_evaluation_flag_without_path() -> None:
    args = parse_args(["--strategy", "momentum_v1", "--evaluation"])

    assert args.strategy == "momentum_v1"
    assert args.evaluation.endswith("configs\\evaluation.yml")


def test_parse_args_accepts_robustness_flag_without_path() -> None:
    args = parse_args(["--strategy", "momentum_v1", "--robustness"])

    assert args.strategy == "momentum_v1"
    assert args.robustness.endswith("configs\\robustness.yml")


def test_parse_args_accepts_strict_flag() -> None:
    args = parse_args(["--strategy", "momentum_v1", "--strict"])

    assert args.strict is True


def test_parse_args_accepts_simulation_config_path() -> None:
    args = parse_args(["--strategy", "momentum_v1", "--simulation", "configs/simulation.yml"])

    assert args.simulation == "configs/simulation.yml"


def test_get_strategy_config_raises_for_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy 'missing_strategy'"):
        get_strategy_config("missing_strategy", {"momentum_v1": {"dataset": "features_daily"}})


def test_run_cli_invokes_research_pipeline_components(monkeypatch, capsys) -> None:
    feature_df = pd.DataFrame({"feature_ret_1d": [0.01, 0.02]})
    strategy_config = {
        "momentum_v1": {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 5, "lookback_long": 20},
        }
    }
    calls: dict[str, object] = {}

    def fake_load_strategies_config(path=None):
        calls["config_path"] = path
        return strategy_config

    def fake_load_features(dataset, start=None, end=None):
        calls["load_features"] = {"dataset": dataset, "start": start, "end": end}
        return feature_df

    def fake_generate_signals(dataset, strategy):
        calls.setdefault("generate_signals", []).append({"dataset": dataset, "strategy_name": strategy.name})
        return dataset.assign(signal=[1, 0])

    def fake_run_backtest(signal_frame):
        calls["run_backtest"] = signal_frame.copy()
        return _results_frame()

    def fake_compute_metrics(results_df):
        calls["compute_metrics"] = results_df.copy()
        return {"cumulative_return": 0.02, "sharpe_ratio": 1.25}

    def fake_save_experiment(strategy_name, results_df, metrics, config):
        calls["save_experiment"] = {
            "strategy_name": strategy_name,
            "results_df": results_df.copy(),
            "metrics": metrics,
            "config": config,
        }
        return Path("artifacts/strategies/run-123")

    monkeypatch.setattr("src.cli.run_strategy.load_strategies_config", fake_load_strategies_config)
    monkeypatch.setattr("src.cli.run_strategy.load_features", fake_load_features)
    monkeypatch.setattr("src.cli.run_strategy.generate_signals", fake_generate_signals)
    monkeypatch.setattr(
        "src.cli.run_strategy.run_backtest",
        lambda signal_frame, execution_config=None, require_managed_signals=False: fake_run_backtest(signal_frame),
    )
    monkeypatch.setattr("src.cli.run_strategy.compute_metrics", fake_compute_metrics)
    monkeypatch.setattr("src.cli.run_strategy.save_experiment", fake_save_experiment)

    result = run_cli(["--strategy", "momentum_v1", "--start", "2025-01-01", "--end", "2025-02-01"])

    assert isinstance(result, StrategyRunResult)
    assert result.strategy_name == "momentum_v1"
    assert result.run_id == "run-123"
    assert calls["load_features"] == {
        "dataset": "features_daily",
        "start": "2025-01-01",
        "end": "2025-02-01",
    }
    assert calls["generate_signals"][0]["dataset"].equals(feature_df)
    assert calls["generate_signals"][0]["strategy_name"] == "momentum_v1"
    assert calls["generate_signals"][1]["strategy_name"] == "buy_and_hold_v1"
    assert calls["run_backtest"]["signal"].tolist() == [1, 0]
    assert calls["save_experiment"]["strategy_name"] == "momentum_v1"
    assert calls["save_experiment"]["metrics"]["cumulative_return"] == 0.02
    assert calls["save_experiment"]["metrics"]["sharpe_ratio"] == 1.25
    assert calls["save_experiment"]["metrics"]["benchmark_total_return"] == pytest.approx(0.02)
    assert calls["save_experiment"]["metrics"]["excess_return"] == pytest.approx(0.0)
    assert calls["save_experiment"]["metrics"]["benchmark_correlation"] == pytest.approx(1.0)
    assert calls["save_experiment"]["config"] == {
        "strategy_name": "momentum_v1",
        "dataset": "features_daily",
        "parameters": {"lookback_short": 5, "lookback_long": 20},
        "start": "2025-01-01",
        "end": "2025-02-01",
        "execution": _default_execution_config_dict(),
        "sanity": {
            "max_abs_period_return": 1.0,
            "max_annualized_return": 25.0,
            "max_equity_multiple": 1000000.0,
            "max_sharpe_ratio": 10.0,
            "min_annualized_volatility_floor": 0.02,
            "min_volatility_trigger_annualized_return": 1.0,
            "min_volatility_trigger_sharpe": 4.0,
            "smoothness_max_drawdown": 0.02,
            "smoothness_min_annualized_return": 0.75,
            "smoothness_min_positive_return_fraction": 0.95,
            "smoothness_min_sharpe": 3.0,
            "strict_sanity_checks": False,
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
            "enabled": False,
            "source": "default",
        },
        "runtime": {
            "execution": {
                "enabled": False,
                "execution_delay": 1,
                "transaction_cost_bps": 0.0,
                "slippage_bps": 0.0,
                "fixed_fee": 0.0,
                "fixed_fee_model": "per_rebalance",
                "slippage_model": "constant",
                "slippage_turnover_scale": 1.0,
                "slippage_volatility_scale": 1.0,
            },
            "sanity": {
                "max_abs_period_return": 1.0,
                "max_annualized_return": 25.0,
                "max_equity_multiple": 1000000.0,
                "max_sharpe_ratio": 10.0,
                "min_annualized_volatility_floor": 0.02,
                "min_volatility_trigger_annualized_return": 1.0,
                "min_volatility_trigger_sharpe": 4.0,
                "smoothness_max_drawdown": 0.02,
                "smoothness_min_annualized_return": 0.75,
                "smoothness_min_positive_return_fraction": 0.95,
                "smoothness_min_sharpe": 3.0,
                "strict_sanity_checks": False,
            },
            "portfolio_validation": {
                "long_only": False,
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
                "strict_sanity_checks": False,
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
                "enabled": False,
                "source": "default",
            },
        },
    }

    stdout = capsys.readouterr().out
    assert "strategy: momentum_v1" in stdout
    assert "run_id: run-123" in stdout
    assert "cumulative_return: 0.020000" in stdout
    assert "sharpe_ratio: 1.250000" in stdout
    assert "Signal diagnostics:" in stdout
    assert "- long: 50% | short: 0% | flat: 50%" in stdout
    assert "- trades: 1 | turnover: 0.50" in stdout
    assert "- avg holding: 1.0 bars" in stdout
    assert "QA Summary:" in stdout
    assert "- status: WARN" in stdout
    assert "- rows: 2 | symbols: 0" in stdout
    assert "- trades: 1 | turnover: 0.50" in stdout
    assert "Benchmark comparison:" in stdout
    assert "- benchmark return: 2%" in stdout
    assert "- excess return: +0%" in stdout
    assert "- correlation: 1.00" in stdout
    assert "Warnings:" in stdout
    assert "- insufficient data for a high-confidence analysis" in stdout
    assert "- strategy is highly correlated with the benchmark (1.00)" in stdout
    assert "- strategy delivered little excess return versus buy and hold" in stdout
    assert result.qa_summary["overall_status"] == "warn"


def test_run_cli_invokes_walk_forward_mode(monkeypatch, capsys) -> None:
    strategy_config = {
        "momentum_v1": {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 5, "lookback_long": 20},
        }
    }
    calls: dict[str, object] = {}
    walk_forward_result = WalkForwardRunResult(
        strategy_name="momentum_v1",
        run_id="wf-123",
        experiment_dir=Path("artifacts/strategies/wf-123"),
        metrics={
            "cumulative_return": 0.03,
            "sharpe_ratio": 1.5,
            "volatility": 0.1,
            "max_drawdown": 0.02,
            "win_rate": 0.5,
        },
        aggregate_summary={"split_count": 2, "cumulative_return": 0.03, "sharpe_ratio": 1.5},
        splits=[],
    )

    monkeypatch.setattr("src.cli.run_strategy.load_strategies_config", lambda path=None: strategy_config)

    def fake_run_walk_forward_experiment(strategy_name, strategy, evaluation_path, strategy_config, execution_config, strict):
        calls["walk_forward"] = {
            "strategy_name": strategy_name,
            "strategy_name_attr": strategy.name,
            "dataset": strategy.dataset,
            "evaluation_path": evaluation_path,
            "strategy_config": strategy_config,
            "execution_config": execution_config.to_dict(),
            "strict": strict,
        }
        return walk_forward_result

    monkeypatch.setattr("src.cli.run_strategy.run_walk_forward_experiment", fake_run_walk_forward_experiment)

    result = run_cli(["--strategy", "momentum_v1", "--evaluation", "configs/evaluation.yml"])

    assert result is walk_forward_result
    assert calls["walk_forward"]["strategy_name"] == "momentum_v1"
    assert calls["walk_forward"]["strategy_name_attr"] == "momentum_v1"
    assert calls["walk_forward"]["dataset"] == "features_daily"
    assert calls["walk_forward"]["evaluation_path"] == Path("configs/evaluation.yml")
    assert calls["walk_forward"]["strategy_config"] == strategy_config["momentum_v1"]
    assert calls["walk_forward"]["execution_config"] == _default_execution_config_dict()
    assert calls["walk_forward"]["strict"] is False

    stdout = capsys.readouterr().out
    assert "strategy: momentum_v1" in stdout
    assert "run_id: wf-123" in stdout
    assert "split_count: 2" in stdout


def test_run_cli_invokes_robustness_mode(monkeypatch, capsys) -> None:
    strategy_config = {
        "momentum_v1": {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 5, "lookback_long": 20},
        }
    }
    robustness_result = RobustnessRunResult(
        strategy_name="momentum_v1",
        run_id="robust-123",
        experiment_dir=Path("artifacts/strategies/robustness/robust-123"),
        summary={
            "variant_count": 4,
            "ranking_metric": "sharpe_ratio",
            "best_variant_id": "variant_0000",
            "best_metric_value": 1.25,
            "metric_spread": 0.3,
            "split_count": 3,
            "threshold_pass_rate": 0.75,
        },
        variants=[],
        variant_metrics=pd.DataFrame(),
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr("src.cli.run_strategy.load_strategies_config", lambda path=None: strategy_config)
    monkeypatch.setattr(
        "src.cli.run_strategy.load_robustness_config",
        lambda path: type(
            "FakeRobustnessConfig",
            (),
            {
                "resolve_strategy_name": staticmethod(lambda strategy_name: strategy_name or "momentum_v1"),
            },
        )(),
    )

    def fake_run_robustness_experiment(strategy_name, **kwargs):
        calls["strategy_name"] = strategy_name
        calls["kwargs"] = kwargs
        return robustness_result

    monkeypatch.setattr("src.cli.run_strategy.run_robustness_experiment", fake_run_robustness_experiment)

    result = run_cli(["--strategy", "momentum_v1", "--robustness"])

    assert result is robustness_result
    assert calls["strategy_name"] == "momentum_v1"
    assert calls["kwargs"]["execution_config"].to_dict() == _default_execution_config_dict()
    stdout = capsys.readouterr().out
    assert "variant_count: 4" in stdout
    assert "ranking_metric: sharpe_ratio" in stdout
    assert "best_variant: variant_0000" in stdout
    assert "split_count: 3" in stdout
    assert "threshold_pass_rate: 75.00%" in stdout


def test_run_cli_rejects_date_filters_in_evaluation_mode() -> None:
    with pytest.raises(ValueError, match="cannot be combined with --evaluation"):
        run_cli(["--strategy", "momentum_v1", "--evaluation", "--start", "2025-01-01"])


def test_run_strategy_experiment_loads_curated_daily_features_and_supports_mean_reversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    feature_df = _write_daily_features_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_strategy.save_experiment", lambda *args, **kwargs: tmp_path / "run-mean-reversion")

    result = run_strategy_experiment("mean_reversion_v1")

    assert isinstance(result, StrategyRunResult)
    assert result.strategy_name == "mean_reversion_v1"
    assert result.run_id == "run-mean-reversion"
    assert {"executed_signal", "delta_position", "abs_delta_position", "turnover", "trade_event", "gross_strategy_return", "net_strategy_return", "execution_friction"}.issubset(
        result.results_df.columns
    )
    assert result.results_df["close"].tolist() == pytest.approx(feature_df["close"].tolist())
    assert result.results_df["feature_ret_1d"].tolist() == pytest.approx(
        feature_df["feature_ret_1d"].fillna(0.0).tolist()
    )


def test_build_strategy_supports_buy_and_hold_baseline() -> None:
    strategy = build_strategy("buy_and_hold_v1", {"dataset": "features_1m", "parameters": {}})

    assert strategy.name == "buy_and_hold_v1"
    assert strategy.dataset == "features_1m"


def test_run_strategy_experiment_is_reproducible_for_repeated_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_daily_features_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    first = run_strategy_experiment("momentum_v1")
    first_metrics = dict(first.metrics)
    first_artifacts = {
        path.relative_to(first.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(first.experiment_dir.rglob("*"))
        if path.is_file()
    }

    second = run_strategy_experiment("momentum_v1")
    second_artifacts = {
        path.relative_to(second.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(second.experiment_dir.rglob("*"))
        if path.is_file()
    }

    assert first.run_id == second.run_id
    assert first_metrics == second.metrics
    assert first_artifacts == second_artifacts


def test_run_strategy_experiment_does_not_write_artifacts_when_temporal_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_frame = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "timeframe": pd.Series(["1D", "1D"], dtype="string"),
            "date": pd.Series(["2025-01-01", "2025-01-02"], dtype="string"),
            "feature_ret_1d": [0.01, 0.02],
            "feature_source_ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-03T00:00:00Z"], utc=True),
        }
    )
    calls: dict[str, int] = {"save_experiment": 0}

    monkeypatch.setattr("src.cli.run_strategy.load_features", lambda dataset, start=None, end=None: invalid_frame)

    def fake_save_experiment(*args, **kwargs):
        calls["save_experiment"] += 1
        return Path("artifacts/strategies/should-not-exist")

    monkeypatch.setattr("src.cli.run_strategy.save_experiment", fake_save_experiment)

    with pytest.raises(ValueError, match="future_feature_timestamp"):
        run_strategy_experiment("buy_and_hold_v1")

    assert calls["save_experiment"] == 0


def test_run_strategy_experiment_strict_sanity_failure_prevents_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_frame = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "timeframe": pd.Series(["1D", "1D"], dtype="string"),
            "date": pd.Series(["2025-01-01", "2025-01-02"], dtype="string"),
            "feature_ret_1d": [0.0, 1.0],
        }
    )
    calls = {"save_experiment": 0}

    monkeypatch.setattr(
        "src.cli.run_strategy.get_strategy_config",
        lambda strategy_name, path=None: {
            "dataset": "features_daily",
            "parameters": {},
            "sanity": {
                "strict_sanity_checks": True,
                "max_abs_period_return": 0.5,
            },
        },
    )
    monkeypatch.setattr("src.cli.run_strategy.load_features", lambda dataset, start=None, end=None: feature_frame)
    def fake_save_experiment(*args, **kwargs):
        calls["save_experiment"] += 1
        return Path("artifacts/strategies/should-not-exist")

    monkeypatch.setattr("src.cli.run_strategy.save_experiment", fake_save_experiment)

    with pytest.raises(ValueError, match="absolute strategy_return exceeds configured maximum"):
        run_strategy_experiment("buy_and_hold_v1")

    assert calls["save_experiment"] == 0


def test_run_strategy_experiment_non_strict_sanity_records_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_frame = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "timeframe": pd.Series(["1D", "1D"], dtype="string"),
            "date": pd.Series(["2025-01-01", "2025-01-02"], dtype="string"),
            "feature_ret_1d": [0.0, 1.0],
        }
    )

    monkeypatch.setattr(
        "src.cli.run_strategy.get_strategy_config",
        lambda strategy_name, path=None: {
            "dataset": "features_daily",
            "parameters": {},
            "sanity": {
                "strict_sanity_checks": False,
                "max_abs_period_return": 0.5,
            },
        },
    )
    monkeypatch.setattr("src.cli.run_strategy.load_features", lambda dataset, start=None, end=None: feature_frame)
    monkeypatch.setattr(
        "src.cli.run_strategy.save_experiment",
        lambda *args, **kwargs: Path("artifacts/strategies/non-strict-sanity"),
    )

    result = run_strategy_experiment("buy_and_hold_v1")

    assert result.metrics["sanity_status"] == "warn"
    assert result.metrics["sanity_issue_count"] >= 1
    assert result.qa_summary["sanity"]["status"] == "warn"
    assert result.qa_summary["overall_status"] == "warn"


def test_run_cli_passes_strict_flag_to_single_strategy_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.cli.run_strategy.get_strategy_config", lambda strategy_name, path=None: {"dataset": "features_daily"})
    calls: dict[str, object] = {}

    def fake_run_strategy_experiment(strategy_name, **kwargs):
        calls["strategy_name"] = strategy_name
        calls["kwargs"] = kwargs
        return StrategyRunResult(
            strategy_name=strategy_name,
            run_id="strict-run",
            metrics={"cumulative_return": 0.0, "sharpe_ratio": 0.0},
            experiment_dir=Path("artifacts/strategies/strict-run"),
            results_df=pd.DataFrame({"signal": [], "strategy_return": [], "equity_curve": []}),
        )

    monkeypatch.setattr("src.cli.run_strategy.run_strategy_experiment", fake_run_strategy_experiment)

    run_cli(["--strategy", "momentum_v1", "--strict"])

    assert calls["strategy_name"] == "momentum_v1"
    assert calls["kwargs"]["strict"] is True


def test_run_strategy_experiment_writes_simulation_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_daily_features_dataset(tmp_path, periods=40)
    monkeypatch.chdir(tmp_path)

    result = run_strategy_experiment(
        "momentum_v1",
        simulation_config={"method": "bootstrap", "num_paths": 4, "path_length": 5, "seed": 13},
    )

    assert result.simulation_result is not None
    simulation_dir = result.experiment_dir / "simulation"
    assert (simulation_dir / "summary.json").exists()
    assert (simulation_dir / "path_metrics.csv").exists()
    metrics_frame = pd.read_csv(simulation_dir / "path_metrics.csv")
    assert len(metrics_frame) == 4
    parent_manifest = (result.experiment_dir / "manifest.json").read_text(encoding="utf-8")
    assert "simulation/summary.json" in parent_manifest
